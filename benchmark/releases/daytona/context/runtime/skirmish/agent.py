"""Time-bounded authoring loop. Setup and final evaluation are outside the model's clock."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import tarfile

from .common import Budget, REFERENCE, REFERENCE_SHA, SCHEMA, canonical, identifier, sha, write_json
from .providers import provider
from .tools import AuthoringTools, TOOLS


def source_receipt():
    root = Path('/workspace')
    paths = sorted((root / 'engine-rs/src').glob('*.rs'))
    paths += [root / 'engine-rs/Cargo.toml', root / 'engine-rs/Cargo.lock',
              root / 'src/polytopia_bot/gpu/rules.json']
    paths += sorted((root / 'src/polytopia_bot/skirmish').glob('*.py'))
    # Pin the actual adapter, serializer, policy, feature encoding and all transitive Python sources.
    paths += sorted((root / 'src/polytopia_bot').rglob('*.py'))
    paths += [p for p in (root/'experiments/modal_skirmish.py',root/'experiments/modal_skirmish_v5.py') if p.exists()]
    return {str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted(set(paths))}


def run_trial(run_id, config, coding_image, app, volume):
    import modal
    if not os.environ.get('MODAL_TASK_ID'):
        raise RuntimeError('authoring runs must execute on Modal')
    identifier(run_id)
    budget = Budget(**config.get('budget', {})).validate()
    config = dict(config)
    params = dict(config.get('parameters', {}))
    token_field = 'max_output_tokens' if config['provider'] == 'responses' else 'max_tokens'
    params[token_field] = min(int(params.get(token_field, budget.max_output_tokens)), budget.max_output_tokens)
    config['parameters'] = params
    root = Path('/results/runs') / run_id
    root.mkdir(parents=True, exist_ok=False)  # Never overwrite/resume a run as a new trial.
    status = dict(schema=SCHEMA, run_id=run_id, model=config['model'], status='setting_up')
    write_json(root / 'status.json', status)
    volume.commit()
    sandbox = None
    try:
        # No secrets, volumes, OIDC tokens, opponent files or other source tree in this container.
        sandbox = modal.Sandbox.create('sleep', 'infinity', image=coding_image, app=app,
                        cpu=2, memory=2048, timeout=budget.seconds + 300, block_network=True,
                        workdir='/bot')
        from .conditions import condition
        treatment = condition(config)
        docs = (Path('/workspace/skirmish/PROMPT.md')).read_text()
        system = (f'Write the strongest Rust bot you can for Polytopia 1v1 within {budget.seconds} seconds. '
                  'You have no internet. Use only the provided tools and public game information. '
                  'Edit, compile, practice and inspect feedback repeatedly. Keep the current source compiling. '
                  'The final successfully compiled current source is frozen at the deadline. '
                  f'Active resource constraint: {budget.constraint}; changed-source revision allowance: {budget.max_revisions}. '
                  'All stated time/request/practice ceilings are enforced. Use feedback to improve your bot, '
                  'then call finish when ready.\n\n' + docs + '\n\nRun framing:\n' + treatment['text'])
        write_json(root / 'protocol.json', dict(schema=SCHEMA, model=config['model'],
            provider=config['provider'], base_url=config['base_url'], parameters=config.get('parameters', {}),
            budget=asdict(budget), treatment=treatment, budget_profile=config.get('budget_profile'), tools=TOOLS, prompt_sha256=sha(system.encode()),
            ruleset='polytopia-2.17.3', size=14, maps='classic', players=2, reference=REFERENCE,
            reference_sha256=REFERENCE_SHA, source_files=source_receipt(),
            sandbox_id=sandbox.object_id, coding_image_id=config.get('_coding_image_id'), block_network=True, rust_toolchain='1.90.0', wasmtime='38.0.0'))
        (root / 'prompt.txt').write_text(system)
        with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
            for name in source_receipt():
                archive.add(Path('/workspace') / name, arcname=name)
            archive.add('/workspace/skirmish', arcname='skirmish')
        api = provider(config, system, TOOLS)
        from .game import warm_reference
        reference_bot = warm_reference('/results/references/hero-1v1/final.npz',REFERENCE_SHA)
        started = time.monotonic()
        deadline = started + budget.seconds
        tools = AuthoringTools(sandbox, root, budget, volume, deadline)
        tools.reference_bot = reference_bot
        tools.source = tools.read('/bot/src/main.rs')
        # Initial compile counts against the wall clock and establishes a fallback starter artifact.
        tools.compile()
        status.update(status='authoring', sandbox_id=sandbox.object_id, budget_seconds=budget.seconds,
                      started_unix=time.time(), requests=0)
        write_json(root / 'status.json', status)
        volume.commit()
        prompt = 'Start by reading /bot/src/main.rs and /reference/rules.json, then write and test your bot.'
        end_reason = 'deadline'
        usage = []
        with (root / 'transcript.jsonl').open('x') as log:
            for index in range(budget.max_requests):
                if tools.remaining() <= 1:
                    break
                try:
                    turn = api.next(prompt, timeout=max(1, min(180, tools.remaining())))
                except TimeoutError:
                    end_reason = 'deadline' if tools.remaining() <= 1 else 'api_timeout'
                    break
                prompt = None
                usage.append(turn['usage'])
                log.write(canonical(dict(type='model', elapsed=time.monotonic() - started, response=turn['raw'])) + '\n')
                log.flush()
                # A response arriving past the deadline is logged but cannot change a submission.
                if tools.remaining() <= 0:
                    break
                for call in turn['calls']:
                    try:
                        args = json.loads(call['arguments'])
                        result = tools.dispatch(call['name'], args)
                    except Exception as error:
                        result = {'error': f'{type(error).__name__}: {str(error)[:500]}'}
                    result['budget_status'] = tools.budget_status()
                    api.result(call, result)
                    log.write(canonical(dict(type='tool', elapsed=time.monotonic() - started,
                        call=call, result=result)) + '\n')
                    log.flush()
                    if tools.finished:
                        break
                if tools.finished:
                    end_reason = 'agent_finished'
                    break
                if not turn['calls']:
                    prompt = (f'{max(0, int(tools.remaining()))} seconds remain. Use compile, campaign and review '
                              'to improve your bot; you may continue editing.')
                status.update(requests=index + 1, practice_games=tools.practice_count, revisions=tools.revisions,
                              elapsed_seconds=time.monotonic() - started)
                write_json(root / 'status.json', status)
                volume.commit()
            else:
                end_reason = 'request_budget'
        status.update(requests=len(usage),practice_games=tools.practice_count,revisions=tools.revisions,
                      elapsed_seconds=time.monotonic()-started)
        # We never quietly score an older version after a broken final edit.
        if tools.artifact is None or sha(tools.source.encode()) != tools.compiled_source:
            status.update(status='failed_submission', reason='current_source_not_compiled')
            write_json(root / 'status.json', status)
            volume.commit()
            return status
        artifact = tools.artifact.read_bytes()
        (root / 'bot.wasm').write_bytes(artifact)
        (root / 'main.rs').write_text(tools.source)
        costs = [u.get('cost') for u in usage]
        manifest = dict(schema=SCHEMA, run_id=run_id, model=config['model'],
            api_cost_usd=sum(costs) if all(isinstance(c, (int, float)) for c in costs) else None,
            wasm_sha256=sha(artifact), source_sha256=sha(tools.source.encode()),
            protocol_sha256=sha((root / 'protocol.json').read_bytes()),
            source_archive_sha256=sha((root / 'source.tar.gz').read_bytes()),
            transcript_sha256=sha((root / 'transcript.jsonl').read_bytes()),
            budget=asdict(budget), treatment=treatment, revisions=tools.revisions,
            budget_profile=config.get('budget_profile'),
            authoring_seconds=min(budget.seconds,time.monotonic()-started),
            controller_wall_seconds=time.monotonic()-started,
            end_reason=end_reason, requests=len(usage), practice_games=tools.practice_count, usage=usage)
        write_json(root / 'submission.json', manifest)
        status.update(status='frozen', wasm_sha256=manifest['wasm_sha256'], end_reason=end_reason,
                      api_cost_usd=manifest['api_cost_usd'],authoring_seconds=manifest['authoring_seconds'],
                      controller_wall_seconds=manifest['controller_wall_seconds'])
        write_json(root / 'status.json', status)
        volume.commit()
        sandbox.terminate()
        sandbox = None
        # Independent random final maps are allocated after the bot is frozen.
        from .tournament import tournament
        report = tournament('final-' + run_id, [run_id], int(config.get('final_maps', 16)), volume)
        status.update(status='completed', final=report)
        write_json(root / 'report.json', status)
        write_json(root / 'status.json', status)
        volume.commit()
        return status
    except Exception as error:
        status.update(status='failed', error=f'{type(error).__name__}: {str(error)[:500]}')
        write_json(root / 'status.json', status)
        volume.commit()
        raise
    finally:
        if sandbox is not None:
            sandbox.terminate()
