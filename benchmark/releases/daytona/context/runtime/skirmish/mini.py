"""Versioned mini-SWE-agent authoring tracks, staged feedback and frozen evaluation."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

from .common import Budget, REFERENCE, REFERENCE_SHA, canonical, identifier, sha, write_json
from .game import WasiBot, play_game, warm_reference
from .mini_worker import MINI_VERSION

MAP_TYPES = ('drylands', 'lakes', 'continents')
TRACKS = {
    'oneshot-20m': dict(stage_seconds=1200, revision_rounds=0),
    'feedback-3r': dict(stage_seconds=300, revision_rounds=3),
    'independent-40m': dict(stage_seconds=2400, revision_rounds=0),
    'feedback-3r-40m': dict(stage_seconds=600, revision_rounds=3),
}
from .protocol import SCHEMA


def track_config(config):
    track = config['track']
    if track not in TRACKS:
        raise ValueError('unknown mini benchmark track')
    values = dict(TRACKS[track])
    if config.get('smoke'):
        values['stage_seconds'] = int(config.get('smoke_seconds', 60))
        if not 10 <= values['stage_seconds'] <= 120:
            raise ValueError('smoke duration must be 10..120 seconds')
    elif any(key in config for key in ('stage_seconds', 'revision_rounds', 'smoke_seconds')):
        raise ValueError('official track budgets cannot be overridden')
    if config.get('provider') != 'openrouter':
        raise ValueError('all benchmark models must use OpenRouter')
    return values


def evaluation_schedule(seeds):
    """One seed per map type, paired seats; seed partitions are checked by the caller."""
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('three distinct map seeds required')
    return [dict(map_type=kind, seed=seed, seat=seat)
            for kind, seed in zip(MAP_TYPES, seeds) for seat in (1, 2)]


def evaluate(artifact, root, reference, budget, seeds, phase, volume):
    rows = []
    for i, job in enumerate(evaluation_schedule(seeds)):
        trace = root / f'{i:02d}-{job["map_type"]}-seat{job["seat"]}.jsonl'
        row = play_game(WasiBot(artifact, budget.fuel, budget.move_seconds), reference,
                        job['seed'], job['seat'], budget, trace, phase=phase,
                        labels=['candidate', REFERENCE], map_type=job['map_type'],cap_policy='engine_draw')
        from .diagnostics import public_summary
        row['public_debrief']=public_summary(trace)
        row['trace_file'] = trace.name
        rows.append(row)
        write_json(root / 'progress.json', dict(completed=len(rows), total=6))
        volume.commit()
    report = dict(games=len(rows), wins=sum(r['result']=='win' for r in rows),
                  losses=sum(r['result']=='loss' for r in rows),draws=sum(r['result']=='draw' for r in rows),
                  unresolved=sum(r['result']=='unresolved' for r in rows),
                  forfeits=sum(r['status']=='forfeit' for r in rows), rows=rows)
    write_json(root / 'report.json', report)
    volume.commit()
    return report


def stop_coder(sandbox):
    # Child processes, daemons and delayed writes must end before freezing source.
    p = sandbox.exec('pkill', '-KILL', '-u', 'coder')
    p.wait()


def start_watchdog(sandbox, deadline, *, stage=0, start_path=None):
    """Capture immutable cutoff source independently of the controller."""
    from .lifecycle import CAPTURE
    if start_path:
        sandbox.exec('rm','-f',start_path).wait()
    setup = ('import os,time,json; from pathlib import Path\n'
             'Path("/verify/watchdog.pid").write_text(str(os.getpid()))\n')
    if start_path:
        setup += (f'p=Path({start_path!r}); waiting=time.monotonic()\n'
                  'while not p.exists():\n'
                  ' if time.monotonic()-waiting>120: raise TimeoutError("no stage start")\n'
                  ' time.sleep(.01)\n'
                  'deadline=json.loads(p.read_text())["deadline_unix"]\n')
    else: setup += f'deadline={deadline!r}\n'
    script=setup+'time.sleep(max(0,deadline-time.time()))\n'+f'stage={int(stage)}\n'+CAPTURE+'\nPath("/verify/watchdog-fired").write_text(str(time.time()))'
    p=sandbox.exec('python','-c',script,timeout=7500)
    ready=sandbox.exec('python','-c',
        'import time; from pathlib import Path;\nwhile not Path("/verify/watchdog.pid").exists(): time.sleep(.01)\nprint("ready")',timeout=5)
    assert ready.stdout.read().strip()=='ready';ready.wait()
    return p


def cancel_watchdog(sandbox):
    script = ('import os,signal; from pathlib import Path; '
              'p=Path("/verify/watchdog.pid"); '
              '\nif p.exists():\n'
              ' try: os.kill(int(p.read_text()),signal.SIGKILL)\n'
              ' except ProcessLookupError: pass\n'
              ' p.unlink()\n')
    p=sandbox.exec('python','-c',script);p.wait()


def freeze(sandbox, root, volume, *, cutoff=None):
    """Persist exact source BEFORE compilation. Idempotently rebuild that source only."""
    from .protocol import submission_quality
    stop_coder(sandbox)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'submission.json').exists():
        saved=json.loads((root/'submission.json').read_text())
        if sha((root/'main.rs').read_bytes())!=saved['source_sha256']:
            raise ValueError('frozen source digest mismatch')
        if saved.get('wasm_sha256') and sha((root/'bot.wasm').read_bytes())!=saved['wasm_sha256']:
            raise ValueError('frozen Wasm digest mismatch')
        return saved
    cutoff_receipt=json.loads((root/'capture.json').read_text()).get('cutoff') if (root/'capture.json').exists() else None
    if not (root/'main.rs').exists():
        if cutoff:
            capture=json.loads(sandbox.filesystem.read_text(cutoff+'/capture.json'));cutoff_receipt=capture
            if capture.get('error'):
                # Deleting or making the submitted file unreadable is an invalid
                # submission, not an infrastructure retry or an older checkpoint.
                (root/'main.rs').write_bytes(b'')
                receipt=dict(source_sha256=sha(b''),source_bytes=0,status='missing_source',
                    returncode=-1,compile_output='No readable main.rs at cutoff: '+capture['error'],
                    submission_quality='compile_failure')
                write_json(root/'capture.json',receipt);write_json(root/'submission.json',receipt)
                volume.commit();return receipt
        source=sandbox.filesystem.read_bytes((cutoff or '/bot/src')+'/main.rs')
        (root/'main.rs').write_bytes(source)
    source=(root/'main.rs').read_bytes()
    receipt=dict(source_sha256=sha(source), source_bytes=len(source),status='source_captured',cutoff=cutoff_receipt)
    if cutoff_receipt and cutoff_receipt['source_sha256']!=receipt['source_sha256']:
        raise ValueError('cutoff source changed during transfer')
    write_json(root/'capture.json',receipt);volume.commit()
    sandbox.filesystem.write_bytes(source, '/verify/src/main.rs')
    p=sandbox.exec('cargo','build','--manifest-path','/verify/Cargo.toml','--release',
                   '--offline','--locked','--target','wasm32-wasip1',timeout=90)
    out=p.stdout.read()+p.stderr.read();p.wait()
    receipt.update(returncode=p.returncode,compile_output=out[-12000:],status='compiled')
    if p.returncode==0:
        data=sandbox.filesystem.read_bytes('/verify/target/wasm32-wasip1/release/skirmish-bot.wasm')
        if not data.startswith(b'\0asm') or len(data)>8*1024*1024:
            receipt.update(returncode=-1,status='invalid_artifact',compile_output='Compiled Wasm exceeds declared 8 MiB limit or has invalid header')
        else:
            (root/'bot.wasm').write_bytes(data);receipt['wasm_sha256']=sha(data)
    starter=sandbox.filesystem.read_bytes('/reference/STARTER.rs') if cutoff else b''
    receipt['submission_quality']=submission_quality(receipt['source_sha256'],sha(starter),receipt['returncode'])
    write_json(root/'submission.json',receipt);volume.commit()
    return receipt


def run_mini_trial(run_id, config, image, app, volume):
    import modal
    if not os.environ.get('MODAL_TASK_ID'):
        raise RuntimeError('benchmark execution must run on Modal')
    identifier(run_id)
    track = track_config(config)
    root = Path('/results/mini-runs') / run_id
    root.mkdir(parents=True, exist_ok=False)
    status = dict(schema=SCHEMA, run_id=run_id, model=config['model'], track=config['track'],
                  smoke=bool(config.get('smoke')), status='setting_up')
    write_json(root/'status.json', status); volume.commit()
    sandbox = None
    try:
        sandbox = modal.Sandbox.create('sleep', 'infinity', image=image, app=app, cpu=2, memory=2048,
                         timeout=14400, block_network=True, workdir='/bot')
        status['sandbox_id']=sandbox.object_id
        write_json(root/'status.json',status);write_json(root/'config.json',config);volume.commit()
        isolation = sandbox.exec('python', '-c',
            'import os,socket,json; s=socket.socket();s.settimeout(2);'
            'print(json.dumps(dict(network_blocked=s.connect_ex(("1.1.1.1",443))!=0,'
            'credential_absent="OPENROUTER_API_KEY" not in os.environ,reference_absent=not os.path.exists("/results"))))')
        checks = json.loads(isolation.stdout.read()); isolation.wait()
        assert all(checks.values()), checks
        budget = Budget(seconds=track['stage_seconds'],max_output_tokens=config['parameters']['max_tokens']).validate()
        # All three map families and reference inference are warmed before any coding clock.
        reference = warm_reference('/results/references/hero-1v1/final.npz', REFERENCE_SHA)
        from .game import reset_map
        from polytopia_bot.studio.environment import RustEnvironment
        for kind in MAP_TYPES:
            env = RustEnvironment(size=14)
            try: reference.select(reset_map(env, 500000, kind))
            finally: env.close()
        from .agent import source_receipt
        system = Path('/workspace/skirmish/mini/SYSTEM.md').read_text()
        rules = Path('/workspace/skirmish/mini/RULES.md').read_bytes()
        protocol = dict(schema=SCHEMA, harness='mini-swe-agent', harness_version=MINI_VERSION,
                        provider='openrouter', model=config['model'], parameters=config['parameters'],
                        track=config['track'], **track, map_types=MAP_TYPES, size=14, players=2,
                        runtime=asdict(budget), reference=REFERENCE, reference_sha256=REFERENCE_SHA,
                        administration=dict(model_calls_per_stage=0,cost_limit=None,command_seconds=60),
                        rules_sha256=sha(rules), system_sha256=sha(system.encode()), source_files=source_receipt(),
                        tools_sha256=sha(Path('/workspace/skirmish/mini/TOOLS.md').read_bytes()),
                        coding_image_id=image.object_id, isolation=checks,
                        authoring_tools=dict(version=1,public_maps=9,opponent_provided=False,
                                             match_runner=False,mechanics_actions_per_replay=32),
                        feedback='six complete candidate-public observation/action JSONL files; private referee logs withheld',
                        smoke=bool(config.get('smoke')))
        from .protocol import EVALUATION
        protocol['evaluation']=EVALUATION
        write_json(root/'protocol.json', protocol)
        (root/'runtime-packages.txt').write_text(subprocess.check_output(
            [sys.executable,'-m','pip','freeze'],text=True))
        # Full source snapshot is on Modal, never downloaded with model artifacts.
        import tarfile
        with tarfile.open(root/'source.tar.gz','w:gz') as archive:
            for name in source_receipt(): archive.add(Path('/workspace')/name, arcname=name)
            archive.add('/workspace/skirmish/mini',arcname='skirmish/mini')
        history = None
        feedback = ''
        stages = []
        for stage in range(track['revision_rounds']+1):
            seconds = track['stage_seconds']
            task = (f'Build the strongest Bardur Polytopia 1v1 Rust bot. This is stage {stage} of '
                    f'{track["revision_rounds"]+1}, with {seconds} seconds of wall time. '
                    'Start with /reference/QUICKSTART.md. Read /reference/RULES.md for the complete rules, tables and implementation, '
                    '/reference/PROTOCOL.md for the wire format, and /bot/src/main.rs for the starter. '
                    'Edit only /bot/src/main.rs for your submission (std and serde_json only). '
                    'Compile locally using cargo build --release --offline --locked --target wasm32-wasip1. '
                    'You may compile and run self-written tests as often as time allows. '
                    'Use `skirmish --help` and /reference/TOOLS.md for the real engine CLI: '
                    'compile, inspect nine public maps, check WASI protocol, and replay short '
                    'explicit action sequences. An annotated fixed example is provided. '
                    'No opponent or automatic practice-match runner is provided. '
                    'The controller freezes your current source when time expires; it then compiles that exact '
                    'source with the original manifest/lockfile. Broken current source fails, with no rollback. '
                    'Use the bash tool for all actions. To end this stage early run '
                    '`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. '
                    f'Each model response allows at most {budget.max_output_tokens} output tokens, including reasoning. '
                    'Keep bash actions compact and split large file writes across commands if needed. '
                    + ('No expert-game feedback is available during this one-shot attempt.' if not track['revision_rounds'] else
                       'After each of the first three stages you receive six full traces against the fixed expert: '
                       'drylands, lakes, continents, both seats. Then you have another coding stage to revise. '
                       'A revision round is a coding stage, not an individual file edit. ') + feedback)
            trajectory = root/f'stage-{stage}.trajectory.json'
            payload = dict(provider=config['provider'], model=config['model'], parameters=config['parameters'], sandbox_id=sandbox.object_id,
                           system=system, task=task, seconds=seconds,trajectory=str(trajectory),history=history)
            status.update(status='authoring',stage=stage)
            write_json(root/'status.json',status);volume.commit()
            from .lifecycle import run_stage_worker
            end_reason,elapsed,timing=run_stage_worker(sandbox,root,stage,payload,start_watchdog,stop_coder,cancel_watchdog,volume)
            receipt = freeze(sandbox, root/f'stage-{stage}', volume,cutoff=f'/verify/cutoffs/stage-{stage}')
            stage_info = dict(stage=stage, authoring_seconds=min(elapsed,seconds),
                              controller_wall_seconds=elapsed, end_reason=end_reason,timing=timing, **receipt)
            if trajectory.exists():
                trajectory_info=json.loads(trajectory.read_text())['info']
                stage_info['agent_exit_status']=trajectory_info.get('exit_status')
                stage_info.update(trajectory_info.get('model_stats',{}))
            stages.append(stage_info)
            write_json(root/'stages.json',stages)
            if end_reason=='agent_error':
                status.update(status='failed_harness',stage=stage,reason='mini_worker_error',stages=stages)
                write_json(root/'status.json',status);write_json(root/'report.json',status);volume.commit()
                return status

            history = str(trajectory) if trajectory.exists() else history
            if receipt['returncode'] != 0:
                if stage == track['revision_rounds']:
                    status.update(status='failed_submission', reason='final_source_compile_failed', stages=stages)
                    write_json(root/'report.json',status);write_json(root/'status.json',status);volume.commit()
                    return status
                feedback = '\nYour frozen source failed to compile. Compiler output:\n'+receipt['compile_output']
            elif stage < track['revision_rounds']:
                status.update(status='feedback_evaluation',stage=stage)
                write_json(root/'status.json',status);volume.commit()
                feedback_root = root/f'feedback-{stage}'
                seeds = [500000+stage*30+i for i in range(3)]
                report = evaluate(root/f'stage-{stage}/bot.wasm', feedback_root, reference, budget, seeds, 'practice', volume)
                destination = f'/feedback/round-{stage}'
                sandbox.exec('mkdir','-p',destination).wait()
                # Complete deterministic public traces are files: no truncation, pagination or private weights.
                for path in feedback_root.glob('*.jsonl'):
                    if '.private.' not in path.name:
                        sandbox.filesystem.write_text(path.read_text(),destination+'/'+path.name)
                public_report = {k:v for k,v in report.items() if k!='rows'}
                public_report['rows'] = [{k:v for k,v in row.items() if k not in ('private_trace_sha256','action_chain_sha256','seconds')}
                                         for row in report['rows']]
                sandbox.filesystem.write_text(canonical(public_report),destination+'/report.json')
                feedback = ('\nComplete feedback from your previous frozen bot is in '+destination+
                            '/report.json and its six JSONL trace files. Read and analyze them before revising. '
                            'Each line is your full public observation and chosen action; opponent private state is withheld. '
                            'Feedback evaluation does not consume coding time.\n'+canonical({k:v for k,v in public_report.items() if k!='rows'}))
            volume.commit()
        sandbox.terminate();sandbox=None
        status.update(status='final_evaluation');write_json(root/'status.json',status);volume.commit()
        # Shared hidden evaluation is cohort-owned and begins only after ALL builds
        # are frozen. No per-run random final maps or extra diagnostic feedback.
        final=None
        final_path = root/f'stage-{track["revision_rounds"]}'
        (root/'bot.wasm').write_bytes((final_path/'bot.wasm').read_bytes())
        (root/'main.rs').write_bytes((final_path/'main.rs').read_bytes())
        costs = []; calls = []
        for i in range(len(stages)):
            path = root/f'stage-{i}.trajectory.json'
            info = json.loads(path.read_text())['info']['model_stats'] if path.exists() else {}
            costs.append(info.get('instance_cost',0));calls.append(info.get('api_calls',0))
        status.update(status='completed',stages=stages,final=final,submission_quality=stages[-1]['submission_quality'],api_cost_usd=sum(costs),model_calls=sum(calls),
                      api_cost_accounting='saved completed calls only; in-flight request billing may be additional',
                      artifact_root=str(root), final_source_sha256=sha((root/'main.rs').read_bytes()),
                      final_wasm_sha256=sha((root/'bot.wasm').read_bytes()))
        status['protocol_sha256']=sha((root/'protocol.json').read_bytes())
        status['trajectory_sha256']={p.name:sha(p.read_bytes()) for p in root.glob('*.trajectory.json')}
        write_json(root/'report.json',status);write_json(root/'status.json',status);volume.commit()
        return status
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {str(error)[:500]}')
        write_json(root/'status.json',status);volume.commit()
        raise
    finally:
        if sandbox is not None: sandbox.terminate()


def load_submission(run_id):
    root = Path('/results/mini-runs')/identifier(run_id)
    report = json.loads((root/'report.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    if report['status'] != 'completed':
        raise ValueError('entrant has no valid final submission')
    if sha((root/'main.rs').read_bytes()) != report['final_source_sha256']:
        raise ValueError('frozen source mismatch')
    if sha((root/'bot.wasm').read_bytes()) != report['final_wasm_sha256']:
        raise ValueError('frozen Wasm mismatch')
    if sha((root/'protocol.json').read_bytes()) != report['protocol_sha256']:
        raise ValueError('protocol mismatch')
    for name,digest in report['trajectory_sha256'].items():
        if sha((root/name).read_bytes())!=digest: raise ValueError('trajectory mismatch')
    return root,report,protocol


def mini_tournament(tournament_id, run_ids, volume, pair_runner):
    from .tournament import summarize
    identifier(tournament_id)
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError('distinct submissions required')
    entrants = {name:load_submission(name) for name in run_ids}
    protocols = [x[2] for x in entrants.values()]
    keys = ('schema','harness_version','track','smoke','rules_sha256','system_sha256',
            'source_files','map_types','size','players','runtime','reference_sha256',
            'tools_sha256','authoring_tools')
    signature = lambda p: canonical({k:p[k] for k in keys})
    if any(signature(p)!=signature(protocols[0]) for p in protocols[1:]):
        raise ValueError('different authoring or runtime tracks cannot share a benchmark ladder')
    root = Path('/results/mini-tournaments')/tournament_id
    root.mkdir(parents=True,exist_ok=True)
    # Draw shared held-out maps only after every entrant is frozen and verified.
    labels = [REFERENCE,*run_ids]
    digest=sha(signature(protocols[0]).encode())
    if (root/'plan.json').exists():
        plan=json.loads((root/'plan.json').read_text())
        if plan['labels']!=labels or plan['protocol_sha256']!=digest:
            raise ValueError('existing tournament plan has a different roster or protocol')
        if (root/'report.json').exists():return json.loads((root/'report.json').read_text())
        groups=plan['groups']
    else:
        seeds = secrets.SystemRandom().sample(range(1_000_000_000,2_000_000_000),3)
        groups = [dict(tournament_id=tournament_id,a=a,b=b,seeds=seeds)
                  for i,a in enumerate(labels) for b in labels[i+1:]]
        write_json(root/'plan.json',dict(labels=labels,groups=groups,protocol_sha256=digest))
    volume.commit()
    rows = [r for part in pair_runner(groups) for r in part]
    rows.sort(key=lambda r:(r['a'],r['b'],r['map_type'],r['seed'],r['seat']))
    report = dict(schema=SCHEMA,track=protocols[0]['track'],smoke=protocols[0]['smoke'],
                  tournament_id=tournament_id,status='completed',**summarize(rows,labels))
    write_json(root/'games.json',rows);write_json(root/'report.json',report);volume.commit()
    return report


def mini_pair(payload, volume):
    a,b=payload['a'],payload['b']
    root = Path('/results/mini-tournaments')/identifier(payload['tournament_id'])/(a+'__'+b)
    entrants = {label:load_submission(label) for label in (a,b) if label != REFERENCE}
    budget = Budget(**next(iter(entrants.values()))[2]['runtime'])
    reference = warm_reference('/results/references/hero-1v1/final.npz',REFERENCE_SHA) if REFERENCE in (a,b) else None
    def bot(label):
        return reference if label==REFERENCE else WasiBot(entrants[label][0]/'bot.wasm',budget.fuel,budget.move_seconds)
    rows=[]
    for i,job in enumerate(evaluation_schedule(payload['seeds'])):
        row=play_game(bot(a),bot(b),job['seed'],job['seat'],budget,root/f'{i:02d}.jsonl',
                      phase='final',labels=[a,b],map_type=job['map_type'])
        rows.append(row)
        write_json(root/'progress.json',dict(completed=len(rows),total=6));volume.commit()
    write_json(root/'games.json',rows);volume.commit()
    return rows


def audit_trace(row, path):
    """Replay every action in the actual Rust engine, verifying the public trace and hash chain."""
    from dataclasses import asdict
    from .game import reset_map
    from polytopia_bot.studio.environment import RustEnvironment
    env=RustEnvironment(size=14)
    public=iter(json.loads(x) for x in path.read_text().splitlines())
    private=path.with_suffix('.private.jsonl')
    assert sha(path.read_bytes())==row['trace_sha256']
    assert sha(private.read_bytes())==row['private_trace_sha256']
    count=0;chain='0'*64
    try:
        reset_map(env,row['seed'],row['map_type'])
        for line in private.read_text().splitlines():
            entry=json.loads(line);expected_chain=entry.pop('chain')
            chain=sha((chain+canonical(entry)).encode());assert chain==expected_chain
            obs=env.observe(int(env.state.current_player))
            assert obs.player==entry['player'] and obs.state_version==entry['version']
            assert obs.visible_state['turn']==entry['turn']
            if obs.player==row['seat']:
                assert next(public)==dict(observation=asdict(obs),action_id=entry['action_id'])
            env.step(action_id=entry['action_id'],expected_state_version=obs.state_version);count+=1
        assert next(public,None) is None
        assert count==row['actions'] and chain==row['action_chain_sha256']
        assert int(env.state.turn)==row['turns']
        if row['status']=='native':
            assert bool(env.state.terminated)
            assert list(env.games.batch.terminal_rewards()[0])[row['winner']-1]>0
        elif row['status']=='draw':
            assert row['winner'] is None and row['reason'] in ('engine_truncation','action_limit')
            if row['reason']=='engine_truncation': assert bool(env.state.truncated)
        elif row['status']=='forfeit':
            assert not bool(env.state.terminated) and row['winner']==3-row['fault_player']
    finally:env.close()
    return count
