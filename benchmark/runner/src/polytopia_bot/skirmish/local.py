"""Local parallel benchmark: host Mini controllers, offline coder/referee containers.

No Modal execution. CPU queue waits are recorded and excluded from coding time.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import uuid

from .common import Budget, canonical, identifier, sha, write_json
from .cohort import make_evaluation_plan, summarize_evaluation, validate_manifest
from .docker_sandbox import DockerSandbox, LocalVolume, docker
from .inference import endpoint
from .local_scheduler import cpu_slot, StageClock, WATCHDOG
from .mini import TRACKS, freeze, stop_coder, evaluation_schedule
from .lifecycle import CAPTURE, capture_cutoff

ROOT = Path(__file__).resolve().parents[3]
CODER_IMAGE = 'polytopia-skirmish-coder:local-v1'
REFEREE_IMAGE = 'polytopia-skirmish-referee:local-v1'
LOCAL_SCHEMA = 'polytopia-local-active-clock-v1'


def image_id(tag):
    return subprocess.run(docker('image', 'inspect', '--format', '{{.Id}}', tag),
                          check=True, capture_output=True, text=True).stdout.strip()


def sources():
    paths = [*(ROOT/'src/polytopia_bot').rglob('*.py'),
             *(ROOT/'engine-rs/src').glob('*.rs'), *(ROOT/'skirmish/mini').glob('*.md'),
             ROOT/'engine-rs/Cargo.toml', ROOT/'engine-rs/Cargo.lock',
             ROOT/'src/polytopia_bot/gpu/rules.json', ROOT/'scripts/build_skirmish_local.py',
             ROOT/'scripts/run_skirmish_local.py',ROOT/'scripts/run_skirmish_v6.py',ROOT/'scripts/build_skirmish_v6.py',ROOT/'scripts/regrade_skirmish_v6.py',*(ROOT/'skirmish/v6').glob('*.md')]
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted(paths)}


def checked_references(manifest, directory):
    refs = manifest['references']
    for ref in refs:
        if ref['kind']=='scripted':
            if sha((ROOT/'src/polytopia_bot/skirmish/reference_panel.py').read_bytes()) != ref['source_sha256']:
                raise ValueError('Scripted reference source changed')
            continue
        if Path(ref['file']).name != ref['file']:
            raise ValueError('Reference file must be a basename')
        digest = ref.get('checkpoint_sha256', ref.get('wasm_sha256'))
        if sha((Path(directory)/ref['file']).read_bytes()) != digest:
            raise ValueError('Reference digest mismatch: '+ref['label'])
        if ref['kind'] in ('expert','hybrid') and sha((Path(directory)/ref['file']).with_suffix('.json').read_bytes()) != ref['config_sha256']:
            raise ValueError('Reference configuration digest mismatch')
    return refs


def worker_payloads(root, jobs):
    for job in jobs:
        yield dict(job, output_relative=f'games/{job["index"]:06d}')


def referee(root, refs_dir, jobs, image, slots, count, *, seconds=900, evaluation=None):
    payload = dict(jobs=jobs, references=[], referee_seconds=seconds, evaluation=evaluation or {})
    # The trusted referee can see private references and traces. The coder cannot.
    with cpu_slot(slots, count):
        name = 'polytopia-referee-'+uuid.uuid4().hex
        try:
            proc = subprocess.run(docker('run', '--rm', '-i', '--name', name,
                '--label', 'polytopia.skirmish.role=referee', '--network', 'none', '--cpus', '2',
                '--memory', '4g', '--memory-swap', '4g', '--pids-limit', '512', '--cap-drop', 'ALL',
                '--security-opt', 'no-new-privileges=true',
                '--mount', f'type=bind,src={root.resolve()},dst=/results',
                '--mount', f'type=bind,src={Path(refs_dir).resolve()},dst=/refs,readonly',
                image, 'python', '-m', 'polytopia_bot.skirmish.local_eval'),
                input=json.dumps(payload), capture_output=True, text=True,
                timeout=(seconds+120)*max(1, len(jobs)))
        finally:
            subprocess.run(docker('rm', '-f', name), capture_output=True, timeout=30)
    if proc.returncode:
        raise RuntimeError('Local referee failed: '+proc.stderr[-1500:])
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith('SKIRMISH_RESULT='):
            return json.loads(line.removeprefix('SKIRMISH_RESULT='))
    raise RuntimeError('Local referee produced no result receipt')




def public_feedback_row(row, destination, index):
    """Publish only game-visible fields and paths mounted in the author sandbox."""
    fields = ('a', 'b', 'seed', 'seat', 'map_type', 'phase', 'status', 'reason',
              'winner', 'result', 'turns', 'actions', 'action_counts', 'cap_kind',
              'evaluation', 'trace_sha256', 'model_label', 'track', 'replicate',
              'stage', 'opponent_label', 'submission_quality')
    public = {key: row[key] for key in fields if key in row}
    filename = f'{index:02d}.jsonl'
    return dict(public, trace_file=filename, trace_path=f'{destination}/{filename}')


def signal_worker_group(process,sig):
    try:os.killpg(process.pid,sig)
    except ProcessLookupError:pass


def run_stage(sandbox, root, stage, payload, slots, count, *, stop_event=None):
    from .telemetry import summarize_timing
    ready, start, timing = [root/f'stage-{stage}.{kind}' for kind in ('ready.json', 'start.json', 'timing.jsonl')]
    payload.update(sandbox_backend=sandbox.backend, sandbox_id=sandbox.object_id,
        ready_path=str(ready), start_path=str(start), timing=str(timing),
        cpu_slots_directory=str(slots) if slots else None, cpu_slots=count)
    sandbox.exec('rm', '-f', '/verify/stage-start.json', '/verify/watchdog.pid').wait()
    launched = time.time(); watchdog = None
    snapshots=[x for x in payload.get('budget_snapshots',[]) if 0<x<1] if payload.get('v6_track',{}).get('revision_rounds')==0 else []
    captured=set()
    with (root/f'stage-{stage}.worker.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'polytopia_bot.skirmish.mini_worker'],
            stdin=subprocess.PIPE, stdout=log, stderr=log, text=True,
            env={**os.environ,'PYTHONPATH':str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')},start_new_session=True)
        try:
            process.stdin.write(json.dumps(payload)); process.stdin.close()
            while not ready.exists():
                if process.poll() is not None:
                    raise RuntimeError('Local Mini worker failed before readiness')
                if time.time()-launched > 120:
                    raise TimeoutError('Local Mini readiness deadline')
                time.sleep(.02)
            watchdog = sandbox.exec('python', '-c', WATCHDOG+f'\nstage={stage}\n'+CAPTURE, timeout=14400)
            check = sandbox.exec('python', '-c', "import time;from pathlib import Path\nwhile not Path('/verify/watchdog.pid').exists():time.sleep(.01)", timeout=60)
            check.wait()
            if check.returncode:
                raise RuntimeError('Local cutoff watchdog did not become ready')
            started = time.time()
            clock = StageClock(start, sandbox)
            clock.save(dict(started_unix=started, deadline_unix=started+payload['seconds'],
                            stage=stage, scheduler_wait_seconds=0))
            reason = 'agent_exit'
            while process.poll() is None:
                state = clock.read()
                deadline = state.get('pause_lease_unix', state['deadline_unix'])
                active_elapsed=time.time()-started-state.get('scheduler_wait_seconds',0)
                if 'paused_unix' in state: active_elapsed-=time.time()-state['paused_unix']
                for fraction in snapshots:
                    if fraction not in captured and active_elapsed>=payload['seconds']*fraction:
                        snap=root/f'budget-{int(fraction*100)}';snap.mkdir(parents=True,exist_ok=True)
                        snap.joinpath('main.rs').write_bytes(sandbox.filesystem.read_bytes('/bot/src/main.rs'))
                        write_json(snap/'sample.json',dict(fraction=fraction,active_seconds=active_elapsed,source_sha256=sha(snap.joinpath('main.rs').read_bytes())))
                        captured.add(fraction)
                if time.time() >= deadline or (stop_event is not None and stop_event.is_set()):
                    signal_worker_group(process,signal.SIGTERM)
                    try:process.wait(timeout=3)
                    except subprocess.TimeoutExpired:signal_worker_group(process,signal.SIGKILL);process.wait()
                    reason='cancelled' if stop_event is not None and stop_event.is_set() else 'deadline';break
                time.sleep(.02)
            ended = time.time()
            if reason != 'deadline' and process.returncode:
                reason = 'agent_error'
            stop_coder(sandbox); capture_cutoff(sandbox, stage)
            sandbox.exec('python', '-c', "import os,signal;from pathlib import Path;p=Path('/verify/watchdog.pid');os.kill(int(p.read_text()),signal.SIGKILL) if p.exists() else None").wait()
            state = clock.read()
            excluded = state.get('scheduler_wait_seconds', 0)
            if 'paused_unix' in state:
                excluded += max(0, ended-state['paused_unix'])
            metrics = summarize_timing(timing, ended)
            metrics.update(scheduler_wait_seconds=excluded, controller_wall_seconds=ended-started,
                           coding_seconds=min(payload['seconds'], max(0, ended-started-excluded)))
            return reason, metrics
        finally:
            if process.poll() is None:
                signal_worker_group(process,signal.SIGKILL);process.wait()


def task(stage, seconds, rounds, feedback):
    return (f'Build the strongest Bardur Polytopia 1v1 Rust bot. Stage {stage} of {rounds+1}; '
        f'{seconds} seconds of coding time. Read /reference/QUICKSTART.md, /reference/RULES.md, '
        '/reference/PROTOCOL.md and /reference/TOOLS.md. Edit only /bot/src/main.rs; std and serde_json only. '
        'Use bash for all actions. Compile with cargo build --release --offline --locked --target wasm32-wasip1. '
        'Use skirmish --help for nine public maps, real-engine inspection, protocol checking and short action replays. '
        'You may run self-written tests. There is no internet, supplied public opponent or automatic practice-match runner. '
        'The current source is frozen at the deadline and independently compiled with the original manifest. '
        'Broken current source fails; there is no rollback. Each model response allows 32768 output tokens including reasoning. '
        'Split large file writes across calls. End a stage early using bash: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT. '
        'Time waiting for a host CPU slot is excluded from your budget; API/model latency and your tool execution count. '
        + ('No expert-game feedback is available during this independent attempt.' if not rounds else
           'After each of the first three stages, you receive six complete public traces against the expert: '
           'drylands, lakes and continents in both seats. Then revise in the next stage.') + feedback)



def v6_task(stage,track,seconds,config,feedback):
    stages=track['revision_rounds']+1
    common=(f'Build the strongest Bardur Polytopia 1v1 Rust bot. Stage {stage+1} of {stages}; '
        f'{seconds} active coding seconds in this stage. Read /reference/QUICKSTART.md, /reference/RULES.md, '
        '/reference/PROTOCOL.md and /reference/TOOLS.md. Edit only /bot/src/main.rs; Rust std and serde_json only. '
        'Use bash; compile offline with cargo build --release --offline --locked --target wasm32-wasip1. '
        'There is no internet, opponent implementation, or automatic practice runner. Self-written tests consume time. '
        'Current source is frozen at cutoff and independently rebuilt; broken current source fails without rollback. '
        'Each model response permits 32768 tokens including reasoning. API/model latency and tool execution consume time; '
        'host CPU queue waits do not. Conversation and workspace persist across stages. '
        'To finish early run only: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT. Unused stage time is not transferred. '
        'Core play is 14x14. Read visible_state.board_size rather than hardcoding coordinates; separate transfer diagnostics use 11/18. '
        'Runtime limits: '+canonical(config['evaluation'])+'. ')
    treatment=('After each of stages 1,2,3, six mixed-reference games (three map families, both seats) supply public traces; '
               'feedback generation is outside your clock, reading is inside it. ' if track['supplied_feedback'] else
               'No opponent feedback is supplied at any stage. ')
    return common+treatment+feedback


def trial(root, cell, refs_dir, coder, evaluator, slots, count, source_receipt):
    run_root = root/'runs'/cell['run_id']; run_root.mkdir(parents=True, exist_ok=False)
    config = cell['config']; track = dict(config.get('v6_track') or TRACKS[cell['track']])
    if config.get('smoke'):
        track['stage_seconds'] = config['smoke_seconds']
    status = dict(status='setting_up', run_id=cell['run_id'], stages=[])
    sandbox = None
    write_json(run_root/'config.json', config)
    try:
        if sources() != source_receipt:
            raise ValueError('Host authoring source changed after cohort allocation')
        sandbox = DockerSandbox.create(coder)
        route = endpoint(config)
        protocol = dict(schema='polytopia-local-v6' if config.get('v6_track') else LOCAL_SCHEMA, execution_backend='local-docker', harness='mini-swe-agent',
            harness_version='2.4.6', inference=route.receipt(), model=cell['model'], parameters=config['parameters'],
            track=cell['track'], **track, source_files=source_receipt, coder_image=coder,
            referee_image=evaluator, sandbox_id=sandbox.object_id, network=False,
            cpu=2, memory_mib=2048, cpu_slots=count, clock='active wall excluding scheduler waits',
            smoke=bool(config.get('smoke')))
        write_json(run_root/'protocol.json', protocol)
        system = (ROOT/'skirmish/mini/SYSTEM.md').read_text()
        history = None; feedback = ''; stages = []
        for stage in range(track['revision_rounds']+1):
            seconds = track['stage_seconds']
            payload = dict(**config, system=system, task=task(stage, seconds, track['revision_rounds'] if track.get('supplied_feedback',True) else 0, feedback),
                seconds=seconds, trajectory=str(run_root/f'stage-{stage}.trajectory.json'), history=history)
            if config.get('v6_track'):payload['task']=v6_task(stage,track,seconds,config,feedback)
            status.update(status='authoring', stage=stage); write_json(run_root/'status.json', status)
            reason, timing = run_stage(sandbox, run_root, stage, payload, slots, count)
            with cpu_slot(slots, count):
                receipt = freeze(sandbox, run_root/f'stage-{stage}', LocalVolume(), cutoff=f'/verify/cutoffs/stage-{stage}')
            usage=dict(input_tokens=0,output_tokens=0,output_limit_hits=0,completed_response_receipts=0)
            trajectory=run_root/f'stage-{stage}.trajectory.json'
            if trajectory.exists():
                data=json.loads(trajectory.read_text())
                prior=[] if history is None else json.loads(Path(history).read_text()).get('messages',[])
                previous_ids={m.get('extra',{}).get('response',{}).get('id') for m in prior if m.get('extra',{}).get('response',{}).get('id')}
                for message in data.get('messages',[]):
                    response=message.get('extra',{}).get('response',{})
                    if not response or response.get('id') in previous_ids:continue
                    u=response.get('usage') or {};usage['input_tokens']+=u.get('prompt_tokens',0);usage['output_tokens']+=u.get('completion_tokens',0)
                    usage['completed_response_receipts']+=1
                    usage['output_limit_hits']+=any(c.get('finish_reason')=='length' for c in response.get('choices',[]))
            stages.append(dict(stage=stage,end_reason=reason,timing=timing,usage=usage,**receipt))
            status['stages'] = stages; write_json(run_root/'stages.json', stages)
            if reason == 'agent_error':
                raise RuntimeError('Local Mini worker error; preserve artifacts and do not score an infrastructure failure')
            history=str(run_root/f'stage-{stage}.trajectory.json') if (run_root/f'stage-{stage}.trajectory.json').exists() else history
            if receipt['returncode']:
                feedback = '\nFrozen source failed to compile:\n'+receipt['compile_output']
            elif stage < track['revision_rounds'] and track.get('supplied_feedback',True):
                candidate = dict(kind='submission', run_id=cell['run_id'], stage=stage,
                    label='candidate', model_label=cell['model_label'], track=cell['track'],
                    replicate=cell['replicate'], submission_quality=receipt['submission_quality'],
                    wasm_sha256=receipt['wasm_sha256'])
                panel=json.loads((root/'allocation.json').read_text())['references']
                expert=next(r for r in panel if r['kind']=='expert')
                feedback_panel=[r for r in panel if r['kind']=='hybrid']+[expert]
                feedback_evaluation=config.get('evaluation')
                jobs = []
                for i, position in enumerate(evaluation_schedule([500000+stage*30+j for j in range(3)])):
                    opponent=feedback_panel[(stage+i//2)%len(feedback_panel)] if config.get('v6_track') else expert
                    job = dict(index=i, candidate=candidate, opponent=opponent, phase='practice', **position,
                        evaluation=config.get('evaluation',{}),
                        output_relative=f'feedback/{cell["run_id"]}/round-{stage}/{i:02d}')
                    job['job_sha256'] = sha(canonical(job).encode()); jobs.append(job)
                status.update(status='feedback_evaluation'); write_json(run_root/'status.json', status)
                rows = referee(root, refs_dir, jobs, evaluator, slots, count, seconds=feedback_evaluation['game_seconds'] if config.get('v6_track') else 180, evaluation=feedback_evaluation)
                destination = f'/feedback/round-{stage}'
                public = []
                for i, row in enumerate(rows):
                    path = root/row['trace_relative']
                    sandbox.filesystem.write_bytes(path.read_bytes(), f'{destination}/{i:02d}.jsonl')
                    public.append(public_feedback_row(row, destination, i))
                sandbox.filesystem.write_text(json.dumps(dict(rows=public)), destination+'/report.json')
                feedback = '\nPrevious frozen bot feedback is in '+destination+'/report.json and six full public JSONL traces.'
        if config.get('v6_track'):
            snapshots=[]
            for snap in sorted(run_root.glob('budget-*')):
                with cpu_slot(slots,count):
                    fresh=DockerSandbox.create(coder)
                    try:proof=freeze(fresh,snap,LocalVolume())
                    finally:fresh.terminate()
                sampled=json.loads((snap/'sample.json').read_text())
                snapshots.append(dict({**sampled,**proof},snapshot=int(snap.name.split('-')[1]),stage=0))
            status['budget_snapshots']=snapshots
        status.update(status='completed' if stages[-1]['returncode']==0 else 'failed_submission',
                      source_sha256=stages[-1]['source_sha256'])
        write_json(run_root/'report.json', status); write_json(run_root/'status.json', status)
        return status
    except Exception as error:
        status.update(status='failed_harness', error=f'{type(error).__name__}: {str(error)[:500]}')
        write_json(run_root/'status.json', status)
        raise
    finally:
        if sandbox is not None:
            sandbox.terminate()


def local_cells(prefix, manifest, *, smoke=False):
    validate_manifest(manifest)
    cells = []
    for rep in range(1 if smoke else manifest['replicates']):
        for entry in manifest['models']:
            for track in manifest['tracks']:
                # Explicit OpenRouter for other vendors; local subscription default
                # for every OpenAI/Anthropic model, including Fable.
                provider = 'auto' if entry['model'].startswith(('openai/', 'anthropic/')) else 'openrouter'
                cfg = dict(provider=provider, model=entry['model'], parameters=entry['parameters'], track=track)
                if smoke:
                    cfg.update(smoke=True, smoke_seconds=30)
                run_id = identifier(f'{prefix}-{entry["label"]}-r{rep+1}-{track}')
                cells.append(dict(run_id=run_id, model_label=entry['label'], model=entry['model'],
                                  replicate=rep+1, track=track, config=cfg))
    return cells


def rebuild_all(root, cells, runs, coder, slots, count):
    def rebuild(item):
        cell, stage = item
        snapshot = root/'runs'/cell['run_id']/(f'budget-{stage["snapshot"]}' if 'snapshot' in stage else f'stage-{stage["stage"]}')
        with cpu_slot(slots, count):
            sandbox = DockerSandbox.create(coder)
            try:
                destination=root/'rebuilds'/cell['run_id']/snapshot.name
                destination.mkdir(parents=True,exist_ok=True)
                source=(snapshot/'main.rs').read_bytes()
                if (destination/'main.rs').exists():
                    if sha((destination/'main.rs').read_bytes())!=sha(source):raise ValueError('rebuild source drift')
                else:(destination/'main.rs').write_bytes(source)
                # Root has no DAC override in the hardened coder container. The
                # verifier owns /verify; compile the immutable captured file there.
                proof=freeze(sandbox,destination,LocalVolume())
                if proof['source_sha256'] != stage['source_sha256'] or proof.get('wasm_sha256') != stage.get('wasm_sha256'):
                    raise ValueError('Independent rebuild mismatch')
            finally:
                sandbox.terminate()
    with ThreadPoolExecutor(max_workers=count) as pool:
        list(pool.map(rebuild, [(c,s) for c in cells for s in runs[c['run_id']]['stages']+runs[c['run_id']].get('budget_snapshots',[])]))


def cohort(prefix, manifest, directory, refs_dir, *, concurrency=24, cpu_slots=6, smoke=False, evaluate_only=False):
    identifier(prefix)
    if not 1 <= concurrency <= 96 or not 1 <= cpu_slots <= 32:
        raise ValueError('Invalid local concurrency')
    root = Path(directory).resolve()/prefix
    coder, evaluator = image_id(CODER_IMAGE), image_id(REFEREE_IMAGE)
    info = json.loads(subprocess.run(docker('info', '--format', '{{json .}}'),
                                    check=True, capture_output=True, text=True).stdout)
    if 2*cpu_slots > info['NCPU'] or 4*cpu_slots*1024**3 > info['MemTotal']:
        raise ValueError('CPU worker slots exceed local VM resources; authoring session count can remain high')
    references = checked_references(manifest, refs_dir)
    slots = Path.home()/'.cache/polytopia-skirmish/cpu-slots'
    cells = local_cells(prefix, manifest, smoke=smoke)
    fingerprint = sources()
    allocation = dict(schema=LOCAL_SCHEMA, manifest=manifest, references=references, cells=cells,
        coder_image=coder, referee_image=evaluator, source_files=fingerprint, cpu_slots=cpu_slots,
        concurrency=concurrency, smoke=smoke)
    if evaluate_only:
        old = json.loads((root/'allocation.json').read_text())
        if old != allocation:
            raise ValueError('Local cohort source/config/image changed; refusing to resume')
        runs = {cell['run_id']: json.loads((root/'runs'/cell['run_id']/'report.json').read_text()) for cell in cells}
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root/'allocation.json', allocation)
        # All credentials and proxy catalog checks belong outside coding clocks.
        for cell in cells:
            endpoint(cell['config']).credential()
        runs = {}
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(trial, root, cell, refs_dir, coder, evaluator, slots, cpu_slots, fingerprint):cell for cell in cells}
            for future in as_completed(futures):
                cell = futures[future]
                try:
                    runs[cell['run_id']] = future.result()
                except Exception as error:
                    runs[cell['run_id']] = dict(status='failed_harness', error=str(error))
                write_json(root/'progress.json', dict(status='authoring', completed=len(runs), total=len(cells), runs=runs))
    if any(r['status'] not in ('completed','failed_submission') for r in runs.values()):
        raise RuntimeError('Authoring has infrastructure failures; no hidden maps or rankings allocated')
    if not (root/'evaluation-plan.json').exists():
        rebuild_all(root, cells, runs, coder, slots, cpu_slots)
        if sources() != fingerprint:
            raise ValueError('Source changed before hidden-map allocation')
        rng = secrets.SystemRandom(); maps = []
        for family in ('drylands','lakes','continents'):
            for seed in rng.sample(range(1_000_000_000, 2_000_000_000), 1 if smoke else 5):
                maps.append(dict(map_type=family, seed=seed))
        plan = make_evaluation_plan(cells, runs, maps, references, smoke=smoke)
        write_json(root/'evaluation-plan.json', plan)
    plan = json.loads((root/'evaluation-plan.json').read_text())
    jobs = list(worker_payloads(root, plan['jobs'])); rows = []
    # Batch jobs to amortize container/JAX startup, while maintaining six CPU slots.
    batches = [jobs[i:i+24] for i in range(0,len(jobs),24)]
    with ThreadPoolExecutor(max_workers=cpu_slots) as pool:
        futures = [pool.submit(referee, root, refs_dir, batch, evaluator, slots, cpu_slots) for batch in batches]
        for future in as_completed(futures):
            rows.extend(future.result())
            write_json(root/'evaluation-progress.json', dict(completed=len(rows), total=len(jobs)))
    report = summarize_evaluation(plan, cells, rows)
    from .tournament import fit_elo
    ratings = {}
    for track in manifest['tracks']:
        pooled = []
        for row in rows:
            job = plan['jobs'][row['job_index']]
            a, b = job['candidate'], job['opponent']
            if a['track']!=track or row['status']=='invalid_submission' or a['stage']!=(3 if track=='feedback-3r-40m' else 0):
                continue
            pooled.append(dict(row, a=a['model_label'], b=b.get('model_label',b['label'])))
        ratings[track] = fit_elo(pooled, [m['label'] for m in manifest['models']]+[r['label'] for r in references])
    report['secondary_elo'] = ratings
    report.update(execution_backend='local-docker', local_schema=LOCAL_SCHEMA, source_files=fingerprint)
    write_json(root/'games.json', sorted(rows,key=lambda r:r['job_index']))
    write_json(root/'report.json', report)
    write_json(root/'progress.json', dict(status=report['status'], games=len(rows), total=len(jobs)))
    return report
