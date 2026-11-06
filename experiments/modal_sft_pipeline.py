"""Configurable, restart-safe expert-SFT pipeline on Modal: generate -> prepare -> train -> evaluate -> summarize.

  modal run --env staging -m experiments.modal_sft_pipeline --config configs/sft-pipeline/smoke.json --run-id sftp-smoke-1
  modal run --detach --env staging -m experiments.modal_sft_pipeline --config configs/sft-pipeline/NAME.json --run-id RUN
  modal run --detach --env staging -m experiments.modal_sft_pipeline --action resume --run-id RUN [--retry STAGE]
  modal run --env staging -m experiments.modal_sft_pipeline --action status --run-id RUN
  python -m experiments.modal_sft_pipeline estimate configs/sft-pipeline/NAME.json      (local, no Modal calls)

Everything a run needs is fixed at launch in ``runs/RUN/protocol.json`` (config, jobs, hashes of every
source/config file) plus a snapshot of src/, scripts/ and configs/ uploaded next to it; every worker
overlays that snapshot onto /workspace first, so a resumed run executes the launch-time code even if the
checkout changed. Each stage is skipped when its verified output exists. The coordinator persists every
spawned call ID (calls.json) before waiting and reattaches to live calls after a restart; it never spawns
a second copy of a worker that may still be running. Status: ``RUN/status.json``.
See docs/sft-pipeline-config-2026-10-19.md.
"""
import concurrent.futures
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time
import traceback
import modal
from experiments.modal_thinking_repaired import image, volume, ROOT
from experiments.modal_expert_sft import training_image

app = modal.App('polytopia-sft-pipeline')
retained = modal.Volume.from_name('polytopia-structured-rl-retained', environment_name='staging')
RUNS = Path('/workspace/runs')
VOLUMES = {'/workspace/runs': volume, '/retained': retained.read_only()}
SNAPSHOT_DIRS = ('src', 'scripts', 'configs')
_cache = {}


# ---------------------------------------------------------------- shared helpers

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str)+'\n')
    tmp.replace(path)


def commit(reload=False):
    """commit()/reload() with backoff: the shared volume sometimes answers 'too many layers'."""
    for attempt in range(40):
        try:
            return volume.reload() if reload else volume.commit()
        except (modal.exception.ResourceExhaustedError, modal.exception.ServiceError, modal.exception.InternalError) as exc:
            print(f'volume {"reload" if reload else "commit"} retry {attempt}: {exc!r}', flush=True)
            time.sleep(min(60, 5*(attempt+1)))
    raise RuntimeError('volume commit kept failing')


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def load(run_id, protocol_sha):
    """Check the protocol hash and overlay the launch-time source snapshot onto /workspace (once per container)."""
    root = RUNS/run_id
    if not (root/'protocol.json').exists():
        commit(reload=True)
    if sha(root/'protocol.json') != protocol_sha:
        raise ValueError('protocol changed')
    p = json.loads((root/'protocol.json').read_text())
    snap = p['source_snapshot']
    if _cache.get('snapshot') not in (None, snap['sha256']):
        raise RuntimeError('container already overlaid with another snapshot')
    if _cache.get('snapshot') is None:
        if sha(root/snap['path']) != snap['sha256']:
            raise ValueError('source snapshot changed')
        with tarfile.open(root/snap['path']) as tar:
            tar.extractall('/workspace', filter='data')
        for rel, h in p['source_hashes'].items():
            if sha(Path('/workspace')/rel) != h:
                raise ValueError(f'source overlay mismatch: {rel}')
        if '/workspace/src' not in sys.path:
            sys.path.insert(0, '/workspace/src')
        _cache['snapshot'] = snap['sha256']
    return root, p


def games_dir(p):
    return RUNS/(p['generation']['reuse_run'] or p['run_id'])/'generate'/'games'


def train_payload(root, p):
    t = p['training']
    return dict(prepared=str(root/'prepare'/'corpus'), output=str(root/'training'/'model'), config=t['config'], seed=t['seed'],
                benchmark=t['benchmark'], checkpoint_every=t['checkpoint_every'])


def resume_point(root, payload):
    """Newest checkpoint with trainer state in an archived training attempt (training.failed-*/) of this run whose
    input matches ``payload``, as {resume_from, step, attempt}; None when there is none (adapter-only checkpoints,
    written by a trainer without resume support, do not count). The trainer re-verifies every hash before using it."""
    best = None
    for ready in root.glob('training.failed-*/model/checkpoint-*.ready.json'):
        cp = ready.parent/ready.name[:-len('.ready.json')]
        attempt = ready.parent.parent
        given = read_json(attempt/'input.json', {})
        if any(given.get(k) != payload[k] for k in ('prepared', 'config', 'seed', 'benchmark')):
            continue
        meta = read_json(cp/'trainer_state.json')
        if not meta or not (cp/'trainer_state.pt').exists():
            continue
        key = (meta['step'], attempt.name)
        if best is None or key > best[0]:
            best = key, dict(resume_from=str(cp), step=meta['step'], attempt=attempt.name)
    return best and best[1]


def env_cached():
    if 'env' not in _cache:
        from polytopia_bot.learning.adapter import GPUEnvironment
        _cache['env'] = GPUEnvironment(14)
    return _cache['env']


# ---------------------------------------------------------------- workers

@app.function(image=image, cpu=2, memory=8192, timeout=4*3600, max_containers=200, retries=1, volumes=VOLUMES)
def generate_chunk(run_id, protocol_sha, job_ids):
    """Play a list of expert games; idempotent per game (result file = done)."""
    root, p = load(run_id, protocol_sha)
    from polytopia_bot.learning import sft_pipeline as sp
    g = p['generation']
    jobs = _cache.setdefault(('jobs', run_id), {j['id']: j for j in g['jobs']})
    renderer = dict(sp.resolve_renderer(p['prompt_renderer']), name=p['prompt_renderer']['name'])
    forbidden = frozenset(g['forbidden_symmetry_hashes'])
    out = []
    for jid in job_ids:
        done = root/'generate'/'results'/f'{jid}.json'
        if done.exists():
            out.append(dict(id=jid, resumed=True)); continue
        job = jobs[jid]
        try:
            expert = sp.make_player(p['players'][job['expert']], job['game_seed'])
            opponent = sp.make_player(p['players'][job['opponent']], job['game_seed']+1)
            result = sp.play_game(job, expert, opponent, env_cached(), root/'generate'/'games', renderer=renderer,
                                  record_trajectory=g['record_trajectory'], forbidden=forbidden)
        except sp.MapExcluded as exc:
            result = dict(job, status='excluded', reason=str(exc), seconds=0)
        except Exception:
            _cache.pop('env', None)  # never reuse an environment that raised mid-game
            atomic_json(root/'generate'/'failures'/f'{jid}.json', dict(id=jid, error=traceback.format_exc()[-3000:],
                                                                      observed_unix=time.time()))
            commit()
            out.append(dict(id=jid, failed=True)); continue
        atomic_json(done, result)
        commit()
        out.append(dict(id=jid, status=result['status']))
    return out


@app.function(image=image, cpu=4, memory=16384, timeout=4*3600, max_containers=100, retries=1, volumes=VOLUMES)
def prepare_shard(run_id, protocol_sha, index, job_ids):
    root, p = load(run_id, protocol_sha)
    commit(reload=True)
    sys.path.insert(0, '/workspace/scripts')
    from sft_pipeline_prepare import shard
    files = [games_dir(p)/f'{j}.jsonl.gz' for j in job_ids]
    result = shard(files, p['training']['config'], p['data'], p['prompt_renderer'],
                   root/'prepare'/'shards'/f'shard-{index:04d}.jsonl', p['fingerprints']['prepare'])
    commit()
    return {k: v for k, v in result.items() if k not in ('families',)}


@app.function(image=image, cpu=4, memory=65536, timeout=8*3600, retries=0, volumes=VOLUMES)
def combine_corpus(run_id, protocol_sha, n_shards):
    root, p = load(run_id, protocol_sha)
    commit(reload=True)
    corpus = root/'prepare'/'corpus'
    if (corpus/'manifest.json').exists():
        return dict(read_json(corpus/'manifest.json'), resumed=True)
    sys.path.insert(0, '/workspace/scripts')
    from sft_pipeline_prepare import combine
    shards = [read_json(root/'prepare'/'shards'/f'shard-{i:04d}.json') for i in range(n_shards)]
    if any(s is None or s['fingerprint'] != p['fingerprints']['prepare'] for s in shards):
        raise ValueError('missing or stale prepared shard')
    manifest = combine(shards, corpus, p['training']['config'], data=p['data'],
        provenance=dict(fingerprint=p['fingerprints']['prepare'], experts={n: p['players'][n] for n in p['generation']['experts']},
                        prompt_renderer=p['prompt_renderer'], data_protocol=p['data'], run_id=run_id))
    commit()
    return manifest


@app.function(image=training_image, gpu='B200:4', cpu=16, memory=131072, timeout=24*3600, max_containers=1, retries=0,
              scaledown_window=2, volumes={'/workspace/runs': volume})
def train(run_id, protocol_sha):
    import subprocess
    root, p = load(run_id, protocol_sha)
    commit(reload=True)
    t, out = p['training'], root/'training'
    if (out/'execution.json').exists() and read_json(out/'model'/'run.json', {}).get('status') == 'completed':
        return dict(read_json(out/'execution.json'), resumed=True)
    out.mkdir(exist_ok=False)  # a leftover directory means a dead attempt: --retry train archives it
    payload = train_payload(root, p)
    resume = resume_point(root, payload)  # set only after --retry train archived an attempt with a usable checkpoint
    if resume:
        payload['resume_from'] = resume['resume_from']
        print('RESUMING', json.dumps(resume), flush=True)
    atomic_json(out/'input.json', payload); commit()
    started = time.monotonic()
    names = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True,
                           text=True, check=True).stdout.split('\n')
    names = [n for n in names if n.strip()]
    gpus = len(names)
    if gpus != t['gpus']:
        raise RuntimeError(f'expected {t["gpus"]} GPUs, container has {gpus}')
    proc = subprocess.Popen(['/workspace/training/.venv/bin/python', '-m', 'torch.distributed.run', '--standalone',
                             f'--nproc_per_node={gpus}', '/workspace/scripts/expert_train_distributed.py',
                             str(out/'input.json')], cwd='/workspace')
    try:
        last, seen = 0, set()
        while proc.poll() is None:  # commit once a minute, and as soon as a checkpoint is ready (so a lost call keeps it)
            ready = {f.name for f in (out/'model').glob('checkpoint-*.ready.json')} if (out/'model').exists() else set()
            if ready - seen or time.monotonic()-last > 60:
                commit(); last, seen = time.monotonic(), ready
            time.sleep(5)
        result = dict(exit_code=proc.returncode, occupied_seconds=time.monotonic()-started, finished_unix=time.time(),
                      gpus=gpus, gpu=names[0], resumed_from=resume)
        run = read_json(out/'model'/'run.json')
        if run:
            result.update(status=run['status'], train_seconds=run.get('train_seconds'), global_step=run.get('global_step'),
                          adapter_files=run.get('adapter_files'))
        atomic_json(out/'execution.json', result); commit()
        if proc.returncode:
            raise RuntimeError(result)
        return result
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=30)
        commit()


@app.function(image=image, gpu='H100', cpu=8, memory=49152, timeout=24*3600, max_containers=12, retries=0,
              scaledown_window=2, volumes=VOLUMES)
def evaluate(run_id, protocol_sha, arm):
    import subprocess
    root, p = load(run_id, protocol_sha)
    commit(reload=True)
    ev = root/'eval'
    if sha(ev/'protocol.json') != p['evaluation']['protocol_sha256']:
        raise ValueError('eval protocol changed')
    if (ev/arm/'report.json').exists():
        return dict(read_json(ev/f'{arm}-execution.json', {}), arm=arm, resumed=True)
    if (ev/arm).exists():
        raise RuntimeError(f'partial arm directory {arm}; --retry evaluate archives it')
    started = time.monotonic()
    proc = subprocess.Popen(['python', '/workspace/scripts/sft_pipeline_eval.py', str(ev), arm], cwd='/workspace')
    try:
        while proc.poll() is None:
            commit(); time.sleep(60)
        result = dict(arm=arm, exit_code=proc.returncode, worker_seconds=time.monotonic()-started, finished_unix=time.time())
        atomic_json(ev/f'{arm}-execution.json', result); commit()
        if proc.returncode:
            raise RuntimeError(result)
        return result
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=15)
        commit()


@app.function(image=image, cpu=2, memory=8192, timeout=4*3600, max_containers=60, retries=1, volumes=VOLUMES)
def reference_chunk(run_id, protocol_sha, player, job_ids):
    """An RL player (the reference) on evaluation jobs, through the same public interface; idempotent per game."""
    root, p = load(run_id, protocol_sha)
    from polytopia_bot.learning import sft_pipeline as sp
    ev = json.loads((root/'eval'/'protocol.json').read_text())
    jobs = {j['id']: j for j in ev['jobs']}
    out = root/'eval'/f'reference-{player}'
    rows = []
    for jid in job_ids:
        if (out/f'{jid}.json').exists():
            rows.append(read_json(out/f'{jid}.json')); continue
        job = dict(jobs[jid], game_seed=jobs[jid]['rng_index'])
        me = sp.make_player(p['players'][player], job['game_seed'])
        opp = sp.make_player(p['players'][job['opponent']], job['game_seed']+1)
        try:
            r = sp.play_game(job, me, opp, env_cached(), out/'games', renderer=None)
        except Exception:
            _cache.pop('env', None)
            raise
        atomic_json(out/f'{jid}.json', r); commit()
        rows.append(r)
    return [{k: r[k] for k in ('id', 'status', 'reward', 'round')} for r in rows]


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes=VOLUMES)
def summarize_eval(run_id, protocol_sha):
    root, p = load(run_id, protocol_sha)
    commit(reload=True)
    sys.path.insert(0, '/workspace/scripts')
    from sft_pipeline_summarize import summarize
    result = summarize(root/'eval')
    commit()
    return result


# ---------------------------------------------------------------- coordinator

def call_state(call_id):
    """('running'|'done'|'failed'|'lost', value) without blocking."""
    fc = modal.FunctionCall.from_id(call_id)
    try:
        return 'done', fc.get(timeout=0)
    except TimeoutError:  # builtin: the call has no output yet
        return 'running', None
    except modal.exception.OutputExpiredError as exc:
        return 'lost', repr(exc)
    except modal.exception.InputCancellation as exc:
        return 'failed', repr(exc)
    except Exception as exc:  # noqa: BLE001 - the remote error itself
        return 'failed', repr(exc)[:3000]


class StageFailed(RuntimeError):
    pass


class Coordinator:
    POLL = 30

    def __init__(self, run_id, protocol_sha):
        self.run_id, self.psha = run_id, protocol_sha
        self.root, self.p = load(run_id, protocol_sha)
        commit(reload=True)
        self.me = modal.current_function_call_id()
        self.calls = read_json(self.root/'calls.json') or dict(calls={}, fanouts={}, archived=[])
        self.status = read_json(self.root/'status.json') or dict(
            run_id=run_id, config=self.p['config']['name'], status='starting', created_unix=time.time(),
            stages={s: dict(state='pending' if s in self.p['stages'] else 'skipped') for s in
                    ('generate', 'prepare', 'train', 'evaluate', 'summarize')}, coordinators=[])
        self._last_publish = 0
        owner = self.status.get('coordinator_call_id')
        if owner and owner != self.me and call_state(owner)[0] == 'running':
            raise RuntimeError(f'coordinator {owner} is still running this run; not starting a second one')
        restart = owner == self.me
        self.status['coordinators'].append(dict(call_id=self.me, started_unix=time.time(), platform_restart=restart,
                                                launcher_sha256=sha(Path(__file__))))
        self.status.update(coordinator_call_id=self.me, status='running', error=None)
        self.publish(force=True)

    # -- persistence
    def save_calls(self):
        atomic_json(self.root/'calls.json', self.calls); commit()

    def publish(self, force=False, **kw):
        self.status.update(kw, observed_unix=time.time())
        if force or time.time()-self._last_publish > 60:
            atomic_json(self.root/'status.json', self.status); commit(); self._last_publish = time.time()

    def stage(self, name):
        return self.status['stages'][name]

    def begin(self, name):
        s = self.stage(name)
        s.setdefault('first_started_unix', time.time())
        s.update(state='running', started_unix=time.time())
        self.publish(force=True, current_stage=name)

    def finish(self, name, **kw):
        s = self.stage(name)
        if kw.get('resumed') and s.get('state') == 'done':  # re-verified after a restart: keep the original timing
            s['resumed'] = True
            return self.publish(force=True)
        s.update(kw, state=kw.get('state', 'done'), finished_unix=time.time())
        s['wall_seconds'] = s['finished_unix'] - s.get('first_started_unix', s['finished_unix'])
        self.publish(force=True)

    # -- calls
    def spawn(self, key, fn, *args, save=True):
        call = fn.spawn(*args)
        self.calls['calls'][key] = dict(call_id=call.object_id, spawned_unix=time.time(), by=self.me)
        if save:
            self.save_calls()
        return call.object_id

    def wait(self, ids, stage, progress=None):
        """Block until every call in ``ids`` ({key: call_id}) is terminal; publishes progress meanwhile."""
        pending, out = dict(ids), {}
        with concurrent.futures.ThreadPoolExecutor(16) as pool:
            while pending:
                states = dict(zip(pending, pool.map(call_state, pending.values())))
                for k, (st, v) in states.items():
                    if st != 'running':
                        out[k] = (st, v); pending.pop(k)
                s = self.stage(stage)
                s['calls'] = dict(done=len(out), running=len(pending), failed=sum(st != 'done' for st, _ in out.values()))
                if progress:
                    try:
                        commit(reload=True)
                        s['progress'] = progress()
                    except Exception as exc:  # noqa: BLE001 - progress is best effort
                        s['progress'] = dict(error=repr(exc)[:200])
                self.publish()
                if pending:
                    time.sleep(self.POLL)
        return out

    def fanout(self, name, stage, plan, spawn, max_rounds, progress=None):
        """Idempotent fan-out in persisted rounds. ``plan()`` -> {item: args} still to do (recomputed only
        when a new round starts); a restarted coordinator resumes the last round's recorded calls."""
        rounds = self.calls['fanouts'].setdefault(name, [])
        for r in range(max_rounds):
            if r == len(rounds):
                commit(reload=True)
                items = plan()
                if not items:
                    break
                rounds.append(dict(items={k: dict(args=a, call_id=None) for k, a in items.items()}, started_unix=time.time()))
                self.save_calls()
            rnd = rounds[r]
            todo = [it for it in rnd['items'].values() if it['call_id'] is None]
            for n, it in enumerate(todo):
                it['call_id'] = spawn(it['args']).object_id
                if n % 25 == 24:
                    self.save_calls()
            self.save_calls()
            outcome = self.wait({k: it['call_id'] for k, it in rnd['items'].items()}, stage, progress)
            rnd['failed_calls'] = {k: v for k, (st, v) in outcome.items() if st != 'done'}
            rnd['finished_unix'] = time.time()
            self.save_calls()
        commit(reload=True)
        return plan()

    # -- stages
    def run(self):
        try:
            if 'evaluate' in self.p['stages'] and self.p['evaluation']['start_independent_arms_early']:
                self.spawn_arms(trained=False)
            for name in ('generate', 'prepare', 'train', 'evaluate', 'summarize'):
                if name in self.p['stages']:
                    getattr(self, 'stage_'+name)()
            self.publish(force=True, status='complete', current_stage=None, finished_unix=time.time())
        except Exception as exc:
            current = self.status.get('current_stage')
            if current and self.stage(current)['state'] == 'running':
                self.stage(current)['state'] = 'failed'
            self.publish(force=True, status='failed', error=traceback.format_exc()[-4000:], finished_unix=time.time())
            if isinstance(exc, StageFailed):
                return self.status
            raise
        return self.status

    def stage_generate(self):
        g = self.p['generation']
        manifest_path = RUNS/(g['reuse_run'] or self.run_id)/'generate'/'manifest.json'
        manifest = read_json(manifest_path)
        if manifest and manifest['fingerprint'] == self.p['fingerprints']['generate']:
            return self.finish('generate', resumed=True, reused_from=g['reuse_run'], summary=manifest['summary'],
                               throughput=manifest['throughput'])
        if g['reuse_run']:
            raise StageFailed(f'reuse_run {g["reuse_run"]} has no generate manifest with this fingerprint')
        self.begin('generate')
        results = self.root/'generate'/'results'
        ids = [j['id'] for j in g['jobs']]

        def missing():
            done = {f.stem for f in results.glob('*.json')} if results.exists() else set()
            return [j for j in ids if j not in done]

        def plan():
            todo = missing()
            chunks = [todo[i:i+g['games_per_call']] for i in range(0, len(todo), g['games_per_call'])]
            return {f'c{i:05d}': c for i, c in enumerate(chunks)}

        def progress():
            n = sum(1 for _ in results.glob('*.json')) if results.exists() else 0
            return dict(games_done=n, games_total=len(ids))
        fn = lambda chunk: generate_chunk.spawn(self.run_id, self.psha, chunk)  # noqa: E731
        left = self.fanout('generate', 'generate', plan, fn, g['max_rounds'], progress)
        rows = [read_json(f) for f in sorted(results.glob('*.json'))]
        played = [r for r in rows if r['status'] != 'excluded']
        failures = [read_json(self.root/'generate'/'failures'/f'{j}.json', dict(id=j)) for j in left]
        s = self.stage('generate')
        wall = time.time() - s['first_started_unix']
        summary = dict(games=len(played), excluded=len(rows)-len(played), failed=len(failures),
                       expert_decisions=sum(r['expert_decisions'] for r in played),
                       forced_decisions=sum(r.get('forced_decisions', 0) for r in played),
                       by_opponent={o: dict(games=sum(r['opponent'] == o for r in played),
                                            wins=sum(r['opponent'] == o and r['status'] == 'native' and r['reward'] > 0 for r in played),
                                            losses=sum(r['opponent'] == o and r['status'] == 'native' and r['reward'] < 0 for r in played),
                                            truncated=sum(r['opponent'] == o and r['status'] == 'truncated' for r in played))
                                     for o in sorted({r['opponent'] for r in played})},
                       median_round=sorted(r['round'] for r in played)[len(played)//2] if played else None,
                       failure_samples=[dict(id=f['id'], error=(f.get('error') or '')[-400:]) for f in failures[:5]])
        container = sum(r['seconds'] for r in played)
        throughput = dict(wall_seconds=wall, games_per_minute=60*len(played)/wall,
                          expert_decisions_per_minute=60*summary['expert_decisions']/wall,
                          game_container_seconds_mean=container/max(1, len(played)),
                          container_hours=container/3600, usd_estimate=container*(2*0.0000131+8*0.00000222))
        if len(failures) > g['max_failed_games']:
            self.finish('generate', state='failed', summary=summary, throughput=throughput)
            raise StageFailed(f'{len(failures)} games failed (> {g["max_failed_games"]}); see generate/failures/, '
                              'then relaunch with --action resume --retry generate')
        manifest = dict(fingerprint=self.p['fingerprints']['generate'], run_id=self.run_id, finished_unix=time.time(),
                        summary=summary, throughput=throughput, failures=[f['id'] for f in failures],
                        games=[{k: r.get(k) for k in ('id', 'expert', 'opponent', 'seed', 'seat', 'status', 'reward', 'round',
                                                      'expert_decisions', 'forced_decisions', 'seconds')} for r in played])
        atomic_json(manifest_path, manifest); commit()
        self.finish('generate', summary=summary, throughput=throughput)

    def shard_plan(self):
        games = read_json(RUNS/(self.p['generation']['reuse_run'] or self.run_id)/'generate'/'manifest.json')['games']
        ids = sorted(r['id'] for r in games if r['status'] in ('native', 'truncated'))
        n = max(1, math.ceil(len(ids)/self.p['data']['games_per_shard']))
        return [ids[i::n] for i in range(n)]

    def stage_prepare(self):
        corpus = self.root/'prepare'/'corpus'/'manifest.json'
        manifest = read_json(corpus)
        if manifest and manifest['fingerprint'] == self.p['fingerprints']['prepare']:
            return self.finish('prepare', resumed=True, summary=self._prepare_summary(manifest),
                               throughput=self.stage('prepare').get('throughput'))
        self.begin('prepare')
        shards = self.shard_plan()
        shard_dir = self.root/'prepare'/'shards'
        fp = self.p['fingerprints']['prepare']

        def ok(i):
            m = read_json(shard_dir/f'shard-{i:04d}.json')
            return m is not None and m['fingerprint'] == fp

        def plan():
            return {f's{i:04d}': i for i in range(len(shards)) if not ok(i)}

        def progress():
            return dict(shards_done=sum(ok(i) for i in range(len(shards))), shards_total=len(shards))
        fn = lambda i: prepare_shard.spawn(self.run_id, self.psha, i, shards[i])  # noqa: E731
        left = self.fanout('prepare', 'prepare', plan, fn, 2, progress)
        if left:
            raise StageFailed(f'{len(left)} prepare shards failed; see calls.json fanouts.prepare, then --retry prepare')
        tokenize_wall = time.time() - self.stage('prepare')['first_started_unix']
        key = 'prepare/combine'
        if key not in self.calls['calls']:
            self.spawn(key, combine_corpus, self.run_id, self.psha, len(shards))
        st, value = self.wait({key: self.calls['calls'][key]['call_id']}, 'prepare')[key]
        commit(reload=True)
        manifest = read_json(corpus)
        if st != 'done' or not manifest:
            raise StageFailed(f'combine {st}: {value}; then --retry prepare')
        markers = [read_json(shard_dir/f'shard-{i:04d}.json') for i in range(len(shards))]
        wall = time.time() - self.stage('prepare')['first_started_unix']
        tokens = sum(m['tokens'] for m in markers)
        throughput = dict(wall_seconds=wall, tokenize_wall_seconds=tokenize_wall, combine_seconds=manifest['combine_seconds'],
                          examples_per_minute=60*manifest['records']/wall, tokens_per_second_tokenize=tokens/tokenize_wall,
                          shard_container_seconds=sum(m['seconds'] for m in markers), shards=len(shards))
        self.finish('prepare', summary=self._prepare_summary(manifest), throughput=throughput)

    @staticmethod
    def _prepare_summary(m):
        return {k: m[k] for k in ('records', 'unique_states', 'duplicates_dropped', 'conflicting_duplicate_labels',
                                  'capped_by_max_examples', 'max_tokens', 'mean_tokens', 'total_tokens', 'families',
                                  'move_fraction', 'source_stats')}

    def train_progress(self):
        commit(reload=True)
        run = read_json(self.root/'training'/'model'/'run.json')
        if not run:
            try:  # PENDING = no container yet (e.g. waiting for GPU capacity)
                cid = self.calls['calls']['train']['call_id']
                nodes, pending = list(modal.FunctionCall.from_id(cid).get_call_graph()), None
                while nodes:
                    n = nodes.pop()
                    if n.function_call_id == cid:
                        pending = n.status == modal.call_graph.InputStatus.PENDING
                    nodes += n.children
            except Exception:  # noqa: BLE001 - best effort
                pending = None
            return dict(phase='waiting_for_gpu' if pending else 'loading',
                        waited_minutes=(time.time()-self.calls['calls']['train']['spawned_unix'])/60)
        logs = run['logs']
        out = dict(status=run['status'], steps=len(logs), planned_steps=run['planned_steps'])
        if logs:
            sps = logs[-1]['elapsed_training_seconds']/len(logs)
            out.update(loss=logs[-1]['loss'], seconds_per_step=sps, eta_hours=(run['planned_steps']-len(logs))*sps/3600)
        return out

    def stage_train(self):
        t, out = self.p['training'], self.root/'training'
        run = read_json(out/'model'/'run.json')
        if run and run['status'] == 'completed' and (out/'execution.json').exists():
            return self.finish('train', resumed=True, summary=self.stage('train').get('summary'),
                               throughput=self.stage('train').get('throughput') or self._train_throughput(run))
        self.begin('train')
        key = 'train'
        if key not in self.calls['calls']:
            if out.exists():
                raise StageFailed('training/ exists without a recorded call; inspect it, then --retry train')
            fn = train.with_options(gpu=f'{t["gpu_type"]}:{t["gpus"]}', timeout=int(t['timeout_hours']*3600))
            self.spawn(key, fn, self.run_id, self.psha)
        st, value = self.wait({key: self.calls['calls'][key]['call_id']}, 'train', self.train_progress)[key]
        commit(reload=True)
        run = read_json(out/'model'/'run.json')
        if st != 'done' or not run or run['status'] != 'completed':
            self.finish('train', state='failed', error=str(value)[:2000])
            raise StageFailed(f'train {st}: {str(value)[:500]}; inspect training/, then --retry train')
        summary = dict(steps=run['global_step'], final_loss=run['logs'][-1]['loss'], first_loss=run['logs'][0]['loss'],
                       corpus_stress=run.get('corpus_stress'), adapter_files=run['adapter_files'],
                       max_gpu_memory_bytes=run.get('max_gpu_memory_bytes'))
        if run.get('resumes'):
            summary['resumed_at_steps'] = [r['step'] for r in run['resumes']]
        self.finish('train', summary=summary, throughput=self._train_throughput(run))

    def _train_throughput(self, run):
        t = self.p['training']
        corpus = read_json(self.root/'prepare'/'corpus'/'manifest.json')
        tokens = corpus['total_tokens']*t['config']['epochs']
        execution = read_json(self.root/'training'/'execution.json', {})
        occupied = execution.get('occupied_seconds') or run['elapsed_seconds']
        return dict(train_seconds=run['train_seconds'], seconds_per_step=run['train_seconds']/run['global_step'],
                    tokens_per_second=tokens/run['train_seconds'], examples_per_second=corpus['records']*t['config']['epochs']/run['train_seconds'],
                    occupied_seconds=occupied, gpu_hours=occupied*t['gpus']/3600,
                    usd_estimate=occupied*t['gpus']*{'B200': 0.001736, 'H200': 0.001261, 'H100': 0.001097}.get(t['gpu_type'], 0.001736))

    def spawn_arms(self, trained):
        ev = self.root/'eval'
        for arm, spec in self.p['evaluation']['arms'].items():
            if spec['trained'] != trained:
                continue
            key = f'evaluate/{arm}'
            if key in self.calls['calls'] or (ev/arm/'report.json').exists():
                continue
            if (ev/arm).exists():
                raise StageFailed(f'eval/{arm} exists without a recorded call; inspect it, then --retry evaluate')
            fn = evaluate.with_options(gpu=self.p['evaluation']['gpu'])
            self.spawn(key, fn, self.run_id, self.psha, arm)

    def stage_evaluate(self):
        e, ev = self.p['evaluation'], self.root/'eval'
        arms = e['arms']
        if all((ev/a/'report.json').exists() for a in arms) and self._references_done():
            return self.finish('evaluate', resumed=True, arms=self.stage('evaluate').get('arms'))
        self.begin('evaluate')
        self.spawn_arms(trained=False)
        self.spawn_arms(trained=True)
        jobs = read_json(ev/'protocol.json')['jobs']
        if e['rl_reference']:
            def plan():
                out = {}
                for pl in e['rl_reference']:
                    todo = [j['id'] for j in jobs if not (ev/f'reference-{pl}'/f'{j["id"]}.json').exists()]
                    for i in range(0, len(todo), 8):
                        out[f'{pl}/{i//8}'] = [pl, todo[i:i+8]]
                return out
            fn = lambda a: reference_chunk.spawn(self.run_id, self.psha, a[0], a[1])  # noqa: E731
            left = self.fanout('reference', 'evaluate', plan, fn, 2)
            self.stage('evaluate')['reference_missing_chunks'] = sorted(left)

        def progress():
            out = {}
            for arm in arms:
                pr = read_json(ev/arm/'games'/'progress.json')
                out[arm] = 'done' if (ev/arm/'report.json').exists() else pr and dict(finished=pr['finished'], allocated=pr['allocated'])
            return out
        keys = {f'evaluate/{a}': self.calls['calls'][f'evaluate/{a}']['call_id'] for a in arms
                if f'evaluate/{a}' in self.calls['calls']}
        outcome = self.wait(keys, 'evaluate', progress)
        commit(reload=True)
        failed = {k.split('/', 1)[1]: v for k, (st, v) in outcome.items() if st != 'done'}
        missing = [a for a in arms if not (ev/a/'report.json').exists()]
        self.finish('evaluate', state='failed' if missing else 'done',
                    arms={a: 'done' if a not in missing else str(failed.get(a, 'missing'))[:500] for a in arms})
        if missing:
            raise StageFailed(f'evaluation arms without a report: {missing}; inspect eval/, then --retry evaluate')

    def _references_done(self):
        jobs = read_json(self.root/'eval'/'protocol.json')['jobs']
        return all((self.root/'eval'/f'reference-{pl}'/f'{j["id"]}.json').exists()
                   for pl in self.p['evaluation']['rl_reference'] for j in jobs)

    def stage_summarize(self):
        self.begin('summarize')
        key = f'summarize/{len(self.status["coordinators"])}'
        self.spawn(key, summarize_eval, self.run_id, self.psha)
        st, small = self.wait({key: self.calls['calls'][key]['call_id']}, 'summarize')[key]
        if st != 'done':
            self.finish('summarize', state='failed', error=str(small)[:2000])
            raise StageFailed(f'summarize {st}: {str(small)[:500]}')
        compact = dict(models={m: {k: v.get(k) for k in ('games', 'wins', 'losses', 'truncated', 'forfeits', 'operational',
                                                          'median_round', 'mean_peak_cities')} for m, v in
                               ((m, d['all']) for m, d in small['models'].items())},
                       paired={k: v['all'] for k, v in small['paired'].items()},
                       eval_throughput={a: {k: v.get(k) for k in ('prompt_tokens_per_second', 'decisions_per_minute',
                                                                  'worker_seconds', 'load_seconds', 'decisions')}
                                        for a, v in small['throughput'].items()})
        self.stage('evaluate')['throughput'] = compact['eval_throughput']
        self.finish('summarize', summary=compact)


@app.function(image=image, cpu=1, memory=4096, timeout=24*3600, max_containers=4, retries=0, volumes=VOLUMES)
def coordinate(run_id, protocol_sha):
    return Coordinator(run_id, protocol_sha).run()


@app.function(image=image, cpu=1, memory=2048, timeout=600, retries=0, volumes=VOLUMES)
def admin(run_id, protocol_sha, action, stage=''):
    """status: current status.json. retry: archive the stage's dead calls and partial outputs so a resumed
    coordinator redoes them; refuses while any of those calls is still running. For train it also reports the
    checkpoint the next train call resumes from (see resume_point)."""
    root, p = load(run_id, protocol_sha)
    commit(reload=True)
    status, calls = read_json(root/'status.json'), read_json(root/'calls.json')
    if action == 'status':
        return dict(status=status, coordinator=status and call_state(status['coordinator_call_id'])[0])
    if action != 'retry' or not calls:
        raise ValueError('action must be status or retry (after a launch)')
    owner = status.get('coordinator_call_id')
    if owner and call_state(owner)[0] == 'running':
        raise RuntimeError('the coordinator is still running; retry only after it stopped')
    stamp = time.strftime('%Y%m%d-%H%M%S')
    moved, archived = [], []
    prefixes = {'generate': ('generate',), 'prepare': ('prepare',), 'train': ('train',),
                'evaluate': ('evaluate', 'reference'), 'summarize': ('summarize',)}[stage]
    for key in [k for k in calls['calls'] if k.split('/')[0] in prefixes]:
        st, _ = call_state(calls['calls'][key]['call_id'])
        if st == 'running':
            raise RuntimeError(f'{key} is still running')
        if st != 'done' or stage in ('train', 'prepare'):
            archived.append(dict(key=key, state=st, **calls['calls'].pop(key)))
            if key.startswith('evaluate/'):
                arm = key.split('/', 1)[1]
                if (root/'eval'/arm).exists() and not (root/'eval'/arm/'report.json').exists():
                    (root/'eval'/arm).rename(root/'eval'/f'{arm}.failed-{stamp}'); moved.append(f'eval/{arm}')
    for name in [n for n in calls['fanouts'] if n in prefixes]:
        for rnd in calls['fanouts'][name]:
            for it in rnd['items'].values():
                if it['call_id'] and call_state(it['call_id'])[0] == 'running':
                    raise RuntimeError(f'a {name} call is still running')
        archived.append(dict(key=f'fanout/{name}', rounds=calls['fanouts'].pop(name)))
    if stage == 'train' and (root/'training').exists() and read_json(root/'training'/'model'/'run.json', {}).get('status') != 'completed':
        (root/'training').rename(root/f'training.failed-{stamp}'); moved.append('training')
    resume = resume_point(root, train_payload(root, p)) if stage == 'train' else None
    calls['archived'].append(dict(stage=stage, unix=time.time(), calls=archived, moved=moved, resume=resume))
    atomic_json(root/'calls.json', calls)
    if status:
        status['stages'][stage]['state'] = 'pending'
        atomic_json(root/'status.json', status)
    commit()
    return dict(archived=[a['key'] for a in archived], moved=moved, resume=resume)


# ---------------------------------------------------------------- launch-side protocol

def snapshot_sources():
    """tar.gz of src/, scripts/, configs/ (no caches) and a hash for every file in it."""
    buf, hashes = io.BytesIO(), {}
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for d in SNAPSHOT_DIRS:
            for f in sorted((ROOT/d).rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts and not f.name.startswith('.'):
                    rel = str(f.relative_to(ROOT))
                    hashes[rel] = sha(f)
                    tar.add(f, arcname=rel, recursive=False)
    return buf.getvalue(), hashes


def build_protocol(cfg, run_id, config_path):
    sys.path.insert(0, str(ROOT/'src'))
    from polytopia_bot.learning import sft_pipeline as sp
    from polytopia_bot.learning.interface import digest
    g, d, t, e = cfg['generation'], cfg['data'], cfg['training'], cfg['evaluation']
    renderer = sp.renderer_spec(cfg['prompt']['renderer'])
    manifest = json.loads((ROOT/g['map_manifest']).read_text())
    # Training recipe: the qualified runtime benchmark's config plus the config's overrides.
    base = json.loads((ROOT/t['base']).read_text())['training']
    config = dict(base['config'], **t['overrides'])
    if 'max_length' in t['overrides'] and t['overrides']['max_length'] > 24576:
        print('WARNING: max_length above 24,576 has not passed the 4 x B200 stress test')
    training = dict(config=config, benchmark=base['benchmark'], benchmark_source=t['base'], seed=t['seed'],
                    checkpoint_every=t['checkpoint_every'], gpus=t['gpus'], gpu_type=t['gpu_type'], timeout_hours=t['timeout_hours'])
    names = set(g['experts']) | {o['name'] for o in g['opponents']} | set(e['opponents']) | set(e['rl_reference'])
    players = {n: sp.player_spec(cfg, n) for n in sorted(names)}
    if g['reuse_run']:
        src = json.loads((ROOT/'runs'/g['reuse_run']/'protocol.json').read_text())
        if src['prompt_renderer'] != renderer:
            raise ValueError('reuse_run was rendered with a different prompt renderer')
        generation = dict(src['generation'], reuse_run=g['reuse_run'])
        players.update(src['players'])
        gen_fp = src['fingerprints']['generate']
        pool_seeds = {j['seed'] for j in generation['jobs']}
        pool_sym = {j['symmetry_hash'] for j in generation['jobs'] if j['symmetry_hash']}
    else:
        pool = sp.map_pool(cfg, manifest)
        jobs = sp.generation_jobs(cfg, pool)
        forbidden = sorted({m['symmetry_hash'] for split, maps in manifest['maps'].items() if split != 'train' for m in maps})
        if {m['symmetry_hash'] for m in pool if m['symmetry_hash']} & set(forbidden):
            raise ValueError('generation map pool contains a held-out split map')
        mf = g['max_failed_games']
        generation = dict(jobs=jobs, forbidden_symmetry_hashes=forbidden, map_pool_size=len(pool),
                          map_pool=g['map_pool'], map_manifest=g['map_manifest'], map_manifest_hash=manifest['manifest_hash'],
                          experts=g['experts'], opponents=g['opponents'], seats=g['seats'],
                          games_per_call=g['games_per_call'], max_rounds=g['max_rounds'],
                          max_failed_games=math.ceil(mf*len(jobs)) if mf < 1 else int(mf),
                          record_trajectory=g['record_trajectory'], reuse_run=None)
        gen_fp = digest(dict(jobs=jobs, players={n: players[n] for n in names if n in g['experts'] or n in
                                                  {o['name'] for o in g['opponents']}},
                             renderer=renderer, forbidden=forbidden, trajectory=g['record_trajectory']))
        pool_seeds = {m['seed'] for m in pool}
        pool_sym = {m['symmetry_hash'] for m in pool if m['symmetry_hash']}
    # Evaluation protocol in the benchmark_expert_sft schema (its own file, hash-linked from here).
    eval_jobs = sp.evaluation_jobs(cfg, manifest)
    if e['maps']['split'] == 'train' or {j['seed'] for j in eval_jobs} & pool_seeds or \
            {j['symmetry_hash'] for j in eval_jobs} & pool_sym:
        raise ValueError('evaluation maps overlap generation maps')
    harness = json.loads((ROOT/e['harness_source']).read_text())
    if (harness['model_id'], harness['revision']) != (config['model_id'], config['revision']):
        raise ValueError('evaluation harness and training config use different base models')
    rank = next(r for r in (8, 16, 32, 64, 128, 256) if r >= max(32, config['lora_rank']))
    arms = sp.eval_arms(cfg, run_id)
    for a, spec in arms.items():
        spec['trained'] = cfg['evaluation']['arms'][spec['model']].get('adapter') == 'trained'
    snapshot, hashes = snapshot_sources()
    rel_config = str(Path(config_path).resolve().relative_to(ROOT))
    if rel_config not in hashes:
        raise ValueError('config must live under configs/ so it is part of the source snapshot')
    keep = ['model_id', 'revision', 'named_fields', 'constrain_actions', 'max_length', 'max_new_tokens', 'thinking', 'seed',
            'sampling', 'concurrency']
    eval_protocol = dict({k: harness[k] for k in keep}, schema='sft-pipeline-eval-v1', run_id=run_id, gpu=e['gpu'],
        serving=dict(harness['serving'], **e['serving']),
        harness=renderer['harness'], prompt_renderer=renderer, lora=dict(max_rank=rank),
        arms={a: {k: v for k, v in s.items() if k != 'trained'} for a, s in arms.items()},
        rl_opponents={n: dict(path=players[n]['path'], sha256=players[n]['sha256'], deterministic=True)
                      for n in e['opponents'] if players[n]['kind'] == 'rl'},
        rl_reference=e['rl_reference'], pairs=e['pairs'], jobs=eval_jobs,
        split=f'{e["maps"]["split"]} offset {e["maps"].get("offset", 0)} count {e["maps"]["count"]}',
        split_manifest_hash=manifest['manifest_hash'],
        source_hashes={k: v for k, v in hashes.items() if k.endswith('.py') or k.startswith('configs/')},
        harness_source=dict(path=e['harness_source'], sha256=sha(ROOT/e['harness_source'])))
    prep_fp = digest(dict(generate=gen_fp, data=d, renderer=renderer,
                          tokenizer={k: config[k] for k in ('model_id', 'revision', 'max_length', 'max_new_tokens')}))
    fingerprints = dict(generate=gen_fp, prepare=prep_fp,
                        train=digest(dict(prepare=prep_fp, training={k: v for k, v in training.items() if k != 'timeout_hours'})))
    protocol = dict(schema='sft-pipeline-protocol-v1', run_id=run_id, created_unix=time.time(), config=cfg,
        config_path=rel_config, config_sha256=hashes[rel_config], stages=cfg['stages'], players=players,
        prompt_renderer=renderer, generation=generation, data=d, training=training,
        evaluation=dict(gpu=e['gpu'], arms=arms, rl_reference=e['rl_reference'],
                        start_independent_arms_early=e['start_independent_arms_early']),
        fingerprints=fingerprints, source_hashes=hashes,
        source_snapshot=dict(path='source/source.tar.gz', sha256=hashlib.sha256(snapshot).hexdigest(), files=len(hashes)),
        launcher_sha256=sha(Path(__file__)), estimate=sp.estimate(cfg, len(generation['jobs'])))
    return protocol, eval_protocol, snapshot


def estimate_local(config_path):
    sys.path.insert(0, str(ROOT/'src'))
    from polytopia_bot.learning import sft_pipeline as sp
    cfg = sp.load_config(config_path)
    manifest = json.loads((ROOT/cfg['generation']['map_manifest']).read_text())
    jobs = sp.generation_jobs(cfg, sp.map_pool(cfg, manifest))
    return sp.estimate(cfg, len(jobs))


@app.local_entrypoint()
def main(action: str = 'launch', config: str = '', run_id: str = '', retry: str = ''):
    local = ROOT/'runs'/run_id
    if not run_id:
        raise ValueError('--run-id required')
    if action in ('status', 'resume'):
        protocol_sha = sha(local/'protocol.json')
        if retry:
            print('RETRY', json.dumps(admin.remote(run_id, protocol_sha, 'retry', retry)))
        if action == 'status':
            status, coordinator = remote_status(run_id)
            print(json.dumps(dict(coordinator=coordinator, status=status), indent=1, default=str)[:20000])
            return
        call = coordinate.spawn(run_id, protocol_sha)
        receipt = dict(app_id=app.app_id, call_id=call.object_id, protocol_sha256=protocol_sha,
                       launcher_sha256=sha(Path(__file__)), resumed_unix=time.time(), retry=retry or None)
        dispatch = read_json(local/'dispatch.json', dict(launches=[]))
        dispatch.setdefault('launches', []).append(receipt)
        atomic_json(local/'dispatch.json', dispatch); print(json.dumps(receipt))
        return
    if action not in ('launch', 'smoke'):
        raise ValueError('action must be launch, smoke, resume or status')
    sys.path.insert(0, str(ROOT/'src'))
    from polytopia_bot.learning import sft_pipeline as sp
    cfg = sp.load_config(config)
    if local.exists():
        raise ValueError(f'{local} exists; use --action resume (the protocol of a run never changes)')
    protocol, eval_protocol, snapshot = build_protocol(cfg, run_id, config)
    local.mkdir(parents=True)
    (local/'eval').mkdir()
    (local/'eval'/'protocol.json').write_text(json.dumps(eval_protocol, indent=2)+'\n')
    protocol['evaluation']['protocol_sha256'] = sha(local/'eval'/'protocol.json')
    (local/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    protocol_sha = sha(local/'protocol.json')
    for attempt in range(20):
        try:
            with volume.batch_upload(force=True) as up:
                up.put_file(local/'protocol.json', f'{run_id}/protocol.json')
                up.put_file(local/'eval'/'protocol.json', f'{run_id}/eval/protocol.json')
                up.put_file(io.BytesIO(snapshot), f'{run_id}/source/source.tar.gz')
            break
        except modal.exception.ResourceExhaustedError as exc:
            print(f'upload retry {attempt}: {exc!r}', flush=True); time.sleep(min(60, 10*(attempt+1)))
    else:
        raise RuntimeError('protocol upload kept failing')
    print('ESTIMATE', json.dumps(protocol['estimate']))
    call = coordinate.spawn(run_id, protocol_sha)
    receipt = dict(app_id=app.app_id, call_id=call.object_id, protocol_sha256=protocol_sha, launcher_sha256=sha(Path(__file__)),
                   launched_unix=time.time())
    atomic_json(local/'dispatch.json', dict(launches=[receipt])); print(json.dumps(receipt))
    if action == 'smoke':
        # Follow the run from the volume (no function calls: in --detach mode only the last triggered
        # function, the coordinator, survives a disconnect of this process).
        follow(run_id, local)


def remote_status(run_id):
    try:
        status = json.loads(b''.join(volume.read_file(f'{run_id}/status.json')))
    except Exception:  # noqa: BLE001 - not written yet
        return None, None
    return status, call_state(status['coordinator_call_id'])[0]


def follow(run_id, local):
    while True:
        s, coordinator = remote_status(run_id)
        s = s or {}
        print(time.strftime('%H:%M:%S'), s.get('status'), s.get('current_stage'), coordinator,
              json.dumps({k: (v.get('state'), v.get('progress') or v.get('calls')) for k, v in s.get('stages', {}).items()},
                         default=str)[:700], flush=True)
        if s.get('status') in ('complete', 'failed') and coordinator != 'running':
            atomic_json(local/'status.json', s)
            print('FINAL', json.dumps(s, indent=1, default=str)[:20000], flush=True)
            return s
        time.sleep(30)

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'estimate':
        print(json.dumps(estimate_local(sys.argv[2]), indent=1))
    else:
        raise SystemExit(__doc__)
