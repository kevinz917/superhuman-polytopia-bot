"""Private, staging-only offline Rust bot benchmark. See skirmish/README.md."""
from pathlib import Path
import json
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')

base = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('curl', 'build-essential', 'pkg-config')
    .run_commands('curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain 1.90.0',
                  '/root/.cargo/bin/rustup target add wasm32-wasip1')
    .env({'PATH': '/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin'}))
coding_image = (base
    .add_local_dir(ROOT / 'skirmish/bot-template', '/bot', copy=True, ignore=['**/target/**'])
    .run_commands('cd /bot && cargo fetch --locked && cargo build --release --offline --locked --target wasm32-wasip1')
    .add_local_dir(ROOT / 'engine-rs/src', '/reference/engine-rs/src', copy=True)
    .add_local_file(ROOT / 'src/polytopia_bot/gpu/rules.json', '/reference/rules.json', copy=True)
    .add_local_file(ROOT / 'skirmish/PROMPT.md', '/reference/README.md', copy=True))
mini_coding_image = (coding_image
    .pip_install('jax==0.11.2', 'numpy==2.4.4', 'maturin==1.9.6', 'wasmtime==38.0.0')
    .add_local_dir(ROOT / 'engine-rs', '/opt/skirmish-build/engine-rs', copy=True, ignore=['**/target/**'])
    .run_commands('cd /opt/skirmish-build/engine-rs && maturin build --release --locked -o /tmp/engine-wheels && pip install /tmp/engine-wheels/*.whl',
                  'rm -rf /opt/skirmish-build /tmp/engine-wheels')
    .apt_install('procps', 'util-linux', 'passwd')
    .run_commands('/usr/sbin/useradd --create-home --shell /bin/bash coder',
                  'cp -a /root/.cargo /opt/cargo && cp -a /root/.rustup /opt/rustup',
                  'cp -a /root/.cargo /home/coder/.cargo && chown -R coder:coder /home/coder/.cargo',
                  'cp -a /bot /verify && chown -R coder:coder /bot',
                  'chmod -R a+rX /opt/cargo /opt/rustup')
    .env({'CARGO_HOME':'/opt/cargo', 'RUSTUP_HOME':'/opt/rustup',
          'PATH':'/opt/cargo/bin:/usr/local/bin:/usr/bin:/bin',
          'PYTHONPATH':'/opt/skirmish-sdk', 'JAX_PLATFORMS':'cpu', 'OMP_NUM_THREADS':'1',
          'XLA_FLAGS':'--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'})
    .add_local_dir(ROOT / 'skirmish/mini', '/reference', copy=True))
# Ship only engine/protocol/map dependencies, never policy/search/opponent code or weights.
sdk_files = ['__init__.py', 'environment.py', 'ruleset.py',
    *[str(p.relative_to(ROOT/'src/polytopia_bot')) for p in sorted((ROOT/'src/polytopia_bot/gpu').glob('*.py'))],
    'gpu/rules.json', 'learning/__init__.py', 'learning/adapter.py', 'learning/interface.py', 'learning/memory.py',
    'rl/__init__.py', 'rl/actions.py', 'rl/cpu_engine.py', 'rl/observation.py', 'rl/rust_engine.py',
    'studio/__init__.py', 'studio/environment.py', 'maps/__init__.py', 'maps/variants.py', 'maps/state.py',
    'skirmish/__init__.py', 'skirmish/common.py', 'skirmish/authoring_tools.py', 'skirmish/wasi.py', 'skirmish/wasi_worker.py']
def exclude_non_sdk(path):
    relative=path.as_posix()
    return relative not in sdk_files and not any(name.startswith(relative+'/') for name in sdk_files)

mini_coding_image = mini_coding_image.add_local_dir(
    ROOT/'src/polytopia_bot','/opt/skirmish-sdk/polytopia_bot',copy=True,ignore=exclude_non_sdk)
mini_coding_image = (mini_coding_image
    .run_commands('cp /reference/skirmish.sh /usr/local/bin/skirmish && chmod 755 /usr/local/bin/skirmish',
        'cp /bot/src/main.rs /reference/STARTER.rs',
        'python -c "from polytopia_bot.skirmish.authoring_tools import build_examples; build_examples(\'/reference/examples\')"'))
runtime_image = (base
    .pip_install('jax==0.11.2', 'numpy==2.4.4', 'maturin==1.9.6', 'wasmtime==38.0.0', 'pytest==9.1.1',
                 'mini-swe-agent==2.4.6', 'modal==1.5.2')
    .add_local_dir(ROOT / 'engine-rs', '/workspace/engine-rs', copy=True, ignore=['**/target/**'])
    .run_commands('cd /workspace/engine-rs && maturin build --release --locked -o /tmp/wheels && pip install /tmp/wheels/*.whl')
    .env({'PYTHONPATH': '/workspace/src:/workspace', 'JAX_PLATFORMS': 'cpu',
          'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1', 'OMP_NUM_THREADS': '1'})
    .add_local_dir(ROOT / 'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT / 'skirmish', '/workspace/skirmish', ignore=['**/target/**'])
    .add_local_dir(ROOT / 'tests', '/workspace/tests', ignore=['**/__pycache__/**']))
app = modal.App('polytopia-skirmish', image=runtime_image)
results = modal.Volume.from_name('polytopia-skirmish', create_if_missing=True)
MOUNTS = {'/results': results}


@app.function(cpu=4, memory=8192, timeout=10800, retries=0, max_containers=2, volumes=MOUNTS)
def trial(run_id: str, config: dict):
    from polytopia_bot.skirmish.agent import run_trial
    sandbox_app = modal.App.lookup('polytopia-skirmish-sandboxes', create_if_missing=True, environment_name='staging')
    return run_trial(run_id, config, modal.Image.from_id(config['_coding_image_id']), sandbox_app, results)


@app.function(cpu=4, memory=8192, timeout=14400, retries=0, max_containers=20, volumes=MOUNTS)
def mini_trial(run_id: str, config: dict):
    from polytopia_bot.skirmish.mini import run_mini_trial
    sandbox_app = modal.App.lookup('polytopia-skirmish-sandboxes', create_if_missing=True, environment_name='staging')
    return run_mini_trial(run_id, config, modal.Image.from_id(config['_coding_image_id']), sandbox_app, results)


@app.function(cpu=4, memory=8192, timeout=1800, retries=0, max_containers=6, volumes=MOUNTS)
def mini_match(payload: dict):
    from polytopia_bot.skirmish.mini import mini_pair
    return mini_pair(payload,results)


@app.function(cpu=4,memory=8192,timeout=300,retries=0,max_containers=32,volumes=MOUNTS)
def mini_match_game(payload: dict):
    import time,uuid
    from dataclasses import replace
    from polytopia_bot.skirmish.common import Budget,REFERENCE,REFERENCE_SHA,write_json
    from polytopia_bot.skirmish.mini import evaluation_schedule,load_submission
    from polytopia_bot.skirmish.game import WasiBot,play_game,warm_reference
    a,b=payload['a'],payload['b'];index=payload['game_index']
    if index not in range(6):raise ValueError('invalid game index')
    entrants={label:load_submission(label) for label in (a,b) if label!=REFERENCE}
    budget=Budget(**next(iter(entrants.values()))[2]['runtime'])
    if 'fuel' in payload:budget=replace(budget,fuel=int(payload['fuel'])).validate()
    reference=warm_reference('/results/references/hero-1v1/final.npz',REFERENCE_SHA) if REFERENCE in (a,b) else None
    def bot(label):
        return reference if label==REFERENCE else WasiBot(entrants[label][0]/'bot.wasm',budget.fuel,budget.move_seconds)
    job=evaluation_schedule(payload['seeds'])[index]
    root=Path('/results/mini-tournaments')/payload['tournament_id']/(a+'__'+b)
    summary=root/f'{index:02d}.summary.json'
    if summary.exists():return json.loads(summary.read_text())
    # Modal can redeliver an input after a transport failure. One distributed claim
    # preserves one canonical game and trace; duplicates await its committed result.
    claims=modal.Dict.from_name('polytopia-skirmish-game-claims-v1',create_if_missing=True)
    key=(payload['tournament_id'],a,b,index)
    owner=uuid.uuid4().hex
    if not claims.put(key,dict(status='running',owner=owner),skip_if_exists=True):
        deadline=time.monotonic()+240
        while time.monotonic()<deadline:
            receipt=claims[key]
            if receipt['status']=='completed':return receipt['row']
            if receipt['status']=='failed':raise RuntimeError(receipt['error'])
            time.sleep(2)
        raise RuntimeError('canonical game has not completed; preserve its artifacts for inspection')
    try:
        row=play_game(bot(a),bot(b),job['seed'],job['seat'],budget,root/f'{index:02d}.jsonl',
                      phase='final',labels=[a,b],map_type=job['map_type'])
        write_json(summary,row);results.commit()
        claims[key]=dict(status='completed',owner=owner,row=row)
        return row
    except Exception as error:
        claims[key]=dict(status='failed',owner=owner,error=f'{type(error).__name__}: {error}')
        raise


@app.function(cpu=1, memory=2048, timeout=7200, retries=0, volumes=MOUNTS)
def mini_ladder(tournament_id: str, run_ids: list[str]):
    from polytopia_bot.skirmish.mini import mini_tournament
    from polytopia_bot.skirmish.common import write_json
    def parallel_games(groups):
        jobs=[dict(**group,game_index=i) for group in groups for i in range(6)]
        for done,row in enumerate(mini_match_game.map(jobs,order_outputs=False),1):
            write_json(Path('/results/mini-tournaments')/tournament_id/'status.json',
                       dict(status='running',completed=done,games=len(jobs)))
            results.commit()
            yield [row]
    return mini_tournament(tournament_id,run_ids,results,pair_runner=parallel_games)


@app.function(cpu=4,memory=8192,timeout=1800,retries=0,volumes=MOUNTS)
def mini_ladder_audit(tournament_id: str):
    from polytopia_bot.skirmish.common import identifier, write_json
    from polytopia_bot.skirmish.mini import MAP_TYPES,audit_trace
    root=Path('/results/mini-tournaments')/identifier(tournament_id)
    rows=json.loads((root/'games.json').read_text())
    actions=sum(mini_audit_game.map([dict(tournament_id=tournament_id,row=row) for row in rows]))
    report=dict(passed=True,games=len(rows),actions=actions)
    write_json(root/'audit.json',report);results.commit()
    return report


@app.function(cpu=1,memory=2048,timeout=3600,retries=0,volumes=MOUNTS)
def mini_fuel_calibration(tournament_id: str, original_id: str):
    from polytopia_bot.skirmish.common import identifier,write_json,sha
    from polytopia_bot.skirmish.tournament import summarize
    from polytopia_bot.skirmish.mini import load_submission
    original=Path('/results/mini-tournaments')/identifier(original_id)
    plan=json.loads((original/'plan.json').read_text())
    for label in plan['labels']:
        if label!='hero-1v1':load_submission(label)
    root=Path('/results/mini-tournaments')/identifier(tournament_id)
    root.mkdir(parents=True,exist_ok=False)
    groups=[dict(**{k:v for k,v in group.items() if k!='tournament_id'},tournament_id=tournament_id,
                 fuel=1_000_000_000_000) for group in plan['groups']]
    write_json(root/'plan.json',dict(labels=plan['labels'],groups=groups,original_id=original_id,
        original_plan_sha256=sha((original/'plan.json').read_bytes()),frozen_sources_unchanged=True,
        override=dict(fuel=1_000_000_000_000),other_limits_unchanged=True))
    results.commit()
    jobs=[dict(**group,game_index=i) for group in groups for i in range(6)]
    rows=[]
    for row in mini_match_game.map(jobs,order_outputs=False):
        rows.append(row)
        write_json(root/'status.json',dict(status='running',completed=len(rows),games=len(jobs)));results.commit()
    base=json.loads((original/'report.json').read_text())
    report=dict(schema='polytopia-mini-fuel-calibration-v1',tournament_id=tournament_id,status='completed',
        track=base['track'],smoke=False,original_id=original_id,fuel=1_000_000_000_000,
        frozen_sources_unchanged=True,other_limits_unchanged=True,**summarize(rows,plan['labels']))
    write_json(root/'games.json',rows);write_json(root/'report.json',report);results.commit()
    return report


@app.function(cpu=4, memory=8192, timeout=1800, retries=0, volumes=MOUNTS)
def mini_inspect(prefix: str, audit: bool = False):
    from polytopia_bot.skirmish.common import identifier, write_json
    from polytopia_bot.skirmish.mini import audit_trace
    identifier(prefix)
    reports = {}
    for root in sorted(Path('/results/mini-runs').glob(prefix+'*')):
        path = root/'report.json' if (root/'report.json').exists() else root/'status.json'
        report = json.loads(path.read_text())
        stages=[]
        for trajectory in sorted(root.glob('*.trajectory.json')):
            data=json.loads(trajectory.read_text());messages=data['messages']
            stats=data['info']['model_stats']
            stats['trajectory']=trajectory.name
            stats['bash_commands']=sum(len(m.get('extra',{}).get('actions',[])) for m in messages)
            stats['feedback_reads']=sum('/feedback/' in a.get('command','') for m in messages
                                        for a in m.get('extra',{}).get('actions',[]))
            stages.append(stats)
        report['trajectory_stats']=stages
        if (root/'stages.json').exists():report['stages']=json.loads((root/'stages.json').read_text())
        for stage in report.get('stages',[]):stage.pop('compile_output',None)
        report['worker_errors']={p.name:p.read_text()[-1800:] for p in root.glob('*.worker.log') if p.stat().st_size}
        if audit:
            actions=games=0
            for summary in [*root.glob('feedback-*/report.json'),*root.glob('final/report.json')]:
                for row in json.loads(summary.read_text())['rows']:
                    actions+=audit_trace(row,summary.parent/row['trace_file']);games+=1
            report['audit']=dict(passed=True,games=games,actions=actions)
            write_json(root/'audit.json',report['audit']);results.commit()
        # Detailed observations, source and complete transcripts remain on Modal.
        if isinstance(report.get('final'),dict):report['final'].pop('rows',None)
        reports[root.name]=report
    return reports


@app.function(cpu=1, memory=2048, timeout=86400, retries=0, max_containers=2, volumes=MOUNTS)
def run_ladder(tournament_id: str, run_ids: list[str], maps: int):
    from polytopia_bot.skirmish.tournament import tournament
    return tournament(tournament_id, run_ids, maps, results,
                      pair_runner=lambda groups: match_group.map(groups,order_outputs=False))


@app.function(cpu=1, memory=1024, timeout=30, volumes={'/results': results})
def get_status(run_id: str, tournament: bool = False):
    from polytopia_bot.skirmish.common import identifier
    root = Path('/results') / ('tournaments' if tournament else 'runs') / identifier(run_id)
    for name in ('report.json', 'status.json'):
        if (root / name).exists():
            return json.loads((root / name).read_text())
    return {'status': 'not_found', 'run_id': run_id}


@app.function(cpu=4, memory=8192, timeout=900, volumes=MOUNTS)
def checks():
    import subprocess
    p = subprocess.run(['python', '-m', 'pytest', '-q', '/workspace/tests/test_skirmish.py',
                        '/workspace/tests/test_skirmish_mini.py', '/workspace/tests/test_skirmish_tools.py',
                        '/workspace/tests/test_skirmish_v5.py',
                        '/workspace/tests/test_studio_environment.py'],
                       capture_output=True, text=True, cwd='/workspace', timeout=800)
    return {'returncode': p.returncode, 'output': p.stdout[-10000:] + p.stderr[-3000:]}


@app.local_entrypoint()
def launch(run_id: str, config: str):
    from modal.config import config as modal_config
    if modal_config.get('environment') != 'staging':
        raise RuntimeError('pass --env staging')
    cfg = json.loads(Path(config).read_text())
    use_mini = cfg.get('harness') == 'mini-swe-agent'
    cfg['_coding_image_id'] = (mini_coding_image if use_mini else coding_image).build(app).object_id
    key_env = cfg.get('key_env', 'OPENROUTER_API_KEY')
    if not os.environ.get(key_env):
        raise RuntimeError(f'{key_env} is required in the controller environment')
    # Secret stays in process memory and is injected only into the trusted controller.
    bound = (mini_trial if use_mini else trial).with_options(
        secrets=[modal.Secret.from_dict({key_env: os.environ[key_env]})])
    call = bound.spawn(run_id, cfg)
    print(json.dumps({'run_id': run_id, 'function_call_id': call.object_id}))
    print(json.dumps(call.get(), indent=2))


@app.local_entrypoint()
def status(run_id: str, tournament: bool = False):
    print(json.dumps(get_status.remote(run_id, tournament), indent=2))


@app.local_entrypoint()
def ladder(tournament_id: str, run_ids: str, maps: int = 16):
    print(json.dumps(run_ladder.remote(tournament_id, run_ids.split(','), maps), indent=2))


@app.local_entrypoint()
def test():
    result = checks.remote()
    print(result['output'])
    if result['returncode']:
        raise SystemExit(result['returncode'])


@app.local_entrypoint()
def mini_batch(prefix: str, smoke: bool = False):
    """Run both official tracks on DeepSeek and Luna high, all through OpenRouter."""
    from modal.config import config as modal_config
    if modal_config.get('environment') != 'staging':
        raise RuntimeError('pass --env staging')
    image_id = mini_coding_image.build(app).object_id
    key = os.environ['OPENROUTER_API_KEY']
    bound = mini_trial.with_options(secrets=[modal.Secret.from_dict({'OPENROUTER_API_KEY':key})])
    calls = []
    for label in ('deepseek','luna-high'):
        for track in ('oneshot-20m','feedback-3r'):
            cfg = json.loads((ROOT/f'configs/skirmish/mini-{label}-{track}.json').read_text())
            cfg['_coding_image_id'] = image_id
            if smoke: cfg.update(smoke=True,smoke_seconds=60)
            run_id = f'{prefix}-{label}-{track}'
            call = bound.spawn(run_id,cfg)
            calls.append((run_id,call))
            print(json.dumps(dict(run_id=run_id,call_id=call.object_id)),flush=True)
    for run_id,call in calls:
        try:
            report = call.get()
            # Small receipts only; source and full trajectories remain on Modal.
            print(json.dumps(dict(run_id=run_id,status=report['status'],api_cost_usd=report.get('api_cost_usd'),
                                  model_calls=report.get('model_calls'), final=report.get('final'),
                                  stages=[{k:v for k,v in stage.items() if k!='compile_output'}
                                          for stage in report.get('stages',[])])),flush=True)
        except Exception as error:
            print(json.dumps(dict(run_id=run_id,error=f'{type(error).__name__}: {str(error)[:500]}')),flush=True)


@app.local_entrypoint()
def mini_review(prefix: str, audit: bool = False):
    if not audit:
        print(json.dumps(mini_inspect.remote(prefix,False),indent=2))
        return
    selected=mini_inspect.remote(prefix,False)
    calls=[mini_inspect.spawn(run_id,True) for run_id in selected]
    merged={}
    for call in calls:merged.update(call.get())
    print(json.dumps(merged,indent=2))


@app.local_entrypoint()
def mini_rank(prefix: str, track: str = '', audit: bool = False):
    calls=[]
    tracks=[track] if track else ['oneshot-20m','feedback-3r']
    if any(t not in ('oneshot-20m','feedback-3r') for t in tracks):raise ValueError('unknown track')
    cohort=mini_inspect.remote(prefix,False) if not audit else {}
    for name in tracks:
        ident=f'{prefix}-{name}'
        if audit:calls.append(mini_ladder_audit.spawn(ident))
        else:
            run_ids=[f'{prefix}-{label}-{name}' for label in ('deepseek','luna-high')]
            states={key:cohort.get(key,{}).get('status') for key in run_ids}
            if any(state not in ('completed','failed_submission') for state in states.values()):
                raise RuntimeError(f'cohort is not ready or has infrastructure failures: {states}')
            excluded=[key for key,state in states.items() if state=='failed_submission']
            if excluded:print(json.dumps(dict(track=name,excluded_failed_submissions=excluded)),flush=True)
            run_ids=[key for key in run_ids if key not in excluded]
            if not run_ids:continue
            calls.append(mini_ladder.spawn(ident,run_ids))
    failures=[]
    for call in calls:
        try:print(json.dumps(call.get(),indent=2))
        except Exception as error:failures.append(str(error))
    if failures:raise RuntimeError('; '.join(failures))


@app.function(cpu=2,memory=2048,timeout=240,retries=0,volumes=MOUNTS)
def mini_probe(image_id: str):
    import time
    from minisweagent.exceptions import TimeExceeded
    from polytopia_bot.skirmish.mini import start_watchdog,cancel_watchdog,freeze
    from polytopia_bot.skirmish.mini_worker import ModalBashEnvironment
    sb=modal.Sandbox.create('sleep','infinity',image=modal.Image.from_id(image_id),
        app=modal.App.lookup('polytopia-skirmish-sandboxes',create_if_missing=True,environment_name='staging'),
        cpu=2,memory=2048,timeout=180,block_network=True,workdir='/bot')
    try:
        env=ModalBashEnvironment(sb,time.time()+90)
        compiled=env.execute(dict(command='cargo build --release --offline --locked --target wasm32-wasip1'))
        assert compiled['returncode']==0,compiled
        denied=env.execute(dict(command='touch /verify/forbidden'))
        assert denied['returncode']!=0,denied
        tools=env.execute(dict(command='skirmish compile && skirmish maps && skirmish check'))
        assert tools['returncode']==0,tools
        inspected=env.execute(dict(command='skirmish inspect --map continents-0 > /bot/map.json && '
            'python3 -c "import json; d=json.load(open(\'/bot/map.json\')); '
            'assert d[\'observation\'][\'visible_state\'][\'board_size\']==14"'))
        assert inspected['returncode']==0,inspected
        absent=env.execute(dict(command='python3 -c "import importlib.util as u; '
            'assert u.find_spec(\'polytopia_bot.rl.policy\') is None; '
            'assert u.find_spec(\'polytopia_bot.rl.search\') is None"'))
        assert absent['returncode']==0,absent
        large=env.execute(dict(command='#'+('x'*300000)+'\nprintf LARGE_COMMAND_OK\n'))
        assert large['returncode']==0 and 'LARGE_COMMAND_OK' in large['output'],large
        unicode=env.execute(dict(command="printf '%s\\n' 'literal:$() café'"))
        assert unicode['returncode']==0 and 'literal:$() café' in unicode['output'],unicode
        expanded=env.execute(dict(command="python3 - <<'PY'\nfrom pathlib import Path\nPath('/bot/src/main.rs').write_text('//'+('z'*250000)+'\\n'+Path('/reference/STARTER.rs').read_text())\nPY"))
        assert expanded['returncode']==0,expanded
        deadline=time.time()+3
        watchdog=start_watchdog(sb,deadline)
        env.deadline=deadline
        try:env.execute(dict(command='(sleep 5; touch /bot/late-write) & wait'))
        except TimeExceeded:pass
        watchdog.wait();cancel_watchdog(sb)
        check=sb.exec('test','!','-e','/bot/late-write');check.wait()
        assert check.returncode==0,'late background write escaped the stage deadline'
        captured=sb.filesystem.read_bytes('/verify/cutoffs/stage-0/main.rs')
        # A root-side post-cutoff mutation must not change the submitted source.
        sb.filesystem.write_text('this is not Rust','/bot/src/main.rs')
        folder=Path('/results/mini-probes')/str(time.time_ns())
        receipt=freeze(sb,folder,results,cutoff='/verify/cutoffs/stage-0')
        assert receipt['returncode']==0,receipt
        from polytopia_bot.skirmish.common import sha
        assert receipt['source_sha256']==sha(captured)
        assert freeze(sb,folder,results,cutoff='/verify/cutoffs/stage-0')==receipt
        return dict(coder_compilation=True,offline_tools=True,no_opponent_package=True,
                    trusted_build_protected=True,background_deadline=True,
                    trusted_compilation=True,large_command_transport=True,large_source=True,unicode_preserved=True,durable_cutoff=True,idempotent_freeze=True,source_sha256=receipt['source_sha256'])
    finally:sb.terminate()


@app.local_entrypoint()
def mini_preflight():
    print(json.dumps(mini_probe.remote(mini_coding_image.build(app).object_id),indent=2))


@app.function(cpu=2,memory=2048,timeout=240,retries=0,volumes=MOUNTS)
def mini_resume_probe(image_id: str):
    import subprocess,sys,time
    root=Path('/results/mini-probes')/('resume-'+str(time.time_ns()));root.mkdir(parents=True)
    sb=modal.Sandbox.create('sleep','infinity',image=modal.Image.from_id(image_id),
        app=modal.App.lookup('polytopia-skirmish-sandboxes',create_if_missing=True,environment_name='staging'),
        cpu=2,memory=2048,timeout=220,block_network=True,workdir='/bot')
    try:
        history=None;stats=[]
        for stage in (0,1):
            path=root/f'stage-{stage}.json'
            cfg=dict(model='openai/gpt-6-luna',parameters=dict(reasoning=dict(effort='high'),max_tokens=1024),
                     sandbox_id=sb.object_id,system='Use the bash tool. Follow the task exactly.',
                     task='This is a protocol test. Immediately run only: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT',
                     seconds=90,deadline_unix=time.time()+90,trajectory=str(path),history=history)
            p=subprocess.Popen([sys.executable,'-m','polytopia_bot.skirmish.mini_worker'],stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
            try:_,error=p.communicate(json.dumps(cfg),timeout=90)
            except subprocess.TimeoutExpired:p.kill();p.communicate();raise RuntimeError('resume probe deadline')
            assert p.returncode==0,error[-2000:]
            data=json.loads(path.read_text());assert data['info']['exit_status']=='Submitted',data['info']
            stats.append(data['info']['model_stats']);history=str(path)
        results.commit()
        return dict(early_submit_then_resume=True,provider='openrouter',model='openai/gpt-6-luna',stats=stats)
    finally:sb.terminate()


@app.local_entrypoint()
def mini_resume_test():
    image_id=mini_coding_image.build(app).object_id
    bound=mini_resume_probe.with_options(secrets=[modal.Secret.from_dict({'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})])
    print(json.dumps(bound.remote(image_id),indent=2))


@app.function(cpu=1,memory=1024,timeout=90,retries=0,volumes=MOUNTS)
def mini_toolchain_receipt(run_id: str):
    from polytopia_bot.skirmish.common import identifier,sha,write_json
    protocol=json.loads((Path('/results/mini-runs')/identifier(run_id)/'protocol.json').read_text())
    sb=modal.Sandbox.create('sleep','infinity',image=modal.Image.from_id(protocol['coding_image_id']),
        app=modal.App.lookup('polytopia-skirmish-sandboxes',create_if_missing=True,environment_name='staging'),
        cpu=1,memory=1024,timeout=60,block_network=True)
    try:
        lock=sb.filesystem.read_text('/verify/Cargo.lock')
        receipt=dict(coding_image_id=protocol['coding_image_id'],lock_sha256=sha(lock.encode()),
                     manifest_sha256=sha(sb.filesystem.read_bytes('/verify/Cargo.toml')),
                     starter_sha256=sha(sb.filesystem.read_bytes('/bot/src/main.rs')))
        root=Path('/results/mini-toolchains')/protocol['coding_image_id']
        root.mkdir(parents=True,exist_ok=True);(root/'Cargo.lock').write_text(lock)
        write_json(root/'receipt.json',receipt);results.commit()
        return dict(lock=lock,receipt=receipt)
    finally:sb.terminate()


@app.local_entrypoint()
def mini_pin_toolchain(run_id: str):
    import hashlib
    data=mini_toolchain_receipt.remote(run_id)
    template=ROOT/'skirmish/bot-template'
    for name,key in [('Cargo.toml','manifest_sha256'),('src/main.rs','starter_sha256')]:
        assert hashlib.sha256((template/name).read_bytes()).hexdigest()==data['receipt'][key]
    (template/'Cargo.lock').write_text(data['lock'])
    print(json.dumps(data['receipt'],indent=2))


@app.local_entrypoint()
def mini_calibrate(prefix: str, audit: bool = False):
    calls=[mini_ladder_audit.spawn(f'{prefix}-{track}-fuel1t') if audit else
           mini_fuel_calibration.spawn(f'{prefix}-{track}-fuel1t',f'{prefix}-{track}')
           for track in ('oneshot-20m','feedback-3r')]
    failures=[]
    for call in calls:
        try:print(json.dumps(call.get(),indent=2))
        except Exception as error:failures.append(str(error))
    if failures:raise RuntimeError('; '.join(failures))


@app.function(cpu=4, memory=8192, timeout=900, volumes=MOUNTS)
def preflight(coding_image_id: str):
    import time
    import tempfile
    from polytopia_bot.skirmish.common import Budget, REFERENCE_SHA
    from polytopia_bot.skirmish.authoring_tools import AuthoringTools
    from polytopia_bot.skirmish.game import ReferenceBot, WasiBot, play_game
    sb = modal.Sandbox.create('sleep', 'infinity', image=modal.Image.from_id(coding_image_id),
                             app=modal.App.lookup('polytopia-skirmish-sandboxes', create_if_missing=True, environment_name='staging'), block_network=True,
                             cpu=2, memory=2048, timeout=500)
    try:
        check = sb.exec('python', '-c',
            'import socket,json,os; s=socket.socket(); s.settimeout(1); '
            'r=s.connect_ex(("1.1.1.1",443)); '
            'print(json.dumps({"network_blocked":r!=0,"reference_absent":not os.path.exists("/results"),'
            '"credential_absent":"OPENROUTER_API_KEY" not in os.environ}))')
        isolation = json.loads(check.stdout.read())
        check.wait()
        assert all(isolation.values()), isolation
        root = Path('/results/preflight') / str(time.time_ns())
        root.mkdir(parents=True)
        budget = Budget(seconds=300, game_seconds=90)
        tools = AuthoringTools(sb, root, budget, results, time.monotonic()+300)
        tools.source = tools.read('/bot/src/main.rs')
        built = tools.compile()
        assert built['returncode'] == 0, built
        ref = ReferenceBot('/results/references/hero-1v1/final.npz', REFERENCE_SHA)
        rows = []
        for seat in (1, 2):
            rows.append(play_game(WasiBot(tools.artifact,budget.fuel,budget.move_seconds),ref,500999,seat,
                budget,root/f'seat-{seat}.jsonl',phase='practice',labels=['starter','hero-1v1']))
        report = dict(isolation=isolation, compile=built, games=rows, artifact_root=str(root))
        from polytopia_bot.skirmish.common import write_json
        write_json(root/'report.json',report)
        results.commit()
        return report
    finally:
        sb.terminate()


@app.local_entrypoint()
def smoke():
    image_id = coding_image.build(app).object_id
    print(json.dumps(preflight.remote(image_id),indent=2))


@app.local_entrypoint()
def reference(checkpoint: str):
    """Upload an already available released checkpoint; never loads model weights locally."""
    import hashlib
    import sys
    sys.path.insert(0, str(ROOT / 'src'))
    from polytopia_bot.skirmish.common import REFERENCE_SHA
    path = Path(checkpoint)
    meta_path = path.with_suffix('.json')
    meta = json.loads(meta_path.read_text())
    if meta['sha256'] != REFERENCE_SHA or hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA:
        raise ValueError('expected the pinned, released hero-1v1 checkpoint')
    with results.batch_upload(force=False) as upload:
        upload.put_file(path, '/references/hero-1v1/final.npz')
        upload.put_file(meta_path, '/references/hero-1v1/final.json')
    print(json.dumps({'reference': 'hero-1v1', 'sha256': REFERENCE_SHA, 'volume': 'polytopia-skirmish'}))


@app.function(cpu=4, memory=8192, timeout=1800, volumes=MOUNTS)
def audit_tournament(tournament_id: str):
    from dataclasses import asdict
    from polytopia_bot.skirmish.common import canonical, identifier, sha, write_json
    from polytopia_bot.studio.environment import RustEnvironment
    root = Path('/results/tournaments') / identifier(tournament_id)
    rows = json.loads((root/'games.json').read_text())
    checked_actions = 0
    for i, row in enumerate(rows):
        public_path = root/'games'/f'{i:06d}.jsonl'
        private_path = public_path.with_suffix('.private.jsonl')
        assert sha(public_path.read_bytes()) == row['trace_sha256']
        assert sha(private_path.read_bytes()) == row['private_trace_sha256']
        public = iter(json.loads(line) for line in public_path.read_text().splitlines())
        env = RustEnvironment(size=14)
        try:
            env.reset(seed=row['seed'],settings={})
            chain = '0'*64
            count = 0
            for line in private_path.read_text().splitlines():
                entry = json.loads(line)
                expected_chain = entry.pop('chain')
                chain = sha((chain+canonical(entry)).encode())
                assert chain == expected_chain
                obs = env.observe(int(env.state.current_player))
                assert obs.player == entry['player'] and obs.state_version == entry['version']
                assert obs.visible_state['turn'] == entry['turn']
                if obs.player == row['seat']:
                    expected = next(public)
                    assert expected == dict(observation=asdict(obs),action_id=entry['action_id'])
                env.step(action_id=entry['action_id'],expected_state_version=obs.state_version)
                count += 1
            assert next(public,None) is None
            assert count == row['actions'] and chain == row['action_chain_sha256']
            assert int(env.state.turn) == row['turns']
            if row['status'] == 'native':
                assert bool(env.state.terminated)
                rewards = list(env.games.batch.terminal_rewards()[0])
                assert rewards[row['winner']-1] > 0
            elif row['status'] == 'forfeit':
                assert not bool(env.state.terminated) and row['winner'] == 3-row['fault_player']
            checked_actions += count
        finally:
            env.close()
    report = dict(tournament_id=tournament_id,passed=True,games=len(rows),actions=checked_actions)
    write_json(root/'audit.json',report)
    results.commit()
    return report


@app.local_entrypoint()
def audit(tournament_id: str):
    print(json.dumps(audit_tournament.remote(tournament_id),indent=2))


@app.function(cpu=1, memory=2048, timeout=30, volumes={'/results': results})
def inspect_run(run_id: str):
    """Small diagnostic metadata only; full source, model transcripts and game traces stay on Modal."""
    from collections import Counter
    from polytopia_bot.skirmish.common import identifier
    root = Path('/results/runs') / identifier(run_id)
    status = json.loads((root/'status.json').read_text())
    tools, errors, compile_codes = Counter(), Counter(), Counter()
    cost, models = 0., Counter()
    path = root/'transcript.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try: entry = json.loads(line)
            except json.JSONDecodeError: continue
            if entry['type'] == 'tool':
                tools[entry['call']['name']] += 1
                if 'error' in entry['result']: errors[entry['result']['error']] += 1
                if entry['call']['name'] == 'compile': compile_codes[str(entry['result'].get('returncode'))] += 1
            else:
                response = entry['response']
                models[response.get('model','unknown')] += 1
                cost += response.get('usage',{}).get('cost') or 0.
    games = [json.loads(p.read_text()) for p in (root/'practice').glob('*.summary.json')]
    return dict(status=status,tools=dict(tools),errors=dict(errors),compile_returncodes=dict(compile_codes),
                models_returned=dict(models),api_reported_cost_usd=cost,
                practice_results=dict(Counter(r['result'] for r in games)),
                practice_reasons=dict(Counter(r['reason'] for r in games)))


@app.local_entrypoint()
def inspect(run_id: str):
    print(json.dumps(inspect_run.remote(run_id),indent=2))


@app.function(cpu=4, memory=8192, timeout=600, volumes=MOUNTS)
def diagnose_guest_exit(fuel: int = 2_000_000_000):
    import subprocess
    from unittest.mock import patch
    from polytopia_bot.skirmish.common import Budget
    from polytopia_bot.skirmish.game import WasiBot, play_game
    original = subprocess.Popen
    def captured(*args, **kwargs):
        kwargs['stderr'] = subprocess.PIPE
        return original(*args, **kwargs)
    with patch('subprocess.Popen', captured):
        a = WasiBot('/results/runs/deepseek-pilot-20261030-03/bot.wasm', fuel, 2.)
        b = WasiBot('/results/runs/gpt56-pilot-20261030-01/bot.wasm', fuel, 2.)
    import time
    row = play_game(a,b,703195,1,Budget(),Path('/results/diagnostics')/f'{time.time_ns()}.jsonl',
                    phase='diagnostic',labels=['deepseek','gpt56'])
    return dict(game=row, a_stderr=a.proc.stderr.read().decode(errors='replace')[-2000:],
                b_stderr=b.proc.stderr.read().decode(errors='replace')[-2000:])


@app.local_entrypoint()
def diagnose(fuel: int = 2_000_000_000):
    print(json.dumps(diagnose_guest_exit.remote(fuel),indent=2))


@app.function(cpu=1, memory=2048, timeout=86400, max_containers=1, retries=0, volumes=MOUNTS)
def study_coordinator(spec: dict, coding_image_id: str):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from polytopia_bot.skirmish.conditions import build_study
    from polytopia_bot.skirmish.common import identifier, write_json
    plan = build_study(spec)
    root = Path('/results/studies')/identifier(plan['study_id'])
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'plan.json',plan)
    results.commit()
    def one(row):
        cfg = dict(row['config'], _coding_image_id=coding_image_id)
        key_env = cfg['key_env']
        bound = trial.with_options(secrets=[modal.Secret.from_dict({key_env:os.environ[key_env]})])
        call = bound.spawn(row['run_id'],cfg)
        write_json(root/'calls'/f'{row["run_id"]}.json',dict(function_call_id=call.object_id))
        results.commit()
        try:
            return dict(run_id=row['run_id'],result=call.get())
        except Exception as error:
            return dict(run_id=row['run_id'],error=f'{type(error).__name__}: {str(error)[:300]}')
    completed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(one,row) for row in plan['assignments']]
        for future in as_completed(futures):
            completed.append(future.result())
            write_json(root/'status.json',dict(status='authoring',completed=len(completed),total=len(futures),runs=completed))
            results.commit()
    results.reload()
    valid = [r['run_id'] for r in completed if (Path('/results/runs')/r['run_id']/'submission.json').exists()]
    for row in completed:
        run_root = Path('/results/runs')/row['run_id']
        if row['run_id'] in valid:
            row['metadata'] = json.loads((run_root/'submission.json').read_text())
        if (run_root/'status.json').exists():
            row['result'] = json.loads((run_root/'status.json').read_text())
    failures = [r for r in completed if r['run_id'] not in valid]
    report = dict(study_id=plan['study_id'],status='completed',allocated=len(completed),
                  valid_submissions=len(valid),failed_submissions=failures,runs=completed,
                  interpretation='Functional/behavioral records; no automatic causal significance claim')
    if valid:
        report['tournaments'] = {}
        for replicate in range(spec['replicates']):
            block = [r['run_id'] for r in plan['assignments'] if r['replicate']==replicate and r['run_id'] in valid]
            if block:
                report['tournaments'][str(replicate)] = run_ladder.remote(
                    f'study-{plan["study_id"]}-r{replicate}',block,int(spec.get('tournament_maps',4)))
    from polytopia_bot.skirmish.conditions import summarize_study
    report['analysis'] = summarize_study(plan,completed,report.get('tournaments',{}))
    write_json(root/'report.json',report)
    write_json(root/'status.json',report)
    results.commit()
    return report


@app.local_entrypoint()
def study(spec: str):
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from polytopia_bot.skirmish.conditions import build_study
    cfg = json.loads(Path(spec).read_text())
    build_study(cfg)  # validate and materialize before any dispatch
    key_vars = {m['config']['key_env'] for m in cfg['models']}
    if any(not os.environ.get(k) for k in key_vars):
        raise RuntimeError('every configured model key environment variable must be set')
    image_id = coding_image.build(app).object_id
    bound = study_coordinator.with_options(secrets=[modal.Secret.from_dict({k:os.environ[k] for k in key_vars})])
    print(json.dumps(bound.remote(cfg,image_id),indent=2))


@app.function(cpu=4,memory=8192,timeout=2000,max_containers=8,retries=0,volumes=MOUNTS)
def match_group(payload: dict):
    from polytopia_bot.skirmish.tournament import play_group
    return play_group(payload,results)


@app.function(cpu=1,memory=2048,timeout=240,retries=0,max_containers=10,volumes=MOUNTS)
def mini_model_probe(prefix: str, entry: dict, image_id: str):
    """Two short actual upstream-agent stages exercise tool calls and provider-safe resume."""
    import subprocess,sys,time
    from polytopia_bot.skirmish.common import identifier,write_json
    from polytopia_bot.skirmish.mini import stop_coder
    root=Path('/results/mini-cohorts')/identifier(prefix)/'probes'/identifier(entry['label'])
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'config.json',entry);results.commit()
    sb=modal.Sandbox.create('sleep','infinity',image=modal.Image.from_id(image_id),
        app=modal.App.lookup('polytopia-skirmish-sandboxes',create_if_missing=True,environment_name='staging'),
        cpu=2,memory=2048,timeout=200,block_network=True,workdir='/bot')
    history=None;stats=[]
    try:
        for stage in range(2):
            task=('This is a transport compatibility test, not a benchmark attempt. '
                  'Do not read rules or write a bot. First call bash with `printf SMOKE_OK`. '
                  'After its result, submit with bash `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. '
                  'Do not combine these two commands. Use no other tools.')
            path=root/f'stage-{stage}.trajectory.json'
            parameters={**entry['parameters'],'max_tokens':2048}
            cfg=dict(model=entry['model'],parameters=parameters,sandbox_id=sb.object_id,
                     system='You are a coding agent. Use the bash tool to follow the task exactly.',task=task,
                     seconds=60,deadline_unix=time.time()+60,trajectory=str(path),history=history)
            with (root/f'stage-{stage}.log').open('w') as log:
                p=subprocess.Popen([sys.executable,'-m','polytopia_bot.skirmish.mini_worker'],
                    stdin=subprocess.PIPE,stdout=log,stderr=log,text=True)
                try:p.communicate(json.dumps(cfg),timeout=60)
                except subprocess.TimeoutExpired:p.kill();p.communicate();raise RuntimeError('probe deadline')
            stop_coder(sb)
            data=json.loads(path.read_text())
            if p.returncode or data['info'].get('exit_status')!='Submitted':
                raise RuntimeError(f"probe exit: {data['info'].get('exit_status')} worker={p.returncode}")
            messages=data['messages']
            if not any(m.get('role')=='tool' and 'SMOKE_OK' in str(m.get('content','')) for m in messages):
                raise RuntimeError('probe did not execute the requested bash command')
            stats.append(data['info']['model_stats']);history=str(path)
        report=dict(label=entry['label'],model=entry['model'],passed=True,stages=stats,
                    api_cost_usd=sum(s['instance_cost'] for s in stats))
    except Exception as error:
        report=dict(label=entry['label'],model=entry['model'],passed=False,
                    error=f'{type(error).__name__}: {error}',stages=stats)
    finally:sb.terminate()
    write_json(root/'report.json',report);results.commit()
    return report


@app.local_entrypoint()
def mini_cohort_probes(prefix: str, manifest: str = 'configs/skirmish/mini-top10.json'):
    from modal.config import config as modal_config
    if modal_config.get('environment')!='staging':raise RuntimeError('pass --env staging')
    data=json.loads(Path(manifest).read_text())
    image_id=mini_coding_image.build(app).object_id
    bound=mini_model_probe.with_options(secrets=[modal.Secret.from_dict(
        {'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})])
    calls=[bound.spawn(prefix,entry,image_id) for entry in data['models']]
    print(json.dumps(dict(prefix=prefix,image_id=image_id,probes=len(calls))),flush=True)
    for call in calls:print(json.dumps(call.get()),flush=True)


@app.function(cpu=1,memory=2048,timeout=14400,retries=0,volumes=MOUNTS)
def mini_cohort_controller(prefix: str, manifest: dict, image_id: str):
    """Dispatch all authoring cells, then both frozen-bot ladders and replay audits."""
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from polytopia_bot.skirmish.common import identifier,write_json,sha,canonical
    identifier(prefix)
    if manifest['tracks']!=['oneshot-20m','feedback-3r']:raise ValueError('official tracks required')
    entries=manifest['models']
    if len({e['label'] for e in entries})!=len(entries):raise ValueError('duplicate labels')
    root=Path('/results/mini-cohorts')/prefix
    results.reload()
    for entry in entries:
        probe=json.loads((root/'probes'/identifier(entry['label'])/'report.json').read_text())
        if not probe['passed']:raise RuntimeError(f"failed compatibility probe: {entry['label']}")
    root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():raise ValueError('cohort already allocated; do not overwrite or launch twice')
    jobs=[dict(run_id=f"{prefix}-{entry['label']}-{track}",
               config=dict(harness='mini-swe-agent',provider='openrouter',model=entry['model'],
                           parameters=entry['parameters'],track=track,_coding_image_id=image_id))
          for entry in entries for track in manifest['tracks']]
    plan=dict(prefix=prefix,manifest=manifest,manifest_sha256=sha(canonical(manifest).encode()),
              coding_image_id=image_id,jobs=jobs,authoring_parallelism=20,game_parallelism=32)
    write_json(root/'plan.json',plan)
    write_json(root/'status.json',dict(status='dispatching',completed=0,total=len(jobs)))
    results.commit()
    bound=mini_trial.with_options(secrets=[modal.Secret.from_dict(
        {'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})])
    calls=[]
    for job in jobs:
        call=bound.spawn(job['run_id'],job['config'])
        calls.append((job['run_id'],call))
        write_json(root/'calls'/f"{job['run_id']}.json",dict(function_call_id=call.object_id))
    results.commit()
    completed=[]
    def get_run(item):
        run_id,call=item
        try:
            report=call.get()
            return dict(run_id=run_id,status=report['status'],model=report.get('model'),
                track=report.get('track'),api_cost_usd=report.get('api_cost_usd'),
                model_calls=report.get('model_calls'),error=report.get('error'))
        except Exception as error:return dict(run_id=run_id,status='infrastructure_error',error=str(error)[:800])
    with ThreadPoolExecutor(max_workers=20) as pool:
        for future in as_completed([pool.submit(get_run,item) for item in calls]):
            completed.append(future.result())
            write_json(root/'status.json',dict(status='authoring',completed=len(completed),total=len(jobs),runs=completed))
            results.commit()
    failures=[r for r in completed if r['status'] not in ('completed','failed_submission')]
    report=dict(prefix=prefix,status='needs_attention' if failures else 'evaluating',runs=completed,
                infrastructure_failures=failures,tournaments={},audits={},errors=[])
    if not failures:
        pending=[]
        for track in manifest['tracks']:
            valid=[r['run_id'] for r in completed if r['track']==track and r['status']=='completed']
            if valid:pending.append(('tournament',track,mini_ladder.spawn(f'{prefix}-{track}',sorted(valid))))
        for row in completed:
            pending.append(('audit',row['run_id'],mini_inspect.spawn(row['run_id'],True)))
        write_json(root/'status.json',report);results.commit()
        def get_eval(item):
            kind,name,call=item
            try:return kind,name,call.get(),None
            except Exception as error:return kind,name,None,str(error)[:800]
        with ThreadPoolExecutor(max_workers=24) as pool:
            for future in as_completed([pool.submit(get_eval,item) for item in pending]):
                kind,name,data,error=future.result()
                if error:report['errors'].append(dict(kind=kind,name=name,error=error))
                elif kind=='tournament':report['tournaments'][name]=data
                else:report['audits'][name]={k:v.get('audit') for k,v in data.items()}
                write_json(root/'status.json',report);results.commit()
        for track in report['tournaments']:
            try:report['tournaments'][track]['audit']=mini_ladder_audit.remote(f'{prefix}-{track}')
            except Exception as error:report['errors'].append(dict(kind='ladder_audit',name=track,error=str(error)[:800]))
        report['status']='completed' if not report['errors'] else 'needs_attention'
    write_json(root/'report.json',report);write_json(root/'status.json',report);results.commit()
    return report


@app.local_entrypoint()
def mini_cohort_start(prefix: str, manifest: str = 'configs/skirmish/mini-top10.json'):
    from modal.config import config as modal_config
    if modal_config.get('environment')!='staging':raise RuntimeError('pass --env staging')
    data=json.loads(Path(manifest).read_text())
    image_id=mini_coding_image.build(app).object_id
    bound=mini_cohort_controller.with_options(secrets=[modal.Secret.from_dict(
        {'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})])
    call=bound.spawn(prefix,data,image_id)
    print(json.dumps(dict(prefix=prefix,controller_call_id=call.object_id,runs=2*len(data['models']))),flush=True)
    print(json.dumps(call.get()),flush=True)


@app.function(cpu=1,memory=2048,timeout=14400,retries=0,volumes=MOUNTS)
def mini_cohort_finish(prefix: str):
    """Recover canonical frozen results after a duplicate-dispatch transport error; never reauthor a bot."""
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from polytopia_bot.skirmish.common import identifier,write_json
    from polytopia_bot.skirmish.mini import load_submission
    root=Path('/results/mini-cohorts')/identifier(prefix)
    results.reload()
    plan=json.loads((root/'plan.json').read_text())
    replacements=json.loads((root/'replacements.json').read_text()) if (root/'replacements.json').exists() else {}
    prior=json.loads((root/'report.json').read_text()) if (root/'report.json').exists() else {}
    rows=[]
    for job in plan['jobs']:
        run_id=replacements.get(job['run_id'],{}).get('run_id',job['run_id'])
        path=Path('/results/mini-runs')/run_id/'report.json'
        if not path.exists():raise RuntimeError(f"run still active: {run_id}")
        result=json.loads(path.read_text())
        if result['status'] not in ('completed','failed_submission'):
            raise RuntimeError(f"nonterminal or genuine infrastructure failure: {job['run_id']}={result['status']}")
        if result['status']=='completed':load_submission(run_id)
        rows.append(dict(run_id=run_id,status=result['status'],model=result['model'],
            track=result['track'],api_cost_usd=result.get('api_cost_usd'),model_calls=result.get('model_calls')))
    report=dict(prefix=prefix,status='evaluating',runs=rows,tournaments={},audits={},errors=[],
                dispatch_errors=prior.get('infrastructure_failures',[]),
                replacements=replacements,
                recovery='Read canonical results; infrastructure-only replacements are listed explicitly.')
    pending=[]
    for track in plan['manifest']['tracks']:
        ids=sorted(r['run_id'] for r in rows if r['status']=='completed' and r['track']==track)
        existing=Path('/results/mini-tournaments')/f'{prefix}-{track}'/'report.json'
        call_path=root/'ladder-calls'/f'{track}.json'
        if existing.exists():report['tournaments'][track]=json.loads(existing.read_text())
        elif call_path.exists():
            pending.append(('tournament',track,modal.FunctionCall.from_id(json.loads(call_path.read_text())['call_id'])))
        else:pending.append(('tournament',track,mini_ladder.spawn(f'{prefix}-{track}',ids)))
    for row in rows:pending.append(('audit',row['run_id'],mini_inspect.spawn(row['run_id'],True)))
    write_json(root/'status.json',report);results.commit()
    def get_one(item):
        kind,name,call=item
        try:return kind,name,call.get(),None
        except Exception as error:return kind,name,None,str(error)[:1000]
    with ThreadPoolExecutor(max_workers=24) as pool:
        for future in as_completed([pool.submit(get_one,item) for item in pending]):
            kind,name,data,error=future.result()
            if error:report['errors'].append(dict(kind=kind,name=name,error=error))
            elif kind=='tournament':report['tournaments'][name]=data
            else:report['audits'][name]={k:v.get('audit') for k,v in data.items()}
            write_json(root/'status.json',report);results.commit()
    audit_calls=[(track,mini_ladder_audit.spawn(f'{prefix}-{track}')) for track in report['tournaments']]
    for track,call in audit_calls:
        try:report['tournaments'][track]['audit']=call.get()
        except Exception as error:report['errors'].append(dict(kind='ladder_audit',name=track,error=str(error)[:1000]))
    report['status']='completed' if not report['errors'] else 'needs_attention'
    write_json(root/'report.json',report);write_json(root/'status.json',report);results.commit()
    return report


@app.local_entrypoint()
def mini_cohort_recover(prefix: str):
    call=mini_cohort_finish.spawn(prefix)
    print(json.dumps(dict(prefix=prefix,recovery_call_id=call.object_id)),flush=True)
    print(json.dumps(call.get()),flush=True)


@app.function(cpu=.25,memory=512,timeout=14400,retries=0,volumes=MOUNTS)
def mini_cohort_recovery_guard(prefix: str):
    import time
    from polytopia_bot.skirmish.common import identifier,write_json
    root=Path('/results/mini-cohorts')/identifier(prefix)
    while True:
        results.reload()
        report_path=root/'report.json'
        if report_path.exists():
            report=json.loads(report_path.read_text())
            if report['status']=='completed':return dict(status='not_needed')
            failures=report.get('infrastructure_failures',[])
            if report['status']=='needs_attention':
                replacements=json.loads((root/'replacements.json').read_text()) if (root/'replacements.json').exists() else {}
                if not failures or any('File exists' not in str(r.get('error') or '') and r['run_id'] not in replacements for r in failures):
                    return dict(status='needs_attention',reason='genuine failure requires inspection')
                plan=json.loads((root/'plan.json').read_text())
                ready=True
                for job in plan['jobs']:
                    run_id=replacements.get(job['run_id'],{}).get('run_id',job['run_id'])
                    path=Path('/results/mini-runs')/run_id/'report.json'
                    if not path.exists():ready=False;break
                    if json.loads(path.read_text())['status'] not in ('completed','failed_submission'):
                        return dict(status='needs_attention',run_id=job['run_id'])
                if ready:
                    write_json(root/'recovery.json',dict(status='recovering',reason='infrastructure recovery',
                        repeated_authoring_attempts=0));results.commit()
                    return mini_cohort_finish.remote(prefix)
        time.sleep(30)


@app.local_entrypoint()
def mini_cohort_guard(prefix: str):
    call=mini_cohort_recovery_guard.spawn(prefix)
    print(json.dumps(dict(prefix=prefix,guard_call_id=call.object_id)),flush=True)
    print(json.dumps(call.get()),flush=True)


@app.function(cpu=1,memory=1024,timeout=60,retries=0)
def mini_accounting_probe():
    import subprocess,sys,os
    script='''import inspect,json
from unittest.mock import patch
from minisweagent.models.openrouter_model import OpenRouterModel
model=OpenRouterModel(model_name="deepseek/deepseek-v4-pro-0813")
print("cost_tracking",model.config.cost_tracking)
print(inspect.getsource(type(model)._calculate_cost))
fixture=dict(choices=[dict(finish_reason="tool_calls",message=dict(role="assistant",content=None,
    tool_calls=[dict(id="zero-cost",type="function",function=dict(name="bash",arguments=json.dumps(dict(command="printf ZERO_COST_OK"))))]))],
    usage=dict(cost=0,prompt_tokens=3,completion_tokens=1,total_tokens=4,cost_details=dict(upstream_inference_cost=.001)))
with patch.object(model,"_query",return_value=fixture):
 out=model.query([model.format_message(role="user",content="Use bash")])
assert out["extra"]["actions"][0]["command"]=="printf ZERO_COST_OK",out
print("zero_cost_response_preserved",True)
'''
    proc=subprocess.run([sys.executable,'-c',script],env={**os.environ,'MSWEA_COST_TRACKING':'ignore_errors'},
                        capture_output=True,text=True,timeout=45)
    return dict(returncode=proc.returncode,output=proc.stdout,stderr=proc.stderr[-1200:])


@app.local_entrypoint()
def mini_check_accounting():print(json.dumps(mini_accounting_probe.remote(),indent=2))


@app.function(cpu=.25,memory=512,timeout=14400,retries=0,volumes=MOUNTS)
def mini_cohort_replace_accounting_failure(prefix: str, old_run_id: str):
    from polytopia_bot.skirmish.common import identifier,write_json
    root=Path('/results/mini-cohorts')/identifier(prefix);results.reload()
    plan=json.loads((root/'plan.json').read_text())
    job=next(j for j in plan['jobs'] if j['run_id']==old_run_id)
    original=Path('/results/mini-runs')/identifier(old_run_id)
    status=json.loads((original/'report.json').read_text())
    if status['status']!='failed_harness' or not any('No valid cost information available' in p.read_text()
            for p in original.glob('*.worker.log')):raise RuntimeError('not the identified accounting-only infrastructure failure')
    replacement_path=root/'replacements.json'
    replacements=json.loads(replacement_path.read_text()) if replacement_path.exists() else {}
    if old_run_id in replacements:raise ValueError('replacement already allocated')
    run_id=old_run_id+'-r1'
    bound=mini_trial.with_options(secrets=[modal.Secret.from_dict({
        'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY'],'MSWEA_COST_TRACKING':'ignore_errors'})])
    call=bound.spawn(run_id,job['config'])
    replacements[old_run_id]=dict(run_id=run_id,call_id=call.object_id,
        reason='OpenRouter reported cost=0; strict accounting aborted the worker.',
        cost_tracking='ignore_errors',same_prompt_budget_provider_and_image=True)
    write_json(replacement_path,replacements);results.commit()
    return call.get()


@app.local_entrypoint()
def mini_cohort_retry_accounting(prefix: str, run_id: str):
    bound=mini_cohort_replace_accounting_failure.with_options(secrets=[modal.Secret.from_dict(
        {'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})])
    call=bound.spawn(prefix,run_id)
    print(json.dumps(dict(prefix=prefix,retry_controller_call_id=call.object_id)),flush=True)
    print(json.dumps(call.get()),flush=True)


@app.function(cpu=.25,memory=512,timeout=14400,retries=0,volumes=MOUNTS)
def mini_cohort_track_ladder(prefix: str, track: str):
    from polytopia_bot.skirmish.common import identifier,write_json
    from polytopia_bot.skirmish.mini import load_submission
    root=Path('/results/mini-cohorts')/identifier(prefix);results.reload()
    plan=json.loads((root/'plan.json').read_text())
    replacements=json.loads((root/'replacements.json').read_text()) if (root/'replacements.json').exists() else {}
    ids=[]
    for job in plan['jobs']:
        if job['config']['track']!=track:continue
        run_id=replacements.get(job['run_id'],{}).get('run_id',job['run_id'])
        path=Path('/results/mini-runs')/run_id/'report.json'
        if not path.exists():raise RuntimeError(f'authoring still active: {run_id}')
        report=json.loads(path.read_text())
        if report['status']=='completed':load_submission(run_id);ids.append(run_id)
        elif report['status']!='failed_submission':raise RuntimeError(f'infrastructure failure: {run_id}')
    if not ids:raise RuntimeError('no valid entrants')
    path=root/'ladder-calls'/f'{track}.json'
    if path.exists():raise ValueError('track ladder already allocated')
    call=mini_ladder.spawn(f'{prefix}-{track}',sorted(ids))
    write_json(path,dict(call_id=call.object_id,entrants=ids));results.commit()
    return call.get()


@app.local_entrypoint()
def mini_cohort_rank_track(prefix: str, track: str):
    call=mini_cohort_track_ladder.spawn(prefix,track)
    print(json.dumps(dict(prefix=prefix,track=track,track_controller_call_id=call.object_id)),flush=True)
    print(json.dumps(call.get()),flush=True)


@app.function(cpu=.25,memory=512,timeout=14400,retries=0,volumes=MOUNTS)
def mini_cohort_replace_lost_worker(prefix: str, old_run_id: str):
    import time
    from polytopia_bot.skirmish.common import identifier,write_json
    root=Path('/results/mini-cohorts')/identifier(prefix);results.reload()
    plan=json.loads((root/'plan.json').read_text())
    job=next(j for j in plan['jobs'] if j['run_id']==old_run_id)
    original=Path('/results/mini-runs')/identifier(old_run_id)
    state=json.loads((original/'status.json').read_text())
    call_data=json.loads((root/'calls'/f'{old_run_id}.json').read_text())
    call=modal.FunctionCall.from_id(call_data['function_call_id'])
    try:call.get(timeout=0)
    except FileExistsError:pass
    else:raise RuntimeError('original call did not fail with the recorded duplicate-root error')
    if state['status']!='authoring' or state['stage_deadline_unix']>time.time()-60 or (original/'report.json').exists():
        raise RuntimeError('original is not an expired orphaned authoring attempt')
    path=root/'replacements.json';replacements=json.loads(path.read_text()) if path.exists() else {}
    if old_run_id in replacements:raise ValueError('already replaced')
    run_id=old_run_id+'-r1'
    bound=mini_trial.with_options(secrets=[modal.Secret.from_dict({
        'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY'],'MSWEA_COST_TRACKING':'ignore_errors'})])
    retry=bound.spawn(run_id,job['config'])
    replacements[old_run_id]=dict(run_id=run_id,call_id=retry.object_id,
        reason='Controller call failed with duplicate-root error; expired original never froze a submission.',
        cost_tracking='ignore_errors',same_prompt_budget_provider_and_image=True)
    write_json(path,replacements);results.commit()
    return retry.get()


@app.local_entrypoint()
def mini_cohort_retry_lost_worker(prefix: str, run_id: str):
    bound=mini_cohort_replace_lost_worker.with_options(secrets=[modal.Secret.from_dict(
        {'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']})])
    call=bound.spawn(prefix,run_id)
    print(json.dumps(dict(prefix=prefix,retry_controller_call_id=call.object_id)),flush=True)
    print(json.dumps(call.get()),flush=True)


@app.function(cpu=4,memory=8192,timeout=1800,retries=0,volumes=MOUNTS)
def mini_restore_sol_submission(prefix: str):
    """Restore only deterministic recorded source writes after the undisclosed size cap."""
    import hashlib,shutil,secrets,time
    from polytopia_bot.skirmish.common import Budget,REFERENCE_SHA,write_json,sha
    from polytopia_bot.skirmish.mini import evaluate
    from polytopia_bot.skirmish.game import warm_reference
    results.reload()
    cohort=Path('/results/mini-cohorts')/prefix
    old_id=prefix+'-sol61-oneshot-20m-r1';new_id=old_id+'-restored'
    old=Path('/results/mini-runs')/old_id;new=old.parent/new_id
    status=json.loads((old/'status.json').read_text())
    if status.get('error')!='ValueError: main.rs exceeds 200000 bytes':raise RuntimeError('unexpected recovery target')
    protocol=json.loads((old/'protocol.json').read_text())
    trajectory=json.loads((old/'stage-0.trajectory.json').read_text())
    assert trajectory['info']['exit_status']=='Submitted'
    operations=json.loads((cohort/'source-recovery/sol-operations.json').read_text())
    # All these writes are literal patches or generation from immutable reference files;
    # none reads practice results or external mutable data to choose source content.
    runner='''import json,subprocess,sys
from pathlib import Path
for op in json.load(open('/tmp/source-operations.json')):
 kind=op['kind']
 if kind in ('write','append'):
  with open(op['path'],'a' if kind=='append' else 'w') as f:f.write(op['text'])
 elif kind=='generator':
  Path(op['path']).write_text(op['text']);subprocess.run([sys.executable,op['path']],cwd='/bot',check=True,timeout=30)
 elif kind=='python':subprocess.run([sys.executable,'-c',op['text']],cwd='/bot',check=True,timeout=30)
 elif kind=='argv':subprocess.run(op['argv'],cwd='/bot',check=True,timeout=30)
'''
    sandboxes=[];passes=[];source=None
    try:
        for index in range(2):
            sb=modal.Sandbox.create('sleep','infinity',image=modal.Image.from_id(protocol['coding_image_id']),
                app=modal.App.lookup('polytopia-skirmish-sandboxes',create_if_missing=True,environment_name='staging'),
                cpu=2,memory=2048,timeout=360,block_network=True,workdir='/bot')
            sandboxes.append(sb)
            assert sha(sb.filesystem.read_bytes('/reference/RULES.md'))==protocol['rules_sha256']
            sb.filesystem.write_text(json.dumps(operations),'/tmp/source-operations.json')
            proc=sb.exec('/usr/sbin/runuser','-u','coder','--','python','-c',runner,timeout=120)
            output=proc.stdout.read()+proc.stderr.read();proc.wait()
            if proc.returncode:raise RuntimeError(output[-2000:])
            source=sb.filesystem.read_text('/bot/src/main.rs')
            passes.append(dict(sha256=sha(source.encode()),lines=len(source.splitlines()),bytes=len(source.encode())))
            (cohort/'source-recovery'/f'sol-pass-{index}.rs').write_text(source)
        assert passes[0]==passes[1],passes
        assert passes[0]['lines']==6425,'must match last recorded wc -l output'
        new.mkdir(parents=True,exist_ok=False);stage=new/'stage-0';stage.mkdir()
        sb=sandboxes[-1]
        sb.filesystem.write_text(source,'/verify/src/main.rs')
        proc=sb.exec('cargo','build','--manifest-path','/verify/Cargo.toml','--release','--offline','--locked',
                    '--target','wasm32-wasip1',timeout=90)
        output=proc.stdout.read()+proc.stderr.read();proc.wait()
        assert proc.returncode==0,output[-3000:]
        wasm=sb.filesystem.read_bytes('/verify/target/wasm32-wasip1/release/skirmish-bot.wasm')
        assert wasm.startswith(b'\0asm')
        (stage/'main.rs').write_text(source);(stage/'bot.wasm').write_bytes(wasm)
        stats=trajectory['info']['model_stats']
        receipt=dict(stage=0,returncode=0,source_sha256=sha(source.encode()),wasm_sha256=sha(wasm),
            compile_output=output[-12000:],authoring_seconds=None,authoring_seconds_bounds=[1173,1200],
            agent_exit_status='Submitted',end_reason='agent_exit',**stats)
        write_json(stage/'submission.json',receipt);write_json(new/'stages.json',[receipt])
        for name in ['protocol.json','source.tar.gz','runtime-packages.txt','stage-0.trajectory.json','stage-0.worker.log']:
            shutil.copyfile(old/name,new/name)
        (new/'main.rs').write_text(source);(new/'bot.wasm').write_bytes(wasm)
        recovery=dict(from_run_id=old_id,method='Replay only recorded deterministic source-writing operations twice in original immutable image.',
            extra_model_calls=0,source_edits_added=0,extra_agent_feedback=0,passes=passes,
            recorded_final_line_count=6425,source_operations_sha256=sha(json.dumps(operations,sort_keys=True).encode()),
            reason='Undisclosed 200 KB source cap blocked freeze; authoring had already submitted and compiled this source.')
        write_json(new/'recovery.json',recovery)
        write_json(new/'status.json',dict(run_id=new_id,model=protocol['model'],track=protocol['track'],status='final_evaluation'))
        results.commit()
    finally:
        for sb in sandboxes:sb.terminate()
    budget=Budget(**protocol['runtime']).validate()
    reference=warm_reference('/results/references/hero-1v1/final.npz',REFERENCE_SHA)
    seeds=secrets.SystemRandom().sample(range(1_000_000_000,2_000_000_000),3)
    final=evaluate(stage/'bot.wasm',new/'final',reference,budget,seeds,'final',results)
    report=dict(run_id=new_id,model=protocol['model'],track=protocol['track'],status='completed',smoke=False,
        stages=[receipt],final=final,api_cost_usd=stats['instance_cost'],model_calls=stats['api_calls'],
        final_source_sha256=sha(source.encode()),final_wasm_sha256=sha(wasm),protocol_sha256=sha((new/'protocol.json').read_bytes()),
        trajectory_sha256={'stage-0.trajectory.json':sha((new/'stage-0.trajectory.json').read_bytes())},source_recovery=recovery)
    write_json(new/'report.json',report);write_json(new/'status.json',report)
    path=cohort/'replacements.json';replacements=json.loads(path.read_text())
    original_id=prefix+'-sol61-oneshot-20m'
    replacements[original_id]['run_id']=new_id
    replacements[old_id]=dict(run_id=new_id,reason='Exact deterministic source-write replay after undisclosed source-cap failure; no extra model coding.',
        recovery=recovery,fresh_authoring_attempt=False)
    write_json(path,replacements);results.commit()
    return dict(run_id=new_id,status='completed',recovery=recovery,final={k:v for k,v in final.items() if k!='rows'})


@app.local_entrypoint()
def mini_cohort_restore_sol(prefix: str):
    print(json.dumps(mini_restore_sol_submission.remote(prefix)),flush=True)


@app.function(cpu=1,memory=2048,timeout=14400,retries=0,volumes=MOUNTS)
def mini_complete_restored_cohort(prefix: str):
    """Finalize immutable recovered source, then run the complete common-map benchmark."""
    from polytopia_bot.skirmish.common import sha,write_json
    results.reload()
    cohort=Path('/results/mini-cohorts')/prefix
    old_id=prefix+'-sol61-oneshot-20m-r1';new_id=old_id+'-restored'
    root=Path('/results/mini-runs')/new_id
    if not (root/'report.json').exists():
        recovery=json.loads((root/'recovery.json').read_text())
        source=(root/'main.rs').read_bytes();wasm=(root/'bot.wasm').read_bytes()
        assert sha(source)==recovery['passes'][0]['sha256']==recovery['passes'][1]['sha256']
        receipt=json.loads((root/'stage-0/submission.json').read_text())
        assert receipt['returncode']==0 and receipt['wasm_sha256']==sha(wasm)
        protocol=json.loads((root/'protocol.json').read_text())
        stats=json.loads((root/'stage-0.trajectory.json').read_text())['info']['model_stats']
        if (root/'final').exists() and not (root/'interrupted-final-diagnostic').exists():
            (root/'final').rename(root/'interrupted-final-diagnostic')
        recovery['isolated_diagnostic']='Interrupted after five games; excluded. Complete shared-map tournament supplies six expert games, as for every entrant.'
        report=dict(run_id=new_id,model=protocol['model'],track=protocol['track'],status='completed',smoke=False,
            stages=[receipt],api_cost_usd=stats['instance_cost'],model_calls=stats['api_calls'],
            final_source_sha256=sha(source),final_wasm_sha256=sha(wasm),protocol_sha256=sha((root/'protocol.json').read_bytes()),
            trajectory_sha256={'stage-0.trajectory.json':sha((root/'stage-0.trajectory.json').read_bytes())},
            source_recovery=recovery,isolated_diagnostic_status='interrupted_and_excluded')
        write_json(root/'report.json',report);write_json(root/'status.json',report);write_json(root/'recovery.json',recovery)
        path=cohort/'replacements.json';replacements=json.loads(path.read_text())
        replacements[prefix+'-sol61-oneshot-20m']['run_id']=new_id
        replacements[old_id]=dict(run_id=new_id,reason='Undisclosed source cap: deterministic source reconstruction, no extra model coding.',
            recovery=recovery,fresh_authoring_attempt=False)
        write_json(path,replacements);results.commit()
    return mini_cohort_finish.remote(prefix)


@app.local_entrypoint()
def mini_cohort_complete_detached(prefix: str):
    call=mini_complete_restored_cohort.spawn(prefix)
    print(json.dumps(dict(prefix=prefix,completion_call_id=call.object_id)),flush=True)
    # Return immediately: status polling must never own the lifetime of the run.


@app.function(cpu=2,memory=4096,timeout=300,retries=0,max_containers=32,volumes=MOUNTS)
def mini_audit_game(payload: dict):
    from polytopia_bot.skirmish.mini import MAP_TYPES,audit_trace
    row=payload['row'];index=MAP_TYPES.index(row['map_type'])*2+row['seat']-1
    path=Path('/results/mini-tournaments')/payload['tournament_id']/(row['a']+'__'+row['b'])/f'{index:02d}.jsonl'
    return audit_trace(row,path)


@app.function(cpu=1,memory=2048,timeout=3600,retries=0,volumes=MOUNTS)
def mini_complete_cohort_audits(prefix: str):
    from collections import defaultdict
    from polytopia_bot.skirmish.common import identifier,write_json
    root=Path('/results/mini-cohorts')/identifier(prefix);results.reload()
    state=json.loads((root/'status.json').read_text())
    if set(state.get('tournaments',{}))!={'oneshot-20m','feedback-3r'}:
        raise RuntimeError('both complete tournaments are required')
    pending=[];counts=defaultdict(lambda:dict(passed=True,games=0,actions=0))
    for track in state['tournaments']:
        tournament_id=f'{prefix}-{track}';folder=Path('/results/mini-tournaments')/tournament_id
        rows=json.loads((folder/'games.json').read_text())
        audit=folder/'audit.json'
        if audit.exists():
            result=json.loads(audit.read_text())
            assert result['passed'] and result['games']==len(rows)
            counts[track]=result
        else:
            pending.extend(dict(track=track,tournament_id=tournament_id,row=row) for row in rows)
    if pending:
        # Reading immutable game logs is a pure operation; preemption can safely replay it.
        for job,actions in zip(pending,mini_audit_game.map(pending)):
            counts[job['track']]['games']+=1;counts[job['track']]['actions']+=actions
    for track,audit in counts.items():
        folder=Path('/results/mini-tournaments')/f'{prefix}-{track}'
        assert audit['games']==len(json.loads((folder/'games.json').read_text()))
        write_json(folder/'audit.json',audit)
        state['tournaments'][track]['audit']=audit
    assert len(state.get('audits',{}))==20
    assert all(a['passed'] for group in state['audits'].values() for a in group.values())
    state['status']='completed';state['errors']=[]
    write_json(root/'report.json',state);write_json(root/'status.json',state);results.commit()
    return dict(status='completed',audits=dict(counts))


@app.local_entrypoint()
def mini_cohort_finish_audits(prefix: str):
    call=mini_complete_cohort_audits.spawn(prefix)
    print(json.dumps(dict(prefix=prefix,audit_completion_call_id=call.object_id)),flush=True)
