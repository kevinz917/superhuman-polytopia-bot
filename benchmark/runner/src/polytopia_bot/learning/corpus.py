"""Qualified teacher episodes and reproducible stratified semantic samples."""
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from .interface import canonical, digest
from .data import RECORD_SCHEMA, example


def _collect_job(directory, job):
    from .adapter import GPUEnvironment
    from .teacher import ScriptedTeacher, WeakBaseline
    root=Path(directory)
    teacher=ScriptedTeacher()
    opponent=WeakBaseline() if job['opponent']=='weak' else ScriptedTeacher(job['opponent'])
    env=GPUEnvironment(14)
    obs=env.reset(seed=job['seed'],settings={})
    (root/'private'/f"{job['id']}.snapshot").write_bytes(env.snapshot())
    count=0
    try:
        with gzip.open(root/f"{job['id']}.jsonl.gz",'xt',compresslevel=1) as stream:
            for decision in range(10000):
                is_teacher=obs.player==job['seat']
                selected=(teacher if is_teacher else opponent).select(obs)
                record=None
                if is_teacher:
                    count+=1
                    record=dict(schema=RECORD_SCHEMA,source='teacher',split='train',episode_id=job['id'],
                        map_hash=job['board_hash'],decision_index=decision,observation=asdict(obs),
                        teacher_hash=teacher.teacher_hash,selected_command=dict(selected.action.command),teacher_scores=selected.scores)
                transition=env.step(action_id=selected.action.action_id,expected_state_version=obs.state_version)
                stream.write(canonical(dict(record=record,observation=None if record else asdict(obs),
                    selection=asdict(selected),after=[asdict(env.observe(p)) for p in (1,2)],
                    outcome=dict(terminated=transition.terminated,truncated=transition.truncated,
                        rewards=transition.rewards,reason=transition.reason)))+'\n')
                if transition.terminated or transition.truncated:break
                obs=env.observe(transition.next_player)
        result=dict(**job,decisions=decision+1,teacher_decisions=count,teacher_hash=teacher.teacher_hash,
            opponent_hash=opponent.teacher_hash,terminated=transition.terminated,
            truncated=transition.truncated or not transition.terminated,reward=transition.rewards[job['seat']],
            reason=transition.reason or 'research_action_limit')
        (root/f"{job['id']}.outcome.json").write_text(json.dumps(result,indent=2)+'\n')
        print(canonical(result),flush=True)
        return result
    finally:
        env.close()


def collect(output, manifest, qualification, *, map_start=0, map_count=30, workers=4):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import hashlib,multiprocessing
    from .research import source_receipt
    root=Path(output)
    qualified=Path(qualification)
    report=json.loads(qualified.read_text())
    if report.get('teacher_qualified') is not True or qualified.parent.joinpath('invalidation.json').exists():
        raise ValueError('teacher qualification required before corpus collection')
    qualification_protocol=json.loads(qualified.parent.joinpath('protocol.json').read_text())
    sources=qualification_protocol['source_sha256']
    repo=Path(__file__).resolve().parents[3]
    if digest({k:v for k,v in manifest.items() if k!='manifest_hash'})!=manifest['manifest_hash'] or qualification_protocol['map_manifest_hash']!=manifest['manifest_hash']:
        raise ValueError('split manifest differs from qualification')
    from polytopia_bot.gpu.snapshot import implementation_digest
    if implementation_digest()!=manifest['simulator_sha256']:
        raise ValueError('simulator changed since split reservation')
    for name in ('adapter','interface','encoding','memory','teacher'):
        relative=f'src/polytopia_bot/learning/{name}.py'
        if hashlib.sha256((repo/relative).read_bytes()).hexdigest()!=sources[relative]:
            raise ValueError('qualified teacher/public interface changed')
    boards=manifest['maps']['train'][map_start:map_start+map_count]
    if len(boards)!=map_count or map_count<1 or not 1<=workers<=8:
        raise ValueError('invalid corpus allocation')
    jobs=[dict(id=f'train-{m["seed"]}-{seat}',seed=m['seed'],board_hash=m['board_hash'],seat=seat,
               opponent=('weak','economy','pressure')[(map_start+i)%3])
          for i,m in enumerate(boards) for seat in (1,2)]
    root.mkdir(parents=True,exist_ok=False);(root/'private').mkdir()
    protocol=dict(schema='raw-teacher-corpus-v1',map_manifest_hash=manifest['manifest_hash'],jobs=jobs,
        qualification=str(qualified),qualification_sha256=hashlib.sha256(qualified.read_bytes()).hexdigest(),
        sources=source_receipt(root),note='All episodes and outcomes retained; raw semantic observations must be rendered under the frozen prompt before SFT.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    games=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending=[pool.submit(_collect_job,str(root),job) for job in jobs]
        for future in as_completed(pending):
            games.append(future.result())
            (root/'games.json').write_text(json.dumps(games,indent=2)+'\n')
    receipt=dict(complete=len(games)==len(jobs),games=len(games),teacher_decisions=sum(g['teacher_decisions'] for g in games),
        native_completions=sum(g['terminated'] for g in games),truncations=sum(g['truncated'] for g in games),
        wins=sum(g['reward']>0 for g in games))
    (root/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def phase(record):
    v=record['observation']['visible_state'];kind=record['selected_command']['kind']
    if v['turn']<=5:return 'opening'
    if v['turn']>=25:return 'late'
    if kind==2:return 'combat'
    if kind in (5,6,7,9):return 'economy'
    if kind in (1,8):return 'exploration'
    return 'other'


def sample(directories, output, *, count=5000, end_turn_fraction=.20):
    """Select by semantic input hash, balanced across phase/action-family strata.

    The result is still unrendered; prompt freeze and token audits are separate.
    """
    if count<1 or not 0<=end_turn_fraction<=1:raise ValueError('invalid sample allocation')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(list);seen={};duplicates=0;files={}
    import hashlib
    for directory in map(Path,directories):
        if directory.joinpath('invalidation.json').exists() or not json.loads((directory/'report.json').read_text())['complete']:
            raise ValueError('completed, valid source corpus required')
        for path in sorted(directory.glob('*.jsonl.gz')):
            files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            for index,line in enumerate(gzip.open(path,'rt')):
                row=json.loads(line)['record']
                if row is None:continue
                if row['source']!='teacher':raise ValueError('fixtures cannot enter a real semantic sample')
                _,_,key=example(row);target=canonical(row['selected_command'])
                if key in seen:
                    if seen[key]!=target:raise ValueError('conflicting labels for identical public input')
                    duplicates+=1;continue
                seen[key]=target
                family=row['selected_command']['kind'];stage=phase(row)
                groups[(stage,family)].append(dict(file=str(path),line=index,input_hash=key,
                    episode_id=row['episode_id'],map_hash=row['map_hash'],decision_index=row['decision_index'],phase=stage,family=family))
    for group in groups.values():group.sort(key=lambda r:r['input_hash'],reverse=True)
    selected=[];ends=0
    while len(selected)<count:
        changed=False
        for group in sorted(groups):
            if not groups[group] or (group[1]==10 and ends>=int(count*end_turn_fraction)):continue
            selected.append(groups[group].pop());ends+=group[1]==10;changed=True
            if len(selected)==count:break
        if not changed:raise ValueError(f'Only {len(selected)} unique examples available under the predeclared end-turn cap')
    chosen={(r['file'],r['line']) for r in selected};written=0
    with (root/'records.jsonl').open('x') as stream:
        for path in files:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=files[path]:raise ValueError('source changed during selection')
            for index,line in enumerate(gzip.open(path,'rt')):
                if (path,index) in chosen:
                    stream.write(canonical(json.loads(line)['record'])+'\n');written+=1
    assert written==count
    report=dict(schema='semantic-sample-v1',count=count,source_files=files,unique_available=len(seen),duplicates=duplicates,
        end_turn_fraction_cap=end_turn_fraction,maps=len({r['map_hash'] for r in selected}),
        episodes=len({r['episode_id'] for r in selected}),phases=dict(Counter(r['phase'] for r in selected)),
        families=dict(Counter(str(r['family']) for r in selected)),
        selection='Round-robin phase/action-family strata; lowest semantic input hashes first; no outcome filtering',
        selected=selected,records_sha256=hashlib.sha256((root/'records.jsonl').read_bytes()).hexdigest())
    (root/'sampling.json').write_text(json.dumps(report,indent=2)+'\n')
    return {k:v for k,v in report.items() if k not in ('selected','source_files')}
