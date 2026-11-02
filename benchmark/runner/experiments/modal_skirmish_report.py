"""Read-only compact evidence export; generated bots and full trajectories stay on Modal."""
from pathlib import Path
from collections import Counter
import json
import modal

app=modal.App('polytopia-mini-report')
volume=modal.Volume.from_name('polytopia-skirmish',environment_name='staging')


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


@app.function(image=modal.Image.debian_slim(python_version='3.12'),cpu=1,memory=1024,timeout=60,
              volumes={'/results':volume.with_mount_options(read_only=True)})
def summary(prefix: str):
    if not prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in prefix):
        raise ValueError('invalid prefix')
    runs={}
    for root in sorted(Path('/results/mini-runs').glob(prefix+'*')):
        status=read(root/'status.json') or {}
        report=read(root/'report.json') or status
        protocol=read(root/'protocol.json') or {}
        stages=read(root/'stages.json') or []
        trajectories=[]
        for path in sorted(root.glob('*.trajectory.json')):
            data=read(path)
            if data:
                messages=data['messages']
                calls=data['info']['model_stats']['api_calls']
                responses=[m['extra']['response'] for m in messages if m.get('extra',{}).get('response')]
                responses=responses[-calls:] if calls else []
                trajectories.append(dict(name=path.name,**data['info']['model_stats'],
                    exit_status=data['info'].get('exit_status'),
                    finish_reasons=dict(Counter(r['choices'][0].get('finish_reason') for r in responses)),
                    completion_tokens=sum((r.get('usage') or {}).get('completion_tokens',0) for r in responses),
                    feedback_reads_cumulative=sum('/feedback/' in a.get('command','')
                        for m in messages for a in m.get('extra',{}).get('actions',[]))))
        evaluations={}
        for directory in [*sorted(root.glob('feedback-*')),root/'final']:
            data=read(directory/'report.json')
            if data:
                evaluations[directory.name]={k:v for k,v in data.items() if k!='rows'}
                evaluations[directory.name]['actions']=sum(r['actions'] for r in data['rows'])
                evaluations[directory.name]['by_map']={kind:dict(
                    games=len(rows:=[r for r in data['rows'] if r['map_type']==kind]),
                    wins=sum(r['result']=='win' for r in rows),losses=sum(r['result']=='loss' for r in rows),
                    unresolved=sum(r['result']=='unresolved' for r in rows),
                    forfeits=sum(r['status']=='forfeit' for r in rows))
                    for kind in ('drylands','lakes','continents')}
            elif (data:=read(directory/'progress.json')):
                evaluations[directory.name]=data
        # Neither source nor model messages are returned.
        stages=[{k:v for k,v in s.items() if k!='compile_output'} for s in stages]
        worker_errors={p.name:p.read_text()[-1500:] for p in root.glob('*.worker.log')
                       if any(word in p.read_text() for word in ('Traceback','Error','retry','Retry'))}
        runs[root.name]=dict(model=report.get('model'),track=report.get('track'),status=report.get('status'),
            stage=status.get('stage'),deadline_unix=status.get('stage_deadline_unix'),
            smoke=report.get('smoke'),stages=stages,trajectories=trajectories,evaluations=evaluations,
            api_cost_usd=report.get('api_cost_usd'),model_calls=report.get('model_calls'),
            final_source_sha256=report.get('final_source_sha256'),final_wasm_sha256=report.get('final_wasm_sha256'),
            protocol_sha256=report.get('protocol_sha256'),harness_version=protocol.get('harness_version'),
            protocol_schema=protocol.get('schema'),runtime=protocol.get('runtime'),
            stage_seconds=protocol.get('stage_seconds'),revision_rounds=protocol.get('revision_rounds'),
            rules_sha256=protocol.get('rules_sha256'),system_sha256=protocol.get('system_sha256'),
            coding_image_id=protocol.get('coding_image_id'),
            parameters=protocol.get('parameters'),isolation=protocol.get('isolation'),worker_errors=worker_errors,
            audit=read(root/'audit.json'),source_bytes=(root/'main.rs').stat().st_size if (root/'main.rs').exists() else None)
        runs[root.name]['source_recovery']=report.get('source_recovery')
        runs[root.name]['isolated_diagnostic_status']=report.get('isolated_diagnostic_status')
    ladders={}
    for root in sorted(Path('/results/mini-tournaments').glob(prefix+'*')):
        data=read(root/'report.json') or read(root/'status.json')
        if data:
            data['audit']=read(root/'audit.json')
            rows=read(root/'games.json') or []
            data['runtime_faults']=[r for r in rows if r['status']=='forfeit']
            data['unresolved_games']=[r for r in rows if r['status']=='unresolved']
            if data.get('original_id'):
                oldroot=root.parent/data['original_id']
                oldrows=read(oldroot/'games.json') or []
                key=lambda r:(r['a'],r['b'],r['map_type'],r['seed'],r['seat'])
                old={key(r):r for r in oldrows}
                exact=prefixes=0
                for row in rows:
                    previous=old[key(row)]
                    index=('drylands','lakes','continents').index(row['map_type'])*2+row['seat']-1
                    relative=Path(row['a']+'__'+row['b'])/f'{index:02d}.private.jsonl'
                    exact+=row['action_chain_sha256']==previous['action_chain_sha256']
                    prefixes+=(root/relative).read_bytes().startswith((oldroot/relative).read_bytes())
                data['calibration_comparison']=dict(games=len(rows),identical_action_chains=exact,
                                                    original_action_prefix_preserved=prefixes)
        ladders[root.name]=data
    cohort=Path('/results/mini-cohorts')/prefix
    return dict(prefix=prefix,runs=runs,ladders=ladders,
                replacements=read(cohort/'replacements.json') or {},
                cohort_status=read(cohort/'status.json'))


@app.local_entrypoint()
def report(prefix: str, output: str = ''):
    data=summary.remote(prefix)
    if output:
        Path(output).write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(dict(output=output,statuses={k:v['status'] for k,v in data['runs'].items()})))
    else:print(json.dumps(data,indent=2))
