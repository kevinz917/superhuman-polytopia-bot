"""Complete local-v6 pipeline; subscription inference on host, games in offline VM."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter,defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
import subprocess
import time

from . import local,v6
from .common import canonical,sha,write_json,identifier
from .docker_sandbox import docker
from .inference import endpoint


def save_progress(root,**state):write_json(root/'progress.json',dict(updated_unix=time.time(),**state))


def run_jobs(root,refs,jobs,image,slots,count,*,name='evaluation',batch_size=12):
    rows=[];batches=[jobs[i:i+batch_size] for i in range(0,len(jobs),batch_size)]
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures=[pool.submit(local.referee,root,refs,batch,image,slots,count,
            seconds=max(j.get('evaluation',{}).get('game_seconds',900) for j in batch),
            evaluation=batch[0].get('evaluation')) for batch in batches]
        for future in as_completed(futures):
            rows.extend(future.result());save_progress(root,status=name,completed=len(rows),total=len(jobs))
    return sorted(rows,key=lambda r:r['job_index'])


def diagnostic_jobs(refs,maps,cfg,*,phase,grid=None):
    jobs=[]
    pairs=list(combinations(refs,2))
    # Freeze all pairings and maps before the first game; all cap settings share them.
    for rounds in grid or [cfg['max_rounds']]:
        evaluation=dict(cfg,max_rounds=rounds)
        for a,b in pairs:
            for m in maps:
                for seat in (1,2):
                    candidate=dict(a,model_label=a['label'],replicate=1,stage=0,track=phase,submission_quality='frozen_reference')
                    j=dict(index=len(jobs),candidate=candidate,opponent=b,**m,seat=seat,phase=phase,evaluation=evaluation)
                    j['job_sha256']=sha(canonical(j).encode());j['output_relative']=f'{phase}/games/{j["index"]:06d}';jobs.append(j)
    return jobs


def cap_report(rows):
    by_horizon=defaultdict(list);by_family=defaultdict(list);blocks=defaultdict(list)
    for r in rows:
        h=r['evaluation']['max_rounds'];by_horizon[h].append(r);by_family[h,r['map_type']].append(r)
        blocks[r['a'],r['b'],r.get('track'),r.get('replicate'),r['seed'],r['seat']].append(r)
    def stats(rs):return dict(games=len(rs),statuses=dict(Counter(r['status'] for r in rs)),
        reasons=dict(Counter(r.get('cap_kind',r['reason']) for r in rs)),
        cap_rate=sum(r['status']=='draw' for r in rs)/len(rs),seconds=sum(r['seconds'] for r in rs),
        mean_actions=sum(r['actions'] for r in rs)/len(rs))
    rankings={}
    for h,rs in by_horizon.items():
        scores=defaultdict(list)
        for r in rs:
            if r['result']=='unresolved':continue
            p={'win':1,'draw':.5,'loss':0}[r['result']];scores[r['a']].append(p);scores[r['b']].append(1-p)
        rankings[h]=sorted([(k,sum(v)/len(v)) for k,v in scores.items()],key=lambda x:-x[1])
    return dict(by_horizon={str(h):stats(rs) for h,rs in by_horizon.items()},
        by_family={f'{h}|{f}':stats(rs) for (h,f),rs in by_family.items()},reference_rankings=rankings,
        changed_outcomes=sum(len({r['result'] for r in rs})>1 for rs in blocks.values()),
        opponents=sorted({r['b'] for r in rows}),
        caveat='diagnostic on this declared opponent/map sample; not proof of horizon independence on all strategies')


def render(report):
    out=['# Polytopia local v6 pilot' if report['pilot'] else '# Polytopia local v6', '',
        f"Status: {report['status']}. Local VM/Docker; CLI proxy only. No paid fallback.",
        '','All authored builds count. Points: win 1, cap draw 0.5, loss/guest forfeit 0. Infrastructure timeouts are unresolved.',
        '',report['uncertainty']['warning'], '']
    for track,ranking in report['ranking'].items():
        out += ['## '+track,'','| Model | Points /100 | 95% interval | Attempts |','|---|---:|---|---|']
        for r in ranking:out.append(f"| {r['model_label']} | {r['score']} | {r['ci95'] if r['ci95'] is not None else 'not estimable'} | {r['attempt_scores']} |")
        out.append('')
    out+=['## Paired-map contrasts','','Builds are resampled independently across models/tracks; layout draws are shared. Intervals are exploratory and unadjusted for multiple comparisons.','']
    for c in report['contrasts']:out.append(f"- {c.get('track',c.get('model_label'))}: {c['a']} − {c['b']}: {c['delta']:.2f}, 95% {c['ci95'] if c['ci95'] is not None else 'not estimable'}")
    out+=['','## Runtime diagnostics','', '```json',json.dumps(report['diagnostics'],indent=2),'```',
        '','## Scope','',report['engine_scope'],
        '','Full opponent breakdown, budget curves, peer matrix, cap sensitivity, transfer and timing are in report.json.']
    return '\n'.join(out)+'\n'


def cohort(prefix,manifest,directory,refs_dir,*,concurrency=8,cpu_slots=4,evaluate_only=False):
    v6.validate_manifest(manifest);identifier(prefix)
    root=Path(directory).resolve()/prefix;refs_dir=Path(refs_dir).resolve()
    coder,evaluator=local.image_id('polytopia-skirmish-coder:local-v6'),local.image_id('polytopia-skirmish-referee:local-v6')
    info=json.loads(subprocess.run(docker('info','--format','{{json .}}'),check=True,capture_output=True,text=True).stdout)
    if cpu_slots*2>info['NCPU'] or cpu_slots*4*1024**3>info['MemTotal']:raise ValueError('VM worker allocation exceeds resources')
    refs=local.checked_references(manifest,refs_dir);cells=v6.cells(prefix,manifest);sources=local.sources()
    allocation=dict(schema=v6.SCHEMA,manifest=manifest,cells=cells,source_files=sources,coder_image=coder,referee_image=evaluator,
        cpu_slots=cpu_slots,concurrency=concurrency,references=refs)
    slots=Path.home()/'.cache/polytopia-skirmish/cpu-slots'
    if evaluate_only:
        if json.loads((root/'allocation.json').read_text())!=allocation:raise ValueError('frozen local allocation changed')
        if (root/'report.json').exists():raise ValueError('completed cohort cannot be overwritten')
        runs={c['run_id']:json.loads((root/'runs'/c['run_id']/'report.json').read_text()) for c in cells}
    else:
        root.mkdir(parents=True,exist_ok=False);write_json(root/'allocation.json',allocation)
        from scripts.check_skirmish_proxy import catalog,mini_check
        available=catalog();write_json(root/'catalog.json',available)
        checks=[]
        # Probe all actual selected models, not substitutes, before their coding clocks.
        for model in manifest['models']:
            check=mini_check(model['model']);checks.append(dict(label=model['label'],**check));write_json(root/'probes.json',checks)
        runs={};save_progress(root,status='authoring',completed=0,total=len(cells))
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures={pool.submit(local.trial,root,c,refs_dir,coder,evaluator,slots,cpu_slots,sources):c for c in cells}
            for future in as_completed(futures):
                c=futures[future]
                try:runs[c['run_id']]=future.result()
                except Exception as error:runs[c['run_id']]=dict(status='failed_harness',error=str(error))
                write_json(root/'runs.json',runs);save_progress(root,status='authoring',completed=len(runs),total=len(cells))
    if any(r['status'] not in ('completed','failed_submission') for r in runs.values()):
        save_progress(root,status='needs_attention',errors={k:v.get('error') for k,v in runs.items() if v['status']=='failed_harness'})
        raise RuntimeError('authoring infrastructure failure; no hidden maps allocated')
    if not (root/'evaluation-plan.json').exists():
        save_progress(root,status='source_rebuild')
        local.rebuild_all(root,cells,runs,coder,slots,cpu_slots)
        if sources!=local.sources():raise ValueError('source drift before map allocation')
        maps=v6.make_maps(manifest['maps_per_family']);plan=v6.make_plan(cells,runs,maps,refs,manifest)
        write_json(root/'evaluation-plan.json',plan)
    plan=json.loads((root/'evaluation-plan.json').read_text())
    jobs=list(local.worker_payloads(root,plan['jobs']))
    rows=run_jobs(root,refs_dir,jobs,evaluator,slots,cpu_slots)
    write_json(root/'games.json',rows);report=v6.summarize(plan,cells,rows,manifest)
    # Separate diagnostic suites never alter primary scores or return to authors.
    if not (root/'diagnostic-plan.json').exists():
        calibration=v6.make_maps(manifest.get('calibration_maps_per_family',1),partition='calibration')
        # Include every panel style, sampled through a frozen ring to keep the pilot bounded.
        calibration_jobs=[]
        for i in range(len(refs)):
            calibration_jobs+=diagnostic_jobs([refs[i],refs[(i+1)%len(refs)]],calibration,manifest['evaluation'],phase='calibration',grid=[100,150,200,300])
        for i,j in enumerate(calibration_jobs):
            j['index']=i;j.pop('job_sha256',None);j.pop('output_relative',None);j['job_sha256']=sha(canonical(j).encode());j['output_relative']=f'calibration/games/{i:06d}'
        transfer=[]
        transfer_maps=v6.make_maps(manifest.get('transfer_maps_per_family',1),v6.TRANSFER_FAMILIES,partition='transfer')
        for c in cells:
            run=runs[c['run_id']];s=run['stages'][-1]
            if s['returncode']:continue
            a=dict(kind='submission',run_id=c['run_id'],stage=s['stage'],label=c['model_label'],model_label=c['model_label'],track=c['track'],replicate=c['replicate'],wasm_sha256=s['wasm_sha256'],submission_quality=s['submission_quality'])
            for size in (11,14,18):
                for m in transfer_maps:
                    for seat in (1,2):
                        ref=next(r for r in refs if r['kind']=='scripted')
                        j=dict(index=len(transfer),candidate=a,opponent=ref,**m,seat=seat,phase='transfer',evaluation=dict(manifest['evaluation'],size=size))
                        j['job_sha256']=sha(canonical(j).encode());j['output_relative']=f'transfer/games/{j["index"]:06d}';transfer.append(j)
        # Engine-valid fixtures are generated once inside the isolated referee image.
        subprocess.run(docker('run','--rm','--network','none','--cpus','2','--memory','4g','--mount',f'type=bind,src={root},dst=/results',evaluator,'python','-m','polytopia_bot.skirmish.v6_tactics'),check=True,capture_output=True,text=True)
        tactical=[];sensitivity=[]
        fixtures=json.loads((root/'tactical-fixtures.json').read_text())
        for c in cells:
            s=runs[c['run_id']]['stages'][-1]
            if s['returncode']:continue
            a=dict(kind='submission',run_id=c['run_id'],stage=s['stage'],label=c['model_label'],model_label=c['model_label'],track=c['track'],replicate=c['replicate'],wasm_sha256=s['wasm_sha256'],submission_quality=s['submission_quality'])
            opponent=next(r for r in refs if r['kind']=='scripted')
            for f in fixtures:
                j=dict(index=len(tactical),candidate=a,opponent=opponent,seed=750000001,map_type='drylands',seat=1,phase='tactical',initial_snapshot=f['snapshot'],expected_command=f['predicate'],tactical_name=f['name'],evaluation=dict(manifest['evaluation'],max_actions=1))
                j['job_sha256']=sha(canonical(j).encode());j['output_relative']=f'tactical/games/{j["index"]:06d}';tactical.append(j)
            sensitivity_panel=[r for r in refs if r['label'] in ('scripted-siege','hero-naval-50','hero-1v1')]
            for horizon in [100,150,200,300]:
                for opponent in sensitivity_panel:
                    for m in calibration:
                        for seat in (1,2):
                            j=dict(index=len(sensitivity),candidate=a,opponent=opponent,**m,seat=seat,phase='generated_cap_sensitivity',evaluation=dict(manifest['evaluation'],max_rounds=horizon))
                            j['job_sha256']=sha(canonical(j).encode());j['output_relative']=f'sensitivity/games/{j["index"]:06d}';sensitivity.append(j)
        write_json(root/'diagnostic-plan.json',dict(calibration=calibration_jobs,transfer=transfer,tactical=tactical,sensitivity=sensitivity))
    diag=json.loads((root/'diagnostic-plan.json').read_text())
    calibration_rows=run_jobs(root,refs_dir,diag['calibration'],evaluator,slots,cpu_slots,name='cap_calibration')
    write_json(root/'cap-calibration-games.json',calibration_rows);report['cap_calibration']=cap_report(calibration_rows)
    transfer_rows=run_jobs(root,refs_dir,diag['transfer'],evaluator,slots,cpu_slots,name='transfer')
    write_json(root/'transfer-games.json',transfer_rows)
    tactical_rows=run_jobs(root,refs_dir,diag['tactical'],evaluator,slots,cpu_slots,name='tactical')
    write_json(root/'tactical-games.json',tactical_rows);report['tactical']=[dict(model_label=r['model_label'],track=r['track'],replicate=r['replicate'],fixture=r['tactical_name'],passed=r['tactical_pass']) for r in tactical_rows]
    sensitivity_rows=run_jobs(root,refs_dir,diag['sensitivity'],evaluator,slots,cpu_slots,name='generated_cap_sensitivity')
    write_json(root/'generated-cap-games.json',sensitivity_rows);report['generated_cap_sensitivity']=cap_report(sensitivity_rows)
    from .cohort import score_rows
    grouped=defaultdict(list)
    for r in transfer_rows:grouped[r['model_label'],r['track'],r['evaluation']['size'],r['map_type']].append(r)
    report['transfer']={'|'.join(map(str,k)):score_rows(v) for k,v in grouped.items()}
    # Descriptive budget curves for staged tracks reuse their frozen snapshots.
    for stage in report['stages']:
        if stage['track']!='independent':report['budget_curves'][f'{stage["model_label"]}|{stage["track"]}|r{stage["replicate"]}|{(stage["stage"]+1)/4}']=stage
    report['timing']={name:dict(stages=[dict(stage=s['stage'],**s['timing'],usage=s.get('usage')) for s in run['stages']],
        inference=json.loads((root/'runs'/name/'protocol.json').read_text())['inference']) for name,run in runs.items()}
    report.update(execution_backend='local-vm-docker',evaluation_revision=manifest.get('evaluation_revision','local-v6-initial'),source_files=sources,allocation_sha256=sha(canonical(allocation).encode()),
        cap_choice='200-round candidate, preregistered; calibration and generated-bot sensitivity are diagnostics, not retrospective rescoring')
    write_json(root/'report.json',report);(root/'report.md').write_text(render(report));save_progress(root,status=report['status'],games=len(rows))
    return report
