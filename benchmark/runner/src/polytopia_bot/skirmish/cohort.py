"""Pure plans and scores for repeated, held-out v5 evaluation on Modal."""
from collections import defaultdict, Counter
import random

from .common import canonical, identifier, sha, REFERENCE
from .protocol import SCHEMA, MAP_TYPES, validate_map_suite, suite_digest

TRACKS = ('independent-40m','feedback-3r-40m')


def validate_manifest(manifest):
    if manifest.get('schema')!='polytopia-mini-cohort-v2' or manifest.get('tracks')!=list(TRACKS):
        raise ValueError('v5 requires both equal-budget 40-minute tracks')
    if type(manifest.get('replicates')) is not int or not 1<=manifest['replicates']<=10:
        raise ValueError('replicates must be 1..10')
    entries=manifest['models']
    if not entries or len({e['label'] for e in entries})!=len(entries) or len({e['model'] for e in entries})!=len(entries):
        raise ValueError('distinct models and labels required')
    for e in entries:
        identifier(e['label'])
        p=e['parameters'];provider=p['provider']
        if not e['model'] or len(provider['only'])!=1 or provider.get('allow_fallbacks') is not False or provider.get('require_parameters') is not True:
            raise ValueError('one pinned provider without fallback required')
        if p['reasoning'].get('effort')!='high' or p['max_tokens']!=32768:
            raise ValueError('same declared high reasoning and 32768-token ceiling required')
    refs=manifest['references']
    if len(refs)<3 or len({r['label'] for r in refs})!=len(refs):
        raise ValueError('at least three fixed evaluation references required')
    for ref in refs:
        identifier(ref['label'])
        if ref['kind'] not in ('expert','frozen'):raise ValueError('unknown reference kind')
        if ref['kind']=='frozen':identifier(ref['run_id'])
    if len([r for r in refs if r['kind']=='expert' and r['label']==REFERENCE])!=1 or sum(r['kind']=='expert' for r in refs)!=1:
        raise ValueError('one hero-1v1 expert anchor is required')
    if {e['label'] for e in entries}&{r['label'] for r in refs}:
        raise ValueError('model and reference labels must be distinct')
    frozen=[r['run_id'] for r in refs if r['kind']=='frozen']
    if len(set(frozen))!=len(frozen):raise ValueError('reference policies must be distinct')
    if manifest.get('maps_per_family')!=5:raise ValueError('official evaluation uses five layouts per family')
    return manifest


def authoring_cells(prefix, manifest, image_id, *, smoke=False):
    validate_manifest(manifest);identifier(prefix)
    if not smoke and manifest['replicates']!=3:raise ValueError('official cohorts require three independent builds')
    jobs=[]
    for rep in range(manifest['replicates']):
        for e in manifest['models']:
            for track in manifest['tracks']:
                run_id=f'{prefix}-{e["label"]}-r{rep+1}-{track}'
                identifier(run_id)
                config=dict(provider='openrouter',harness='mini-swe-agent',model=e['model'],
                    parameters=e['parameters'],track=track,_coding_image_id=image_id)
                if smoke:config.update(smoke=True,smoke_seconds=30)
                jobs.append(dict(run_id=run_id,model_label=e['label'],model=e['model'],replicate=rep+1,
                                 track=track,config=config))
    return jobs


def validate_run_protocol(cell, protocol, execution_source, image_id):
    from .mini import track_config
    from .mini_worker import MINI_VERSION
    from .protocol import EVALUATION
    expected=dict(schema=SCHEMA,model=cell['model'],provider='openrouter',harness='mini-swe-agent',
        harness_version=MINI_VERSION,track=cell['track'],source_files=execution_source,
        coding_image_id=image_id,parameters=cell['config']['parameters'],evaluation=EVALUATION,
        smoke=bool(cell['config'].get('smoke')),**track_config(cell['config']))
    for key,value in expected.items():
        if canonical(protocol.get(key))!=canonical(value):raise ValueError('run protocol mismatch: '+key)
    return True


def artifact(cell, stage, run_id=None):
    return dict(kind='submission',run_id=run_id or cell['run_id'],stage=stage,
                label=f'{cell["model_label"]}-r{cell["replicate"]}-s{stage}',
                model_label=cell['model_label'],replicate=cell['replicate'],track=cell['track'])


def make_evaluation_plan(cells, runs, maps, references, *, smoke=False):
    """Called AFTER all authoring is terminal; all stages/tracks share exactly these maps.

    Primary evaluates every frozen stage against the same fixed panel. Secondary
    peer games pair independent replicate N against replicate N, never select a
    best attempt, and use one shared layout per family (both seats).
    """
    schedule=validate_map_suite(maps)
    primary=[];finals=[];invalid=[]
    for cell in cells:
        run=runs[cell['run_id']]
        if run['status'] not in ('completed','failed_submission'):
            raise ValueError('all authoring must be terminal without unresolved infrastructure failures')
        count=4 if cell['track']=='feedback-3r-40m' else 1
        stages={s['stage']:s for s in run['stages']}
        if set(stages)!=set(range(count)):raise ValueError('missing frozen stage receipts')
        for stage in range(count):
            candidate=artifact(cell,stage,run.get('effective_run_id'))
            good=stages[stage]['returncode']==0
            candidate.update(source_sha256=stages[stage]['source_sha256'],
                             wasm_sha256=stages[stage].get('wasm_sha256'),
                             submission_quality=stages[stage]['submission_quality'])
            if stage==count-1 and good:finals.append(candidate)
            for ref in references:
                for position in schedule:
                    primary.append(dict(candidate=candidate,opponent=ref,**position,
                                        phase='heldout',invalid_submission=not good))
            if not good:invalid.append(candidate)
    peer=[]
    peer_maps=[next(m for m in maps if m['map_type']==kind) for kind in MAP_TYPES]
    for i,a in enumerate(finals):
        for b in finals[i+1:]:
            if a['track']!=b['track'] or a['replicate']!=b['replicate'] or a['model_label']==b['model_label']:continue
            for position in validate_map_suite(peer_maps):
                peer.append(dict(candidate=a,opponent=b,**position,phase='peer',invalid_submission=False))
    jobs=[]
    for index,job in enumerate([*primary,*peer]):
        job=dict(index=index,**job);job['job_sha256']=sha(canonical(job).encode());jobs.append(job)
    return dict(schema=SCHEMA,smoke=smoke,maps=maps,maps_sha256=suite_digest(maps),references=references,
                primary_games=len(primary),peer_games=len(peer),invalid_stages=invalid,jobs=jobs,
                cap_policy='engine_draw',peer_policy='same replicate, distinct models, one layout per family, both seats')


def game_points(row):
    if row['result']=='unresolved':return None
    return {'win':1.,'loss':0.,'draw':.5}[row['result']]


def score_rows(rows):
    if not rows:raise ValueError('cannot score empty schedule')
    values=[game_points(r) for r in rows];missing=sum(v is None for v in values)
    points=sum(v for v in values if v is not None)
    return dict(games=len(rows),wins=sum(r['result']=='win' for r in rows),
                losses=sum(r['result']=='loss' for r in rows),draws=sum(r['result']=='draw' for r in rows),
                unresolved=missing,score=None if missing else 100*points/len(rows),
                score_bounds=[100*points/len(rows),100*(points+missing)/len(rows)])


def summarize_evaluation(plan, cells, rows):
    """All attempts count. Broken frozen stages contribute scheduled zeroes, no invented Elo."""
    if len(rows)!=len(plan['jobs']) or {r['job_index'] for r in rows}!=set(range(len(plan['jobs']))):
        raise ValueError('evaluation is incomplete or duplicated')
    for row in rows:
        if row['job_sha256']!=plan['jobs'][row['job_index']]['job_sha256']:
            raise ValueError('game result belongs to a different frozen plan')
    grouped=defaultdict(list)
    for row in rows:
        if row['phase']=='heldout':grouped[row['model_label'],row['track'],row['replicate'],row['stage']].append(row)
    stages=[]
    for (model,track,rep,stage),part in sorted(grouped.items()):
        score=score_rows(part)
        stages.append(dict(model_label=model,track=track,replicate=rep,stage=stage,**score,
            submission_quality=part[0]['submission_quality'],
            by_opponent={op:score_rows([r for r in part if r['opponent_label']==op]) for op in sorted({r['opponent_label'] for r in part})},
            by_map={kind:score_rows([r for r in part if r['map_type']==kind]) for kind in MAP_TYPES}))
    ranking={}
    for track in sorted({c['track'] for c in cells}):
        final_stage=3 if track=='feedback-3r-40m' else 0
        ranking[track]=[]
        for model in sorted({c['model_label'] for c in cells}):
            selected=[s for s in stages if s['model_label']==model and s['track']==track and s['stage']==final_stage]
            expected=sum(c['model_label']==model and c['track']==track for c in cells)
            if len(selected)!=expected:raise ValueError('model attempt missing from aggregate')
            scores=[s['score'] for s in selected]
            value=None if any(s is None for s in scores) else sum(scores)/len(scores)
            ranking[track].append(dict(model_label=model,score=value,attempt_scores=scores,
                submission_quality=dict(Counter(s['submission_quality'] for s in selected)),
                score_bounds=[sum(s['score_bounds'][i] for s in selected)/len(selected) for i in (0,1)]))
            if value is not None:
                final_rows=[r for r in rows if r['phase']=='heldout' and r['model_label']==model and r['track']==track and r['stage']==final_stage]
                blocks=defaultdict(list)
                for r in final_rows:blocks[r['replicate'],r['map_type'],r['seed']].append(game_points(r))
                reps=sorted({r['replicate'] for r in final_rows})
                seeds={kind:sorted({r['seed'] for r in final_rows if r['map_type']==kind}) for kind in MAP_TYPES}
                rng=random.Random(0);samples=[]
                for _ in range(400):
                    chosen_reps=rng.choices(reps,k=len(reps))
                    chosen_maps={kind:rng.choices(seed,k=len(seed)) for kind,seed in seeds.items()}
                    points=[p for rep in chosen_reps for kind,seed in chosen_maps.items() for s in seed for p in blocks[rep,kind,s]]
                    samples.append(100*sum(points)/len(points))
                samples.sort()
                ranking[track][-1]['score_ci95']=[samples[9],samples[389]]
        ranking[track].sort(key=lambda r:(r['score'] is not None,r['score'] or 0),reverse=True)
    changes=[]
    for model,track,rep,_ in grouped:
        if track!='feedback-3r-40m' or any(c['model_label']==model and c['replicate']==rep for c in changes):continue
        attempts=sorted([s for s in stages if s['model_label']==model and s['track']==track and s['replicate']==rep],key=lambda s:s['stage'])
        values=[s['score'] for s in attempts]
        changes.append(dict(model_label=model,replicate=rep,stage_scores=values,
                            final_minus_initial=None if values[0] is None or values[-1] is None else values[-1]-values[0]))
    return dict(schema=SCHEMA,status='completed' if all(r['result']!='unresolved' for r in rows) else 'provisional',
        smoke=plan['smoke'],maps_sha256=plan['maps_sha256'],games=len(rows),
        primary_games=plan['primary_games'],peer_games=plan['peer_games'],stages=stages,ranking=ranking,revision_changes=changes,
        uncertainty='400 bootstrap resamples over independent builds and shared map blocks stratified by family; three builds remain preliminary',
        scoring='mean points (win=1, draw=0.5, loss=0), all attempts, equal maps/seats/opponents; compile failure=0; unresolved blocks point estimate')
