"""Versioned local benchmark plans and uncertainty; v5 protocols remain untouched."""
from collections import Counter, defaultdict
from itertools import combinations
import math
import random

from .common import canonical, identifier, sha
from .cohort import score_rows

SCHEMA = 'polytopia-local-v6'
TRACKS = ('independent', 'segmented-control', 'feedback')
CORE_FAMILIES = ('drylands', 'lakes', 'continents')
TRANSFER_FAMILIES = ('pangea', 'archipelago', 'highlands')
DEFAULT_EVALUATION = dict(max_rounds=200, max_actions=40000, fuel=4_000_000_000_000,
    move_seconds=2., game_seconds=1800, exact_rounds=True, size=14)


def validate_manifest(m):
    if m.get('schema') != SCHEMA or m.get('tracks') != list(TRACKS):
        raise ValueError('v6 requires independent, segmented-control and feedback tracks')
    pilot = m.get('pilot', False)
    for key, low, high in [('replicates',1 if pilot else 5,20),('maps_per_family',1 if pilot else 10,128),
                            ('peer_maps_per_family',1 if pilot else 5,128),('coding_seconds',120 if pilot else 2400,7200)]:
        if type(m.get(key)) is not int or not low <= m[key] <= high:
            raise ValueError('invalid v6 '+key)
    if m['coding_seconds'] % 4 or m['peer_maps_per_family'] > m['maps_per_family']:
        raise ValueError('four equal coding stages and bounded peer map count required')
    if len({e['label'] for e in m['models']}) != len(m['models']) or not m['models']:
        raise ValueError('distinct nonempty model roster required')
    for e in m['models']:
        identifier(e['label'])
        if not e['model'].startswith(('openai/','anthropic/')):
            raise ValueError('this local pilot requires explicit subscription models')
        if e.get('parameters',{}).get('max_tokens') != 32768:
            raise ValueError('declare common per-call ceiling')
    refs = m['references']
    if len(refs) < 5 or len({r['label'] for r in refs}) != len(refs):
        raise ValueError('v6 requires at least five distinct frozen references')
    if sum(r['label']=='hero-1v1' for r in refs) != 1:
        raise ValueError('fixed hero anchor required')
    if set(e['label'] for e in m['models']) & set(r['label'] for r in refs):
        raise ValueError('model/reference labels collide')
    validate_evaluation(m['evaluation'])
    if m['bootstrap_resamples'] < 2000:
        raise ValueError('at least 2000 bootstrap resamples required')
    return m


def validate_evaluation(cfg):
    expected = set(DEFAULT_EVALUATION)
    if set(cfg) != expected or cfg['exact_rounds'] is not True:
        raise ValueError('complete exact-round evaluation configuration required')
    if cfg['size'] not in (11,14,18) or not 1 <= cfg['max_rounds'] <= 1000:
        raise ValueError('invalid board/horizon')
    if not 100 <= cfg['max_actions'] <= 200000 or not 1000 <= cfg['fuel'] <= 10**14:
        raise ValueError('invalid action/fuel safety allowance')
    if not 0 < cfg['move_seconds'] <= 30 or not 10 <= cfg['game_seconds'] <= 7200:
        raise ValueError('invalid timeout')
    return cfg


def track_config(manifest, track):
    if track not in TRACKS: raise ValueError('unknown v6 track')
    stages = 1 if track == 'independent' else 4
    return dict(stage_seconds=manifest['coding_seconds']//stages, revision_rounds=stages-1,
                supplied_feedback=track=='feedback')


def final_stage(track):
    return 0 if track == 'independent' else 3


def cells(prefix, manifest):
    validate_manifest(manifest)
    out=[]
    for rep in range(1,manifest['replicates']+1):
        for e in manifest['models']:
            for track in TRACKS:
                run_id=identifier(f'{prefix}-{e["label"]}-r{rep}-{track}')
                cfg=dict(provider='cliproxyapi',model=e['model'],parameters=e['parameters'],track=track,
                    v6_track=track_config(manifest,track),evaluation=manifest['evaluation'],
                    budget_snapshots=manifest.get('budget_snapshots',[.25,.5,.75,1.]))
                out.append(dict(run_id=run_id,model_label=e['label'],model=e['model'],replicate=rep,track=track,config=cfg))
    # Randomize dispatch order before authoring; persist it, never choose by outcomes.
    random.Random(manifest.get('dispatch_seed',61031)).shuffle(out)
    return out


def make_maps(count, families=CORE_FAMILIES, *, partition='heldout', sample=None):
    ranges={'heldout':(1_000_000_000,1_500_000_000),'transfer':(1_500_000_000,1_800_000_000),
            'calibration':(700_000_000,800_000_000),'feedback':(500000,599999)}
    lo,hi=ranges[partition]
    seeds=(sample or random.SystemRandom().sample)(range(lo,hi),count*len(families))
    return [dict(map_type=f,seed=s) for i,f in enumerate(families) for s in seeds[i*count:(i+1)*count]]


def make_plan(cells_,runs,maps,refs,manifest):
    validate_manifest(manifest)
    if Counter(x['map_type'] for x in maps) != Counter({f:manifest['maps_per_family'] for f in CORE_FAMILIES}):
        raise ValueError('unbalanced hidden core suite')
    if len({m['seed'] for m in maps}) != len(maps) or any(not 10**9<=m['seed']<1_500_000_000 for m in maps):
        raise ValueError('invalid hidden map partition')
    primary=[];finals=[];invalid=[];snapshots=[]
    for cell in cells_:
        run=runs[cell['run_id']]
        if run['status'] not in ('completed','failed_submission'): raise ValueError('unresolved authoring')
        count=final_stage(cell['track'])+1
        if {s['stage'] for s in run['stages']} != set(range(count)): raise ValueError('missing frozen stage')
        for s in run['stages']:
            a=dict(kind='submission',run_id=cell['run_id'],stage=s['stage'],label=f'{cell["model_label"]}-r{cell["replicate"]}-s{s["stage"]}',
                model_label=cell['model_label'],replicate=cell['replicate'],track=cell['track'],
                source_sha256=s['source_sha256'],wasm_sha256=s.get('wasm_sha256'),submission_quality=s['submission_quality'])
            if s['stage']==count-1 and s['returncode']==0: finals.append(a)
            if s['returncode']:invalid.append(a)
            for ref in refs:
                for m in maps:
                    for seat in (1,2):primary.append(dict(candidate=a,opponent=ref,**m,seat=seat,phase='heldout',invalid_submission=bool(s['returncode'])))
        for snap in run.get('budget_snapshots',[]):
            a=dict(kind='budget_snapshot',run_id=cell['run_id'],stage=snap['stage'],snapshot=snap['snapshot'],
                label=f'{cell["model_label"]}-r{cell["replicate"]}-b{snap["snapshot"]}',model_label=cell['model_label'],
                replicate=cell['replicate'],track=cell['track'],wasm_sha256=snap.get('wasm_sha256'),
                source_sha256=snap['source_sha256'],submission_quality=snap['submission_quality'],budget_fraction=snap['fraction'])
            for ref in refs:
                for m in maps:
                    for seat in (1,2):snapshots.append(dict(candidate=a,opponent=ref,**m,seat=seat,phase='budget',invalid_submission=bool(snap['returncode'])))
    peer=[]
    peer_maps=[m for f in CORE_FAMILIES for m in [x for x in maps if x['map_type']==f][:manifest['peer_maps_per_family']]]
    for a,b in combinations(finals,2):
        if a['track']!=b['track'] or a['replicate']!=b['replicate'] or a['model_label']==b['model_label']:continue
        for m in peer_maps:
            for seat in (1,2):peer.append(dict(candidate=a,opponent=b,**m,seat=seat,phase='peer',invalid_submission=False))
    jobs=[]
    for i,j in enumerate(primary+peer+snapshots):
        job=dict(index=i,**j,evaluation=manifest['evaluation']);job['job_sha256']=sha(canonical(job).encode());jobs.append(job)
    return dict(schema=SCHEMA,pilot=manifest['pilot'],maps=maps,maps_sha256=sha(canonical(maps).encode()),references=refs,jobs=jobs,
        primary_games=len(primary),peer_games=len(peer),budget_games=len(snapshots),evaluation=manifest['evaluation'],invalid_stages=invalid)


def percentile(values,p):
    vals=sorted(values);at=(len(vals)-1)*p;lo=int(at);hi=math.ceil(at)
    return vals[lo]+(vals[hi]-vals[lo])*(at-lo)


def interval(samples):return [percentile(samples,.025),percentile(samples,.975)]


def bootstrap_scores(rows, resamples=4000, seed=0):
    """Crossed build/map bootstrap. Shared maps remain shared across model contrasts.

    Builds are independent across models/tracks, NOT paired by their arbitrary index.
    All references and both seats travel with a layout. Opponent panel is fixed.
    """
    blocks=defaultdict(list)
    for r in rows:blocks[r['model_label'],r['track'],r['replicate'],r['map_type'],r['seed']].append({'win':1.,'loss':0.,'draw':.5}[r['result']])
    keys=sorted({(r['model_label'],r['track']) for r in rows})
    reps={k:sorted({r['replicate'] for r in rows if (r['model_label'],r['track'])==k}) for k in keys}
    maps={f:sorted({r['seed'] for r in rows if r['map_type']==f}) for f in sorted({r['map_type'] for r in rows})}
    rng=random.Random(seed);samples={k:[] for k in keys}
    for _ in range(resamples):
        chosen={f:rng.choices(s,k=len(s)) for f,s in maps.items()}
        for k in keys:
            points=[p for rep in rng.choices(reps[k],k=len(reps[k])) for f,ss in chosen.items() for s in ss for p in blocks[*k,rep,f,s]]
            samples[k].append(100*sum(points)/len(points))
    return samples


def summarize(plan,cells_,rows,manifest):
    if len(rows)!=len(plan['jobs']) or {r['job_index'] for r in rows} != set(range(len(rows))):raise ValueError('incomplete or duplicated evaluation')
    for r in rows:
        if r['job_sha256']!=plan['jobs'][r['job_index']]['job_sha256']:raise ValueError('frozen job mismatch')
        if not r.get('audit',{}).get('passed'):raise ValueError('unaudited result')
    primary=[r for r in rows if r['phase']=='heldout'];grouped=defaultdict(list)
    for r in primary:grouped[r['model_label'],r['track'],r['replicate'],r['stage']].append(r)
    stages=[dict(model_label=m,track=t,replicate=p,stage=s,**score_rows(rs),
        by_map={f:score_rows([r for r in rs if r['map_type']==f]) for f in CORE_FAMILIES}) for (m,t,p,s),rs in grouped.items()]
    final=[r for r in primary if r['stage']==final_stage(r['track'])]
    complete=all(r['result']!='unresolved' for r in rows)
    estimable=manifest['replicates']>=2 and manifest['maps_per_family']>=2
    samples=bootstrap_scores(final,manifest['bootstrap_resamples']) if complete and estimable else {}
    ranking={};by_opponent={};diagnostics={}
    for t in TRACKS:
        ranking[t]=[]
        for m in manifest['models']:
            rs=[r for r in final if r['track']==t and r['model_label']==m['label']]
            score=score_rows(rs);key=(m['label'],t)
            ranking[t].append(dict(model_label=m['label'],**score,ci95=interval(samples[key]) if samples else None,
                attempt_scores=[s['score'] for s in stages if s['model_label']==m['label'] and s['track']==t and s['stage']==final_stage(t)]))
            by_opponent['|'.join(key)]={ref['label']:score_rows([r for r in rs if r['opponent_label']==ref['label']]) for ref in plan['references']}
            diagnostics['|'.join(key)]=dict(statuses=dict(Counter(r['status'] for r in rs)),reasons=dict(Counter(r['reason'] for r in rs)),
                by_map={f:dict(games=sum(r['map_type']==f for r in rs),cap_draws=sum(r['status']=='draw' and r['map_type']==f for r in rs)) for f in CORE_FAMILIES})
        ranking[t].sort(key=lambda x:x['score'] if x['score'] is not None else -1,reverse=True)
    contrasts=[]
    if complete:
        for t in TRACKS:
            for a,b in combinations([m['label'] for m in manifest['models']],2):
                vals=[x-y for x,y in zip(samples[a,t],samples[b,t])] if samples else []
                point={r['model_label']:r['score'] for r in ranking[t]}
                contrasts.append(dict(kind='model_difference',track=t,a=a,b=b,delta=point[a]-point[b],ci95=interval(vals) if vals else None))
        for m in manifest['models']:
            for a,b in [('feedback','segmented-control'),('segmented-control','independent'),('feedback','independent')]:
                vals=[x-y for x,y in zip(samples[m['label'],a],samples[m['label'],b])] if samples else []
                point_a=next(r['score'] for r in ranking[a] if r['model_label']==m['label'])
                point_b=next(r['score'] for r in ranking[b] if r['model_label']==m['label'])
                contrasts.append(dict(kind='track_difference',model_label=m['label'],a=a,b=b,delta=point_a-point_b,ci95=interval(vals) if vals else None))
    # Preserve descriptive cross-play, including cycles; Elo remains a secondary projection.
    from .tournament import fit_elo
    peer={}
    for t in TRACKS:
        pool=[]
        for r in rows:
            j=plan['jobs'][r['job_index']];a,b=j['candidate'],j['opponent']
            if r['phase'] not in ('heldout','peer') or a['track']!=t or a['stage']!=final_stage(t) or r['status']=='invalid_submission':continue
            pool.append(dict(r,a=a['model_label'],b=b.get('model_label',b['label'])))
        labels=[m['label'] for m in manifest['models']]+[r['label'] for r in plan['references']]
        pairs={}
        for r in pool:
            if r['phase']!='peer':continue
            k=r['a']+'|'+r['b'];pairs.setdefault(k,[]).append(r)
        matrix={k:score_rows(v) for k,v in pairs.items()};cycles=[]
        def points(a,b):
            if a+'|'+b in matrix:return matrix[a+'|'+b]['score']
            if b+'|'+a in matrix:return 100-matrix[b+'|'+a]['score']
        names=[m['label'] for m in manifest['models']]
        for a,b,c in combinations(names,3):
            for x,y,z in [(a,b,c),(a,c,b)]:
                if all(points(u,v) is not None and points(u,v)>50 for u,v in [(x,y),(y,z),(z,x)]):cycles.append([x,y,z,x])
        peer[t]=dict(elo=fit_elo(pool,labels),matchups=matrix,descriptive_cycles=cycles)
    budget={}
    for r in rows:
        if r['phase']=='budget':
            a=plan['jobs'][r['job_index']]['candidate'];budget.setdefault(f'{r["model_label"]}|{r["track"]}|{a["budget_fraction"]}',[]).append(r)
    return dict(schema=SCHEMA,status='completed' if complete else 'provisional',pilot=manifest['pilot'],
        games=len(rows),primary_games=plan['primary_games'],peer_games=plan['peer_games'],ranking=ranking,
        stages=stages,by_opponent=by_opponent,diagnostics=diagnostics,contrasts=contrasts,secondary_peer=peer,
        budget_curves={k:score_rows(v) for k,v in budget.items()},
        uncertainty=dict(method='crossed independent-build/shared-map bootstrap, fixed opponent panel',resamples=manifest['bootstrap_resamples'],effective_resamples=manifest['bootstrap_resamples'] if samples else 0,intervals_estimable=estimable,
            warning='intervals are not estimable: fewer than two independent builds or two layouts per family' if not estimable else 'pilot; too few builds/maps for leaderboard claims' if manifest['pilot'] else 'intervals condition on the fixed opponent panel; multiplicity unadjusted'),
        engine_scope='implementation-defined Bardur rules; replay consistency is not official-game fidelity')
