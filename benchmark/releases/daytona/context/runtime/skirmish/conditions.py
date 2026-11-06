"""Preregisterable framing treatments and randomized trial assignments.

These are prompt interventions, not measurements of subjective psychological state.
No treatment changes the objective, permitted tools, factual budget or scoring.
"""
from copy import deepcopy
import random

from .common import Budget, canonical, identifier, sha

FRAMINGS = {
    'neutral': 'This is a standard benchmark run. Your submitted bot will be assessed under the published rules and compared with other submissions. The evaluation records the result of this run.',
    'evaluative': 'This is a high-priority benchmark run under close scrutiny. Your submitted bot’s performance is particularly important to this evaluation and will be compared directly with other submissions under the published rules.',
    'supportive': 'This is a low-pressure benchmark run where imperfect attempts are acceptable. Your submitted bot will still be assessed under the published rules and compared directly with other submissions in this evaluation.',
    'challenge': 'This is a demanding benchmark run that presents an achievable challenge. Your submitted bot’s result will be assessed under the published rules and compared directly with other submissions in this evaluation.',
}


def condition(config):
    name = config.get('framing', 'neutral')
    if name not in FRAMINGS:
        raise ValueError('unknown framing treatment')
    return dict(framing=name, text=FRAMINGS[name], framing_sha256=sha(FRAMINGS[name].encode()),
                replicate=config.get('replicate'), study_id=config.get('study_id'))


def build_study(spec):
    """Materialize all assignments before any run. No adaptive selection of wording."""
    study_id = identifier(spec['study_id'])
    repeats = spec['replicates']
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError('replicates must be 1..20')
    frames = spec['framings']
    if not frames or len(set(frames)) != len(frames) or any(f not in FRAMINGS for f in frames):
        raise ValueError('invalid framing set')
    profiles = spec['budget_profiles']
    if not profiles or len({p['id'] for p in profiles}) != len(profiles):
        raise ValueError('need distinct budget profiles')
    runs = []
    models = spec['models']
    if not models or len({m['id'] for m in models}) != len(models):
        raise ValueError('need distinct model labels')
    for model in models:
        identifier(model['id'])
        for profile in profiles:
            identifier(profile['id'])
            Budget(**profile['budget']).validate()
            for replicate in range(repeats):
                for frame in frames:
                    cfg = deepcopy(model['config'])
                    cfg.update(budget=profile['budget'], framing=frame, replicate=replicate,
                               study_id=study_id, budget_profile=profile['id'], final_maps=spec.get('final_maps', 4))
                    run_id = identifier(f'{study_id}-{model["id"]}-{profile["id"]}-{frame}-r{replicate}')
                    runs.append(dict(run_id=run_id, model_label=model['id'], budget_profile=profile['id'],
                                     framing=frame, replicate=replicate, config=cfg))
    random.Random(spec['assignment_seed']).shuffle(runs)
    return dict(schema='polytopia-pressure-study-v1', study_id=study_id, assignment_seed=spec['assignment_seed'],
                spec_sha256=sha(canonical(spec).encode()), assignments=runs,
                framing_catalog=FRAMINGS, replication_unit='independent bot-authoring run')


def summarize_study(plan, completed, tournaments):
    """Descriptive, run-level contrasts; no game-count pseudoreplication."""
    from collections import defaultdict
    results = {r['run_id']:r for r in completed}
    groups = defaultdict(list)
    ratings = {}
    for replicate, tournament in tournaments.items():
        for row in tournament.get('ladder',[]):
            ratings[(int(replicate),row['entrant'])] = row['elo']
    for row in plan['assignments']:
        result = results.get(row['run_id'],{})
        groups[(row['model_label'],row['budget_profile'],row['framing'])].append((row,result))
    cells = []
    for (model,profile,frame),members in sorted(groups.items()):
        costs = [r['metadata']['api_cost_usd'] for _,r in members
                 if r.get('metadata',{}).get('api_cost_usd') is not None]
        cells.append(dict(model=model,budget_profile=profile,framing=frame,allocated=len(members),
                          valid_submissions=sum('metadata' in r for _,r in members),
                          completed_evaluations=sum(r.get('result',{}).get('status')=='completed' for _,r in members),
                          api_reported_cost_usd=sum(costs) if costs else None))
    contrasts = []
    for (model,profile,frame),members in sorted(groups.items()):
        if frame=='neutral': continue
        controls = {row['replicate']:row for row,_ in groups.get((model,profile,'neutral'),[])}
        differences = []
        for row,_ in members:
            control = controls.get(row['replicate'])
            if control is None: continue
            a = ratings.get((row['replicate'],row['run_id']))
            b = ratings.get((row['replicate'],control['run_id']))
            if a is not None and b is not None: differences.append(a-b)
        interval = None
        if len(differences)>=5 and len(set(differences))>1:
            rng = random.Random(0)
            draws = sorted(sum(rng.choices(differences,k=len(differences)))/len(differences) for _ in range(2000))
            interval = [draws[49],draws[1949]]
        contrasts.append(dict(model=model,budget_profile=profile,framing=frame,control='neutral',
            paired_authoring_runs=len(differences),mean_elo_delta=sum(differences)/len(differences) if differences else None,
            paired_run_bootstrap_ci95=interval,conditional_on_valid_submissions=True))
    return dict(cells=cells,contrasts=contrasts,replication_unit='independent bot-authoring run',
                caveat='Elo contrasts condition on valid bots; report submission attrition alongside strength. '
                       'Tiny or constant samples receive no confidence interval. Map-level uncertainty is separate.')
