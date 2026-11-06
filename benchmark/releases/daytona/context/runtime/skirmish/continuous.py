"""The single, continuous Polytopia Skirmish benchmark contract."""
from collections import Counter
from pathlib import Path
from .common import canonical, sha

SCHEMA = 'polytopia-skirmish-continuous-v1'
CODING_SECONDS = 2400
EVALUATION = dict(max_rounds=200, max_actions=40000, fuel=4_000_000_000_000,
                  move_seconds=2., game_seconds=1800, exact_rounds=True, size=14)
FAMILIES = ('drylands', 'lakes', 'continents')


def source_files():
    from .local import sources
    root = Path(__file__).resolve().parents[3]
    result = sources()
    paths = [*(root/'harbor').rglob('*'), *(root/'skirmish/continuous').rglob('*'),
             root/'configs/skirmish/continuous-v1.json', root/'scripts/build_skirmish_harbor.py',
             root/'scripts/run_skirmish_harbor.py', root/'scripts/check_skirmish_harbor.py',
             root/'scripts/summarize_skirmish_harbor.py', root/'pyproject.toml',
             root/'scripts/build_skirmish_daytona.py', root/'scripts/run_skirmish_split.py',
             root/'scripts/check_skirmish_daytona.py', root/'tests/test_skirmish_split.py',
             root/'scripts/check_skirmish_proxy.py',
             root/'configs/skirmish/local-v1-top12.json',
             *(root/'tests').glob('test_skirmish*.py'),
             root/'tests/test_skirmish_harbor.py', root/'docs/skirmish/CONTINUOUS.md',
             root/'README.md', root/'skirmish/README.md']
    for path in paths:
        if path.is_file() and '__pycache__' not in path.parts:
            result[str(path.relative_to(root))] = sha(path.read_bytes())
    return result


def evaluation_jobs(manifest, maps, *, demo=False):
    """One frozen source; every opponent, family and seat has equal weight."""
    if manifest['schema'] != SCHEMA or manifest['coding_seconds'] != CODING_SECONDS:
        raise ValueError('continuous-v1 has exactly one 2400-second authoring protocol')
    if manifest['evaluation'] != EVALUATION:
        raise ValueError('continuous-v1 evaluation contract changed')
    if manifest['maps_per_family'] != 10:
        raise ValueError('standard continuous-v1 uses ten layouts per family')
    refs = manifest['references']
    if len(refs) != 6 or len({r['label'] for r in refs}) != 6 or sum(r['label']=='hero-1v1' for r in refs) != 1:
        raise ValueError('six distinct references including the champion required')
    if len({m['seed'] for m in maps}) != len(maps):
        raise ValueError('map seeds must be unique')
    expected = 1 if demo else manifest['maps_per_family']
    if Counter(m['map_type'] for m in maps) != dict.fromkeys(FAMILIES, expected):
        raise ValueError('incomplete or unbalanced map suite')
    if any(not 1_000_000_000 <= m['seed'] < 1_500_000_000 for m in maps):
        raise ValueError('held-out seed partition required')
    jobs = []
    for ref in refs:
        for m in maps:
            for seat in (1, 2):
                job = dict(index=len(jobs), candidate=dict(kind='submission', run_id='candidate',
                    stage=0, label='submission', model_label='submission', track='continuous',
                    replicate=1), opponent=ref, **m, seat=seat, phase='heldout', evaluation=EVALUATION)
                job['job_sha256'] = sha(canonical(job).encode())
                job['output_relative'] = f'games/{job["index"]:06d}'
                jobs.append(job)
    return jobs


def rewards(rows, expected_games):
    if len(rows) != expected_games or len({r['job_index'] for r in rows}) != expected_games:
        raise ValueError('incomplete evaluation')
    if any(not r['audit']['passed'] for r in rows):
        raise ValueError('unaudited evaluation')
    if any(r['result'] == 'unresolved' for r in rows):
        raise RuntimeError('infrastructure outcome unresolved; do not award a zero score')
    points = {'win': 1., 'draw': .5, 'loss': 0.}
    hero = [r for r in rows if r['opponent_label'] == 'hero-1v1']
    return dict(reward=sum(points[r['result']] for r in rows)/len(rows),
        champion_win_rate=sum(r['result']=='win' for r in hero)/len(hero),
        champion_points=sum(points[r['result']] for r in hero)/len(hero),
        games=expected_games, runtime_fault_rate=sum(r['status']=='forfeit' for r in rows)/len(rows),
        compilation_loss_rate=sum(r['status']=='invalid_submission' for r in rows)/len(rows))
