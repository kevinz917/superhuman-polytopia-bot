"""Predeclared map-paired prompt selection; no teacher-agreement optimization."""
import random
from collections import defaultdict


def conservative_score(game):
    if game['status'] != 'native':
        return 0.
    return 1. if game['reward'] > 0 else .5 if game['reward'] == 0 else 0.


def paired_interval(candidate, original, *, resamples=10000, seed=17029):
    a = {(g['board_hash'], g['seat']): conservative_score(g) for g in candidate}
    b = {(g['board_hash'], g['seat']): conservative_score(g) for g in original}
    if len(a) != len(candidate) or len(b) != len(original) or a.keys() != b.keys() or not a:
        raise ValueError('paired evaluation requires identical unique map/seat blocks')
    maps = defaultdict(list)
    for key in sorted(a):
        maps[key[0]].append(a[key]-b[key])
    if any(len(v)!=2 for v in maps.values()):
        raise ValueError('every map requires both seats')
    differences = [sum(v)/2 for v in maps.values()]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(differences, k=len(differences)))/len(differences) for _ in range(resamples))
    return dict(difference=sum(differences)/len(differences), lower95=boot[int(resamples*.025)],
                upper95=boot[min(resamples-1,int(resamples*.975))], maps=len(maps), resamples=resamples, seed=seed)


def select_prompt(games, candidates, *, maps_required=10):
    """Require a complete registered stage; ambiguous results retain original.

    A revision must complete >=95% natively, truncate <5%, produce >=95%
    valid first attempts, and improve conservative score by >=10 percentage
    points with a positive lower map-bootstrap bound. Ties use candidate order.
    """
    expected = maps_required*2
    grouped = {name: [g for g in games if g['candidate']==name] for name in candidates}
    if 'original' not in grouped or sum(map(len,grouped.values())) != len(games):
        raise ValueError('unexpected candidate set')
    if any(len(rows)!=expected or any(g['status'] not in ('native','truncated','forfeit') for g in rows) for rows in grouped.values()):
        raise ValueError('incomplete prompt gameplay allocation')
    summaries = {}
    for name, rows in grouped.items():
        native = sum(g['status']=='native' for g in rows)/expected
        truncated = sum(g['status']=='truncated' for g in rows)/expected
        decisions = sum(g['model_decisions'] for g in rows)
        valid = sum(g['first_attempt_valid'] for g in rows)/max(decisions,1)
        pair = paired_interval(rows, grouped['original'])
        summaries[name] = dict(native_fraction=native,truncation_fraction=truncated,
            first_attempt_valid_fraction=valid,conservative_score=sum(map(conservative_score,rows))/expected,
            paired=pair,eligible=name!='original' and native>=.95 and truncated<.05 and valid>=.95 and
            pair['difference']>=.10 and pair['lower95']>0)
    eligible = [name for name in candidates if summaries[name]['eligible']]
    chosen = max(eligible,key=lambda name:summaries[name]['conservative_score']) if eligible else 'original'
    return dict(selected=chosen,summaries=summaries,
                reason='Clear paired improvement with reliability gates' if eligible else 'No clear qualified improvement; retain original')
