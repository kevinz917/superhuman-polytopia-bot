"""Summarize the expert-SFT comparison on Modal: outcomes, strategy signals, paired map bootstrap."""
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import zlib

FAM = {1: 'move', 2: 'attack', 3: 'ability', 4: 'recruit', 5: 'harvest', 6: 'build', 7: 'research',
       8: 'capture', 9: 'city_reward', 10: 'end_turn', 11: 'resign'}


def rows(path):
    dec, buf = zlib.decompressobj(31), b''
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            try:
                buf += dec.decompress(chunk)
            except zlib.error:
                break
            *lines, buf = buf.split(b'\n')
            for line in lines:
                try:
                    yield json.loads(line)
                except ValueError:
                    return


def trace_metrics(path, seat):
    acts, combat, abilities = Counter(), Counter(), Counter()
    peak, owned = 1, {1: set(), 2: set()}
    for r in rows(path):
        if 'selected_command' not in r:
            continue
        obs = r['observation']; actor = obs['player']; cmd = r['selected_command']; after = r['after']
        who = 'model' if actor == seat else 'opp'
        if who == 'model':
            acts[FAM.get(cmd['kind'], str(cmd['kind']))] += 1
            if cmd['kind'] == 3:
                abilities[{7: 'recover', 8: 'disband', 9: 'heal'}.get(cmd['parameter'], 'other')] += 1
        if cmd['kind'] == 2:
            tgt = cmd['target']
            hit = [u for u in obs['visible_state']['units'] if u['tile'] == tgt and u['owner'] == 3-actor]
            alive = [u for u in after[actor-1]['visible_state']['units'] if u['tile'] == tgt and u['owner'] == 3-actor]
            combat[f'{who}_kills'] += bool(hit and not alive)
        if cmd['kind'] == 8:
            combat[f'{who}_captures'] += 1
        now = {p: {c['tile'] for c in after[p-1]['visible_state']['cities'] if c['owner'] == p} for p in (1, 2)}
        for p in (1, 2):
            if now[p] & owned[3-p]:
                combat[('model' if p == seat else 'opp')+'_takes_city'] += len(now[p] & owned[3-p])
        owned = now
        peak = max(peak, len(now[seat]))
    return dict(actions=dict(acts), combat=dict(combat), abilities=dict(abilities), peak_cities=peak)


def llm_arm(root, arm, jobs):
    games = {g['id']: g for g in json.loads((root/arm/'games'/'games.json').read_text())}
    out = {}
    for j in jobs:
        g = games.get(j['id'])
        m = trace_metrics(root/arm/'games'/(j['id']+'.jsonl.gz'), j['seat'])
        out[j['id']] = dict(seed=j['seed'], seat=j['seat'], status=g['status'] if g else 'missing',
                            reason=g['reason'] if g else None, reward=g['reward'] if g else None,
                            round=g['round'] if g else None, model_moves=g['model_decisions'] if g else None, **m)
    return out


def rl_arm(root, jobs):
    games = json.loads((root/'rl-reference'/'games.json').read_text())
    by = {(g['seed'], g['seat']): g for g in games}
    out = {}
    for j in jobs:
        g = by.get((j['seed'], j['seat']))
        status = 'missing' if g is None else 'native' if g['terminated'] else 'truncated' if g['truncated'] else 'unresolved'
        out[j['id']] = dict(seed=j['seed'], seat=j['seat'], status=status, reward=g and g['reward'],
                            round=g and g['turn'], model_moves=g and g['actions'], peak_cities=g and g['max_cities'])
    return out


def win(g):
    return int(g['status'] == 'native' and (g['reward'] or 0) > 0)


def paired(a, b, jobs):
    maps = sorted({j['seed'] for j in jobs})
    diffs = [statistics.mean(win(a[j['id']])-win(b[j['id']]) for j in jobs if j['seed'] == m) for m in maps]
    rng = random.Random(1701)
    boot = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(20000))
    return dict(a_minus_b=statistics.mean(diffs), map_cluster_bootstrap_95=[boot[499], boot[19499]], maps=len(maps),
                maps_a_better=sum(d > 0 for d in diffs), maps_b_better=sum(d < 0 for d in diffs))


def aggregate(arm):
    gs = list(arm.values())
    agg = dict(games=len(gs), statuses=dict(Counter(f"{g['status']}/{g.get('reason')}" for g in gs)),
               wins=sum(win(g) for g in gs), losses=sum(g['status'] == 'native' and (g['reward'] or 0) < 0 for g in gs),
               forfeits=sum(g['status'] == 'forfeit' for g in gs),
               truncated=sum(g['status'] == 'truncated' for g in gs),
               operational=sum(g['status'] not in ('native', 'forfeit', 'truncated') for g in gs),
               median_round=statistics.median(g['round'] for g in gs if g['round'] is not None),
               mean_peak_cities=statistics.mean(g['peak_cities'] for g in gs if g['peak_cities'] is not None))
    if 'actions' in gs[0]:
        acts, combat, ab = Counter(), Counter(), Counter()
        for g in gs:
            acts.update(g['actions']); combat.update(g['combat']); ab.update(g['abilities'])
        agg.update(actions=dict(acts.most_common()), combat=dict(combat), abilities=dict(ab))
    return agg


def summarize(root):
    root = Path(root)
    p = json.loads((root/'protocol.json').read_text())
    jobs = p['jobs']
    arms = {a: llm_arm(root, a, jobs) for a in p['arms']}
    arms['rl_expert'] = rl_arm(root, jobs)
    result = dict(schema='expert-sft-summary-v1', arms={a: aggregate(v) for a, v in arms.items()},
                  paired={f'{a}_vs_{b}': paired(arms[a], arms[b], jobs)
                          for a, b in [('expert_sft', 'base'), ('old_sft', 'base'), ('expert_sft', 'old_sft'), ('rl_expert', 'base')]
                          if a in arms and b in arms},
                  games={a: v for a, v in arms.items()},
                  note='Unresolved/operational games never count as wins or losses; truncated = round-100 horizon.')
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    small = {k: v for k, v in result.items() if k != 'games'}
    (root/'summary-small.json').write_text(json.dumps(small, indent=2)+'\n')
    return small
