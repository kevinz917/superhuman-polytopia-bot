"""12-hour cost matrix for continuing the hero runs, from the 2026-10-28 measurements
(docs/hero-daytona-2026-10-28.md, docs/hero-throughput-2026-10-28.md).

    python experiments/daytona/cost_matrix.py [--hours 8] [--reclaim-loss .1]

Rate = min(learner cap, actors x live rate per actor). Daytona spot learners lose `reclaim_loss` of the clock to
reclaims (two H100 learners and one H200 were reclaimed within ~2 h on 2026-10-28; no RTX PRO 6000 was).
"""
import argparse
import json

D_GPU = {'H100': 2.27, 'H200': 2.61, 'RTX-PRO-6000': 1.74, 'RTX-5090': 0.74, 'RTX-4090': 0.57}   # Daytona spot $/h
D_VCPU, D_MEM = 0.0504, 0.0162
M_GPU = {'H100': 3.95, 'H200': 4.54, 'A10G': 1.10}                                                # Modal $/h
M_CORE, M_MEM = 0.0472, 0.008
M_EGRESS_PER_B = {'1v1': 5.6, '4p': 8.0}   # docs/hero-runs-plan-2026-10-26.md

# name, run, platform, learner (gpu, n), learner cap/s, actor (gpu, memory GiB), live rate per actor, actors, basis
SCENARIOS = [
    ('A', '1v1', 'Modal', ('H100', 4), 208000, ('A10G', 96), 13600, 17, 'live smoke on main, 2026-10-28'),
    ('B', '1v1', 'Daytona', ('RTX-PRO-6000', 4), 125000, ('RTX-PRO-6000', 64), 29500, 5, 'live smoke 118k/s on 4 actors'),
    ('C', '1v1', 'Daytona', ('H100', 2), 160000, ('RTX-PRO-6000', 64), 29500, 6, 'learner: 2x H100 smoke 1.48 s/batch'),
    ('D', '1v1', 'Daytona', ('RTX-5090', 4), 120000, ('RTX-PRO-6000', 64), 29500, 5, 'learner: bench 165k train only'),
    ('E', '4p', 'Modal', ('H200', 4), 92000, ('A10G', 96), 14700, 7, 'live smoke on main, 2026-10-28'),
    ('F', '4p', 'Daytona', ('RTX-PRO-6000', 4), 65000, ('RTX-PRO-6000', 160), 20000, 5, 'learner live 1.91 s/batch; actor bench 28k'),
    ('G', '4p', 'Daytona', ('RTX-PRO-6000', 4), 65000, ('RTX-4090', 112), 6300, 11, 'actor live 6.3k (Zen 3 hosts)'),
]


def hourly(platform, run, learner, actor, actors, rate):
    if platform == 'Modal':
        lh = learner[1] * M_GPU[learner[0]] + 32 * M_CORE + 64 * M_MEM
        ah = M_GPU[actor[0]] + 16 * M_CORE + actor[1] * M_MEM
        return lh, ah, rate * 3600 / 1e9 * M_EGRESS_PER_B[run]
    lh = learner[1] * D_GPU[learner[0]] + 16 * learner[1] * D_VCPU + 128 * D_MEM
    ah = D_GPU[actor[0]] + 16 * D_VCPU + actor[1] * D_MEM
    return lh, ah, 0.


def table(hours, loss):
    rows = []
    for name, run, platform, learner, cap, actor, per, actors, basis in SCENARIOS:
        rate = min(cap, actors * per) * (1 - (loss if platform == 'Daytona' else 0))
        lh, ah, eh = hourly(platform, run, learner, actor, actors, rate)
        usd_h = lh + actors * ah + eh
        rows.append(dict(option=name, run=run, platform=platform, learner=f'{learner[1]}x {learner[0]}',
                         actors=f'{actors}x {actor[0]}', samples_per_s=int(rate), usd_per_h=round(usd_h, 2),
                         samples_b=round(rate * hours * 3600 / 1e9, 2), usd=round(usd_h * hours),
                         usd_per_b=round(usd_h / (rate * 3600 / 1e9), 1), basis=basis))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=float, default=8)
    ap.add_argument('--reclaim-loss', type=float, default=.1)
    a = ap.parse_args()
    for r in table(a.hours, a.reclaim_loss):
        print(json.dumps(r))
