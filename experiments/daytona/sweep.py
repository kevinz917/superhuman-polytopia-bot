"""Iso-FLOP net-size sweep for the 1v1 bot on Daytona (docs/isoflop-sweep-plan-2026-10-28.md).

    python experiments/daytona/sweep.py plan                     # arms, updates, fork points, projected hours and $
    python experiments/daytona/sweep.py stage-anchors            # hero-1v1 snapshots (Modal staging) -> the volume
    python experiments/daytona/sweep.py init                     # random-init parents of every size on the volume
    python experiments/daytona/sweep.py launch ARM [--hours H]   # a trunk / replicate, or a fork once its trunk passed
    python experiments/daytona/hero.py keep-remote ARM --max-usd X   # then the hero keeper runs the arm (and its
                                                                 # final round vs the budget's anchors)
    python experiments/daytona/sweep.py matches [--boxes 40]     # neighbours within a budget, replicates, anchor chain
    python experiments/daytona/sweep.py fit                      # Bradley-Terry Elo, N_opt per budget, exponent b

Arms (BUDGETS in B128 = the compute of 1 B samples on 4x128, ~3.7e18 FLOPs; samples = budget / relative size):
  sw-<size>        the trunk: C3, constant LR to 0.75 C3 then decay; snapshot_at saves the C1 and C2 fork points
  sw-<size>-c1/c2  cooldown forks: resume the trunk at 0.75 C_k, decay over the last 0.25 C_k (distributed.fork_run)
  sw-<size>-c1-s2  second seed at C1, trained from its own random init
Every arm is a hero.PLANS entry (hero.py imports plans()), so hero.py's launch, keep, status, stop and down run it.
Learner rates for 6x160 / 8x192 and their actor counts are projections from the 4x128 numbers: bench them first.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIZES = {'4x128': (4, 128), '6x160': (6, 160), '8x192': (8, 192)}   # blocks x width
BUDGETS = {'c1': .25, 'c2': .75, 'c3': 2.25}                           # B128
BATCH, MINIBATCH = 65536, 4096   # 262k would leave the 8x192 C1 arm ~210 updates; 65k matched it per sample (Oct 23)
SEEDS = (811, 812)
REFERENCE_CONV = 4 * 2 * 9 * 128 * 128
# Projected learner samples/s on 4x RTX PRO 6000: 4x128 is the live ~112k/s (assembly cap); the bigger nets scale by
# FLOPs from the 125k/s cap (GPU-bound); one PRO 6000 actor sandbox makes ~29.5k/s of 4x128 (docs/hero-daytona).
LEARNER_RATE = {'4x128': 112000, '6x160': 55000, '8x192': 28000}
ACTORS = {'4x128': 4, '6x160': 2, '8x192': 1}
# Anchors: hero-1v1 snapshots (4x128, same rules and recipe, batch 262k, constant-LR phase) bracketing each budget's
# 4x128 arm (C1 = 0.25 B ~ hero update 950, C2 ~ 2,860, C3 ~ 8,580). The chain of consecutive anchors ties the
# budgets onto one Elo scale.
ANCHOR_UPDATES = {'c1': (750, 1250), 'c2': (2500, 3500), 'c3': (7500, 10000)}
ANCHOR_DIR = '/results/sweep/anchors'
PARENT_DIR = '/results/sweep/parents'
ARENA_DIR = '/results/sweep/arena'
MATCH_MAPS, MAPS_PER_BOX, SEED_BASE = 128, 16, 540000   # 128 maps x 2 seats = 256 games per pair


def conv(size):
    blocks, width = SIZES[size]
    return blocks * 2 * 9 * width * width


def updates(size, budget):
    """Updates for `budget` B128 on `size`, a multiple of 4 so that 0.75 x updates (the fork point) is whole."""
    u = BUDGETS[budget] * 1e9 * REFERENCE_CONV / conv(size) / BATCH
    return int(round(u / 4)) * 4


def anchor_path(u):
    return f'{ANCHOR_DIR}/hero1-u{u}.npz'


def anchors(budget):
    return {f'hero1-u{u}': anchor_path(u) for u in ANCHOR_UPDATES[budget]}


def base_config():
    cfg = json.loads((ROOT / 'experiments/hero/hero-1v1.json').read_text())
    cfg.pop('parent', None)
    cfg.update(batch=BATCH, minibatch=MINIBATCH, learning_rate=3e-4, learning_rate_end=1e-4, lr_decay_fraction=.25,
               checkpoint_every=20, snapshot_every=250, historical_every=250, critic_path='')
    return cfg


def arms():
    """name -> dict(size, budget, seed, updates, fork=(trunk, update) or None, snapshot_at)."""
    out = {}
    for size in SIZES:
        u = {b: updates(size, b) for b in BUDGETS}
        forks = {b: u[b] * 3 // 4 for b in ('c1', 'c2')}
        out[f'sw-{size}'] = dict(size=size, budget='c3', seed=SEEDS[0], updates=u['c3'], fork=None,
                                 snapshot_at=sorted(forks.values()))
        for b, at in forks.items():
            out[f'sw-{size}-{b}'] = dict(size=size, budget=b, seed=SEEDS[0], updates=u[b], fork=(f'sw-{size}', at),
                                         snapshot_at=[])
        out[f'sw-{size}-c1-s2'] = dict(size=size, budget='c1', seed=SEEDS[1], updates=u['c1'], fork=None, snapshot_at=[])
    return out


def plans():
    """hero.PLANS entries: hero-1v1's 1v1 resources, the arm's own LearnerConfig (run_config), a random-init parent,
    no periodic ladder, and a final round of MATCH_MAPS maps x 2 seats vs each of the budget's two anchors."""
    out = {}
    for name, arm in arms().items():
        cfg = dict(base_config(), seed=arm['seed'], training_updates=arm['updates'], snapshot_at=arm['snapshot_at'])
        out[name] = dict(
            sweep=arm, run_config=cfg, players=2, parent=f"{PARENT_DIR}/{arm['size']}.npz".replace('/results', '/vol/results'),
            learner=dict(gpu=4, gpu_type=['RTX-PRO-6000'], cpu=64, memory=128, disk=100),
            actors=ACTORS[arm['size']], actor=dict(gpu=1, gpu_type=['RTX-PRO-6000', 'RTX-5090'], cpu=16, memory=64, disk=40),
            processes=48, games=24, bench_per_actor=29500, learner_cap=LEARNER_RATE[arm['size']],
            eval=dict(kind='ladder', interval=10 ** 9, maps=MATCH_MAPS // 2, maps_per_box=MAPS_PER_BOX // 2,
                      final_maps=MATCH_MAPS, seed_base=SEED_BASE + 50000, references=anchors(arm['budget'])))
    return out


def projected(name, arm):
    """(samples, hours, usd) of the arm's own training (a fork: from its fork point)."""
    import hero
    p = hero.PLANS[name]
    start = arm['fork'][1] if arm['fork'] else 0
    samples = (arm['updates'] - start) * BATCH
    hours = samples / LEARNER_RATE[arm['size']] / 3600 / .9   # 10% of the clock lost to spot reclaims
    usd = hours * (hero.hourly(p['learner']) + p['actors'] * hero.hourly(p['actor']))
    return samples, hours, usd


# ---- commands
def cmd_plan(a):
    total = 0.
    for name, arm in arms().items():
        samples, hours, usd = projected(name, arm)
        total += usd
        flops = 3136 * conv(arm['size']) * arm['updates'] * BATCH
        print(json.dumps(dict(arm=name, size=arm['size'], budget=arm['budget'], seed=arm['seed'], updates=arm['updates'],
                              fork=arm['fork'], snapshot_at=arm['snapshot_at'] or None, flops=f'{flops:.2e}',
                              own_samples_B=round(samples / 1e9, 3), hours=round(hours, 2), usd=round(usd))))
    print(json.dumps(dict(training_usd=round(total), note='projected; bench 6x160/8x192 learner and actor rates first')))


def _hub_python(h, fn, spec):
    """Run node.py `fn` on the hub from a scratch copy of the published code, leaving /workspace (the live hero
    keeper's code) untouched."""
    import hero
    sha = hero.code_sha()
    if 'ok' not in hero.sh(h, f'test -f /vol/code/{sha}/OK && echo ok', check=False):
        raise SystemExit(f'code {sha} not published: hero.py publish')
    d = f'/tmp/sweep-code-{sha}'
    return hero.sh(h, f'set -e; mkdir -p {d} && cd {d} && (test -f CODE_SHA || tar xzf /vol/code/{sha}/code.tar.gz) && '
                      f'echo {sha} > CODE_SHA && echo {hero._quote(json.dumps(spec))} > /tmp/sweep-spec.json && '
                      f'JAX_PLATFORMS=cpu PYTHONPATH={d}/src /opt/venv/bin/python experiments/daytona/node.py {fn} '
                      f'/tmp/sweep-spec.json', timeout=1800)


def cmd_stage_anchors(a):
    """hero-1v1 snapshots from the Modal staging volume to /vol/results/sweep/anchors (weights + their .json)."""
    import hero
    import tempfile
    h = hero.hub()
    with tempfile.TemporaryDirectory() as tmp:
        for u in sorted({u for us in ANCHOR_UPDATES.values() for u in us}):
            for ext in ('npz', 'json'):
                dst = f'/vol{ANCHOR_DIR}/hero1-u{u}.{ext}'
                if 'ok' in hero.sh(h, f'test -f {dst} && echo ok', check=False):
                    continue
                local = Path(tmp) / f'u{u}.{ext}'
                subprocess.run(['modal', 'volume', 'get', '--env', 'staging', 'polytopia-structured-rl',
                                f'distributed/hero-1v1/snapshots/update-{u}.{ext}', str(local)], check=True)
                hero.put(h, dst, local.read_bytes())
            print(json.dumps(dict(anchor=f'hero1-u{u}')), flush=True)


def cmd_init(a):
    import hero
    h = hero.hub()
    for size, (blocks, width) in SIZES.items():
        out = f'/vol{PARENT_DIR}/{size}.npz'
        print(_hub_python(h, 'sweep_parent', dict(size=14, blocks=blocks, width=width, players=2, out=out)).strip()[-300:])


def cmd_launch(a):
    import hero
    arm = arms()[a.arm]
    h = hero.hub()
    folder = f'/vol/results/distributed/{a.arm}'
    if arm['fork'] and 'ok' not in hero.sh(h, f'test -f {folder}/latest.npz && echo ok', check=False):
        trunk, at = arm['fork']
        if 'ok' not in hero.sh(h, f'test -f /vol/results/distributed/{trunk}/snapshots/update-{at}-critic.npz && echo ok',
                               check=False):
            raise SystemExit(f'{trunk} has not reached update {at} yet (or not synced it): launch {a.arm} later')
        print(_hub_python(h, 'sweep_fork', dict(trunk=trunk, fork=a.arm, update=at)).strip()[-300:])
    elif not arm['fork'] and 'ok' not in hero.sh(h, f"test -f {hero.PLANS[a.arm]['parent']} && echo ok", check=False):
        raise SystemExit('no random-init parent on the volume: sweep.py init')
    samples, hours, usd = projected(a.arm, arm)
    a.hours = a.hours or max(2., round(hours * 1.5 + 1, 1))   # training_updates stops it; hours: TTL and backstop
    print(json.dumps(dict(arm=a.arm, projected_hours=round(hours, 2), projected_usd=round(usd),
                          keeper=f'hero.py keep-remote {a.arm} --max-usd {round(usd * 1.5)}')), flush=True)
    hero.cmd_launch(argparse.Namespace(run=a.arm, hours=a.hours, actors=a.actors, dry_run=a.dry_run,
                                       learner_gpu=a.learner_gpu, on_demand=False, actor_gpu=None, actor_memory=None))


def match_pairs():
    """(tag, candidate, opponent): neighbouring sizes within each budget, each C1 replicate vs its first seed, and
    consecutive anchors."""
    final = lambda arm: f'/results/distributed/{arm}/final.npz'
    name = lambda size, budget: f'sw-{size}' if budget == 'c3' else f'sw-{size}-{budget}'
    sizes, pairs = list(SIZES), []
    for budget in BUDGETS:
        for small, big in zip(sizes, sizes[1:]):
            pairs.append((name(big, budget), name(small, budget)))
    pairs += [(f'sw-{size}-c1-s2', f'sw-{size}-c1') for size in sizes]
    out = [(f'{a}-vs-{b}', final(a), final(b)) for a, b in pairs]
    chain = sorted({u for us in ANCHOR_UPDATES.values() for u in us})
    out += [(f'hero1-u{hi}-vs-hero1-u{lo}', anchor_path(hi), anchor_path(lo)) for lo, hi in zip(chain, chain[1:])]
    return out


def cmd_matches(a):
    """Every match_pairs() pair over MATCH_MAPS maps, MAPS_PER_BOX maps per 4-vCPU sandbox, at most --boxes at once
    (the org's CPU sandboxes share 250 vCPU with the hero keeper's evals). Finished parts are skipped on a rerun."""
    import hero
    h, sha = hero.hub(), hero.code_sha()
    jobs = []
    for i, (tag, cand, opp) in enumerate(match_pairs()):
        if 'ok' not in hero.sh(h, f'test -f /vol{cand} && test -f /vol{opp} && echo ok', check=False):
            print(json.dumps(dict(skip=tag, reason='a checkpoint is missing')), flush=True)
            continue
        for lo in range(0, MATCH_MAPS, MAPS_PER_BOX):
            seed = SEED_BASE + 1000 * i + lo
            jobs.append(dict(name=f'{tag}-{seed}', candidate=cand, opponent=opp, seed_base=seed, maps=MAPS_PER_BOX,
                             map_mix='proposed-v1', size=14, games=4, processes=4, reference=tag,
                             out=f'/vol{ARENA_DIR}/{tag}-{seed}.json'))
    done = lambda: set(hero.sh(h, f'ls /vol{ARENA_DIR} 2>/dev/null; true').split())
    have = done()
    todo = [j for j in jobs if Path(j['out']).name not in have]
    running, lost = {}, []   # out -> (sandbox, started)
    print(json.dumps(dict(jobs=len(jobs), todo=len(todo))), flush=True)
    while todo or running:
        while todo and len(running) < a.boxes:
            j = todo.pop(0)
            try:
                s = hero.create(f'sw-m{len(todo)}-{int(time.time()) % 10000}', dict(run='sweep', role='eval'),
                                ttl_minutes=240, **hero.EVAL_BOX)
                hero.bootstrap(s, sha)
                hero.start(s, 'hero_arena', j, env=dict(JAX_PLATFORMS='cpu'), log='eval')
                running[Path(j['out']).name] = (s, time.time())
            except Exception as error:
                print(json.dumps(dict(box_error=repr(error)[:300])), flush=True)
                todo.append(j)
                time.sleep(60)
                break
        time.sleep(60)
        have = done()
        for out in [o for o in running if o in have or time.time() - running[o][1] > 3 * 3600]:
            if out not in have:   # a part that never wrote its result: left out of the fit, rerun `matches` for it
                lost.append(out)
            hero.delete(running.pop(out)[0])
        print(json.dumps(dict(at=time.time(), running=len(running), todo=len(todo), lost=len(lost))), flush=True)


def bradley_terry(results, fixed, reps=300, seed=0):
    """results: [(a, b, score of a, games)]. Elo per player (fixed = 0) and a parametric bootstrap (binomial resample of
    each pair's score). One virtual drawn game per pair keeps 0-n pairs finite."""
    import numpy as np
    players = sorted({p for a, b, _, _ in results for p in (a, b)})
    idx = {p: i for i, p in enumerate(players)}
    pa, pb = np.array([idx[a] for a, *_ in results]), np.array([idx[b] for _, b, *_ in results])
    games = np.array([g for *_, g in results], int)
    score = np.array([s for _, _, s, _ in results], float)
    n = games + 1.

    def fit(won):
        s, r = won + .5, np.zeros(len(players))
        for _ in range(3000):
            p = 1 / (1 + np.exp(r[pb] - r[pa]))
            grad = np.zeros_like(r)
            np.add.at(grad, pa, s - n * p)
            np.add.at(grad, pb, n * p - s)
            r += .5 * grad / max(n.max(), 1.)
            r -= r[idx[fixed]]
        return r * 400 / math.log(10)
    rng = np.random.default_rng(seed)
    point = fit(score)
    boot = np.array([fit(rng.binomial(games, np.clip(score / np.maximum(games, 1), 0, 1))) for _ in range(reps)])
    return players, point, boot


def cmd_fit(a):
    """Gather every arm's final round vs its anchors and the sweep matches; Elo per arm; per budget the quadratic in
    log N through the three sizes and its argmax N_opt (bootstrap CI; 'edge' when it is a minimum or off the range);
    the slope b of log N_opt over log C."""
    import numpy as np
    import hero
    h, results = hero.hub(), []
    for name, arm in arms().items():
        rows = json.loads(hero.sh(h, f'cat /vol/runs/{name}/eval-history.json 2>/dev/null || echo []'))
        final = [r for r in rows if r.get('final')]
        for ref, t in (final[-1]['vs'] if final else {}).items():
            results.append((name, ref, t['wins'] + .5 * t['draws_or_truncations'], t['games']))
    tags = {}
    for line in hero.sh(h, f'cd /vol{ARENA_DIR} 2>/dev/null && for f in *.json; do python3 -c "import json,sys; '
                           f'd=json.load(open(sys.argv[1])); print(json.dumps([sys.argv[1], d[\'wins\'], d[\'losses\'], '
                           f'd.get(\'draws_or_truncations\', 0), d[\'games\']]))" $f; done; true').splitlines():
        f, w, l, d, g = json.loads(line)
        tag = f.rsplit('-', 1)[0]
        t = tags.setdefault(tag, [0., 0])
        t[0] += w + .5 * d
        t[1] += g
    for tag, (score, games) in tags.items():
        cand, opp = tag.split('-vs-')
        results.append((cand, opp, score, games))
    if not results:
        raise SystemExit('no results yet')
    players, point, boot = bradley_terry(results, fixed='hero1-u750')
    elo = dict(zip(players, point))
    lo, hi = dict(zip(players, np.percentile(boot, 2.5, axis=0))), dict(zip(players, np.percentile(boot, 97.5, axis=0)))
    table = {p: dict(elo=round(elo[p]), ci=[round(lo[p]), round(hi[p])]) for p in players}
    nopt = {}
    for budget in BUDGETS:
        names = [f'sw-{s}' if budget == 'c3' else f'sw-{s}-{budget}' for s in SIZES]
        if not all(n in elo for n in names):
            continue
        x = np.log([conv(s) for s in SIZES])

        def argmax(y):
            c2, c1, _ = np.polyfit(x, y, 2)
            v = -c1 / (2 * c2) if c2 < 0 else None
            return float(np.exp(v)) if v is not None and x[0] <= v <= x[-1] else None
        reps = [argmax([b[players.index(n)] for n in names]) for b in boot]
        inside = [r for r in reps if r is not None]
        nopt[budget] = dict(point=argmax([elo[n] for n in names]), edge_fraction=round(1 - len(inside) / len(reps), 3),
                            ci=[float(np.percentile(inside, 2.5)), float(np.percentile(inside, 97.5))] if inside else None)
    fitted = [(BUDGETS[b], v['point']) for b, v in nopt.items() if v['point']]
    slope = float(np.polyfit(np.log([c for c, _ in fitted]), np.log([n for _, n in fitted]), 1)[0]) if len(fitted) >= 2 else None
    out = dict(elo=table, n_opt=nopt, exponent_b=slope, results=len(results))
    print(json.dumps(out, indent=1))
    (ROOT / 'runs' / 'hero-daytona').mkdir(parents=True, exist_ok=True)
    (ROOT / 'runs' / 'hero-daytona' / 'sweep-fit.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('plan')
    sub.add_parser('stage-anchors')
    sub.add_parser('init')
    q = sub.add_parser('launch'); q.add_argument('arm', choices=arms()); q.add_argument('--hours', type=float)
    q.add_argument('--actors', type=int); q.add_argument('--learner-gpu'); q.add_argument('--dry-run', action='store_true')
    q = sub.add_parser('matches'); q.add_argument('--boxes', type=int, default=40)
    sub.add_parser('fit')
    a = ap.parse_args()
    globals()['cmd_' + a.cmd.replace('-', '_')](a)
