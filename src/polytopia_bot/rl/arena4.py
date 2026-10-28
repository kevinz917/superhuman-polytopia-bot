"""Four-seat evaluation: seat-rotated tables of policies on engine-rs four-seat games, and a rating fit.

A table is four entrants (policy names, repeats allowed) on one map; it is played four times with the seats
rotated, so every entrant sits in every seat once per map. Each game's result is the engine's rank rewards
(+1, +1/3, -1/3, -1; the seats still in at the round cap share their places). Ratings are a Bradley-Terry
(Elo-scale) fit to the pairwise comparisons inside each game: entrant A beats B when A's reward is higher, ties
count half. Pairs of the same policy are skipped. This is the standard decomposition of a multiplayer ranking
into pairwise results (e.g. TrueSkill's free-for-all model uses the same pairwise likelihoods).
"""
import json
import math
from itertools import combinations

import numpy as np

PLAYERS = 4


def rotations(entrants):
    """The four seat assignments of one table: entrant k sits in seat (k + r) % 4 in rotation r."""
    out = []
    for r in range(PLAYERS):
        seats = [None] * PLAYERS
        for k, name in enumerate(entrants):
            seats[(k + r) % PLAYERS] = name
        out.append(seats)
    return out


def play(policies, tables, *, size=18, maps='four-v1', seeds, lanes=32, deterministic=True, max_turns=None,
         seed=0, specs=None, log=None):
    """Play every (table, map seed) with all four rotations. policies: {name: params} (four-seat networks);
    tables: list of 4-entrant lists; seeds: map seeds (one table per seed, cycling through `tables`).
    Returns a list of game records {seat_names, rewards, turns, terminated, map_seed, kind}."""
    import jax
    import jax.numpy as jnp
    from . import network
    from .actions import Choice
    from .rust_engine import RustGames
    from polytopia_bot.maps.stream import MapSource
    from polytopia_bot.maps.state import stack

    source = MapSource(maps, size, players=PLAYERS)
    jobs = []
    for i, map_seed in enumerate(seeds):
        entrants = tables[i % len(tables)]
        for seats in rotations(entrants):
            jobs.append((int(map_seed), seats))
    lanes = min(lanes, len(jobs))
    engine = RustGames(size, lanes, map_batch=True, max_turns=max_turns, players=PLAYERS)
    sample = jax.jit(jax.vmap(lambda p, x, m, k: network.sample(p, x, m, k, size=size, deterministic=deterministic),
                              in_axes=(None, 0, 0, 0)))

    layouts = {}
    def layout(map_seed):
        if map_seed not in layouts:
            layouts[map_seed] = source.layout(map_seed)
        return layouts[map_seed]

    pending = list(range(len(jobs)))
    lane_job = [None] * lanes
    def maps_for(indices):
        rows = []
        for j in indices:
            map_seed = jobs[j][0] if j is not None else jobs[0][0]
            rows.append((map_seed, layout(map_seed)[1]))
        return stack(rows)
    first = [pending.pop(0) if pending else None for _ in range(lanes)]
    lane_job = first
    states, prepared = engine.reset(maps_for(lane_job))
    next_job = [pending.pop(0) if pending else None for _ in range(lanes)]
    records = []
    key = jax.random.key(seed)
    turns = np.zeros(lanes, np.int32)
    while any(j is not None for j in lane_job):
        current = np.asarray(states.current_player)
        choice = [np.zeros(lanes, np.int32) for _ in range(3)]
        by_policy = {}
        for lane, j in enumerate(lane_job):
            name = jobs[j][1][current[lane] - 1] if j is not None else next(iter(policies))
            by_policy.setdefault(name, []).append(lane)
        for name, idx in by_policy.items():
            key, k = jax.random.split(key)
            idx = np.asarray(idx)
            take = lambda a: jnp.take(a, idx, axis=0)
            out = sample(policies[name], jax.tree.map(take, prepared.features), jax.tree.map(take, prepared.masks),
                         jax.random.split(k, len(idx)))[0]
            for c in range(3):
                choice[c][idx] = np.asarray(out[c])
        tick, status = engine.tick(states, Choice(*(jnp.asarray(c) for c in choice)), maps_for(next_job))
        if np.any(np.asarray(status)):
            raise RuntimeError(f'illegal action on lanes {np.nonzero(np.asarray(status))[0].tolist()}')
        turns = np.maximum(turns, np.asarray(states.turn))
        done = np.asarray(tick.done)
        for lane in np.nonzero(done)[0]:
            j = lane_job[lane]
            if j is not None:
                map_seed, seats = jobs[j]
                records.append(dict(map_seed=map_seed, kind=layout(map_seed)[0], seat_names=list(seats),
                                    rewards=[float(r) for r in np.asarray(tick.rewards)[lane]],
                                    terminated=bool(np.asarray(tick.terminated)[lane]), turns=int(turns[lane])))
                if log is not None and len(records) % 16 == 0:
                    log(json.dumps(dict(stage='arena4_progress', games=len(records), of=len(jobs))))
            lane_job[lane] = next_job[lane]
            next_job[lane] = pending.pop(0) if pending else None
            turns[lane] = 0
        states, prepared = tick.states, tick.prepared
    return records


def pairwise(records):
    """{(a, b): [a's wins over b, games compared]} with a < b, ties counted half, same-name pairs skipped."""
    out = {}
    for rec in records:
        for i, j in combinations(range(PLAYERS), 2):
            a, b = rec['seat_names'][i], rec['seat_names'][j]
            if a == b:
                continue
            ra, rb = rec['rewards'][i], rec['rewards'][j]
            if a > b:
                a, b, ra, rb = b, a, rb, ra
            cell = out.setdefault((a, b), [0., 0])
            cell[0] += 1. if ra > rb else .5 if ra == rb else 0.
            cell[1] += 1
    return out


def fit_ratings(records, anchor=None, iterations=2000):
    """Bradley-Terry strengths by minorization-maximization, on the Elo scale (400 log10), mean 0 or `anchor` at 0.
    A half point of prior (one tie against a virtual average entrant) keeps undefeated or winless entrants finite."""
    pairs = pairwise(records)
    names = sorted({n for rec in records for n in rec['seat_names']})
    index = {n: i for i, n in enumerate(names)}
    k = len(names)
    wins = np.full(k, .5)
    games = np.zeros((k, k))
    for (a, b), (w, n) in pairs.items():
        i, j = index[a], index[b]
        wins[i] += w
        wins[j] += n - w
        games[i, j] += n
        games[j, i] += n
    p = np.ones(k)
    for _ in range(iterations):
        denom = (games / (p[:, None] + p[None, :])).sum(1) + 1. / (p + 1.)
        new = wins / denom
        new /= np.exp(np.log(new).mean())
        if np.max(np.abs(np.log(new) - np.log(p))) < 1e-10:
            p = new
            break
        p = new
    elo = 400. * np.log10(p)
    if anchor is not None:
        elo -= elo[index[anchor]]
    return {n: float(elo[index[n]]) for n in names}


def summary(records, anchor=None):
    """Per entrant: games, mean rank reward, first places, mean place (1-4, ties averaged); pairwise matrix;
    Bradley-Terry ratings; and cap / map statistics."""
    per = {}
    for rec in records:
        rewards = np.asarray(rec['rewards'])
        for seat, name in enumerate(rec['seat_names']):
            r = rewards[seat]
            # place = 1 + seats strictly ahead + half the other seats tied
            place = 1 + float((rewards > r).sum()) + .5 * float((rewards == r).sum() - 1)
            row = per.setdefault(name, dict(games=0, reward=0., first=0, place=0., by_seat=[0.] * PLAYERS,
                                            seat_games=[0] * PLAYERS))
            row['games'] += 1
            row['reward'] += float(r)
            row['first'] += int(r == 1.)
            row['place'] += place
            row['by_seat'][seat] += float(r)
            row['seat_games'][seat] += 1
    table = {}
    for name, row in per.items():
        n = row['games']
        table[name] = dict(games=n, mean_reward=row['reward'] / n, first_rate=row['first'] / n,
                           mean_place=row['place'] / n,
                           reward_by_seat=[s / max(g, 1) for s, g in zip(row['by_seat'], row['seat_games'])])
    pairs = {f'{a} vs {b}': dict(score=w / n, games=n) for (a, b), (w, n) in sorted(pairwise(records).items())}
    capped = sum(not r['terminated'] for r in records)
    return dict(games=len(records), capped=capped, cap_rate=capped / max(len(records), 1),
                mean_turns=float(np.mean([r['turns'] for r in records])) if records else None,
                entrants=table, pairwise=pairs, ratings=fit_ratings(records, anchor) if records else {})


def score_interval(w, n, z=1.96):
    """Wilson interval of a pairwise score (ties as halves)."""
    if n == 0:
        return (0., 1.)
    phat = w / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)
