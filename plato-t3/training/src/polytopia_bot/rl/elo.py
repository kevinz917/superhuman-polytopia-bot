"""Bradley–Terry ratings for a round-robin of paired-seat checkpoint matches.

Input rows are single games: dict(a, b, seed, seat, score) with score the points
of `a` (1 win, 0 loss). NO-SCORE RULE: a game that reaches the round cap is a loss for every
participant, so a capped game (score None, or capped=True) counts as a game played with no points for either side. Strengths are fitted with
Hunter's MM algorithm; every played pair gets `prior` virtual draws so an
undefeated player keeps a finite rating. Ratings are on the Elo scale
(400·log10 strength) relative to `anchor`. Confidence intervals resample map
seeds with replacement: every pair plays the same maps, so a map that favours
one style moves all of that map's games together.
"""
import numpy as np

ELO = 400 / np.log(10)


def tables(rows, players, decisive_only=False):
    index = {p: i for i, p in enumerate(players)}
    n = len(players)
    points, games = np.zeros((n, n)), np.zeros((n, n))
    for r in rows:
        capped = r.get('capped') or r['score'] is None or r['score'] == .5
        if decisive_only and capped:
            continue
        i, j = index[r['a']], index[r['b']]
        if not capped:
            points[i, j] += r['score']
            points[j, i] += 1 - r['score']
        games[i, j] += 1
        games[j, i] += 1
    return points, games


def fit(points, games, prior=.5, iterations=20000, tol=1e-12):
    played = games > 0
    w = points + prior * played
    g = games + 2 * prior * played
    strength = np.ones(len(points))
    for _ in range(iterations):
        denominator = (g / (strength[:, None] + strength[None, :])).sum(axis=1)
        updated = w.sum(axis=1) / denominator
        updated /= np.exp(np.log(updated).mean())
        if np.max(np.abs(np.log(updated) - np.log(strength))) < tol:
            strength = updated
            break
        strength = updated
    return np.log(strength)


def ratings(rows, players, anchor, *, resamples=2000, seed=0, prior=.5, decisive_only=False):
    a = players.index(anchor)
    point = fit(*tables(rows, players, decisive_only), prior=prior)
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r['seed'], []).append(r)
    seeds = sorted(by_seed)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        sample = [r for s in rng.choice(seeds, len(seeds)) for r in by_seed[s]]
        draws.append(fit(*tables(sample, players, decisive_only), prior=prior))
    draws = np.asarray(draws)
    elo = (point - point[a]) * ELO
    rel = (draws - draws[:, [a]]) * ELO
    low, high = np.percentile(rel, [2.5, 97.5], axis=0)
    rank = np.argsort(np.argsort(-draws, axis=1), axis=1) + 1
    order = np.argsort(-elo)
    return [dict(player=players[i], elo=round(float(elo[i]), 1), ci95=[round(float(low[i]), 1), round(float(high[i]), 1)],
                 rank_ci95=[int(np.percentile(rank[:, i], 2.5)), int(np.percentile(rank[:, i], 97.5))])
            for i in order]


def pair_table(rows, players):
    out = []
    for x in range(len(players)):
        for y in range(x + 1, len(players)):
            a, b = players[x], players[y]
            games = [r for r in rows if {r['a'], r['b']} == {a, b}]
            cap = lambda r: r.get('capped') or r['score'] is None or r['score'] == .5
            score = [.5 if cap(r) else (r['score'] if r['a'] == a else 1 - r['score']) for r in games]   # .5 marks a cap (a loss for both) below
            truncations = sum(s == .5 for s in score)
            out.append(dict(a=a, b=b, games=len(games), a_wins=sum(s == 1 for s in score),
                            b_wins=sum(s == 0 for s in score), truncations=truncations,
                            truncation_rate=round(truncations / max(len(games), 1), 4),
                            a_score=round(sum(x == 1 for x in score) / max(len(games), 1), 4)))   # caps score 0 for a
    return out
