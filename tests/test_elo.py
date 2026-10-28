"""Bradley–Terry fit on synthetic round-robins with known strengths."""
import numpy as np

from polytopia_bot.rl.elo import ELO, fit, pair_table, ratings, tables


def synthetic(true_elo, maps=64, seed=0, draw=0.):
    rng = np.random.default_rng(seed)
    players = list(true_elo)
    rows = []
    for x, a in enumerate(players):
        for b in players[x + 1:]:
            p = 1 / (1 + 10 ** ((true_elo[b] - true_elo[a]) / 400))
            for s in range(maps):
                for seat in (1, 2):
                    u = rng.random()
                    score = .5 if u < draw else float(rng.random() < p)
                    rows.append(dict(a=a, b=b, seed=300000 + s, seat=seat, score=score))
    return players, rows


def test_mm_fit_recovers_exact_two_player_odds():
    points = np.array([[0., 48.], [16., 0.]])
    games = np.array([[0., 64.], [64., 0.]])
    s = fit(points, games, prior=0)
    assert abs((s[0] - s[1]) * ELO - 400 * np.log10(3)) < 1e-6


def test_ratings_recover_order_and_scale_with_anchor_at_zero():
    true = dict(a=0., b=150., c=300., d=-200.)
    players, rows = synthetic(true, maps=200)
    out = {r['player']: r for r in ratings(rows, players, 'a', resamples=200)}
    assert out['a']['elo'] == 0 and out['a']['ci95'] == [0., 0.]
    assert [r['player'] for r in ratings(rows, players, 'a', resamples=10)] == ['c', 'b', 'a', 'd']
    for name, value in true.items():
        assert abs(out[name]['elo'] - value) < 60
        assert out[name]['ci95'][0] - 1e-9 <= out[name]['elo'] <= out[name]['ci95'][1] + 1e-9


def test_undefeated_player_stays_finite_and_draws_count_half():
    rows = [dict(a='x', b='y', seed=s, seat=1, score=1.) for s in range(10)]
    rows += [dict(a='y', b='z', seed=s, seat=1, score=.5) for s in range(10)]
    out = {r['player']: r['elo'] for r in ratings(rows, ['x', 'y', 'z'], 'y', resamples=20)}
    assert np.isfinite(out['x']) and out['x'] > 300 and abs(out['z']) < 1e-6
    points, games = tables(rows, ['x', 'y', 'z'], decisive_only=True)
    assert games[1, 2] == 0 and points[0, 1] == 10


def test_pair_table_reports_truncations_from_either_side():
    rows = [dict(a='x', b='y', seed=1, seat=1, score=1.), dict(a='y', b='x', seed=1, seat=2, score=.5),
            dict(a='y', b='x', seed=2, seat=1, score=1.)]
    (row,) = pair_table(rows, ['x', 'y'])
    assert (row['a_wins'], row['b_wins'], row['truncations'], row['games']) == (1, 1, 1, 3)
    assert row['truncation_rate'] == round(1 / 3, 4) and row['a_score'] == .5
