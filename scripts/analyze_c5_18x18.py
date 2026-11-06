"""Curves and ladder tables for the c5-18x18 arm (plain Python, no numpy: the Mac has none).

    python3 scripts/analyze_c5_18x18.py curves <training.json> [window=50]
    python3 scripts/analyze_c5_18x18.py ladder <arena-with-details.json> [...]

`curves` prints per-window training-time records vs the frozen parent (summing the per-update actor stats of each
learner row), the self-play cap rate, critic value std, entropy and throughput. `ladder` prints W-L-cap, win rate
with a 95% Wilson interval, decisive share, mean turns, cap rate, per map type and the seat split of a frozen-engine
arena file (needs `details`, i.e. the /results/distributed/arena/<name>.json copy, not the summary)."""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from polytopia_bot.rl.training_log import read as read_log  # noqa: E402  training.json or training.jsonl


def wilson(w, n, z=1.96):
    if n == 0:
        return 0., 0., 0.
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def pct(x):
    return f'{100 * x:.0f}%'


def curves(path, window=50):
    rows = read_log(path)
    print(f'updates {rows[-1]["update"]}, samples {rows[-1]["samples"] / 1e6:.0f}M, elapsed {rows[-1]["elapsed"] / 60:.0f} min, '
          f'mean {sum(r["seconds"] for r in rows) / len(rows):.1f} s/update '
          f'(train {sum(r["train_seconds"] for r in rows) / len(rows):.1f} s)')
    print('| updates | vs frozen parent W–L–cap (sampled) | % decisive | self-play games | cap rate | value std | '
          'value mean | entropy | approx KL | samples/s |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for start in range(0, rows[-1]['update'], window):
        block = [r for r in rows if start < r['update'] <= start + window]
        if not block:
            continue
        # each row's `actors` holds the actor reports merged since the previous update (not cumulative)
        vs, games, caps = {}, 0, 0
        for r in block:
            a = r['actors']
            games += a['selfplay_games']
            caps += a['selfplay_truncations']
            for name, rec in a.get('vs', {}).items():
                agg = vs.setdefault(name, [0, 0, 0])
                for j in range(3):
                    agg[j] += rec[j]
        rec = next(iter(vs.values()), [0, 0, 0])
        dec = rec[0] + rec[1]
        mean = lambda k: sum(r.get(k, 0.) for r in block) / len(block)
        print(f'| {block[0]["update"]}–{block[-1]["update"]} | {rec[0]}–{rec[1]}–{rec[2]} | '
              f'{pct(rec[0] / dec) if dec else "–"} | {games} | {pct(caps / games) if games else "–"} | '
              f'{mean("value_std"):.3f} | {mean("value_mean"):+.3f} | {mean("entropy"):.2f} | '
              f'{mean("approx_kl"):.4f} | {mean("received_samples_per_second") / 1e3:.1f}k |')


def ladder(paths):
    for path in paths:
        a = json.load(open(path))
        rows = a.get('details')
        if not rows:
            print(f'{path}: no details (summary only)')
            continue
        n = len(rows)
        w = sum(r['reward'] > 0 for r in rows)
        l = sum(r['reward'] < 0 for r in rows)
        cap = sum(r['truncated'] for r in rows)
        p, lo, hi = wilson(w, n)
        turns = sum(r.get('turn', 0) for r in rows) / n
        print(f'\n### {a.get("name", path)}\ncandidate {a.get("candidate")}\nopponent {a.get("opponent")}\n'
              f'seeds {a.get("seed_base")}+{a.get("maps")}, size {a.get("size")}, {a.get("map_mix")}, '
              f'deterministic {a.get("deterministic")}/{a.get("opponent_deterministic")}, {a.get("engine")}')
        print(f'W–L–cap {w}–{l}–{n - w - l} (cap {cap}); win rate {pct(p)} [{pct(lo)}, {pct(hi)}] of all games; '
              f'{pct(w / max(w + l, 1))} of decisive [{pct(wilson(w, w + l)[1])}, {pct(wilson(w, w + l)[2])}]; '
              f'cap rate {pct(cap / n)}; mean turns {turns:.0f}')
        print('| seat | games | W–L–cap | win rate (all) | of decisive |')
        print('|---|---|---|---|---|')
        for seat in (1, 2):
            rs = [r for r in rows if r['seat'] == seat]
            ws = sum(r['reward'] > 0 for r in rs)
            ls = sum(r['reward'] < 0 for r in rs)
            ps, los, his = wilson(ws, len(rs))
            print(f'| candidate moves {"first" if seat == 1 else "second"} | {len(rs)} | {ws}–{ls}–{len(rs) - ws - ls} | '
                  f'{pct(ps)} [{pct(los)}, {pct(his)}] | {pct(ws / max(ws + ls, 1))} |')
        first = sum(1 for r in rows if (r['reward'] > 0) == (r['seat'] == 1) and r['reward'] != 0)
        print(f'first mover won {pct(first / max(w + l, 1))} of decisive games')
        by = defaultdict(list)
        for r in rows:
            by[r.get('map', a.get('map_mix'))].append(r)
        if len(by) > 1:
            print('| map type | games | W–L–cap | win rate | of decisive | mean turns |')
            print('|---|---|---|---|---|---|')
            for t, rs in sorted(by.items()):
                ws = sum(r['reward'] > 0 for r in rs)
                ls = sum(r['reward'] < 0 for r in rs)
                print(f'| {t} | {len(rs)} | {ws}–{ls}–{len(rs) - ws - ls} | {pct(ws / len(rs))} | '
                      f'{pct(ws / max(ws + ls, 1))} | {sum(r.get("turn", 0) for r in rs) / len(rs):.0f} |')


if __name__ == '__main__':
    if sys.argv[1] == 'curves':
        curves(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 50)
    else:
        ladder(sys.argv[2:])
