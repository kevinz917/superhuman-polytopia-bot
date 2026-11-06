"""Training-curve readout of a distributed run's training.json in 50-update windows, with the UPGO statistics.

    python scripts/analyze_c5_upgo.py runs/c5-upgo/training.json [--window 50] [--kill-hours 2]

Per window: sampled record vs frozen c3 (W-L-cap, % of decisive), self-play cap rate, entropy, critic value std,
raw GAE / UPGO advantage std, fraction of UPGO returns that followed the trajectory, UPGO-GAE correlation.
--kill-hours prints the vs-c3 decisive rate over the last N hours (the pre-registered kill rule: < 45% -> stop).
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from polytopia_bot.rl.training_log import read as read_log  # noqa: E402  training.json or training.jsonl


def windows(history, size):
    for lo in range(0, len(history), size):
        rows = history[lo:lo + size]
        yield rows[0]['update'], rows[-1]['update'], rows


def mean(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else float('nan')


def record(rows, name='c3'):
    w = l = c = 0
    for r in rows:
        rec = r.get('actors', {}).get('vs', {}).get(name)
        if rec:
            w, l, c = w + rec[0], l + rec[1], c + rec[2]
    return w, l, c


def cap_rate(rows):
    games = sum(r['actors'].get('selfplay_games', 0) for r in rows)
    caps = sum(r['actors'].get('selfplay_truncations', 0) for r in rows)
    return caps / games if games else float('nan')


def main():
    path = sys.argv[1]
    size = int(sys.argv[sys.argv.index('--window') + 1]) if '--window' in sys.argv else 50
    kill_hours = float(sys.argv[sys.argv.index('--kill-hours') + 1]) if '--kill-hours' in sys.argv else None
    history = read_log(path)
    upgo = any('raw_upgo_std' in r for r in history)
    head = ['updates', 'samples(M)', 'hours', 'vs c3 W-L-cap', 'decisive%', 'cap%', 'entropy', 'value_std', 'adv_std']
    if upgo:
        head += ['upgo_std', 'follow%', 'upgo>0%', 'corr', 'upgo_loss']
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for lo, hi, rows in windows(history, size):
        w, l, c = record(rows)
        cells = [f'{lo}-{hi}', f'{rows[-1]["samples"] / 1e6:.0f}', f'{rows[-1]["elapsed"] / 3600:.2f}', f'{w}-{l}-{c}',
                 f'{100 * w / max(w + l, 1):.0f}', f'{100 * cap_rate(rows):.0f}', f'{mean(rows, "entropy"):.2f}',
                 f'{mean(rows, "value_std"):.3f}', f'{mean(rows, "raw_advantage_std"):.3f}']
        if upgo:
            cells += [f'{mean(rows, "raw_upgo_std"):.3f}', f'{100 * mean(rows, "upgo_follow_fraction"):.0f}',
                      f'{100 * mean(rows, "upgo_positive_fraction"):.0f}', f'{mean(rows, "upgo_gae_corr"):.2f}',
                      f'{mean(rows, "upgo_loss"):.4f}']
        print('| ' + ' | '.join(cells) + ' |')
    last = history[-1]
    print(f'\ntotal: {last["update"]} updates, {last["samples"] / 1e6:.0f}M samples, {last["elapsed"] / 3600:.2f} h, '
          f'vs c3 overall {record(history)}')
    if kill_hours:
        rows = [r for r in history if r['elapsed'] >= last['elapsed'] - 3600 * kill_hours]
        w, l, c = record(rows)
        print(f'kill rule window (last {kill_hours:g} h, updates {rows[0]["update"]}-{last["update"]}): '
              f'{w}-{l}-{c} = {100 * w / max(w + l, 1):.1f}% of decisive (stop below 45%)')


if __name__ == '__main__':
    main()
