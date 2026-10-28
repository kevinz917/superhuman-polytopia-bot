"""Training curves of a distributed run from its training.json, per window of updates (pure Python).

    python scripts/analyze_c5_training.py runs/c5/c5-omni-s1-training.json [--window 50] [--hours 2] [--opponent c3]

Per window: sampled record vs the frozen opponent named --opponent (the `frozen` entry's name, default c3; W-L-cap, win share of decisive games), mean policy
entropy, critic value std, self-play turn-cap rate, warrior share of recruits, samples/s. --hours prints the
kill-rule check (decisive win share vs c3 over the last N hours; docs/c5-omni-long: stop an arm under 45% over
its 2-h window). Rows are the learner's `update` rows (polytopia_bot.rl.distributed.Learner.run).
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from polytopia_bot.rl.training_log import read as read_log  # noqa: E402  training.json or training.jsonl

OPPONENT = 'c3'


def windows(rows, window):
    out = []
    for start in range(0, (rows[-1]['update'] // window + 1) * window, window):
        part = [r for r in rows if start < r['update'] <= start + window]
        if part:
            out.append((start + 1, part[-1]['update'], part))
    return out


def summarize(part):
    w = l = t = 0
    games = caps = 0
    for r in part:
        a = r.get('actors', {})
        rec = a.get('vs', {}).get(OPPONENT)
        if rec:
            w, l, t = w + rec[0], l + rec[1], t + rec[2]
        games += a.get('selfplay_games', 0)
        caps += a.get('selfplay_truncations', 0)
    mean = lambda k: sum(r[k] for r in part) / len(part)
    ws = [r['actors'].get('warrior_share') for r in part if r.get('actors', {}).get('warrior_share') is not None]
    return dict(w=w, l=l, t=t, decisive=w / (w + l) if w + l else None, entropy=mean('entropy'),
                value_std=mean('value_std'), approx_kl=mean('approx_kl'), cap_rate=caps / games if games else None,
                warrior=sum(ws) / len(ws) if ws else None, sps=mean('received_samples_per_second'),
                hours=part[-1]['elapsed'] / 3600, samples=part[-1]['samples'])


def main():
    global OPPONENT
    path = sys.argv[1]
    OPPONENT = next((sys.argv[sys.argv.index(f) + 1] for f in ('--opponent', '--frozen') if f in sys.argv), 'c3')
    window = int(sys.argv[sys.argv.index('--window') + 1]) if '--window' in sys.argv else 50
    hours = float(sys.argv[sys.argv.index('--hours') + 1]) if '--hours' in sys.argv else 2.
    rows = read_log(path)
    print(f"{path}: {len(rows)} updates, {rows[-1]['elapsed'] / 3600:.2f} h, {rows[-1]['samples'] / 1e6:.0f}M samples")
    print(f"{'updates':>10s} {'hours':>5s} {'vs ' + OPPONENT + ' W-L-cap':>14s} {'dec%':>5s} {'entropy':>7s} {'vstd':>5s} {'kl':>6s} "
          f"{'cap%':>5s} {'warrior':>7s} {'samples/s':>9s}")
    for lo, hi, part in windows(rows, window):
        s = summarize(part)
        pct = lambda x: f'{100 * x:4.0f}%' if x is not None else '    -'
        print(f"{lo:4d}-{hi:4d} {s['hours']:5.2f} {s['w']:5d}-{s['l']:4d}-{s['t']:3d} {pct(s['decisive'])} {s['entropy']:7.3f} "
              f"{s['value_std']:5.3f} {s['approx_kl']:6.4f} {pct(s['cap_rate'])} {pct(s['warrior'])} {s['sps']:9.0f}")
    recent = [r for r in rows if r['elapsed'] > rows[-1]['elapsed'] - hours * 3600]
    s = summarize(recent)
    print(f"last {hours:g} h (updates {recent[0]['update']}-{recent[-1]['update']}): vs {OPPONENT} {s['w']}-{s['l']}-{s['t']}, "
          f"decisive {100 * s['decisive']:.1f}%" if s['decisive'] is not None else f'no vs-{OPPONENT} games yet')


if __name__ == '__main__':
    main()
