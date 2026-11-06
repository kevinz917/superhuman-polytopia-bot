"""Where a distributed RL run's time went, from the learner's per-update training.json.

    python scripts/analyze_learner_bottlenecks.py <training.json> [--window 900] [--split 18900]

Prints (1) a table per time window: consumed samples/s (batch x updates over wall time), actor steps/s, connections,
steps/s per connection, ready_wait / asm_wait / inbox, train, assembler work (asm_total - asm_wait), the gap per
update outside `seconds` (status callback, logging, checkpoints), decisions per self-play game, decided share;
(2) medians of every timing field before and after --split seconds of learner clock; (3) the checkpoint stall
(gap after every checkpoint_every-th update) by update range. Read-only; runs on the JSON alone."""
import json
import statistics as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from polytopia_bot.rl.training_log import read as read_log  # noqa: E402  training.json or training.jsonl

BATCH = 262144


def windows(d, width):
    out = {}
    for r in d:
        out.setdefault(int(r['elapsed'] // width), []).append(r)
    print('t(min) upd consumed/s steps/s conns steps/s/conn ready_wait asm_wait inbox train asm_work gap/upd '
          'dec/game decided%')
    for k in sorted(out):
        rs = out[k]
        dt = rs[-1]['elapsed'] - rs[0]['elapsed'] + rs[0]['seconds']
        steps = sum(r['actors']['steps'] for r in rs) / dt
        conns = [r['connections'] for r in rs]
        games = sum(r['actors']['selfplay_games'] for r in rs)
        trunc = sum(r['actors']['selfplay_truncations'] for r in rs)
        dpg = sum(r['actors']['learner_decisions'] for r in rs) / max(games, 1)
        gap = (dt - sum(r['seconds'] for r in rs)) / len(rs)
        work = st.mean(r['asm_total_seconds'] - r['asm_wait_seconds'] for r in rs)
        m = lambda key: st.mean(r[key] for r in rs)
        print(f"{k * width / 60:5.0f} {len(rs):4d} {len(rs) * BATCH / dt / 1e3:7.1f}k {steps / 1e3:6.1f}k "
              f"{min(conns):3d}-{max(conns):3d} {steps / st.mean(conns):5.0f} {m('ready_wait_seconds'):6.2f} "
              f"{m('asm_wait_seconds'):6.2f} {m('inbox'):6.1f} {m('train_seconds'):5.2f} {work:5.2f} {gap:5.3f} "
              f"{dpg:6.0f} {100 * (1 - trunc / max(games, 1)):5.1f}")


def phases(d, split):
    keys = ['seconds', 'train_seconds', 'ready_wait_seconds', 'device_batch_seconds', 'asm_total_seconds',
            'asm_wait_seconds', 'asm_stream_seconds', 'asm_views_seconds', 'asm_value_seconds', 'asm_gae_seconds',
            'asm_concat_seconds', 'asm_normalize_seconds', 'receiver_decompress_seconds', 'receiver_unpickle_seconds',
            'batch_age_seconds', 'inbox', 'stale_dropped', 'mean_lag', 'messages']
    early = [r for r in d if 600 < r['elapsed'] < split]
    late = [r for r in d if r['elapsed'] >= split]
    print(f'\nmedians: early (600 s .. {split} s, {len(early)} updates) | late ({len(late)} updates)')
    for k in keys:
        a = st.median(r[k] for r in early) if early else float('nan')
        b = st.median(r[k] for r in late) if late else float('nan')
        print(f'  {k:30s} {a:9.3f} | {b:9.3f}')


def checkpoints(d, every=50):
    gap = {d[i]['update']: d[i]['elapsed'] - d[i - 1]['elapsed'] - d[i]['seconds'] for i in range(1, len(d))}
    other = [g for u, g in gap.items() if (u - 1) % every]
    print(f'\ngap per update outside `seconds`: median {st.median(other):.3f} s (non-checkpoint updates)')
    last = max(gap)
    edges = [1, 1000, 3000, 5000, 7000, last + 1]
    for lo, hi in zip(edges, edges[1:]):
        ck = [g for u, g in gap.items() if lo <= u < hi and (u - 1) % every == 0]
        if ck:
            print(f'  checkpoint stall, updates {lo}-{hi - 1}: median {st.median(ck):.2f} s over {len(ck)} checkpoints')
    total = sum(gap.values())
    print(f'  total time outside `seconds`: {total:.0f} s of {d[-1]["elapsed"]:.0f} s '
          f'({100 * total / d[-1]["elapsed"]:.1f}%)')


def main():
    path = sys.argv[1]
    width = float(sys.argv[sys.argv.index('--window') + 1]) if '--window' in sys.argv else 900.
    split = float(sys.argv[sys.argv.index('--split') + 1]) if '--split' in sys.argv else 18900.
    d = read_log(path)
    windows(d, width)
    phases(d, split)
    checkpoints(d)


if __name__ == '__main__':
    main()
