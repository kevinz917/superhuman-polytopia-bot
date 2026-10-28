"""Per-phase timings of several engine-rs builds on the same warmed states, interleaved, one pinned core
(experiments/modal_rust_ab.py::bench). Shape of scripts/bench_rust_engine.py's Rust half.

The first revision warms `games` games with `warm` ticks of random legal play (mean turn ~46 at 400 on 14x14);
every revision then imports that state. Each metric is the median of `repeats` calls, the revisions taking
turns call by call; one-shot metrics restore the warmed state (import_, untimed) before every call. `play_*` is
the median per-tick time over `repeats` consecutive ticks of random play from the warmed state (the steady state
an actor sees, including any cross-tick caching); every revision plays the same choices, the revisions taking turns
tick by tick, over three passes of the sequence.

    python scripts/bench_rust_ab.py --revs base,work --size 14
"""
import argparse
import json
import os
import statistics
import time

import numpy as np

from rust_ab_common import load_rev, rules_json, sample


def has_kw(fn, kw):
    return kw in (getattr(fn, '__text_signature__', None) or '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--revs', default='base,work')
    ap.add_argument('--size', type=int, default=14)
    ap.add_argument('--games', type=int, default=24)
    ap.add_argument('--warm', type=int, default=400)
    ap.add_argument('--repeats', type=int, default=25)
    args = ap.parse_args()
    if os.environ.get('PIN_CORE'):
        os.sched_setaffinity(0, {int(os.environ['PIN_CORE'])})
    from polytopia_bot.gpu import ENGINE_REVISION
    size, games, reps = args.size, args.games, args.repeats
    names = args.revs.split(',')
    mods = {n: load_rev(n) for n in names}
    eng = {n: m.GameBatch(size, games, rules_json(), ENGINE_REVISION, None, True, 10_000) for n, m in mods.items()}
    rng = np.random.default_rng(0)
    seeds = np.arange(games, dtype=np.int32) + 500_000
    w = eng[names[0]]
    w.reset(seeds)
    masks = w.legal_masks()
    for _ in range(args.warm):
        masks = w.tick(*sample(masks, size, rng), seeds, True)['masks']
    saved = w.export()
    f, s, t = sample(masks, size, rng)
    mw = masks.shape[1]
    res = dict(size=size, games=games, warm=args.warm, repeats=reps,
               mean_turn=float(np.asarray(saved['turn']).mean()),
               mean_units=float((np.asarray(saved['units.owner']) > 0).sum(1).mean()), revs={})
    # the play sequence: choices drawn once (from the first revision's masks) and replayed by every revision
    w.import_(saved)
    play, m = [], masks
    for _ in range(reps):
        c = sample(m, size, rng)
        play.append(c)
        m = w.tick(*c, seeds, True)['masks']

    metrics = {
        'step': lambda b: b.step(f, s, t),
        'masks': lambda b: b.legal_masks(),
        'masks_packed': lambda b: b.legal_masks(packed=True),
        'observe_both': lambda b: (b.observe(), b.observe_opponent()),
        'tick': lambda b: b.tick(f, s, t, seeds, True),
        'tick_one_view': lambda b: b.tick(f, s, t, seeds, False),
        'tick_packed': lambda b: b.tick(f, s, t, seeds, True, None, True),
        'export': lambda b: b.export(),
    }
    need = {'masks_packed': ('legal_masks', 'packed'), 'tick_packed': ('tick', 'packed_masks')}
    out = {n: {} for n in names}
    for metric, fn in metrics.items():
        samples = {n: [] for n in names}
        live = [n for n in names if metric not in need or has_kw(getattr(eng[n], need[metric][0]), need[metric][1])]
        for _ in range(reps):
            for n in live:
                eng[n].import_(saved)
                a = time.perf_counter()
                fn(eng[n])
                samples[n].append(1e3 * (time.perf_counter() - a))
        for n in live:
            out[n][metric] = round(statistics.median(samples[n]), 4)
    # play: the same choices for every revision, revisions taking turns tick by tick, three passes over the sequence
    for mode in ('dense', 'packed'):
        samples = {n: [] for n in names}
        live = [n for n in names if mode == 'dense' or has_kw(eng[n].tick, 'packed_masks')]
        for _ in range(3):
            for n in live:
                eng[n].import_(saved)
            for c in play:
                for n in live:
                    a = time.perf_counter()
                    eng[n].tick(*c, seeds, True) if mode == 'dense' else eng[n].tick(*c, seeds, True, None, True)
                    samples[n].append(1e3 * (time.perf_counter() - a))
        for n in live:
            out[n][f'play_{mode}'] = round(statistics.median(samples[n]), 4)
    # per-family mask profile: five interleaved rounds, median per family
    prof = {n: [] for n in names}
    for _ in range(5):
        for n in names:
            eng[n].import_(saved)
            prof[n].append(eng[n].profile_masks(max(reps // 5, 5)))
    for n in names:
        out[n]['profile'] = {k: round(statistics.median(p[k] for p in prof[n]), 4) for k in prof[n][0]}
        if hasattr(eng[n], 'profile_phases'):
            # phase totals over the play sequence, ms per tick (calls = per-game calls)
            eng[n].import_(saved)
            eng[n].profile_phases(True)
            for c in play:
                eng[n].tick(*c, seeds, True)
            ph = eng[n].phase_profile()
            eng[n].profile_phases(False)
            out[n]['phases'] = {k: (int(calls), round(ms / len(play), 4)) for k, (calls, ms) in sorted(ph.items())}
    res['revs'] = out
    res['mask_width'] = mw
    print(json.dumps(res))


if __name__ == '__main__':
    main()
