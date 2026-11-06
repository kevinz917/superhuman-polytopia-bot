"""What does a checkpoint's self-play look like? Plays N sampled self-play games on the CPU and reports, per game and
in total: rounds played, how it ended (win / cap), action counts per family, turns spent at peace, embassies, and
attacks. For diagnosing degenerate equilibria (e.g. both seats at peace until the round cap).

    python experiments/curio/selfplay_probe.py CKPT.npz [--games 8] [--seed 600000] [--max-turns 200] [--no-peace]
"""
import argparse
import collections
import json
import os

import numpy as np



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--games', type=int, default=8)
    ap.add_argument('--seed', type=int, default=600000)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--no-peace', action='store_true', help='the training option LearnerConfig.no_peace')
    args = ap.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import jax
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import checkpoint_template
    from polytopia_bot.rl.actions import FAMILIES
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.maps.stream import MapSource
    params = load_checkpoint(args.ckpt, checkpoint_template(args.ckpt))
    g = args.games
    engine, source = RustGames(14, g, map_batch=True, max_turns=args.max_turns, no_peace=args.no_peace), MapSource('proposed-v1', 14)
    states, prepared = engine.reset(source.batch(range(args.seed, args.seed + g))[0])
    act = jax.jit(jax.vmap(lambda x, m, k: network.sample(params, x, m, k, size=14)[0], in_axes=(0, 0, 0)))
    fam = collections.Counter()
    per = [collections.Counter() for _ in range(g)]
    peace_turns = np.zeros(g, int)
    done = np.zeros(g, bool)
    result = [None] * g
    last_turn = np.zeros(g, int)
    key = jax.random.key(args.seed)
    for _ in range(200000):
        live = ~done
        if not live.any():
            break
        key, sub = jax.random.split(key)
        choice = act(prepared.features, prepared.masks, jax.random.split(sub, g))
        f = np.asarray(choice.family)
        for i in np.nonzero(live)[0]:
            name = FAMILIES[int(f[i])] if int(f[i]) < len(FAMILIES) else str(int(f[i]))
            fam[name] += 1
            per[i][name] += 1
        ex = engine.batch.export()
        turn = np.asarray(ex['turn'])
        peace = np.asarray(ex['players.peace']).reshape(g, -1) if 'players.peace' in ex else None
        for i in np.nonzero(live)[0]:
            if turn[i] != last_turn[i]:
                last_turn[i] = turn[i]
                if peace is not None and peace[i].any():
                    peace_turns[i] += 1
        tick, status = engine.tick(states, choice, source.batch(range(args.seed + g, args.seed + 2 * g))[0])
        d, t = np.asarray(tick.done), np.asarray(tick.truncated)
        for i in np.nonzero(d & ~done)[0]:
            done[i] = True
            result[i] = dict(rounds=int(last_turn[i]), ended='cap' if t[i] else 'win',
                             reward=[float(x) for x in np.asarray(tick.rewards)[i]], peace_rounds=int(peace_turns[i]),
                             families=dict(per[i].most_common()))
        states, prepared = tick.states, tick.prepared
    total = sum(fam.values())
    print(json.dumps(dict(games=g, ended=collections.Counter(r['ended'] for r in result if r),
                          mean_rounds=float(np.mean([r['rounds'] for r in result if r])),
                          mean_peace_rounds=float(np.mean([r['peace_rounds'] for r in result if r])),
                          family_share={k: round(v / total, 4) for k, v in fam.most_common()}), indent=1))
    for i, r in enumerate(result):
        print(i, json.dumps(r))


if __name__ == '__main__':
    main()
