"""Fog-of-war value-head check, no Modal (port of experiments/modal_distributed_rl.py head_check_fn).

    python head_check.py --policy RUN/final.npz --head RUN-foghead/final.npz --references-json '{"c3": ..., "champ": ...}'
                         [--games 32] [--seed-base 397000] [--max-ticks 4000] --out OUT.json

1) Which parameters differ between `policy` and `head`: only critic_hidden / value may (policy_unchanged).
2) How well each value head (`head` plus the references, name -> checkpoint) predicts the acting seat's result
   along `policy`'s sampled self-play games on proposed-v1 maps (decided games only): rank AUC of the value for
   wins vs losses, sign accuracy, and the mean value in wins and losses, overall and by game turn.
Reference points: champion fog head 0.975 vs c3 0.963; bn4 fog head 0.939 vs c3 0.943; an untrained head ~0.72.
Runs on the CPU (JAX_PLATFORMS=cpu); 32 games take a few minutes on one core group.
"""
import argparse
import json
import os
from pathlib import Path


def head_check(policy, head, references, games, seed_base, max_ticks):
    import jax
    import numpy as np
    from polytopia_bot.rl import network
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.rl.distributed import checkpoint_template
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.maps.stream import MapSource
    load = lambda p: load_checkpoint(p, checkpoint_template(p))
    base, trained = load(policy), load(head)
    changed = {}
    for (path, x), (_, y) in zip(jax.tree_util.tree_flatten_with_path(base)[0],
                                 jax.tree_util.tree_flatten_with_path(trained)[0]):
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            changed[jax.tree_util.keystr(path)] = float(np.abs(np.asarray(x) - np.asarray(y)).max())
    heads = {'head': trained, **{name: load(p) for name, p in references.items()}}
    value = jax.jit(jax.vmap(lambda p, x: network.trunk(p, x)[2], in_axes=(None, 0)))
    act = jax.jit(jax.vmap(lambda x, m, k: network.sample(base, x, m, k, size=14)[0], in_axes=(0, 0, 0)))
    engine, source = RustGames(14, games, map_batch=True), MapSource('proposed-v1', 14)
    states, prepared = engine.reset(source.batch(range(seed_base, seed_base + games))[0])
    lanes, seats, turns, values = [], [], [], {n: [] for n in heads}
    outcome, finished = np.zeros((games, 2), np.float32), np.zeros(games, bool)
    key = jax.random.key(seed_base)
    for _ in range(max_ticks):
        live = np.nonzero(~finished)[0]
        if not len(live):
            break
        for n, p in heads.items():
            values[n].append(np.asarray(value(p, prepared.features))[live])
        lanes.append(live)
        seats.append(np.asarray(states.current_player)[live])
        turns.append(np.asarray(states.turn)[live])
        key, sub = jax.random.split(key)
        choice = act(prepared.features, prepared.masks, jax.random.split(sub, games))
        tick, status = engine.tick(states, choice, source.batch(range(seed_base + games, seed_base + 2 * games))[0])
        assert not np.any(np.asarray(status)[live]), 'illegal action'
        done, truncated = np.asarray(tick.done), np.asarray(tick.truncated)
        for i in np.nonzero(done & ~finished)[0]:
            finished[i] = True
            outcome[i] = 0. if truncated[i] else np.asarray(tick.rewards)[i]
        states, prepared = tick.states, tick.prepared
    lanes, seats, turns = np.concatenate(lanes), np.concatenate(seats), np.concatenate(turns)
    result = outcome[lanes, seats - 1]
    keep = finished[lanes] & (result != 0)

    def score(v, mask):
        win, loss = v[mask & (result > 0)], v[mask & (result < 0)]
        if not len(win) or not len(loss):
            return None
        order = np.argsort(np.concatenate([win, loss]), kind='stable')
        ranks = np.empty(len(order))
        ranks[order] = np.arange(1, len(order) + 1)
        auc = (ranks[:len(win)].sum() - len(win) * (len(win) + 1) / 2) / (len(win) * len(loss))
        return dict(auc=round(float(auc), 4), sign_accuracy=round(float(np.mean(np.sign(v[mask]) == result[mask])), 4),
                    mean_win=round(float(win.mean()), 4), mean_loss=round(float(loss.mean()), 4), n=int(mask.sum()))

    phases = dict(all=keep, turns_0_9=keep & (turns < 10), turns_10_19=keep & (turns >= 10) & (turns < 20),
                  turns_20_plus=keep & (turns >= 20))
    return dict(policy=policy, head=head, changed_parameters=changed,
                policy_unchanged=all(('critic_hidden' in k or "'value'" in k) for k in changed),
                games=games, finished=int(finished.sum()), decided=int(np.sum(finished & (outcome[:, 0] != 0))),
                positions=int(keep.sum()),
                heads={n: {ph: score(np.concatenate(v), m) for ph, m in phases.items()} for n, v in values.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--head', required=True)
    ap.add_argument('--references-json', default='{}')
    ap.add_argument('--games', type=int, default=32)
    ap.add_argument('--seed-base', type=int, default=397000)
    ap.add_argument('--max-ticks', type=int, default=4000)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    out = head_check(args.policy, args.head, json.loads(args.references_json), args.games, args.seed_base,
                     args.max_ticks)
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
