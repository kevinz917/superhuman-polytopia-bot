"""Cross-architecture head-to-head on the CUDA engine (engine-cu, host-pointer mode): one game per lane, paired maps.

    python gpu_arena.py '{"a": "/results/.../final.npz", "b": "random", "maps": 1024, "seed_base": 300000}'

Every map seed is played twice with the seats swapped (lane 2k: A is seat 1, lane 2k+1: A is seat 2); all games run at
once and each lane counts only its first game. Policies: a checkpoint path (any architecture network_spec loads,
sampled T=1 decisions by default as the trainer plays; deterministic argmax can stall early policies) or 'random' (uniform over the legal paths: family, then source, then
target). Outcome per game: a terminated game by the engine reward; a game truncated at the round limit by the score
at the cap (the scores BEFORE the final action, as the trainer's cap_score_reward books it), a tie is a draw.
Reports both the terminal-only and the score-at-cap results, paired-map score and a map-clustered 95% interval.
Map seeds 300000+ are outside the training table [200000, 300000). `no_peace: true` plays under the no_peace rule
(default false = engine default, peace allowed; every session E arena so far used the default).
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


class RustArena:
    """The same paired arena on engine-rs, without a CUDA library or torch."""

    def __init__(self, size, games, no_peace):
        from polytopia_engine_rs import GameBatch
        from polytopia_bot.gpu import ENGINE_REVISION
        from polytopia_bot.ruleset import rules_json, training_round_limit
        self.batch = GameBatch(size, games, rules_json(), ENGINE_REVISION, training_round_limit(size),
                               True, 10_000, 2, no_peace)
        self.mask_width = self.batch.mask_width

    def reset(self, seeds, layouts):
        self.batch.reset_layouts(seeds, *layouts)

    def observe(self):
        return self.batch.observe()

    def legal_masks(self, packed=True):
        return self.batch.legal_masks(packed=packed)

    def tick(self, family, source, target, seeds, layouts, both_views=False):
        args = [np.ascontiguousarray(x, np.int32) for x in (family, source, target, seeds)]
        out = dict(self.batch.tick(*args, both_views, layouts, True))
        out['scores'] = out['score']
        return out


def _policy(spec_or_path: str):
    from polytopia_bot.rl.distributed import _template_params, network_spec
    from polytopia_bot.rl.runner import load_checkpoint
    if spec_or_path == 'random':
        return None, 'random'
    spec = network_spec(spec_or_path)
    return load_checkpoint(spec_or_path, _template_params(**spec)), spec['architecture']


def _random_choice(masks, key, size):
    """Uniform legal path: family, source, target, each uniform over the legal entries given the prefix."""
    from polytopia_bot.rl.actions import family_mask, source_mask, target_mask
    k1, k2, k3 = jax.random.split(key, 3)
    pick = lambda m, k: jax.random.categorical(k, jnp.where(m, 0., -1e9)).astype(jnp.int32)
    f = pick(family_mask(masks), k1)
    s = pick(source_mask(masks, f, size), k2)
    return f, s, pick(target_mask(masks, f, s, size), k3)


def run(cfg: dict) -> dict:
    from polytopia_bot.gpu.actions import Masks
    from polytopia_bot.maps.stream import MapSource
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import leaf_shapes
    from polytopia_bot.rl.cpu_engine import view_features, view_layout
    from polytopia_bot.rl.gpu_engine import GpuGames
    size, maps, seed_base = int(cfg.get('size', 14)), int(cfg['maps']), int(cfg.get('seed_base', 300000))
    deterministic = bool(cfg.get('deterministic', False))
    G = 2 * maps
    layout = view_layout(size, 2)
    offsets, at = {}, 0
    for name, shape in layout:
        offsets[name] = at
        at += int(np.prod(shape))
    player_at = offsets['player']
    source = MapSource(cfg.get('mix', 'classic'), size, players=2)
    rows = [source.layout(seed_base + k)[1] for k in range(maps)]
    lay = [np.ascontiguousarray(np.repeat(np.stack([np.asarray(getattr(r, f), np.int32) for r in rows]), 2, axis=0))
           for f in ('terrain', 'structure', 'resource', 'capitals')]
    seeds = np.repeat(np.arange(seed_base, seed_base + maps, dtype=np.int32), 2)
    backend = cfg.get('backend', 'cuda')
    if backend not in ('rust', 'cpu', 'cuda'):
        raise ValueError('backend must be rust, cpu, or cuda')
    env = (RustArena(size, G, bool(cfg.get('no_peace', False))) if backend == 'rust' else
           GpuGames(size=size, games=G, players=2, no_resign=True, no_peace=bool(cfg.get('no_peace', False)),
                    max_actions=10_000, device=-1 if backend == 'cpu' else int(cfg.get('device', 0))))
    env.reset(seeds, tuple(lay))
    a_seat = np.where(np.arange(G) % 2 == 0, 1, 2)
    params = {k: _policy(cfg[k]) for k in ('a', 'b')}
    mask_width = env.mask_width
    shapes = leaf_shapes(size)

    def dense(packed):
        bits = jnp.unpackbits(packed, axis=1)[:, :mask_width].astype(bool)
        leaves, at = [], 0
        for shape in shapes:
            k = int(np.prod(shape)) if shape else 1
            leaves.append(bits[:, at:at + k].reshape((bits.shape[0],) + tuple(shape)))
            at += k
        return Masks(*leaves)

    @jax.jit
    def decide(p_a, p_b, view, packed, use_a, key):
        features = jax.vmap(lambda v: view_features(v, layout, size))(view)
        masks = dense(packed)
        keys = jax.random.split(key, view.shape[0])
        out = []
        for p in (p_a, p_b):
            if p is None:
                out.append(jax.vmap(lambda m, k: jnp.stack(_random_choice(m, k, size)))(masks, keys))
            else:
                out.append(jax.vmap(lambda x, m, k: jnp.stack(
                    network.sample(p, x, m, k, size=size, deterministic=deterministic)[0]))(features, masks, keys))
        return jnp.where(use_a[:, None], out[0], out[1])

    view, packed = env.observe(), env.legal_masks(packed=True)
    prev_scores = np.zeros((G, 2), np.int64)
    finished = np.zeros(G, bool)
    outcome = np.zeros(G, np.int8)          # A's terminal-only result: 1 win, -1 loss, 0 draw/truncated
    outcome_cap = np.zeros(G, np.int8)      # A's result with score at the cap deciding truncated games
    truncated = np.zeros(G, bool)
    key = jax.random.key(int(cfg.get('seed', 0)))
    ticks, started = 0, time.time()
    while not finished.all():
        mover = view[:, player_at].astype(np.int64)
        key, sub = jax.random.split(key)
        choice = np.asarray(decide(params['a'][0], params['b'][0], jnp.asarray(view), jnp.asarray(packed),
                                   jnp.asarray(mover == a_seat), sub))
        out = env.tick(choice[:, 0], choice[:, 1], choice[:, 2], seeds, tuple(lay), both_views=False)
        if np.any(out['status']):
            raise RuntimeError(f"illegal policy actions on lanes {np.flatnonzero(out['status']).tolist()}")
        ticks += 1
        new = out['done'] & ~finished
        for g in np.flatnonzero(new):
            seat = a_seat[g] - 1
            if out['truncated'][g]:
                truncated[g] = True
                mine, theirs = prev_scores[g, seat], prev_scores[g, 1 - seat]
                outcome_cap[g] = np.sign(mine - theirs)
            else:
                r = float(out['rewards'][g, seat])
                outcome[g] = outcome_cap[g] = np.sign(r)
        finished |= new
        prev_scores = out['scores'].astype(np.int64)
        view, packed = out['view'], out['masks']
        if ticks % 250 == 0:
            print(json.dumps(dict(ticks=ticks, finished=int(finished.sum()), seconds=round(time.time() - started, 1),
                                  mean_turn=float(out['view'][:, offsets['turn']].mean()))), file=sys.stderr, flush=True)
        if ticks >= int(cfg.get('max_ticks', 6000)):
            # unfinished games are decided by the current score, as a truncation would be; counted in `unfinished`
            for g in np.flatnonzero(~finished):
                seat = a_seat[g] - 1
                outcome_cap[g] = np.sign(prev_scores[g, seat] - prev_scores[g, 1 - seat])
                truncated[g] = True
            break

    def summary(res):
        wins, losses = int((res > 0).sum()), int((res < 0).sum())
        score = (res.astype(np.float64) + 1) / 2
        per_map = score.reshape(maps, 2).mean(1)
        se = per_map.std(ddof=1) / np.sqrt(maps) if maps > 1 else None
        return dict(wins=wins, losses=losses, draws=G - wins - losses, score=float(score.mean()),
                    ci95=None if se is None else [float(per_map.mean() - 1.96 * se), float(per_map.mean() + 1.96 * se)])
    unfinished = int((~finished).sum())
    return dict(backend=backend, unfinished=unfinished, no_peace=bool(cfg.get('no_peace', False)), a=cfg['a'], b=cfg['b'], arch_a=params['a'][1], arch_b=params['b'][1], maps=maps, games=G,
                seed_base=seed_base, deterministic=deterministic, ticks=ticks, seconds=round(time.time() - started, 1),
                truncated=int(truncated.sum()), terminal_only=summary(outcome), score_at_cap=summary(outcome_cap))


if __name__ == '__main__':
    print('RESULT ' + json.dumps(run(json.loads(sys.argv[1]))), flush=True)
