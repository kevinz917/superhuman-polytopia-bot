"""Cross-architecture head-to-head on the CUDA engine (engine-cu, host-pointer mode): one game per lane, paired maps.

    python gpu_arena.py '{"a": "/results/.../final.npz", "b": "random", "maps": 1024, "seed_base": 300000}'

Every map seed is played twice with the seats swapped (lane 2k: A is seat 1, lane 2k+1: A is seat 2); all games run at
once and each lane counts only its first game. Policies: a checkpoint path (any architecture network_spec loads,
sampled T=1 decisions by default as the trainer plays; deterministic argmax can stall early policies) or 'random' (uniform over the legal paths: family, then source, then
target). Outcome per game (NO-SCORE RULE): +1 if A won by elimination, -1 otherwise; a game capped at
the round limit or still unfinished at max_ticks is a LOSS for A, never decided on score.
Returns `no_score` (paired-map score, map-clustered 95% interval) and `per_map`: for every map its seed and, per seat
([A in seat 1, A in seat 2]), the first game's no-score outcome, finish reason (win / loss / capped / unfinished) and
final round (1-based round of the deciding action, read before the engine auto-resets the lane; capped = max_turns;
unfinished = the round reached), plus `final_round` quantiles and `capped_share`. `historical` keeps the old
terminal_only / cap_loss summaries as diagnostics only: never decision or continuation evidence.
Map seeds 300000+ are outside the training table [200000, 300000). `no_peace: true` plays under the no_peace rule
(default false = engine default, peace allowed; every session E arena so far used the default).
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


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
    if cfg.get('table'):
        # stage 5 (Session L): boards of the REAL game generators from a full-game engine map table on held-out
        # seeds; `map_type` keeps the first `maps` seeds >= seed_base of that type (types: the table's order)
        with np.load(cfg['table'].replace('{size}', str(size))) as z:
            lo, kind, types = int(z['lo']), z['kind'], [str(t) for t in z['types']]
            want = types.index(cfg['map_type']) if cfg.get('map_type') else None
            idx = [i for i in range(seed_base - lo, len(kind)) if want is None or int(kind[i]) == want][:maps]
            if len(idx) < maps:
                raise ValueError(f'table has {len(idx)} {cfg.get("map_type")} boards from seed {seed_base}, need {maps}')
            lay = [np.ascontiguousarray(np.repeat(np.asarray(z[f][idx], np.int32), 2, axis=0))
                   for f in ('terrain', 'structure', 'resource', 'capitals')]
        seeds = np.repeat(np.asarray(idx, np.int32) + lo, 2)
    else:
        source = MapSource(cfg.get('mix', 'classic'), size, players=2)
        rows = [source.layout(seed_base + k)[1] for k in range(maps)]
        lay = [np.ascontiguousarray(np.repeat(np.stack([np.asarray(getattr(r, f), np.int32) for r in rows]), 2, axis=0))
               for f in ('terrain', 'structure', 'resource', 'capitals')]
        seeds = np.repeat(np.arange(seed_base, seed_base + maps, dtype=np.int32), 2)
    # max_turns: None = the harness default for the board; stage 5 passes 200;
    # max_actions out of reach so no action cap binds first
    env = GpuGames(size=size, games=G, players=2, no_resign=True, no_peace=bool(cfg.get('no_peace', False)),
                   max_turns=cfg.get('max_turns'), max_actions=int(cfg.get('max_actions', 2 ** 31 - 1)), device=0)
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
    finished = np.zeros(G, bool)
    outcome = np.zeros(G, np.int8)          # A's terminal-only result: 1 win, -1 loss, 0 draw/truncated (historical)
    outcome_cap = np.zeros(G, np.int8)      # NO-SCORE RULE: a truncated game is a LOSS for A (the model under test)
    truncated = np.zeros(G, bool)
    final_round = np.zeros(G, np.int32)     # first game's round of the deciding action (1-based), before the lane resets
    key = jax.random.key(int(cfg.get('seed', 0)))
    ticks, started = 0, time.time()
    while not finished.all():
        mover = view[:, player_at].astype(np.int64)
        turn_before = view[:, offsets['turn']].astype(np.int32)     # the engine resets done lanes before returning views
        key, sub = jax.random.split(key)
        choice = np.asarray(decide(params['a'][0], params['b'][0], jnp.asarray(view), jnp.asarray(packed),
                                   jnp.asarray(mover == a_seat), sub))
        out = env.tick(choice[:, 0], choice[:, 1], choice[:, 2], seeds, tuple(lay), both_views=False)
        ticks += 1
        new = out['done'] & ~finished
        final_round[new] = turn_before[new] + 1
        for g in np.flatnonzero(new):
            seat = a_seat[g] - 1
            if out['truncated'][g]:
                truncated[g] = True
                outcome_cap[g] = -1
            else:
                r = float(out['rewards'][g, seat])
                outcome[g] = outcome_cap[g] = np.sign(r)
        finished |= new
        view, packed = out['view'], out['masks']
        if ticks % 250 == 0:
            print(json.dumps(dict(ticks=ticks, finished=int(finished.sum()), seconds=round(time.time() - started, 1),
                                  mean_turn=float(out['view'][:, offsets['turn']].mean()))), file=sys.stderr, flush=True)
        if cfg.get('max_ticks') and ticks >= int(cfg['max_ticks']):
            # unfinished games are LOSSES for A (no-score rule); counted in `unfinished`, finish 'unfinished'
            outcome_cap[~finished] = -1
            final_round[~finished] = view[~finished, offsets['turn']].astype(np.int32) + 1
            break

    def summary(res):
        wins, losses = int((res > 0).sum()), int((res < 0).sum())
        score = (res.astype(np.float64) + 1) / 2
        per_map = score.reshape(maps, 2).mean(1)
        se = per_map.std(ddof=1) / np.sqrt(maps)
        return dict(wins=wins, losses=losses, draws=G - wins - losses, score=float(score.mean()),
                    ci95=[float(per_map.mean() - 1.96 * se), float(per_map.mean() + 1.96 * se)])
    unfinished = int((~finished).sum())
    # the no-score rule: a game not won by elimination (capped or unfinished) is a loss for A
    no_score = np.where(finished & ~truncated & (outcome > 0), 1, -1).astype(np.int8)
    finish = np.where(~finished, 'unfinished', np.where(truncated, 'capped', np.where(no_score > 0, 'win', 'loss')))
    q = np.percentile(final_round, [10, 25, 50, 75, 90])
    return dict(unfinished=unfinished, no_peace=bool(cfg.get('no_peace', False)), a=cfg['a'], b=cfg['b'], arch_a=params['a'][1], arch_b=params['b'][1], maps=maps, games=G,
                seed_base=seed_base, deterministic=deterministic, ticks=ticks, seconds=round(time.time() - started, 1),
                truncated=int((truncated & finished).sum()), no_score=summary(no_score),
                capped_share=float((finish == 'capped').mean()),
                final_round=dict(zip(('q10', 'q25', 'q50', 'q75', 'q90'), q.tolist()), max=int(final_round.max())),
                per_map=dict(seed=seeds[::2].tolist(), outcome=no_score.reshape(maps, 2).tolist(),
                             finish=finish.reshape(maps, 2).tolist(), final_round=final_round.reshape(maps, 2).tolist()),
                historical=dict(note='diagnostics only, never decision evidence (no-score rule)',
                                terminal_only=summary(outcome), cap_loss=summary(outcome_cap)),
                max_turns=cfg.get('max_turns'), max_actions=int(cfg.get('max_actions', 2 ** 31 - 1)), mix=cfg.get('mix', 'classic'),
                table=cfg.get('table'), map_type=cfg.get('map_type'), size=size)


if __name__ == '__main__':
    print('RESULT ' + json.dumps(run(json.loads(sys.argv[1]))), flush=True)
