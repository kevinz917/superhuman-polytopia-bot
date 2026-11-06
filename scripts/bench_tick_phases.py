"""Where one actor tick spends its time, phase by phase, on one pinned CPU core.

Splits the production tick (cpu_engine.CpuGames, fast=True, both views) into engine step vs
prepare (legal masks, observe/encode, packing) and the masks into their pieces, and the policy
forward into trunk vs heads; also times the omniscient-critic path. Classic maps (map generation
was 2% of the tick in bench_actor_profile). Run on Modal only (experiments/modal_tick_phases.py).

argv: games seconds_warmup repeats
"""
import json
import os
import statistics
import sys
import time


def run(games, warm_seconds, repeats, core=None):
    if core is not None:
        os.sched_setaffinity(0, {core})
    from polytopia_bot.rl.distributed import _env_setup, _template_params, _critic_template, omniscient_features
    _env_setup()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import Choice, flatten_choice
    from polytopia_bot.rl.cpu_engine import CpuGames, view_features, _move_rows, pack_view
    from polytopia_bot.rl.observation import encode_view
    from polytopia_bot.gpu import actions as engine_actions, core, kernels, structures, military, economy, capture
    from polytopia_bot.gpu.kernels import research_mask
    from polytopia_bot.gpu.rules import load_rules
    from polytopia_bot.proposals import fast_engine

    size = 14
    n = size * size
    rules = load_rules()
    engine = CpuGames(size, games, fast=True, both_views=True)
    slow = CpuGames(size, games, fast=False, both_views=True) if os.environ.get('SLOW', '1') == '1' else None
    params = _template_params(size)
    critic = _critic_template(size)
    act = jax.jit(jax.vmap(lambda p, x, m, k: network.sample(p, x, m, k, size=size), in_axes=(None, 0, 0, 0)))
    trunk = jax.jit(jax.vmap(lambda p, x: network.trunk(p, x), in_axes=(None, 0)))
    critic_value = jax.jit(jax.vmap(lambda c, x: network.trunk(c, x)[2], in_axes=(None, 0)))
    opp_features = jax.jit(jax.vmap(lambda v: view_features(v, engine.layout, size)))

    @jax.jit
    def act_subset(p, features, masks, index, key):
        take = lambda a: jnp.take(a, index, axis=0)
        return act(p, jax.tree.map(take, features), jax.tree.map(take, masks), jax.random.split(key, index.shape[0]))

    rng = np.random.default_rng(0)
    seeds = jnp.asarray(rng.integers(200000, 300000, games).astype(np.int32))
    states, prepared = engine.reset(seeds)
    key = jax.random.key(0)

    # --- engine pieces, each as lax.map over the games like the production tick
    def step_one(args):
        s, choice = args
        action = engine_actions.decode(s, flatten_choice(choice, size), size=size)
        return core.step(s, action, s.state_version, rules, size=size).state

    def masks_fast(s):
        return fast_engine._legal_masks_with_move(s, rules, size, _move_rows(s, rules, size))

    def masks_full(s):
        return engine_actions.legal_masks(s, rules, size=size)

    def move_rows(s):
        return _move_rows(s, rules, size)

    def move_full(s):
        return jax.vmap(lambda i: kernels.movement_targets(s, i, rules, size=size))(jnp.arange(n))

    def attack_rows(s):
        return jax.vmap(lambda i: military.attack_mask(s, i, rules, size=size))(jnp.arange(n))

    def build(s):
        return structures.build_mask(s, rules, size=size)

    def validity_like(s):
        """The masks core.step recomputes to validate one action (whole-board ones; move/attack for one source)."""
        return (structures.build_mask(s, rules, size=size), military.summon_mask(s, rules), economy.harvest_mask(s),
                economy.forest_masks(s), economy.reward_mask(s), capture.capture_mask(s, rules),
                structures.destroy_mask(s, rules), research_mask(s, rules),
                kernels.movement_targets(s, jnp.int32(0), rules, size=size),
                military.attack_mask(s, jnp.int32(0), rules, size=size))

    def observe_encode(s):
        return encode_view(kernels.observe(s, s.current_player), size)

    def observe_pack_both(s):
        return (pack_view(kernels.observe(s, s.current_player), engine.layout),
                pack_view(kernels.observe(s, 3 - s.current_player), engine.layout))

    mapped = {}
    with fast_engine.patched('removal', 'path', 'buildmask'):
        for name, fn in dict(step=step_one, masks_fast=masks_fast, move_rows=move_rows, attack_rows=attack_rows,
                             build_mask=build, validity_like=validity_like, observe_encode=observe_encode,
                             observe_pack_both=observe_pack_both).items():
            if name == 'step':
                f = jax.jit(lambda S, C: jax.lax.map(fn, (S, C)))
                mapped[name] = f.lower(states, Choice(*(jnp.zeros(games, jnp.int32),) * 3)).compile()
            else:
                f = jax.jit(lambda S, fn=fn: jax.lax.map(fn, S))
                mapped[name] = f.lower(states).compile()
    # unpatched variants (what the frozen engine does)
    for name, fn in dict(masks_full=masks_full, move_full=move_full).items():
        f = jax.jit(lambda S, fn=fn: jax.lax.map(fn, S))
        mapped[name] = f.lower(states).compile()

    # --- warm up: play the template policy for a while so states are mid-game
    keys = jax.random.split(key, games)
    t0 = time.monotonic()
    ticks = 0
    while time.monotonic() - t0 < warm_seconds:
        key, k = jax.random.split(key)
        choice, path, logp, values = act(params, prepared.features, prepared.masks, jax.random.split(k, games))
        tick, status = engine.tick(states, choice, seeds)
        states, prepared = tick.states, tick.prepared
        ticks += 1
    turn = np.asarray(states.turn)
    choice = act(params, prepared.features, prepared.masks, jax.random.split(key, games))[0]
    choice_np = Choice(*(np.asarray(a) for a in choice))

    def timed(fn, *args):
        out = []
        for _ in range(repeats):
            a = time.perf_counter()
            r = fn(*args)
            jax.block_until_ready(r)
            out.append(1e3 * (time.perf_counter() - a))
        return statistics.median(out)

    index = jnp.arange(games, dtype=jnp.int32)
    full = omniscient_features(prepared.features, opp_features(prepared.opp_view))
    res = dict(games=games, warm_ticks=ticks, mean_turn=float(turn.mean()))
    res['tick_full'] = timed(lambda: engine.tick(states, choice, seeds))
    if slow is not None:
        res['tick_slow'] = timed(lambda: slow.tick(states, choice, seeds))
    res['step'] = timed(lambda: mapped['step'](states, choice))
    for name in ('masks_fast', 'masks_full', 'move_rows', 'move_full', 'attack_rows', 'build_mask', 'validity_like',
                 'observe_encode', 'observe_pack_both'):
        res[name] = timed(lambda name=name: mapped[name](states))
    res['policy_sample'] = timed(lambda: act(params, prepared.features, prepared.masks, keys))
    res['policy_subset_gather'] = timed(lambda: act_subset(params, prepared.features, prepared.masks, index, key))
    res['policy_trunk'] = timed(lambda: trunk(params, prepared.features))
    res['opp_decode'] = timed(lambda: opp_features(prepared.opp_view))
    res['critic_forward'] = timed(lambda: critic_value(critic, full))
    res['critic_total'] = timed(lambda: critic_value(critic, omniscient_features(prepared.features, opp_features(prepared.opp_view))))
    res['host_copy_view'] = timed(lambda: np.asarray(prepared.view).copy())
    return res


if __name__ == '__main__':
    games, warm, repeats = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    core = int(os.environ['PIN_CORE']) if os.environ.get('PIN_CORE') else None
    print(json.dumps(run(games, warm, repeats, core)))
