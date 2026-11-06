"""Production-shaped actor tick profile: where one actor process spends its time per phase.

Mirrors _actor_process (policy forward on own lanes, pool-opponent forward on a subset, omniscient
critic forward, engine tick with both views and proposed-v1 maps, row bookkeeping, GAE segments,
message packing + zstd). Run on Modal only (experiments/modal_actor_profile.py).

argv: processes games seconds maps(classic|proposed-v1) omni(0/1) opponent_share(0..1)
"""
import json
import multiprocessing as mp
import os
import pickle
import sys
import time


def worker(games, seconds, maps, omni, opp_share, out, core=None):
    if core is not None:
        os.sched_setaffinity(0, {core})
    from polytopia_bot.rl.distributed import (_env_setup, _template_params, _critic_template, omniscient_features,
                                              SeatTrajectories, stack_rows, _compress)
    _env_setup()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import Choice
    from polytopia_bot.rl.cpu_engine import CpuGames, view_features
    t0 = time.monotonic()
    size = 14
    source = None
    if maps != 'classic':
        from polytopia_bot.maps.stream import MapSource
        from polytopia_bot.maps.state import stack as stack_maps
        source = MapSource(maps, size)
    engine = CpuGames(size, games, fast=True, map_batch=source is not None, both_views=omni)
    params = _template_params(size)
    rival = _template_params(size)
    act = jax.jit(jax.vmap(lambda p, x, m, k: network.sample(p, x, m, k, size=size), in_axes=(None, 0, 0, 0)))

    @jax.jit
    def act_subset(p, features, masks, index, key):
        take = lambda a: jnp.take(a, index, axis=0)
        return act(p, jax.tree.map(take, features), jax.tree.map(take, masks), jax.random.split(key, index.shape[0]))
    buckets = sorted({b for b in (4, 8, 12, 16, 24, 32, 48, 64) if b < games} | {games})

    def run_subset(p, prepared, lanes_idx, key):
        n = len(lanes_idx)
        b = next(x for x in buckets if x >= n)
        index = np.zeros(b, np.int32)
        index[:n] = lanes_idx
        o = act_subset(p, prepared.features, prepared.masks, jnp.asarray(index), key)
        return jax.tree.map(lambda a: np.asarray(a)[:n], o)

    if omni:
        critic = _critic_template(size=size)
        opp_features = jax.jit(jax.vmap(lambda v: view_features(v, engine.layout, size)))
        critic_value = jax.jit(jax.vmap(lambda c, x: network.trunk(c, x)[2], in_axes=(None, 0)))

    rng = np.random.default_rng(os.getpid())
    lane_maps = [None] * games

    def refill(idx):
        for i, seed in zip(idx, rng.integers(200000, 300000, len(idx))):
            lane_maps[i] = int(seed) if source is None else (int(seed), source.layout(int(seed))[1])

    def next_maps():
        return np.array(lane_maps, np.int32) if source is None else stack_maps(lane_maps)
    refill(range(games))
    states, prepared = engine.reset(next_maps())
    refill(range(games))
    # fixed opponent lanes (production: lanes vs pool opponents, learner moves on ~half their decisions)
    n_opp = int(round(opp_share * games))
    opp_lanes = np.zeros(games, bool)
    opp_lanes[:n_opp] = True
    traj = SeatTrajectories(games, 32, .995, .99)
    key = jax.random.key(0)
    # compile everything once
    mine = np.nonzero(~opp_lanes)[0]
    run_subset(params, prepared, mine, key)
    if n_opp:
        run_subset(rival, prepared, np.nonzero(opp_lanes)[0], key)
    if omni:
        critic_value(critic, omniscient_features(prepared.features, opp_features(prepared.opp_view)))
    zero = Choice(*(jnp.zeros(games, jnp.int32),) * 3)
    engine.tick(states, zero, next_maps())
    compile_s = time.monotonic() - t0

    t = dict(policy=0., opponent=0., critic=0., engine=0., rows=0., maps=0., pack=0.)
    steps = ticks = sent = 0
    pending = []
    start = time.monotonic()
    version = 0
    while time.monotonic() - start < seconds:
        current = np.asarray(states.current_player)
        # opponent lanes: the rival moves when current seat == 2 (half the decisions on those lanes)
        opponent_turn = opp_lanes & (current == 2)
        mine = np.nonzero(~opponent_turn)[0]
        key, k1, k2 = jax.random.split(key, 3)
        c_np = [np.zeros(games, np.int32) for _ in range(3)]
        a = time.perf_counter()
        choice, path, logp, values = run_subset(params, prepared, mine, k1)
        for j in range(3):
            c_np[j][mine] = choice[j]
        fam_np, src_np, tgt_np = (np.packbits(x, axis=-1) for x in path)
        b = time.perf_counter()
        t['policy'] += b - a
        if omni:
            full = omniscient_features(prepared.features, opp_features(prepared.opp_view))
            values = np.asarray(critic_value(critic, full))[mine]
            opp_view_np = np.asarray(prepared.opp_view)
            c = time.perf_counter()
            t['critic'] += c - b
            b = c
        idx = np.nonzero(opponent_turn)[0]
        if len(idx):
            r = run_subset(rival, prepared, idx, k2)[0]
            for j in range(3):
                c_np[j][idx] = r[j]
        c = time.perf_counter()
        t['opponent'] += c - b
        choice = Choice(*(jnp.asarray(x) for x in c_np))
        view_np = np.asarray(prepared.view)
        for r_, i in enumerate(mine):
            row = dict(view=view_np[i], family_mask=fam_np[r_], source_mask=src_np[r_], target_mask=tgt_np[r_],
                       choice=np.array([c_np[0][i], c_np[1][i], c_np[2][i]], np.int32),
                       logp=np.float32(logp[r_]), value=np.float32(values[r_]), version=np.int32(version))
            if omni:
                row['opp_view'] = opp_view_np[i]
            traj.add(i, int(current[i]), row)
        d = time.perf_counter()
        t['rows'] += d - c
        nm = next_maps()
        e = time.perf_counter()
        t['maps'] += e - d
        tick, status = engine.tick(states, choice, nm)
        jax.block_until_ready(tick.states.turn)
        f = time.perf_counter()
        t['engine'] += f - e
        done = np.asarray(tick.done)
        if done.any():
            rewards = np.array(tick.rewards, np.float32)
            boot = np.zeros((games, 2), np.float32)
            for i in np.nonzero(done)[0]:
                traj.finish(i, rewards[i], True, boot[i])
            refill([int(i) for i in np.nonzero(done)[0]])
        g = time.perf_counter()
        t['maps'] += g - f
        states, prepared = tick.states, tick.prepared
        pending.extend(traj.take())
        if len(pending) >= 512:
            body = _compress(pickle.dumps(dict(type='data', batch=stack_rows(pending)), protocol=5))
            sent += len(body)
            pending = []
        t['pack'] += time.perf_counter() - g
        steps += games
        ticks += 1
    el = time.monotonic() - start
    import resource
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    out.put(dict(games=games, compile=compile_s, rss_mib=rss_mib, steps_per_s=steps / el, ms_per_tick=1e3 * el / ticks,
                 bytes_per_step=sent / steps, **{f'{k}_ms': 1e3 * v / ticks for k, v in t.items()}))


def main(processes, games, seconds, maps, omni, opp_share):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    cores = sorted(os.sched_getaffinity(0))
    ps = [ctx.Process(target=worker, args=(games, seconds, maps, omni, opp_share, q, cores[i % len(cores)]))
          for i in range(processes)]
    for p in ps:
        p.start()
    rows = [q.get() for _ in ps]
    for p in ps:
        p.join()
    keys = [k for k in rows[0] if k not in ('games',)]
    return dict(processes=processes, games=games, maps=maps, omni=omni, opp_share=opp_share, affinity=len(cores),
                total_steps_per_s=sum(r['steps_per_s'] for r in rows),
                **{k: sum(r[k] for r in rows) / len(rows) for k in keys})


if __name__ == '__main__':
    print(json.dumps(main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], bool(int(sys.argv[5])),
                          float(sys.argv[6]))))
