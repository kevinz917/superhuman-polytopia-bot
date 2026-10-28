"""Actor tick split on the Rust engine: policy forward vs engine vs feature encode vs rows/upload, per sample.

Replays distributed._actor_process's self-play loop (learner_values, omniscient views, map mix, segment rows,
zstd-7 upload frames to a null socket) without a learner. Per-phase wall time, so a run with a tiny network
gives the CPU cost an actor keeps when the forward moves to a GPU.
argv: size players games seconds blocks width maps
"""
import json
import os
import pickle
import sys
import time


class NullSocket:
    def sendall(self, data):
        pass


def main(size, players, games, seconds, blocks, width, maps):
    from polytopia_bot.rl.distributed import (_env_setup, _template_params, SegmentCollector, stack_rows, send_msg)
    _env_setup()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import Choice
    from polytopia_bot.rl.rust_engine import RustGames
    from polytopia_bot.maps.stream import MapSource
    from polytopia_bot.maps.state import stack as stack_maps

    rng = np.random.default_rng(int(os.environ.get('SEED', 0)))
    source = MapSource(maps, size, players=players)
    engine = RustGames(size, games, map_batch=True, both_views=True, players=players)
    encode_s = [0.]
    raw_features = engine._features

    def timed_features(view):
        t = time.perf_counter()
        out = jax.block_until_ready(raw_features(view))
        encode_s[0] += time.perf_counter() - t
        return out
    engine._features = timed_features
    params = _template_params(size, blocks=blocks, width=width, players=players)
    act = jax.jit(jax.vmap(lambda p, x, m, k: network.sample(p, x, m, k, size=size), in_axes=(None, 0, 0, 0)))

    lane_maps = [None] * games

    def refill(idx):
        for i in idx:
            seed = int(rng.integers(200000, 300000))
            lane_maps[i] = (seed, source.layout(seed)[1])
    refill(range(games))
    states, prepared = engine.reset(stack_maps(lane_maps))
    refill(range(games))
    traj = SegmentCollector(games, 32, players)
    key = jax.random.key(0)
    out = act(params, prepared.features, prepared.masks, jax.random.split(key, games))   # compile
    jax.block_until_ready(out)

    t_act = t_tick = t_rows = t_send = t_maps = 0.
    steps = finished = 0
    pending, pending_segments = [], []
    sock = NullSocket()
    start = time.perf_counter()
    encode_s[0] = 0.
    while time.perf_counter() - start < seconds:
        t0 = time.perf_counter()
        key, k1 = jax.random.split(key)
        choice, path, logp, _ = jax.tree.map(np.asarray, act(params, prepared.features, prepared.masks,
                                                                 jax.random.split(k1, games)))
        t1 = time.perf_counter()
        current = np.asarray(states.current_player)
        fam, src, tgt = (np.packbits(a, axis=-1) for a in path)
        view, opp = np.asarray(prepared.view), np.asarray(prepared.opp_view)
        for i in range(games):
            traj.add(i, int(current[i]), dict(view=view[i], family_mask=fam[i], source_mask=src[i],
                                              target_mask=tgt[i], choice=np.array([choice[0][i], choice[1][i],
                                              choice[2][i]], np.int32), logp=np.float32(logp[i]),
                                              version=np.int32(0), end_turn=bool(choice[0][i] == 0),
                                              opp_view=opp[i]))
        t2 = time.perf_counter()
        enc0 = encode_s[0]
        tick, status = engine.tick(states, Choice(*(jnp.asarray(a) for a in choice)), stack_maps(lane_maps))
        t3 = time.perf_counter()
        if np.any(np.asarray(status) != 0):
            raise RuntimeError('illegal actions')
        steps += games
        done = np.asarray(tick.done)
        if done.any():
            final_views, terminated, rewards = np.asarray(tick.final_views), np.asarray(tick.terminated), np.asarray(tick.rewards)
            idx = [int(i) for i in np.nonzero(done)[0]]
            for i in idx:
                traj.finish(i, rewards[i], bool(terminated[i]), final_views[i])
                finished += 1
            tm = time.perf_counter()
            refill(idx)
            t_maps += time.perf_counter() - tm
        states, prepared = tick.states, tick.prepared
        rows, segments = traj.take()
        offset = np.int32(len(pending_segments))
        for r in rows:
            r['seg'] = r['seg'] + offset
        pending.extend(rows)
        pending_segments.extend(segments)
        t4 = time.perf_counter()
        if len(pending) >= 512:
            send_msg(sock, dict(type='data', batch=stack_rows(pending),
                                segments=stack_rows(pending_segments) if pending_segments else None), level=7)
            pending, pending_segments = [], []
        t5 = time.perf_counter()
        t_act += t1 - t0
        t_rows += (t2 - t1) + (t4 - t3)
        t_tick += (t3 - t2) - (encode_s[0] - enc0)
        t_send += t5 - t4
    wall = time.perf_counter() - start
    ms = lambda t: round(1e3 * t / steps, 4)
    return dict(size=size, players=players, games=games, net=f'{blocks}x{width}', steps=steps, finished=finished,
                samples_per_s=round(steps / wall, 1), total_ms=ms(wall), act_ms=ms(t_act),
                engine_ms=ms(t_tick - t_maps), encode_ms=ms(encode_s[0]), maps_ms=ms(t_maps), rows_ms=ms(t_rows),
                send_ms=ms(t_send))


if __name__ == '__main__':
    a = sys.argv[1:]
    print(json.dumps(main(int(a[0]), int(a[1]), int(a[2]), float(a[3]), int(a[4]), int(a[5]), a[6])))
