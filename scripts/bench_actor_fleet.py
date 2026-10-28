"""Actor-fleet throughput in one container: local CPU policy forwards vs the in-node inference server.

Runs the real actor processes (distributed.run_actor_container) against a stub learner in this process that
answers hello with template weights and acknowledges data messages. Measures learner decisions per second
(the training samples/s) over the last `measure` seconds, after compile and warm-up. With a server, the
learner rows' logp (served, possibly bf16) is re-scored in fp32 on a sample of rows (the PPO ratio at equal
weights).
Run on Modal only (experiments/modal_inference_server.py).

argv: JSON {processes, games, seconds, mode (local|cuda|cpu), size, players, maps, blocks, width, measure,
            serve_options, logp_check, weights, learner}
weights: a checkpoint .npz (e.g. a hero run's latest.npz) served instead of the noised template; learner: config
fields the actors get from the stub (e.g. coded_views, upload_zstd_level) as the live run had them.
"""
from dataclasses import asdict
import json
import os
import socket
import sys
import threading
import time


def stub_learner(size, players, maps, blocks, width, send_samples=512, keep_rows=0, weights=None, learner=None):
    """A socket server speaking the learner protocol: fixed weights, no pool, counts what actors send."""
    from polytopia_bot.rl.distributed import LearnerConfig, _template_params, recv_msg, send_msg
    import jax
    import numpy as np
    cfg = asdict(LearnerConfig(size=size, players=players, maps=maps, send_samples=send_samples,
                               historical_probability=0., self_play_probability=1., omniscient_critic=True,
                               learner_values=True, turn_gamma=.99, turn_lambda=.95))
    cfg['network'] = dict(size=size, blocks=blocks, extra_blocks=0, width=width, architecture='normalized-v2',
                          players=players)
    cfg['pbrs_scale'] = 0.
    cfg.update(learner or {})
    params = _template_params(**cfg['network'])
    leaves = [np.asarray(a, np.float32) for a in jax.tree.leaves(params)]
    if weights:
        with np.load(weights, allow_pickle=False) as archive:
            leaves = [np.asarray(archive[f'p{i}'], np.float32) for i in range(len(leaves))]
    else:
        # template weights are near-uniform; scale the heads up so the logp check sees a peaked policy like a trained one
        rng = np.random.default_rng(0)
        leaves = [a + rng.normal(0, .05, a.shape).astype(np.float32) if a.ndim >= 2 else a for a in leaves]
    events, rows, lock = [], [], threading.Lock()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', 0))
    srv.listen(512)

    def handle(conn):
        try:
            while True:
                msg = recv_msg(conn)
                if msg['type'] == 'hello':
                    send_msg(conn, dict(config=cfg, version=0, weights=leaves, critic_weights=None, pool_spec=[],
                                        pool_params={}), compress=False)
                    continue
                st = msg['stats']
                with lock:
                    events.append((time.monotonic(), st['learner_decisions'], st['steps'], st.get('serve_wait_s', 0.),
                                   time.time() - st['started'], st.get('serve_pack_s', 0.), st.get('engine_s', 0.)))
                    if keep_rows and msg.get('batch') is not None and len(rows) < keep_rows:
                        b = msg['batch']
                        take = min(64, len(b['logp']))
                        rows.append({k: b[k][:take] for k in ('view', 'family_mask', 'source_mask', 'target_mask',
                                                                 'choice', 'logp')})
                send_msg(conn, dict(version=0, stop=False), compress=False)
        except (ConnectionError, OSError, EOFError):
            pass
        finally:
            conn.close()

    def accept():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    threading.Thread(target=accept, daemon=True).start()
    return srv.getsockname(), events, rows, srv, (cfg['network'], leaves)


def logp_check(rows, spec, leaves, size, players):
    """fp32 (HIGHEST) logp of the served choices vs the logp the actor recorded."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from polytopia_bot.rl import network
    from polytopia_bot.rl.actions import Choice, PathMasks
    from polytopia_bot.rl.cpu_engine import view_features, view_layout
    from polytopia_bot.rl.distributed import _template_params
    from polytopia_bot.rl.inference_server import wire_shapes
    shapes = wire_shapes(size, players)
    layout = view_layout(size, players)
    params = jax.tree.structure(_template_params(**spec)).unflatten([jnp.asarray(a) for a in leaves])
    unpack = lambda a, bits: jnp.asarray(np.unpackbits(a, axis=-1)[:, :bits].astype(bool))
    view = np.concatenate([r['view'] for r in rows])
    path = PathMasks(unpack(np.concatenate([r['family_mask'] for r in rows]), shapes['family_bits']),
                     unpack(np.concatenate([r['source_mask'] for r in rows]), shapes['source_bits']),
                     unpack(np.concatenate([r['target_mask'] for r in rows]), shapes['target_bits']))
    choice = np.concatenate([r['choice'] for r in rows])
    served = np.concatenate([r['logp'] for r in rows]).astype(np.float64)

    @jax.jit
    def score(v, c, m):
        feats = jax.vmap(lambda x: view_features(x, layout, size))(v)
        return jax.vmap(lambda x, cc, mm: network.evaluate(params, x, Choice(*cc), mm)[0])(feats, c, m)
    fp32 = np.concatenate([np.asarray(score(jnp.asarray(view[i:i + 256]), jnp.asarray(choice[i:i + 256].T).T,
                                            jax.tree.map(lambda a: a[i:i + 256], path)))
                           for i in range(0, len(view), 256)]).astype(np.float64)
    d = served - fp32
    ratio = np.exp(fp32 - served)
    return dict(rows=int(len(d)), mean_abs_dlogp=float(np.mean(np.abs(d))), p99_abs_dlogp=float(np.quantile(np.abs(d), .99)),
                max_abs_dlogp=float(np.max(np.abs(d))), ratio_outside_clip=float(np.mean(np.abs(ratio - 1) > .2)),
                ratio_p01=float(np.quantile(ratio, .01)), ratio_p99=float(np.quantile(ratio, .99)),
                mean_served_logp=float(np.mean(served)), approx_kl=float(np.mean(served - fp32)))


def main(a):
    os.environ['JAX_PLATFORMS'] = 'cpu'   # actors; the server picks its own platform
    from polytopia_bot.rl.distributed import run_actor_container
    keep = 400 if a.get('logp_check') else 0
    address, events, rows, srv, (spec, leaves) = stub_learner(a['size'], a['players'], a['maps'], a['blocks'],
                                                              a['width'], keep_rows=keep, weights=a.get('weights'),
                                                              learner=a.get('learner'))
    started = time.monotonic()
    summary = run_actor_container(address, a['processes'], a['games'], report=lambda d: None, report_every=15.,
                                  should_stop=lambda: time.monotonic() - started > a['seconds'],
                                  inference=None if a['mode'] == 'local' else a['mode'],
                                  serve_options=a.get('serve_options'))
    srv.close()
    # The window ends at the last data message, not at return: shutdown produces no samples.
    end = max(e[0] for e in events) if events else time.monotonic()
    measure = a.get('measure', 90)
    window = [e for e in events if e[0] >= end - measure]
    decisions = sum(e[1] for e in window)
    span = (end - min(e[0] for e in window)) if window else float('nan')
    frac = lambda j: round(sum(e[j] for e in window) / max(1e-9, sum(e[4] for e in window)), 3) if window else None
    out = dict(mode=a['mode'], processes=a['processes'], games=a['games'], size=a['size'], players=a['players'],
               net=f"{a['blocks']}x{a['width']}", serve_options=a.get('serve_options'), measure_s=round(span, 1),
               decisions_per_s=round(decisions / span, 1) if window else None,
               per_process_decisions_per_s=round(decisions / span / a['processes'], 1) if window else None,
               actor_wait_fraction=frac(3), actor_pack_fraction=frac(5), actor_engine_fraction=frac(6),
               first_compile_s=summary.get('first_compile_seconds'), restarts=summary.get('restarts'),
               errors=summary.get('errors'), total_sent=summary.get('sent_samples'), loadavg=os.getloadavg(),
               cpus=len(os.sched_getaffinity(0)), memory_peak_gib=summary.get('memory_peak_gib'),
               server_exitcode=summary.get('server_exitcode'))
    if rows:
        out['logp_check'] = logp_check(rows, spec, leaves, a['size'], a['players'])
    return out


if __name__ == '__main__':
    print(json.dumps(main(json.loads(sys.argv[1]))))
