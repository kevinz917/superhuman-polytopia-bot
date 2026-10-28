"""Per-node batched policy inference for the actor fleet (the SEED-RL / Duper "predictor" layout, in-node).

One server process per container owns the node's GPU and the policy weights. Each actor process keeps its
games on the CPU and, once per tick, writes the packed public views and packed legal masks (straight from the
Rust engine, RustGames(served=True)) of the lanes that need a move into its own shared-memory slot, then signals
the server over a pipe. The server batches the requests of every actor that asked within a short deadline,
decodes views and masks on the GPU, runs `network.sample` once per parameter set (learner weights, or a pool
opponent), writes choices, path masks, log-probabilities and values back into the slots and signals each actor.
Training rows still leave the node exactly as before; nothing else crosses a network.

The device side is pipelined: up to `inflight` batches are dispatched (JAX runs them asynchronously) while the
server keeps collecting the next requests, so with more actor processes than cores the GPU stays busy while the
actors whose batch is in flight wait (a blocked actor costs no CPU).

Precision: `trunk_dtype='bfloat16'` runs the conv torso on bf16 operands (network.TRUNK_DTYPE; heads, norms and
logp stay fp32); `precision='DEFAULT'` lets fp32 matmuls use TF32. The learner recomputes logp in fp32, so the
behaviour policy differs from the learner's by the rounding of these paths (measured by the smoke's logp check).

Protocol (multiprocessing pipes, one per actor, in order; every reply follows its request):
  client -> server  ('register', shm_name, size, games, players)    reply ('ok',)
  client -> server  ('params', name, version, leaves, spec)          reply ('id', name, id)  (older versions ignored)
  client -> server  ('request',)                                     reply ('done',) or ('error', text)
  client -> server  ('close',)
A request fills rows 0..count-1 of the slot (lanes needing a forward, in the order the client chose); which[row]
names the parameter set. Replies land in the same rows. Pool parameter sets are never removed from the server (the
pool is at most ~10 networks); the learner's set is replaced in place when a newer version arrives.
"""
from multiprocessing import shared_memory
from multiprocessing.connection import wait
import json
import os
import time
import traceback

import numpy as np

BUCKETS = (32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 16384, 32768)


def mask_leaf_shapes(size):
    """The Rust engine's legal-mask leaves, in wire order (rust_engine.split_masks; gpu.actions.Masks fields)."""
    n = size * size
    return ((n, n), (n, n), (n, 128), (n, 128), (128,), (n,), (n, 3), (n, 9), *((n,),) * 6, (6,), (), ())


def wire_shapes(size, players=2):
    """Shapes shared by client and server: packed view width, mask bits, path-mask bit counts."""
    import jax
    import jax.numpy as jnp
    from polytopia_bot.gpu.actions import Masks
    from .actions import Choice, path_masks
    from .cpu_engine import view_layout
    leaves = mask_leaf_shapes(size)
    masks = Masks(*(jax.ShapeDtypeStruct(s, jnp.bool_) for s in leaves))
    path = jax.eval_shape(lambda m: path_masks(m, Choice(jnp.int32(0), jnp.int32(0), jnp.int32(0)), size), masks)
    width = sum(int(np.prod(s)) for _, s in view_layout(size, players))
    return dict(view_width=width, mask_shapes=leaves, mask_bits=sum(int(np.prod(s)) for s in leaves),
                family_bits=int(path.family.shape[0]), source_bits=int(path.source.shape[0]),
                target_bits=int(path.target.shape[0]))


class SlotLayout:
    """Byte layout of one actor's shared-memory slot (request fields, then reply fields)."""

    def __init__(self, games, shapes):
        nbytes = lambda bits: (bits + 7) // 8
        self.fields = (('count', np.int32, (1,)), ('view', np.int16, (games, shapes['view_width'])),
                       ('mask', np.uint8, (games, nbytes(shapes['mask_bits']))),
                       ('which', np.int16, (games,)), ('choice', np.int32, (games, 3)),
                       ('family', np.uint8, (games, nbytes(shapes['family_bits']))),
                       ('source', np.uint8, (games, nbytes(shapes['source_bits']))),
                       ('target', np.uint8, (games, nbytes(shapes['target_bits']))),
                       ('logp', np.float32, (games,)), ('value', np.float32, (games,)))
        self.offsets, at = {}, 0
        for name, dtype, shape in self.fields:
            at = (at + 63) // 64 * 64
            self.offsets[name] = at
            at += int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.nbytes = at

    def shape(self, name):
        return next(shape for n, _, shape in self.fields if n == name)

    def arrays(self, buf):
        return {name: np.ndarray(shape, dtype, buffer=buf, offset=self.offsets[name])
                for name, dtype, shape in self.fields}


def _bucket(n):
    return next((b for b in BUCKETS if b >= n), n)


def serve(conns, *, platform=None, deadline_ms=4., seed=0, log=None, trunk_dtype='bfloat16', precision='DEFAULT',
          inflight=2, log_every=30., cores=None):
    """The server process body. conns: one pipe end per actor. Returns when every actor has closed.
    cores: CPUs this process is pinned to (run_actor_container reserves them; None leaves the affinity alone)."""
    if cores:
        os.sched_setaffinity(0, set(cores))
    if platform:
        os.environ['JAX_PLATFORMS'] = platform
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    # read at network import: set before it (the server is a fresh spawned process)
    os.environ['POLYTOPIA_TRUNK_DTYPE'] = trunk_dtype
    os.environ['POLYTOPIA_MATMUL_PRECISION'] = precision
    os.environ.setdefault('POLYTOPIA_HEAD_PRECISION', 'HIGHEST')
    import jax
    import jax.numpy as jnp
    from polytopia_bot.gpu.actions import Masks
    from . import network
    from .cpu_engine import view_features, view_layout
    from .distributed import _template_params
    say = log or (lambda d: print(json.dumps(d, default=str), flush=True))
    say(dict(stage='server_start', device=str(jax.devices()[0]), actors=len(conns), trunk_dtype=trunk_dtype,
             precision=precision, inflight=inflight, deadline_ms=deadline_ms))

    state = dict(shapes=None, slot_layout=None, forward=None, size=None)
    slots, params, ids, structures = {}, {}, {}, {}
    key = [jax.random.key(seed)]
    stats = dict(calls=0, lanes=0, gather_s=0., dispatch_s=0., finish_s=0., scatter_s=0., device_wait_s=0.,
                 compiles=0, compile_s=0., idle_s=0., warmed=0, max_inflight=0)
    compiled = set()
    # Gather buffers: inflight + 1 sets used in turn, so a set is only rewritten once the flight that read it is
    # finished (dispatch waits for the oldest flight when `inflight` are out), whatever JAX's host-buffer semantics.
    # Each set holds one (views, masks) pair per job of a flight, grown to the largest bucket seen.
    rings, dispatched = [[] for _ in range(inflight + 1)], [0]
    started_wall, started_cpu = time.monotonic(), time.process_time()

    def buffers(ring, job, rows):
        while len(ring) <= job:
            ring.append(None)
        have = ring[job]
        if have is None or have[0].shape[0] < rows:
            layout = state['slot_layout']
            have = ring[job] = (np.empty((rows,) + layout.shape('view')[1:], np.int16),
                                np.empty((rows,) + layout.shape('mask')[1:], np.uint8))
        return have

    def cpu_stats():
        """The server's CPU time vs wall time since start (well under 1 while never idle: starved of a core) and
        its involuntary context switches."""
        import resource
        wall = time.monotonic() - started_wall
        return dict(wall_s=round(wall, 1), cpu_s=round(time.process_time() - started_cpu, 1),
                    involuntary_switches=resource.getrusage(resource.RUSAGE_SELF).ru_nivcsw,
                    affinity=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)

    def setup(size, games, players):
        shapes = wire_shapes(size, players)
        layout = view_layout(size, players)
        sizes = [int(np.prod(s)) for s in shapes['mask_shapes']]
        edges = np.cumsum([0] + sizes)

        def forward(p, views, packed, k):
            bits = jnp.unpackbits(packed, axis=-1)[:, :edges[-1]].astype(bool)
            masks = Masks(*[bits[:, a:b].reshape((bits.shape[0],) + s)
                            for a, b, s in zip(edges[:-1], edges[1:], shapes['mask_shapes'], strict=True)])
            feats = jax.vmap(lambda v: view_features(v, layout, size))(views)
            keys = jax.random.split(k, views.shape[0])
            choice, path, logp, value = jax.vmap(lambda x, m, kk: network.sample(p, x, m, kk, size=size))(feats, masks, keys)
            return (jnp.stack(choice, 1), jnp.packbits(path.family, axis=-1), jnp.packbits(path.source, axis=-1),
                    jnp.packbits(path.target, axis=-1), logp, value)
        state.update(shapes=shapes, slot_layout=SlotLayout(games, shapes), forward=jax.jit(forward), size=size)

    def to_tree(leaves, spec):
        tag = tuple(sorted(spec.items()))
        if tag not in structures:
            structures[tag] = jax.tree.structure(_template_params(**spec))
        return structures[tag].unflatten([jnp.asarray(a, jnp.float32 if a.dtype == np.float16 else a.dtype)
                                          for a in leaves])

    def dispatch(batch):
        """Gather the batch's rows per parameter set and start the forwards (asynchronously)."""
        arrays = {c: slots[c][1] for c in batch}
        by_id = {}
        for c in batch:
            n = int(arrays[c]['count'][0])
            which = arrays[c]['which'][:n]
            pids = np.unique(which)
            for pid in pids:
                # one parameter set per request is the common case: a plain slice
                idx = slice(0, n) if len(pids) == 1 else np.nonzero(which == pid)[0]
                by_id.setdefault(int(pid), []).append((c, idx))
        jobs = []
        ring = rings[dispatched[0] % len(rings)]
        dispatched[0] += 1
        for job, (pid, parts) in enumerate(by_id.items()):
            key[0], k = jax.random.split(key[0])
            t0 = time.perf_counter()
            sizes = [(idx.stop - idx.start) if isinstance(idx, slice) else len(idx) for _, idx in parts]
            n = sum(sizes)
            b = _bucket(n)
            vbuf, mbuf = buffers(ring, job, b)
            at = 0
            for (c, idx), m in zip(parts, sizes):   # one copy per request straight into the batch
                if isinstance(idx, slice):
                    vbuf[at:at + m] = arrays[c]['view'][idx]
                    mbuf[at:at + m] = arrays[c]['mask'][idx]
                else:
                    np.take(arrays[c]['view'], idx, axis=0, out=vbuf[at:at + m], mode='clip')
                    np.take(arrays[c]['mask'], idx, axis=0, out=mbuf[at:at + m], mode='clip')
                at += m
            if b > n:   # pad with copies of a real row: every padded lane is a legal state
                vbuf[n:b] = vbuf[0]
                mbuf[n:b] = mbuf[0]
            views, masks = vbuf[:b], mbuf[:b]
            t1 = time.perf_counter()
            fresh = b not in compiled
            out = state['forward'](params[pid][1], views, masks, k)   # async on the device
            if fresh:
                jax.block_until_ready(out)
                compiled.add(b)
                stats['compiles'] += 1
                stats['compile_s'] += time.perf_counter() - t1
            stats['gather_s'] += t1 - t0
            stats['dispatch_s'] += time.perf_counter() - t1
            jobs.append((parts, n, out, fresh))
        return dict(batch=batch, jobs=jobs, arrays=arrays)

    def finish(flight):
        t0 = time.perf_counter()
        for parts, n, out, fresh in flight['jobs']:
            t = time.perf_counter()
            host = [np.asarray(o)[:n] for o in out]   # waits for the device
            stats['device_wait_s'] += time.perf_counter() - t
            t = time.perf_counter()
            at = 0
            for c, idx in parts:
                a = flight['arrays'][c]
                m = (idx.stop - idx.start) if isinstance(idx, slice) else len(idx)
                for field, o in zip(('choice', 'family', 'source', 'target', 'logp', 'value'), host, strict=True):
                    a[field][idx] = o[at:at + m]
                at += m
            stats['scatter_s'] += time.perf_counter() - t
            if not fresh:
                stats['calls'] += 1
                stats['lanes'] += n
        for c in flight['batch']:
            if c in slots:
                c.send(('done',))
        stats['finish_s'] += time.perf_counter() - t0

    def fail(batch, text):
        for c in batch:
            try:
                c.send(('error', text[-4000:]))
            except OSError:
                pass

    def warm(pid, games):
        stats['warmed'] = 1
        width = state['slot_layout'].shape('view')[1]
        mask_bytes = state['slot_layout'].shape('mask')[1]
        top = _bucket(len(conns) * games)
        t = time.perf_counter()
        for b in BUCKETS:
            if b > top:
                break
            jax.block_until_ready(state['forward'](params[pid][1], np.zeros((b, width), np.int16),
                                                   np.full((b, mask_bytes), 255, np.uint8), key[0]))
            compiled.add(b)
        stats['compiles'] += len(compiled)
        stats['compile_s'] += time.perf_counter() - t
        say(dict(stage='server_warm', buckets=sorted(compiled), seconds=round(time.perf_counter() - t, 1)))

    def handle(c, msg):
        nonlocal first
        kind = msg[0]
        if kind == 'register':
            _, shm_name, size, games, players = msg
            if state['forward'] is None:
                setup(size, games, players)
            if c in slots:   # a restarted actor on the same pipe
                slots.pop(c)[0].close()
            shm = shared_memory.SharedMemory(name=shm_name)
            slots[c] = (shm, state['slot_layout'].arrays(shm.buf))
            c.send(('ok',))
        elif kind == 'params':
            _, name, version, leaves, spec = msg
            if name not in ids:
                ids[name] = len(ids)
            pid = ids[name]
            if pid not in params or params[pid][0] < version:
                params[pid] = (version, to_tree(leaves, spec))
            c.send(('id', name, pid))
            if not stats['warmed'] and state['forward'] is not None:
                warm(pid, state['slot_layout'].shape('view')[0])
        elif kind == 'request':
            pending.append(c)
            first = first if first is not None else time.monotonic()
        elif kind == 'close':
            conns.remove(c)
            if c in pending:
                pending.remove(c)
            if c in slots:
                slots.pop(c)[0].close()
            try:
                c.close()
            except OSError:
                pass

    conns = list(conns)
    pending, first, flights = [], None, []
    deadline = deadline_ms / 1e3
    last_log = time.monotonic()
    while conns:
        # finish the oldest in-flight batch once its device work is done
        while flights and all(o.is_ready() for _, _, out, _ in flights[0]['jobs'] for o in out):
            flight = flights.pop(0)
            try:
                finish(flight)
            except Exception:
                fail(flight['batch'], traceback.format_exc())
        waiting = sum(len(f['batch']) for f in flights)
        full = pending and len(pending) >= len(slots) - waiting
        if pending and (full or time.monotonic() - first >= deadline):
            if len(flights) >= inflight:   # the pipeline is full: wait for the oldest
                flight = flights.pop(0)
                try:
                    finish(flight)
                except Exception:
                    fail(flight['batch'], traceback.format_exc())
            batch, pending, first = pending, [], None
            try:
                flights.append(dispatch(batch))
                stats['max_inflight'] = max(stats['max_inflight'], len(flights))
            except Exception:
                fail(batch, traceback.format_exc())
        if flights:
            timeout = 0.0005
        elif pending:
            timeout = max(0., deadline - (time.monotonic() - first))
        else:
            timeout = None
        t_wait = time.perf_counter()
        ready = wait(conns, timeout)
        if timeout is None:
            stats['idle_s'] += time.perf_counter() - t_wait
        for c in ready:
            try:
                msg = c.recv()
            except (EOFError, OSError):
                msg = ('close',)
            try:
                handle(c, msg)
            except Exception:
                text = traceback.format_exc()
                say(dict(stage='server_error', error=text[-2000:]))
                try:
                    c.send(('error', text[-4000:]))
                except OSError:
                    pass
        if time.monotonic() - last_log > log_every:
            last_log = time.monotonic()
            say(dict(stage='server', **stats, **cpu_stats(), actors=len(slots), pending=len(pending), flights=len(flights)))
    for flight in flights:
        try:
            finish(flight)
        except Exception:
            pass
    say(dict(stage='server_stop', **stats, **cpu_stats()))
    return stats


class InferenceClient:
    """An actor's side of the server: one shared-memory slot and one pipe."""

    def __init__(self, conn, size, games, players=2):
        self.conn, self.games = conn, games
        self.layout = SlotLayout(games, wire_shapes(size, players))
        self.shm = shared_memory.SharedMemory(create=True, size=self.layout.nbytes)
        self.a = self.layout.arrays(self.shm.buf)
        self.ids = {}
        self.pack_s = self.wait_s = 0.
        conn.send(('register', self.shm.name, size, games, players))
        self._expect('ok')

    def _expect(self, kind):
        while True:   # a restarted actor may find a stale reply meant for its predecessor on the pipe
            reply = self.conn.recv()
            if reply[0] == kind:
                return reply
            if reply[0] == 'error':
                raise RuntimeError('inference server: ' + reply[1])

    def publish(self, name, version, leaves, spec):
        self.conn.send(('params', name, int(version), leaves, dict(spec)))
        self.ids[name] = self._expect('id')[2]

    def sample(self, groups, view, masks):
        """groups: {name: lane indices}. view: [games, width] int16; masks: the engine's packed [games, mask_bytes]
        uint8 rows (RustGames(served=True)). Returns {name: (choice [3 arrays], (family, source, target) packed
        path bits, logp, value)} as numpy."""
        a = self.a
        t = time.perf_counter()
        masks = np.asarray(masks)
        if masks.ndim != 2 or masks.shape[1] != a['mask'].shape[1]:
            raise ValueError(f'packed mask rows of shape {masks.shape}, the wire format has {a["mask"].shape[1]} bytes')
        rows, at = {}, 0
        for name, idx in groups.items():
            n = len(idx)
            rows[name] = slice(at, at + n)
            a['which'][at:at + n] = self.ids[name]
            a['view'][at:at + n] = view[idx]
            a['mask'][at:at + n] = masks[idx]
            at += n
        a['count'][0] = at
        self.pack_s += time.perf_counter() - t
        t = time.perf_counter()
        self.conn.send(('request',))
        self._expect('done')
        self.wait_s += time.perf_counter() - t
        out = {}
        for name in groups:
            r = rows[name]
            out[name] = ([a['choice'][r, j].copy() for j in range(3)],
                         (a['family'][r].copy(), a['source'][r].copy(), a['target'][r].copy()),
                         a['logp'][r].copy(), a['value'][r].copy())
        return out

    def close(self):
        try:
            self.conn.send(('close',))
        except (OSError, BrokenPipeError):
            pass
        self.shm.close()
        self.shm.unlink()
