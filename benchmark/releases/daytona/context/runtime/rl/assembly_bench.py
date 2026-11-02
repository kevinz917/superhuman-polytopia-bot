"""A6 learner batch assembly: equality gate, real-message capture and the throughput bench (experiments/modal_distributed_rl.py
assembly_* entrypoints; write-up docs/a6-batch-assembly-2026-10-22.md).

The gate feeds one message sequence to the host path (stream_views=False: concatenate on the host, one put) and to
the streamed path (DeviceRows) of the same Learner and checks that the device training batches are bitwise equal in
every field and that the batch statistics in `info` are identical (timings excepted).
"""
import dataclasses
import json
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np

from .distributed import END_FAMILY, VALUE_CHUNK, _new_actor_stats, decode_body, encode_msg, recv_msg, send_msg


# ----------------------------------------------------------------------------- messages

def synthetic_messages(rng, width, cells, count, rows=(40, 90), *, omni=True, turn_clock=False, shaping=False,
                       segment=8):
    """Messages shaped like an actor's (SegmentCollector rows + segment table, stack_rows layout): random packed
    views of `width` int16, all-legal masks, segments of up to `segment` rows; every segment is either cut
    (boundary views) or terminal (zeros)."""
    out = []
    for _ in range(count):
        m = int(rng.integers(*rows))
        lengths = []
        while sum(lengths) < m:
            lengths.append(int(min(rng.integers(1, segment + 1), m - sum(lengths))))
        seg = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
        pos = np.concatenate([np.arange(n, dtype=np.int32) for n in lengths])
        batch = dict(view=rng.integers(0, 3, (m, width)).astype(np.int16),
                     family_mask=np.packbits(np.ones((m, 17), bool), axis=-1),
                     source_mask=np.packbits(np.ones((m, cells), bool), axis=-1),
                     target_mask=np.packbits(np.ones((m, max(cells, 128)), bool), axis=-1),
                     choice=rng.integers(0, 16, (m, 3)).astype(np.int32),
                     logp=rng.normal(-2., .5, m).astype(np.float32), version=np.zeros(m, np.int32), seg=seg, pos=pos)
        if omni:
            batch['opp_view'] = rng.integers(0, 3, (m, width)).astype(np.int16)
        if turn_clock:
            batch['end_turn'] = batch['choice'][:, 0] == END_FAMILY
        if shaping:
            batch['phi'] = rng.normal(0., .1, m).astype(np.float32)
        terminal = rng.random(len(lengths)) < .3
        segs = dict(terminal=terminal, reward=np.where(terminal, rng.choice([-1., 1.], len(lengths)), 0.).astype(np.float32),
                    length=np.array(lengths, np.int32),
                    boot_view=np.where(terminal[:, None], 0, rng.integers(0, 3, (len(lengths), width))).astype(np.int16))
        if omni:
            segs['boot_opp_view'] = np.where(terminal[:, None], 0, rng.integers(0, 3, (len(lengths), width))).astype(np.int16)
        if shaping:
            segs['phi_next'] = rng.normal(0., .1, len(lengths)).astype(np.float32)
        out.append((batch, segs))
    return out


def with_options(messages, rng, *, turn_clock=False, shaping=False, stale_version=None):
    """Copies of messages with the turn-clock flag / shaping potentials added, and (stale_version=v) each
    message's rows before a random cut tagged version 0 and the rest v (versions grow along a segment)."""
    out = []
    for batch, segs in messages:
        batch, segs = dict(batch), dict(segs)
        m = len(batch['logp'])
        if turn_clock and 'end_turn' not in batch:
            batch['end_turn'] = batch['choice'][:, 0] == END_FAMILY
        if shaping and 'phi' not in batch:
            batch['phi'] = rng.normal(0., .1, m).astype(np.float32)
            segs['phi_next'] = rng.normal(0., .1, len(segs['length'])).astype(np.float32)
        if stale_version is not None:
            cut = int(rng.integers(0, int(batch['pos'].max()) + 2))
            batch['version'] = np.where(batch['pos'] < cut, 0, stale_version).astype(np.int32)
        out.append((batch, segs))
    return out


def decode(body):
    """A captured message file (any send_msg framing) -> (batch, segments)."""
    size, kind = struct.unpack('!QB', body[:9])
    msg = decode_body(body[9:], kind)
    return msg['batch'], msg['segments']


def frame(batch, segs, actor=0):
    """What an actor's send_msg puts on the wire for this data message (today's framing)."""
    return encode_msg(dict(type='data', actor=actor, have=[], batch=batch, stats=_new_actor_stats(), segments=segs))


# ----------------------------------------------------------------------------- gate

def assemble_once(learner, messages, stream, coded=False):
    """One batch from `messages` (cycled as needed) through _assemble + _device_batch with stream_views=stream
    (coded: coded_views, and `messages` must be view_codec-encoded)."""
    learner.config = dataclasses.replace(learner.config, stream_views=stream, coded_views=coded)
    learner.deadline = float('inf')
    while not learner.inbox.empty():
        learner.inbox.get()
    stop = threading.Event()

    def feed():
        i = 0
        while not stop.is_set():
            batch, segs = messages[i % len(messages)]
            try:
                learner.inbox.put((dict(batch), dict(segs) if segs is not None else None), timeout=.1)
                i += 1
            except Exception:
                pass
    thread = threading.Thread(target=feed, daemon=True)
    thread.start()
    try:
        data, info = learner._assemble()
        batch = learner._device_batch(data)
        learner.jax.block_until_ready(batch)
    finally:
        stop.set()
        thread.join()
        while not learner.inbox.empty():
            learner.inbox.get()
    return batch, info


def bitwise_equal(a, b):
    import jax
    import jax.numpy as jnp
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if jnp.issubdtype(a.dtype, jnp.floating):
        kind = {2: jnp.int16, 4: jnp.int32, 8: jnp.int64}[a.dtype.itemsize]
        a, b = jax.lax.bitcast_convert_type(a, kind), jax.lax.bitcast_convert_type(b, kind)
    return bool(jax.jit(lambda x, y: jnp.all(x == y))(a, b))


def gate(learner, messages):
    """Host path vs streamed path on the same messages: per-field bitwise equality, shardings, info (timings
    excepted). The first message cycle is identical for both since each assembles from the start of `messages`."""
    host, host_info = assemble_once(learner, messages, False)
    streamed, streamed_info = assemble_once(learner, messages, True)
    fields = {k: (k in host and k in streamed and bitwise_equal(host[k], streamed[k])
                  and host[k].sharding.is_equivalent_to(streamed[k].sharding, host[k].ndim))
              for k in sorted(set(host) | set(streamed))}
    stats = lambda info: {k: v for k, v in info.items() if not k.endswith('_seconds')}
    timings = lambda info: {k: round(v, 4) for k, v in info.items() if k.endswith('_seconds')}
    info_equal = stats(host_info) == stats(streamed_info)
    return dict(equal=all(fields.values()) and info_equal, fields=fields, info_equal=info_equal,
                info=stats(host_info), info_streamed=None if info_equal else stats(streamed_info),
                rows=int(len(host['logp'])), host_timings=timings(host_info), streamed_timings=timings(streamed_info))


def coded_gate(learner, messages):
    """Dense streamed path vs coded views (view_codec) on the same messages: bitwise batch fields and info."""
    from .view_codec import encode
    dense, dense_info = assemble_once(learner, messages, True)
    coded, coded_info = assemble_once(learner, [encode(dict(b), dict(s)) for b, s in messages], True, coded=True)
    learner.config = dataclasses.replace(learner.config, coded_views=False)
    fields = {k: (k in dense and k in coded and bitwise_equal(dense[k], coded[k])
                  and dense[k].sharding.is_equivalent_to(coded[k].sharding, dense[k].ndim))
              for k in sorted(set(dense) | set(coded))}
    stats = lambda info: {k: v for k, v in info.items() if not k.endswith('_seconds')}
    info_equal = stats(dense_info) == stats(coded_info)
    return dict(equal=all(fields.values()) and info_equal, fields=fields, info_equal=info_equal,
                info=stats(dense_info), info_coded=None if info_equal else stats(coded_info),
                rows=int(len(dense['logp'])), dense_timings={k: round(v, 4) for k, v in dense_info.items() if k.endswith('_seconds')},
                coded_timings={k: round(v, 4) for k, v in coded_info.items() if k.endswith('_seconds')})


# ----------------------------------------------------------------------------- capture (CPU container)

def capture(learner, out_dir, count, processes, games, log=print, skip=0):
    """Real actor messages: actor processes on this container play against `learner` (served on localhost; it is
    never trained, so every row is version 0) and the first `count` data messages are written to out_dir in the
    send_msg framing. skip: data messages dropped first (the opening positions, which compress far better than
    mid-game ones)."""
    from .distributed import run_actor_container
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    learner.bind_host = '127.0.0.1'
    learner.serve()
    stop = threading.Event()
    thread = threading.Thread(target=run_actor_container, args=(('127.0.0.1', learner.port), processes, games),
                              kwargs=dict(should_stop=stop.is_set), daemon=True)
    thread.start()
    rows, started = [], time.monotonic()
    for i in range(skip):
        learner.inbox.get(timeout=1800)
        if i % 500 == 0:
            log(json.dumps(dict(stage='skipped', messages=i, seconds=time.monotonic() - started)))
    for i in range(count):
        batch, segs = learner.inbox.get(timeout=1800)
        (out / f'msg-{i:04d}.bin').write_bytes(frame(batch, segs, actor=i))
        rows.append(len(batch['logp']))
        if i % 16 == 0:
            log(json.dumps(dict(stage='captured', messages=i + 1, rows=int(sum(rows)), seconds=time.monotonic() - started)))
    learner.stop = True
    stop.set()
    while not learner.inbox.empty():   # unblock receiver threads so the actors see stop
        learner.inbox.get()
    thread.join(timeout=120)
    return dict(messages=count, rows=rows, width=int(batch['view'].shape[1]), keys=sorted(batch),
                segment_keys=sorted(segs), bytes=sum(p.stat().st_size for p in out.glob('msg-*.bin')))


# ----------------------------------------------------------------------------- throughput clients

def restamp(batch, version, lag, segment=32):
    """batch with its rows' versions set as a live actor's would be at learner `version`: the message is `lag`
    versions behind and versions grow along each segment, the oldest third of a segment's rows 2 behind that."""
    batch = dict(batch)
    behind = 2 - np.minimum(2, np.asarray(batch['pos']) * 3 // segment)
    batch['version'] = np.maximum(0, version - lag - behind).astype(np.int32)
    return batch


def client_process(port, paths, connections, first_actor, seconds, coded=False, lag=None):
    """Unthrottled actor stand-in: `connections` threads, each sends the captured messages round robin and reads
    the learner's replies (weights included), until the learner says stop or `seconds` pass. coded: the messages
    go out view_codec-encoded, as actors with coded_views send them. lag=(low, high): every send restamps the
    rows' versions `lag` (uniform in [low, high]) behind the learner's last reply and re-encodes the frame, so the
    learner's max_lag filter drops a live run's share of stale rows (without lag every row is version 0)."""
    from .view_codec import encode
    if coded:
        messages = [encode(*decode(Path(p).read_bytes())) for p in paths]
    else:
        messages = [decode(Path(p).read_bytes()) for p in paths]   # captured files may predate the framing
    bodies = None if lag else [frame(*m) for m in messages]
    end = time.time() + seconds

    def one(c):
        for _ in range(600):
            try:
                sock = socket.create_connection(('127.0.0.1', port), timeout=600)
                break
            except OSError:
                time.sleep(1)
        else:
            return
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # live actors run relayed (lever 2): their replies carry versions, not weights
        send_msg(sock, dict(type='hello', actor=first_actor + c, have=[], relayed=bool(lag)))
        version = recv_msg(sock).get('version', 0)
        i = first_actor + c
        rng = np.random.default_rng(first_actor + c)
        try:
            while time.time() < end:
                if bodies is None:
                    batch, segs = messages[i % len(messages)]
                    sock.sendall(frame(restamp(batch, version, int(rng.integers(lag[0], lag[1] + 1))), segs))
                else:
                    sock.sendall(bodies[i % len(bodies)])
                i += 1
                reply = recv_msg(sock)
                version = reply.get('version', version)
                if reply.get('stop'):
                    break
        except (OSError, ConnectionError, EOFError):
            pass
        sock.close()
    threads = [threading.Thread(target=one, args=(c,), daemon=True) for c in range(connections)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def summarize(history, skip=3):
    rows = [r for r in history if r['update'] > skip]
    if not rows:
        return dict(updates=len(history))
    keys = ['seconds', 'ready_wait_seconds', 'train_seconds', 'device_batch_seconds', 'inbox', 'messages'] + sorted(
        k for k in rows[0] if k.startswith(('asm_', 'receiver_')))
    span = rows[-1]['elapsed'] - rows[0]['elapsed'] + rows[0]['seconds']
    batch = rows[0]['samples'] // rows[0]['update']
    return dict(updates=len(history), steady_updates=len(rows),
                samples_per_second=len(rows) * batch / span,
                **{f'mean_{k}': float(np.mean([r[k] for r in rows])) for k in keys if k in rows[0]})


# ----------------------------------------------------------------------------- upload codec (lever 4)

def _shuffle(a):
    """Byte planes of an int16 array (all low bytes, then all high bytes): small values compress better."""
    return np.ascontiguousarray(a.view(np.uint8).reshape(-1, 2).T)


def _narrow(a):
    """int16 -> int8 when every value fits (lossless), else None."""
    return a.astype(np.int8) if a.size and -128 <= a.min() and a.max() <= 127 else None


def codec_bench(paths, levels=(1, 3, 6), repeats=3):
    """Bytes and single-thread seconds per real actor message for zstd levels on today's kind-2 body, and on two
    cheap lossless reformats of the int16 view arrays (byte planes; int8 when the values fit). Encode time is the
    actor's cost (pickle + zstd on the sender thread, which shares the process's pinned core); decode time is the
    learner receiver's (plus the inverse reformat)."""
    import pickle
    import zstandard
    views = ('view', 'opp_view', 'boot_view', 'boot_opp_view')
    msgs = [decode(Path(p).read_bytes()) for p in paths]
    rows = sum(len(b['logp']) for b, _ in msgs)

    def body(batch, segs, reformat):
        batch, segs = dict(batch), dict(segs)
        for part in (batch, segs):
            for k in views:
                if k in part:
                    if reformat == 'shuffle':
                        part[k] = _shuffle(part[k])
                    elif reformat == 'int8':
                        n = _narrow(part[k])
                        part[k] = part[k] if n is None else n
        obj = dict(type='data', actor=0, have=[], batch=batch, stats=_new_actor_stats(), segments=segs)
        buffers = []
        head = pickle.dumps(obj, protocol=5, buffer_callback=buffers.append)
        raws = [b.raw() for b in buffers]
        index = struct.pack(f'!QI{len(raws)}Q', len(head), len(raws), *(r.nbytes for r in raws))
        return b''.join([index, head, *raws])

    out = dict(messages=len(msgs), rows=rows, view_min=int(min(b['view'].min() for b, _ in msgs)),
               view_max=int(max(b['view'].max() for b, _ in msgs)),
               int8_fits=float(np.mean([_narrow(b['view']) is not None for b, _ in msgs])), variants={})
    for reformat in ('none', 'shuffle', 'int8'):
        for level in levels:
            if reformat != 'none' and level not in (1, 3):
                continue
            cctx, dctx = zstandard.ZstdCompressor(level=level), zstandard.ZstdDecompressor()
            raw_bytes = wire = 0
            encode_s = decode_s = 0.
            for _ in range(repeats):
                raw_bytes = wire = 0
                for batch, segs in msgs:
                    t = time.perf_counter()
                    data = body(batch, segs, reformat)
                    packed = cctx.compress(data)
                    encode_s += time.perf_counter() - t
                    t = time.perf_counter()
                    plain = dctx.decompress(packed)
                    # the learner's inverse, over the whole body (an upper bound: only the view arrays need it)
                    if reformat == 'shuffle':
                        np.frombuffer(plain, np.uint8)[:len(plain) // 2 * 2].reshape(2, -1).T.copy()
                    elif reformat == 'int8':
                        np.frombuffer(plain, np.int8).astype(np.int16)
                    decode_s += time.perf_counter() - t
                    raw_bytes += len(data)
                    wire += len(packed)
            out['variants'][f'{reformat}-zstd{level}'] = dict(
                raw_bytes_per_row=raw_bytes / rows, wire_bytes_per_row=wire / rows, ratio=raw_bytes / wire,
                encode_ms_per_message=1e3 * encode_s / repeats / len(msgs),
                decode_ms_per_message=1e3 * decode_s / repeats / len(msgs),
                encode_us_per_row=1e6 * encode_s / repeats / rows)
    return out


# ----------------------------------------------------------------------------- in-process stack sampler

class StackSampler:
    """Samples every thread's Python stack every `interval` seconds (sys._current_frames; no ptrace needed).
    report(): per thread kind (the thread's target name), the share of samples and the hottest frames."""

    def __init__(self, interval=.005, depth=4):
        self.interval, self.depth = interval, depth
        self.counts, self.totals, self.stop = {}, {}, threading.Event()
        self.thread = threading.Thread(target=self._run, name='stack-sampler', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _kind(self, name):
        return name[name.index('(') + 1:name.rindex(')')] if '(' in name and name.endswith(')') else name.split('-')[0]

    def _run(self):
        import sys
        me = threading.get_ident()
        while not self.stop.wait(self.interval):
            names = {t.ident: t.name for t in threading.enumerate()}
            for ident, frame in sys._current_frames().items():
                if ident == me:
                    continue
                kind = self._kind(names.get(ident, 'unknown'))
                stack, f = [], frame
                while f is not None and len(stack) < self.depth:
                    stack.append(f'{Path(f.f_code.co_filename).name}:{f.f_code.co_name}:{f.f_lineno}')
                    f = f.f_back
                key = (kind, ' <- '.join(stack))
                self.counts[key] = self.counts.get(key, 0) + 1
                self.totals[kind] = self.totals.get(kind, 0) + 1

    def report(self, top=12):
        self.stop.set()
        out = {}
        for kind, total in sorted(self.totals.items(), key=lambda x: -x[1]):
            rows = sorted(((c, s) for (k, s), c in self.counts.items() if k == kind), reverse=True)[:top]
            out[kind] = dict(samples=total, top=[dict(share=round(c / total, 3), stack=s) for c, s in rows])
        return out
