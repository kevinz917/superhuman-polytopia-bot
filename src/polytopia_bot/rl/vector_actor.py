"""Vectorized actor: one process per container steps every game of the container and runs the policy on the
node's GPU itself (docs/throughput-multiples-2026-11-06.md).

It replaces, per container, the 48 actor processes + in-node inference server (run_actor_container with
inference='cuda'), whose single Python server loop capped a container at ~30k decisions/s (2026-11-06 bench:
actors waiting 84% of the time, GPU 34% busy, cores idle). Here the per-lane work is done in bulk:

* the Rust engine steps a group of a few hundred games per call on worker threads (GameBatch.set_threads, GIL
  released) and returns the views and packed masks of every game in one array each;
* the policy forward of a whole group is one jitted call on the GPU (no shared-memory slots, pipes, gathers or
  per-actor replies); groups are pipelined, so the GPU runs one group's forward while the CPU steps another;
* training rows are not built one Python dict at a time: per-row fields go into ring arrays by fancy index, the
  views stay in the tick arrays the engine returned, and a message gathers its rows from those arrays once;
* maps for finished games are generated ahead of time in worker processes;
* messages are compressed by several sender threads (multi-threaded zstd, GIL released) on their own learner
  connections, so compression never runs on the stepping thread.

The learner sees exactly the old actors' protocol: hello, data messages of SegmentCollector rows (view,
opp_view, family/source/target masks, choice, logp, version, end_turn, seg, pos) and segment tables, stats, and
replies carrying weights and the pool. Game logic (opponent assignment per block of lanes, episode outcomes,
cap score, stats) follows distributed._actor_process line for line; tests/test_vector_actor.py checks the
segment emission against SegmentCollector.
"""
import json
import os
import queue
import struct
import sys
import threading
import time

import numpy as np

from .distributed import (END_FAMILY, SUMMON_FAMILY, _connect, _new_actor_stats, _template_params, episode_outcome,
                          other_seats, recv_msg)


# ----------------------------------------------------------------------------- wire (multi-threaded zstd)

def encode_frame(obj, level=7, threads=2):
    """distributed.encode_msg's kind-2 frame (zstd of index + out-of-band pickle + raw buffers), streamed into a
    multi-threaded zstd compressor without first joining the buffers into one bytes object."""
    import pickle
    import zstandard
    buffers = []
    head = pickle.dumps(obj, protocol=5, buffer_callback=buffers.append)
    raws = [b.raw() for b in buffers]
    index = struct.pack(f'!QI{len(raws)}Q', len(head), len(raws), *(r.nbytes for r in raws))
    total = len(index) + len(head) + sum(r.nbytes for r in raws)
    cctx = zstandard.ZstdCompressor(level=level, threads=threads, write_content_size=True)
    obj_ = cctx.compressobj(size=total)
    parts = [obj_.compress(index), obj_.compress(head)]
    for r in raws:
        parts.append(obj_.compress(r))
    parts.append(obj_.flush())
    body = b''.join(parts)
    return struct.pack('!QB', len(body), 2) + body


# ----------------------------------------------------------------------------- map prefetch

def _map_worker(mix, size, players, jobs, results):
    from polytopia_bot.maps.stream import MapSource
    source = MapSource(mix, size, players=players)
    while True:
        job = jobs.get()
        if job is None:
            return
        i, seed = job
        kind, layout = source.layout(int(seed))
        results.put((i, int(seed), kind, tuple(np.asarray(a, np.int32) for a in (layout.terrain, layout.structure,
                                                                                   layout.resource, layout.capitals))))


class MapPrefetch:
    """Layouts for the seeds the actor will need, generated ahead by worker processes and handed out in seed
    order (so a run's map sequence depends only on its rng, not on worker timing). 'classic' maps come from the
    engine and are cheap; the other kinds are Python generators (tens of ms each)."""

    def __init__(self, mix, size, players, seeds, workers=4, ahead=256):
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        self.seeds, self.ahead = seeds, ahead   # seeds(): the next seed from the actor's rng
        self.jobs, self.results = ctx.Queue(), ctx.Queue()
        self.procs = [ctx.Process(target=_map_worker, args=(mix, size, players, self.jobs, self.results), daemon=True)
                      for _ in range(workers)]
        for p in self.procs:
            p.start()
        self.submitted, self.next, self.ready = 0, 0, {}
        self.wait_s = 0.
        self._fill()

    def _fill(self):
        while self.submitted - self.next < self.ahead:
            self.jobs.put((self.submitted, int(self.seeds())))
            self.submitted += 1

    def get(self):
        """(seed, kind, (terrain, structure, resource, capitals)) of the next seed."""
        t = time.perf_counter()
        while self.next not in self.ready:
            i, seed, kind, layout = self.results.get()
            self.ready[i] = (seed, kind, layout)
        self.wait_s += time.perf_counter() - t
        out = self.ready.pop(self.next)
        self.next += 1
        self._fill()
        return out

    def close(self):
        for _ in self.procs:
            self.jobs.put(None)
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


MAP_TABLE_DIRS = ('/results/maps', os.path.expanduser('~/.cache/polytopia/maps'))


class MapTable:
    """Every board of a map mix for a seed range, precomputed (experiments/modal_map_table.py): the same layouts
    MapSource.layout(seed) returns, looked up instead of generated."""

    def __init__(self, path, seeds):
        with np.load(path) as z:
            self.lo, self.hi = int(z['lo']), int(z['hi'])
            self.types = [str(t) for t in z['types']]
            self.kind = z['kind']
            self.arrays = tuple(np.ascontiguousarray(z[k], np.int32) for k in ('terrain', 'structure', 'resource', 'capitals'))
        self.seeds, self.wait_s = seeds, 0.

    @staticmethod
    def find(mix, size, players, lo, hi):
        name = f'table-{mix}-{size}-{players}p-{lo}-{hi}.npz'
        for d in MAP_TABLE_DIRS:
            if os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
        return None

    def get(self):
        seed = int(self.seeds())
        i = seed - self.lo
        return seed, self.types[int(self.kind[i])], tuple(a[i] for a in self.arrays)

    def close(self):
        pass


# ----------------------------------------------------------------------------- rows

class RowStore:
    """Training rows of every lane of the actor, without per-row Python objects.

    A row is an integer id. Its small fields live in ring arrays indexed by id % capacity; its two views stay in
    the arrays the engine returned for that tick (`tick_views[slot]`, rows of the lanes that moved), referenced by
    (slot, local lane). Per (lane, seat) the ids of the current partial segment sit in `seg_rows[lane, seat, :count]`.
    Segments are emitted exactly as SegmentCollector emits them: a seat's (segment + 1)-th row cuts the first
    `segment` rows, with the extra row as the boundary (and the first row of the next segment); a finished game
    emits each seat's remaining rows. emit() returns nothing; take() hands the emitted segments over."""

    FIELDS = (('family_mask', np.uint8), ('source_mask', np.uint8), ('target_mask', np.uint8), ('choice', np.int32),
              ('logp', np.float32), ('version', np.int32), ('end_turn', np.bool_))

    def __init__(self, lanes, players, segment, widths, capacity=1 << 18, turn_clock=True, omni=True):
        self.lanes, self.players, self.segment = lanes, players, segment
        self.capacity, self.turn_clock, self.omni = capacity, turn_clock, omni
        shape = dict(family_mask=(widths['family'],), source_mask=(widths['source'],), target_mask=(widths['target'],),
                     choice=(3,), logp=(), version=(), end_turn=())
        self.ring = {k: np.zeros((capacity,) + shape[k], dt) for k, dt in self.FIELDS}
        self.slot = np.zeros(capacity, np.int64)     # the tick slot holding the row's views
        self.local = np.zeros(capacity, np.int32)    # the row's index in that slot's view arrays
        self.next_id = 0
        self.seg_rows = np.zeros((lanes, players, segment + 1), np.int64)
        self.count = np.zeros((lanes, players), np.int32)
        self.views = {}                               # slot -> (view [m, vw], opp_view [m, vw] or None)
        self.next_slot = 0
        # emitted segments not yet taken: row ids (concatenated) and per-segment table entries
        self.out_rows, self.out_len, self.out_term, self.out_reward = [], [], [], []
        self.out_boot = []                            # per segment: ('row', id) or ('view', (view, opp)) or None
        self.emitted_rows = 0
        self.spilled = 0

    def add(self, lanes, seats, view, opp_view, local, fields):
        """One tick's learner rows: lanes [m] (global), seats [m] (1-based), the tick's view/opp_view arrays (kept by
        reference, no copy) and the rows' indexes `local` [m] in them, fields {name: [m, ...]}. Returns the ids."""
        m = len(lanes)
        if m == 0:
            return np.zeros(0, np.int64)
        ids = self.next_id + np.arange(m, dtype=np.int64)
        self.next_id += m
        at = ids % self.capacity
        for k, a in fields.items():
            self.ring[k][at] = a
        slot = self.next_slot
        self.next_slot += 1
        self.views[slot] = (view, opp_view)
        self.slot[at] = slot
        self.local[at] = local
        s = seats - 1
        c = self.count[lanes, s]
        self.seg_rows[lanes, s, c] = ids
        self.count[lanes, s] = c + 1
        full = np.flatnonzero(c + 1 > self.segment)
        if len(full):
            fl, fs = lanes[full], s[full]
            heads = self.seg_rows[fl, fs, :self.segment]
            boots = self.seg_rows[fl, fs, self.segment]
            for j in range(len(full)):
                self.out_rows.append(heads[j].copy())
                self.out_len.append(self.segment)
                self.out_term.append(False)
                self.out_reward.append(0.)
                self.out_boot.append(('row', int(boots[j])))
            self.emitted_rows += self.segment * len(full)
            self.seg_rows[fl, fs, 0] = boots
            self.count[fl, fs] = 1
        return ids

    def finish(self, lane, rewards, terminated, final_views):
        """SegmentCollector.finish for one lane: final_views [players, vw] (copied when a boundary needs them)."""
        for seat in range(1, self.players + 1):
            c = int(self.count[lane, seat - 1])
            if c:
                if terminated:
                    boot = None
                else:
                    others = final_views[2 - seat] if self.players == 2 else final_views[other_seats(seat, self.players)]
                    boot = ('view', (np.array(final_views[seat - 1]), np.array(others)))
                self.out_rows.append(self.seg_rows[lane, seat - 1, :c].copy())
                self.out_len.append(c)
                self.out_term.append(bool(terminated))
                self.out_reward.append(float(rewards[seat - 1]))
                self.out_boot.append(boot)
                self.emitted_rows += c
        self.count[lane] = 0

    def drop(self, lane, seats):
        """Forget a lane's partial rows of the given seats (1-based): a pool game records the learner seat only."""
        for seat in seats:
            self.count[lane, seat - 1] = 0

    def _live(self):
        """Every live row id: partial segments (seg_rows[l, s, :count]), emitted rows and their row boundaries."""
        parts = []
        mask = np.arange(self.segment + 1)[None, None, :] < self.count[..., None]
        if mask.any():
            parts.append(self.seg_rows[mask])
        parts.extend(self.out_rows)
        boots = [b[1] for b in self.out_boot if b is not None and b[0] == 'row']
        if boots:
            parts.append(np.array(boots, np.int64))
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    def release(self, max_age=96):
        """Free the tick slots no live row references. Rows still live in slots older than `max_age` slots (a seat
        whose segment stays open while the other seat plays long turns) are first copied into one fresh compact
        slot under new ids, so a few old rows never pin whole tick arrays (or outlive the ring)."""
        live = self._live()
        cutoff = self.next_slot - max_age
        if len(live):
            old = np.unique(live[self.slot[live % self.capacity] < cutoff])
            if len(old):
                self._spill(old)
                live = self._live()
        keep_from = int(self.slot[live % self.capacity].min()) if len(live) else self.next_slot
        for slot in [x for x in self.views if x < keep_from]:
            del self.views[slot]

    def _spill(self, old):
        """Move rows `old` (sorted unique ids) to new ids whose views sit in one new slot of copies."""
        order, slots, local, refs = self._refs(old)
        old_sorted = old[order]
        width = next(iter(refs.values()))[0].shape[1]
        view = np.empty((len(old), width), np.int16)
        opp = np.empty((len(old), width), np.int16) if self.omni else None
        for a, b in _runs(slots):
            v, o = refs[int(slots[a])]
            np.take(v, local[a:b], axis=0, out=view[a:b])
            if opp is not None:
                np.take(o, local[a:b], axis=0, out=opp[a:b])
        new = self.next_id + np.arange(len(old), dtype=np.int64)
        self.next_id += len(old)
        src, dst = old_sorted % self.capacity, new % self.capacity
        for k in self.ring:
            self.ring[k][dst] = self.ring[k][src]
        slot = self.next_slot
        self.next_slot += 1
        self.views[slot] = (view, opp)
        self.slot[dst] = slot
        self.local[dst] = np.arange(len(old), dtype=np.int32)
        self.spilled += len(old)
        # remap every reference: old_sorted[j] -> new[j]
        key = np.argsort(old_sorted)
        ks, kn = old_sorted[key], new[key]

        def remap(a):
            j = np.searchsorted(ks, a)
            j = np.minimum(j, len(ks) - 1)
            hit = ks[j] == a
            out = a.copy()
            out[hit] = kn[j[hit]]
            return out
        mask = np.arange(self.segment + 1)[None, None, :] < self.count[..., None]
        if mask.any():
            self.seg_rows[mask] = remap(self.seg_rows[mask])
        self.out_rows = [remap(r) for r in self.out_rows]
        self.out_boot = [('row', int(remap(np.array([b[1]], np.int64))[0])) if b is not None and b[0] == 'row' else b
                         for b in self.out_boot]

    def check(self):
        """Raise if the live rows span more ids than the ring holds (their small fields would be overwritten)."""
        live = self._live()
        if len(live) and self.next_id - int(live.min()) > self.capacity:
            raise RuntimeError('RowStore ring overflow: raise capacity')

    def _refs(self, ids):
        """(slots, local indexes) of rows `ids` sorted by slot, the permutation applied, and the slot arrays."""
        at = ids % self.capacity
        slots = self.slot[at]
        order = np.argsort(slots, kind='stable')
        slots = slots[order]
        return order, slots, self.local[at][order], {int(x): self.views[int(x)] for x in np.unique(slots)}

    def take(self):
        """The emitted segments as a MessageJob (cheap: the view gathers run in job.build(), on a sender thread),
        or None. Rows are ordered by tick slot, not by segment: the learner reads (seg, pos) only."""
        if not self.out_rows:
            return None
        lengths = np.array(self.out_len, np.int32)
        ids = np.concatenate(self.out_rows)
        seg = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
        pos = np.concatenate([np.arange(n, dtype=np.int32) for n in lengths])
        order, slots, local, refs = self._refs(ids)
        ids, seg, pos = ids[order], seg[order], pos[order]
        at = ids % self.capacity
        small = {k: self.ring[k][at] for k, _ in self.FIELDS if k != 'end_turn' or self.turn_clock}
        row_segs = np.array([j for j, b in enumerate(self.out_boot) if b is not None and b[0] == 'row'], np.int64)
        boot_rows = None
        if len(row_segs):
            bids = np.array([self.out_boot[j][1] for j in row_segs], np.int64)
            border, bslots, blocal, brefs = self._refs(bids)
            refs.update(brefs)
            boot_rows = (row_segs[border], bslots, blocal)
        boot_views = [(j, b[1]) for j, b in enumerate(self.out_boot) if b is not None and b[0] == 'view']
        job = MessageJob(small, seg, pos, slots, local, refs, boot_rows, boot_views,
                         dict(terminal=np.array(self.out_term, bool), reward=np.array(self.out_reward, np.float32),
                              length=lengths), self.omni)
        self.out_rows, self.out_len, self.out_term, self.out_reward, self.out_boot = [], [], [], [], []
        self.emitted_rows = 0
        return job


def _runs(slots):
    cuts = np.flatnonzero(np.diff(slots)) + 1
    return zip(np.concatenate(([0], cuts)).tolist(), np.concatenate((cuts, [len(slots)])).tolist())


class MessageJob:
    """A data message whose views are still references into tick arrays; build() gathers them (numpy take
    releases the GIL, so several sender threads build in parallel with the stepping thread)."""

    def __init__(self, small, seg, pos, slots, local, refs, boot_rows, boot_views, table, omni):
        self.small, self.seg, self.pos, self.slots, self.local = small, seg, pos, slots, local
        self.refs, self.boot_rows, self.boot_views, self.table, self.omni = refs, boot_rows, boot_views, table, omni
        self.rows = len(seg)

    def _gather(self, slots, local, which, out):
        for a, b in _runs(slots):
            np.take(self.refs[int(slots[a])][which], local[a:b], axis=0, out=out[a:b])

    def build_coded(self, threads=4):
        """The message in view_codec's coded layout (sparse row-to-row deltas; rows in segment order), encoded by
        engine-rs coo_deltas straight from the tick arrays: equal to view_codec.encode of the dense message."""
        from polytopia_engine_rs import coo_deltas
        if not self.omni:
            raise ValueError('coded vector-actor messages need the omniscient two-view rows')
        order = np.lexsort((self.pos, self.seg))
        seg, pos = self.seg[order], self.pos[order]
        slots, local = self.slots[order], self.local[order]
        batch = {k: v[order] for k, v in self.small.items()}
        batch['seg'], batch['pos'] = seg, pos
        arrays, index = [], {}
        for slot, (v, o) in self.refs.items():
            index[slot] = len(arrays)
            arrays += [v, o]
        width = arrays[0].shape[1]
        m, n_seg = len(seg), len(self.table['length'])
        src = np.empty((m, 2, 2), np.int64)
        iv = np.array([index[int(x)] for x in slots], np.int64) if m else np.zeros(0, np.int64)
        src[:, 0, 0], src[:, 1, 0] = iv, iv + 1
        src[:, :, 1] = local[:, None]
        base = np.full((m, 2, 2), -1, np.int64)
        cont = np.flatnonzero(pos > 0)
        base[cont] = src[cont - 1]
        r, st, c, v = coo_deltas(arrays, src, base, threads)
        batch.update(view_coo_row=r, view_coo_stream=st, view_coo_col=c, view_coo_val=v,
                     view_shape=np.array([m, 2, width], np.int32))
        # boundaries: from the segment's last row (non-terminal), zero views (terminal: no entries)
        bsrc = np.full((n_seg, 2, 2), -1, np.int64)
        if self.boot_rows is not None:
            where, bslots, blocal = self.boot_rows
            bi = np.array([index[int(x)] for x in bslots], np.int64)
            bsrc[where, 0, 0], bsrc[where, 1, 0] = bi, bi + 1
            bsrc[where, 0, 1] = bsrc[where, 1, 1] = blocal
        if self.boot_views:
            ev = np.stack([b[0] for _, b in self.boot_views])
            eo = np.stack([b[1] for _, b in self.boot_views])
            at = len(arrays)
            arrays += [ev, eo]
            js = np.array([j for j, _ in self.boot_views], np.int64)
            bsrc[js, 0, 0], bsrc[js, 1, 0] = at, at + 1
            bsrc[js, 0, 1] = bsrc[js, 1, 1] = np.arange(len(js))
        terminal = np.asarray(self.table['terminal'], bool)
        last = np.cumsum(np.asarray(self.table['length'], np.int64)) - 1
        bbase = np.full((n_seg, 2, 2), -1, np.int64)
        live = np.flatnonzero(~terminal)
        bbase[live] = src[last[live]]
        bsrc[terminal] = -1
        r, st, c, v = coo_deltas(arrays, bsrc, bbase, threads)
        segs = dict(self.table)
        segs.update(boot_coo_row=r, boot_coo_stream=st, boot_coo_col=c, boot_coo_val=v,
                    boot_shape=np.array([n_seg, 2, width], np.int32))
        self.refs = None
        return batch, segs

    def build(self):
        any_ref = next(iter(self.refs.values()))[0]
        width = any_ref.shape[1]
        n, n_seg = self.rows, len(self.table['length'])
        batch = dict(self.small)
        batch['view'] = np.empty((n, width), np.int16)
        self._gather(self.slots, self.local, 0, batch['view'])
        if self.omni:
            batch['opp_view'] = np.empty((n, width), np.int16)
            self._gather(self.slots, self.local, 1, batch['opp_view'])
        batch['seg'], batch['pos'] = self.seg, self.pos
        segs = dict(self.table)
        segs['boot_view'] = np.zeros((n_seg, width), np.int16)
        if self.omni:
            segs['boot_opp_view'] = np.zeros((n_seg, width), np.int16)
        if self.boot_rows is not None:
            where, bslots, blocal = self.boot_rows
            tmp = np.empty((len(where), width), np.int16)
            self._gather(bslots, blocal, 0, tmp)
            segs['boot_view'][where] = tmp
            if self.omni:
                self._gather(bslots, blocal, 1, tmp)
                segs['boot_opp_view'][where] = tmp
        for j, (v, o) in self.boot_views:
            segs['boot_view'][j] = v
            if self.omni:
                segs['boot_opp_view'][j] = o
        self.refs = None
        return batch, segs


class ViewPool:
    """Reusable (view, opp_view) output arrays for GameBatch.tick_into. An array is free again once nothing but
    the pool references it (the row store dropped its tick slot, the messages that gathered from it are built and
    the device copy that read it is done), judged by its reference count. Reusing arrays keeps the engine's
    worker threads off fresh pages (first-touch faults serialize them on the address-space lock)."""

    def __init__(self, rows, width, omni):
        self.rows, self.width, self.omni = rows, width, omni
        self.items, self.at = [], 0
        self.allocated = 0

    def get(self):
        n = len(self.items)
        for k in range(n):
            j = (self.at + k) % n
            item = self.items[j]
            # references when free: the list `item`, the getrefcount argument
            if sys.getrefcount(item[0]) <= 2 and (item[1] is None or sys.getrefcount(item[1]) <= 2):
                self.at = j + 1
                return item[0], item[1]
        view = np.zeros((self.rows, self.width), np.int16)
        opp = np.zeros((self.rows, self.width), np.int16) if self.omni else np.zeros((0, self.width), np.int16)
        view.fill(1)   # touch every page now, in this thread
        view.fill(0)
        opp.fill(0)
        self.items.append([view, opp if self.omni else None])
        self.allocated += 1
        return view, opp


# ----------------------------------------------------------------------------- the actor

def _bucket(n, top):
    b = 32
    while b < n:
        b *= 2
    return min(b, top) if n <= top else n


def run_vector_actor(address, actor_id, lanes=1152, *, groups=2, threads=None, block=24, send_rows=8192,
                     senders=4, zstd_threads=2, map_workers=4, platform='cuda', trunk_dtype='bfloat16',
                     precision='DEFAULT', should_stop=None, report=None, report_every=20., connect_timeout=900,
                     seed=None, profile=False, zstd_level=None, spill_age=512, act_dtype='float32',
                     stop_check_every=5., pad_stem=0):
    """Run until the learner says stop (or should_stop()). report(dict) gets progress summaries.

    lanes: games in the container, split into `groups` engine batches stepped in turn (the GPU runs one group's
    forward while the CPU steps the next); threads: engine worker threads per tick (default: the CPUs minus the
    senders); block: lanes sharing one pool opponent (the old per-process unit, pool_per_process); send_rows:
    emitted rows per data message."""
    os.environ['JAX_PLATFORMS'] = platform
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ['POLYTOPIA_TRUNK_DTYPE'] = trunk_dtype   # read when network is imported: before it
    os.environ['POLYTOPIA_MATMUL_PRECISION'] = precision
    os.environ.setdefault('POLYTOPIA_HEAD_PRECISION', 'HIGHEST')
    os.environ['POLYTOPIA_ACT_DTYPE'] = act_dtype   # network.ACT_DTYPE: bf16 activations in this policy forward
    os.environ['POLYTOPIA_PAD_STEM'] = str(pad_stem)   # network.PAD_STEM: stem input channels padded (same values)
    import jax
    import jax.numpy as jnp
    from polytopia_bot.gpu import ENGINE_REVISION
    from polytopia_bot.gpu.actions import Masks
    from polytopia_bot.ruleset import rules_json, training_round_limit
    from polytopia_engine_rs import GameBatch
    from . import network
    from .cpu_engine import view_features, view_layout
    from .inference_server import wire_shapes
    from .distributed import send_msg

    started = time.monotonic()
    clock = time.perf_counter
    say = report or (lambda d: None)
    # ---- connections: the first one says hello and carries the config; the senders each open their own
    socks = []

    def hello(sock):
        send_msg(sock, dict(type='hello', actor=actor_id, have=[], relayed=False))
        return recv_msg(sock)
    sock = _connect(address, connect_timeout)
    first = hello(sock)
    socks.append(sock)
    cfg = first['config']
    size, segment, players = cfg['size'], cfg['segment'], int(cfg.get('players', 2))
    if not cfg.get('learner_values'):
        raise ValueError('the vector actor needs learner_values (no actor-side critic)')
    if float(cfg.get('pbrs_coef', 0.) or 0.) > 0:
        raise ValueError('the vector actor does not support PBRS')
    omni = bool(cfg.get('omniscient_critic'))
    turn_clock = cfg.get('turn_gamma') is not None
    rng = np.random.default_rng((cfg['seed'], actor_id) if seed is None else seed)
    map_mix = cfg.get('maps', 'classic')
    spec = cfg.get('network', dict(size=size))
    structure = jax.tree.structure(_template_params(**spec))
    device = jax.devices()[0]

    def to_tree(leaves):
        return jax.device_put(structure.unflatten([jnp.asarray(a, jnp.float32 if a.dtype == np.float16 else a.dtype)
                                                   for a in leaves]), device)

    params = {'learner': (first['version'], to_tree(first['weights']))}
    pool_spec = first['pool_spec']
    for name, leaves in first['pool_params'].items():
        params[name] = (0, to_tree(leaves))
    shared = dict(params=None, pool=None, stop=False, error=None)
    lock = threading.Lock()
    max_turns = cfg.get('max_turns')
    limit = training_round_limit(size) if max_turns is None else (None if max_turns == 0 else int(max_turns))

    # ---- the forward (inference_server.serve's, with the lane gather on the device)
    shapes = wire_shapes(size, players)
    layout = view_layout(size, players)
    vw = shapes['view_width']
    sizes = [int(np.prod(s)) for s in shapes['mask_shapes']]
    edges = np.cumsum([0] + sizes)

    def forward(p, views, packed, idx, k):
        views, packed = jnp.take(views, idx, axis=0), jnp.take(packed, idx, axis=0)
        bits = jnp.unpackbits(packed, axis=-1)[:, :edges[-1]].astype(bool)
        masks = Masks(*[bits[:, a:b].reshape((bits.shape[0],) + s)
                        for a, b, s in zip(edges[:-1], edges[1:], shapes['mask_shapes'], strict=True)])
        feats = jax.vmap(lambda v: view_features(v, layout, size))(views)
        keys = jax.random.split(k, views.shape[0])
        choice, path, logp, value = jax.vmap(lambda x, m, kk: network.sample(p, x, m, kk, size=size))(feats, masks, keys)
        return (jnp.stack(choice, 1), jnp.packbits(path.family, axis=-1), jnp.packbits(path.source, axis=-1),
                jnp.packbits(path.target, axis=-1), logp)
    forward = jax.jit(forward)
    key = [jax.random.key(int(rng.integers(1 << 31)))]

    # ---- engine groups
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    threads = threads or max(1, cpus - 2)
    per = -(-lanes // groups)
    bounds = [(g * per, min(lanes, (g + 1) * per)) for g in range(groups)]
    batches = []
    for a, b in bounds:
        gb = GameBatch(size, b - a, rules_json(), ENGINE_REVISION, limit, True, 10_000, players,
                       bool(cfg.get('no_peace', False)))
        gb.set_threads(threads)
        if profile:
            gb.profile_threads(True)
        batches.append(gb)
    classic = map_mix == 'classic'
    seed_fn = lambda: int(rng.integers(cfg['map_seed_low'], cfg['map_seed_high']))
    prefetch = None
    if not classic:
        table = MapTable.find(map_mix, size, players, cfg['map_seed_low'], cfg['map_seed_high'])
        prefetch = (MapTable(table, seed_fn) if table else
                    MapPrefetch(map_mix, size, players, seed_fn, workers=map_workers))
    n = size * size
    # per lane: the map it will play next (seed, kind, layout), and the kind it is playing
    lane_seed = np.zeros(lanes, np.int32)
    lane_kind = ['classic'] * lanes
    playing = ['classic'] * lanes
    lay = None if classic else [(np.zeros((b - a, n), np.int32), np.zeros((b - a, n), np.int32),
                                 np.zeros((b - a, n), np.int32), np.zeros((b - a, players), np.int32)) for a, b in bounds]

    def refill(lane):
        g = lane // per
        i = lane - bounds[g][0]
        if classic:
            lane_seed[lane] = seed_fn()
            return
        seed_, kind, (t, s_, r_, c_) = prefetch.get()
        lane_seed[lane], lane_kind[lane] = seed_, kind
        lay[g][0][i], lay[g][1][i], lay[g][2][i], lay[g][3][i] = t, s_, r_, c_

    t_maps = clock()
    for lane in range(lanes):
        refill(lane)
    for g, gb in enumerate(batches):
        a, b = bounds[g]
        if classic:
            gb.reset(np.ascontiguousarray(lane_seed[a:b]))
        else:
            gb.reset_layouts(np.ascontiguousarray(lane_seed[a:b]), *lay[g])
    for lane in range(lanes):
        playing[lane] = lane_kind[lane]
        refill(lane)
    first_maps_s = clock() - t_maps

    def observe(g):
        gb = batches[g]
        d = gb.scalars()
        ex = gb.export()
        cur = dict(current_player=np.asarray(d['current_player']), turn=np.asarray(d['turn']),
                   score=np.asarray(ex['players.score']), killed_turn=np.asarray(ex['players.killed_turn']),
                   resigned_turn=np.asarray(ex['players.resigned_turn']))
        return dict(view=np.asarray(gb.observe()), opp=np.asarray(gb.observe_opponent()) if omni else None,
                    masks=np.asarray(gb.legal_masks(packed=True)), **cur)
    state = [observe(g) for g in range(groups)]
    masks_width = state[0]['masks'].shape[1]
    pools = [ViewPool(b - a, vw, omni) for a, b in bounds]
    mask_ring = [[np.zeros((b - a, masks_width), np.uint8) for _ in range(2)] for a, b in bounds]
    final_ring = [[np.zeros(((b - a) * players, vw), np.int16) for _ in range(2)] for a, b in bounds]
    ring_at = [0] * groups

    # ---- opponents per block of lanes (pool_per_process semantics of the old actor processes)
    blocks = -(-lanes // block)
    per_process = bool(cfg.get('pool_per_process', False))
    p_self = [max(0., 1. - sum(p for _, p in pool_spec))]

    def draw_process_opponent():
        names = [nm for nm, _ in pool_spec if nm in params]
        probs = np.array([p for nm, p in pool_spec if nm in params], np.float64)
        if not names or probs.sum() <= 0:
            return None
        return names[int(rng.choice(len(names), p=probs / probs.sum()))]

    def draw_mode():
        if per_process and rng.random() < p_self[0]:
            return None
        return draw_process_opponent()
    block_opp = [draw_mode() for _ in range(blocks)]
    finished_since = [0] * blocks

    def assign(lane):
        opp = block_opp[lane // block]
        if opp is None or (not per_process and rng.random() < p_self[0]):
            return None, 0
        return opp, int(rng.integers(1, players + 1))
    lane_opp = [assign(lane) for lane in range(lanes)]
    lane_seat = np.array([lo[1] for lo in lane_opp], np.int32)     # numpy mirrors of lane_opp for dispatch
    lane_pool = np.array([lo[0] is not None for lo in lane_opp])

    def set_lane(lane, value):
        lane_opp[lane] = value
        lane_seat[lane], lane_pool[lane] = value[1], value[0] is not None

    widths = dict(family=-(-shapes['family_bits'] // 8), source=-(-shapes['source_bits'] // 8),
                  target=-(-shapes['target_bits'] // 8))
    store = RowStore(lanes, players, segment, widths, capacity=1 << 20, turn_clock=turn_clock, omni=omni)
    stats = _new_actor_stats()
    timing = {}

    def lap(name, since):
        now = clock()
        timing[name] = timing.get(name, 0.) + (now - since)
        return now

    # ---- senders: compress + send on their own connections; replies update weights and the pool
    outbox = queue.Queue(maxsize=senders + 1)
    send_timing = dict(build_s=0., zstd_s=0., raw_rows=0)
    level = int(cfg.get('upload_zstd_level', 1) if zstd_level is None else zstd_level)
    coded = bool(cfg.get('coded_views'))   # the learner decodes view_codec deltas on its devices

    def sender(k):
        try:
            sock_k = socks[0] if k == 0 else None
            if sock_k is None:
                sock_k = _connect(address, connect_timeout)
                hello(sock_k)
            while True:
                item = outbox.get()
                if item is None:
                    return
                t0 = clock()
                job = item.pop('job')
                item['batch'], item['segments'] = job.build_coded(zstd_threads) if coded else job.build()
                t1 = clock()
                frame = encode_frame(item, level=level, threads=zstd_threads)
                with lock:
                    send_timing['build_s'] += t1 - t0
                    send_timing['zstd_s'] += clock() - t1
                    send_timing['raw_rows'] += job.rows
                sock_k.sendall(frame)
                reply = recv_msg(sock_k)
                with lock:
                    if reply.get('weights') is not None and (shared['params'] is None or reply['version'] > shared['params'][0]):
                        shared['params'] = (reply['version'], reply['weights'])
                    if reply.get('pool_params') is not None:
                        shared['pool'] = (reply['pool_spec'], reply['pool_params'])
                    if reply.get('stop'):
                        shared['stop'] = True
        except Exception as error:
            shared['error'] = repr(error)
            shared['stop'] = True
    send_threads = [threading.Thread(target=sender, args=(k,), daemon=True) for k in range(senders)]
    for t in send_threads:
        t.start()

    # ---- pipeline
    pending = [None] * groups      # per group: list of (pid, idx, n, outputs) in flight

    dispatcher = None

    def dispatch(g):
        """Lane groups by parameter set (main thread), then the device puts and forwards on the dispatcher thread."""
        nonlocal dispatcher
        st = state[g]
        a, b = bounds[g]
        cp = st['current_player']
        opp_turn = lane_pool[a:b] & (cp != lane_seat[a:b])
        groups_ = {'learner': np.flatnonzero(~opp_turn)}
        by_name = {}
        for i in np.flatnonzero(opp_turn).tolist():
            by_name.setdefault(lane_opp[a + i][0], []).append(i)
        for nm in sorted(by_name):
            groups_[nm] = np.array(by_name[nm], np.int64)
        if dispatcher is None:
            from concurrent.futures import ThreadPoolExecutor
            dispatcher = ThreadPoolExecutor(1)
        pending[g] = dict(future=dispatcher.submit(launch, st['view'], st['masks'], b - a, groups_), opp_turn=opp_turn)

    def launch(view, masks, m, groups_):
        dv = jax.device_put(view, device)
        dm = jax.device_put(masks, device)
        jobs = []
        for nm, idx in groups_.items():
            if not len(idx):
                continue
            bsz = _bucket(len(idx), m)
            pad = np.full(bsz, idx[0], np.int32)
            pad[:len(idx)] = idx
            key[0], k = jax.random.split(key[0])
            out = forward(params[nm][1], dv, dm, jax.device_put(pad, device), k)
            jobs.append((nm, idx, out, params[nm][0]))
        jax.block_until_ready(dv)   # the host arrays may be reused once their copies are on the device
        jax.block_until_ready(dm)
        return jobs

    def collect(g):
        """Host copies of the group's forwards: per-lane choice [m, 3], packed path masks, logp."""
        a, b = bounds[g]
        m = b - a
        flight = pending[g]
        jobs = flight['future'].result()
        choice = np.zeros((m, 3), np.int32)
        fam = np.zeros((m, widths['family']), np.uint8)
        src = np.zeros((m, widths['source']), np.uint8)
        tgt = np.zeros((m, widths['target']), np.uint8)
        logp = np.zeros(m, np.float32)
        version = np.zeros(m, np.int32)
        for nm, idx, out, ver in jobs:
            host = [np.asarray(o)[:len(idx)] for o in out]
            choice[idx], fam[idx], src[idx], tgt[idx], logp[idx] = host
            version[idx] = ver
        return choice, fam, src, tgt, logp, version, flight['opp_turn']

    cap_reward = cfg.get('truncation_reward')
    cap_score = float(cfg.get('cap_score_reward', 0.) or 0.)
    last_report, last_release = time.monotonic(), 0
    last_send = time.monotonic()
    ticks = 0
    for g in range(groups):
        dispatch(g)
    summary = dict(actor=actor_id, lanes=lanes, groups=groups, threads=threads, sent_rows=0, messages=0,
                   first_maps_s=round(first_maps_s, 2), started=time.time(), spilled=0)
    stats['started'] = time.time()
    from concurrent.futures import ThreadPoolExecutor
    stepper = ThreadPoolExecutor(1)   # engine ticks (worker threads inside, GIL released) overlap the main thread

    def run_tick(g, choice, view, opp, masks, final):
        a, b = bounds[g]
        t0 = clock()
        out = batches[g].tick_into(np.ascontiguousarray(choice[:, 0]), np.ascontiguousarray(choice[:, 1]),
                                   np.ascontiguousarray(choice[:, 2]), np.ascontiguousarray(lane_seed[a:b]),
                                   view, opp, masks, final, omni, None if classic else lay[g], True)
        return out, clock() - t0

    def update_params():
        nonlocal pool_spec
        with lock:
            new_params, shared['params'] = shared['params'], None
            new_pool, shared['pool'] = shared['pool'], None
        if new_params is not None and new_params[0] > params['learner'][0]:
            params['learner'] = (new_params[0], to_tree(new_params[1]))
            stats['weight_updates'] += 1
        if new_pool is not None:
            pool_spec, new = new_pool
            for nm, leaves in new.items():
                params[nm] = (0, to_tree(leaves))
            for nm in [x for x in params if x != 'learner' and x not in {y for y, _ in pool_spec}]:
                del params[nm]
            p_self[0] = max(0., 1. - sum(p for _, p in pool_spec))
            for k_ in range(blocks):
                if block_opp[k_] is not None and block_opp[k_] not in params:
                    block_opp[k_] = draw_mode()
            for lane in range(lanes):
                if lane_opp[lane][0] is not None and lane_opp[lane][0] not in params:
                    set_lane(lane, (None, 0))

    def front(g):
        """The group's forwards -> its learner rows -> its tick submitted to the stepper."""
        a, b = bounds[g]
        st = state[g]
        t = clock()
        choice, fam, src, tgt, logp, version, opp_turn = collect(g)
        t = lap('t_wait_gpu', t)
        mine = np.flatnonzero(~opp_turn)
        cp = st['current_player']
        if len(mine):
            fields = dict(family_mask=fam[mine], source_mask=src[mine], target_mask=tgt[mine], choice=choice[mine],
                          logp=logp[mine], version=version[mine], end_turn=choice[mine, 0] == END_FAMILY)
            store.add(a + mine, cp[mine].astype(np.int64), st['view'], st['opp'], mine.astype(np.int32), fields)
            stats['learner_decisions'] += len(mine)
            summ = choice[mine][choice[mine, 0] == SUMMON_FAMILY, 2]
            if len(summ):
                u, c = np.unique(summ, return_counts=True)
                for unit, cnt in zip(u.tolist(), c.tolist()):
                    stats['summons'][unit] = stats['summons'].get(unit, 0) + cnt
        t = lap('t_rows', t)
        view, opp = pools[g].get()
        ring_at[g] ^= 1
        masks, final = mask_ring[g][ring_at[g]], final_ring[g][ring_at[g]]
        lap('t_pool', t)
        return stepper.submit(run_tick, g, choice, view, opp, masks, final), (view, opp, masks, final)

    def back(g, submitted):
        """The group's tick result: outcomes of finished games, the new state, the next forwards dispatched."""
        a, b = bounds[g]
        st = state[g]
        future, (view_out, opp_out, masks_out, final_out) = submitted
        t = clock()
        out, engine_s = future.result()
        t = lap('t_wait_engine', t)
        timing['t_engine'] = timing.get('t_engine', 0.) + engine_s
        status = np.asarray(out['status'])
        if status.any():
            raise RuntimeError(f'illegal policy actions on lanes {(a + np.flatnonzero(status)).tolist()}')
        stats['steps'] += b - a
        done = np.asarray(out['done'])
        if done.any():
            terminated, truncated = np.asarray(out['terminated']), np.asarray(out['truncated'])
            rewards = np.array(out['rewards'], np.float32)
            final_views = final_out.reshape(b - a, players, vw)
            score, final_turn = st['score'], st['turn']
            out_seats = ((st['killed_turn'] > 0) | (st['resigned_turn'] > 0)).sum(1)
            for i in np.flatnonzero(done):
                lane = a + int(i)
                name, seat = lane_opp[lane]
                stats['finished'] = stats.get('finished', 0) + 1
                stats['turns'] = stats.get('turns', 0) + int(final_turn[i])
                stats['seats_out'] = stats.get('seats_out', 0) + int(out_seats[i]) + int(terminated[i])
                if truncated[i] and cap_score > 0:
                    lead = float(np.sign(score[i, 0] - score[i, 1]))
                    rewards[i], ended = np.array([lead, -lead], np.float32) * cap_score, True
                    stats['cap_score_results'] = stats.get('cap_score_results', 0) + 1
                else:
                    rewards[i], ended = episode_outcome(rewards[i], terminated[i], truncated[i], cap_reward)
                if name is None:
                    store.finish(lane, rewards[i], ended, final_views[i])
                    stats['selfplay_games'] += 1
                    stats['selfplay_truncations'] += int(truncated[i])
                else:
                    store.drop(lane, other_seats(seat, players))
                    store.finish(lane, rewards[i], ended, final_views[i])
                    rec = stats['vs'].setdefault(name, [0, 0, 0])
                    r = 0. if truncated[i] and players == 2 else rewards[i][seat - 1]
                    rec[0 if r > 0 else 1 if r < 0 else 2] += 1
                rec = stats['maps'].setdefault(playing[lane], [0, 0])
                rec[0] += 1
                rec[1] += int(truncated[i])
                playing[lane] = lane_kind[lane]
                k_ = lane // block
                finished_since[k_] += 1
                if finished_since[k_] >= 4 * block:
                    block_opp[k_], finished_since[k_] = draw_mode(), 0
                set_lane(lane, assign(lane))
                refill(lane)
        st['view'], st['masks'] = view_out, masks_out
        st['opp'] = opp_out if omni else None
        for k_ in ('current_player', 'turn', 'score', 'killed_turn', 'resigned_turn'):
            st[k_] = np.asarray(out[k_])
        t = lap('t_done', t)
        dispatch(g)
        lap('t_dispatch', t)

    futures = [None] * groups
    stop_checked, stopping = time.monotonic(), False

    def stop_requested():
        """should_stop() at most every stop_check_every seconds (it may be a remote lookup)."""
        nonlocal stop_checked, stopping
        if should_stop is not None and time.monotonic() - stop_checked > stop_check_every:
            stop_checked = time.monotonic()
            stopping = bool(should_stop())
        return stopping
    while not shared['stop'] and not stop_requested():
        for g in range(groups):
            t = clock()
            update_params()
            lap('t_params', t)
            futures[g] = front(g)
            h = (g - 1) % groups
            if futures[h] is not None and h != g:
                back(h, futures[h])
                futures[h] = None
            elif groups == 1:
                back(g, futures[g])
                futures[g] = None
            ticks += 1
            # messages
            if store.emitted_rows >= send_rows or (store.emitted_rows and time.monotonic() - last_send > report_every):
                t = clock()
                last_send = time.monotonic()
                store.check()
                job = store.take()
                stats.update({k: v for k, v in timing.items()})
                with lock:
                    stats.update({'send_' + k: v for k, v in send_timing.items()})
                    for k in send_timing:
                        send_timing[k] = 0.
                if prefetch is not None:
                    stats['t_maps_wait'] = prefetch.wait_s
                    prefetch.wait_s = 0.
                stats['spilled'] = store.spilled
                summary['spilled'] += store.spilled
                store.spilled = 0
                message = dict(type='data', actor=actor_id, have=sorted(x for x in params if x != 'learner'),
                               job=job, stats=stats)
                take_s = clock() - t
                t = clock()
                outbox.put(message)
                summary['sent_rows'] += job.rows
                summary['messages'] += 1
                stats, timing = _new_actor_stats(), {}
                timing['t_put'] = clock() - t
                timing['t_take'] = take_s
            if ticks - last_release >= 32:
                t = clock()
                store.release(max_age=spill_age)
                last_release = ticks
                lap('t_release', t)
        if report is not None and time.monotonic() - last_report > report_every:
            last_report = time.monotonic()
            say(dict(summary, elapsed=time.monotonic() - started, ticks=ticks, slots=len(store.views)))
    for g in range(groups):   # let the last ticks finish before the engines go away
        if futures[g] is not None:
            futures[g][0].result()
    stepper.shutdown()
    if dispatcher is not None:
        dispatcher.shutdown()
    summary['pool_arrays'] = [p.allocated for p in pools]
    if profile:   # engine phases summed over the groups' worker threads: {phase: (calls, ms)}
        total = {}
        for gb in batches:
            for k, (calls, ms) in gb.phase_profile_threads().items():
                c0, m0 = total.get(k, (0, 0.))
                total[k] = (c0 + calls, m0 + ms)
        summary['engine_profile'] = {k: (c, round(m, 1)) for k, (c, m) in sorted(total.items(), key=lambda x: -x[1][1])}
    for _ in send_threads:
        outbox.put(None)
    for t in send_threads:
        t.join(timeout=30)
    if prefetch is not None:
        prefetch.close()
    for s in socks:
        try:
            s.close()
        except OSError:
            pass
    summary['error'] = shared['error']
    return summary
