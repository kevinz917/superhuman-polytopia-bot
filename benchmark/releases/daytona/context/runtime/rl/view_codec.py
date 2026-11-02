"""Lossless compact wire format for the packed views of training rows (actor -> learner).

A message's rows come in whole segments (SegmentCollector), each a seat's consecutive decisions; one decision
changes a few tiles, so consecutive views of a seat are ~99% equal. Every view stream (the mover's view and,
with the omniscient critic, each other seat's view) is sent as sparse int16 deltas: row r carries
view[r] - view[r-1] when r continues the segment of r-1 (pos > 0), else view[r] itself. A segment's boundary
view (boot_*) is sent as its delta from the segment's last row. Only the nonzero entries travel, as
(row, column, value) triples.

The learner rebuilds the dense int16 views on the device (`dense_rows`): scatter the deltas into a zero
[rows, width] int32 block, cumulative-sum along rows, subtract the running sum at each segment's first row.
Exact: every partial sum is a true view value (int16 range), so int32 never overflows.

Shapes: a message with m rows and k = players - 1 other seats has streams [m, 1 + k, width]
(stream 0 = own view). coo = dict(row int32 [z], stream int8 [z], col int16 [z], val int16 [z]).
"""
import numpy as np


def streams_of(batch):
    """[m, S, width] int16: own view then the other seats' views (opp_view [m, width] or [m, k, width])."""
    view = batch['view']
    if 'opp_view' not in batch:
        return view[:, None, :]
    opp = batch['opp_view']
    return np.concatenate([view[:, None, :], opp if opp.ndim == 3 else opp[:, None, :]], 1)


def _sparse(delta):
    r, s, c = np.nonzero(delta)
    return dict(row=r.astype(np.int32), stream=s.astype(np.int8), col=c.astype(np.int16 if delta.shape[-1] < 32768 else np.int32),
                val=delta[r, s, c].astype(np.int16))


def encode(batch, segs):
    """Replace batch['view'] / ['opp_view'] and segs['boot_view'] / ['boot_opp_view'] by sparse deltas.
    Returns new (batch, segs) dicts; other fields are shared, not copied."""
    x = streams_of(batch)
    m, n_streams, width = x.shape
    cont = np.asarray(batch['pos']) > 0
    delta = x.copy()
    delta[1:][cont[1:]] -= x[:-1][cont[1:]]
    out = {k: v for k, v in batch.items() if k not in ('view', 'opp_view')}
    out.update({f'view_coo_{k}': v for k, v in _sparse(delta).items()},
               view_shape=np.array([m, n_streams, width], np.int32))
    sout = dict(segs)
    if segs is not None and 'boot_view' in segs:
        boot = streams_of(dict(view=segs['boot_view'],
                               **({'opp_view': segs['boot_opp_view']} if 'boot_opp_view' in segs else {})))
        n_seg = len(segs['length'])
        # each segment's last row: rows are whole segments in order, so the last row of seg j is cumsum(length)-1
        last = np.cumsum(np.asarray(segs['length'])) - 1
        terminal = np.asarray(segs['terminal']).astype(bool)
        bdelta = boot.copy()
        bdelta[~terminal] -= x[last[~terminal]]   # terminal boundaries are zero views: sent as they are
        sout = {k: v for k, v in segs.items() if k not in ('boot_view', 'boot_opp_view')}
        sout.update({f'boot_coo_{k}': v for k, v in _sparse(bdelta).items()},
                    boot_shape=np.array([n_seg, n_streams, width], np.int32))
    return out, sout


def decode_numpy(batch, segs):
    """Reference decoder (host, numpy): the dense views back, as encode's inputs had them."""
    m, n_streams, width = (int(v) for v in batch['view_shape'])
    delta = np.zeros((m, n_streams, width), np.int32)
    np.add.at(delta, (batch['view_coo_row'], batch['view_coo_stream'].astype(np.int64), batch['view_coo_col'].astype(np.int64)),
              batch['view_coo_val'].astype(np.int32))
    x = delta.copy()
    for r in range(1, m):
        if batch['pos'][r] > 0:
            x[r] += x[r - 1]
    x = x.astype(np.int16)
    out = {k: v for k, v in batch.items() if not k.startswith('view_coo_') and k != 'view_shape'}
    out['view'] = x[:, 0]
    if n_streams > 1:
        out['opp_view'] = x[:, 1] if n_streams == 2 else x[:, 1:]
    sout = segs
    if segs is not None and 'boot_shape' in segs:
        n_seg = int(segs['boot_shape'][0])
        b = np.zeros((n_seg, n_streams, width), np.int32)
        np.add.at(b, (segs['boot_coo_row'], segs['boot_coo_stream'].astype(np.int64), segs['boot_coo_col'].astype(np.int64)),
                  segs['boot_coo_val'].astype(np.int32))
        last = np.cumsum(np.asarray(segs['length'])) - 1
        terminal = np.asarray(segs['terminal']).astype(bool)
        b[~terminal] += x[last[~terminal]]
        b = b.astype(np.int16)
        sout = {k: v for k, v in segs.items() if not k.startswith('boot_coo_') and k != 'boot_shape'}
        sout['boot_view'] = b[:, 0]
        if n_streams > 1:
            sout['boot_opp_view'] = b[:, 1] if n_streams == 2 else b[:, 1:]
    return out, sout


def dense_rows(rows, coo_row, coo_stream, coo_col, coo_val, start, n_streams, width, base=None, base_row=None):
    """Device rebuild of a block of `rows` dense rows [rows, n_streams, width] int16 (jit with rows, n_streams,
    width static). coo_*: the deltas; coo_row indexes this block (padding entries carry val 0). start: bool
    [rows], the row begins a segment (its delta is from zero, or from base[base_row[r]]). base/base_row: the
    boundary views' rebuild: base = dense rows, base_row[r] the row r's delta is from (< 0: from zero)."""
    import jax
    import jax.numpy as jnp
    acc = jnp.zeros((rows, n_streams, width), jnp.int32).at[coo_row, coo_stream, coo_col].add(coo_val.astype(jnp.int32))
    if base_row is not None:   # boundary views: each row is its own one-row segment on top of its base row
        add = jnp.where((base_row >= 0)[:, None, None], base[jnp.maximum(base_row, 0)].astype(jnp.int32), 0)
        return (acc + add).astype(jnp.int16)
    run = jnp.cumsum(acc, axis=0)
    before = jnp.concatenate([jnp.zeros((1, n_streams, width), jnp.int32), run[:-1]], 0)
    seg_start = jax.lax.cummax(jnp.where(start, jnp.arange(rows), 0), axis=0)
    return (run - before[seg_start]).astype(jnp.int16)


# ----------------------------------------------------------------------------- learner side

def _round(n, step):
    """n rounded up to a bucket >= step: 8 buckets per doubling (<= 12.5% padding), so batches of slowly varying
    size reuse a few compiled shapes."""
    b = step
    while b * 2 <= n:
        b *= 2
    eighth = max(b // 8, 1)
    return b if n <= b else -(-n // eighth) * eighth


_SHARD_FNS = {}


def _shard_fn(slots, length, rows_local, segs_local, two_seats):
    """jit: one device's shard of the batch's views from the raw deltas of the messages that reach it. Each raw
    row r sits at (row_slot[r], row_pos[r]) of a [slots, length, streams, width] int16 layout (segment-major; the
    last slot is an all-zero spare that padding points at); entries are scattered there by their raw row, scanned
    along each segment in int16 (every partial sum is a true view value), the segments' boundary views are
    rebuilt on their last rows (base), and the fresh rows / boundaries are scattered into this shard's
    [rows_local] / [segs_local] buffers (destinations outside the shard carry the dropped end index)."""
    key = (slots, length, rows_local, segs_local, two_seats)
    if key not in _SHARD_FNS:
        import jax
        import jax.numpy as jnp

        def build(row_slot, row_pos, coo, bcoo, src, dst, base, bdst, n_streams, width):
            c_row, c_stream, c_col, c_val = coo
            r = c_row.astype(jnp.int32)
            acc = jnp.zeros((slots, length, n_streams, width), jnp.int16).at[
                row_slot[r], row_pos[r], c_stream.astype(jnp.int32), c_col.astype(jnp.int32)].add(c_val)
            dense = jnp.cumsum(acc, axis=1, dtype=jnp.int16)
            b_slot, b_stream, b_col, b_val = bcoo
            boot = dense[base[0], base[1]].at[b_slot, b_stream.astype(jnp.int32), b_col.astype(jnp.int32)].add(b_val)
            take = dense[row_slot[src], row_pos[src]]
            out_view = jnp.zeros((rows_local, width), jnp.int16).at[dst].set(take[:, 0], mode='drop')
            out_bview = jnp.zeros((segs_local, width), jnp.int16).at[bdst].set(boot[:, 0], mode='drop')
            out_opp = out_bopp = None
            if n_streams > 1:
                opp, bopp = (take[:, 1], boot[:, 1]) if two_seats else (take[:, 1:], boot[:, 1:])
                out_opp = jnp.zeros((rows_local,) + opp.shape[1:], jnp.int16).at[dst].set(opp, mode='drop')
                out_bopp = jnp.zeros((segs_local,) + bopp.shape[1:], jnp.int16).at[bdst].set(bopp, mode='drop')
            return out_view, out_opp, out_bview, out_bopp
        _SHARD_FNS[key] = jax.jit(build, static_argnums=(8, 9))
    return _SHARD_FNS[key]


class CodedRows:
    """A batch's coded views (learner side): add() records each message's deltas (stale rows included: they anchor
    the delta chain), finish() rebuilds blocks of whole messages on the device and gathers the fresh rows into the
    batch order. Returns the same {'view', 'opp_view'} / boundary dicts _score builds from dense views."""

    def __init__(self, learner):
        self.learner = learner
        self.parts, self.fresh_rows, self.segs = [], 0, 0
        self.shape = None

    def add(self, part, seg, pos, fresh, bpart, lengths, terminal):
        """seg/pos: the message's rows' segment (message-local) and position, before the stale filter."""
        m, n_streams, width = (int(v) for v in part['view_shape'])
        if self.shape is None:
            self.shape = (n_streams, width)
        elif self.shape != (n_streams, width):
            raise ValueError('coded views: messages with different view shapes in one batch')
        lengths = np.asarray(lengths)
        terminal = np.asarray(terminal).astype(bool)
        self.parts.append(dict(m=m, seg=np.asarray(seg), pos=np.asarray(pos),
                               coo=(part['view_coo_row'], part['view_coo_stream'], part['view_coo_col'],
                                    part['view_coo_val']),
                               fresh=np.nonzero(np.asarray(fresh))[0], dst0=self.fresh_rows,
                               bcoo=(bpart['boot_coo_row'], bpart['boot_coo_stream'], bpart['boot_coo_col'],
                                     bpart['boot_coo_val']),
                               lengths=lengths, terminal=terminal, seg0=self.segs))
        self.fresh_rows += len(self.parts[-1]['fresh'])
        self.segs += len(lengths)

    def finish(self, n, n_pad, n_seg, seg_pad):
        """({'view' [n_pad, W], 'opp_view' [n_pad, (k,) W]}, the same for the segments' boundaries) on the
        devices, data-sharded; rows past n / n_seg are zero. Device i rebuilds, in one call, the messages whose
        fresh rows or boundaries fall in its 1/D of the batch (a message straddling a shard edge goes to both)
        into its own shard. Host work is one concatenation of those messages' raw delta arrays plus per-row and
        per-segment index arrays; the per-entry slot/position lookup runs on the device."""
        import jax
        learner = self.learner
        n_streams, width = self.shape
        two = n_streams == 2
        devs = list(learner.mesh.devices.flat)
        d = len(devs)
        rows_local, segs_local = n_pad // d, seg_pad // d
        parts = self.parts
        nf = np.array([len(p['fresh']) for p in parts])
        ns = np.array([len(p['lengths']) for p in parts])
        dst0, seg0 = np.array([p['dst0'] for p in parts]), np.array([p['seg0'] for p in parts])
        def shard(i):
            dev = devs[i]
            lo, blo = i * rows_local, i * segs_local
            need = ((dst0 < lo + rows_local) & (dst0 + nf > lo)) | ((seg0 < blo + segs_local) & (seg0 + ns > blo))
            hull = np.nonzero(need)[0]
            mine = parts[hull[0]:hull[-1] + 1] if len(hull) else []
            m = np.array([p['m'] for p in mine], np.int64)
            k = np.array([len(p['lengths']) for p in mine], np.int64)
            roff, sat = np.cumsum(m) - m, np.cumsum(k) - k
            R, S = int(m.sum()), int(k.sum())
            spare = _round(S + 1, 64) - 1          # bucketed slots; the last one is the zero spare
            slots = spare + 1
            length = 8
            while mine and length < max(int(p['lengths'].max()) for p in mine):
                length *= 2
            R_pad = _round(R + 1, 1024)          # raw row R..: padding rows at (spare, 0)
            cat = lambda xs, total, fill, dt: np.concatenate(
                [np.asarray(x).astype(dt, copy=False) for x in xs] + [np.full(total - sum(len(x) for x in xs), fill, dt)])
            row_slot = cat([p['seg'] + s_ for p, s_ in zip(mine, sat)], R_pad, spare, np.int32)
            row_pos = cat([p['pos'] for p in mine], R_pad, 0, np.int32)
            z = sum(len(p['coo'][0]) for p in mine)
            Z = _round(z, 1 << 18)
            col_dt = mine[0]['coo'][2].dtype if mine else np.int16
            coo = (cat([p['coo'][0] + np.int32(r) for p, r in zip(mine, roff)], Z, R, np.int32),
                   cat([p['coo'][1] for p in mine], Z, 0, np.int8), cat([p['coo'][2] for p in mine], Z, 0, col_dt),
                   cat([p['coo'][3] for p in mine], Z, 0, np.int16))
            bz = sum(len(p['bcoo'][0]) for p in mine)
            BZ = _round(bz, 1 << 14)
            bcol_dt = mine[0]['bcoo'][2].dtype if mine else np.int16
            bcoo = (cat([p['bcoo'][0] + np.int32(s_) for p, s_ in zip(mine, sat)], BZ, spare, np.int32),
                    cat([p['bcoo'][1] for p in mine], BZ, 0, np.int8), cat([p['bcoo'][2] for p in mine], BZ, 0, bcol_dt),
                    cat([p['bcoo'][3] for p in mine], BZ, 0, np.int16))
            F = sum(len(p['fresh']) for p in mine)
            F_pad = _round(F + 1, 1024)
            src = cat([p['fresh'] + r for p, r in zip(mine, roff)], F_pad, R, np.int32)
            dst = np.concatenate([np.arange(len(p['fresh'])) + p['dst0'] for p in mine]) if mine else np.zeros(0, np.int64)
            dst = cat([np.where((dst >= lo) & (dst < lo + rows_local), dst - lo, rows_local)], F_pad, rows_local, np.int32)
            # boundary base: each segment's last row; terminal (and padding) slots read the spare slot's row 0 (zero)
            last = np.concatenate([np.where(p['terminal'], -1, p['lengths'] - 1) for p in mine]) if mine else np.zeros(0)
            base = (cat([np.where(last >= 0, np.arange(S), spare)], slots, spare, np.int32),
                    cat([np.maximum(last, 0)], slots, 0, np.int32))
            bglob = np.concatenate([np.arange(len(p['lengths'])) + p['seg0'] for p in mine]) if mine else np.zeros(0, np.int64)
            bdst = cat([np.where((bglob >= blo) & (bglob < blo + segs_local), bglob - blo, segs_local)], slots,
                       segs_local, np.int32)
            args = jax.device_put((row_slot, row_pos, coo, bcoo, src, dst, base, bdst), dev)
            return _shard_fn(slots, length, rows_local, segs_local, two)(*args, n_streams, width)

        # one thread per device: the host prep is numpy concatenation (releases the GIL) and JAX dispatch is
        # thread-safe, so the shards are prepared and launched in parallel
        if d == 1:
            outs = [shard(0)]
        else:
            if self.learner.__dict__.get('_codec_pool') is None:
                from concurrent.futures import ThreadPoolExecutor
                self.learner._codec_pool = ThreadPoolExecutor(d, thread_name_prefix='codec')
            outs = list(self.learner._codec_pool.map(shard, range(d)))

        def merged(j, total):
            if d == 1:
                return outs[0][j]
            return jax.make_array_from_single_device_arrays((total,) + outs[0][j].shape[1:], learner.data_sharding,
                                                            [o[j] for o in outs])
        views, boots = dict(view=merged(0, n_pad)), dict(view=merged(2, seg_pad))
        if outs[0][1] is not None:
            views['opp_view'], boots['opp_view'] = merged(1, n_pad), merged(3, seg_pad)
        return views, boots
