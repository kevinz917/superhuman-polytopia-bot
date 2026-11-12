"""GPU-resident PPO trainer for the hero-1v1 recipe (docs/hero-recipe-map-2026-11-06.md) on one GPU.

The CUDA engine (rl.gpu_engine.GpuGames, device-pointer mode) steps thousands of games into torch device tensors,
every tensor is handed to JAX zero-copy through dlpack, and the production network / PPO code (rl.network, rl.ppo,
rl.distributed's critic, GAE and schedule functions) trains on the same card. The recipe pieces and where each lives:

  * omniscient critic (map section 1): `distributed._critic_template` / `initial_critic`; value pass per VALUE_CHUNK rows
    of `critic_features(view_features(view), opp_view)` through `network.trunk(critic, x)[2]` (_values); loss =
    `ppo.loss(..., value_coef=0.)` + .5 * (.5 * MSE) (D:), one `ppo.adam_step` over (params, critic) (_train_step);
  * same-seat 32-decision segments bootstrapping on the seat's next decision (SegmentCollector / RowStore semantics):
    `SegmentLog` (host bookkeeping; rows live in the device ring); `distributed.segment_gae` with the turn clock;
  * opponent pool (map section 2): `Lanes` (blocks of 24 lanes), fp16-rounded snapshots
    every historical_every updates, grouped sampling per parameter set ;
  * rewards and terminals (section 3): engine terminal rewards, cap score from the PREVIOUS tick's `scores` (the state
    before the final action), `distributed.episode_outcome` otherwise (_outcomes);
  * schedule / precision (section 4): `scheduled_learning_rate(training_progress(...))`, bf16 trunk / TF32 / HIGHEST heads
    from the environment the CLI sets before JAX imports;
  * maps (section 5): the precomputed `MapTable` on the device, seeds uniform in [lo, hi) per game per step, the seed is
    also the game RNG seed;
  * checkpoints (section 6): `runner.save_checkpoint` (p{i} leaves + o{i} Adam leaves of (params, critic)), latest /
    latest-critic / snapshots/update-N / final, protocol.json, training.jsonl, resume from latest.npz.

Rows: every engine step writes every game's (view, opp_view, packed path masks, choice, logp, critic value) into slot
t % L of a device ring of L = ring + assembly-slack steps (torch tensors). The host SegmentLog knows which (step, game)
rows are learner rows, their (game, seat) segment and position; a segment closes at the seat's 33rd row (the first 32
form the segment, the 33rd is its boundary and the first row of the next one), at the game's end, or when its first
row is `ring` steps old (forced: boundary = the seat's value of the game's CURRENT state, i.e. the post-tick views the
step that cuts it starts from, seen from that seat; counted and reported as forced_close_fraction; --ring is the lever
against it). When the closed rows reach `batch`, exactly `batch` rows (oldest segments first; the tail of the
straddling segment is dropped as the production learner drops rows past `batch`) are gathered from the ring, GAE'd per
segment with the critic values, normalized over the batch (floor advantage_floor) and trained on in `minibatch` pieces
(epochs from the config).

Critic values (value_mode): the production learner scores every row and boundary with the pre-update critic at
assembly. 'rollout' (default) runs the critic on each set's state right after encoding it, every step (the call
overlaps the other sets' engine ticks) and keeps the value in the ring: identical to the assembly score for rows
recorded since the last update (the critic only changes at an update); the rows of segments that were open at the
previous update were valued by the previous critic, so the assembly re-scores exactly those (the ring slot's critic
version is tracked; ~half the batch, see `value_sources`), the boundary of a 33rd-row cut is that row's ring value,
and only forced-cut boundaries (the other seat's orientation), truncation boundaries from the final-view store and
stale boundaries need the small value call. 'rollout-stale' skips the re-score (a documented deviation: those rows
keep values one update old). 'assembly' is the production path (every value at assembly); value_check computes both
and reports max |diff|.

The rollout step per set (fused, the default): ONE jitted call (`fused`) does the encode, the critic value and the
learner's sampling of the set's state and returns the choice (both [Gs, 3] for the ring and [3, Gs] for the engine's
inputs, read in place), the packed PathMasks rows, (logp, value), the mover seats and the pool key: one dispatch and
one host wait per set per step. The tick's outcomes (scores, rewards, status, done / terminated / truncated) leave the
device in one non-blocking copy into pinned memory staged right after the engine sync and read after the fused call
is dispatched (they overlap it); they are booked BEFORE this step's rows are recorded, which the SegmentLog needs
(a finished game's segments close before the restarted game's first row is recorded, and the lanes' new seats decide
whose move the row is), so the copy cannot be deferred a step. The ring takes a set's rows as five slice copies (view,
opp view, masks, choice, (logp, value) as a [2, Gs] block); the engine inputs are the fused call's [3, Gs] output;
the seeds and the map rows come from pinned buffers; the layouts are two index_selects. `fused=False` (--unfused)
keeps the separate encode / values_set / act calls for A/B timing. The pool opponents' act_subset runs only on steps
where a set has pool lanes (never during the first historical_every updates).

Several GPUs (devices=N > 1, one process, CUDA ordinals 0..N-1; devices=1 is the single-GPU path above unchanged): data
parallel as the production learner. Each device is a shard with its own engine sets (games / sets / ring are PER
DEVICE), ring, batch buffers, SegmentLog, Lanes, map draws and rollout key: device 0 draws from the run's streams (the
config seed), device d > 0 from streams seeded by (seed, d). The rollouts run in lockstep: per set index every device's
set is finished and its fused call dispatched (they run concurrently, each on its GPU) before any is awaited, booked and
its tick launched; the engine driver makes the batch's device current around every call (pc_cuda.cu enter_device).
The update fires when the SUM of the devices' closed rows reaches `batch`. Device d then contributes
row_shares(closed, batch)[d] rows: batch // N each, a device short of that gives all it has and the shortfall comes
from the others (water filling). Each device gathers its oldest segments covering its share from its own ring, scores
them with its replica of the critic and runs segment GAE on its own segments; the advantages are normalized with the
mean / std of the whole batch (floor advantage_floor). rebalance_plan moves the surplus rows of a device that gave more
than batch // N to the devices that gave fewer, so every device holds exactly batch // N rows (the static shard of the
update; no move when the shares are equal). The update is the production step: (params, critic) and Adam replicated
over a 1-D mesh ('data'), per epoch each device permutes its own rows (fold_in(key, device)), each step takes
minibatch // N of them (grad_accum micro-batches of that), the gradients and metrics are pmean'd, and one
ppo.adam_step (one global-norm clip) is applied identically on every device; the rollouts then read their device's
replica. What differs from production at the batch boundary: production assembles the globally oldest `batch` rows
from ONE queue, here each device takes its own oldest share, so (a) each device drops its own straddling segment's tail
(up to N partial segments dropped per update instead of one; dropped_rows reports them) and (b) a device's share is a
little older or younger than the global cut would make it; versions, pool snapshots, checkpoints, lr schedule and
training.jsonl count updates exactly as with one device (samples = updates x batch).

Device threads (dp_threads, the default with several devices; dp_threads=0 = the lockstep loop above on the main
thread): one host thread per device (_DeviceWorker) owns the shard and steps its sets on its own (_step_shard: the
single-device round robin, every tick queued), with the device's own step counter and critic-version table; the
GIL is released in the engine's ctypes calls, the JAX waits and dispatches, torch's copies and most numpy, so the
devices' host work overlaps instead of running in sequence. Each step announces its closed rows on a condition; the
main thread waits until their SUM reaches `batch`, pauses the threads (each completes the step it is in, so a device
can close a few rows past the trigger, which stay closed for the next update: the update still takes exactly
`batch` rows, the oldest of each device's share) and, once all are parked, computes row_shares; the threads then take
and assemble their shares in parallel (gather, value pass, GAE); the main thread normalizes over the whole batch,
rebalances, runs the shard_map update and the bookkeeping, and starts the next rollout. Timing keys of the threads
are per device (per_device[d].timing_ms); `rollout` (main thread) is the wall time from the start of the rollouts to
the parked batch and `assemble_devices` the parallel assembly.
"""
import gc
import json
import os
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from polytopia_bot.ruleset import RULESET_ID, training_round_limit
from . import training_log
from .distributed import (END_FAMILY, VALUE_CHUNK, LearnerConfig, _critic_template, _template_params, critic_features,
                          episode_outcome, feature_shapes, network_spec, scheduled_learning_rate, segment_gae,
                          training_progress, warmed_learning_rate)
from .vector_actor import MAP_TABLE_DIRS, MapTable, _bucket


def pool_call_shape(n_members, max_lanes, Gs):
    """(Ka, B) of one act_group call: the members of a structure group on the move padded to a power of two, and the
    largest member's lane count bucketed (_bucket: 32, 64, ... capped at Gs). Each distinct pair is one XLA executable
    of act_group (and of merge / the reshape for Ka * B)."""
    return 1 << (n_members - 1).bit_length(), _bucket(max_lanes, Gs)


GC_NO_AUTO_FULL = 2 ** 30    # V3L_GC=scheduled: generation-2 threshold no run reaches (full collections only at _gc_point)


def pool_call_shapes(capacity, Gs):
    """Every (Ka, B) _pool_grouped can produce for a group of at most `capacity` members on sets of Gs lanes (Session A,
    after T1 v3l-s5m: a shape first met mid-run compiles on that rank alone while every other rank waits in the next
    collective, the all-rank ~10 s stalls)."""
    kas = sorted({pool_call_shape(m, 1, Gs)[0] for m in range(1, capacity + 1)})
    bs = sorted({pool_call_shape(1, n, Gs)[1] for n in range(1, Gs + 1)})
    return [(ka, b) for ka in kas for b in bs]

RING_SLACK_STEPS = 8          # ring steps past `ring` + batch / games: closed segments wait this long for the assembly
DIPLOMATIC_FAMILY = 14        # Masks field order (actions.py: RECOVER, DIPLOMATIC, END, RESIGN = 9, 14, 15, 16)
BOOT_STORE = 16384            # final-view boundaries (truncation bootstrap, cap_score_reward 0 only) kept per run


def _ceil_to(n, k):
    return -(-n // k) * k


VALUE_MODES = ('rollout', 'rollout-stale', 'assembly')


def value_sources(take, step_version, version):
    """Where the assembly takes each critic value from when the rollout stored them in the ring. `take` is
    SegmentLog.take's result; step_version[step % L] is the update count the critic had when that step's rows were
    valued (L = len(step_version)); `version` is the current update count (the critic the production learner scores
    with). Returns (row_stale [n], boundary_ring [S], boundary_call [S]): a stale row was valued by an older critic and
    is re-scored at assembly; a boundary comes from the ring when it is the seat's own row (a 33rd-row cut:
    boot_swap False, no stored final views) valued by the current critic, needs the small value call otherwise
    (forced cut = the other seat's orientation, a truncation's final views, or a stale row), and a terminal segment
    has no boundary. The ring row of a boundary is (boot_step, game), exactly the row the assembly path gathers."""
    L = len(step_version)
    segs = take['segments']
    row_stale = step_version[take['step'] % L] != version
    nonterminal = ~np.asarray(segs['terminal'], bool)
    own_row = (segs['boot_store'] < 0) & ~np.asarray(segs['boot_swap'], bool)
    current = step_version[segs['boot_step'] % L] == version
    boundary_ring = nonterminal & own_row & current
    boundary_call = nonterminal & ~boundary_ring
    return row_stale, boundary_ring, boundary_call


def row_shares(closed, batch):
    """Data parallel: the rows each device contributes to one update. closed [N] = each device's closed rows (their sum
    >= batch). Every device is asked for batch // N (the first batch % N devices one more); a device that has fewer
    gives all it has and its shortfall is spread evenly over the devices with rows to spare (water filling: the share
    is min(closed, level) for the highest level the batch affords, the last few rows one each to the first devices
    above the level). Returns int64 [N] with sum == batch and share <= closed."""
    closed = np.asarray(closed, np.int64)
    if closed.ndim != 1 or len(closed) == 0 or (closed < 0).any():
        raise ValueError('closed must be a non-empty vector of row counts')
    if int(closed.sum()) < batch:
        raise ValueError(f'{int(closed.sum())} closed rows cannot fill a batch of {batch}')
    lo, hi = 0, int(closed.max())                  # the largest level with sum(min(closed, level)) <= batch
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if int(np.minimum(closed, mid).sum()) <= batch:
            lo = mid
        else:
            hi = mid - 1
    share = np.minimum(closed, lo)
    rest = int(batch) - int(share.sum())           # < the number of devices above the level
    share[np.flatnonzero(closed > lo)[:rest]] += 1
    return share


def rebalance_plan(shares, per):
    """Data parallel: moves that leave every device with exactly `per` rows of the global batch (the update's static
    per-device shard; sum(shares) == len(shares) * per). Device d holds its rows at [0, shares[d]); a device with a
    surplus keeps [0, per) and sends [per, shares[d]), a device short of `per` receives at [shares[d], per), greedily in
    device order. Returns [(src, src_lo, dst, dst_lo, count)]; empty when every share is `per`."""
    shares = np.asarray(shares, np.int64)
    if int(shares.sum()) != len(shares) * per:
        raise ValueError('shares must sum to devices * per')
    give = [[d, per, int(s) - per] for d, s in enumerate(shares) if s > per]       # [src, next row, rows left]
    take = [[d, int(s), per - int(s)] for d, s in enumerate(shares) if s < per]     # [dst, next slot, slots left]
    moves, i, j = [], 0, 0
    while i < len(give) and j < len(take):
        k = min(give[i][2], take[j][2])
        moves.append((give[i][0], give[i][1], take[j][0], take[j][1], k))
        give[i][1] += k
        give[i][2] -= k
        take[j][1] += k
        take[j][2] -= k
        i += give[i][2] == 0
        j += take[j][2] == 0
    return moves


def pool_warm_on(env, size):
    """Session A's pool pre-compile (V3L_POOL_WARM=1, the default) for this process's board size, unless the size is
    listed in V3L_POOL_WARM_SKIP_SIZES ('30' or '20,30'). q8/q9/q10: on 30x30 the warm-up's
    all-shape pass is the new element in every OOM; T1 ran 30x30 without it (lazy compiles, 5 of 76 stalls on 30)."""
    if env.get('V3L_POOL_WARM', '1') != '1':
        return False
    skip = {int(x) for x in env.get('V3L_POOL_WARM_SKIP_SIZES', '').split(',') if x.strip()}
    return int(size) not in skip


def torch_partition(env):
    """torch's share of the card when V3L_TORCH_MEM_PARTITION=<margin> is set (stage 5 T2):
    1 - XLA_PYTHON_CLIENT_MEM_FRACTION - margin, else None (unchanged behaviour). With preallocation off, JAX grows on
    demand up to its fraction while torch's caching allocator keeps every block it ever took; q6/q8 30x30 ranks (view
    width 39227) died in update 2 when JAX's 11 GiB stale-row value chunk found the card full (CUDA_ERROR_OUT_OF_MEMORY).
    The cap makes torch release cached blocks before it crosses into JAX's share. No numerics change."""
    margin = env.get('V3L_TORCH_MEM_PARTITION', '')
    if margin == '':
        return None
    jax_frac = float(env.get('XLA_PYTHON_CLIENT_MEM_FRACTION') or .45)
    frac = 1. - jax_frac - float(margin)
    if not (0.1 <= frac < 1.):
        raise ValueError(f'torch memory fraction {frac:.3f} (jax {jax_frac}, margin {margin}) out of [0.1, 1)')
    return frac


def size_groups(spec, n_procs, rank, default):
    """(this process's board size, [size of every rank]) from 'size:processes,...' in rank order (stage 5); '' = every
    process plays `default`."""
    if not spec:
        return int(default), [int(default)] * n_procs
    sizes = []
    for part in spec.split(','):
        size, count = (int(x) for x in part.split(':'))
        if size < 8 or count < 1:
            raise ValueError(f'bad size group {part!r}')
        sizes += [size] * count
    if len(sizes) != n_procs:
        raise ValueError(f'V3L_SIZE_GROUPS {spec!r} covers {len(sizes)} processes, the run has {n_procs}')
    return sizes[rank], sizes


def size_games(spec, size, default, kind=int):
    """Games per process for board `size` from 'size:games,...' (V3L_SIZE_GAMES, stage 5); sizes not listed (or '')
    play `default`. kind=float: the same for V3L_SIZE_QUOTA weights."""
    for part in (spec.split(',') if spec else []):
        sz, value = part.split(':')
        value = kind(value)
        if value <= 0:
            raise ValueError(f'bad size entry {part!r}')
        if int(sz) == int(size):
            return value
    return kind(default)


def process_rows(games, batch, minibatch, accum):
    """Every process's rows per update, proportional to `games` (its games x its size's quota weight: the rate it
    closes rows), each a whole number of (batch // minibatch) steps x grad_accum micro-batches so every process runs
    the same minibatch steps."""
    steps = batch // minibatch
    unit = steps * accum
    total = sum(games)
    rows = [int(batch * g / total) // unit * unit for g in games]
    if min(rows) < unit:
        raise ValueError(f'a process gets {min(rows)} rows per update, below one step x accum ({unit})')
    return rows


def _shard_on(a, device):
    """The copy of the replicated jax.Array `a` that lives on `device` (a committed single-device array, no copy)."""
    for s in a.addressable_shards:
        if s.device == device:
            return s.data
    raise ValueError(f'array has no shard on {device}')


class _Checkpointer:
    """Writes checkpoints on a background thread. The loop hands over host copies (jax.device_get on the loop thread:
    a consistent snapshot of (params, Adam state, critic) at one version) and a function doing the writes (log rows,
    npz files, volume commit); submit() joins the previous write first so the files land in order. Errors surface on
    the next wait()/submit()."""

    def __init__(self):
        self.thread, self.error, self.seconds, self.writes = None, None, 0., 0

    def submit(self, fn):
        self.wait()

        def run():
            t0 = time.perf_counter()
            try:
                fn()
            except BaseException as exc:   # noqa: BLE001 - re-raised on the loop thread
                self.error = exc
            self.seconds = time.perf_counter() - t0
            self.writes += 1

        self.thread = threading.Thread(target=run, name='checkpoint-writer', daemon=True)
        self.thread.start()

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            exc, self.error = self.error, None
            raise RuntimeError('the background checkpoint write failed') from exc


class _DeviceWorker:
    """One device's host thread (data parallel, dp_threads): owns the shard (its sets, ring, SegmentLog, Lanes, fused
    calls, engine ticks, outcome staging) and runs the main thread's commands on it one at a time: 'init' (the initial
    state), 'roll' (steps until the trainer pauses the rollouts; returns the steps run), 'assemble' rows (the shard's
    take / gather / value pass / GAE; returns the part), 'exit'. The thread makes its GPU the torch current device
    first, so its torch allocations, streams, events and dlpack imports are on that GPU; its JAX work is committed to
    the shard's device by its inputs. An error is reported to the trainer (which stops the other threads' rollouts)
    and re-raised by result() on the main thread."""

    def __init__(self, trainer, sh):
        self.trainer, self.sh = trainer, sh
        self.inbox, self.outbox = queue.Queue(), queue.Queue()
        self.thread = threading.Thread(target=self._main, name=f'dp-device-{sh.index}', daemon=True)
        self.thread.start()

    def call(self, cmd, *args):
        self.inbox.put((cmd, args))

    def result(self):
        kind, value = self.outbox.get()
        if kind == 'error':
            raise RuntimeError(f'device {self.sh.index} thread failed') from value
        return value

    def _main(self):
        tr, sh = self.trainer, self.sh
        tr.torch.cuda.set_device(sh.index)
        run = dict(init=lambda: tr._init_shard(sh), roll=lambda: tr._roll_shard(sh),
                   assemble=lambda rows: tr._assemble_part(sh, rows))
        while True:
            cmd, args = self.inbox.get()
            if cmd == 'exit':
                return
            try:
                self.outbox.put(('ok', run[cmd](*args)))
            except BaseException as exc:   # noqa: BLE001 - re-raised on the main thread by result()
                tr._worker_failed(exc)
                self.outbox.put(('error', exc))


# ----------------------------------------------------------------------------- opponents

def rule_block_kind(kinds, Gs, block, always=False):
    """Per-block pool kind (0 = no-peace sets, 1 = peace sets) from the per-set rules `kinds` (Gs games per set), or None
    when every set has the same rule and `always` is off (the single-pool trainer: kind 0 for all, monitors off).
    always: keep the kinds even when
    uniform, so an all-peace run draws from its 'peace' pool (kind 1) and keeps the peace monitors (mixed_rules)."""
    if len(set(kinds)) <= 1 and not always:
        return None
    if Gs % block:
        raise ValueError(f'mixed rules per set need opponent blocks aligned to sets: {Gs} % {block}')
    return np.repeat(np.array(kinds, np.int8), Gs // block)


class Lanes:
    """Per-lane opponents and learner seats, per block of `block` lanes (as in vector_actor):
    a block draws its opponent (None = self-play with probability p_self, else a pool name by normalized
    probabilities; pool_per_process semantics), redrawn after 4 * block finished games; a lane gets its learner seat
    uniformly in {1..players} when its game ends; evicted pool names fall back to self-play.

    Session L stage 4 (2026-11-08): `block_kind` [blocks] gives every block a pool kind (0 = no_peace sets, 1 =
    peace-legal sets); set_pool takes either one spec (every kind, the default behaviour) or {kind: spec}, and each
    kind has its own self-play probability."""

    def __init__(self, games, players, block, rng, per_process=True, block_kind=None, fresh_draw=False):
        # fresh_draw (stage 4): the first set_pool with any pool mass redraws every block at once (no game has rows
        # yet), instead of leaving them self-play until each block has finished 4 * block games (~1,000+ updates)
        self.fresh_draw = bool(fresh_draw)
        self.G, self.players, self.block, self.rng, self.per_process = games, players, block, rng, per_process
        self.blocks = -(-games // block)
        self.block_kind = np.zeros(self.blocks, np.int8) if block_kind is None else np.asarray(block_kind, np.int8)
        if len(self.block_kind) != self.blocks:
            raise ValueError('block_kind needs one entry per block')
        self.kinds = sorted(set(self.block_kind.tolist()) | {0})
        self.pool_spec = {k: [] for k in self.kinds}
        self.available = set()
        self.p_self_kind = {k: 1. for k in self.kinds}
        self.p_self = 1.                    # kind 0's (the single-pool trainer's attribute)
        self.codes, self.code_name = {}, {}
        self.block_opp = [None] * self.blocks
        self.finished_since = np.zeros(self.blocks, np.int64)
        self.lane_name = [None] * games
        self.lane_seat = np.zeros(games, np.int32)
        self.lane_code = np.full(games, -1, np.int32)
        self.lane_pool = np.zeros(games, bool)
        # stage 5 T2 league: seat_hint[lane] = the seat (1-based)
        # a pool game's learner must take in the lane's CURRENT game (its Bardur seat), 0 = any (uniform); mirror[lane] =
        # the current game is Bardur vs Bardur; an opponent in mirror_only (hero-1v1, a Bardur-only bot) plays only on
        # mirror games, other games of its blocks fall back to self-play
        self.seat_hint = np.zeros(games, np.int8)
        self.mirror = np.ones(games, bool)
        self.mirror_only = set()
        self.block_opp = [self.draw_mode(int(self.block_kind[k])) for k in range(self.blocks)]
        for lane in range(games):
            self.set_lane(lane, self.assign(lane))

    def code(self, name):
        if name not in self.codes:
            self.codes[name] = len(self.codes)
            self.code_name[self.codes[name]] = name
        return self.codes[name]

    def draw_opponent(self, kind=0):
        spec = self.pool_spec[kind]
        names = [nm for nm, _ in spec if nm in self.available]
        probs = np.array([p for nm, p in spec if nm in self.available], np.float64)
        if not names or probs.sum() <= 0:
            return None
        return names[int(self.rng.choice(len(names), p=probs / probs.sum()))]

    def draw_mode(self, kind=0):
        if self.per_process and self.rng.random() < self.p_self_kind[kind]:
            return None
        return self.draw_opponent(kind)

    def assign(self, lane):
        k = lane // self.block
        opp = self.block_opp[k]
        if opp is None or (not self.per_process and self.rng.random() < self.p_self_kind[int(self.block_kind[k])]):
            return None, 0
        if opp in self.mirror_only and not self.mirror[lane]:
            return None, 0
        seat = int(self.rng.integers(1, self.players + 1))      # drawn always: the stream is unchanged without hints
        return opp, int(self.seat_hint[lane]) or seat

    def set_lane(self, lane, value):
        name, seat = value
        self.lane_name[lane] = name
        self.lane_seat[lane] = seat
        self.lane_pool[lane] = name is not None
        self.lane_code[lane] = self.code(name) if name is not None else -1

    def set_pool(self, spec, available):
        """New pool spec [(name, probability)] (or {kind: spec}) and the names with parameters: p_self, evicted
        blocks and lanes."""
        by_kind = spec if isinstance(spec, dict) else {k: spec for k in self.kinds}
        self.pool_spec = {k: list(by_kind.get(k, ())) for k in self.kinds}
        self.available = set(available)
        self.p_self_kind = {k: max(0., 1. - sum(p for _, p in self.pool_spec[k])) for k in self.kinds}
        self.p_self = self.p_self_kind[0]
        for k in range(self.blocks):
            if self.block_opp[k] is not None and self.block_opp[k] not in self.available:
                self.block_opp[k] = self.draw_mode(int(self.block_kind[k]))
        for lane in range(self.G):
            if self.lane_name[lane] is not None and self.lane_name[lane] not in self.available:
                self.set_lane(lane, (None, 0))
        if self.fresh_draw and any(self.pool_spec[k] for k in self.kinds):
            self.fresh_draw = False
            for k in range(self.blocks):
                self.block_opp[k] = self.draw_mode(int(self.block_kind[k]))
                for lane in range(k * self.block, min((k + 1) * self.block, self.G)):
                    self.set_lane(lane, self.assign(lane))

    def set_tribes(self, lanes, tribes, learner_tribe):
        """The seat tribes [k, players] of the games `lanes` now play: seat hints (the learner_tribe seat; a mirror or a
        game without it: any seat) and the mirror flags."""
        own = np.asarray(tribes) == learner_tribe
        one = own.sum(1) == 1
        self.seat_hint[lanes] = np.where(one, own.argmax(1) + 1, 0)
        self.mirror[lanes] = own.all(1)

    def game_over(self, lanes):
        """The lanes whose games just finished: block redraws and new (opponent, seat) per lane."""
        for lane in lanes.tolist():
            k = lane // self.block
            self.finished_since[k] += 1
            if self.finished_since[k] >= 4 * self.block:
                self.block_opp[k], self.finished_since[k] = self.draw_mode(int(self.block_kind[k])), 0
            self.set_lane(lane, self.assign(lane))


# ----------------------------------------------------------------------------- rows and segments

def resume_optimizer(opt, version, critic_warmup, init):
    """The Adam state a resumed run continues with. A run that resumes INSIDE its critic warm-up (version <
    critic_warmup; T3 from u88890 with critic_warmup 89090) starts a fresh Adam state: the warm-up zeroes
    the policy's gradients, but a resumed first moment would keep moving the policy along its last direction for
    ~1 / (1 - beta1 / sqrt(beta2)) ~ 10 updates' worth. Otherwise the saved state, exactly as before."""
    return init() if critic_warmup and version < critic_warmup else opt


class SegmentLog:
    """Host bookkeeping of the learner rows in the device ring and their same-seat segments (SegmentCollector /
    RowStore semantics). A row is (step, game); per (game, seat) the steps of the open segment's rows sit in
    `steps[game, seat, :count]`. Closed segments queue as events (arrays per close) until take() hands out the
    oldest `batch` rows."""

    KEYS = ('game', 'seat', 'steps', 'length', 'terminal', 'reward', 'boot_step', 'boot_swap', 'boot_store')

    def __init__(self, games, players, segment, max_age):
        self.G, self.players, self.segment, self.max_age = games, players, segment, max_age
        self.W = segment + 1
        self.count = np.zeros((games, players), np.int32)
        self.first = np.full((games, players), -1, np.int64)
        self.steps = np.full((games, players, self.W), -1, np.int64)
        self.events = []
        self.closed_rows = 0
        self.closed_segments = 0
        self.recorded = 0
        self.forced_rows = 0
        self.forced_segments = 0
        self.cut_segments = 0
        self.finished_segments = 0

    def _close(self, g, s, length, terminal, reward, boot_step, boot_swap, boot_store):
        k = len(g)
        if k == 0:
            return
        steps = self.steps[g, s].copy()
        steps[np.arange(self.W)[None, :] >= length[:, None]] = -1
        self.events.append(dict(game=g.astype(np.int64), seat=s.astype(np.int32), steps=steps,
                                length=length.astype(np.int32), terminal=np.asarray(terminal, bool),
                                reward=np.asarray(reward, np.float32), boot_step=boot_step.astype(np.int64),
                                boot_swap=np.asarray(boot_swap, bool), boot_store=boot_store.astype(np.int64)))
        self.closed_rows += int(length.sum())
        self.closed_segments += k

    def force_close(self, t, games, mover0):
        """Open segments of `games` whose first row is max_age steps old: closed with the boundary at the seat's
        current state (the ring row of step t, swapped when the seat is not the mover)."""
        cnt = self.count[games]
        old = (cnt > 0) & (self.first[games] <= t - self.max_age)
        if not old.any():
            return 0
        gi, si = np.nonzero(old)
        g, s = games[gi], si.astype(np.int32)
        length = cnt[gi, si]
        z = np.zeros(len(g), np.float32)
        self._close(g, s, length, np.zeros(len(g), bool), z, np.full(len(g), t, np.int64), s != mover0[gi],
                    np.full(len(g), -1, np.int64))
        self.count[g, s] = 0
        self.first[g, s] = -1
        self.forced_rows += int(length.sum())
        self.forced_segments += len(g)
        return len(g)

    def record(self, t, games, seats0):
        """This step's learner rows: games (global indices) and the mover seat (0-based) of each. A seat's
        (segment + 1)-th row cuts the first `segment` rows (boundary = this row) and starts the next segment."""
        if len(games) == 0:
            return
        c = self.count[games, seats0]
        self.steps[games, seats0, c] = t
        new = c == 0
        if new.any():
            self.first[games[new], seats0[new]] = t
        c1 = c + 1
        self.count[games, seats0] = c1
        full = c1 > self.segment
        if full.any():
            gf, sf = games[full], seats0[full]
            k = len(gf)
            self._close(gf, sf, np.full(k, self.segment, np.int32), np.zeros(k, bool), np.zeros(k, np.float32),
                        np.full(k, t, np.int64), np.zeros(k, bool), np.full(k, -1, np.int64))
            self.steps[gf, sf, 0] = t
            self.count[gf, sf] = 1
            self.first[gf, sf] = t
            self.cut_segments += k
        self.recorded += len(games)

    def finish(self, games, rewards, ended, seats_mask, store_idx):
        """Finished games: rewards [k, players], ended [k] (terminal), seats_mask [k, players] = the seats whose
        rows are emitted (self-play: all; a pool game: the learner seat; the others' rows are dropped), store_idx [k]
        = the row of the final views in the boundary store (-1 when the segment is terminal)."""
        if len(games) == 0:
            return
        cnt = self.count[games]
        has = (cnt > 0) & seats_mask
        if has.any():
            gi, si = np.nonzero(has)
            g, s = games[gi], si.astype(np.int32)
            length = cnt[gi, si]
            term = ended[gi]
            if (~term & (store_idx[gi] < 0)).any():
                raise RuntimeError('a truncation bootstrap needs the final views in the boundary store')
            last = self.steps[g, s, length - 1]
            self._close(g, s, length, term, rewards[gi, si], last, s == 1, np.where(term, -1, store_idx[gi]))
            self.finished_segments += len(g)
        gi, si = np.nonzero(seats_mask)
        self.count[games[gi], si] = 0
        self.first[games[gi], si] = -1

    def take(self, batch, t, ring_len):
        """The oldest closed segments covering `batch` rows (the last one may straddle): flat rows (step, game, seg,
        pos) of every row of those segments, the per-segment tables, and n (>= batch; the rows past batch are dropped
        by the caller after GAE). t = the last recorded step; rows older than ring_len steps are gone."""
        if self.closed_rows < batch:
            return None
        cat = {k: np.concatenate([e[k] for e in self.events]) for k in self.KEYS}
        length = cat['length']
        cum = np.cumsum(length, dtype=np.int64)
        used = int(np.searchsorted(cum, batch, side='left')) + 1
        n = int(cum[used - 1])
        seg = np.repeat(np.arange(used, dtype=np.int32), length[:used])
        start = cum[:used] - length[:used]
        pos = (np.arange(n, dtype=np.int64) - np.repeat(start, length[:used])).astype(np.int32)
        steps = cat['steps'][:used][seg, pos]
        game = cat['game'][:used][seg]
        if (steps < 0).any():
            raise RuntimeError('segment bookkeeping error: a row has no step')
        if t - int(steps.min()) >= ring_len:
            raise RuntimeError(f'row {t - int(steps.min())} steps old has left the device ring ({ring_len} steps); '
                               f'raise --ring or lower --batch / raise --games')
        rest = {k: v[used:] for k, v in cat.items()}
        self.events = [rest] if len(rest['length']) else []
        self.closed_rows -= n
        self.closed_segments -= used
        return dict(step=steps, game=game, seg=seg, pos=pos, n=n, segments={k: cat[k][:used] for k in self.KEYS})


# ----------------------------------------------------------------------------- trainer

class GpuResidentTrainer:
    """GPU-resident trainer of the hero recipe on one GPU or data parallel on `devices` GPUs (module docstring). `config`: the recipe dict (experiments/hero/*.json) or a
    LearnerConfig; `run_dir`: /results/distributed/{run_id}; `parent`: the policy checkpoint to continue (its
    network_spec sets the architecture), '' = random init of `network` = dict(width, blocks, architecture, ...)."""
    # Session A profiling knobs (_setup_train_knobs, called by run()): off unless set
    train_profile, train_bench, _bench_left, gc_mode, _gc_log, gc_full_every = False, (), -1, '', None, 1
    gc_refreeze, gc_deep_every, gc_last, value_tail = False, 0, {}, 0

    def __init__(self, config, run_dir, parent='', *, critic_path=None, training_seconds=None, games=4096, sets=4,
                 ring=160, minibatch=None, batch=None, grad_accum=None, checkpoint_every=None, snapshot_every=None,
                 map_table=None, device=0, seed=None, block=24, network=None, value_mode='rollout', value_check=False,
                 fused=True, devices=1, dp_threads=None, log=print, commit=lambda: None):
        import torch
        import jax
        self.torch, self.jax = torch, jax
        if value_mode not in VALUE_MODES:
            raise ValueError(f'value_mode must be one of {VALUE_MODES}')
        self.value_mode, self.value_check, self.fused = value_mode, bool(value_check), bool(fused)
        cfg = dict(asdict(config) if isinstance(config, LearnerConfig) else config)
        if critic_path is not None:
            cfg['critic_path'] = critic_path
        if training_seconds is not None:
            cfg['training_seconds'] = int(training_seconds)   # the LR decay runs over the run's own length
        if minibatch:
            cfg['minibatch'] = int(minibatch)
        if batch:
            cfg['batch'] = int(batch)
        if grad_accum:
            cfg['grad_accum'] = int(grad_accum)
        if checkpoint_every:
            cfg['checkpoint_every'] = int(checkpoint_every)
        if snapshot_every:
            cfg['snapshot_every'] = int(snapshot_every)
        if seed is not None:
            cfg['seed'] = int(seed)
        cfg['frozen'] = tuple(tuple(x) for x in cfg.get('frozen', ()))
        self.config = cfg = LearnerConfig(**cfg)
        cfg.validate()
        self._check_supported(cfg)
        self.root, self.parent, self.log, self.commit = Path(run_dir), parent, log, commit
        # multi-process data parallel (Claude session A, 2026-11-08; docs/claude-gpu-resident-multinode-design.md): one
        # process per container, jax.distributed initialized by the caller before this constructor. Every process runs
        # the threaded data-parallel trainer over its local GPUs; the update's mesh, gradient pmean and advantage
        # statistics span all processes. Rank 0 writes checkpoints / logs to run_dir; rank r > 0 logs to run_dir/rank<r>
        # and resumes from rank 0's checkpoints.
        self.n_procs, self.rank = jax.process_count(), jax.process_index()
        self.ckpt_root = self.root
        if self.rank:
            self.root = self.root / f'rank{self.rank}'
        self.is_writer = self.rank == 0
        self.G, self.sets, self.ring, self.block = int(games), int(sets), int(ring), int(block)
        if self.sets < 1 or self.G % self.sets:
            raise ValueError('games must be a positive multiple of sets')
        self.Gs = self.G // self.sets
        self.map_table_path = map_table
        self.device_index = device
        # data parallel (module docstring, "Several GPUs"): devices 0 = every visible GPU; games / sets / ring per device
        n_dev = int(devices) if devices else torch.cuda.device_count()
        if n_dev < 1 or (n_dev > 1 and device != 0):
            raise ValueError('devices must be >= 1 (0 = all visible); several devices use the CUDA ordinals 0..N-1')
        if n_dev > torch.cuda.device_count() or n_dev > len(jax.local_devices()):
            raise ValueError(f'devices={n_dev}: torch sees {torch.cuda.device_count()} GPUs, JAX {len(jax.local_devices())}')
        if self.n_procs > 1 and n_dev * self.n_procs != len(jax.devices()):
            raise ValueError(f'multi-process: every process must drive all its local GPUs; '
                             f'{n_dev} x {self.n_procs} != {len(jax.devices())} global devices')
        # one GPU per process (n_dev == 1, several processes) is the GIL-free layout: each process runs the single-device
        # lockstep loop on its own interpreter; only the update is collective (session A, 2026-11-08)
        n_global = n_dev * self.n_procs
        if n_global > 1:
            local = cfg.minibatch // n_global
            if cfg.batch % n_global or cfg.minibatch % n_global or local % cfg.grad_accum:
                raise ValueError(f'devices={n_global}: batch and minibatch must split evenly over the devices and the '
                                 f'per-device minibatch {local} must be a multiple of grad_accum {cfg.grad_accum}')
        self.n_devices = n_dev
        self.n_global = n_global
        # Stage 5 (Session L, mixed sizes in one run, grouped by size, not padded): V3L_SIZE_GROUPS
        # = 'size:processes,...' in rank order (e.g. '11:8,14:12,18:12') gives every process ONE board size; the
        # parameter tree is size-independent (transformer_v3 plays any size with the base-size weights), each process
        # gathers / scores / GAEs its own rows, computes its gradient locally and the gradients are pmean'd over all
        # devices before the one Adam step (_mixed). Unset: every process plays cfg.size (the recipe, unchanged).
        self.play_size, self.size_of_rank = size_groups(os.environ.get('V3L_SIZE_GROUPS', ''), self.n_procs, self.rank,
                                                        cfg.size)
        # V3L_SIZE_GAMES = 'size:games,...' (stage 5): games per process for a board size (default --games). A process's
        # share of every batch is proportional to its games (it produces rows in that proportion), rounded to whole
        # minibatch steps x grad_accum; gradients are then averaged weighted by rows (apply_mean), so the update is the
        # mean over all rows exactly as with equal shares.
        self.games_of_rank = [size_games(os.environ.get('V3L_SIZE_GAMES', ''), sz, self.G) for sz in self.size_of_rank]
        # V3L_SIZE_QUOTA = 'size:weight,...' (default 1): a size's rows per game per update relative to the others. Big
        # boards close rows more slowly per game; the update waits for the last process, so quotas proportional to
        # each process's row RATE (games x relative ticks/s) keep every GPU busy. Rows are averaged by count.
        self.quota_of_rank = [g * size_games(os.environ.get('V3L_SIZE_QUOTA', ''), sz, 1.0, float)
                              for g, sz in zip(self.games_of_rank, self.size_of_rank)]
        self.mixed = len(set(self.size_of_rank)) > 1 or len(set(self.quota_of_rank)) > 1
        if self.mixed and n_dev != 1:
            raise ValueError('mixed board sizes need one GPU per process (the size is per process)')
        self.G = self.games_of_rank[self.rank]
        if self.G % self.sets:
            raise ValueError(f'games {self.G} of this process must be a multiple of sets {self.sets}')
        self.Gs = self.G // self.sets
        self.rows_of_rank = process_rows(self.quota_of_rank, cfg.batch, cfg.minibatch, cfg.grad_accum) if self.mixed \
            else [cfg.batch // self.n_procs] * self.n_procs
        self.process_rows = self.rows_of_rank[self.rank]
        self.async_ticks = self.sets > 1 or n_dev > 1     # several devices: every tick queued, synced next step
        # several devices: one host thread per device for the rollout and its share of the assembly (dp_threads None =
        # on); 0 = the lockstep loop on the main thread. One device never starts a thread.
        self.threaded = n_dev > 1 and (dp_threads is None or bool(dp_threads))
        self.workers, self.last_roll_steps = [], []
        self._worker_error = None
        self._pause = threading.Event()
        self._cond = threading.Condition()
        restarted = self.ckpt_root.exists() and (self.ckpt_root / 'latest.npz').exists()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'snapshots').mkdir(exist_ok=True)
        self.rng = np.random.default_rng(cfg.seed)
        self.version = 0
        self.resume_history, self.resume_elapsed = [], 0.
        self.historical = []                          # (name, params on device, fp16-rounded)
        self._build_network(network)
        self._build_engine()
        self._build_jax()
        if restarted:
            self._resume()
        self._place()
        self._set_pool()

    # ---- setup
    @staticmethod
    def _check_supported(cfg):
        unsupported = dict(players=cfg.players != 2, anchor_coef=cfg.anchor_coef, unit_kl_coef=cfg.unit_kl_coef,
                           unit_entropy_coef=cfg.unit_entropy_coef, upgo_coef=cfg.upgo_coef, pbrs_coef=cfg.pbrs_coef,
                           ema_decay=cfg.ema_decay, advantage_keep=cfg.advantage_keep < 1,
                           omniscient_critic=not cfg.omniscient_critic, learner_values=not cfg.learner_values,
                           entropy_target=cfg.entropy_target is not None, kl_adapt_target=cfg.kl_adapt_target is not None)
        bad = [k for k, v in unsupported.items() if v]
        if bad:
            raise NotImplementedError(f'the GPU-resident trainer implements the hero recipe only; unsupported: {bad}')

    def _build_network(self, network):
        jax = self.jax
        from .runner import Config, load_checkpoint
        from .observation import OBS_VERSION
        cfg = self.config
        if self.parent:
            self.network = network_spec(self.parent)
            template = _template_params(**self.network)
            self.params = load_checkpoint(self.parent, template)
            parent_config = json.loads(Path(self.parent).with_suffix('.json').read_text())['config']
        else:
            spec = dict(size=cfg.size, blocks=4, extra_blocks=0, width=128, architecture='normalized-v2', obs_version=OBS_VERSION)
            spec.update(network or {})
            self.network = spec
            board, glob = feature_shapes(cfg.players, spec.get('obs_version', 1))
            self.params = self._fresh(board, glob, spec)
            parent_config = dict(size=cfg.size, width=spec['width'], blocks=spec['blocks'], extra_blocks=spec.get('extra_blocks', 0),
                                 architecture=spec['architecture'], players=cfg.players, obs_version=spec.get('obs_version', 1),
                                 no_resign=True, reward_mode='native', global_pool=bool(spec.get('global_pool', False)),
                                 block_pool=int(spec.get('block_pool', 0) or 0), attention=int(spec.get('attention', 0)))
        self.run_config = Config(**{**parent_config, 'learning_rate': cfg.learning_rate, 'entropy_coef': cfg.entropy_coef,
                                    'gamma': cfg.gamma, 'gae_lambda': cfg.gae_lambda, 'minibatch': cfg.minibatch,
                                    'epochs': cfg.epochs, 'seed': cfg.seed})
        template = _template_params(**self.network)
        self.structure = jax.tree.structure(template)
        from .transformer_v3 import is_v3
        self.v3 = is_v3(self.network['architecture'])
        # the omniscient critic: fresh template or the parent's (critic_path), as initial_critic
        self.critic = _critic_template(**self.network)
        if cfg.critic_path:
            self.critic = load_checkpoint(cfg.critic_path, self.critic)
            self.log(json.dumps(dict(stage='critic_loaded', path=str(cfg.critic_path))))
        self.frozen = {}
        # Session L stage 4 / Session S: a frozen opponent may have its own architecture when frozen_any_arch() (the
        # same board size; the features are the learner's view_features, as the GPU arena plays mixed pairs); it is
        # loaded with its own template and act_subset dispatches network.sample on its parameter tree.
        any_arch = self.frozen_any_arch()
        for name, path in [(n, pth) for n, pth, _ in cfg.frozen] + list(self.extra_opponents()):
            spec = network_spec(path)
            if spec != self.network:
                if not any_arch or spec['size'] != self.network['size'] or spec.get('players', 2) != cfg.players:
                    raise ValueError(f'frozen opponent {name} is a {spec} network; the learner is {self.network}')
                self.frozen[name] = self._rounded(load_checkpoint(path, _template_params(**spec)))
                self.log(json.dumps(dict(stage='frozen_cross_architecture', name=name, path=str(path), spec=spec)))
                continue
            self.frozen[name] = self._rounded(load_checkpoint(path, template))
        self.frozen_probability = {name: p for name, _, p in cfg.frozen}

    # ---- stage-4 hooks: the defaults are the recipe's behaviour
    def frozen_any_arch(self):
        return False

    def extra_opponents(self):
        """[(name, checkpoint path)] frozen opponents beyond cfg.frozen, drawn only through an overridden pool_spec."""
        return []

    def set_no_peace(self, g, k):
        """The engine rule of game set k on global device g: no_peace (True) or peace legal (False)."""
        return bool(self.config.no_peace)

    def _fresh(self, board, glob, spec):
        from . import network
        return network.init(self.jax.random.key(self.config.seed), board, glob, size=spec['size'], width=spec['width'],
                            blocks=spec['blocks'], architecture=spec['architecture'], extra_blocks=spec.get('extra_blocks', 0),
                            global_pool=spec.get('global_pool', False), block_pool=spec.get('block_pool', False),
                            attention=spec.get('attention', 0))

    def _rounded(self, params):
        """The pool's transport copy: float16-rounded leaves back in float32 (Learner._leaves + actor upcast)."""
        import jax.numpy as jnp
        return self.jax.tree.map(lambda a: a.astype(jnp.float16).astype(jnp.float32)
                                 if jnp.issubdtype(a.dtype, jnp.floating) else a, params)

    def _on(self, index):
        """The torch CUDA device context of a shard's GPU (CUDA ordinal `index`): its events, synchronizes and default
        allocations go to that GPU (one device: a no-op, the device is current since _build_engine)."""
        import contextlib
        return contextlib.nullcontext() if self.n_devices == 1 else self.torch.cuda.device(index)

    def _build_engine(self):
        torch = self.torch
        from .cpu_engine import view_layout
        cfg = self.config
        size, players, G = self.play_size, cfg.players, self.G
        self.device = torch.device('cuda', self.device_index)
        torch.cuda.set_device(self.device)
        self.torch_mem_fraction = torch_partition(os.environ)
        if self.torch_mem_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(self.torch_mem_fraction, self.device)
        self.layout = view_layout(size, players)
        offsets, at = {}, 0
        for name, shape in self.layout:
            offsets[name] = at
            at += int(np.prod(shape))
        self.vw, self.player_at = at, offsets['player']
        shapes = dict(self.layout)
        self.peace_at, self.peace_w = offsets.get('peace', 0), int(np.prod(shapes.get('peace', (0,))))
        self.turn_at = offsets.get('turn', 0)
        limit = training_round_limit(size) if cfg.max_turns is None else int(cfg.max_turns)
        self.round_limit = limit
        self.late_round = int(0.75 * limit) if limit else 150      # stage-4 monitor: live games past 75% of the cap
        # maps: the precomputed table on each device, drawn per game per step (only finished games use their seed)
        path = self.map_table_path or MapTable.find(cfg.maps, size, players, cfg.map_seed_low, cfg.map_seed_high)
        path = path and str(path).replace('{size}', str(size))     # stage 5: one table per board size
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f'map table for {cfg.maps} {size} {players}p [{cfg.map_seed_low}, {cfg.map_seed_high}) '
                                    f'not found in {MAP_TABLE_DIRS}; pass map_table')
        table = MapTable(path, None)
        if table.lo != cfg.map_seed_low or table.hi != cfg.map_seed_high:
            raise ValueError(f'map table seeds [{table.lo}, {table.hi}) differ from the config')
        self.map_lo, self.map_hi, self.map_types, self.map_kind = table.lo, table.hi, table.types, table.kind
        # stage 5 T2 league: seat tribes per table row (full-game engine b12 tables), the learner's tribe (Bardur = 4) and the rows
        # whose seats are both that tribe (mirror opponents' blocks draw their seeds from them)
        self.map_tribes = getattr(table, 'tribes', None)
        self.learner_tribe = int(getattr(self, 'league_learner_tribe', 4))
        self.mirror_rows = None
        if self.map_tribes is not None:
            self.mirror_rows = np.flatnonzero((self.map_tribes == self.learner_tribe).all(1)).astype(np.int64)
        n = size * size
        self.fam_bytes, self.src_bytes, self.tgt_bytes = 3, -(-n // 8), -(-max(n, 128) // 8)
        self.row_mask_bytes = self.fam_bytes + self.src_bytes + self.tgt_bytes
        # the device ring: `ring` steps of addressable rows plus the slack closed segments wait for the assembly (each
        # device assembles about batch / devices rows)
        self.L = self.ring + -(-(self.process_rows if self.n_procs > 1 else cfg.batch // self.n_global) // G) + RING_SLACK_STEPS
        # stage 4: pool lanes record only the learner's seat, so a batch closes in more steps (Session S's
        # V3S_RING_EXTRA_STEPS); ring_extra_steps (class or instance attribute) adds that headroom (default 0)
        self.L += int(getattr(self, 'ring_extra_steps', 0))
        # a device's assembled rows (its share + the straddling segment's tail). One GPU per process: the share is
        # exactly this process's rows (no rebalancing across processes), so the buffers need not hold the whole batch
        # (stage 5: 30 GB per GPU at 18x18 otherwise)
        share = self.process_rows if self.n_procs > 1 and self.n_devices == 1 else cfg.batch
        self.N_cap = share + VALUE_CHUNK
        self.shards = []
        for d in range(self.n_devices):
            with self._on(self.device_index + d):
                self.shards.append(self._build_shard(d, table, limit))
        sh0 = self.shards[0]
        self.mask_bytes, self.mask_width = sh0.gsets[0].env.mask_bytes, sh0.gsets[0].env.mask_width
        self.ring_bytes = sh0.ring_bytes

    def _build_shard(self, d, table, limit):
        """One device's engines, ring, batch buffers and host bookkeeping. Device 0 draws from the run's streams (the
        config seed: rng, map generator), so one device is the single-GPU trainer exactly; device d > 0 has its own
        streams seeded from (seed, d)."""
        torch = self.torch
        from .gpu_engine import GpuGames
        cfg = self.config
        size, players, G, Gs, L = self.play_size, cfg.players, self.G, self.Gs, self.L
        n = size * size

        class Shard:
            pass

        sh = Shard()
        sh.d = d
        sh.index = self.device_index + d
        sh.device = torch.device('cuda', sh.index)
        sh.jax_device = self.jax.local_devices()[sh.index]
        sh.g = self.rank * self.n_devices + d      # global device index: the rank-0 device 0 keeps the run's streams
        hw = getattr(sh.jax_device, 'local_hardware_id', sh.index)
        if hw is not None and hw != sh.index:
            raise RuntimeError(f'JAX device {sh.index} is CUDA ordinal {hw}; torch and JAX must agree on the GPU order')
        if sh.g == 0:
            sh.rng = self.rng
            sh.map_gen = torch.Generator().manual_seed(int(cfg.seed))
        else:
            sh.rng = np.random.default_rng([int(cfg.seed), sh.g])
            sh.map_gen = torch.Generator().manual_seed(int(np.random.SeedSequence([int(cfg.seed), sh.g]).generate_state(1)[0]))
        sh.clock, sh.calls = {}, {}
        # the table as two device tensors ([3, M, n] terrain / structure / resource and [M, players] capitals): a set's
        # layout rows are two index_selects per step, not four
        sh.table3 = torch.from_numpy(np.stack(table.arrays[:3])).to(sh.device)
        sh.table_caps = torch.from_numpy(table.arrays[3]).to(sh.device)
        sh.playing_kind = np.zeros(G, np.int8)
        sh.playing_tribes = np.full((G, players), self.learner_tribe, np.int8)

        class GameSet:
            pass

        sh.gsets = []
        for k in range(self.sets):
            gs = GameSet()
            gs.k, gs.lo, gs.hi = k, k * Gs, (k + 1) * Gs
            gs.games = np.arange(gs.lo, gs.hi)
            gs.no_peace = bool(self.set_no_peace(sh.g, k))
            gs.peace = not gs.no_peace
            gs.env = GpuGames(size=size, games=Gs, players=players, max_turns=limit, no_resign=True,
                              max_actions=int(getattr(cfg, 'max_actions', None) or 2 ** 31 - 1),
                              no_peace=gs.no_peace, device=sh.index)   # Kevin 28cd1b5: pass the recipe's no_peace
            gs.peace_steps = torch.zeros((), dtype=torch.int64, device=sh.device)      # game-steps at peace
            gs.diplo = torch.zeros((), dtype=torch.int64, device=sh.device)            # learner diplomatic decisions
            gs.late_steps = torch.zeros((), dtype=torch.int64, device=sh.device)       # game-steps past late_round
            if gs.env.view_width != self.vw:
                raise RuntimeError(f'packed view width differs: production layout {self.vw} vs C engine {gs.env.view_width}')
            seeds = self._draw_seeds(sh, Gs)
            sh.playing_kind[gs.lo:gs.hi] = self.map_kind[seeds - self.map_lo]
            if self.map_tribes is not None:
                sh.playing_tribes[gs.lo:gs.hi] = self.map_tribes[seeds - self.map_lo]
            host_layouts = tuple(np.ascontiguousarray(a[seeds - self.map_lo]) for a in table.arrays)
            gs.env.reset(seeds, host_layouts)
            gs.seeds = torch.from_numpy(seeds).to(sh.device)
            gs.seeds_host = seeds
            # the next step's seeds and table rows: pinned host buffers, two non-blocking H2D copies per step (the tick
            # synchronizes torch's stream before it reads them, which also orders the host's next write after the copy)
            gs.seeds_pin = torch.empty(Gs, dtype=torch.int32).pin_memory()
            gs.rows_pin = torch.empty(Gs, dtype=torch.int64).pin_memory()
            gs.rows = torch.zeros(Gs, dtype=torch.int64, device=sh.device)
            gs.layouts3 = torch.zeros((3, Gs, n), dtype=torch.int32, device=sh.device)
            gs.layout_caps = torch.zeros((Gs, players), dtype=torch.int32, device=sh.device)
            gs.layouts = (gs.layouts3[0], gs.layouts3[1], gs.layouts3[2], gs.layout_caps)
            gs.out = gs.env.alloc_device_outputs(sh.device)
            # the first state's masks and views from the host-pointer calls, into the tensors the engine writes afterwards
            gs.out['masks'].copy_(torch.from_numpy(gs.env.legal_masks(packed=True)))
            view = gs.env.observe()
            mover = view[:, self.player_at].astype(np.int32)
            others = np.stack([gs.env.observe(player=p) for p in range(1, players + 1)], 1)      # [Gs, players, vw]
            opp = others[np.arange(Gs), np.where(mover == 1, 1, 0)]
            gs.out['view'].copy_(torch.from_numpy(view))
            gs.out['opp_view'][:, 0].copy_(torch.from_numpy(np.ascontiguousarray(opp)))
            gs.inputs_buf = torch.zeros((3, Gs), dtype=torch.int32, device=sh.device)   # unfused: family / source / target rows
            # the tick's outcomes: one device staging buffer (scores, rewards, status, then the three flags, as bytes; every
            # typed segment starts 4-byte aligned) copied to pinned host memory in one non-blocking transfer (_stage_outcomes)
            # and read after the set's JAX work is dispatched (_outcomes)
            seg_bytes = (4 * players * Gs, 4 * players * Gs, 4 * Gs, 3 * Gs)
            offs = np.concatenate([[0], np.cumsum(seg_bytes)])
            total = int(offs[-1])
            gs.stage_dev = torch.empty(total, dtype=torch.uint8, device=sh.device)
            gs.stage_host = torch.empty(total, dtype=torch.uint8).pin_memory()
            sl = [slice(int(offs[i]), int(offs[i + 1])) for i in range(4)]
            gs.stage_in = (gs.stage_dev[sl[0]].view(torch.int32).view(Gs, players), gs.stage_dev[sl[1]].view(torch.float32).view(Gs, players),
                           gs.stage_dev[sl[2]].view(torch.int32), gs.stage_dev[sl[3]])
            host = gs.stage_host.numpy()
            gs.stage_out = dict(scores=host[sl[0]].view(np.int32).reshape(Gs, players), rewards=host[sl[1]].view(np.float32).reshape(Gs, players),
                                status=host[sl[2]].view(np.int32), flags=host[sl[3]].reshape(3, Gs))
            gs.stage_event = torch.cuda.Event()
            gs.features = gs.player = gs.mover = gs.value = None
            gs.in_flight, gs.pending, gs.hold = False, None, None
            sh.gsets.append(gs)
        for gs in sh.gsets:
            # JAX aliases of the engine's output tensors, imported once (the buffers are fixed; every read is dispatched
            # after the tick that wrote them completed on the host, every tick after the reads completed)
            gs.view_j = self._to_jax(gs.out['view'])
            gs.opp_j = self._to_jax(gs.out['opp_view'][:, 0])
            gs.masks_j = self._to_jax(gs.out['masks'])
        z = lambda shape, dt: torch.zeros(shape, dtype=dt, device=sh.device)
        sh.ring_view = z((L * G, self.vw), torch.int16)
        sh.ring_opp = z((L * G, self.vw), torch.int16)
        sh.ring_masks = z((L * G, self.row_mask_bytes), torch.uint8)
        sh.ring_choice = z((L * G, 3), torch.int32)
        sh.ring_fv = z((2, L * G), torch.float32)                   # row 0 logp, row 1 the critic's value (value_mode rollout)
        sh.ring_logp, sh.ring_value = sh.ring_fv[0], sh.ring_fv[1]
        sh.ring_bytes = sum(x.numel() * x.element_size() for x in
                            (sh.ring_view, sh.ring_opp, sh.ring_masks, sh.ring_choice, sh.ring_fv))
        # the assembled batch (fixed capacity: the device's share + the straddling segment's tail < VALUE_CHUNK rows)
        N_cap = self.N_cap
        sh.buf_view, sh.buf_opp = z((N_cap, self.vw), torch.int16), z((N_cap, self.vw), torch.int16)
        sh.buf_masks = z((N_cap, self.row_mask_bytes), torch.uint8)
        sh.buf_choice, sh.buf_logp = z((N_cap, 3), torch.int32), z((N_cap,), torch.float32)
        sh.buf_value = z((N_cap,), torch.float32)
        # scratch pair for the small value calls (stale rows, boundaries): one VALUE_CHUNK of (view, opp) rows
        sh.tmp_view, sh.tmp_opp = z((VALUE_CHUNK, self.vw), torch.int16), z((VALUE_CHUNK, self.vw), torch.int16)
        sh.boot_store = None   # [BOOT_STORE, players, vw] final views (truncation bootstraps), allocated on first use
        sh.boot_cursor = 0
        sh.seglog = SegmentLog(G, players, cfg.segment, self.ring)
        block_kind = rule_block_kind([int(gs.peace) for gs in sh.gsets], Gs, self.block,
                                     bool(getattr(self, 'rule_kinds_always', False)))
        self.mixed_rules = block_kind is not None or bool(getattr(self, 'mixed_rules', False))
        sh.lanes = Lanes(G, players, self.block, sh.rng, bool(cfg.pool_per_process), block_kind=block_kind,
                         fresh_draw=bool(getattr(self, 'pool_fresh_draw', False)))
        if self.map_tribes is not None:
            sh.lanes.mirror_only = set(getattr(self, 'league_mirror_only', ()))
            sh.lanes.set_tribes(np.arange(G), sh.playing_tribes, self.learner_tribe)
        sh.prev_score = np.zeros((G, players), np.int64)
        sh.stats = dict(steps=0, finished=0, terminated=0, truncated=0, cap_score_results=0, nonok=0, turns=0,
                        selfplay_games=0, selfplay_truncations=0, vs={}, maps={}, tribes={},
                        peace_finished=0, peace_truncated=0, peace_selfplay_games=0, peace_selfplay_truncations=0,
                        peace_set_steps=0)
        sh.key = sh.params = sh.critic = None
        sh.pool_params = {}
        # the shard's step counter (ring slot t % L) and the critic version each slot was valued with: per device, since
        # the device threads step independently between updates (lockstep: equal on every device)
        sh.t = 0
        sh.h_step_version = np.full(L, -1, np.int64)
        torch.cuda.synchronize(sh.device)
        return sh

    def _draw_seeds(self, sh, m, lo=None):
        seeds = self.torch.randint(self.map_lo, self.map_hi, (m,), generator=sh.map_gen, dtype=self.torch.int64).numpy().astype(np.int32)
        lanes = getattr(sh, 'lanes', None)
        if lo is not None and lanes is not None and lanes.mirror_only and self.mirror_rows is not None and len(self.mirror_rows):
            # stage 5 T2: lanes of blocks whose opponent plays only mirrors start their next game on a mirror row
            blocks = (lo + np.arange(m)) // lanes.block
            want = np.array([lanes.block_opp[b] in lanes.mirror_only for b in blocks.tolist()], bool)
            if want.any():
                pick = self.torch.randint(0, len(self.mirror_rows), (int(want.sum()),), generator=sh.map_gen,
                                          dtype=self.torch.int64).numpy()
                seeds[want] = (self.mirror_rows[pick] + self.map_lo).astype(np.int32)
        return seeds

    def _build_jax(self):
        import functools
        import jax
        import jax.numpy as jnp
        from polytopia_bot.gpu.actions import Masks
        from . import network, ppo, transformer_v3
        from .actions import Choice, PathMasks, leaf_shapes
        from .cpu_engine import view_features
        cfg = self.config
        # V3 (rl/transformer_v3.py): the critic branch reads the policy's public tokens, so every value call takes
        # value_params = (params, critic); V2 and the CNNs keep value_params = critic (_value_params)
        v3 = self.v3

        def critic_value(value_params, x):
            if v3:
                return transformer_v3.value(value_params[0], value_params[1], x)
            return network.trunk(value_params, x)[2]
        size, layout, Gs = self.play_size, self.layout, self.Gs
        n = size * size
        mask_width, player_at = self.mask_width, self.player_at
        fb, sb = self.fam_bytes, self.src_bytes
        encode_one = lambda v: view_features(v, layout, size)
        self.encode_one = encode_one

        @jax.jit
        def encode(views):
            return jax.vmap(encode_one)(views), views[:, player_at].astype(jnp.int32)

        def split_masks(bits):
            leaves, at = [], 0
            for shape in leaf_shapes(size):
                k = int(np.prod(shape)) if shape else 1
                leaves.append(bits[:, at:at + k].reshape((bits.shape[0],) + tuple(shape)))
                at += k
            assert at == mask_width, (at, mask_width)
            return Masks(*leaves)

        def dense_masks(packed):
            return split_masks(jnp.unpackbits(packed, axis=1)[:, :mask_width].astype(bool))

        sample_one = functools.partial(network.sample, size=size)

        def sample(params, features, masks, key):
            keys = jax.random.split(key, features.board.shape[0])
            return jax.vmap(sample_one, in_axes=(None, 0, 0, 0))(params, features, masks, keys)

        @jax.jit
        def act(params, features, packed, key):
            """The learner's decisions for every game of a set: choice [Gs, 3], row masks (packed PathMasks:
            family 17 / source n / target max(n, 128) bits), behaviour logp."""
            choice, path, logp, _ = sample(params, features, dense_masks(packed), key)
            row_masks = jnp.concatenate([jnp.packbits(path.family, axis=-1), jnp.packbits(path.source, axis=-1),
                                         jnp.packbits(path.target, axis=-1)], -1)
            return jnp.stack(choice, 1), row_masks, logp

        @jax.jit
        def act_subset(params, view, packed, idx, key):
            """A pool opponent's decisions for the lanes idx (padded with Gs: the gather clamps, the scatter drops); the
            rows' features are encoded from the gathered views (view_features is per row: the same features)."""
            f = jax.vmap(encode_one)(jnp.take(view, idx, axis=0, mode='clip'))
            masks = dense_masks(jnp.take(packed, idx, axis=0, mode='clip'))
            choice, _, _, _ = sample(params, f, masks, key)
            return jnp.stack(choice, 1)

        def act_rows(params, view, packed, idx, key):
            f = jax.vmap(encode_one)(jnp.take(view, idx, axis=0, mode='clip'))
            masks = dense_masks(jnp.take(packed, idx, axis=0, mode='clip'))
            choice, _, _, _ = sample(params, f, masks, key)
            return jnp.stack(choice, 1)

        @jax.jit
        def act_group(stacked, which, view, packed, idx2, key):
            """Stage 4 (Session L): the pool opponents of ONE parameter structure in one executable. stacked = their
            parameters stacked on a leading axis, which [Ka] = the members on the move this step (padded by
            repeating a member whose idx row is all padding), idx2 [Ka, B] = each member's lanes (padded with Gs).
            Member j's rows are exactly act_subset(member params, view, packed, idx2[j], keys[j]) with keys =
            split(key, Ka) (lax.map: one member after another inside the executable, no host round trip each).
            Returns [Ka * B, 3] in idx2's row-major order."""
            sel = jax.tree.map(lambda a: jnp.take(a, which, axis=0), stacked)
            keys = jax.random.split(key, which.shape[0])
            out = jax.lax.map(lambda a: act_rows(a[0], view, packed, a[1], a[2]), (sel, idx2, keys))
            return out.reshape(-1, 3)

        @jax.jit
        def act_group_vmap(stacked, which, view, packed, idx2, key):
            """act_group with the members batched (vmap over the stacked parameters: one forward for all of them)
            instead of one after another; same keys, same rows."""
            sel = jax.tree.map(lambda a: jnp.take(a, which, axis=0), stacked)
            keys = jax.random.split(key, which.shape[0])
            out = jax.vmap(lambda p, i, k: act_rows(p, view, packed, i, k))(sel, idx2, keys)
            return out.reshape(-1, 3)

        @jax.jit
        def merge(choice3, idx, sub):
            """The pool lanes' choices into the set's: [Gs, 3] for the ring and [3, Gs] for the engine's inputs."""
            c = choice3.at[idx].set(sub, mode='drop')
            return c, c.T

        with_value = self.value_mode != 'assembly'

        @jax.jit
        def fused(params, critic, view, opp, packed, key):
            """One set's whole JAX work for a step in one executable: encode (features, mover seat), act (the learner's
            choice, packed PathMasks rows, behaviour logp) and values_set (the critic on critic_features) are the same
            functions as the separate jits, just traced together. Returns choice [Gs, 3], choice.T [3, Gs] (the
            engine's family / source / target inputs, read in place), row masks, (logp, value) [2, Gs] ([1, Gs] in
            value_mode assembly), mover [Gs] int32, and the pool opponents' key (the second half of `key`'s split, as
            _launch splits it unfused)."""
            features = jax.vmap(encode_one)(view)
            mover = view[:, player_at].astype(jnp.int32)
            k_learner, k_pool = jax.random.split(key)
            if v3 and with_value:
                # one public encoder pass for the policy's sample and the critic branch's value
                f = jax.vmap(lambda a, b: critic_features(a, b, layout, size))(features, opp)
                keys = jax.random.split(k_learner, features.board.shape[0])
                choice, path, logp, value = jax.vmap(
                    lambda x, m, k: transformer_v3.sample_and_value(params, critic, x, m, k, size=size))(
                    f, dense_masks(packed), keys)
                row_masks = jnp.concatenate([jnp.packbits(path.family, axis=-1), jnp.packbits(path.source, axis=-1),
                                             jnp.packbits(path.target, axis=-1)], -1)
                choice3 = jnp.stack(choice, 1)
                return choice3, choice3.T, row_masks, jnp.stack([logp, value], 0), mover, k_pool
            choice, path, logp, _ = sample(params, features, dense_masks(packed), k_learner)
            row_masks = jnp.concatenate([jnp.packbits(path.family, axis=-1), jnp.packbits(path.source, axis=-1),
                                         jnp.packbits(path.target, axis=-1)], -1)
            choice3 = jnp.stack(choice, 1)
            rows = [logp]
            if with_value:
                f = jax.vmap(lambda a, b: critic_features(a, b, layout, size))(features, opp)
                value_params = (params, critic) if v3 else critic
                rows.append(jax.vmap(lambda x: critic_value(value_params, x))(f))
            return choice3, choice3.T, row_masks, jnp.stack(rows, 0), mover, k_pool

        @jax.jit
        def values_chunk(value_params, view, opp):
            """The assembly value pass: a chunk of (view, opp) rows -> the critic's value (distributed.local_values)."""
            f = jax.vmap(encode_one)(view)
            f = jax.vmap(lambda a, b: critic_features(a, b, layout, size))(f, opp)
            return jax.vmap(lambda x: critic_value(value_params, x))(f)

        @jax.jit
        def values_set(value_params, features, opp):
            """The in-rollout value of a set's current state from its already-encoded mover features and the opp view:
            the same critic_features -> critic value as values_chunk."""
            f = jax.vmap(lambda a, b: critic_features(a, b, layout, size))(features, opp)
            return jax.vmap(lambda x: critic_value(value_params, x))(f)

        def unpack_bits(a, width):
            return jnp.unpackbits(a, axis=-1)[..., :width].astype(bool)

        def to_batch(mb):
            features = jax.vmap(encode_one)(mb['view'])
            masks = PathMasks(unpack_bits(mb['masks'][:, :fb], 17), unpack_bits(mb['masks'][:, fb:fb + sb], n),
                              unpack_bits(mb['masks'][:, fb + sb:], max(n, 128)))
            choice = Choice(mb['choice'][:, 0], mb['choice'][:, 1], mb['choice'][:, 2])
            return dict(features=features, masks=masks, choice=choice, logp=mb['logp'], advantage=mb['advantage'],
                        returns=mb['returns'],
                        full_features=jax.vmap(lambda f, o: critic_features(f, o, layout, size))(features, mb['opp_view']))

        def loss_v3(trainable, mb, entropy_coef):
            """The same objective as `loss` (PPO surrogate + entropy, + .5 * the critic's .5 MSE) with ONE public
            encoder pass shared by the policy heads and the critic branch."""
            params, critic = trainable
            batch = to_batch(mb)
            logp, v, path_entropy = jax.vmap(transformer_v3.evaluate, in_axes=(None, None, 0, 0, 0))(
                params, critic, batch['full_features'], batch['choice'], batch['masks'])
            total, metrics = ppo.policy_objective(logp, path_entropy, batch, clip=cfg.clip, entropy_coef=entropy_coef)
            critic_loss = .5 * jnp.mean((v - batch['returns']) ** 2)
            total = total + .5 * critic_loss
            return total, dict(metrics, value_loss=critic_loss, value_mean=v.mean(), value_std=v.std())

        def loss(trainable, mb, entropy_coef):
            if v3:
                return loss_v3(trainable, mb, entropy_coef)
            params, critic = trainable
            batch = to_batch(mb)
            total, metrics = ppo.loss(params, batch, clip=cfg.clip, entropy_coef=entropy_coef, value_coef=0.)
            v = jax.vmap(lambda x: network.trunk(critic, x)[2])(batch['full_features'])
            critic_loss = .5 * jnp.mean((v - batch['returns']) ** 2)
            total = total + .5 * critic_loss
            return total, dict(metrics, value_loss=critic_loss, value_mean=v.mean(), value_std=v.std())

        accum = int(cfg.grad_accum)
        mb_rows = cfg.minibatch
        micro = mb_rows // accum

        @jax.jit
        def train_step(trainable, opt, dev, idx, learning_rate, entropy_coef, policy_weight):
            """One minibatch (rows idx of the device batch): features re-encoded from the stored views, the loss of
            D: the gradient (mean over grad_accum micro-batches), one ppo.adam_step over (params, critic)."""
            if accum == 1:
                mb = jax.tree.map(lambda a: a[idx], dev)
                (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(trainable, mb, entropy_coef)
            else:
                def part(acc, j):
                    sub = jax.lax.dynamic_slice(idx, (j * micro,), (micro,))
                    mb = jax.tree.map(lambda a: a[sub], dev)
                    (_, m), g = jax.value_and_grad(loss, has_aux=True)(trainable, mb, entropy_coef)
                    return jax.tree.map(jnp.add, acc, g), m
                zero = jax.tree.map(jnp.zeros_like, trainable)
                grads, ms = jax.lax.scan(part, zero, jnp.arange(accum))
                grads = jax.tree.map(lambda g: g / accum, grads)
                metrics = jax.tree.map(jnp.mean, ms)
            pgrads, cgrads = grads
            pgrads = jax.tree.map(lambda x: x * policy_weight, pgrads)   # critic warm-up: the policy stays put
            trainable, opt, norm = ppo.adam_step(trainable, (pgrads, cgrads), opt, learning_rate=learning_rate)
            return trainable, opt, dict(metrics, grad_norm=norm)

        self._encode, self._act, self._act_subset, self._merge, self._fused = encode, act, act_subset, merge, fused
        self._act_group = act_group
        self._act_group_variants = dict(map=act_group, vmap=act_group_vmap)
        if os.environ.get('V3L_POOL_GROUP', 'map') == 'vmap':
            self._act_group = act_group_vmap
        self._values_chunk, self._values_set, self._train_step = values_chunk, values_set, train_step

        if self.n_global > 1:
            # data parallel: the production learner's step (D:) as a shard_map over a 1-D mesh of the local
            # devices: replicated (params, critic) and Adam state, each device's rows sharded P('data'), a per-device
            # minibatch of minibatch // N rows from that device's own permutation, gradients (and metrics) pmean'd
            # over 'data' before the ONE ppo.adam_step every device applies identically
            from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
            N = self.n_global
            mesh = Mesh(np.array([sh.jax_device for sh in self.shards]) if self.n_procs == 1 else
                        np.array(sorted(jax.devices(), key=lambda dv: (dv.process_index, dv.id))), ('data',))
            self.mesh = mesh
            self.data_sharding = NamedSharding(mesh, P('data'))
            self.replicated = NamedSharding(mesh, P())
            per = cfg.batch // N
            local_mb = cfg.minibatch // N
            if self.mixed:      # this process's share; the same number of minibatch steps on every process
                per = self.process_rows
                local_mb = per // (cfg.batch // cfg.minibatch)
            micro_dp = local_mb // accum
            self.per_device_rows, self.local_minibatch = per, local_mb

            def local_step(trainable, opt, dev, perm, i, learning_rate, entropy_coef, policy_weight):
                idx = jax.lax.dynamic_slice(perm, (i * local_mb,), (local_mb,))
                if accum == 1:
                    mb = jax.tree.map(lambda a: a[idx], dev)
                    (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(trainable, mb, entropy_coef)
                else:
                    def part(acc, j):
                        sub = jax.lax.dynamic_slice(idx, (j * micro_dp,), (micro_dp,))
                        mb = jax.tree.map(lambda a: a[sub], dev)
                        (_, m), g = jax.value_and_grad(loss, has_aux=True)(trainable, mb, entropy_coef)
                        return jax.tree.map(jnp.add, acc, g), m
                    zero = jax.tree.map(jnp.zeros_like, trainable)
                    grads, ms = jax.lax.scan(part, zero, jnp.arange(accum))
                    grads = jax.tree.map(lambda g: g / accum, grads)
                    metrics = jax.tree.map(jnp.mean, ms)
                grads = jax.lax.pmean(grads, 'data')
                metrics = jax.lax.pmean(metrics, 'data')
                pgrads, cgrads = grads
                pgrads = jax.tree.map(lambda x: x * policy_weight, pgrads)   # critic warm-up: the policy stays put
                trainable, opt, norm = ppo.adam_step(trainable, (pgrads, cgrads), opt, learning_rate=learning_rate)
                return trainable, opt, dict(metrics, grad_norm=norm)

            self._train_step_dp = jax.jit(jax.shard_map(
                local_step, mesh=mesh, in_specs=(P(), P(), P('data'), P('data'), P(), P(), P(), P()),
                out_specs=(P(), P(), P()), check_vma=False))

            # mixed sizes (_mixed): the rows' shapes differ between processes, so the minibatch's gradient is computed
            # by a per-process jit on the local device and only the gradients (the size-independent parameter tree)
            # cross the devices: [1, ...] per device -> pmean -> the same adam_step as local_step
            def make_local_grad(accum):
                # Session A (T3 speed): a factory so the profiler can time other grad_accum splits of the SAME rows on
                # the same data; production is make_local_grad(cfg.grad_accum), the identical traced function
                micro_dp = local_mb // accum

                def local_grad(trainable, dev, perm, i, entropy_coef):
                    idx = jax.lax.dynamic_slice(perm, (i * local_mb,), (local_mb,))
                    if accum == 1:
                        mb = jax.tree.map(lambda a: a[idx], dev)
                        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(trainable, mb, entropy_coef)
                    else:
                        def part(acc, j):
                            sub = jax.lax.dynamic_slice(idx, (j * micro_dp,), (micro_dp,))
                            mb = jax.tree.map(lambda a: a[sub], dev)
                            (_, m), g = jax.value_and_grad(loss, has_aux=True)(trainable, mb, entropy_coef)
                            return jax.tree.map(jnp.add, acc, g), m
                        zero = jax.tree.map(jnp.zeros_like, trainable)
                        grads, ms = jax.lax.scan(part, zero, jnp.arange(accum))
                        grads = jax.tree.map(lambda g: g / accum, grads)
                        metrics = jax.tree.map(jnp.mean, ms)
                    w = jnp.float32(local_mb)          # this process's rows in the step: the mean is weighted by rows
                    return (jax.tree.map(lambda g: (g * w)[None], grads), jax.tree.map(lambda m: jnp.reshape(m * w, (1,)), metrics),
                            jnp.reshape(w, (1,)))
                return local_grad
            local_grad = make_local_grad(accum)

            def apply_mean(trainable, opt, grads, metrics, weight, learning_rate, policy_weight):
                total = jax.lax.psum(weight[0], 'data')
                grads = jax.tree.map(lambda g: jax.lax.psum(g[0], 'data') / total, grads)
                metrics = jax.tree.map(lambda m: jax.lax.psum(m[0], 'data') / total, metrics)
                pgrads, cgrads = grads
                pgrads = jax.tree.map(lambda x: x * policy_weight, pgrads)
                trainable, opt, norm = ppo.adam_step(trainable, (pgrads, cgrads), opt, learning_rate=learning_rate)
                return trainable, opt, dict(metrics, grad_norm=norm)

            self._local_grad = jax.jit(local_grad)
            self._apply_mean = jax.jit(jax.shard_map(
                apply_mean, mesh=mesh, in_specs=(P(), P(), P('data'), P('data'), P('data'), P(), P()),
                out_specs=(P(), P(), P()), check_vma=False))

            # Session A (2026-11-09, 32-GPU stage 5): the per-leaf path above costs ~65 ms of fixed overhead per
            # minibatch step (one global array and one cross-node pmean per gradient/metric leaf, ~140 of them), so the
            # train step did not shrink with more GPUs. Flat path: local_grad's (grads, metrics) concatenated into ONE
            # float32 vector on the device -> one global array -> one psum -> unflattened inside the jitted step ->
            # the same adam_step. Same math as apply_mean (row-weighted: the vector carries grads x rows, metrics x rows
            # and the rows; the sums are divided by the summed rows). V3L_MIXED_FLAT=0 restores the per-leaf path.
            self.mixed_flat = os.environ.get('V3L_MIXED_FLAT', '1') == '1'

            def make_local_grad_flat(fn):
                def local_grad_flat(trainable, dev, perm, i, entropy_coef):
                    return jnp.concatenate([x.reshape(-1).astype(jnp.float32)
                                            for x in jax.tree.leaves(fn(trainable, dev, perm, i, entropy_coef))])[None]
                return local_grad_flat
            self._local_grad_flat = jax.jit(make_local_grad_flat(local_grad))
            # V3L_TRAIN_BENCH (profiling only): the flat local gradient with another grad_accum split, built on demand
            self._bench_flat = lambda a: jax.jit(make_local_grad_flat(make_local_grad(a)))

            def make_apply_flat(struct):
                # struct: jax.eval_shape of local_grad, leaves [1, ...]; the flat vector is their concatenation
                leaves, treedef = jax.tree.flatten(struct)
                parts = [(int(np.prod(l.shape)), tuple(l.shape[1:]), l.dtype) for l in leaves]

                def apply_flat(trainable, opt, flat, learning_rate, policy_weight):
                    v = jax.lax.psum(flat[0], 'data')
                    out, o = [], 0
                    for n, shape, dtype in parts:
                        out.append(v[o:o + n].reshape(shape))
                        o += n
                    grads, metrics, rows = jax.tree.unflatten(treedef, out)
                    (pgrads, cgrads), metrics = jax.tree.map(lambda x: x / rows, (grads, metrics))
                    (pgrads, cgrads), metrics = jax.tree.map(lambda x, l: x.astype(l.dtype), ((pgrads, cgrads), metrics),
                                                             (struct[0], struct[1]))
                    pgrads = jax.tree.map(lambda x: x * policy_weight, pgrads)
                    trainable, opt, norm = ppo.adam_step(trainable, (pgrads, cgrads), opt, learning_rate=learning_rate)
                    return trainable, opt, dict(metrics, grad_norm=norm)
                return jax.jit(jax.shard_map(apply_flat, mesh=mesh, in_specs=(P(), P(), P('data'), P(), P()),
                                             out_specs=(P(), P(), P()), check_vma=False))
            self._make_apply_flat, self._apply_flat = make_apply_flat, None
            local_perm = jax.jit(lambda kd, g: jax.random.permutation(
                jax.random.fold_in(jax.random.wrap_key_data(kd), g), per).astype(jnp.int32))
            self._local_perm = lambda key, sh: local_perm(jax.device_put(np.asarray(jax.random.key_data(key)), sh.jax_device),
                                                          jax.device_put(np.int32(sh.g), sh.jax_device))
            # each device's permutation of its own `per` rows (production: permutation(fold_in(key, axis_index), local)),
            # laid out [N * per] and sharded so device d holds its own
            perms = jax.jit(lambda kd: jax.vmap(lambda d: jax.random.permutation(
                jax.random.fold_in(jax.random.wrap_key_data(kd), d), per))(jnp.arange(N)).reshape(-1).astype(jnp.int32),
                out_shardings=self.data_sharding)
            # the key enters as host key data (identical on every process), so the jit spanning several processes gets
            # a replicated host value instead of one process's single-device array; same permutation values
            self._perms = lambda key: perms(np.asarray(jax.random.key_data(key)))

            def local_norm(adv, floor):
                n = jax.lax.psum(jnp.float32(adv.shape[0]), 'data')
                mean = jax.lax.psum(adv.sum(), 'data') / n
                var = jax.lax.psum(((adv - mean) ** 2).sum(), 'data') / n
                std = jnp.sqrt(var)
                return (adv - mean) / jnp.maximum(std, floor), std
            self._normalize_dp = jax.jit(jax.shard_map(local_norm, mesh=mesh, in_specs=(P('data'), P()),
                                                       out_specs=(P('data'), P()), check_vma=False))
            # mixed mode: a few host floats summed over every process (the batch statistics)
            self._psum_small = jax.jit(jax.shard_map(lambda x: jax.lax.psum(x[0], 'data'), mesh=mesh,
                                                     in_specs=P('data'), out_specs=P(), check_vma=False))
        self.params = jax.device_put(self.params)
        self.critic = jax.device_put(self.critic)
        self.opt = ppo.adam_init((self.params, self.critic))
        self.n_params = int(sum(x.size for x in jax.tree.leaves(self.params)))
        self.n_critic = int(sum(x.size for x in jax.tree.leaves(self.critic)))
        self.key = jax.random.key(int(self.rng.integers(1 << 31)))
        # the rollout keys: device 0 continues self.key's stream (one device: exactly the single-GPU trainer), device
        # d > 0 its own key from its own rng, committed to its GPU
        for sh in self.shards:
            sh.key = self.key if sh.g == 0 and self.n_global == 1 else jax.device_put(
                self.key if sh.g == 0 else jax.random.key(int(sh.rng.integers(1 << 31))), sh.jax_device)

    def _resume(self):
        """Learner._resume: weights, Adam state, critic, update count, the log and the pool from the run folder."""
        jax = self.jax
        from . import ppo
        from .runner import load_checkpoint
        cfg = self.config
        template = _template_params(**self.network)
        latest = self.ckpt_root / 'latest.npz'          # rank 0's folder (= self.root on rank 0)
        meta = json.loads(latest.with_suffix('.json').read_text())
        version = int(meta['stage'].rsplit('-', 1)[1])
        params = load_checkpoint(latest, template)
        critic = load_checkpoint(self.ckpt_root / 'latest-critic.npz', _critic_template(**self.network))
        opt_leaves, opt_structure = jax.tree.flatten(ppo.adam_init((params, critic)))
        with np.load(latest, allow_pickle=False) as archive:
            values = [archive[f'o{i}'] for i in range(len(opt_leaves))]
        if any(x.shape != y.shape for x, y in zip(opt_leaves, values, strict=True)):
            raise ValueError('optimizer state in latest.npz does not match the network')
        self.params, self.critic = jax.device_put(params), jax.device_put(critic)
        self.opt = resume_optimizer(opt_structure.unflatten([jax.numpy.asarray(v) for v in values]), version,
                                    cfg.critic_warmup, lambda: ppo.adam_init((self.params, self.critic)))
        history = [row for row in training_log.read(self.ckpt_root) if row['update'] <= version]
        if history and history[-1]['update'] != version:
            raise ValueError(f'training log ends at update {history[-1]["update"]}, latest.npz is update {version}')
        training_log.rewrite(self.root / training_log.JSONL, history)
        self.resume_history, self.resume_elapsed = history, float(history[-1]['elapsed']) if history else 0.
        self.version = version
        self.historical = []
        if cfg.historical_probability > 0:
            for u in range(cfg.historical_every, version + 1, cfg.historical_every):
                snap = self.ckpt_root / 'snapshots' / f'update-{u}.npz'
                if snap.exists():
                    self.historical.append((f'hist-u{u}', self._rounded(jax.device_put(load_checkpoint(snap, template)))))
            self.historical = self.historical[-cfg.historical_keep:]
        self.log(json.dumps(dict(stage='learner_resumed', update=version, elapsed=self.resume_elapsed,
                                 pool=[n for n, _ in self.historical])))

    # ---- pool
    def pool_spec(self):
        cfg = self.config
        spec = [(name, p) for name, p in self.frozen_probability.items()]
        if self.historical:
            each = cfg.historical_probability / len(self.historical)
            spec += [(name, each) for name, _ in self.historical]
        return spec

    def _set_pool(self):
        self.pool_params = dict(self.frozen, **dict(self.historical))
        for sh in self.shards:
            sh.pool_params = {name: self._local(p, sh) for name, p in self.pool_params.items()}
            sh.lanes.set_pool(self.pool_spec(), self.pool_params.keys())
            if getattr(self, 'pool_batched', False):
                self._group_pool(sh)

    def _group_pool(self, sh):
        """Stage 4 (Session L): the shard's pool parameters grouped by structure (tree + leaf shapes and dtypes), each
        group stacked once on the shard's device; sh.pool_group_of[name] = (group, member)."""
        jax = self.jax
        groups = {}
        for name, prm in sh.pool_params.items():
            groups.setdefault(self._structure(prm), []).append(name)
        # Session A (T1 stalls): the historical members' group is stacked at a fixed capacity (its frozen members +
        # historical_keep; padding repeats member 0, which `which` never selects, so every row is unchanged) so a
        # growing history never changes the executables' shapes; V3L_POOL_CAPACITY=0 restores the exact stack.
        pad = os.environ.get('V3L_POOL_CAPACITY', '1') == '1'
        cfg = self.config
        hist_keep = cfg.historical_keep if cfg.historical_probability > 0 else 0
        hist_key = self._historical_structure() if pad and hist_keep else None
        sh.pool_stacks, sh.pool_group_of = [], {}
        for gi, (key, names) in enumerate(groups.items()):
            cap = len(names)
            if pad and key == hist_key:
                cap = max(cap, sum(not n.startswith('hist-') for n in names) + hist_keep)
            members = [sh.pool_params[n] for n in names]
            members += [members[0]] * (cap - len(names))
            stacked = jax.tree.map(lambda *xs: jax.numpy.stack(xs), *members)
            sh.pool_stacks.append((stacked, len(names)))
            for j, n in enumerate(names):
                sh.pool_group_of[n] = (gi, j)
        sh.pool_warm_due = True

    def _structure(self, prm):
        """A pool member's parameter structure: tree + leaf shapes and dtypes (one act_group executable family)."""
        leaves, tree = self.jax.tree.flatten(prm)
        return (str(tree), tuple((tuple(x.shape), str(x.dtype)) for x in leaves))

    def _historical_structure(self):
        """The structure of a future historical member (_rounded(self.params)), without computing it."""
        if self.historical:
            return self._structure(self.historical[-1][1])
        return self._structure(self.jax.eval_shape(self._rounded, self.params))

    def _warm_pool(self, sh, gs, choice3, key):
        """Session A (T1 v3l-s5m stalls): compile every act_group / merge / reshape shape of the shard's current stacks
        once, before the first pool decision that could need it (V3L_POOL_WARM=1, the default). The calls use all-padding
        lanes (idx = Gs: the gather clamps, merge's scatter drops) and `key` = the caller's pool key, read but never
        split or returned, and their outputs are discarded, so no decision, RNG stream or ring row changes. Repeats only
        for a stack signature not warmed yet.
        Session A 2026-11-09 (T2): the key MUST be the real pool key (typed jax.random.key, same
        device). A legacy PRNGKey(0) uint32[2] is a different argument type, so every warm executable missed and each
        real shape compiled again on first use: T2's first update compiled ~140 executables (~750 s per rank) and ~8 late
        compiles per rank (6-13 s each) stalled all 32 ranks."""
        jax = self.jax
        sh.pool_warm_due = False
        sig = tuple(tuple(tuple(x.shape) for x in jax.tree.leaves(stacked)) for stacked, _ in sh.pool_stacks)
        done = sh.__dict__.setdefault('pool_warmed', set())
        if sig in done:
            return
        t0 = time.perf_counter()
        put = (lambda a: jax.numpy.asarray(a)) if self.n_devices == 1 else (lambda a: jax.device_put(a, sh.jax_device))
        calls, out = 0, None
        for stacked, _ in sh.pool_stacks:
            capacity = int(jax.tree.leaves(stacked)[0].shape[0])
            for ka, B in pool_call_shapes(capacity, self.Gs):
                which_j, idx_j = put(np.zeros(ka, np.int32)), put(np.full((ka, B), self.Gs, np.int32))
                sub = self._act_group(stacked, which_j, gs.view_j, gs.masks_j, idx_j, key)
                out = self._merge(choice3, idx_j.reshape(-1), sub)
                calls += 1
        if out is not None:
            jax.block_until_ready(out)
        done.add(sig)
        self.log(json.dumps(dict(stage='pool_warm', shard=sh.index, groups=len(sh.pool_stacks), calls=calls,
                                 seconds=round(time.perf_counter() - t0, 2))))

    # ---- placement (data parallel)
    def _replicate(self, tree):
        """One copy per device of the mesh (NamedSharding P()); one device: unchanged."""
        if self.n_global == 1:
            return tree
        if self.n_procs == 1:
            return self.jax.device_put(tree, self.replicated)
        # several processes: every process holds identical host values (same init seed / same checkpoint); build the
        # global replicated arrays from them
        return self.jax.tree.map(lambda a: self.jax.make_array_from_callback(
            np.shape(a), self.replicated, lambda idx, h=np.asarray(self._host(a)): h[idx]), tree)

    def _host(self, a):
        """numpy copy of a (possibly multi-process replicated) array: its local replica."""
        if isinstance(a, self.jax.Array) and not a.is_fully_addressable:
            return np.asarray(a.addressable_data(0))
        return np.asarray(a)

    def _local(self, tree, sh):
        """The shard's own copy of a replicated tree (no copy: the replica on its GPU); one device: the tree itself."""
        if self.n_global == 1:
            return tree
        return self.jax.tree.map(lambda a: _shard_on(a, sh.jax_device), tree)

    def _place(self):
        """After the build (and a resume): replicate (params, critic, Adam state) and the pool over the devices, then
        hand each shard its local copies for the rollout."""
        if self.n_global > 1:
            self.params, self.critic, self.opt = self._replicate((self.params, self.critic, self.opt))
            self.frozen = {name: self._replicate(p) for name, p in self.frozen.items()}
            self.historical = [(name, self._replicate(p)) for name, p in self.historical]
        self._publish()

    def _publish(self):
        """The rollout's weights on every device: after the placement and after every update (the update's outputs are
        replicated over the mesh; each shard reads its own replica)."""
        for sh in self.shards:
            sh.params, sh.critic = self._local(self.params, sh), self._local(self.critic, sh)

    def _value_params(self, sh):
        """What the shard's value calls read: the critic, or for V3 (policy, critic-branch) since its critic reads
        the public encoder's tokens."""
        return (sh.params, sh.critic) if self.v3 else sh.critic

    # ---- rollout
    def _to_jax(self, t):
        import jax.dlpack
        return jax.dlpack.from_dlpack(t)

    def _to_torch(self, a):
        import torch.utils.dlpack
        self.jax.block_until_ready(a)
        return torch.utils.dlpack.from_dlpack(a)

    def _timed(self, name, fn, sh=None):
        t0 = time.perf_counter()
        res = fn()
        self.jax.block_until_ready(res)
        self._add(name, (time.perf_counter() - t0) * 1000., sh)
        return res

    def _add(self, name, ms, sh=None):
        """Timing: the run-wide clock (per call, averaged over sets and devices) and, with several devices, the shard's.
        Device threads write only their shard's clock (no shared dict across threads); _timing merges them."""
        if not (self.threaded and sh is not None):
            self.clock[name] = self.clock.get(name, 0.) + ms
            self.calls[name] = self.calls.get(name, 0) + 1
        if sh is not None and self.n_devices > 1:
            sh.clock[name] = sh.clock.get(name, 0.) + ms
            sh.calls[name] = sh.calls.get(name, 0) + 1

    def _lap(self, name, since, sh=None):
        now = time.perf_counter()
        self._add(name, (now - since) * 1000., sh)
        return now

    def _finish(self, sh, gs):
        """Complete the set's queued tick and stage its outcomes for the host (one non-blocking copy). Fused: the JAX
        work is dispatched by _launch / _dispatch. Unfused: encode the new state (and value it with the critic in
        value_mode rollout), book the tick's outcomes while that runs, then read the mover seats."""
        if gs.in_flight:
            t0 = time.perf_counter()
            gs.env.sync()
            self._lap('engine_wait', t0, sh)
            gs.in_flight = False
        if gs.pending is not None:
            self._stage_outcomes(gs)
        if self.fused:
            return
        t0 = time.perf_counter()
        # dispatched, not awaited: the outcomes' host bookkeeping below overlaps them (the engine's outputs are complete
        # after sync(); the outcomes read the staged copy, not the views the encode reads)
        gs.features, gs.player = self._encode(gs.view_j)
        if self.value_mode != 'assembly':
            gs.value = self._values_set(self._value_params(sh), gs.features, gs.opp_j)
        dispatched = time.perf_counter()
        if gs.pending is not None:
            self._outcomes(sh, gs, gs.pending)
            gs.pending = None
        t1 = self._lap('outcomes', dispatched, sh)
        gs.mover = np.asarray(gs.player).astype(np.int32)        # waits for the encode; the value call runs on
        now = time.perf_counter()
        self._add('encode', ((dispatched - t0) + (now - t1)) * 1000., sh)

    def _stage_outcomes(self, gs):
        """The completed tick's scores / rewards / status / flags into the set's staging buffer (four device copies on
        torch's stream) and from there to pinned host memory in ONE non-blocking transfer, marked by an event.
        _outcomes waits on the event; by then the set's JAX work has been dispatched and the copy is long done. The
        copy is read in the same step, before the step's rows are recorded: SegmentLog.finish must close a finished
        game's segments before the restarted game's first row is recorded, and Lanes.game_over assigns the new seats
        that decide whose move that row is, so the outcomes cannot be consumed a step later. (Several devices: called
        inside the shard's device context, so the copies and the event are on its GPU's current stream.)"""
        out = gs.out
        scores, rewards, status, flags = gs.stage_in
        scores.copy_(out['scores'])
        rewards.copy_(out['rewards'])
        status.copy_(out['status'])
        self.torch.cat((out['done'], out['terminated'], out['truncated']), out=flags)
        gs.stage_host.copy_(gs.stage_dev, non_blocking=True)
        gs.stage_event.record()

    def _outcomes(self, sh, gs, t):
        """The tick of step t: finished games (rewards, cap score from the previous tick's scores, segment finishes,
        new opponents and seats), statistics, the previous score for the next tick. Reads the staged host copy."""
        cfg = self.config
        gs.stage_event.synchronize()
        host = gs.stage_out
        done, term_all, trunc_all = host['flags'].astype(bool)
        status = host['status']
        scores = host['scores'].astype(np.int64)
        st = sh.stats
        st['steps'] += self.Gs
        st['nonok'] += int((status != 0).sum())
        if done.any():
            d = np.flatnonzero(done)
            g = gs.games[d]
            term, trunc = term_all[d], trunc_all[d]
            rewards = host['rewards'][d].astype(np.float32)
            ended = np.zeros(len(d), bool)
            cap = float(cfg.cap_score_reward or 0.)
            prev = sh.prev_score[g]
            if cap > 0:
                lead = np.sign(prev[:, 0] - prev[:, 1]).astype(np.float32)
                capped = trunc
                rewards[capped] = (np.stack([lead, -lead], 1) * cap)[capped]
                ended[capped] = True
                st['cap_score_results'] += int(capped.sum())
            else:
                capped = np.zeros(len(d), bool)
            rest = ~capped
            for i in np.flatnonzero(rest):
                rewards[i], ended[i] = episode_outcome(rewards[i], term[i], trunc[i], cfg.truncation_reward)
            store_idx = np.full(len(d), -1, np.int64)
            boot = np.flatnonzero(~ended)
            if len(boot):
                store_idx[boot] = self._store_final_views(sh, gs, d[boot])
            pool = sh.lanes.lane_pool[g]
            seats = sh.lanes.lane_seat[g]
            seats_mask = np.ones((len(d), cfg.players), bool)
            seats_mask[pool] = np.arange(cfg.players)[None, :] == (seats[pool] - 1)[:, None]
            sh.seglog.finish(g, rewards, ended, seats_mask, store_idx)
            # statistics (actor stats of the production run)
            st['finished'] += len(d)
            st['terminated'] += int(term.sum())
            st['truncated'] += int(trunc.sum())
            st['selfplay_games'] += int((~pool).sum())
            st['selfplay_truncations'] += int((~pool & trunc).sum())
            tag = ''
            if gs.peace and self.mixed_rules:
                # stage 4: peace-legal sets' outcomes separately (collusion monitors); vs records '<name>|peace'
                tag = '|peace'
                st['peace_finished'] += len(d)
                st['peace_truncated'] += int(trunc.sum())
                st['peace_selfplay_games'] += int((~pool).sum())
                st['peace_selfplay_truncations'] += int((~pool & trunc).sum())
            for i in np.flatnonzero(pool):
                name = sh.lanes.lane_name[g[i]] + tag
                rec = st['vs'].setdefault(name, [0, 0, 0])
                # the learner seat's outcome after episode_outcome: under cap_loss_both (stage 5) a
                # capped game is -1 = a LOSS here too (was recorded as a draw); a score-rule run keeps its 0 draw
                r = float(rewards[i, seats[i] - 1])
                rec[0 if r > 0 else 1 if r < 0 else 2] += 1
            for i in range(len(d)):
                kind = self.map_types[int(sh.playing_kind[g[i]])]
                rec = st['maps'].setdefault(kind, [0, 0])
                rec[0] += 1
                rec[1] += int(trunc[i])
            if self.map_tribes is not None:
                # per opponent tribe (the non-Bardur seat; the mirror under the learner's own tribe): [games, capped,
                # pool games, pool wins of the learner seat]
                for i in range(len(d)):
                    tr = sh.playing_tribes[g[i]]
                    other = int(tr[1] if tr[0] == self.learner_tribe else tr[0])
                    rec = st['tribes'].setdefault(str(other), [0, 0, 0, 0])
                    rec[0] += 1
                    rec[1] += int(trunc[i])
                    if pool[i]:
                        rec[2] += 1
                        rec[3] += int(rewards[i, seats[i] - 1] > 0)
            # the finished lanes restarted inside this tick on the seeds and layouts it was given
            sh.playing_kind[g] = self.map_kind[gs.seeds_host[d] - self.map_lo]
            if self.map_tribes is not None:
                sh.playing_tribes[g] = self.map_tribes[gs.seeds_host[d] - self.map_lo]
                sh.lanes.set_tribes(g, sh.playing_tribes[g], self.learner_tribe)
            sh.lanes.game_over(g)
        sh.prev_score[gs.lo:gs.hi] = scores

    def _store_final_views(self, sh, gs, d):
        """Final views [k, players, vw] of truncated games that bootstrap (cap_score_reward 0): rows of the store."""
        torch = self.torch
        if sh.boot_store is None:
            sh.boot_store = torch.zeros((BOOT_STORE, self.config.players, self.vw), dtype=torch.int16, device=sh.device)
        k = len(d)
        if k > BOOT_STORE:
            raise RuntimeError('too many truncation bootstraps in one step for the boundary store')
        rows = (sh.boot_cursor + np.arange(k)) % BOOT_STORE
        src = gs.out['final_view'][torch.from_numpy(d).to(sh.device)]
        sh.boot_store[torch.from_numpy(rows).to(sh.device)] = src
        sh.boot_cursor += k
        return sh.boot_cursor - k + np.arange(k)   # absolute store positions (row = position % BOOT_STORE)

    def _dispatch(self, sh, gs, key):
        """The set's fused call, dispatched and not awaited (its GPU runs it while the host moves on): returns
        (result, dispatch milliseconds)."""
        t0 = time.perf_counter()
        res = self._fused(sh.params, sh.critic, gs.view_j, gs.opp_j, gs.masks_j, key)
        return res, (time.perf_counter() - t0) * 1000.

    def _launch(self, sh, gs, t, key, dispatched=None):
        """Decisions of every game of the set from its current state (learner, and the pool opponents on their
        lanes), the learner rows into the ring, the new maps, and the tick. Fused: one dispatch of `fused` (the
        staged outcomes are booked while it runs; `dispatched` = _dispatch's result when the caller dispatched it
        already, the data-parallel step does so for every device first) and one host wait; the engine reads the choice
        from the call's [3, Gs] output in place."""
        import jax
        torch = self.torch
        lo, hi, Gs = gs.lo, gs.hi, self.Gs
        lanes = sh.lanes
        choiceT = None
        if self.fused:
            res, dispatch_ms = dispatched if dispatched is not None else self._dispatch(sh, gs, key)
            t1 = time.perf_counter()
            if gs.pending is not None:
                self._outcomes(sh, gs, gs.pending)
                gs.pending = None
            t1 = self._lap('outcomes', t1, sh)
            jax.block_until_ready(res)                                  # the one host wait of the set's JAX work
            choice3, choiceT, row_masks, fv, mover_j, k_pool = res
            mover = np.asarray(mover_j)
            now = time.perf_counter()
            self._add('fused', dispatch_ms + (now - t1) * 1000., sh)
        else:
            mover = gs.mover
            k_learner, k_pool = jax.random.split(key)
            choice3, row_masks, logp = self._timed('sample', lambda: self._act(sh.params, gs.features, gs.masks_j, k_learner), sh)
        # pool opponents: lanes where it is a pool opponent's move, grouped by parameter set (no call while the pool is empty)
        opp_turn = lanes.lane_pool[lo:hi] & (mover != lanes.lane_seat[lo:hi])
        if opp_turn.any() and getattr(self, 'pool_batched', False):
            t0 = time.perf_counter()
            choice3, choiceT, k_pool = self._pool_grouped(sh, gs, opp_turn, choice3, choiceT, k_pool)
            jax.block_until_ready(choice3)
            self._lap('pool', t0, sh)
        elif opp_turn.any():
            t0 = time.perf_counter()
            codes = lanes.lane_code[lo:hi]
            for code in np.unique(codes[opp_turn]).tolist():
                idx = np.flatnonzero(opp_turn & (codes == code))
                name = lanes.code_name[code]
                bsz = _bucket(len(idx), Gs)
                pad = np.full(bsz, Gs, np.int32)
                pad[:len(idx)] = idx
                k_pool, k_sub = jax.random.split(k_pool)
                idx_j = jax.numpy.asarray(pad) if self.n_devices == 1 else jax.device_put(pad, sh.jax_device)
                sub = self._act_subset(sh.pool_params[name], gs.view_j, gs.masks_j, idx_j, k_sub)
                choice3, choiceT = self._merge(choice3, idx_j, sub)
            jax.block_until_ready(choice3)
            self._lap('pool', t0, sh)
        t0 = time.perf_counter()
        # rows: force-close stale open segments against this state (their boundary = this state's ring row, oriented
        # to the seat at assembly), then record this step's learner rows
        sh.seglog.force_close(t, gs.games, mover - 1)
        rec = np.flatnonzero(~opp_turn)
        sh.seglog.record(t, gs.games[rec], (mover[rec] - 1).astype(np.int32))
        # the ring slot: five slice copies (view, opp view, masks, choice, (logp, value) block)
        slot = t % self.L
        base = slot * self.G + lo
        sh.ring_view[base:base + Gs].copy_(gs.out['view'])
        sh.ring_opp[base:base + Gs].copy_(gs.out['opp_view'][:, 0])
        sh.ring_masks[base:base + Gs].copy_(self._to_torch(row_masks))
        choice_t = self._to_torch(choice3)
        sh.ring_choice[base:base + Gs].copy_(choice_t)           # end_turn = choice[:, 0] == END_FAMILY at assembly
        if gs.peace and self.mixed_rules:
            # stage-4 collusion monitors (device-side, read once per update): game-steps at peace and the learner's
            # diplomatic decisions in this peace-legal set
            v = gs.out['view'][:, self.peace_at:self.peace_at + self.peace_w]
            gs.peace_steps += (v != 0).any(1).sum()
            gs.late_steps += (gs.out['view'][:, self.turn_at] >= self.late_round).sum()
            learner = torch.from_numpy(~opp_turn).to(sh.device, non_blocking=True)
            gs.diplo += ((choice_t[:, 0] == DIPLOMATIC_FAMILY) & learner).sum()
            sh.stats['peace_set_steps'] += Gs
        if self.fused:
            fv_t = self._to_torch(fv)
            sh.ring_fv[:fv_t.shape[0], base:base + Gs].copy_(fv_t)
        else:
            sh.ring_logp[base:base + Gs].copy_(self._to_torch(logp))
            if gs.value is not None:
                sh.ring_value[base:base + Gs].copy_(self._to_torch(gs.value))
        # the engine's family / source / target inputs: the [3, Gs] JAX output read in place (held until the tick is
        # synced: the next _launch replaces it after this set's _finish), or a transposed copy unfused
        if choiceT is not None:
            choiceT_t = self._to_torch(choiceT)
            gs.hold = (choiceT, choiceT_t)
        else:
            choiceT_t = gs.inputs_buf
            choiceT_t.copy_(choice_t.t())
        inputs = (choiceT_t[0], choiceT_t[1], choiceT_t[2])
        # maps for the games that finish in this tick: fresh seeds (also the game RNG seeds) and their table rows, from
        # the pinned buffers (two H2D copies), two index_selects
        seeds = self._draw_seeds(sh, Gs, gs.lo)
        gs.seeds_host = seeds
        gs.seeds_pin.numpy()[:] = seeds
        gs.rows_pin.numpy()[:] = seeds.astype(np.int64) - self.map_lo
        gs.seeds.copy_(gs.seeds_pin, non_blocking=True)
        gs.rows.copy_(gs.rows_pin, non_blocking=True)
        torch.index_select(sh.table3, 1, gs.rows, out=gs.layouts3)
        torch.index_select(sh.table_caps, 0, gs.rows, out=gs.layout_caps)
        self._lap('book', t0, sh)
        t0 = time.perf_counter()
        gs.env.tick_device(*inputs, gs.seeds, *gs.layouts, True, gs.out, sync=not self.async_ticks)
        gs.pending = t
        if not self.async_ticks:
            self._lap('tick', t0, sh)
        else:
            gs.in_flight = True

    def _pool_grouped(self, sh, gs, opp_turn, choice3, choiceT, k_pool):
        """The pool opponents' decisions with one act_group call per parameter structure (stage 4)."""
        jax = self.jax
        if getattr(sh, 'pool_warm_due', False) and pool_warm_on(os.environ, getattr(self, 'play_size', 0)):
            self._warm_pool(sh, gs, choice3, k_pool)
        lanes, lo, hi, Gs = sh.lanes, gs.lo, gs.hi, self.Gs
        codes = lanes.lane_code[lo:hi]
        by_group = {}
        for code in np.unique(codes[opp_turn]).tolist():
            gi, j = sh.pool_group_of[lanes.code_name[code]]
            by_group.setdefault(gi, []).append((j, np.flatnonzero(opp_turn & (codes == code))))
        for gi in sorted(by_group):
            members = by_group[gi]
            stacked, K = sh.pool_stacks[gi]
            ka, B = pool_call_shape(len(members), max(len(ix) for _, ix in members), Gs)   # one executable per pair
            which = np.full(ka, members[0][0], np.int32)
            idx2 = np.full((ka, B), Gs, np.int32)
            for r, (j, ix) in enumerate(members):
                which[r] = j
                idx2[r, :len(ix)] = ix
            k_pool, k_sub = jax.random.split(k_pool)
            put = (lambda a: jax.numpy.asarray(a)) if self.n_devices == 1 else (lambda a: jax.device_put(a, sh.jax_device))
            which_j, idx_j = put(which), put(idx2)
            sub = self._act_group(stacked, which_j, gs.view_j, gs.masks_j, idx_j, k_sub)
            choice3, choiceT = self._merge(choice3, idx_j.reshape(-1), sub)
        return choice3, choiceT, k_pool

    def _step(self):
        """One engine step of every set of every device. With N sets the loop is a round robin finish(k) -> launch(k):
        when set k is finished (its tick from the previous step awaited), sets k+1.. still run their previous ticks and
        sets ..k-1 already run this step's, so N-1 ticks are in flight behind every set's JAX work (the fused call, or
        encode, critic value and sampling unfused, plus the bookkeeping); the engine is hidden once (N-1) x JAX(set) >=
        tick(set), and the step period is max(N x JAX(set), tick(set) + JAX(set)) otherwise. Several devices: set k of
        every device in lockstep; fused, each device's set k is finished and its call dispatched first (the calls run
        concurrently on their GPUs), then each is awaited, booked and its tick launched, so the host waits once per set
        index for the slowest device rather than N times in sequence. Timing keys per set-step: engine_wait, fused
        (dispatch + the host wait; encode / sample unfused), outcomes, pool, book; `step` per step."""
        import jax
        t = self.shards[0].t
        for sh in self.shards:
            sh.h_step_version[t % self.L] = self.version            # the critic every set's rows of this step are valued with
        keys = []
        for sh in self.shards:
            sh.key, *ks = jax.random.split(sh.key, self.sets + 1)
            keys.append(ks)
        t0 = time.perf_counter()
        if self.n_devices == 1:
            sh, ks = self.shards[0], keys[0]
            for gs, k in zip(sh.gsets, ks):
                if self.sets > 1:
                    self._finish(sh, gs)
                self._launch(sh, gs, t, k)
                if self.sets == 1:
                    self._finish(sh, gs)
        else:
            for k in range(self.sets):
                pending = []
                for sh, ks in zip(self.shards, keys):
                    gs = sh.gsets[k]
                    with self._on(sh.index):
                        self._finish(sh, gs)
                        if self.fused:
                            pending.append((sh, gs, ks[k], self._dispatch(sh, gs, ks[k])))
                        else:
                            self._launch(sh, gs, t, ks[k])
                for sh, gs, key, dispatched in pending:
                    with self._on(sh.index):
                        self._launch(sh, gs, t, key, dispatched)
        self._lap('step', t0)
        for sh in self.shards:
            sh.t += 1

    def _step_shard(self, sh):
        """One engine step of one device's sets on its own host thread (dp_threads): the single-device round robin
        finish(k) -> launch(k) over the shard's sets; every tick is queued (synced by the next step's finish)."""
        import jax
        t = sh.t
        sh.h_step_version[t % self.L] = self.version
        sh.key, *ks = jax.random.split(sh.key, self.sets + 1)
        t0 = time.perf_counter()
        for gs, k in zip(sh.gsets, ks):
            self._finish(sh, gs)
            self._launch(sh, gs, t, k)
        self._lap('step', t0, sh)
        sh.t += 1

    # ---- device threads (dp_threads)
    def _start_workers(self):
        self.workers = [_DeviceWorker(self, sh) for sh in self.shards]
        for w in self.workers:
            w.call('init')
        for w in self.workers:
            w.result()

    def _stop_workers(self):
        for w in self.workers:
            w.call('exit')
        for w in self.workers:
            w.thread.join(timeout=60)
        self.workers = []

    def _init_shard(self, sh):
        for gs in sh.gsets:
            self._finish(sh, gs)                   # the initial state's features (unfused) / nothing in flight yet

    def _roll_shard(self, sh):
        """A device thread's rollout: steps until the main thread pauses the rollouts (the global closed rows reached
        the batch, or the time is up); each step's closed rows are announced on the condition. Returns the steps run."""
        steps = 0
        while not self._pause.is_set() and self._worker_error is None:
            self._step_shard(sh)
            steps += 1
            with self._cond:
                self._cond.notify_all()
        return steps

    def _worker_failed(self, exc):
        with self._cond:
            if self._worker_error is None:
                self._worker_error = exc
            self._cond.notify_all()

    def _roll_until_batch(self, deadline):
        """Threaded rollout to the next update: every device thread steps on its own; the main thread waits until the
        SUM of the devices' closed rows reaches `batch` (or the deadline), pauses the threads (each finishes the step it
        is in, so a device may close a few rows past the trigger; they stay closed for the next update) and waits for
        all of them to park. Returns True when a batch is ready."""
        batch = self.process_rows      # several processes: this process's share
        t0 = time.perf_counter()
        self._pause.clear()
        for w in self.workers:
            w.call('roll')
        ready = lambda: self._worker_error is not None or self._seg('closed_rows') >= batch
        with self._cond:
            self._cond.wait_for(ready, timeout=max(deadline - time.monotonic(), 0.))
        self._pause.set()
        steps = [w.result() for w in self.workers]     # raises a device thread's error
        self._lap('rollout', t0)
        self.last_roll_steps = steps
        return self._seg('closed_rows') >= batch

    # ---- assembly and update
    def _assemble_part(self, sh, rows):
        """A device's share (None when it is 0) with the shard's 'gae' lap (data parallel; on the device's thread)."""
        if not rows:
            return None
        part = self._assemble_shard(sh, rows)
        self._lap('gae', part['t0'], sh)
        return part

    def _assemble_shard(self, sh, rows):
        """One device's share: its oldest `rows` closed rows gathered from its ring into its batch buffers, scored with
        the critic (rows and boundaries), segment GAE with the turn clock. Returns the raw advantages / returns of the
        first `rows` rows (the straddling segment's tail past them is dropped, as the production learner drops rows
        past `batch`), the values of every gathered row, and the info counts."""
        torch = self.torch
        cfg = self.config
        G, L, C = self.G, self.L, VALUE_CHUNK
        t0 = time.perf_counter()
        take = sh.seglog.take(rows, sh.t - 1, L)
        n, segs = take['n'], take['segments']
        S = len(segs['length'])
        dev64 = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.int64)).to(sh.device)
        flat = dev64((take['step'] % L) * G + take['game'])
        torch.index_select(sh.ring_view, 0, flat, out=sh.buf_view[:n])
        torch.index_select(sh.ring_opp, 0, flat, out=sh.buf_opp[:n])
        torch.index_select(sh.ring_masks, 0, flat, out=sh.buf_masks[:n])
        torch.index_select(sh.ring_choice, 0, flat, out=sh.buf_choice[:n])
        torch.index_select(sh.ring_logp, 0, flat, out=sh.buf_logp[:n])
        # boundaries: the ring row of the boundary step oriented to the segment's seat (own view first); final views
        # from the store at a truncation bootstrap; terminal segments carry a (ignored) valid row
        S_pad = _ceil_to(S, C)
        bflat = dev64((segs['boot_step'] % L) * G + segs['game'])
        bv = torch.index_select(sh.ring_view, 0, bflat)
        bo = torch.index_select(sh.ring_opp, 0, bflat)
        swap = torch.from_numpy(segs['boot_swap']).to(sh.device)[:, None]
        boot_view = torch.zeros((S_pad, self.vw), dtype=torch.int16, device=sh.device)
        boot_opp = torch.zeros((S_pad, self.vw), dtype=torch.int16, device=sh.device)
        boot_view[:S] = torch.where(swap, bo, bv)
        boot_opp[:S] = torch.where(swap, bv, bo)
        stored = np.flatnonzero(segs['boot_store'] >= 0)
        if len(stored):
            if (sh.boot_cursor - segs['boot_store'][stored]).max() > BOOT_STORE:
                raise RuntimeError('a truncation boundary left the boundary store before its batch assembled')
            srows = dev64(segs['boot_store'][stored] % BOOT_STORE)
            seat = dev64(segs['seat'][stored])
            own = sh.boot_store[srows, seat]
            opp = sh.boot_store[srows, 1 - seat]
            idx = dev64(stored)
            boot_view[idx], boot_opp[idx] = own, opp
        end_turn = (sh.buf_choice[:n, 0] == END_FAMILY).cpu().numpy()
        torch.cuda.synchronize(sh.device)
        t0 = self._lap('gather', t0, sh)
        info = {}
        if self.value_mode == 'assembly':
            values = self._value_rows(sh, sh.buf_view, sh.buf_opp, n)
            boots = self._value_rows(sh, boot_view, boot_opp, S)
        else:
            # the rollout's values from the ring; stale rows (an older critic) and the boundaries that are not a
            # current own row re-scored with the small call (value_sources)
            row_stale, b_ring, b_call = value_sources(take, sh.h_step_version, self.version)
            refresh = self.value_mode == 'rollout'
            if not refresh:   # rollout-stale: every own-row boundary from the ring, whatever critic valued it
                nonterm_b = ~segs['terminal']
                b_ring = nonterm_b & (segs['boot_store'] < 0) & ~segs['boot_swap']
                b_call = nonterm_b & ~b_ring
            torch.index_select(sh.ring_value, 0, flat, out=sh.buf_value[:n])
            values = sh.buf_value[:n].cpu().numpy().copy()
            boots = np.zeros(S, np.float32)
            if b_ring.any():
                boots[b_ring] = sh.ring_value[bflat[torch.from_numpy(b_ring).to(sh.device)]].cpu().numpy()
            stale = np.flatnonzero(row_stale) if refresh else np.zeros(0, np.int64)
            if len(stale):
                values[stale] = self._values_gathered(sh, sh.buf_view, sh.buf_opp, dev64(stale))
            call = np.flatnonzero(b_call)
            if len(call):
                boots[call] = self._values_gathered(sh, boot_view, boot_opp, dev64(call))
            info.update(stale_rows=int(row_stale.sum()), stale_rows_rescored=int(len(stale)), boundary_calls=int(len(call)),
                        boundary_from_ring=int(b_ring.sum()))
            if self.value_check:
                ref_v = self._value_rows(sh, sh.buf_view, sh.buf_opp, n)
                ref_b = self._value_rows(sh, boot_view, boot_opp, S)
                nonterm = ~segs['terminal']
                info.update(value_check_rows_max_abs_diff=float(np.abs(ref_v - values).max()) if n else 0.,
                            value_check_boundaries_max_abs_diff=float(np.abs(ref_b[nonterm] - boots[nonterm]).max()) if nonterm.any() else 0.,
                            value_check_fresh_rows_max_abs_diff=float(np.abs(ref_v[~row_stale] - values[~row_stale]).max())
                            if (~row_stale).any() else 0.)
        t0 = self._lap('value', t0, sh)
        turn_clock = cfg.turn_gamma is not None
        gamma, lam = (cfg.turn_gamma, cfg.turn_lambda) if turn_clock else (cfg.gamma, cfg.gae_lambda)
        adv, ret = segment_gae(values, take['seg'], take['pos'], segs['length'], segs['terminal'], segs['reward'], boots,
                               gamma, lam, end_turn=end_turn if turn_clock else None)
        return dict(adv=adv[:rows], ret=ret[:rows], values=values, n=n, segments=S, terminal=np.asarray(segs['terminal'], bool),
                    stored=len(stored), info=info, t0=t0)

    def _assemble(self):
        """The oldest `batch` closed rows: gathered from the ring(s), scored with the critic (rows and boundaries),
        segment GAE with the turn clock, advantages normalized over the batch. Returns (device batch, info).

        Several devices: device d contributes row_shares(closed, batch)[d] rows (batch // N each unless a device is
        short; its shortfall comes from the others), assembled and GAE'd on its own GPU from its own segments; the
        advantages are normalized with the mean / std of the WHOLE batch (the per-device raw advantages are host arrays:
        concatenated, the same two-pass mean / std as one device); rebalance_plan then moves the few surplus rows so
        every device holds exactly batch // N rows (the update's static shard), and the batch is one global jax.Array
        sharded P('data') over the mesh, each shard aliasing its device's batch buffers."""
        cfg = self.config
        if self.n_global == 1:
            r = self._assemble_shard(self.shards[0], cfg.batch)
            return self._single_batch(self.shards[0], r)
        return self._global_batch()

    def _single_batch(self, sh, r):
        jax = self.jax
        cfg = self.config
        adv, ret, values, n, S = r['adv'], r['ret'], r['values'], r['n'], r['segments']
        info = r['info']
        raw_std = float(adv.std())
        norm = ((adv - adv.mean()) / max(raw_std, cfg.advantage_floor)).astype(np.float32)
        dev = dict(view=self._to_jax(sh.buf_view), opp_view=self._to_jax(sh.buf_opp), masks=self._to_jax(sh.buf_masks),
                   choice=self._to_jax(sh.buf_choice), logp=self._to_jax(sh.buf_logp),
                   advantage=jax.numpy.asarray(np.concatenate([norm, np.zeros(self.N_cap - cfg.batch, np.float32)])),
                   returns=jax.numpy.asarray(np.concatenate([ret.astype(np.float32), np.zeros(self.N_cap - cfg.batch, np.float32)])))
        self._lap('gae', r['t0'])
        info.update(rows_gathered=n, dropped_rows=n - cfg.batch, segments=S, terminal_fraction=float(r['terminal'].mean()),
                    raw_advantage_std=raw_std, return_mean=float(ret.mean()), value_mean_batch=float(values.mean()),
                    boundary_store_rows=int(r['stored']))
        return dev, info

    def _psum2(self, v):
        """Sum of this process's small float vector `v` over every process (mixed mode)."""
        x = self.jax.device_put(np.asarray(v, np.float32)[None], self.shards[0].jax_device)
        return tuple(float(t) for t in np.asarray(self._host(self._psum_small(self._global_rows(x)))))

    def _global_batch(self):
        jax, torch = self.jax, self.torch
        cfg = self.config
        self.phase = 'assemble'
        N, per = self.n_devices, self.per_device_rows
        closed = np.array([sh.seglog.closed_rows for sh in self.shards], np.int64)
        shares = row_shares(closed, self.process_rows)
        t0 = time.perf_counter()
        if self.threaded:
            # every device's take / gather / value pass / GAE on its own thread, in parallel
            for w, rows in zip(self.workers, shares.tolist()):
                w.call('assemble', rows)
            parts = [w.result() for w in self.workers]
        else:
            parts = []
            for sh, rows in zip(self.shards, shares.tolist()):
                with self._on(sh.index):
                    parts.append(self._assemble_part(sh, rows))
        t0 = self._lap('assemble_devices', t0)
        live = [p for p in parts if p is not None]
        adv_all = np.concatenate([p['adv'] for p in live])
        ret_all = np.concatenate([p['ret'] for p in live])
        device_stats = self.n_procs > 1     # several processes: the batch statistics on the GPUs (no host allgather)
        if not device_stats:
            raw_std = float(adv_all.std())
            mean = adv_all.mean()
        # every device's host advantages / returns, `per` rows each, after the rebalance moves
        adv_d = [np.zeros(per, adv_all.dtype) for _ in range(N)]
        ret_d = [np.zeros(per, ret_all.dtype) for _ in range(N)]
        for d, p in enumerate(parts):
            if p is not None:
                k = min(int(shares[d]), per)
                adv_d[d][:k], ret_d[d][:k] = p['adv'][:k], p['ret'][:k]
        moves = rebalance_plan(shares, per)
        names = ('buf_view', 'buf_opp', 'buf_masks', 'buf_choice', 'buf_logp')
        for src, slo, dst, dlo, k in moves:
            adv_d[dst][dlo:dlo + k] = parts[src]['adv'][slo:slo + k]
            ret_d[dst][dlo:dlo + k] = parts[src]['ret'][slo:slo + k]
            a, b = self.shards[src], self.shards[dst]
            with self._on(b.index):
                for name in names:
                    getattr(b, name)[dlo:dlo + k].copy_(getattr(a, name)[slo:slo + k])
        for sh in self.shards:
            torch.cuda.synchronize(sh.device)
        scale = None if device_stats else max(raw_std, cfg.advantage_floor)

        def global_of(per_device):
            arrays = list(per_device)
            return jax.make_array_from_single_device_arrays((self.n_global * per,) + tuple(arrays[0].shape[1:]),
                                                            self.data_sharding, arrays)

        def rows_of(name):
            # imported inside each device's context: torch's __dlpack__ orders the import after ITS current device's
            # stream (the buffers are complete anyway: synchronized above)
            arrays = []
            for sh in self.shards:
                with self._on(sh.index):
                    arrays.append(self._to_jax(getattr(sh, name)[:per]))
            return global_of(arrays)

        if self.mixed:
            # every process keeps its own rows, advantages and returns (their counts and widths are its own); the
            # whole batch's mean / std come from two psums of (count, sum) and (sum of squared deviations)
            sh = self.shards[0]
            a = adv_d[0].astype(np.float64)
            self.phase = 'psum_stats'
            n_tot, s_tot = self._psum2(np.array([len(a), a.sum()], np.float32))
            mean_all = s_tot / n_tot
            sq_tot, _ = self._psum2(np.array([((a - mean_all) ** 2).sum(), 0.], np.float32))
            raw_std = float(np.sqrt(sq_tot / n_tot))
            norm = ((a - mean_all) / max(raw_std, cfg.advantage_floor)).astype(np.float32)
            dev = dict(view=self._to_jax(sh.buf_view[:per]), opp_view=self._to_jax(sh.buf_opp[:per]),
                       masks=self._to_jax(sh.buf_masks[:per]), choice=self._to_jax(sh.buf_choice[:per]),
                       logp=self._to_jax(sh.buf_logp[:per]), advantage=jax.device_put(norm, sh.jax_device),
                       returns=jax.device_put(ret_d[0].astype(np.float32), sh.jax_device))
            device_stats = False
        else:
          dev = dict(view=rows_of('buf_view'), opp_view=rows_of('buf_opp'), masks=rows_of('buf_masks'),
                   choice=rows_of('buf_choice'), logp=rows_of('buf_logp'),
                   advantage=(global_of(jax.device_put(((a - mean) / scale).astype(np.float32), sh.jax_device)
                                        for a, sh in zip(adv_d, self.shards)) if not device_stats else None),
                   returns=global_of(jax.device_put(r.astype(np.float32), sh.jax_device) for r, sh in zip(ret_d, self.shards)))
        if device_stats:
            # the WHOLE batch's two-pass mean / std on the devices (two psums in one jitted program over NCCL), then the
            # same normalization (floor advantage_floor); replaces two host process_allgathers per update (session A,
            # multinode2 profile: dp_batch 140 ms at 16 GPUs vs 45 ms at 8)
            raw = global_of(jax.device_put(a.astype(np.float32), sh.jax_device) for a, sh in zip(adv_d, self.shards))
            dev['advantage'], raw_std_dev = self._normalize_dp(raw, np.float32(cfg.advantage_floor))
            raw_std = float(self._host(raw_std_dev))
        self._lap('dp_batch', t0)
        info = {}
        for p in live:
            for k, v in p['info'].items():
                info[k] = max(info.get(k, 0.), v) if k.startswith('value_check') else info.get(k, 0) + v
        n_total = sum(p['n'] for p in live)
        segments = sum(p['segments'] for p in live)
        values = np.concatenate([p['values'] for p in live])
        info.update(rows_gathered=n_total, dropped_rows=n_total - cfg.batch, segments=segments,
                    terminal_fraction=float(np.concatenate([p['terminal'] for p in live]).mean()),
                    raw_advantage_std=raw_std, return_mean=float(ret_all.mean()), value_mean_batch=float(values.mean()),
                    boundary_store_rows=int(sum(p['stored'] for p in live)), row_shares=shares.tolist(),
                    closed_rows_at_update=closed.tolist(), rebalanced_rows=int(sum(m[4] for m in moves)))
        return dev, info

    # ---- Session A (T3 speed): profiling and GC knobs; timing only, the update's math is unchanged
    def _setup_train_knobs(self):
        """V3L_TRAIN_PROFILE=1: per-update train-phase timings in the log row (each minibatch step's local gradient and
        global apply waited for: the steps are serialised, so it measures, it does not speed up). V3L_TRAIN_BENCH=1,2,4
        (with the profile): once, V3L_TRAIN_BENCH_AT updates after the start, this process's local gradient of
        minibatch 0 under each grad_accum split, timed and compared with the production one (results discarded).
        V3L_GC=freeze: gc.freeze() once the trainer is built (the long-lived setup objects leave the collected
        generations, so a full collection no longer walks them). V3L_GC_LOG=1 (or the profile): the collector's
        pauses per update (gc_ms, gc_collections, gc_gen2).
        V3L_GC=scheduled): freeze as above, then no AUTOMATIC full (generation-2)
        collection - its threshold is raised out of reach, generations 0/1 stay automatic - and one timed
        gc.collect(2) right after _publish every V3L_GC_FULL_EVERY updates (default 1). Every rank has just left the
        update's collectives there, so the full pauses line up instead of one rank's pause holding all ranks at the next
        collective. The collector frees memory only; no array, RNG stream or decision changes.
        V3L_GC_REFREEZE=1: gc.freeze() right after every scheduled collection.
        The l-ibqual2 point cost ms/update on ranks 1-7 (80 on rank 0) with nothing to free: it walks the
        long-lived objects made after the start-up freeze (JAX's trace / lowering caches of the first updates' compiles,
        the run's tables). Frozen, the next point walks only what the update left alive. Objects that become cyclic
        garbage after being frozen wait for a deep point: V3L_GC_DEEP_EVERY=M (0 = never) unfreezes, collects everything
        and refreezes every M updates (logged gc_point_deep / gc_point_collected / gc_frozen)."""
        self.train_profile = os.environ.get('V3L_TRAIN_PROFILE', '0') == '1'
        self.train_bench = [int(x) for x in os.environ.get('V3L_TRAIN_BENCH', '').split(',') if x.strip()]
        self._bench_left = int(os.environ.get('V3L_TRAIN_BENCH_AT', '3')) if self.train_bench else -1
        self.gc_mode = os.environ.get('V3L_GC', '')
        if self.gc_mode not in ('', 'freeze', 'scheduled'):
            raise SystemExit(f'V3L_GC={self.gc_mode!r}: expected "", "freeze" or "scheduled"')
        self.gc_full_every = int(os.environ.get('V3L_GC_FULL_EVERY', '1'))
        if self.gc_full_every < 1:
            raise SystemExit(f'V3L_GC_FULL_EVERY={self.gc_full_every}: expected >= 1')
        self.value_tail = int(os.environ.get('V3L_VALUE_TAIL', '0'))
        if self.value_tail < 0 or self.value_tail >= VALUE_CHUNK or (self.value_tail and VALUE_CHUNK % self.value_tail):
            raise SystemExit(f'V3L_VALUE_TAIL={self.value_tail}: expected 0 or a divisor of {VALUE_CHUNK} below it')
        self.gc_refreeze = os.environ.get('V3L_GC_REFREEZE', '0') == '1'
        self.gc_deep_every = int(os.environ.get('V3L_GC_DEEP_EVERY', '0'))
        if os.environ.get('V3L_GC_REFREEZE', '0') not in ('0', '1') or self.gc_deep_every < 0:
            raise SystemExit('V3L_GC_REFREEZE expects 0 or 1, V3L_GC_DEEP_EVERY an integer >= 0')
        if (self.gc_refreeze or self.gc_deep_every) and self.gc_mode != 'scheduled':
            raise SystemExit('V3L_GC_REFREEZE / V3L_GC_DEEP_EVERY need V3L_GC=scheduled')
        if self.gc_deep_every and (not self.gc_refreeze or self.gc_deep_every % self.gc_full_every):
            raise SystemExit('V3L_GC_DEEP_EVERY needs V3L_GC_REFREEZE=1 and a multiple of V3L_GC_FULL_EVERY')
        self.gc_last = {}
        self._gc_log = None
        if self.train_profile or os.environ.get('V3L_GC_LOG', '0') == '1':
            log, start = dict(ms=0., n=0, gen2=0), {}

            def on_gc(phase, info):
                if phase == 'start':
                    start['t'] = time.perf_counter()
                elif 't' in start:
                    log['ms'] += (time.perf_counter() - start.pop('t')) * 1e3
                    log['n'] += 1
                    log['gen2'] += info.get('generation') == 2
            gc.callbacks.append(on_gc)
            self._gc_log = log
        if self.gc_mode in ('freeze', 'scheduled'):
            gc.collect()
            gc.freeze()
        if self.gc_mode == 'scheduled':
            t0, t1, _ = gc.get_threshold()
            gc.set_threshold(t0, t1, GC_NO_AUTO_FULL)

    def _gc_point(self):
        """V3L_GC=scheduled: the full collection at the fixed point after _publish (every gc_full_every updates);
        returns its wall ms, or None when not scheduled / not due."""
        if self.gc_mode != 'scheduled' or (self.version + 1) % self.gc_full_every:
            return None
        t0 = time.perf_counter()
        deep = bool(self.gc_deep_every) and (self.version + 1) % self.gc_deep_every == 0
        if deep:
            gc.unfreeze()
        collected = gc.collect(2)
        if self.gc_refreeze:
            gc.freeze()
        ms = (time.perf_counter() - t0) * 1e3
        self.gc_last = dict(gc_point_collected=collected)
        if deep:      # the permanent generation's size only here (counting it walks the list)
            self.gc_last.update(gc_point_deep=True, gc_frozen=gc.get_freeze_count())
        return ms

    def _bench_due(self):
        if self._bench_left < 0:
            return False
        self._bench_left -= 1
        return self._bench_left == 0

    def _train_bench(self, trainable, local, perm, entropy_coef, reps=3):
        """Profiling only: minibatch 0's flat local gradient (grads x rows, metrics x rows, rows) under each
        grad_accum split in V3L_TRAIN_BENCH, median wall ms over `reps` waited calls after a compiling one, and its
        difference from the production executable's vector. Nothing it computes enters the update."""
        jax = self.jax
        sh = self.shards[0]
        mine = jax.tree.map(lambda a: _shard_on(a, sh.jax_device), trainable)
        args = (mine, local, perm, np.int32(0), np.float32(entropy_coef))
        ref = np.asarray(self._local_grad_flat(*args).block_until_ready())
        out = {}
        for a in self.train_bench:
            if a < 1 or self.local_minibatch % a:
                out[str(a)] = dict(skipped=f'local minibatch {self.local_minibatch} not a multiple of {a}')
                continue
            fn = self._local_grad_flat if a == int(self.config.grad_accum) else self._bench_flat(a)
            t0 = time.perf_counter()
            v = np.asarray(fn(*args).block_until_ready())
            first = (time.perf_counter() - t0) * 1e3
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn(*args).block_until_ready()
                ts.append((time.perf_counter() - t0) * 1e3)
            d = np.abs(v - ref)
            out[str(a)] = dict(ms=round(float(np.median(ts)), 2), first_ms=round(first, 1),
                               max_abs=float(d.max()), rel_l2=float(np.linalg.norm(v - ref) / max(np.linalg.norm(ref), 1e-30)))
        return dict(rows=int(self.local_minibatch), **out)

    def _global_rows(self, a):
        """A process's [1, ...] local array as its row of the [N, ...] global array sharded P('data') (mixed sizes)."""
        return self.jax.make_array_from_single_device_arrays((self.n_global,) + tuple(a.shape[1:]), self.data_sharding, [a])

    def _value_rows(self, sh, view_t, opp_t, k):
        """The assembly value pass: the first k rows of the device pair (view_t, opp_t) (at least ceil(k, VALUE_CHUNK)
        rows long) through values_chunk, VALUE_CHUNK at a time; numpy [k]."""
        C = VALUE_CHUNK
        out = [self._values_chunk(self._value_params(sh), self._to_jax(view_t[c * C:(c + 1) * C]), self._to_jax(opp_t[c * C:(c + 1) * C]))
               for c in range(_ceil_to(k, C) // C)]
        return np.concatenate([np.asarray(v) for v in out])[:k] if out else np.zeros(0, np.float32)

    def _values_gathered(self, sh, src_view, src_opp, flat):
        """The small value call: rows `flat` (device int64 [k]) of the device pair (src_view, src_opp) gathered into
        the shard's scratch chunk, VALUE_CHUNK at a time; numpy [k]. Each chunk's result is read before the scratch is
        refilled (the torch gather must not overtake the JAX read of the aliased scratch).
        V3L_VALUE_TAIL=T: the last partial chunk goes through in T-row pieces (the
        first T rows of the scratch) instead of one full VALUE_CHUNK. Every update's boundary call (42-300 rows in
        l-ibqual2) and the stale rows' remainder were padded to 8192 rows; the values are the same per-row critic
        function (rows are independent; a different chunk shape may round differently in the last bit)."""
        torch = self.torch
        C, T = VALUE_CHUNK, self.value_tail
        k = int(flat.shape[0])
        out = []
        c = 0
        while c < k:
            width = T if T and k - c < C else C
            idx = flat[c:c + width]
            m = int(idx.shape[0])
            torch.index_select(src_view, 0, idx, out=sh.tmp_view[:m])
            torch.index_select(src_opp, 0, idx, out=sh.tmp_opp[:m])
            out.append(np.asarray(self._values_chunk(self._value_params(sh), self._to_jax(sh.tmp_view[:width]),
                                                     self._to_jax(sh.tmp_opp[:width])))[:m])
            c += width
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def _update(self, started):
        """One PPO update on the assembled batch; then the version, the pool snapshot and the bookkeeping. Several
        devices: the same loop over minibatches, each one shard_map'd step (per-device minibatch // N rows, pmean'd
        gradients, one adam_step); the permutation is per device (fold_in(key, device)) as in production."""
        import jax
        cfg = self.config
        tick = time.monotonic()
        dev, info = self._assemble()
        assembled = time.monotonic()
        elapsed = assembled - started
        learning_rate = scheduled_learning_rate(cfg.learning_rate, cfg.learning_rate_end, cfg.lr_decay_fraction,
                                                training_progress(cfg, self.version, elapsed))
        learning_rate = warmed_learning_rate(learning_rate, self.version, cfg.lr_warmup_updates)
        weight = np.float32(0. if self.version < cfg.critic_warmup else 1.)
        trainable = (self.params, self.critic)
        metrics_all = []
        steps = cfg.batch // cfg.minibatch
        prof = dict(local_ms=[], apply_ms=[]) if self.train_profile else None
        for _ in range(cfg.epochs):
            self.update_key, sub = jax.random.split(self.update_key)
            acc, count = None, 0
            if self.n_global == 1:
                perm = jax.random.permutation(sub, cfg.batch).astype(jax.numpy.int32)
                for i in range(steps):
                    idx = jax.lax.dynamic_slice(perm, (i * cfg.minibatch,), (cfg.minibatch,))
                    trainable, self.opt, m = self._train_step(trainable, self.opt, dev, idx, np.float32(learning_rate),
                                                              np.float32(cfg.entropy_coef), weight)
                    acc = m if acc is None else jax.tree.map(jax.numpy.add, acc, m)
                    count += 1
            elif self.mixed:
                sh = self.shards[0]
                perm = self._local_perm(sub, sh)
                local = dict(dev, advantage=_shard_on(dev['advantage'], sh.jax_device),
                             returns=_shard_on(dev['returns'], sh.jax_device))
                if prof is not None:
                    t0 = time.perf_counter()
                    jax.block_until_ready((local, perm))
                    prof['wait_batch_ms'] = (time.perf_counter() - t0) * 1e3
                    if self._bench_due():
                        prof['bench'] = self._train_bench(trainable, local, perm, cfg.entropy_coef)
                for i in range(self.per_device_rows // self.local_minibatch):
                    self.phase = f'train_step_{i}'
                    if prof is not None:
                        t0 = time.perf_counter()
                    mine = jax.tree.map(lambda a: _shard_on(a, sh.jax_device), trainable)
                    if self.mixed_flat:
                        if self._apply_flat is None:
                            self._apply_flat = self._make_apply_flat(jax.eval_shape(
                                self._local_grad, mine, local, perm, np.int32(i), np.float32(cfg.entropy_coef)))
                        flat = self._local_grad_flat(mine, local, perm, np.int32(i), np.float32(cfg.entropy_coef))
                        if prof is not None:     # timing only: the same calls, each waited for
                            flat.block_until_ready()
                            t1 = time.perf_counter()
                            prof['local_ms'].append((t1 - t0) * 1e3)
                        trainable, self.opt, m = self._apply_flat(trainable, self.opt, self._global_rows(flat),
                                                                  np.float32(learning_rate), weight)
                        if prof is not None:
                            jax.block_until_ready((trainable, self.opt, m))
                            prof['apply_ms'].append((time.perf_counter() - t1) * 1e3)
                    else:
                        g, m, w = self._local_grad(mine, local, perm, np.int32(i), np.float32(cfg.entropy_coef))
                        g, m, w = jax.tree.map(self._global_rows, g), jax.tree.map(self._global_rows, m), self._global_rows(w)
                        trainable, self.opt, m = self._apply_mean(trainable, self.opt, g, m, w, np.float32(learning_rate),
                                                                  weight)
                    acc = m if acc is None else jax.tree.map(jax.numpy.add, acc, m)
                    count += 1
            else:
                perm = self._perms(sub)
                for i in range(self.per_device_rows // self.local_minibatch):
                    trainable, self.opt, m = self._train_step_dp(trainable, self.opt, dev, perm, np.int32(i),
                                                                 np.float32(learning_rate), np.float32(cfg.entropy_coef),
                                                                 weight)
                    acc = m if acc is None else jax.tree.map(jax.numpy.add, acc, m)
                    count += 1
            t0 = time.perf_counter()
            metrics = {k: float(self._host(v)) / count for k, v in acc.items()}
            if prof is not None:
                prof['metrics_ms'] = prof.get('metrics_ms', 0.) + (time.perf_counter() - t0) * 1e3
            metrics_all.append(metrics)
            if not all(np.isfinite(list(metrics.values()))):
                raise RuntimeError(f'nonfinite PPO metrics {metrics}')
            if metrics['approx_kl'] > cfg.target_kl and self.version >= cfg.critic_warmup:
                break
        self.params, self.critic = trainable
        t0 = time.perf_counter()
        jax.block_until_ready(self.params)
        t1 = time.perf_counter()
        self._publish()
        gc_point_ms = self._gc_point()
        trained = time.monotonic()
        if gc_point_ms is not None:
            info['gc_point_ms'] = round(gc_point_ms, 2)
            info.update(self.gc_last)
        if prof is not None:
            loc, app = prof.pop('local_ms'), prof.pop('apply_ms')
            prof.update(block_ms=(t1 - t0) * 1e3, publish_ms=(time.perf_counter() - t1) * 1e3, steps=len(loc),
                        local_sum_ms=sum(loc), local_max_ms=max(loc, default=0.), local_first_ms=loc[0] if loc else 0.,
                        apply_sum_ms=sum(app), apply_max_ms=max(app, default=0.), apply_first_ms=app[0] if app else 0.)
            info['train_profile'] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in prof.items()}
        if self._gc_log is not None:
            info.update(gc_ms=round(self._gc_log['ms'], 2), gc_collections=self._gc_log['n'], gc_gen2=self._gc_log['gen2'])
            self._gc_log.update(ms=0., n=0, gen2=0)
        self.clock['update'] = self.clock.get('update', 0.) + (trained - assembled) * 1000.
        self.version += 1
        if cfg.critic_warmup and self.version == cfg.critic_warmup:
            from . import ppo
            self.opt = self._replicate(ppo.adam_init((self.params, self.critic)))
            self.log(json.dumps(dict(stage='critic_warmup_done', update=self.version)))
        if self.version % cfg.historical_every == 0 and cfg.historical_probability > 0:
            self.historical = (self.historical + [(f'hist-u{self.version}', self._rounded(self.params))])[-cfg.historical_keep:]
            self._set_pool()
        mean = {k: float(np.mean([m[k] for m in metrics_all])) for k in metrics_all[0]}
        if hasattr(self, '_compiles'):     # this rank's XLA compiles so far (Session A: a late compile = an all-rank stall)
            info.update(xla_compiles=self._compiles[0], xla_compile_seconds=round(self._compiles[1], 2))
        return dict(info, learning_rate=learning_rate, epochs_run=len(metrics_all), assemble_seconds=assembled - tick,
                    train_seconds=trained - assembled, **mean)

    # ---- checkpoints
    def _host_state(self):
        """Host copies of (params, Adam state, critic) and the stage: a consistent snapshot of this version, taken on
        the loop thread before the next update replaces the device arrays."""
        jax = self.jax
        h = lambda t: jax.tree.map(self._host, t)
        return dict(stage=f'update-{self.version}', params=h(self.params), opt=h(self.opt), critic=h(self.critic))

    def _write(self, name, state):
        from .runner import save_checkpoint
        save_checkpoint(self.root / f'{name}.npz', state['params'], self.run_config, stage=state['stage'], optimizer=state['opt'])
        save_checkpoint(self.root / f'{name}-critic.npz', state['critic'], self.run_config, stage=state['stage'])

    def _save(self, name):
        """Synchronous save (the run's end); joins a background write first so the files land in order."""
        self.ckpt.wait()
        self._write(name, self._host_state())

    def _checkpoint(self, rows, names):
        """Log rows + checkpoints on the background writer: only the host copy stalls the loop; the compression, the
        files and the volume commit run off it. Returns the loop's share (seconds); the writer's own time is reported
        as checkpoint_write_seconds of the next update."""
        t0 = time.perf_counter()
        if not self.is_writer:      # several processes: rank r > 0 keeps its own log rows only (rank 0 owns checkpoints)
            training_log.append(self.root / training_log.JSONL, rows)
            return time.perf_counter() - t0
        state = self._host_state()
        path, commit, write = self.root / training_log.JSONL, self.commit, self._write

        def job():
            training_log.append(path, rows)
            for name in names:
                write(name, state)
            commit()

        self.ckpt.submit(job)
        return time.perf_counter() - t0

    def peace_stats(self, st=None):
        """Stage-4 collusion-monitor counters of this process (cumulative): peace-legal games finished / capped (all
        and self-play), game-steps in peace-legal sets, of them at peace, and the learner's diplomatic decisions."""
        st = st or self._stats()
        out = {k: int(st[k]) for k in ('peace_finished', 'peace_truncated', 'peace_selfplay_games',
                                        'peace_selfplay_truncations', 'peace_set_steps')}
        out['peace_steps'] = int(sum(int(gs.peace_steps.item()) for sh in self.shards for gs in sh.gsets))
        out['diplomatic'] = int(sum(int(gs.diplo.item()) for sh in self.shards for gs in sh.gsets))
        out['late_steps'] = int(sum(int(gs.late_steps.item()) for sh in self.shards for gs in sh.gsets))
        return out

    # ---- aggregates over the devices (one device: the shard's own objects)
    def _stats(self):
        """The actor statistics summed over the devices (vs / maps records added up per name)."""
        if self.n_devices == 1:
            return self.shards[0].stats
        out = {}
        for sh in self.shards:
            for k, v in sh.stats.items():
                if isinstance(v, dict):
                    acc = out.setdefault(k, {})
                    for name, rec in v.items():
                        acc[name] = [a + b for a, b in zip(acc.get(name, [0] * len(rec)), rec)]
                else:
                    out[k] = out.get(k, 0) + v
        return out

    def _seg(self, attr):
        """A SegmentLog counter summed over the devices."""
        return sum(getattr(sh.seglog, attr) for sh in self.shards)

    def _timing(self):
        """Mean milliseconds per call of every timing key: the run-wide clock plus, with device threads, the shards'
        (which then hold the rollout and assembly keys alone)."""
        clock, calls = dict(self.clock), dict(self.calls)
        if self.threaded:
            for sh in self.shards:
                for k, v in sh.clock.items():
                    clock[k] = clock.get(k, 0.) + v
                    calls[k] = calls.get(k, 0) + sh.calls.get(k, 0)
        return {k: round(clock[k] / max(calls.get(k, 1), 1), 2) for k in clock}

    def _per_device(self):
        """Several devices: each device's mean timing per call (engine_wait, fused, outcomes, pool, book, gather, value,
        gae) and its row / game counters."""
        if self.n_devices == 1:
            return None
        return [dict(device=sh.index, timing_ms={k: round(sh.clock[k] / max(sh.calls.get(k, 1), 1), 2) for k in sh.clock},
                     learner_rows=sh.seglog.recorded, closed_rows=sh.seglog.closed_rows, finished=sh.stats['finished'],
                     forced_close_fraction=sh.seglog.forced_rows / max(sh.seglog.recorded, 1)) for sh in self.shards]

    # ---- the run
    def _start_progress(self):
        """Stage 5 (Session L, after an all-rank hang with no stdout kept): every rank writes
        <its run dir>/progress.json every V3L_PROGRESS_S seconds (30) from a daemon thread: the phase it is in, engine
        steps, closed rows vs its quota, its update, and the XLA compiles so far (count, seconds). A hung run then
        shows which rank is where (rollout / a collective / compiling)."""
        import threading
        jax = self.jax
        self.phase, self._compiles = 'start', [0, 0.]

        def on_event(event, duration, **_):
            if 'backend_compile' in event:
                self._compiles[0] += 1
                self._compiles[1] += float(duration)
        try:
            jax.monitoring.register_event_duration_secs_listener(on_event)
        except Exception:
            pass
        every = float(os.environ.get('V3L_PROGRESS_S', '30'))
        path = self.root / 'progress.json'
        self._progress_stop = threading.Event()

        def loop():
            started = time.time()
            while not self._progress_stop.is_set():
                try:
                    d = dict(t=time.time(), seconds=round(time.time() - started, 1), rank=self.rank, size=self.play_size,
                             games=self.G, phase=self.phase, update=self.version,
                             steps=int(sum(sh.stats['steps'] for sh in self.shards)),
                             finished=int(sum(sh.stats['finished'] for sh in self.shards)),
                             closed_rows=int(self._seg('closed_rows')), process_rows=int(self.process_rows),
                             compiles=self._compiles[0], compile_seconds=round(self._compiles[1], 1))
                    tmp = path.with_suffix('.tmp')
                    tmp.write_text(json.dumps(d))
                    tmp.replace(path)
                except Exception as e:                  # never let the monitor take the trainer down
                    self.phase_error = repr(e)[:200]
                self._progress_stop.wait(every)
        threading.Thread(target=loop, name='v3l-progress', daemon=True).start()

    def run(self):
        import jax
        from .runner import write_json
        torch = self.torch
        cfg = self.config
        training_seconds = float(cfg.training_seconds)
        self.ckpt = _Checkpointer()
        self._setup_train_knobs()
        self._start_progress()
        write_json(self.root / 'protocol.json', dict(
            config=asdict(cfg), parent=self.parent, devices=self.n_devices, trainer='gpu_resident', games=self.G, sets=self.sets,
            ring=self.ring, ring_steps=self.L, value_mode=self.value_mode,
            architecture='GPU-resident: the CUDA engine steps every game on the learner GPU; same-seat segments, '
                         'omniscient critic, pool and schedule of the hero recipe; on-policy (no actor lag)',
            engine='rl.gpu_engine.GpuGames (engine-cu, CUDA)', ruleset=RULESET_ID))
        gpu_name = torch.cuda.get_device_name(self.device_index)
        if self.n_devices > 1:
            gpu_name = f'{self.n_devices} x {gpu_name}' if self.n_procs == 1 else \
                f'{self.n_procs} processes x {self.n_devices} x {gpu_name}'
        self.log(json.dumps(dict(stage='gpu_resident_start', gpu=gpu_name, devices=self.n_devices, dp_threads=self.threaded,
                                 games=self.G,
                                 games_total=self.G * self.n_devices, sets=self.sets, games_per_set=self.Gs,
                                 fused=self.fused, ring=self.ring,
                                 ring_steps=self.L, vw=self.vw, mask_bytes=self.mask_bytes, params=self.n_params,
                                 critic_params=self.n_critic, network=self.network, batch=cfg.batch, minibatch=cfg.minibatch,
                                 grad_accum=cfg.grad_accum, round_limit=self.round_limit, map_table=str(self.map_table_path or ''),
                                 trunk=os.environ.get('POLYTOPIA_TRUNK_DTYPE', 'float32'),
                                 matmul=os.environ.get('POLYTOPIA_MATMUL_PRECISION', 'HIGHEST'),
                                 heads=os.environ.get('POLYTOPIA_HEAD_PRECISION', ''),
                                 jax_mem_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', ''), value_mode=self.value_mode,
                                 value_check=self.value_check, ring_rows=self.L * self.G,
                                 ring_row_bytes=self.ring_bytes // (self.L * self.G), ring_gib=round(self.ring_bytes / 2 ** 30, 2),
                                 batch_buffers_gib=round(2 * self.shards[0].buf_view.numel() * 2 / 2 ** 30, 2), resumed_at=self.version,
                                 per_device_minibatch=cfg.minibatch // self.n_global, processes=self.n_procs,
                                 rank=self.rank, global_devices=self.n_global)))
        self.clock, self.calls = {}, {}
        for sh in self.shards:
            sh.clock, sh.calls = {}, {}
        history = list(self.resume_history)
        logged = len(history)
        started = time.monotonic() - self.resume_elapsed
        self.update_key = jax.random.key(cfg.seed + self.version)
        finished = lambda: cfg.training_updates and self.version >= cfg.training_updates

        vote_every = int(os.environ.get('POLYTOPIA_DP_VOTE_EVERY', '10'))

        def keep_going():
            """Several processes must stop at the same update (the next update's collectives would hang otherwise).
            The update-count stop is identical on every process without communication; the wall-clock stop is voted
            every vote_every updates only (a host allgather per update cost ~50 ms at 16 GPUs), so a time-limited run
            may overrun its deadline by up to vote_every updates."""
            if finished():
                return False
            local = time.monotonic() - started < training_seconds
            if self.n_procs == 1:
                return local
            if self.version % vote_every and self.version != first_version:
                return True
            from jax.experimental import multihost_utils
            return bool(multihost_utils.process_allgather(np.array([local], np.int32)).all())
        first_version = self.version
        steady_t = steady_samples = None
        last_report = time.monotonic()
        steps_at_update, rows_at_update, finished_at_update = 0, 0, 0
        bookkeeping = 0.
        if self.threaded:
            self._start_workers()                  # each thread: its device current, the initial state's features
        else:
            for sh in self.shards:
                with self._on(sh.index):
                    for gs in sh.gsets:
                        self._finish(sh, gs)       # the initial state's features
        self.phase = 'vote'
        go = keep_going()
        while go:
            self.phase = 'rollout'
            if self.threaded:
                # several processes: no wall-clock break inside a rollout (all must reach the update); keep_going decides
                deadline = started + training_seconds if self.n_procs == 1 else time.monotonic() + 1e6
                if not self._roll_until_batch(deadline):
                    break                          # the time ran out before the next batch
            else:
                self._step()
                if self._seg('closed_rows') < self.process_rows:   # several processes: this process's share
                    continue                       # no vote between engine steps (votes are collective)
            tick = time.monotonic()
            self.phase = 'update'
            info = self._update(started)
            self.phase = 'bookkeeping'
            now = time.monotonic()
            st = self._stats()
            steps, rows = st['steps'], self._seg('recorded')
            row = dict(update=self.version, samples=self.version * cfg.batch, seconds=now - last_report, elapsed=now - started,
                       update_seconds=now - tick, bookkeeping_seconds=bookkeeping,
                       checkpoint_write_seconds=self.ckpt.seconds, checkpoint_writes=self.ckpt.writes,
                       engine_steps=steps - steps_at_update, learner_rows=rows - rows_at_update,
                       samples_per_second=cfg.batch / max(now - last_report, 1e-9),
                       finished=st['finished'] - finished_at_update, finished_total=st['finished'],
                       terminated=st['terminated'], truncated=st['truncated'], cap_score_results=st['cap_score_results'],
                       nonok=st['nonok'], selfplay_games=st['selfplay_games'],
                       forced_close_fraction=self._seg('forced_rows') / max(self._seg('recorded'), 1),
                       forced_segments=self._seg('forced_segments'), cut_segments=self._seg('cut_segments'),
                       finished_segments=self._seg('finished_segments'),
                       pool=[n for n, _ in self.historical], vs=dict(st['vs']), maps=dict(st['maps']), tribes=dict(st.get('tribes', {})),
                       timing_ms=self._timing(), **info)
            if self.mixed_rules:
                row['peace'] = self.peace_stats(st)
            if self.n_devices > 1:
                row['per_device'] = self._per_device()
                if self.threaded:
                    row['dp_threads'] = True
                    row['roll_steps'] = list(self.last_roll_steps)    # steps each device ran for this update
            last_report = now
            steps_at_update, rows_at_update, finished_at_update = steps, rows, st['finished']
            history.append(row)
            self.log(json.dumps(dict(stage='update', **row)))
            if self.version - first_version == 1:
                steady_t, steady_samples = time.monotonic(), self.version * cfg.batch
            snapshot = self.version % cfg.snapshot_every == 0 or self.version in {int(u) for u in cfg.snapshot_at}
            if self.version % cfg.checkpoint_every == 0 or snapshot:
                names = ['latest'] + ([f'snapshots/update-{self.version}'] if snapshot else [])
                self.phase = 'checkpoint'
                bookkeeping = self._checkpoint(history[logged:], names)
                logged = len(history)
            else:
                bookkeeping = 0.
            self.phase = 'vote'
            go = keep_going()
        elapsed = time.monotonic() - started
        if self.threaded:
            self._stop_workers()                       # parked since the last rollout
        self.ckpt.wait()                               # the last background write (its log rows precede these)
        training_log.append(self.root / training_log.JSONL, history[logged:])
        write_json(self.root / training_log.JSON, history)
        if self.is_writer:
            self._save('latest')
            self._save('final')
            write_json(self.root / 'training-complete.json', dict(updates=self.version, seconds=elapsed))
            self.commit()
        st = self._stats()
        updates = self.version - first_version
        steady = None
        if steady_t is not None and updates > 1:
            steady = round((self.version * cfg.batch - steady_samples) / max(time.monotonic() - steady_t, 1e-9))
        jax_stats = jax.local_devices()[0].memory_stats() or {}
        samples_per_s = round(updates * cfg.batch / max(elapsed - self.resume_elapsed, 1e-9))
        self.result = dict(
            ok=True, gpu=gpu_name, run_dir=str(self.root), updates=updates, version=self.version, samples=self.version * cfg.batch,
            seconds=round(elapsed, 1), games=self.G, sets=self.sets, games_per_set=self.Gs, fused=self.fused, ring=self.ring,
            ring_steps=self.L, batch=cfg.batch,
            minibatch=cfg.minibatch, grad_accum=cfg.grad_accum, value_mode=self.value_mode, ring_gib=round(self.ring_bytes / 2 ** 30, 2),
            checkpoint_writes=self.ckpt.writes, params=self.n_params, critic_params=self.n_critic,
            samples_per_s=samples_per_s, samples_per_s_steady=steady,
            engine_steps=st['steps'], learner_rows=self._seg('recorded'), finished=st['finished'], terminated=st['terminated'],
            truncated=st['truncated'], cap_score_results=st['cap_score_results'], nonok_fraction=st['nonok'] / max(st['steps'], 1),
            forced_close_fraction=self._seg('forced_rows') / max(self._seg('recorded'), 1),
            timing_ms=self._timing(),
            gpu_mem_gib=dict(torch_max_allocated=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                             jax_peak_in_use=round(jax_stats.get('peak_bytes_in_use', 0) / 2 ** 30, 2)),
            pool=[n for n, _ in self.historical], vs=dict(st['vs']), maps=dict(st['maps']), tribes=dict(st.get('tribes', {})),
            metrics={k: round(float(v), 5) for k, v in history[-1].items() if isinstance(v, float)} if history else {})
        if self.n_devices > 1:
            per_dev = self._per_device()
            for rec, sh, dev in zip(per_dev, self.shards, jax.local_devices()[:self.n_devices]):
                ms = dev.memory_stats() or {}
                rec.update(torch_max_allocated_gib=round(torch.cuda.max_memory_allocated(sh.device) / 2 ** 30, 2),
                           jax_peak_in_use_gib=round(ms.get('peak_bytes_in_use', 0) / 2 ** 30, 2))
            self.result.update(devices=self.n_devices, dp_threads=self.threaded, games_total=self.G * self.n_devices,
                               per_device=per_dev,
                               samples_per_s_per_device=round(samples_per_s / self.n_global),
                               per_device_minibatch=cfg.minibatch // self.n_global, processes=self.n_procs,
                               rank=self.rank, global_devices=self.n_global)
        return history

    def close(self):
        if self.workers:
            self._pause.set()                      # a thread still rolling parks, then takes 'exit'
            self._stop_workers()
        for sh in self.shards:
            for gs in sh.gsets:
                gs.env.close()
