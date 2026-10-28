"""Duper-style decoupled PPO: CPU actor fleet, trajectory stream, multi-GPU learner.

Roles (after the Duper rl_ray_deploy layout, adapted to Modal and JAX):

* Actors (CPU containers, many processes): each process steps a block of games
  with engine-rs (rust_engine.RustGames: the real game's rules, polytopia_bot.ruleset) and runs the policy on the
  CPU, batched over its games. Segments of each learning seat's decisions are
  streamed to the learner with their boundary (the next decision's views, or
  the outcome); with learner_values (default) the actor runs no critic and the
  learner scores every row with its current critic and computes GAE when it
  assembles the batch (Stage A1, 2026-10-21). With learner_values=False the
  actor computes GAE from its own copy of the critic as before (bootstrapped at
  the segment boundary, zero at native endings, critic value at truncations).
  A background sender thread compresses and ships batches; the learner's reply
  carries the newest published weights and opponent pool. A process hosts one
  pool opponent for all its lanes or none (pool_per_process), so the rival
  forward is one wide batch instead of a few lanes every tick.
* Learner (one GPU node, N devices): receiver threads decompress messages into
  a bounded queue (backpressure blocks actors when the learner falls behind).
  The trainer assembles fixed batches, drops samples older than max_lag policy
  versions, scores them (learner_values), normalizes advantages over the batch
  and runs clipped PPO epochs with data parallelism (shard_map, pmean
  gradients) over every local GPU. Weights are published every publish_every
  updates.

Self-play games record both seats (each is the current policy). Games against a
pool opponent (historical snapshot or frozen rival) record the learner seat only.
The protocol is pickle over TCP between containers of one Modal workspace
(i6pn private network); it is not exposed to untrusted peers.
"""
from dataclasses import dataclass, asdict, field
import json
import os
import pickle
import queue
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np

from polytopia_bot.ruleset import RULESET_ID
from . import training_log

SUMMON_FAMILY = 2
END_FAMILY = 15                     # Masks field order: ..., diplomatic, end, resign
WARRIOR = 2
VIEW_KEYS = ('view', 'opp_view')    # packed int16 public views, the bulk of a row
TRAIN_KEYS = ('family_mask', 'source_mask', 'target_mask', 'choice', 'logp', 'advantage', 'returns', 'upgo_advantage')
VALUE_CHUNK = 8192                  # rows per device chunk of the learner's value pass
VALUE_SUB = 1024                    # four seats: rows per sub-chunk inside a value-pass chunk


# ----------------------------------------------------------------------------- wire

def _compress(data, level=1):
    import zstandard
    return zstandard.ZstdCompressor(level=level).compress(data)


def _decompress(data):
    import zstandard
    return zstandard.ZstdDecompressor().decompress(data)


# Frame: !QB (body length, kind) + body. Kind 0: pickle. Kind 1: zstd(pickle). Kind 2 (what send_msg compresses
# to, A6): zstd(!QI + n*Q index, pickle, raw buffers): protocol-5 out-of-band buffers, so the receiver's numpy
# arrays are read-only views of the one decompressed bytes object instead of copies made under the GIL.
def encode_msg(obj, compress=True, level=1):
    if not compress:
        return _frame(pickle.dumps(obj, protocol=5), 0)
    buffers = []
    head = pickle.dumps(obj, protocol=5, buffer_callback=buffers.append)
    raws = [b.raw() for b in buffers]
    index = struct.pack(f'!QI{len(raws)}Q', len(head), len(raws), *(r.nbytes for r in raws))
    return _frame(_compress(b''.join([index, head, *raws]), level), 2)


def _frame(body, kind):
    return struct.pack('!QB', len(body), kind) + body


def decode_body(body, kind):
    if kind == 0:
        return pickle.loads(body)
    raw = _decompress(body)
    return _unpickle(raw, kind)


def _unpickle(raw, kind):
    if kind == 1:
        return pickle.loads(raw)
    view = memoryview(raw)
    n_head, n_buf = struct.unpack_from('!QI', view, 0)
    sizes = struct.unpack_from(f'!{n_buf}Q', view, 12)
    at = 12 + 8 * n_buf
    head, at = view[at:at + n_head], at + n_head
    buffers = []
    for size in sizes:
        buffers.append(view[at:at + size])
        at += size
    return pickle.loads(head, buffers=buffers)


def send_msg(sock, obj, *, compress=True, level=1):
    """Returns the bytes put on the wire. level: zstd level of a compressed frame."""
    frame = encode_msg(obj, compress, level)
    sock.sendall(frame)
    return len(frame)


_WAITALL = getattr(socket, 'MSG_WAITALL', 0)


def _recv_exact(sock, n):
    """n bytes into one buffer. MSG_WAITALL lets a blocking socket wait in the kernel for the whole frame: one GIL
    release per message instead of one per network chunk (~64 KB), which with ~900 receiver threads starved the
    learner's assembler thread (hero runs, 2026-10-28). Sockets with a timeout are non-blocking underneath and
    may return early; the loop covers that."""
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got, _WAITALL)
        if not k:
            raise ConnectionError('peer closed')
        got += k
    return buf


def recv_msg(sock, clock=None):
    """clock (optional dict): adds the seconds spent in zstd (releases the GIL) and pickle (holds it), and the
    frame's bytes to clock['bytes'] when that key is present."""
    size, kind = struct.unpack('!QB', _recv_exact(sock, 9))
    body = _recv_exact(sock, size)
    if clock is not None and 'bytes' in clock:
        clock['bytes'] += 9 + size
    if clock is None or not kind:
        return decode_body(body, kind)
    t = time.perf_counter()
    raw = _decompress(body)
    u = time.perf_counter()
    msg = _unpickle(raw, kind)
    clock['decompress'] += u - t
    clock['unpickle'] += time.perf_counter() - u
    return msg


# ----------------------------------------------------------------------------- config

@dataclass
class LearnerConfig:
    size: int = 14
    players: int = 2                # seats per game; 4 = four-seat free-for-all (docs/four-player-2026-10-24.md):
                                    # engine-rs only, rank rewards (also at the round cap), learner_values required
    gamma: float = .995
    gae_lambda: float = .99
    segment: int = 64               # learner decisions per seat segment
    batch: int = 131072             # samples per PPO update
    minibatch: int = 8192           # global minibatch (split across devices)
    grad_accum: int = 1             # each device's minibatch gradient is the mean over this many equal micro-batches
                                    # (the default loss is a mean of per-sample terms, so the step is the same; peak
                                    # memory drops ~grad_accum x). Four-seat critics need it: 4 views x ~155 channels
    epochs: int = 2
    learning_rate: float = 1e-4
    learning_rate_end: float = None # learning-rate decay: None = constant learning_rate. A number: the rate runs
    lr_decay_fraction: float = .1   # linearly from learning_rate to learning_rate_end over the last lr_decay_fraction
                                    # of training_seconds (a polish phase at the end of the run)
    entropy_coef: float = .005
    entropy_target: float = None    # adaptive entropy: None = fixed entropy_coef. A number: after every update the
    entropy_target_end: float = None  # coefficient moves in log space by entropy_adapt_rate * (target - entropy),
    entropy_adapt_rate: float = .05 # clipped to [1e-4, .05]; the target runs linearly from entropy_target to
                                    # entropy_target_end (default: constant) over training_seconds
    clip: float = .2
    target_kl: float = .03
    max_lag: int = 6                # drop samples from older policy versions
    publish_every: int = 2
    self_play_probability: float = .5
    historical_probability: float = .2
    historical_every: int = 25
    historical_keep: int = 8
    frozen: tuple = ()              # (name, path, probability)
    unit_kl_coef: float = 0.        # KL(pi || reference) on summon unit choice only
    unit_kl_reference: str = ''
    unit_entropy_coef: float = 0.   # bonus on the exact entropy of the unit-type choice at summon decisions
    turn_gamma: float = None        # turn clock: discount (and turn_lambda) only at the seat's own end-turn decisions,
    turn_lambda: float = None       # 1 within a turn; None = per-decision gamma/gae_lambda as before
    anchor_coef: float = 0.         # k3 KL(pi || anchor) on the taken actions (ppo.loss); anchor = anchor_path or parent
    anchor_path: str = ''
    cap_score_reward: float = 0.    # > 0: a game at the round cap ends with +/-this for the seat ahead on score (0 = bootstrap)
    pbrs_coef: float = 0.           # zero-sum potential shaping: per-step reward gamma*Phi(s') - Phi(s), Phi antisymmetric
    pbrs_anneal_updates: int = 0    # Phi scale = pbrs_coef * max(0, 1 - update / anneal_updates); 0 = constant
    omniscient_critic: bool = False # separate critic on the union of both seats' views (policy input unchanged)
    critic_path: str = ''           # omniscient critic initialised from this checkpoint (a parent's final-critic.npz)
                                    # instead of a fresh template, so a continuation keeps its parent's value function
    advantage_keep: float = 1.      # fraction of each batch kept for the update, largest |advantage| first (Generals.io: .25)
    ema_decay: float = 0.           # > 0: EMA of the policy is what checkpoints/latest.npz hold (raw in *-raw.npz)
    truncation_reward: float = None # None: bootstrap the critic at the turn cap; a number: terminal reward
                                    # for both seats (0 = the cap counts as a draw, as in evaluation)
    checkpoint_every: int = 10
    snapshot_every: int = 50
    training_seconds: int = 3600
    seed: int = 20261019
    send_samples: int = 512         # actor message size; smaller = fresher weights on actors
    map_seed_low: int = 200000
    map_seed_high: int = 300000     # dev evaluation starts at 300000, final at 400000
    maps: str = 'classic'           # map mix (polytopia_bot.maps.stream): 'classic' = today's mapgen.reset maps
    max_turns: int = None           # round cap; None = the default for the board size (ruleset.round_limit)
    critic_warmup: int = 0          # first N updates train only the critic-only parameters (critic_hidden, value);
                                    # every other gradient is zeroed, so the policy is exactly unchanged meanwhile
    advantage_floor: float = .05    # batch advantages are divided by max(std, floor). 18x18 raw std is ~0.024-0.04
                                    # (14x14: ~0.05-0.065), so .05 halves the policy gradient against the entropy bonus
    upgo_coef: float = 0.           # AlphaStar UPGO: an extra PPO-clipped surrogate on the upgoing-return advantage
                                    # G^U - V (upgo_returns), normalized like the GAE advantage. 0 = off: rows carry
                                    # no UPGO fields and the loss is exactly today's
    # Stage A actor throughput (docs/10x-training-plan-2026-10-21.md); each is exact except learner_values, which
    # scores GAE with the learner's current critic instead of the actor's (<= max_lag stale) copy
    learner_values: bool = True     # A1: actors run no critic; the learner computes values and GAE per segment
    # trusted_step, trusted_check_every, build_rows: knobs of the retired JAX actor engine; engine-rs validates every
    # action itself. Kept so older run configs still parse; they have no effect.
    trusted_step: bool = True
    trusted_check_every: int = 64
    build_rows: bool = True
    # Actor engine: engine-rs is the only one (the real game's rules, polytopia_bot.ruleset; the actor image builds the
    # polytopia_engine_rs wheel and an actor fails fast without it). The field stays for older run configs.
    engine: str = 'rust'
    pool_per_process: bool = True   # A4: a process hosts one pool opponent for all its lanes or none
    # A6 (docs/a6-batch-assembly-2026-10-22.md): with learner_values, each message's packed views go to the devices
    # as it is dequeued (DeviceRows) instead of one host concatenation + pageable put per batch. Exact: the training
    # batch and the batch statistics are bitwise those of the host path (False, kept as the reference).
    stream_views: bool = True
    assembly_threads: int = 4       # A6 host-page copy threads (0 = copy on the assembler thread)
    # Coded views (rl/view_codec.py): actors send each view stream as sparse row-to-row int16 deltas (~28x fewer
    # bytes than the dense views on 18x18 four seats) and the learner rebuilds the dense views on the device, so
    # the host never decompresses, copies or uploads dense views. Exact (lossless); needs learner_values.
    coded_views: bool = False
    # Four seats: rows per serial sub-chunk of the learner's value pass (the 4-view critic stem's memory). 1024 fits
    # an 80 GB H100; a B200 runs whole VALUE_CHUNK chunks (value_sub >= VALUE_CHUNK: no sub-chunking).
    value_sub: int = VALUE_SUB
    # Lever 4 (docs/levers-egress-2026-10-22.md): zstd level of the actors' sample uploads. Lossless; on real
    # mid-game messages level 7 is 150 vs 186 wire bytes per row at level 1 (-19% upload egress) for ~13 us more
    # actor CPU per row (~0.6% of a process); the learner's decompression is no slower.
    upload_zstd_level: int = 7

    def validate(self):
        if self.upgo_coef < 0:
            raise ValueError('upgo_coef must be >= 0')
        if self.batch % self.minibatch or self.minibatch < 1 or self.epochs < 1:
            raise ValueError('batch must be a multiple of minibatch')
        if self.grad_accum < 1 or self.minibatch % self.grad_accum:
            raise ValueError('minibatch must be a multiple of grad_accum')
        if self.grad_accum > 1 and (self.unit_entropy_coef or self.unit_kl_coef):
            raise ValueError('grad_accum: the unit entropy / KL terms are ratios of sums, not per-sample means')
        frozen_mass = sum(p for _, _, p in self.frozen)
        if not 0 <= self.self_play_probability <= 1 or self.self_play_probability + self.historical_probability + frozen_mass > 1 + 1e-9:
            raise ValueError('opponent probabilities exceed one')
        if bool(self.unit_kl_coef) != bool(self.unit_kl_reference):
            raise ValueError('unit KL needs coefficient and reference together')
        if not 0 < self.advantage_keep <= 1 or not 0 <= self.ema_decay < 1 or self.pbrs_coef < 0 or self.cap_score_reward < 0:
            raise ValueError('invalid advantage_keep / ema_decay / pbrs / cap score')
        if (self.turn_gamma is None) != (self.turn_lambda is None):
            raise ValueError('turn clock needs turn_gamma and turn_lambda together')
        if self.critic_path and not self.omniscient_critic:
            raise ValueError('critic_path needs omniscient_critic')
        if self.map_seed_high > 300000:
            raise ValueError('training maps must stay below the development evaluation seeds')
        if self.trusted_check_every < 0:
            raise ValueError('trusted_check_every must be >= 0')
        if self.players not in (2, 4):
            raise ValueError('players must be 2 or 4')
        if self.engine != 'rust':
            raise ValueError('engine-rs is the only actor engine (engine=rust); the JAX engine was retired 2026-10-27')
        if self.players > 2 and (self.engine != 'rust' or not self.learner_values or self.pbrs_coef or self.cap_score_reward
                                 or self.truncation_reward is not None):
            raise ValueError('four seats need the rust engine and learner_values, and use rank rewards at the cap '
                             '(no pbrs, cap_score_reward or truncation_reward)')


# ----------------------------------------------------------------------------- trajectories

def gae(values, bootstrap, last_reward, terminal, gamma, lam, end_turn=None, phi=None, phi_next=0.):
    """Single-seat sequence. Rewards are zero except a native outcome on the last decision.

    end_turn (bool per decision) switches to the turn clock: gamma and lam apply only on the step after an
    end-turn decision and are 1 inside a turn (semi-MDP discounting by turns, not by atomic actions).
    phi (potential per decision state, seat-signed) adds potential-based shaping g_t*phi_{t+1} - phi_t, with
    phi_next the potential after the last decision (0 at a terminal state)."""
    n = len(values)
    rewards, g, gl, next_values = _sequence_terms(values, bootstrap, last_reward, terminal, gamma, lam, end_turn,
                                                  phi, phi_next)
    delta = rewards + g * next_values - values
    adv = np.zeros(n, np.float32)
    running = 0.
    for t in range(n - 1, -1, -1):
        running = delta[t] + gl[t] * running
        adv[t] = running
    return adv, adv + values


def _sequence_terms(values, bootstrap, last_reward, terminal, gamma, lam, end_turn, phi, phi_next):
    """Per-decision rewards (outcome and shaping), discount g, trace gamma*lambda gl and next-state values of one
    seat sequence, shared by gae and upgo_returns (see gae for the clocks)."""
    n = len(values)
    rewards = np.zeros(n, np.float32)
    if terminal:
        rewards[-1] = last_reward
    if end_turn is None:
        g = np.full(n, gamma, np.float32)
        gl = np.full(n, gamma * lam, np.float32)
    else:
        end_turn = np.asarray(end_turn, bool)
        g = np.where(end_turn, gamma, 1.).astype(np.float32)
        gl = np.where(end_turn, gamma * lam, 1.).astype(np.float32)
    if phi is not None:
        phi = np.asarray(phi, np.float32)
        rewards = rewards + g * np.append(phi[1:], 0. if terminal else phi_next).astype(np.float32) - phi
    next_values = np.append(values[1:], 0. if terminal else bootstrap).astype(np.float32)
    return rewards, g, gl, next_values


def upgo_returns(values, bootstrap, last_reward, terminal, gamma, end_turn=None, phi=None, phi_next=0.):
    """Upgoing returns (UPGO, Vinyals et al. 2019, Nature, Methods) of a single-seat sequence:

        G_t = r_t + g_t * (G_{t+1}  if Q_{t+1} >= V_{t+1}  else  V_{t+1}),   Q_{t+1} = r_{t+1} + g_{t+1} V_{t+2},

    i.e. the return follows the trajectory while the taken action did at least as well as the critic expected
    and bootstraps from the critic otherwise. The last decision bootstraps the critic (the outcome at a terminal).
    V is whatever `values` holds (the omniscient critic's value under omniscient_critic). Same clocks and shaping
    as gae. Returns (G, follow) with follow[t] = G_t continued the trajectory rather than bootstrapping V_{t+1}
    (False for the last decision, which has no next action to judge)."""
    values = np.asarray(values, np.float32)
    rewards, g, _, next_values = _sequence_terms(values, bootstrap, last_reward, terminal, gamma, 1., end_turn,
                                                 phi, phi_next)
    n = len(values)
    q = rewards + g * next_values            # one-step estimate of Q(s_t, a_t)
    out = np.zeros(n, np.float32)
    follow = np.zeros(n, bool)
    out[-1] = q[-1]
    for t in range(n - 2, -1, -1):
        follow[t] = q[t + 1] >= values[t + 1]
        out[t] = rewards[t] + g[t] * (out[t + 1] if follow[t] else values[t + 1])
    return out, follow


def scheduled_learning_rate(base, end, decay_fraction, frac):
    """Learning rate at fraction frac of training_seconds: base until the last decay_fraction of the run, then
    linear to end at frac = 1. end None = constant base."""
    if end is None:
        return float(base)
    if frac >= 1. or (decay_fraction > 0 and frac - (1. - decay_fraction) >= decay_fraction):
        return float(end)
    if decay_fraction <= 0:
        return float(base)
    t = max((frac - (1. - decay_fraction)) / decay_fraction, 0.)
    return float(base + t * (end - base))


def adapt_entropy_coef(coef, entropy, target, rate, low=1e-4, high=.05):
    """One adaptive-entropy step: raise the entropy bonus while the policy is below the target entropy, lower it
    above, multiplicatively (log-space step rate * (target - entropy)), clipped to [low, high]."""
    return float(np.clip(coef * np.exp(rate * (target - entropy)), low, high))


def episode_outcome(rewards, terminated, truncated, cap_reward=None):
    """Seat rewards and terminal flag used for GAE at an episode end.

    With cap_reward set, the engine's turn cap ends the episode with that reward for both seats instead
    of a critic bootstrap; the observation carries the turn number, so this stays Markov. More than two seats:
    the engine's rank rewards end the episode at the cap too (the seats still in share their places).
    """
    rewards = np.asarray(rewards, np.float32)
    if len(rewards) > 2:
        return rewards, bool(terminated or truncated)
    if truncated and cap_reward is not None:
        return np.full(2, cap_reward, np.float32), True
    return rewards, bool(terminated)


def other_seats(seat, players):
    """0-based indices of the seats other than `seat` (1-based) in turn order after it."""
    return [(seat - 1 + k) % players for k in range(1, players)]


class SeatTrajectories:
    """Per (lane, seat) decision lists; emits GAE segments of fixed maximum length."""

    def __init__(self, lanes, segment, gamma, lam, turn_clock=False, shaping=False, upgo=False, players=2):
        """turn_clock: rows carry 'end_turn' and gamma/lam are per turn (see gae); the flag is dropped on emit.
        shaping: rows carry 'phi' (potential of the decision state for the acting seat); dropped on emit.
        upgo: emitted rows also carry 'upgo_advantage' (upgoing return minus value) and 'upgo_follow'."""
        self.segment, self.gamma, self.lam, self.turn_clock, self.shaping = segment, gamma, lam, turn_clock, shaping
        self.upgo, self.players = upgo, players
        self.rows = [[[] for _ in range(players)] for _ in range(lanes)]
        self.out = []

    def add(self, lane, seat, row):
        rows = self.rows[lane][seat - 1]
        rows.append(row)
        if len(rows) > self.segment:
            head = rows[:self.segment]
            nxt = rows[self.segment]
            self._emit(head, bootstrap=nxt['value'], reward=0., terminal=False, phi_next=nxt.get('phi', 0.))
            del rows[:self.segment]

    def finish(self, lane, rewards, terminated, bootstrap_values):
        """rewards/bootstrap_values are seat-indexed [players]."""
        for seat in range(1, self.players + 1):
            rows = self.rows[lane][seat - 1]
            if rows:
                self._emit(rows, bootstrap=float(bootstrap_values[seat - 1]), reward=float(rewards[seat - 1]),
                           terminal=bool(terminated))
            self.rows[lane][seat - 1] = []

    def drop(self, lane):
        self.rows[lane] = [[] for _ in range(self.players)]

    def _emit(self, rows, *, bootstrap, reward, terminal, phi_next=None):
        values = np.array([r['value'] for r in rows], np.float32)
        end_turn = np.array([r.pop('end_turn') for r in rows], bool) if self.turn_clock else None
        phi = np.array([r.pop('phi') for r in rows], np.float32) if self.shaping else None
        if phi is not None and phi_next is None:
            phi_next = float(phi[-1])   # bootstrap at a cut without the next state (truncation): no shaping step
        adv, ret = gae(values, bootstrap, reward, terminal, self.gamma, self.lam, end_turn, phi, phi_next or 0.)
        for r, a, g in zip(rows, adv, ret):
            r['advantage'], r['returns'] = a, g
        if self.upgo:
            up, follow = upgo_returns(values, bootstrap, reward, terminal, self.gamma, end_turn, phi, phi_next or 0.)
            for r, u, f in zip(rows, up - values, follow):
                r['upgo_advantage'], r['upgo_follow'] = np.float32(u), bool(f)
        self.out.extend(rows)

    def take(self):
        out, self.out = self.out, []
        return out


def stack_rows(rows):
    keys = rows[0].keys()
    return {k: np.stack([r[k] for r in rows]) for k in keys}


class SegmentCollector:
    """Actor-side counterpart of SeatTrajectories when the learner computes values (learner_values).

    Rows carry no value, advantage or returns. Each emitted segment is a group of rows tagged (seg, pos) plus one
    entry of a per-segment table: its boundary views (the next decision of the same seat at a segment cut, the
    final state's views at a truncation; zeros at a native ending), the terminal flag and outcome, and the
    shaping potential after the last decision. The learner scores the rows and the boundary with its current
    critic and runs segment_gae; the numbers equal SeatTrajectories' for the same values.
    """

    def __init__(self, lanes, segment, players=2):
        self.segment, self.players = segment, players
        self.rows = [[[] for _ in range(players)] for _ in range(lanes)]
        self.out, self.segments = [], []

    def add(self, lane, seat, row):
        rows = self.rows[lane][seat - 1]
        rows.append(row)
        if len(rows) > self.segment:
            head, nxt = rows[:self.segment], rows[self.segment]
            self._emit(head, boot=nxt, terminal=False, reward=0.)
            del rows[:self.segment]

    def finish(self, lane, rewards, terminated, final_views):
        """rewards: seat-indexed [players]; final_views: [players, width] packed public view per seat of the final
        state (the bootstrap at a truncation; unused at a native ending). The boundary's opp_view is the other
        seat's view, or with more seats the others' [players - 1, width] in turn order after the seat."""
        for seat in range(1, self.players + 1):
            rows = self.rows[lane][seat - 1]
            if rows:
                others = final_views[2 - seat] if self.players == 2 else final_views[other_seats(seat, self.players)]
                boot = None if terminated else dict(view=final_views[seat - 1], opp_view=others)
                self._emit(rows, boot=boot, terminal=bool(terminated), reward=float(rewards[seat - 1]))
            self.rows[lane][seat - 1] = []

    def drop(self, lane):
        self.rows[lane] = [[] for _ in range(self.players)]

    def _emit(self, rows, *, boot, terminal, reward):
        first = rows[0]
        seg = dict(terminal=terminal, reward=np.float32(reward), length=np.int32(len(rows)),
                   boot_view=boot['view'] if boot is not None else np.zeros_like(first['view']))
        if 'opp_view' in first:
            seg['boot_opp_view'] = boot['opp_view'] if boot is not None else np.zeros_like(first['opp_view'])
        if 'phi' in first:
            # SeatTrajectories: the next decision's potential at a cut; no shaping step at a finish
            seg['phi_next'] = np.float32(boot['phi'] if boot is not None and 'phi' in boot else rows[-1]['phi'])
        index = np.int32(len(self.segments))
        for pos, r in enumerate(rows):
            r['seg'], r['pos'] = index, np.int32(pos)
        self.segments.append(seg)
        self.out.extend(rows)

    def take(self):
        """(rows, segments) since the last take; rows' seg indexes this segments list."""
        out, segments = self.out, self.segments
        self.out, self.segments = [], []
        return out, segments


def segment_gae(values, seg, pos, length, terminal, reward, boot_values, gamma, lam, *, end_turn=None, phi=None,
                phi_next=None, upgo=False):
    """GAE of flat rows grouped into segments (SegmentCollector), vectorized over segments.

    values [R] per row, seg/pos [R] the row's segment and position, length/terminal/reward/boot_values [S] per
    segment (boot_values: the critic's value of the boundary, ignored when terminal). Equals gae() applied to
    each segment (and upgo_returns with upgo=True): same clocks (end_turn per row) and shaping (phi per row,
    phi_next per segment). Rows missing from the front of a segment (dropped as stale) do not change the
    others' numbers, since GAE at t reads only t and later. Returns (advantage, returns) per row, plus
    (upgo_advantage, upgo_follow) with upgo=True."""
    values = np.asarray(values, np.float32)
    seg, pos, length = np.asarray(seg), np.asarray(pos), np.asarray(length)
    n_seg, width = len(length), int(length.max()) if len(length) else 1
    rows = np.arange(n_seg)
    last = length - 1
    V = np.zeros((n_seg, width), np.float32)
    valid = np.zeros((n_seg, width), bool)
    V[seg, pos] = values
    valid[seg, pos] = True
    terminal = np.asarray(terminal, bool)
    next_v = np.zeros_like(V)
    next_v[:, :-1] = V[:, 1:]
    next_v[rows, last] = np.where(terminal, 0., np.asarray(boot_values, np.float32)).astype(np.float32)
    rewards = np.zeros_like(V)
    rewards[rows, last] = np.where(terminal, np.asarray(reward, np.float32), 0.).astype(np.float32)
    if end_turn is None:
        g = np.full_like(V, gamma)
        gl = np.full_like(V, gamma * lam)
    else:
        E = np.zeros((n_seg, width), bool)
        E[seg, pos] = np.asarray(end_turn, bool)
        g = np.where(E, gamma, 1.).astype(np.float32)
        gl = np.where(E, gamma * lam, 1.).astype(np.float32)
    if phi is not None:
        P = np.zeros_like(V)
        P[seg, pos] = np.asarray(phi, np.float32)
        P_next = np.zeros_like(V)
        P_next[:, :-1] = P[:, 1:]
        P_next[rows, last] = np.where(terminal, 0., np.asarray(phi_next, np.float32)).astype(np.float32)
        rewards = (rewards + g * P_next - P).astype(np.float32)
    delta = (rewards + g * next_v - V).astype(np.float32)
    A = np.zeros_like(V)
    running = np.zeros(n_seg, np.float32)
    for t in range(width - 1, -1, -1):
        running = np.where(valid[:, t], (delta[:, t] + gl[:, t] * running).astype(np.float32), np.float32(0.))
        A[:, t] = running
    adv = A[seg, pos]
    out = (adv, (adv + values).astype(np.float32))
    if upgo:
        q = (rewards + g * next_v).astype(np.float32)
        G = np.zeros_like(V)
        follow = np.zeros((n_seg, width), bool)
        G[rows, last] = q[rows, last]
        for t in range(width - 2, -1, -1):
            f = q[:, t + 1] >= V[:, t + 1]
            inner = t < last
            nxt = np.where(f, G[:, t + 1], V[:, t + 1])
            G[:, t] = np.where(inner, (rewards[:, t] + g[:, t] * nxt).astype(np.float32), G[:, t])
            follow[:, t] = f & inner & valid[:, t]
        out = out + ((G[seg, pos] - values).astype(np.float32), follow[seg, pos])
    return out


# Zero-sum potential: Phi_1 = -Phi_2 by construction (log ratios flip sign), 0 when the seats are level.
PBRS_TERMS = (('stars', 2.), ('units', 1.6), ('territory', 1.3), ('cities', 1.5))  # (quantity, ratio for +-1)


def potentials(states, seat, scale):
    """Phi for `seat` (1/2 per lane) of batched engine states: scale * sum 0.3*clip(log((x_s+1)/(x_o+1))/log k, -1, 1)."""
    stars = np.asarray(states.players.stars, np.float32)                                   # [G, 2]
    def counts(owner):
        owner = np.asarray(owner)
        return np.stack([(owner == 1).sum(1), (owner == 2).sum(1)], 1).astype(np.float32)  # [G, 2]
    quantities = dict(stars=stars, units=counts(states.units.owner), territory=counts(states.tiles.owner),
                      cities=counts(states.cities.owner))
    seat = np.asarray(seat)
    rows = np.arange(len(seat))
    phi = np.zeros(len(seat), np.float32)
    for name, k in PBRS_TERMS:
        x = quantities[name]
        own, other = x[rows, seat - 1], x[rows, 2 - seat]
        phi += .3 * np.clip(np.log((own + 1.) / (other + 1.)) / np.log(k), -1., 1.)
    return scale * phi


def feature_shapes(players=2):
    """(board channels, global features) of encode_view for games with `players` seats (board-size independent)."""
    import jax
    import jax.numpy as jnp
    from .cpu_engine import view_features, view_layout
    layout = view_layout(14, players)
    width = sum(int(np.prod(s)) for _, s in layout)
    shapes = jax.eval_shape(lambda: view_features(jnp.zeros(width, jnp.int16).at[0].set(1), layout, 14))
    return shapes.board.shape[-1], shapes.global_state.shape[-1]


def _critic_template(size, blocks=2, extra_blocks=0, width=64, architecture='normalized-v2', players=2,
                     global_pool=False):
    """Omniscient critic: a normalized-v2 trunk on the union of every seat's view (channels x players)."""
    import jax
    from . import network
    board, glob = feature_shapes(players)
    return network.init(jax.random.key(1), players * board, players * glob, size=size, width=width, blocks=blocks,
                        architecture=architecture, extra_blocks=extra_blocks, global_pool=global_pool)


def initial_critic(config, network, log=print):
    """The omniscient critic a learner starts from: None without one, a fresh template to warm up, or, with
    critic_path, a parent's saved critic (its final-critic.npz) loaded into that template. c6-credit (2026-10-21)
    showed that every continuation from a fresh critic first regresses below its parent for longer than an hour of
    training recovers, so continuations should load the parent's critic and skip or shorten the warm-up."""
    if not config.omniscient_critic:
        return None
    template = _critic_template(**network)
    if not config.critic_path:
        return template
    from .runner import load_checkpoint
    critic = load_checkpoint(config.critic_path, template)
    log(json.dumps(dict(stage='critic_loaded', path=str(config.critic_path))))
    return critic


def omniscient_features(own, opp):
    """Features for the omniscient critic: own view and the other seat's view concatenated along channels."""
    import jax.numpy as jnp
    from .observation import Features
    return Features(jnp.concatenate((own.board, opp.board), -1), jnp.concatenate((own.global_state, opp.global_state), -1))


def critic_features(own, opp_packed, layout, size):
    """One sample's omniscient-critic features from its own Features and the packed views of the other seats
    ([width] with two seats, [players - 1, width] or its flattening with more): own then each other seat in turn
    order, concatenated along channels. Two seats: exactly omniscient_features(own, view_features(opp))."""
    import jax
    import jax.numpy as jnp
    from .cpu_engine import view_features
    from .observation import Features
    width = sum(int(np.prod(s)) for _, s in layout)
    others = jax.vmap(lambda a: view_features(a, layout, size))(opp_packed.reshape(-1, width))
    board = jnp.concatenate((own.board, *(others.board[k] for k in range(others.board.shape[0]))), -1)
    glob = jnp.concatenate((own.global_state, others.global_state.reshape(-1)), -1)
    return Features(board, glob)


# ----------------------------------------------------------------------------- actor

def _env_setup():
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
        ' --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 ' +
        os.environ.get('EXTRA_XLA_FLAGS', '')).strip()


# ----------------------------------------------------------------------------- weight relay

RELAY_WEIGHTS, RELAY_POOL = 'weights.pkl', 'pool.pkl'


def _write_atomic(path, obj):
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_bytes(pickle.dumps(obj, protocol=5))
    os.replace(tmp, path)


class WeightRelay:
    """One learner connection per actor container for the published weights and the opponent pool (lever 2,
    docs/levers-egress-2026-10-22.md). Without it the learner sent the same ~1 MB fp16 policy to every actor
    process every publish_every updates, ~2/3 of the overnight run's network egress.

    The relay long-polls the learner (the learner answers as soon as a newer version or pool is published, else
    after a few seconds with nothing) and writes what it gets to files in a container-local directory (/dev/shm)
    by atomic replace: weights.pkl = (session, version, weights, critic_weights), pool.pkl = (session, pool_version,
    pool_spec, {name: leaves}). The arrays are the learner's own transport copies, so processes read bit-identical
    weights. Processes keep their own connections for sample uploads; RelayReader fills in their replies."""

    def __init__(self, address, directory, resolve_address=None, connect_timeout=900):
        self.address, self.dir, self.resolve_address = tuple(address), Path(directory), resolve_address
        self.connect_timeout = connect_timeout
        self.stats = dict(fetches=0, weight_versions=0, pool_versions=0, bytes=0, reconnects=0, error=None)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=30)

    def _run(self):
        sock, failures = None, 0
        version, pool_version, pool, session = -1, -1, {}, None
        while not self.stopped.is_set():
            try:
                if sock is None:
                    sock = _connect(self.address, self.connect_timeout)
                send_msg(sock, dict(type='relay', version=version, pool_version=pool_version, have=sorted(pool)))
                clock = dict(decompress=0., unpickle=0., bytes=0)
                reply = recv_msg(sock, clock)
                failures = 0
                self.stats['fetches'] += 1
                self.stats['bytes'] += clock['bytes']
                if reply['session'] != session:   # a replaced learner: its versions restart, take everything anew
                    version, pool_version, pool, session = -1, -1, {}, reply['session']
                if reply.get('weights') is not None:
                    version = reply['version']
                    _write_atomic(self.dir / RELAY_WEIGHTS, (session, version, reply['weights'], reply.get('critic_weights')))
                    self.stats['weight_versions'] += 1
                if reply.get('pool_spec') is not None:
                    names = {n for n, _ in reply['pool_spec']}
                    pool = {**{n: leaves for n, leaves in pool.items() if n in names}, **reply['pool_params']}
                    pool_version = reply['pool_version']
                    _write_atomic(self.dir / RELAY_POOL, (session, pool_version, reply['pool_spec'], pool))
                    self.stats['pool_versions'] += 1
                if reply.get('stop'):
                    return
            except (OSError, ConnectionError, EOFError) as error:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                sock, failures = None, failures + 1
                self.stats['reconnects'] += 1
                if failures > 12:
                    self.stats['error'] = repr(error)
                    return
                time.sleep(min(30, 2 ** min(failures, 5)))
                if self.resolve_address is not None:
                    fresh = self.resolve_address()
                    if fresh is None:
                        return
                    self.address = tuple(fresh)
            except Exception as error:   # surfaced in the container summary; processes time out waiting
                self.stats['error'] = repr(error)
                return


class RelayReader:
    """Process side of WeightRelay: fills a reply from a relayed connection (which carries the version, the session
    and, when the pool changed, the pool spec and version, but no arrays) with the relay's weights and pool. It waits
    until the relay holds at least the version the learner announced, so a process adopts a new policy at the same
    reply as on the direct path (or a newer one, if the relay already has it)."""

    def __init__(self, directory, timeout=600):
        self.dir, self.timeout = Path(directory), timeout

    def _read(self, name, session, at_least):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                item = pickle.loads((self.dir / name).read_bytes())
            except FileNotFoundError:
                item = None
            if item is not None and item[0] == session and item[1] >= at_least:
                return item
            if time.monotonic() > deadline:
                raise TimeoutError(f'relay has no {name} of version >= {at_least} after {self.timeout} s')
            time.sleep(.02)

    def complete(self, reply, known_version=-1, have=()):
        """known_version: the version this process already runs; have: pool names it already holds."""
        reply = dict(reply)
        if reply.get('weights') is None and reply.get('version', -1) > known_version:
            _, version, weights, critic_weights = self._read(RELAY_WEIGHTS, reply['session'], reply['version'])
            reply.update(version=version, weights=weights, critic_weights=critic_weights)
        if reply.get('pool_spec') is not None and reply.get('pool_params') is None:
            _, pool_version, spec, params = self._read(RELAY_POOL, reply['session'], reply['pool_version'])
            reply.update(pool_version=pool_version, pool_spec=spec,
                         pool_params={n: leaves for n, leaves in params.items() if n not in set(have)})
        return reply


def actor_process(address, actor_id, games, stop_event=None, *, report_every=30., fast=True, connect_timeout=900,
                  progress=None, core=None, relay_dir=None, inference=None):
    """One actor process; failures are reported through the optional progress queue.

    core pins the process to one CPU: XLA's CPU thread pools otherwise oversubscribe a
    container running one process per core (measured 1.7x slower unpinned).
    relay_dir: the container's WeightRelay directory (weights and pool come from there, not the learner).
    inference: a pipe to the node's inference server (inference_server.serve); None runs the policy here.
    """
    if core is not None:
        os.sched_setaffinity(0, {core})
    try:
        return _actor_process(address, actor_id, games, stop_event, report_every=report_every, fast=fast,
                              connect_timeout=connect_timeout, progress=progress, relay_dir=relay_dir,
                              inference=inference)
    except BaseException:
        import traceback
        if progress is not None:
            progress.put(('error', actor_id, traceback.format_exc()))
        raise


def _actor_process(address, actor_id, games, stop_event=None, *, report_every=30., fast=True, connect_timeout=900,
                   progress=None, relay_dir=None, inference=None):
    """One actor process: games stepped on CPU, local or served inference, streamed segments."""
    note = (lambda *event: progress.put(event)) if progress is not None else (lambda *event: None)
    started = time.monotonic()
    _env_setup()
    import jax
    import jax.numpy as jnp
    from . import network
    from .actions import Choice
    from .rust_engine import RustGames   # ImportError here means the actor image lacks the engine-rs wheel

    relay = RelayReader(relay_dir) if relay_dir is not None else None
    sock = _connect(address, connect_timeout)
    note('connected', actor_id, time.monotonic() - started)
    send_msg(sock, dict(type='hello', actor=actor_id, have=[], relayed=relay is not None))
    hello = recv_msg(sock)
    if relay is not None:
        hello = relay.complete(hello)
    cfg = hello['config']
    size, segment = cfg['size'], cfg['segment']
    players = int(cfg.get('players', 2))
    rng = np.random.default_rng((cfg['seed'], actor_id))
    map_mix = cfg.get('maps', 'classic')
    source = None
    if map_mix != 'classic':
        from polytopia_bot.maps.stream import MapSource
        from polytopia_bot.maps.state import stack as stack_maps
        source = MapSource(map_mix, size, players=players)
    omni = bool(cfg.get('omniscient_critic'))
    learner_values = bool(cfg.get('learner_values', False))   # A1: no critic here, the learner scores the rows
    engine = RustGames(size, games, map_batch=source is not None, max_turns=cfg.get('max_turns'), both_views=omni,
                       players=players, served=inference is not None)   # served: packed masks, no encode here
    note('engine', actor_id, 'rust')
    note('compiled', actor_id, time.monotonic() - started)
    spec = cfg.get('network', dict(size=size))
    structure = jax.tree.structure(_template_params(**spec))
    to_f32 = lambda leaves: [jnp.asarray(a, jnp.float32 if a.dtype == np.float16 else a.dtype) for a in leaves]
    unflatten = lambda leaves: structure.unflatten(to_f32(leaves))
    params, version = unflatten(hello['weights']), hello['version']
    critic = None
    if omni and not learner_values:
        from .cpu_engine import view_features
        cstructure = jax.tree.structure(_critic_template(**spec))
        unflatten_critic = lambda leaves: cstructure.unflatten(to_f32(leaves))
        critic = unflatten_critic(hello['critic_weights'])
        opp_features = jax.jit(jax.vmap(lambda v: view_features(v, engine.layout, size)))
        critic_value = jax.jit(jax.vmap(lambda c, x: network.trunk(c, x)[2], in_axes=(None, 0)))
    shaping = float(cfg.get('pbrs_coef', 0.)) > 0
    pbrs_scale = float(cfg.get('pbrs_scale', 0.))
    pool = {name: unflatten(leaves) for name, leaves in hello['pool_params'].items()}
    pool_spec = hello['pool_spec']
    client = None
    if inference is not None:
        if critic is not None or bool(cfg.get('pbrs_coef', 0.)):
            raise ValueError('served inference needs learner_values (no actor-side critic) and no PBRS')
        from .inference_server import InferenceClient
        client = InferenceClient(inference, size, games, players)
        client.publish('learner', version, hello['weights'], spec)
        for name, leaves in hello['pool_params'].items():
            client.publish(name, 0, leaves, spec)

    act = jax.jit(jax.vmap(lambda p, x, m, k: network.sample(p, x, m, k, size=size), in_axes=(None, 0, 0, 0)))
    value = jax.jit(jax.vmap(lambda p, x: network.trunk(p, x)[2], in_axes=(None, 0)))

    @jax.jit
    def act_subset(p, features, masks, index, key):
        """Policy forward on the gathered lanes only (index padded to a bucket size)."""
        take = lambda a: jnp.take(a, index, axis=0)
        return act(p, jax.tree.map(take, features), jax.tree.map(take, masks), jax.random.split(key, index.shape[0]))
    buckets = sorted({b for b in (4, 8, 12, 16, 24, 32, 48, 64) if b < games} | {games})

    def run_subset(p, lanes_idx, key):
        n = len(lanes_idx)
        b = next(x for x in buckets if x >= n)
        index = np.zeros(b, np.int32)
        index[:n] = lanes_idx
        out = act_subset(p, prepared.features, prepared.masks, jnp.asarray(index), key)
        return jax.tree.map(lambda a: np.asarray(a)[:n], out)

    def infer(groups, key):
        """Policy forwards for this tick. groups: {name: lane indices}, 'learner' = current weights, else a
        pool entry. Returns {name: (choice [3 arrays], packed (family, source, target) path masks, logp, value)}."""
        if client is not None:
            t = time.perf_counter()
            out = client.sample(groups, np.asarray(prepared.view), prepared.masks)
            stats['serve_s'] = stats.get('serve_s', 0.) + (time.perf_counter() - t)
            stats['serve_wait_s'] = stats.get('serve_wait_s', 0.) + client.wait_s
            stats['serve_pack_s'] = stats.get('serve_pack_s', 0.) + client.pack_s
            client.wait_s = client.pack_s = 0.
            return out
        out = {}
        for name, idx in groups.items():
            key, k = jax.random.split(key)
            choice, path, logp, values = run_subset(params if name == 'learner' else pool[name], idx, k)
            out[name] = (list(choice), tuple(np.packbits(a, axis=-1) for a in path), logp, values)
        return out

    def map_seeds(n):
        return rng.integers(cfg['map_seed_low'], cfg['map_seed_high'], n).astype(np.int32)

    # Each lane holds its next map, regenerated only after a game on it ends.
    lane_maps, lane_kind = [None] * games, ['classic'] * games
    def refill(lanes_idx):
        for i, seed in zip(lanes_idx, map_seeds(len(lanes_idx))):
            if source is None:
                lane_maps[i] = int(seed)
            else:
                lane_kind[i], layout = source.layout(int(seed))
                lane_maps[i] = (int(seed), layout)
    def next_maps():
        return np.array(lane_maps, np.int32) if source is None else stack_maps(lane_maps)
    refill(range(games))
    playing = list(lane_kind)
    states, prepared = engine.reset(next_maps())
    refill(range(games))
    # lane opponent: None = self-play (both seats learn), else pool name; learner seat 1/2
    per_process = bool(cfg.get('pool_per_process', False))   # A4: the whole process plays the pool opponent, or none
    p_self = max(0., 1. - sum(p for _, p in pool_spec))

    def draw_process_opponent():
        names = [n for n, _ in pool_spec if n in pool]
        probs = np.array([p for n, p in pool_spec if n in pool], np.float64)
        if not names or probs.sum() <= 0:
            return None
        return names[int(rng.choice(len(names), p=probs / probs.sum()))]

    def draw_mode():
        """The process's current pool opponent; with pool_per_process, None (all lanes self-play) with the
        self-play probability, so the time-averaged share of lanes per opponent is the same as per-lane draws."""
        if per_process and rng.random() < p_self:
            return None
        return draw_process_opponent()
    process_opponent = draw_mode()

    def assign():
        """(None, 0) = self-play (every seat learns), else (pool name, the learner's seat); the pool plays the rest."""
        if process_opponent is None or (not per_process and rng.random() < p_self):
            return None, 0
        return process_opponent, int(rng.integers(1, players + 1))
    lanes = [assign() for _ in range(games)]
    turn_clock = cfg.get('turn_gamma') is not None
    extras = dict(shaping=shaping, upgo=bool(cfg.get('upgo_coef', 0.)))
    if learner_values:
        traj = SegmentCollector(games, segment, players)
    elif turn_clock:
        traj = SeatTrajectories(games, segment, cfg['turn_gamma'], cfg['turn_lambda'], turn_clock=True, players=players,
                                **extras)
    else:
        traj = SeatTrajectories(games, segment, cfg['gamma'], cfg['gae_lambda'], players=players, **extras)

    outbox = queue.Queue(maxsize=2)
    shared = dict(version=version, params=None, pool=None, pool_spec=None, stop=False, error=None)
    stats = _new_actor_stats()

    connection = dict(sock=sock, version=version)   # version: the newest weights the sender has handed over

    upload_level = int(cfg.get('upload_zstd_level', 1))
    coded_views = bool(cfg.get('coded_views')) and learner_values
    if coded_views:
        from .view_codec import encode as encode_views

    def exchange(item):
        """Send one message; on a dropped tunnel connection reconnect (with backoff) and resend."""
        for attempt in range(8):
            try:
                send_msg(connection['sock'], item, level=upload_level)
                return recv_msg(connection['sock'])
            except (OSError, ConnectionError, EOFError):
                try:
                    connection['sock'].close()
                except OSError:
                    pass
                time.sleep(min(30, 2 ** attempt))
                connection['sock'] = _connect(address, 300)
                send_msg(connection['sock'], dict(type='hello', actor=actor_id, have=[], relayed=relay is not None))
                again = recv_msg(connection['sock'])
                if relay is not None:
                    again = relay.complete(again)
                shared['params'] = (again['version'], again['weights'], again.get('critic_weights'))
                connection['version'] = again['version']
                shared['pool'] = (again['pool_spec'], again['pool_params'])
                shared['pbrs_scale'] = again.get('config', {}).get('pbrs_scale')
                note('reconnected', actor_id, attempt)
        raise ConnectionError('learner unreachable after retries')

    def sender():
        try:
            while True:
                item = outbox.get()
                if item is None:
                    return
                reply = exchange(item)
                if relay is not None:
                    reply = relay.complete(reply, connection['version'], item['have'])
                if reply.get('weights') is not None:
                    connection['version'] = reply['version']
                    shared['params'] = (reply['version'], reply['weights'], reply.get('critic_weights'))
                if reply.get('pbrs_scale') is not None:
                    shared['pbrs_scale'] = reply['pbrs_scale']
                if reply.get('pool_params') is not None:
                    shared['pool'] = (reply['pool_spec'], reply['pool_params'])
                shared['stop'] = reply.get('stop', False)
        except Exception as error:  # surfaced by the main loop
            shared['error'] = repr(error)
            shared['stop'] = True
    thread = threading.Thread(target=sender, daemon=True)
    thread.start()

    key = jax.random.key(int(rng.integers(1 << 31)))
    pending, pending_segments = [], []
    last_report = time.monotonic()
    send_samples = cfg.get('send_samples', 2048)
    finished_since_rotation = 0
    ticks = 0
    while not shared['stop'] and not (stop_event is not None and stop_event.is_set()):
        if shared['params'] is not None:
            version, leaves, cleaves = shared['params']
            params, shared['params'] = unflatten(leaves), None
            if client is not None:
                client.publish('learner', version, leaves, spec)
            if critic is not None and cleaves is not None:
                critic = unflatten_critic(cleaves)
            stats['weight_updates'] += 1
        if shared.get('pbrs_scale') is not None:
            pbrs_scale, shared['pbrs_scale'] = float(shared['pbrs_scale']), None
        if shared['pool'] is not None:
            pool_spec, new = shared['pool']
            shared['pool'] = None
            for name, leaves in new.items():
                pool[name] = unflatten(leaves)
                if client is not None:
                    client.publish(name, 0, leaves, spec)
            for name in list(pool):
                if name not in {n for n, _ in pool_spec}:
                    del pool[name]
            p_self = max(0., 1. - sum(p for _, p in pool_spec))
            if process_opponent is not None and process_opponent not in pool:
                process_opponent = draw_mode()
            # Any lane whose opponent just left the pool (an evicted historical snapshot)
            # continues as self-play; previously only the process opponent was checked.
            lanes = [lane if lane[0] is None or lane[0] in pool else (None, 0) for lane in lanes]

        current = np.asarray(states.current_player)
        opponent_turn = np.array([name is not None and current[i] != seat for i, (name, seat) in enumerate(lanes)])
        key, k1 = jax.random.split(key)
        c_np = [np.zeros(games, np.int32) for _ in range(3)]
        mine = np.nonzero(~opponent_turn)[0]
        groups = {'learner': mine} if len(mine) else {}
        # Usually one pool opponent per process; lanes keep theirs until their game ends.
        for opp_name in sorted({lanes[i][0] for i in range(games) if opponent_turn[i]}):
            groups[opp_name] = np.nonzero(opponent_turn & np.array([lane[0] == opp_name for lane in lanes]))[0]
        forwards = infer(groups, k1)
        if len(mine):
            choice, (fam_np, src_np, tgt_np), logp, values = forwards['learner']
            for j in range(3):
                c_np[j][mine] = choice[j]
            if critic is not None:   # the critic, not the policy's value head, scores the states
                full = omniscient_features(prepared.features, opp_features(prepared.opp_view))
                values = np.asarray(critic_value(critic, full))[mine]
            if shaping:
                phi_now = potentials(states, current[mine], pbrs_scale)
            if omni:
                opp_view_np = np.asarray(prepared.opp_view)
        for opp_name, idx in groups.items():
            if opp_name != 'learner':
                for j in range(3):
                    c_np[j][idx] = forwards[opp_name][0][j]
        choice = Choice(*(jnp.asarray(a) for a in c_np))
        view_np = np.asarray(prepared.view)
        for r, i in enumerate(mine):
            row = dict(view=view_np[i], family_mask=fam_np[r], source_mask=src_np[r],
                       target_mask=tgt_np[r], choice=np.array([c_np[0][i], c_np[1][i], c_np[2][i]], np.int32),
                       logp=np.float32(logp[r]), version=np.int32(version))
            if not learner_values:
                row['value'] = np.float32(values[r])
            if turn_clock:
                row['end_turn'] = bool(c_np[0][i] == END_FAMILY)
            if shaping:
                row['phi'] = np.float32(phi_now[r])
            if omni:
                row['opp_view'] = opp_view_np[i]
            traj.add(i, int(current[i]), row)
            stats['learner_decisions'] += 1
            if c_np[0][i] == SUMMON_FAMILY:
                unit = int(c_np[2][i])
                stats['summons'][unit] = stats['summons'].get(unit, 0) + 1
        ticks += 1
        t_engine = time.perf_counter()
        tick, status = engine.tick(states, choice, next_maps())
        stats['engine_s'] = stats.get('engine_s', 0.) + (time.perf_counter() - t_engine)
        status = np.asarray(status)
        if np.any(status != 0):
            raise RuntimeError(f'illegal policy actions on lanes {np.nonzero(status)[0].tolist()}')
        stats['steps'] += games
        done = np.asarray(tick.done)
        if done.any():
            terminated, truncated = np.asarray(tick.terminated), np.asarray(tick.truncated)
            rewards = np.array(tick.rewards, np.float32)
            boot = np.zeros((games, players), np.float32)
            if learner_values:
                boot = np.asarray(tick.final_views)        # [games, 2, width]: the learner scores a truncation
            elif truncated.any():
                if omni:
                    f = tick.final_features                      # [games, seat, ...] own-view features per seat
                    seat = lambda j: jax.tree.map(lambda a: a[:, j], f)
                    boot = np.stack([np.asarray(critic_value(critic, omniscient_features(seat(0), seat(1)))),
                                     np.asarray(critic_value(critic, omniscient_features(seat(1), seat(0))))], 1)
                else:
                    flat = jax.tree.map(lambda a: a.reshape((-1,) + a.shape[2:]), tick.final_features)
                    boot = np.asarray(value(params, flat)).reshape(games, 2)
            cap_reward = cfg.get('truncation_reward')
            cap_score = float(cfg.get('cap_score_reward', 0.) or 0.)
            score = np.asarray(states.players.score)               # state before the final action
            final_turn = np.asarray(states.turn)
            out_seats = ((np.asarray(states.players.killed_turn) > 0) | (np.asarray(states.players.resigned_turn) > 0)).sum(1)
            for i in np.nonzero(done)[0]:
                name, seat = lanes[i]
                stats['finished'] = stats.get('finished', 0) + 1
                stats['turns'] = stats.get('turns', 0) + int(final_turn[i])
                # seats out before the final action (+1 for the elimination that ended a native finish)
                stats['seats_out'] = stats.get('seats_out', 0) + int(out_seats[i]) + int(terminated[i])
                if truncated[i] and cap_score > 0:
                    # the round cap resolves by score lead (zero-sum; a tie is a draw), ending the episode
                    lead = float(np.sign(score[i, 0] - score[i, 1]))
                    rewards[i], ended = np.array([lead, -lead], np.float32) * cap_score, True
                    stats['cap_score_results'] = stats.get('cap_score_results', 0) + 1
                else:
                    rewards[i], ended = episode_outcome(rewards[i], terminated[i], truncated[i], cap_reward)
                if name is None:
                    traj.finish(i, rewards[i], ended, boot[i])
                    stats['selfplay_games'] += 1
                    stats['selfplay_truncations'] += int(truncated[i])
                else:
                    # Record the learner seat only; drop any partial opponent-seat rows.
                    for other in other_seats(seat, players):
                        traj.rows[i][other] = []
                    traj.finish(i, rewards[i], ended, boot[i])
                    rec = stats['vs'].setdefault(name, [0, 0, 0])
                    # two seats: win / loss / cap; more: above / below the mean place / exactly at it
                    r = 0. if truncated[i] and players == 2 else rewards[i][seat - 1]
                    rec[0 if r > 0 else 1 if r < 0 else 2] += 1
                    if players > 2:
                        place = stats.setdefault('vs_reward', {}).setdefault(name, [0., 0])
                        place[0] += float(r)
                        place[1] += 1
                rec = stats['maps'].setdefault(playing[i], [0, 0])
                rec[0] += 1
                rec[1] += int(truncated[i])
                playing[i] = lane_kind[i]
                finished_since_rotation += 1
                if finished_since_rotation >= 4 * games:
                    process_opponent, finished_since_rotation = draw_mode(), 0
                lanes[i] = assign()
            refill([int(i) for i in np.nonzero(done)[0]])
        states, prepared = tick.states, tick.prepared
        if learner_values:
            rows, segments = traj.take()
            offset = np.int32(len(pending_segments))
            for r in rows:
                r['seg'] = r['seg'] + offset
            pending.extend(rows)
            pending_segments.extend(segments)
        else:
            pending.extend(traj.take())
        now = time.monotonic()
        if len(pending) >= send_samples or now - last_report > report_every:
            batch_rows = stack_rows(pending) if pending else None
            seg_rows = stack_rows(pending_segments) if pending_segments else None
            if coded_views and batch_rows is not None:   # sparse row-to-row view deltas (view_codec)
                batch_rows, seg_rows = encode_views(batch_rows, seg_rows)
            message = dict(type='data', actor=actor_id, have=sorted(pool), batch=batch_rows, stats=stats,
                           segments=seg_rows)
            note('sent', actor_id, len(pending), stats['steps'])
            outbox.put(message)
            pending, pending_segments, stats, last_report = [], [], _new_actor_stats(), now
    outbox.put(None)
    thread.join(timeout=30)
    connection['sock'].close()
    if client is not None:
        client.close()
    if shared['error']:
        raise RuntimeError(shared['error'])  # nonzero exit: the container may restart this process
    return None


def _new_actor_stats():
    return dict(steps=0, learner_decisions=0, selfplay_games=0, selfplay_truncations=0, vs={}, summons={},
                maps={}, weight_updates=0, started=time.time())


def _connect(address, timeout):
    host, port = address
    deadline = time.monotonic() + timeout
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=60)
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def _template_params(size, blocks=2, extra_blocks=0, width=64, architecture='normalized-v2', players=2,
                     global_pool=False):
    """Parameter tree of a normalized-v2 network; `size` only sets the parameter_embedding row count."""
    import jax
    from . import network
    # Feature widths do not depend on the board size; traced on 14x14.
    board, glob = feature_shapes(players)
    return network.init(jax.random.key(0), board, glob, size=size, width=width, blocks=blocks,
                        architecture=architecture, extra_blocks=extra_blocks, global_pool=global_pool)


def network_spec(path):
    """Architecture of a saved checkpoint, from its JSON config (older checkpoints have no extra_blocks, and are
    two-seat networks without global pooling)."""
    config = json.loads(Path(path).with_suffix('.json').read_text())['config']
    spec = dict(size=config['size'], blocks=config['blocks'], extra_blocks=config.get('extra_blocks', 0),
                width=config['width'], architecture=config['architecture'])
    if config.get('players', 2) != 2:
        spec['players'] = config['players']
    if config.get('global_pool'):
        spec['global_pool'] = True
    return spec


def checkpoint_template(path):
    return _template_params(**network_spec(path))


def _memory_gib(pids):
    """(cgroup memory in use or None, summed proportional set size of pids) in GiB. PSS splits shared pages (the
    JAX and engine libraries every process maps) between processes; VmRSS is the fallback where smaps_rollup is
    missing (gVisor), which counts shared pages once per process and so overstates."""
    cgroup = None
    for name in ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
        try:
            cgroup = int(Path(name).read_text()) / 2 ** 30
            break
        except (OSError, ValueError):
            pass
    total = 0
    for pid in pids:
        for name, key in (('smaps_rollup', 'Pss:'), ('status', 'VmRSS:')):
            try:
                line = next(l for l in Path(f'/proc/{pid}/{name}').read_text().splitlines() if l.startswith(key))
                total += int(line.split()[1]) * 1024
                break
            except (OSError, StopIteration, ValueError):
                pass
    return cgroup, total / 2 ** 30


def run_actor_container(address, processes, games, *, first_id=0, fast=True, report=None, report_every=20.,
                        should_stop=None, resolve_address=None, relay=True, inference=None, serve_options=None):
    """Spawn processes in one container; returns when all exit. report(dict) gets progress summaries.
    processes=0: one per CPU in the container's affinity set (a Modal cpu=16 container exposes 32), each pinned.
    resolve_address() is asked for the learner's current address before a crashed process is restarted (a
    replaced learner container publishes a new tunnel); None means the learner is gone and the container stops.
    relay: one WeightRelay connection fetches weights and pool for all processes (False: every process gets them
    on its own connection, the pre-lever-2 path).
    inference: None runs the policy in every actor process; 'cuda' (or 'cpu', for tests) starts one in-node
    inference server (inference_server.serve, kwargs serve_options) that batches every process's forwards. With a
    server, processes may exceed the cores (pinned round-robin): an actor waiting on its batch costs no CPU.
    serve_options may also hold `servers` (default 1: server processes sharing the GPU, each batching a contiguous
    share of the actors; the single server's Python loop capped hero-ffa4's containers) and `server_cores` (default
    0: servers unpinned; k: each server gets k cores of its own, which the actors are not pinned to, so 48 busy
    actor processes cannot starve a server of CPU)."""
    import multiprocessing as mp
    import shutil
    import tempfile
    ctx = mp.get_context('spawn')
    events = ctx.Queue()
    cores = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
    processes = processes or len(cores) or os.cpu_count()
    relay_dir = weight_relay = None
    if relay:
        relay_dir = tempfile.mkdtemp(prefix='polytopia-relay-', dir='/dev/shm' if os.path.isdir('/dev/shm') else None)
        weight_relay = WeightRelay(address, relay_dir, resolve_address).start()

    pipes = [ctx.Pipe() for _ in range(processes)] if inference else None
    servers = []
    if inference:
        from .inference_server import serve
        options = dict(serve_options or {})
        count, per = max(1, int(options.pop('servers', 1))), int(options.pop('server_cores', 0))
        if per and len(cores) >= count * per + 1:
            reserved, cores = [cores[k * per:(k + 1) * per] for k in range(count)], cores[count * per:]
        else:
            reserved = [None] * count
        share = -(-processes // count)
        for k in range(count):
            ends = [s for s, _ in pipes[k * share:(k + 1) * share]]
            if ends:
                servers.append(ctx.Process(target=serve, args=(ends,), daemon=True,
                                           kwargs=dict(options, platform=inference, cores=reserved[k],
                                                       seed=options.get('seed', 0) + k)))
                servers[-1].start()
        for s, _ in pipes:   # only the servers hold their ends: a dead server is an EOF to its actors, not a hang
            s.close()

    def spawn(i):
        return ctx.Process(target=actor_process, args=(address, first_id + i, games),
                           kwargs=dict(fast=fast, progress=events, core=cores[i % len(cores)] if cores else None,
                                       relay_dir=relay_dir, inference=pipes[i][1] if pipes else None), daemon=False)
    procs = [spawn(i) for i in range(processes)]
    for p in procs:
        p.start()
    summary = dict(processes=processes, connected=0, compiled=0, sent_samples=0, steps=0, errors=[],
                   first_compile_seconds=None, started=time.time(), restarts=0, reconnects=0, affinity=len(cores),
                   memory_gib=None, memory_peak_gib=None, pss_gib=None, pss_peak_gib=None)

    def sample_memory():
        cgroup, pss = _memory_gib([os.getpid()] + [p.pid for p in procs if p.pid is not None])
        summary['memory_gib'], summary['pss_gib'] = cgroup, pss
        if cgroup is not None:
            summary['memory_peak_gib'] = max(summary['memory_peak_gib'] or 0., cgroup)
        summary['pss_peak_gib'] = max(summary['pss_peak_gib'] or 0., pss)
    last = last_memory = 0.
    restarts = [0] * processes
    while any(p.is_alive() for p in procs):
        for i, p in enumerate(procs):
            # Exit code 0 means the learner said stop; anything else is a crash worth a bounded restart.
            if not p.is_alive() and p.exitcode not in (0, None) and restarts[i] < 20:
                if resolve_address is not None:
                    fresh = resolve_address()
                    if fresh is None:
                        for q in procs:
                            q.terminate()
                        restarts = [99] * processes
                        summary['learner_gone'] = True
                        break
                    if tuple(fresh) != tuple(address):
                        address = tuple(fresh)
                        summary['address_changes'] = summary.get('address_changes', 0) + 1
                restarts[i] += 1
                summary['restarts'] += 1
                procs[i] = spawn(i)
                procs[i].start()
        try:
            event = events.get(timeout=5)
            kind = event[0]
            if kind == 'connected':
                summary['connected'] += 1
            elif kind == 'compiled':
                summary['compiled'] += 1
            elif kind == 'engine':
                summary['engine'] = event[2]
                summary['first_compile_seconds'] = summary['first_compile_seconds'] or event[2]
            elif kind == 'sent':
                summary['sent_samples'] += event[2]
                summary['steps'] += event[3]
            elif kind == 'reconnected':
                summary['reconnects'] += 1
            elif kind == 'error':
                summary['errors'] = (summary['errors'] + [event[2][-2000:]])[-5:]
        except queue.Empty:
            pass
        if time.monotonic() - last_memory > 10:
            last_memory = time.monotonic()
            sample_memory()
        if report is not None and time.monotonic() - last > report_every:
            last = time.monotonic()
            if should_stop is not None and should_stop():
                for p in procs:
                    p.terminate()
                restarts = [99] * processes
            summary['alive'] = sum(p.is_alive() for p in procs)
            summary['elapsed'] = time.time() - summary['started']
            if weight_relay is not None:
                summary['relay'] = dict(weight_relay.stats)
            report(dict(summary))
    for p in procs:
        p.join()
    summary['exitcodes'] = [p.exitcode for p in procs]
    if servers:
        for _, c in pipes:   # terminated actors sent no 'close': the parent's ends are the servers' last writers
            c.close()
        for server in servers:
            server.join(timeout=60)
            if server.is_alive():
                server.terminate()
        summary['server_exitcode'] = next((s.exitcode for s in servers if s.exitcode), servers[0].exitcode)
    if weight_relay is not None:
        weight_relay.stop()
        summary['relay'] = dict(weight_relay.stats)
        shutil.rmtree(relay_dir, ignore_errors=True)
    if report is not None:
        report(dict(summary))
    return summary


# ----------------------------------------------------------------------------- learner

class CodedRowsMarker:
    """_score's streams argument for coded views: the batch's CodedRows."""

    def __init__(self, rows):
        self.rows = rows


class DeviceRows:
    """A6: rows of packed views staged in reusable host pages of VALUE_CHUNK rows and sent to the devices a page
    at a time as messages arrive.

    Page j holds rows [j*VALUE_CHUNK, (j+1)*VALUE_CHUNK) and lives on device j % D, as local page j // D of that
    device's buffer. Each message is copied into the current host page (a preallocated, already-touched array
    from the learner's pool, so no page faults), and a full page goes out as one async put plus one donated
    fixed-shape write into the device buffer: a batch costs ~2 dispatches per page instead of a fresh 8 GB host
    concatenation and one pageable put, and the upload overlaps the wait for the next messages (and the previous
    batch's training). Rows never written stay zero, as the host path's padding. finish(n_pad) returns the value
    pass's data-sharded arrays without a copy when the buffers have the final size: device i's shard is its pages
    in order, so every value-pass chunk is exactly one page, the same rows the host path's chunk held.
    position(rows, n_pad) maps rows to indexes in that layout (the identity on one device). release() returns the
    host pages to the pool once the device copies are done."""

    def __init__(self, learner, pages):
        self.learner, self.jax = learner, learner.jax
        self.devices = list(learner.mesh.devices.flat)
        self.pages = max(1, pages)      # local pages per device buffer; grown on demand
        self.buffers = {}               # key -> [one buffer per device]
        self.host, self.waiting, self.used = None, [], []  # page being filled; full pages awaiting copies; sent
        self.last = None                # finish()'s arrays, the last device writes
        self.rows = 0
        self.sends = []                 # page sends queued on the learner's send thread, in page order

    def _zeros(self, device, rows, like):
        import jax.numpy as jnp
        return jnp.zeros((rows,) + like.shape[1:], like.dtype, device=device)

    def _page(self, like):
        pool = self.learner.host_pages.setdefault((like.shape[1:], like.dtype.str), [])
        return pool.pop() if pool else np.zeros((VALUE_CHUNK,) + like.shape[1:], like.dtype)

    def add(self, arrays, keep=None):
        """arrays: {key: [m, ...] host rows}, the same m for every key; keep: the indexes of the rows to add (None:
        all). The copies into the host page, the gather of the kept rows included, run on the learner's copy
        threads (numpy copies and takes release the GIL); a full page is sent once its copies are done."""
        C, pool = VALUE_CHUNK, self.learner.copy_pool
        m = len(keep) if keep is not None else len(next(iter(arrays.values()))) if arrays else 0
        at = 0
        while at < m:
            j, off = divmod(self.rows, C)
            if self.host is None:
                self.host = dict(j=j, pages={k: self._page(a) for k, a in arrays.items()}, copies=[])
            take = min(m - at, C - off)
            for k, a in arrays.items():
                dst = self.host['pages'][k][off:off + take]
                if keep is None:
                    src = a[at:at + take]
                    if pool is None:
                        dst[...] = src
                    else:
                        self.host['copies'].append(pool.submit(np.copyto, dst, src))
                else:
                    idx = keep[at:at + take]   # mode='clip': valid indexes, and 'raise' would buffer the out copy
                    if pool is None:
                        np.take(a, idx, axis=0, out=dst, mode='clip')
                    else:
                        self.host['copies'].append(pool.submit(np.take, a, idx, 0, dst, 'clip'))
            at += take
            self.rows += take
            if off + take == C:
                self.waiting.append((self.host, C))
                self.host = None
        self._poll()

    def _poll(self, block=False):
        """Send the waiting full pages whose copies are done (all of them with block). With the learner's send
        thread the device puts run there, one at a time in page order, while this thread takes the next messages."""
        while self.waiting and (block or all(f.done() for f in self.waiting[0][0]['copies'])):
            self._submit(*self.waiting.pop(0))

    def _submit(self, page, filled):
        pool = self.learner.send_pool
        if pool is None:
            self._send(page, filled)
        else:
            self.sends.append(pool.submit(self._send, page, filled))

    def _drain(self):
        for f in self.sends:
            f.result()
        self.sends = []

    def _send(self, page, filled):
        """Host page j (rows [0, filled) valid) to device j % D, local page j // D."""
        jax, d, C = self.jax, len(self.devices), VALUE_CHUNK
        i, q = page['j'] % d, page['j'] // d
        for f in page['copies']:
            f.result()
        host = page['pages']
        keys = sorted(host)
        for k in keys:
            if filled < C:
                host[k][filled:] = 0
            if k not in self.buffers:
                self.buffers[k] = [self._zeros(dev, self.pages * C, host[k]) for dev in self.devices]
        if q >= self.pages:
            self._grow(q + 1)
        pieces = jax.device_put([host[k] for k in keys], self.devices[i])
        bufs = self.learner._write_rows(tuple(self.buffers[k][i] for k in keys), tuple(pieces), np.int32(q * C))
        for k, b in zip(keys, bufs):
            self.buffers[k][i] = b
        self.used.append(host)

    def release(self):
        """Return the host pages to the learner's pool (after the device writes that read them are done)."""
        self.jax.block_until_ready(self.last)
        for page in self.used:
            for a in page.values():
                self.learner.host_pages[(a.shape[1:], a.dtype.str)].append(a)
        self.used = []

    def _grow(self, pages):
        import jax.numpy as jnp
        for k, bufs in self.buffers.items():
            self.buffers[k] = [jnp.concatenate([b, self._zeros(dev, (pages - self.pages) * VALUE_CHUNK, b)])
                               for b, dev in zip(bufs, self.devices)]
        self.pages = pages

    def finish(self, n_pad):
        """{key: [n_pad, ...] data-sharded}; n_pad is a multiple of VALUE_CHUNK * devices."""
        jax, d = self.jax, len(self.devices)
        self._poll(block=True)
        if self.host is not None:
            self._submit(self.host, self.rows % VALUE_CHUNK)
            self.host = None
        self._drain()
        local = n_pad // d
        pages = local // VALUE_CHUNK
        if pages > self.pages:
            self._grow(pages)
        out = {}
        for k, bufs in self.buffers.items():
            if self.pages > pages:
                bufs = [b[:local] for b in bufs]
            out[k] = jax.make_array_from_single_device_arrays((n_pad,) + bufs[0].shape[1:], self.learner.data_sharding,
                                                              bufs)
        self.buffers, self.last = {}, out
        return out

    def position(self, rows, n_pad):
        d = len(self.devices)
        rows = np.asarray(rows, np.int64)
        if d == 1:
            return rows
        j, r = np.divmod(rows, VALUE_CHUNK)
        return (j % d) * (n_pad // d) + (j // d) * VALUE_CHUNK + r


class Learner:
    def __init__(self, output, config: LearnerConfig, parent, *, port=7070, commit=lambda: None,
                 parent_config=None, log=print, status=None):
        config.validate()
        self.config, self.root, self.commit, self.log = config, Path(output), commit, log
        self.status = status
        # Modal restarts a preempted learner container with the same inputs: tolerate an existing run folder
        # (the run restarts from the parent; its earlier files are kept) instead of dying with FileExistsError.
        restarted = self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'snapshots').mkdir(exist_ok=True)
        if restarted:
            log(json.dumps(dict(stage='learner_restarted_in_existing_folder', root=str(self.root))))
        import jax
        from .runner import load_checkpoint, Config
        from .cpu_engine import view_layout
        self.jax = jax
        self.layout = view_layout(config.size, config.players)
        self.network = network_spec(parent)
        if self.network.get('players', 2) != config.players:
            raise ValueError(f"parent is a {self.network.get('players', 2)}-seat network; config.players={config.players}")
        template = _template_params(**self.network)
        self.structure = jax.tree.structure(template)
        self.params = load_checkpoint(parent, template)
        self.parent = parent
        self.run_config = Config(**{**json.loads(Path(parent).with_suffix('.json').read_text())['config'],
                                    'learning_rate': config.learning_rate, 'entropy_coef': config.entropy_coef,
                                    'gamma': config.gamma, 'gae_lambda': config.gae_lambda,
                                    'minibatch': config.minibatch, 'epochs': config.epochs, 'seed': config.seed})
        from . import ppo
        self.opt = None  # built with the trainer: Adam over (policy, critic)
        self.reference = load_checkpoint(config.unit_kl_reference, template) if config.unit_kl_reference else None
        self.anchor = load_checkpoint(config.anchor_path or parent, template) if config.anchor_coef else None
        self.critic = initial_critic(config, self.network, log)  # fresh (warm it up) or the parent's via critic_path
        self.ema = None
        for name, path, _ in config.frozen:   # actors run pool opponents with the learner's parameter structure
            if network_spec(path) != self.network:
                raise ValueError(f'frozen opponent {name} is a {network_spec(path)} network; the learner is {self.network}')
        self.frozen = {name: self._leaves(load_checkpoint(path, template)) for name, path, _ in config.frozen}
        self.frozen_probability = {name: p for name, _, p in config.frozen}
        self.historical = []  # (name, leaves)
        self.version = 0
        self.learner_values = bool(config.learner_values)
        self.entropy_coef = float(config.entropy_coef)   # adapted per update when entropy_target is set
        # published weights and the pool change under publish_cond, which wakes the long-polling weight relays;
        # pool_version counts pool changes; session tells relays a replaced learner (whose versions restart)
        self.publish_cond = threading.Condition()
        self.published = (0, self._leaves(self.params), self._critic_leaves())
        self.pool_version = 0
        self.session = os.urandom(8).hex()
        self.inbox = queue.Queue(maxsize=128)
        self.port = port
        self.stop = False
        self.connections = 0
        self.relays = 0                                              # WeightRelay connections (one per container)
        self.wire = dict(weight_bytes=0, reply_bytes=0, received_bytes=0)   # per update, reset at each row
        self.lock = threading.Lock()
        self.actor_stats = []
        self.received_samples = 0
        self.receiver_seconds = dict(decompress=0., unpickle=0.)   # summed over receiver threads, reset per update
        self.row_pages, self.boot_pages = 1, 1                       # DeviceRows sizes, learned from the last batch
        self.host_pages = {}                                         # DeviceRows' reusable host pages by (shape, dtype)
        from concurrent.futures import ThreadPoolExecutor
        self.copy_pool = ThreadPoolExecutor(config.assembly_threads) if config.assembly_threads else None
        # DeviceRows' page uploads (device_put + buffer write), off the assembler thread; one thread keeps page order
        self.send_pool = ThreadPoolExecutor(1) if config.assembly_threads else None
        self.rng = np.random.default_rng(config.seed)
        self.resume_history, self.resume_elapsed = [], 0.
        self._build_trainer()
        if restarted and (self.root / 'latest.npz').exists():
            self._resume()

    # ---- pool / weights
    def _critic_leaves(self):
        """The omniscient critic for actors, or None: with learner_values no actor runs it."""
        return self._leaves(self.critic) if self.critic is not None and not self.learner_values else None

    def _leaves(self, params):
        """Transport copy in float16. Actors record logp under the weights they actually run,
        so the rounding changes the behaviour policy only, never PPO's ratio bookkeeping."""
        return [np.asarray(a).astype(np.float16) if np.issubdtype(np.asarray(a).dtype, np.floating) else np.asarray(a)
                for a in self.jax.tree.leaves(params)]

    def _resume(self):
        """Continue a run whose container was replaced: weights, Adam state, the omniscient critic, the update
        count, the training log and the self-play pool come back from latest.npz (+ latest-critic.npz and the
        snapshots), and the clock resumes at the last logged elapsed time, so training_seconds still bounds the
        run as a whole. Each checkpoint appends to training.jsonl before it saves latest.npz, so the log reaches at least
        latest's update (older runs: training.json, written together with latest.npz)."""
        jax = self.jax
        from . import ppo
        from .runner import load_checkpoint
        cfg = self.config
        template = _template_params(**self.network)
        latest = self.root / 'latest.npz'
        meta = json.loads(latest.with_suffix('.json').read_text())
        version = int(meta['stage'].rsplit('-', 1)[1])
        raw = self.root / 'latest-raw.npz' if self.ema is not None else latest
        params = load_checkpoint(raw, template)
        critic = load_checkpoint(self.root / 'latest-critic.npz', _critic_template(**self.network)) if self.critic is not None else None
        opt_template = ppo.adam_init((params, critic))
        opt_leaves, opt_structure = jax.tree.flatten(opt_template)
        with np.load(raw, allow_pickle=False) as archive:
            values = [archive[f'o{i}'] for i in range(len(opt_leaves))]
        if any(x.shape != y.shape for x, y in zip(opt_leaves, values, strict=True)):
            raise ValueError('optimizer state in latest.npz does not match the network')
        opt = opt_structure.unflatten([jax.numpy.asarray(v) for v in values])
        self.params = jax.device_put(params, self.replicated)
        if critic is not None:
            self.critic = jax.device_put(critic, self.replicated)
        self.opt = jax.device_put(opt, self.replicated)
        if self.ema is not None:
            self.ema = jax.device_put(load_checkpoint(latest, template), self.replicated)
        history = [row for row in training_log.read(self.root) if row['update'] <= version]
        if history and history[-1]['update'] != version:
            raise ValueError(f'training log ends at update {history[-1]["update"]}, latest.npz is update {version}')
        # The log may run past latest.npz (rows appended, then the container died before the save) or be an older
        # run's training.json: from here on training.jsonl holds exactly the rows up to the resumed update.
        training_log.rewrite(self.root / training_log.JSONL, history)
        self.resume_history = history
        self.resume_elapsed = float(history[-1]['elapsed']) if history else 0.
        self.version = version
        self.published = (version, self._leaves(self.params), self._critic_leaves())
        self.historical = []
        if cfg.historical_probability > 0:
            for u in range(cfg.historical_every, version + 1, cfg.historical_every):
                snap = self.root / 'snapshots' / f'update-{u}.npz'
                if snap.exists():
                    self.historical.append((f'hist-u{u}', self._leaves(load_checkpoint(snap, template))))
            self.historical = self.historical[-cfg.historical_keep:]
        self.log(json.dumps(dict(stage='learner_resumed', update=version, elapsed=self.resume_elapsed,
                                 pool=[n for n, _ in self.historical], critic=critic is not None)))

    def pool_spec(self):
        cfg = self.config
        spec = [(name, p) for name, p in self.frozen_probability.items()]
        if self.historical:
            each = cfg.historical_probability / len(self.historical)
            spec += [(name, each) for name, _ in self.historical]
        return spec

    def pool_params(self, names):
        everything = dict(self.frozen, **dict(self.historical))
        return {name: everything[name] for name, _ in self.pool_spec() if name not in names}

    def effective_config(self):
        cfg = asdict(self.config)
        cfg['network'] = self.network  # actors rebuild the parameter tree from this
        cfg['pbrs_scale'] = self.pbrs_scale()
        hist = self.config.historical_probability if self.historical else 0.
        cfg['self_play_probability'] = 1. - hist - sum(self.frozen_probability.values())
        return cfg

    def pbrs_scale(self):
        cfg = self.config
        if cfg.pbrs_coef <= 0:
            return 0.
        if cfg.pbrs_anneal_updates <= 0:
            return cfg.pbrs_coef
        return cfg.pbrs_coef * max(0., 1. - self.version / cfg.pbrs_anneal_updates)

    # ---- network service
    def serve(self):
        server = socket.socket(socket.AF_INET6 if ':' in self.bind_host else socket.AF_INET, socket.SOCK_STREAM)
        self.listening = True
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.bind_host, self.port))
        server.listen(4096)
        self.server = server
        def accept():
            while not self.stop:
                try:
                    conn, _ = server.accept()
                except OSError:
                    return
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        threading.Thread(target=accept, daemon=True).start()

    def _publish(self, published=None, historical=None):
        """Replace the published weights and/or the historical pool (bumping pool_version) and wake the relays."""
        with self.publish_cond:
            if published is not None:
                self.published = published
            if historical is not None:
                self.historical = historical
                self.pool_version += 1
            self.publish_cond.notify_all()

    def _published_state(self):
        with self.publish_cond:
            return self.published, self.pool_version, self.pool_spec()

    def _send(self, conn, reply):
        """Uncompressed reply; its bytes count as weight egress when it carries arrays (weights or pool)."""
        n = send_msg(conn, reply, compress=False)
        key = 'weight_bytes' if reply.get('weights') is not None or reply.get('pool_params') else 'reply_bytes'
        with self.lock:
            self.wire[key] += n

    def _handle(self, conn):
        """One connection: an actor process (hello, then data messages) or a container's WeightRelay. A process
        that says relayed=True in its hello gets no arrays on this connection: replies carry the version, the
        session and (when the pool changed) the pool spec and version; RelayReader fills in the rest."""
        with self.lock:
            self.connections += 1
        known_version, known_pool, relayed, relay = -1, set(), False, False
        clock = dict(decompress=0., unpickle=0., bytes=0)
        try:
            while True:
                msg = recv_msg(conn, clock)
                with self.lock:
                    self.wire['received_bytes'] += clock.pop('bytes')
                    for k, v in clock.items():
                        self.receiver_seconds[k] += v
                clock = dict(decompress=0., unpickle=0., bytes=0)
                if msg['type'] == 'relay':
                    if not relay:
                        relay = True
                        with self.lock:
                            self.relays += 1
                    known_v, known_pv = msg['version'], msg['pool_version']
                    with self.publish_cond:   # long poll: answer at the next publish or pool change
                        self.publish_cond.wait_for(lambda: self.published[0] > known_v or self.pool_version != known_pv
                                                   or self.stop, timeout=5)
                    (version, leaves, cleaves), pool_version, spec = self._published_state()
                    reply = dict(version=version, session=self.session, pool_version=pool_version, stop=self.stop)
                    if version > known_v:
                        reply['weights'], reply['critic_weights'] = leaves, cleaves
                    if pool_version != known_pv:
                        reply['pool_spec'], reply['pool_params'] = spec, self.pool_params(set(msg.get('have', ())))
                    self._send(conn, reply)
                    continue
                if msg['type'] == 'hello':
                    relayed = bool(msg.get('relayed'))
                    (version, leaves, cleaves), pool_version, spec = self._published_state()
                    reply = dict(config=self.effective_config(), version=version, session=self.session,
                                 pool_spec=spec, pool_version=pool_version)
                    if not relayed:
                        reply.update(weights=leaves, critic_weights=cleaves, pool_params=self.pool_params(set()))
                    self._send(conn, reply)
                    known_version, known_pool = version, {n for n, _ in spec}
                    continue
                if msg.get('batch') is not None:
                    self.inbox.put((msg['batch'], msg.get('segments')))  # blocks when the learner is behind
                with self.lock:
                    self.actor_stats.append(msg['stats'])
                (version, leaves, cleaves), pool_version, spec = self._published_state()
                reply = dict(version=version, stop=self.stop, pbrs_scale=self.pbrs_scale(), session=self.session)
                if version > known_version:
                    if not relayed:
                        reply['weights'], reply['critic_weights'] = leaves, cleaves
                    known_version = version
                names = {n for n, _ in spec}
                if names != known_pool:
                    reply['pool_spec'], reply['pool_version'] = spec, pool_version
                    if not relayed:
                        reply['pool_params'] = self.pool_params(set(msg.get('have', ())))
                    known_pool = names
                self._send(conn, reply)
        except (ConnectionError, OSError, EOFError):
            pass
        finally:
            with self.lock:
                self.connections -= 1
                self.relays -= int(relay)
            conn.close()

    # ---- training
    def _build_trainer(self):
        jax = self.jax
        import jax.numpy as jnp
        from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
        from . import network, ppo
        from .actions import Choice, PathMasks
        from .cpu_engine import view_features
        cfg = self.config
        devices = jax.devices()
        self.devices = len(devices)
        if cfg.minibatch % self.devices or cfg.batch % self.devices:
            raise ValueError('batch and minibatch must split evenly over devices')
        mesh = Mesh(np.array(devices), ('data',))
        self.mesh = mesh
        self.data_sharding = NamedSharding(mesh, P('data'))
        self.replicated = NamedSharding(mesh, P())
        size, layout = cfg.size, self.layout
        n = size * size
        local_mb = cfg.minibatch // self.devices
        accum = cfg.grad_accum
        if local_mb % accum:
            raise ValueError('the per-device minibatch must be a multiple of grad_accum')
        micro_mb = local_mb // accum
        omni = self.critic is not None

        def unpack_bits(a, width):
            bits = jnp.unpackbits(a, axis=-1)
            return bits[..., :width].astype(bool)

        def to_batch(mb):
            features = jax.vmap(lambda v: view_features(v, layout, size))(mb['view'])
            masks = PathMasks(unpack_bits(mb['family_mask'], 17), unpack_bits(mb['source_mask'], n),
                              unpack_bits(mb['target_mask'], max(n, 128)))
            choice = Choice(mb['choice'][:, 0], mb['choice'][:, 1], mb['choice'][:, 2])
            batch = dict(features=features, masks=masks, choice=choice, logp=mb['logp'],
                         advantage=mb['advantage'], returns=mb['returns'])
            if 'upgo_advantage' in mb:
                batch['upgo_advantage'] = mb['upgo_advantage']
            if omni:
                batch['full_features'] = jax.vmap(lambda f, o: critic_features(f, o, layout, size))(features, mb['opp_view'])
            return batch

        def target_log_distribution(params, features, choice, tmask):
            tiles, g, _ = network.trunk(params, features)
            logits = network.target_logits(params, tiles, g, choice.family, choice.source)
            return network.log_distribution(logits, tmask)

        def loss(trainable, reference, anchor, mb, entropy_coef):
            params, critic = trainable
            batch = to_batch(mb)
            if anchor is not None:
                batch['anchor_logp'] = jax.lax.stop_gradient(jax.vmap(
                    lambda f, c, m: network.evaluate(anchor, f, c, m)[0])(batch['features'], batch['choice'], batch['masks']))
            total, metrics = ppo.loss(params, batch, clip=cfg.clip, entropy_coef=entropy_coef,
                                      anchor_coef=cfg.anchor_coef, value_coef=0. if omni else .5,
                                      upgo_coef=cfg.upgo_coef)
            if omni:
                # The policy's own value head is unused; the omniscient critic regresses the returns.
                v = jax.vmap(lambda x: network.trunk(critic, x)[2])(batch['full_features'])
                critic_loss = .5 * jnp.mean((v - batch['returns']) ** 2)
                total = total + .5 * critic_loss
                metrics = dict(metrics, value_loss=critic_loss, value_mean=v.mean(), value_std=v.std())
            if reference is not None or cfg.unit_entropy_coef:
                lp = jax.vmap(target_log_distribution, in_axes=(None, 0, 0, 0))(params, batch['features'], batch['choice'], batch['masks'].target)
                summon = (batch['choice'].family == SUMMON_FAMILY).astype(jnp.float32)
            if cfg.unit_entropy_coef:
                unit_entropy = -jnp.sum(jnp.where(batch['masks'].target, jnp.exp(lp) * lp, 0.), -1)
                unit_entropy = jnp.sum(unit_entropy * summon) / jnp.maximum(summon.sum(), 1.)
                total = total - cfg.unit_entropy_coef * unit_entropy
                metrics = dict(metrics, unit_entropy=unit_entropy)
            if reference is not None:
                lq = jax.lax.stop_gradient(jax.vmap(target_log_distribution, in_axes=(None, 0, 0, 0))(
                    reference, batch['features'], batch['choice'], batch['masks'].target))
                kl = jnp.sum(jnp.where(batch['masks'].target, jnp.exp(lp) * (lp - lq), 0.), -1)
                unit_kl = jnp.sum(kl * summon) / jnp.maximum(summon.sum(), 1.)
                total = total + cfg.unit_kl_coef * unit_kl
                metrics = dict(metrics, unit_kl=unit_kl, summon_fraction=summon.mean())
            return total, metrics

        critic_only = ('critic_hidden', 'value')

        def epoch(trainable, opt, reference, anchor, batch, key, policy_weight, entropy_coef, learning_rate):
            local = batch['logp'].shape[0]
            steps = local // local_mb          # advantage_keep < 1 shrinks the batch; shapes stay static per run
            perm = jax.random.permutation(jax.random.fold_in(key, jax.lax.axis_index('data')), local)
            def step(carry, i):
                trainable, opt = carry
                idx = jax.lax.dynamic_slice(perm, (i * local_mb,), (local_mb,))
                if accum == 1:
                    mb = jax.tree.map(lambda a: a[idx], batch)
                    (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(trainable, reference, anchor, mb,
                                                                                 entropy_coef)
                else:
                    def micro(acc, j):
                        part = jax.lax.dynamic_slice(idx, (j * micro_mb,), (micro_mb,))
                        mb = jax.tree.map(lambda a: a[part], batch)
                        (_, m), g = jax.value_and_grad(loss, has_aux=True)(trainable, reference, anchor, mb,
                                                                           entropy_coef)
                        return jax.tree.map(jnp.add, acc, g), m
                    zero = jax.tree.map(jnp.zeros_like, trainable)
                    grads, ms = jax.lax.scan(micro, zero, jnp.arange(accum))
                    grads = jax.tree.map(lambda g: g / accum, grads)
                    metrics = jax.tree.map(jnp.mean, ms)
                grads = jax.lax.pmean(grads, 'data')
                pgrads, cgrads = grads
                # critic warm-up: only the critic-side parameters move (the policy's own value head, or the
                # omniscient critic); everything the policy reads is exactly unchanged
                pgrads = {k: g if (k in critic_only and not omni) else jax.tree.map(lambda x: x * policy_weight, g)
                          for k, g in pgrads.items()}
                metrics = jax.lax.pmean(metrics, 'data')
                trainable, opt, norm = ppo.adam_step(trainable, (pgrads, cgrads), opt, learning_rate=learning_rate)
                return (trainable, opt), dict(metrics, grad_norm=norm)
            (trainable, opt), metrics = jax.lax.scan(step, (trainable, opt), jnp.arange(steps))
            return trainable, opt, jax.tree.map(jnp.mean, metrics)

        shard = jax.shard_map(epoch, mesh=mesh, in_specs=(P(), P(), P(), P(), P('data'), P(), P(), P(), P()),
                              out_specs=(P(), P(), P()), check_vma=False)
        jitted = jax.jit(shard)

        def train_epoch(trainable, opt, reference, anchor, batch, key, policy_weight, entropy_coef=None,
                        learning_rate=None):
            # entropy coefficient and learning rate are inputs, not compiled-in constants, so schedules can move them
            return jitted(trainable, opt, reference, anchor, batch, key, policy_weight,
                          np.float32(cfg.entropy_coef if entropy_coef is None else entropy_coef),
                          np.float32(cfg.learning_rate if learning_rate is None else learning_rate))
        self.train_epoch = train_epoch

        # learner_values: the critic (or the policy's value head) over every assembled row, in fixed chunks per
        # device; rows are padded to a multiple of VALUE_CHUNK * devices (_put_views)
        def local_values(net, view, opp_view):
            chunks = view.shape[0] // VALUE_CHUNK
            def one(xs):
                v, o = xs
                f = jax.vmap(lambda a: view_features(a, layout, size))(v)
                if omni:
                    f = jax.vmap(lambda a, b: critic_features(a, b, layout, size))(f, o)
                return jax.vmap(lambda x: network.trunk(net, x)[2])(f)
            sub = min(cfg.value_sub, VALUE_CHUNK)
            if cfg.players > 2 and sub < VALUE_CHUNK:
                # four seats: the critic's 4-view stem may need sub-chunks of value_sub rows to fit in memory
                whole = one
                def one(xs):
                    v, o = xs
                    k = v.shape[0] // sub
                    return jax.lax.map(whole, (v.reshape(k, sub, -1), o.reshape(k, sub, -1))).reshape(-1)
            opp = opp_view.reshape(chunks, VALUE_CHUNK, -1) if omni else view.reshape(chunks, VALUE_CHUNK, -1)
            return jax.lax.map(one, (view.reshape(chunks, VALUE_CHUNK, -1), opp)).reshape(-1)
        self._values = jax.jit(jax.shard_map(local_values, mesh=mesh, in_specs=(P(), P('data'), P('data')),
                                             out_specs=P('data'), check_vma=False))
        self._gather = jax.jit(lambda a, idx: a[idx], out_shardings=self.data_sharding)
        self._write_rows = jax.jit(lambda bufs, xs, at: tuple(
            jax.lax.dynamic_update_slice(b, x, (at,) + (0,) * (x.ndim - 1)) for b, x in zip(bufs, xs)), donate_argnums=0)
        self.params = jax.device_put(self.params, self.replicated)
        if self.critic is not None:
            self.critic = jax.device_put(self.critic, self.replicated)
        self.opt = jax.device_put(ppo.adam_init((self.params, self.critic)), self.replicated)
        if cfg.ema_decay:
            self.ema = self.params
        if self.reference is not None:
            self.reference = jax.device_put(self.reference, self.replicated)
        if self.anchor is not None:
            self.anchor = jax.device_put(self.anchor, self.replicated)

    def _assemble(self):
        """One training batch from the inbox: rows of whole messages (newest first in arrival order), the stale
        ones dropped, scored with the current critic under learner_values (values + GAE per segment; the views go
        to the devices once, here, and the training batch is gathered from them), cut to `batch` rows, advantages
        normalized. Returns (data, info); data holds host arrays of the training fields plus, under
        learner_values, the device views and the row index to gather. info's asm_*_seconds time the phases.
        With stream_views (A6) the views skip the host: DeviceRows writes each message's to the devices as it
        is dequeued, and only the small per-row fields are concatenated."""
        cfg = self.config
        started = time.monotonic()
        coded = self.learner_values and cfg.coded_views
        stream = self.learner_values and cfg.stream_views and not coded
        if stream:
            rows, bounds = DeviceRows(self, self.row_pages), DeviceRows(self, self.boot_pages)
        if coded:
            from .view_codec import CodedRows
            coded_rows = CodedRows(self)
        chunks, segments, have, stale, seg_offset = [], [], 0, 0, 0
        lag = []
        clock = dict(wait=0., stream=0.)
        messages = 0
        while have < cfg.batch:
            t = time.monotonic()
            try:
                batch, segs = self.inbox.get(timeout=30)
            except queue.Empty:
                clock['wait'] += time.monotonic() - t
                if time.monotonic() > self.deadline:
                    return None, None
                continue
            clock['wait'] += time.monotonic() - t
            messages += 1
            self.received_samples += len(batch['logp'])
            if coded:
                batch, segs = dict(batch), dict(segs) if segs is not None else None
                part = {k: batch.pop(k) for k in list(batch) if k.startswith('view_coo_') or k == 'view_shape'}
                bpart = {k: segs.pop(k) for k in list(segs) if k.startswith('boot_coo_') or k == 'boot_shape'}
                if 'view_shape' not in part:
                    raise ValueError('coded_views: an actor sent dense views (actor and learner configs differ)')
            fresh = batch['version'] >= self.version - cfg.max_lag
            if coded:
                coded_rows.add(part, batch['seg'], batch['pos'], fresh, bpart, segs['length'], segs['terminal'])
            stale += int((~fresh).sum())
            keep = None if fresh.all() else np.flatnonzero(fresh)
            if stream:   # the dense views go to DeviceRows, which gathers the kept rows on its copy threads
                views = {k: batch[k] for k in VIEW_KEYS if k in batch}
                batch = {k: v for k, v in batch.items() if k not in views}
            if keep is not None:
                # Versions only grow along a segment, so stale rows are a segment's oldest and dropping them
                # leaves the remaining rows' GAE unchanged (segment_gae reads t and later only).
                batch = {k: v[keep] for k, v in batch.items()}
            if len(batch['logp']):
                if 'seg' in batch:
                    batch['seg'] = batch['seg'] + np.int32(seg_offset)
                if stream:
                    t = time.monotonic()
                    rows.add(views, keep)
                    clock['stream'] += time.monotonic() - t
                chunks.append(batch)
                have += len(batch['logp'])
                lag.append(self.version - batch['version'])
            if segs is not None:
                if stream:
                    t = time.monotonic()
                    segs = dict(segs)
                    bounds.add({k: segs.pop(k) for k in ('boot_view', 'boot_opp_view') if k in segs})
                    clock['stream'] += time.monotonic() - t
                segments.append(segs)
                seg_offset += len(segs['length'])
        n = have
        t = time.monotonic()
        if self.learner_values:
            # pad to the value pass's chunking in the one concatenation (no second copy of the views)
            padded = -(-n // (VALUE_CHUNK * self.devices)) * VALUE_CHUNK * self.devices
            if padded > n:
                chunks.append({k: np.zeros((padded - n,) + v.shape[1:], v.dtype) for k, v in chunks[0].items()})
        data = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        lag = np.concatenate(lag)
        info = dict(stale_dropped=stale, mean_lag=float(lag.mean()), max_lag=int(lag.max()), messages=messages)
        views = None
        if self.learner_values:
            segs = {k: np.concatenate([c[k] for c in segments]) for k in segments[0]}
            clock['concat'] = time.monotonic() - t
            views = self._score(data, segs, n, info, clock, (rows, bounds) if stream else
                                CodedRowsMarker(coded_rows) if coded else None)
        else:
            clock['concat'] = time.monotonic() - t
        t = time.monotonic()
        keep = np.arange(cfg.batch)
        adv = data['advantage'][keep]
        data['advantage'] = ((adv - adv.mean()) / max(adv.std(), cfg.advantage_floor)).astype(np.float32)
        data['returns'] = data['returns'][keep]
        info.update(raw_advantage_std=float(adv.std()), return_mean=float(data['returns'].mean()))
        if 'upgo_advantage' in data:
            # UPGO advantage G^U - V, normalized like the GAE advantage; the follow flag is a statistic only
            u, follow = data['upgo_advantage'][keep], data.pop('upgo_follow')[keep]
            info.update(raw_upgo_std=float(u.std()), raw_upgo_mean=float(u.mean()),
                        upgo_follow_fraction=float(follow.mean()), upgo_positive_fraction=float((u > 0).mean()),
                        upgo_gae_corr=float(np.corrcoef(u, adv)[0, 1]) if u.std() > 0 and adv.std() > 0 else 0.)
            data['upgo_advantage'] = ((u - u.mean()) / max(u.std(), cfg.advantage_floor)).astype(np.float32)
        if cfg.advantage_keep < 1:
            # Generals.io-style: train on the largest-|advantage| fraction of the batch (a multiple of the minibatch)
            n_keep = max(cfg.minibatch, int(cfg.batch * cfg.advantage_keep) // cfg.minibatch * cfg.minibatch)
            idx = np.sort(np.argpartition(-np.abs(data['advantage']), n_keep - 1)[:n_keep])
            data['advantage'], data['returns'] = data['advantage'][idx], data['returns'][idx]
            if 'upgo_advantage' in data:
                data['upgo_advantage'] = data['upgo_advantage'][idx]
            keep = keep[idx]
            info['kept'] = n_keep
        host = {k: data[k][keep] for k in TRAIN_KEYS if k in data and k not in ('advantage', 'returns', 'upgo_advantage')}
        host.update({k: data[k] for k in ('advantage', 'returns', 'upgo_advantage') if k in data})
        if views is None:
            host.update({k: data[k][keep] for k in VIEW_KEYS if k in data})
        elif stream:
            keep = rows.position(keep, len(data['logp']))   # the gather reads the device layout
        clock['normalize'] = time.monotonic() - t
        info.update({f'asm_{k}_seconds': v for k, v in clock.items()}, asm_total_seconds=time.monotonic() - started)
        return dict(host=host, views=views, keep=keep), info

    def _put_views(self, rows, n_pad):
        """The packed views of host rows (already padded to a multiple of VALUE_CHUNK * devices) on the devices."""
        jax = self.jax
        out = {}
        for k in VIEW_KEYS:
            if rows.get(k) is not None:
                a = rows[k]
                if len(a) < n_pad:
                    a = np.concatenate([a, np.zeros((n_pad - len(a),) + a.shape[1:], a.dtype)])
                out[k] = jax.device_put(a, self.data_sharding)
        return out

    def _score(self, data, segs, n, info, clock, streams=None):
        """learner_values: values of every row and segment boundary from the current critic, then GAE per
        segment (segment_gae) into data['advantage'] / ['returns'] (+ UPGO). Returns the device views.
        streams: (rows, bounds) DeviceRows already holding the views (stream_views); else they are put here."""
        cfg = self.config
        net = self.critic if self.critic is not None else self.params
        t = time.monotonic()
        n_pad = len(data['logp'])
        n_seg = len(segs['length'])
        seg_pad = -(-n_seg // (VALUE_CHUNK * self.devices)) * VALUE_CHUNK * self.devices
        if isinstance(streams, CodedRowsMarker):
            views, boots = streams.rows.finish(n, n_pad, n_seg, seg_pad)
            row_at, seg_at = slice(0, n), slice(0, n_seg)
        elif streams is None:
            views = self._put_views(data, n_pad)
            boots = self._put_views(dict(view=segs['boot_view'], opp_view=segs.get('boot_opp_view')), seg_pad)
            row_at, seg_at = slice(0, n), slice(0, n_seg)
        else:
            rows, bounds = streams
            self.row_pages, self.boot_pages = rows.pages, bounds.pages
            views = rows.finish(n_pad)
            boots = {k[len('boot_'):]: v for k, v in bounds.finish(seg_pad).items()}
            row_at, seg_at = rows.position(np.arange(n), n_pad), bounds.position(np.arange(n_seg), seg_pad)
        if 'opp_view' not in views:
            boots.pop('opp_view', None)
        clock['views'] = time.monotonic() - t
        t = time.monotonic()
        values = np.asarray(self._values(net, views['view'], views.get('opp_view', views['view'])))[row_at]
        boot_values = np.asarray(self._values(net, boots['view'], boots.get('opp_view', boots['view'])))[seg_at]
        if streams is not None and not isinstance(streams, CodedRowsMarker):
            for stream in streams:
                stream.release()
        clock['value'] = time.monotonic() - t
        t = time.monotonic()
        turn_clock = cfg.turn_gamma is not None
        gamma, lam = (cfg.turn_gamma, cfg.turn_lambda) if turn_clock else (cfg.gamma, cfg.gae_lambda)
        out = segment_gae(values, data['seg'][:n], data['pos'][:n], segs['length'], segs['terminal'], segs['reward'],
                          boot_values, gamma, lam, end_turn=data['end_turn'][:n] if turn_clock else None,
                          phi=data['phi'][:n] if 'phi' in data else None, phi_next=segs.get('phi_next'),
                          upgo=cfg.upgo_coef > 0)
        pad = len(data['logp']) - n
        grow = lambda a: np.concatenate([a, np.zeros((pad,) + a.shape[1:], a.dtype)]) if pad else a
        data['advantage'], data['returns'] = grow(out[0]), grow(out[1])
        if cfg.upgo_coef > 0:
            data['upgo_advantage'], data['upgo_follow'] = grow(out[2]), grow(out[3])
        info.update(value_mean_batch=float(values.mean()), segments=int(n_seg),
                    terminal_fraction=float(np.asarray(segs['terminal']).mean()))
        clock['gae'] = time.monotonic() - t
        return views

    def _device_batch(self, data):
        """The training batch on the devices: host fields put, views gathered from the value pass's copy."""
        jax = self.jax
        batch = jax.device_put(data['host'], self.data_sharding)
        if data['views'] is not None:
            keep = jax.device_put(np.asarray(data['keep'], np.int32), self.replicated)
            for k, a in data['views'].items():
                batch[k] = self._gather(a, keep)
            data['views'] = None   # release the padded copy; the gathered batch is what trains
        return batch

    def synthetic_batch(self, n):
        """Shape-correct fake batch (all-legal masks) for learner throughput benchmarks."""
        size = self.config.size
        cells = size * size
        rng = np.random.default_rng(0)
        width = sum(int(np.prod(s)) for _, s in self.layout)
        others = () if self.config.players == 2 else (self.config.players - 1,)
        extra = dict(opp_view=np.zeros((n,) + others + (width,), np.int16)) if self.critic is not None else {}
        if self.config.upgo_coef:
            extra['upgo_advantage'] = rng.normal(size=n).astype(np.float32)
        return dict(view=np.zeros((n, width), np.int16), **extra,
                    family_mask=np.packbits(np.ones((n, 17), bool), axis=-1),
                    source_mask=np.packbits(np.ones((n, cells), bool), axis=-1),
                    target_mask=np.packbits(np.ones((n, max(cells, 128)), bool), axis=-1),
                    choice=np.zeros((n, 3), np.int32), logp=np.full(n, -3., np.float32),
                    value=np.zeros(n, np.float32), version=np.zeros(n, np.int32),
                    advantage=rng.normal(size=n).astype(np.float32), returns=rng.normal(size=n).astype(np.float32))

    def benchmark(self, epochs=4):
        jax = self.jax
        data = self.synthetic_batch(self.config.batch)
        data = {k: v for k, v in data.items() if k in TRAIN_KEYS + VIEW_KEYS}
        rows = []
        key = jax.random.key(0)
        for i in range(epochs):
            t = time.monotonic()
            batch = jax.device_put(data, self.data_sharding)
            jax.block_until_ready(batch)
            put = time.monotonic() - t
            (self.params, self.critic), self.opt, metrics = self.train_epoch(
                (self.params, self.critic), self.opt, self.reference, self.anchor, batch, key, np.float32(1.))
            jax.block_until_ready(self.params)
            rows.append(dict(epoch=i, put_seconds=put, epoch_seconds=time.monotonic() - t - put,
                             samples_per_second=self.config.batch / (time.monotonic() - t - put),
                             loss=float(metrics['loss'])))
            self.log(json.dumps(dict(stage='learner_benchmark', **rows[-1])))
        return rows

    def _aggregate_actor_stats(self):
        with self.lock:
            rows, self.actor_stats = self.actor_stats, []
        out = dict(steps=0, learner_decisions=0, selfplay_games=0, selfplay_truncations=0, vs={}, summons={})
        for r in rows:
            for k in ('steps', 'learner_decisions', 'selfplay_games', 'selfplay_truncations'):
                out[k] += r[k]
            for k in ('finished', 'turns', 'seats_out'):   # game length and eliminations of every finished game
                if k in r:
                    out[k] = out.get(k, 0) + r[k]
            for name, rec in r['vs'].items():
                agg = out['vs'].setdefault(name, [0, 0, 0])
                for j in range(3):
                    agg[j] += rec[j]
            for unit, c in r['summons'].items():
                out['summons'][int(unit)] = out['summons'].get(int(unit), 0) + c
            for kind, (n_games, n_trunc) in r.get('maps', {}).items():
                agg = out.setdefault('maps', {}).setdefault(kind, [0, 0])
                agg[0] += n_games
                agg[1] += n_trunc
            for name, (total_reward, n_games) in r.get('vs_reward', {}).items():   # four seats: mean place reward
                agg = out.setdefault('vs_reward', {}).setdefault(name, [0., 0])
                agg[0] += total_reward
                agg[1] += n_games
        total = sum(out['summons'].values())
        out['warrior_share'] = out['summons'].get(WARRIOR, 0) / total if total else None
        return out

    def snapshot(self):
        """Host copy of the training state at this update (a few MB), taken on the training thread so a background
        save writes exactly this update's state while training moves on."""
        jax = self.jax
        return dict(version=self.version, params=jax.device_get(self.params), opt=jax.device_get(self.opt),
                    critic=jax.device_get(self.critic) if self.critic is not None else None,
                    ema=jax.device_get(self.ema) if self.ema is not None else None)

    def save(self, name, snap=None):
        """<name>.npz is the deployed policy (the EMA when ema_decay is set, raw weights otherwise); the raw
        weights with their Adam state go to <name>-raw.npz when an EMA exists; an omniscient critic to <name>-critic.npz.
        snap: a snapshot() to write (default: the current state)."""
        from .runner import save_checkpoint
        snap = snap or self.snapshot()
        params, opt = snap['params'], snap['opt']
        stage = f'update-{snap["version"]}'
        if snap['ema'] is not None:
            save_checkpoint(self.root / f'{name}.npz', snap['ema'], self.run_config, stage=stage)
            save_checkpoint(self.root / f'{name}-raw.npz', params, self.run_config, stage=stage, optimizer=opt)
        else:
            save_checkpoint(self.root / f'{name}.npz', params, self.run_config, stage=stage, optimizer=opt)
        if snap['critic'] is not None:
            save_checkpoint(self.root / f'{name}-critic.npz', snap['critic'], self.run_config, stage=stage)

    # ---- bookkeeping off the training thread (lever 3, docs/levers-egress-2026-10-22.md)
    def _start_bookkeeper(self):
        """One background thread runs the status put (a Modal Dict write, ~0.14 s) and the checkpoints (training
        log append, saves, volume commit) in submission order. Status puts coalesce: only the newest row is sent.
        The queue is bounded, so a training thread that outruns its checkpoints waits instead of piling them up."""
        self.jobs, self.bookkeeping_error = queue.Queue(maxsize=4), None
        self._status_row = None

        def work():
            while True:
                job = self.jobs.get()
                try:
                    if job is None:
                        return
                    job()
                except Exception:
                    import traceback
                    self.bookkeeping_error = traceback.format_exc()
                    self.log(json.dumps(dict(stage='bookkeeping_error', error=self.bookkeeping_error[-2000:])))
                finally:
                    self.jobs.task_done()
        self.bookkeeper = threading.Thread(target=work, daemon=True)
        self.bookkeeper.start()

    def _post_status(self, row):
        with self.lock:
            queued, self._status_row = self._status_row is not None, row
        if not queued:
            self.jobs.put(self._push_status)

    def _push_status(self):
        with self.lock:
            row, self._status_row = self._status_row, None
        try:
            self.status(row)
        except Exception as error:
            self.log(json.dumps(dict(stage='status_error', error=repr(error))))

    def _post_checkpoint(self, rows, names):
        """Append rows to training.jsonl, save the snapshot under each name, then commit; latest.npz never runs
        ahead of the log, so a resume finds its update in the log."""
        snap = self.snapshot()
        def job():
            training_log.append(self.root / training_log.JSONL, rows)
            for name in names:
                self.save(name, snap)
            self.commit()
        self.jobs.put(job)

    def _drain_bookkeeper(self):
        self.jobs.put(None)
        self.bookkeeper.join()
        if self.bookkeeping_error:
            raise RuntimeError(f'bookkeeping failed: {self.bookkeeping_error}')

    def run(self, bind_host='::'):
        import jax
        cfg = self.config
        self.bind_host = bind_host
        from .runner import write_json
        write_json(self.root / 'protocol.json', dict(config=asdict(cfg), parent=self.parent, devices=self.devices,
            architecture='Duper-style: CPU actor processes (engine-rs, local inference) stream GAE '
                         'segments over TCP; learner node trains data-parallel PPO over every local GPU, '
                         'publishes weights every publish_every updates; samples older than max_lag versions dropped',
            engine='rust_engine.RustGames (engine-rs)', ruleset=RULESET_ID))
        self.serve()
        self.log(json.dumps(dict(stage='learner_listening', port=self.port, devices=self.devices)))
        history = list(self.resume_history)
        logged, bookkeeping = len(history), 0.   # rows already in training.jsonl; post-row seconds of the last update
        self._start_bookkeeper()
        started = time.monotonic() - self.resume_elapsed
        key = jax.random.key(cfg.seed + self.version)   # a resumed run must not replay the first updates' keys
        last = time.monotonic()
        received_at_last = 0
        self.deadline = started + cfg.training_seconds
        # Assembly (decompressed rows -> one fixed batch) overlaps GPU training.
        self.ready, self.assembler_error = queue.Queue(maxsize=1), None
        def assembler():
            try:
                while time.monotonic() < self.deadline:
                    data, info = self._assemble()
                    if data is None:
                        return
                    info['assembled_at'] = time.monotonic()
                    self.ready.put((data, info))
            except Exception:
                import traceback
                self.assembler_error = traceback.format_exc()
        threading.Thread(target=assembler, daemon=True).start()
        while time.monotonic() < self.deadline:
            tick = time.monotonic()
            item = None
            while item is None and time.monotonic() < self.deadline:
                try:
                    item = self.ready.get(timeout=10)
                except queue.Empty:
                    if self.assembler_error:
                        raise RuntimeError(self.assembler_error)
            if item is None:
                break
            data, info = item
            assembled = time.monotonic()
            info['ready_wait_seconds'] = assembled - tick
            info['batch_age_seconds'] = assembled - info.pop('assembled_at')
            if self.version == 0:
                self.log(json.dumps(dict(stage='first_batch_assembled', seconds=assembled - started)))
            batch = self._device_batch(data)
            info['device_batch_seconds'] = time.monotonic() - assembled
            metrics_all = []
            learning_rate = scheduled_learning_rate(cfg.learning_rate, cfg.learning_rate_end, cfg.lr_decay_fraction,
                                                    (assembled - started) / cfg.training_seconds)
            if cfg.learning_rate_end is not None:
                info['learning_rate'] = learning_rate
            for _ in range(cfg.epochs):
                key, sub = jax.random.split(key)
                weight = np.float32(0. if self.version < cfg.critic_warmup else 1.)
                (self.params, self.critic), self.opt, metrics = self.train_epoch(
                    (self.params, self.critic), self.opt, self.reference, self.anchor, batch, sub, weight,
                    self.entropy_coef, learning_rate)
                metrics = {k: float(v) for k, v in metrics.items()}
                metrics_all.append(metrics)
                if not all(np.isfinite(list(metrics.values()))):
                    raise RuntimeError(f'nonfinite PPO metrics {metrics}')
                if metrics['approx_kl'] > cfg.target_kl and self.version >= cfg.critic_warmup:
                    break
            trained = time.monotonic()
            if cfg.entropy_target is not None:
                # adaptive entropy: move the coefficient (log space) toward the scheduled target entropy
                frac = min(max((trained - started) / cfg.training_seconds, 0.), 1.)
                end = cfg.entropy_target if cfg.entropy_target_end is None else cfg.entropy_target_end
                target = cfg.entropy_target + frac * (end - cfg.entropy_target)
                entropy = float(np.mean([m['entropy'] for m in metrics_all]))
                self.entropy_coef = adapt_entropy_coef(self.entropy_coef, entropy, target, cfg.entropy_adapt_rate)
                info.update(entropy_coef=self.entropy_coef, entropy_target=target)
            self.version += 1
            if cfg.critic_warmup and self.version == cfg.critic_warmup:
                # Fresh Adam for the policy phase: the global step count advanced while policy moments stayed
                # zero, and bias correction would otherwise inflate the first policy steps ~3x.
                from . import ppo
                self.opt = jax.device_put(ppo.adam_init(jax.device_get((self.params, self.critic))), self.replicated)
                self.log(json.dumps(dict(stage='critic_warmup_done', update=self.version)))
            if self.ema is not None:
                d = cfg.ema_decay
                self.ema = jax.tree.map(lambda e, p: d * e + (1. - d) * p, self.ema, self.params)
            if self.version % cfg.publish_every == 0:
                self._publish(published=(self.version, self._leaves(self.params), self._critic_leaves()))
            if self.version % cfg.historical_every == 0 and cfg.historical_probability > 0:
                self._publish(historical=(self.historical + [(f'hist-u{self.version}', self._leaves(self.params))])[-cfg.historical_keep:])
            actors = self._aggregate_actor_stats()
            with self.lock:
                receiver, self.receiver_seconds = self.receiver_seconds, dict(decompress=0., unpickle=0.)
                wire, self.wire = self.wire, dict(weight_bytes=0, reply_bytes=0, received_bytes=0)
                relays = self.relays
            info.update({f'receiver_{k}_seconds': v for k, v in receiver.items()})
            now = time.monotonic()
            row = dict(update=self.version, samples=self.version * cfg.batch, seconds=now - tick,
                       assemble_seconds=assembled - tick, train_seconds=trained - assembled, epochs_run=len(metrics_all),
                       received_samples_per_second=(self.received_samples - received_at_last) / max(now - last, 1e-9),
                       connections=self.connections, relays=relays, inbox=self.inbox.qsize(), elapsed=now - started,
                       bookkeeping_seconds=bookkeeping, **wire,
                       **info, **{k: float(np.mean([m[k] for m in metrics_all])) for k in metrics_all[0]},
                       actors=actors)
            last, received_at_last = now, self.received_samples
            history.append(row)
            self.log(json.dumps(dict(stage='update', **row)))
            if self.status is not None:
                self._post_status(row)
            if self.version % cfg.checkpoint_every == 0:
                names = ['latest'] + ([f'snapshots/update-{self.version}'] if self.version % cfg.snapshot_every == 0 else [])
                self._post_checkpoint(history[logged:], names)
                logged = len(history)
            if self.bookkeeping_error:
                raise RuntimeError(f'bookkeeping failed: {self.bookkeeping_error}')
            bookkeeping = time.monotonic() - now   # this update's work after its row; logged in the next row
        with self.publish_cond:
            self.stop = True
            self.publish_cond.notify_all()
        if self.copy_pool is not None:   # idle copy threads would hold the container open; a new pool starts none
            from concurrent.futures import ThreadPoolExecutor
            self.copy_pool.shutdown()
            self.copy_pool = ThreadPoolExecutor(cfg.assembly_threads)
            self.send_pool.shutdown()
            self.send_pool = ThreadPoolExecutor(1)
        self._drain_bookkeeper()
        training_log.append(self.root / training_log.JSONL, history[logged:])
        write_json(self.root / training_log.JSON, history)   # once, for readers of finished runs
        self.save('final')
        write_json(self.root / 'training-complete.json', dict(updates=self.version, seconds=time.monotonic() - started))
        self.commit()
        time.sleep(5)
        return history
