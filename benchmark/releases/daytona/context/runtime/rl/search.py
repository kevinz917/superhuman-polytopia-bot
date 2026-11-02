"""Inference-time search over whole turns with a frozen policy/value network, on engine-rs.

A Polytopia turn is a sequence of sub-actions ending in END. The search unit is
one full turn: at the start of the searcher's turn the engine state is cloned
into K lanes, lane 0 plays the greedy policy (so K=1 is exactly the raw
deterministic bot) and lanes 1..K-1 sample the policy at a temperature. Each
lane is scored and the best lane's action list is replayed on the real game.

Scoring (depth d):
  d = 0  the critic's value at the decision where the lane's policy chose END,
         i.e. from the searcher's own view, on its own turn (the states the
         critic was trained on);
  d >= 1 after END both sides keep playing (rollout policy = the searcher's own
         network, greedy unless rollout_temperature > 0) for d opponent turns,
         and the lane is scored by the critic at the searcher's next decision.
A lane that ends the game is scored by the native outcome (+1/-1), and a lane
that hits the turn cap by 0 (no winner).

The game has fog of war and the lanes run on the true engine state, so search
sees consequences a player could not (what a move into fog reveals, the seeded
ruin reward, and with d >= 1 the rollout opponent plays from its true view).
TurnSearcher(determinize=True) removes the leak: lanes run on a world sampled
from the searcher's view (belief.py) and the plan is committed only up to the
first revealed difference (docs/test-time-search-2026-10-19.md).

The lanes are games of engine-rs (the real game's rules, polytopia_bot.ruleset)
stepped in lockstep from the host: each step reads every lane's legal masks and
packed view from Rust and samples all lanes in one jitted network call. States
passed in and returned are single-game NumPy States (gpu/state.py layout, no
game axis), as RustGames exports them. The JAX-engine search this replaces was
retired with the research rules on 2026-10-27.
"""
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import network
from .actions import Choice, family_mask, source_mask, target_mask

END_FAMILY = 15
NOOP = (1, 0, 0)   # an attack from a tile onto itself: always rejected, so a finished lane's game is left untouched


def _nets(params):
    """(policy params, value-scorer params or None). TurnSearcher(value_params=...) passes a (policy, scorer)
    pair; a plain parameter tree scores lanes with the policy network's own value head."""
    return params if isinstance(params, tuple) else (params, None)


class Plan(NamedTuple):
    actions: jax.Array      # [T, 3] chosen lane's sub-actions of the searcher's turn
    count: jax.Array        # number of valid rows in actions
    best: jax.Array         # chosen lane
    scores: jax.Array       # [K] lane scores (searcher's perspective)
    steps: jax.Array        # [K] simulated engine steps per lane
    complete: jax.Array     # [K] lane reached a score before the step budget ran out


def tempered_sample(params, features, masks, key, *, size, temperature, greedy, force=False, forced=(0, 0)):
    """network.sample with logits / temperature; greedy picks the argmax at every level.

    force=True fixes (family, source) to `forced`; the target is then drawn as usual."""
    tiles, g, value = network.trunk(params, features)
    keys = jax.random.split(key, 3)

    def draw(logits, mask, k):
        lp = network.log_distribution(logits, mask)
        sampled = jax.random.categorical(k, lp / temperature)
        return jnp.where(greedy, jnp.argmax(lp), sampled).astype(jnp.int32)

    f = jnp.where(force, forced[0], draw(network.dense(g, params['family']), family_mask(masks), keys[0]))
    s = jnp.where(force, forced[1], draw(network.dense(tiles, params['source'])[:, f], source_mask(masks, f, size), keys[1]))
    t = draw(network.target_logits(params, tiles, g, f, s), target_mask(masks, f, s, size), keys[2])
    return Choice(f, s, t), value


def root_first_actions(params, features, masks, key, *, size, lanes):
    """Gumbel-top-k over the joint (family, source) of the first sub-action, without replacement.

    Returns (family, source, joint log-prior, valid) per lane; lane 0 is the hierarchical argmax
    (the greedy policy's first sub-action), lanes 1.. are distinct Gumbel-top-(K-1) draws."""
    n = size * size
    tiles, g, _ = network.trunk(params, features)
    fm = family_mask(masks)
    lf = network.log_distribution(network.dense(g, params['family']), fm)
    src = network.dense(tiles, params['source'])
    families = jnp.arange(len(fm))
    sm = jax.vmap(lambda f: source_mask(masks, f, size))(families)
    ls = jax.vmap(lambda f: network.log_distribution(src[:, f], sm[f]))(families)
    legal = fm[:, None] & sm
    joint = jnp.where(legal, lf[:, None] + ls, -jnp.inf)
    f0 = jnp.argmax(lf)
    s0 = jnp.argmax(ls[f0])
    _, idx = jax.lax.top_k((joint + jax.random.gumbel(key, joint.shape)).ravel(), lanes)
    idx = idx.at[0].set(f0 * n + s0)
    valid = legal.ravel()[idx]
    return idx // n, idx % n, jnp.where(valid, joint.ravel()[idx], 0.), valid


def halving_schedule(lanes, budget):
    """Sequential halving: (reply lanes per round, survivors per round K, K/2, ..., 2).

    Every round runs the same number of reply lanes, spread round-robin over its survivors."""
    rounds = max(1, int(round(math.log2(lanes))))
    width = max(budget // rounds, lanes)
    return width, tuple(max(lanes >> r, 2) for r in range(rounds))


def _check_options(*, lanes, depth, rollout_temperature, root, replies):
    if root not in ('sequence', 'first'):
        raise ValueError('root must be sequence or first')
    if replies and (root != 'sequence' or depth != 1 or lanes < 2 or rollout_temperature <= 0):
        raise ValueError('replies needs root=sequence, depth=1, lanes>=2 and rollout_temperature>0')


def one(state):
    """A single-game state (no game axis) as a 1-game batch."""
    return jax.tree.map(lambda a: np.asarray(a)[None], state)


def first(states, i=0):
    """Game i of a batched state, without the game axis."""
    return jax.tree.map(lambda a: np.asarray(a)[i], states)


def take(states, index):
    return jax.tree.map(lambda a: np.ascontiguousarray(np.asarray(a)[index]), states)


class Lanes:
    """k games of engine-rs stepped in lockstep (the search lanes)."""

    def __init__(self, size, k):
        from .rust_engine import RustGames, split_masks
        self.size, self.k, self._split = size, k, split_masks
        self.games = RustGames(size, k)
        self.batch = self.games.batch

    def load(self, states):
        self.games.load(states)

    def masks(self):
        return self._split(self.batch.legal_masks(), self.size)

    def views(self, player=None):
        return self.batch.observe(player)

    def scalars(self):
        return {k: np.asarray(v) for k, v in self.batch.scalars().items()}

    def step(self, family, source, target):
        i32 = lambda a: np.ascontiguousarray(a, np.int32)
        return np.asarray(self.batch.step(i32(family), i32(source), i32(target)))

    def states(self):
        return self.games.states()

    def rewards(self):
        return np.asarray(self.batch.terminal_rewards())


class _Nets:
    """Jitted network calls over a batch of lanes (packed views in, choices / values out)."""

    def __init__(self, size):
        from .cpu_engine import view_features, view_layout
        layout = view_layout(size)
        features = lambda v: view_features(v, layout, size)

        def sample(params, view, masks, key, temperature, greedy, force, ff, fs):
            return tempered_sample(params, features(view), masks, key, size=size, temperature=temperature,
                                   greedy=greedy, force=force, forced=(ff, fs))
        self.sample = jax.jit(jax.vmap(sample, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0)))
        self.value = jax.jit(jax.vmap(lambda params, view: network.trunk(params, features(view))[2], in_axes=(None, 0)))
        self.root_first = jax.jit(lambda params, view, masks, key, lanes: root_first_actions(
            params, features(view), masks, key, size=size, lanes=lanes), static_argnums=4)
        self.split = jax.jit(jax.vmap(jax.random.split))


class Planner:
    """plan(params, state, key) -> Plan for a fixed lane count and depth.

    root='sequence': lanes 1.. sample their whole turn at `temperature`.
    root='first': lanes take distinct Gumbel-top-k first sub-actions, then play the turn greedily.
    A non-greedy lane is chosen only if its score beats the greedy lane by `margin`; with
    root='first' the selection adds prior_coef * log-prior of the first sub-action.
    after_end=True (d=0 only) scores every lane at the same point in time: the searcher's own view of the
    state right after its END, instead of the value at the END decision.

    replies=n > 0 (robust d=1): the K candidate turns are played to END once, then scored by averaging
    the critic over sampled opponent replies (at rollout_temperature), with a budget of n replies spent
    by sequential halving (K, K/2, ..., 2 survivors; the greedy turn always survives). The final pick
    needs a mean beating the greedy turn's by `margin`."""

    def __init__(self, *, size, lanes, depth=0, temperature=1., rollout_temperature=0., max_turn_actions=96,
                 max_steps=None, root='sequence', margin=0., prior_coef=0., after_end=False, replies=0):
        _check_options(lanes=lanes, depth=depth, rollout_temperature=rollout_temperature, root=root, replies=replies)
        self.size, self.lanes, self.depth, self.root = size, lanes, depth, root
        self.temperature, self.rollout_temperature = temperature, rollout_temperature
        self.max_turn_actions, self.margin, self.prior_coef = max_turn_actions, margin, prior_coef
        self.after_end, self.replies = after_end, replies
        self.max_steps = max_steps or max_turn_actions * (1 + 2 * depth)
        self.nets = _Nets(size)
        self.main = Lanes(size, lanes)
        if replies:
            self.width, self.schedule = halving_schedule(lanes, replies)
            self.reply_lanes = Lanes(size, self.width)

    def run(self, lanes, start, keys, me, params, greedy_root, ff, fs, *, depth, after_end, changes=0, done=None):
        """Plays every lane until it is scored (the lockstep runner). Returns the per-lane record."""
        k, T = lanes.k, self.max_turn_actions
        policy, scorer = _nets(params)
        lanes.load(start)
        count, steps = np.zeros(k, np.int32), np.zeros(k, np.int32)
        changes = np.broadcast_to(np.asarray(changes, np.int32), (k,)).copy()
        done = np.zeros(k, bool) if done is None else np.asarray(done, bool).copy()
        score, actions = np.zeros(k, np.float32), np.zeros((k, T, 3), np.int32)
        rollout = max(self.rollout_temperature, 1e-3)
        while not done.all():
            mover = lanes.scalars()['current_player']
            view, masks = lanes.views(), lanes.masks()
            pair = self.nets.split(keys)
            keys, draws = pair[:, 0], pair[:, 1]
            in_root = changes == 0
            greedy = np.where(in_root, greedy_root | (self.root == 'first'), self.rollout_temperature <= 0)
            temp = np.where(in_root, self.temperature, rollout).astype(np.float32)
            force = (self.root == 'first') & in_root & (count == 0) & ~greedy_root
            choice, value = self.nets.sample(policy, view, masks, draws, temp, greedy, force, ff, fs)
            fam, src, tgt = (np.asarray(x) for x in choice)
            value = np.asarray(value)
            horizon = (depth > 0) & (changes == 2 * depth)
            post_end = (depth == 0) & after_end & (changes == 1)
            root_end = (depth == 0) & (not after_end) & in_root & (fam == END_FAMILY)
            if scorer is not None:   # a separate value network scores the lanes
                value = np.asarray(self.nets.value(scorer, view))
            own_value = np.where(mover == me, value, -value)
            if after_end:
                if (post_end & ~done).any():   # not the mover's turn: the searcher's own observation
                    mine = np.asarray(self.nets.value(scorer if scorer is not None else policy, lanes.views(me)))
                    value = np.where(post_end, mine, value)
                horizon = horizon | post_end
            active = ~done
            record = in_root & ~horizon & active
            rows = np.nonzero(record)[0]
            actions[rows, count[rows]] = np.stack([fam, src, tgt], 1)[rows]
            count += record
            go = active & ~(horizon | root_end)
            status = lanes.step(np.where(go, fam, NOOP[0]), np.where(go, src, NOOP[1]), np.where(go, tgt, NOOP[2]))
            if np.any(status[go] != 0):
                raise RuntimeError('search: engine-rs rejected a sampled legal action')
            after = lanes.scalars()
            ended = go & (after['terminated'] | after['truncated'])
            outcome = np.where(after['terminated'], lanes.rewards()[:, me - 1], 0.) if ended.any() else np.zeros(k)
            changes += go & (after['current_player'] != mover)
            steps += active
            out_of_budget = (steps >= self.max_steps) | (in_root & (count >= self.max_turn_actions) & ~root_end)
            newly = active & (horizon | root_end | ended | out_of_budget)
            score = np.where(newly, np.where(horizon | root_end, value, np.where(ended, outcome, own_value)),
                             score).astype(np.float32)
            done |= newly
        final = lanes.scalars()
        return dict(actions=actions, count=count, changes=changes, steps=steps, score=score,
                    terminated=final['terminated'], truncated=final['truncated'])

    def __call__(self, params, state, key):
        return self._robust(params, state, key) if self.replies else self._plain(params, state, key)

    def _plain(self, params, state, key):
        k = self.lanes
        me = int(state.current_player)
        start = jax.tree.map(lambda a: np.ascontiguousarray(np.broadcast_to(np.asarray(a), (k,) + np.shape(a))), state)
        greedy_root = np.arange(k) == 0
        if self.root == 'first':
            self.main.load(one(state))   # lane 0 holds the root while its masks and view are read
            masks = jax.tree.map(lambda m: m[0], self.main.masks())
            ff, fs, prior, valid = (np.asarray(x) for x in self.nets.root_first(
                _nets(params)[0], self.main.views()[0], masks, jax.random.fold_in(key, 1), k))
        else:
            ff = fs = np.zeros(k, np.int32)
            prior, valid = np.zeros(k, np.float32), np.ones(k, bool)
        out = self.run(self.main, start, jax.random.split(key, k), me, params, greedy_root, ff, fs,
                       depth=self.depth, after_end=self.after_end)
        adjusted = out['score'] + self.prior_coef * prior - self.margin * ~greedy_root
        best = int(np.argmax(np.where(valid | greedy_root, adjusted, -np.inf)))   # ties keep the greedy lane 0
        return Plan(out['actions'][best], np.int32(out['count'][best]), np.int32(best), out['score'], out['steps'],
                    out['steps'] < self.max_steps)

    def _robust(self, params, state, key):
        k, width = self.lanes, self.width
        me = int(state.current_player)
        greedy_root = np.arange(k) == 0
        zeros = np.zeros(k, np.int32)
        key, draw = jax.random.split(key)
        start = jax.tree.map(lambda a: np.ascontiguousarray(np.broadcast_to(np.asarray(a), (k,) + np.shape(a))), state)
        # Stage A: each candidate turn up to and including END (the after-END value is unused).
        turns = self.run(self.main, start, jax.random.split(draw, k), me, params, greedy_root, zeros, zeros,
                         depth=0, after_end=True)
        ends = self.main.states()
        # A candidate that ended the game or ran out of budget keeps its stage-A score and gets no replies.
        open_ = (turns['changes'] == 1) & ~turns['terminated'] & ~turns['truncated'] & (turns['steps'] < self.max_steps)
        total = np.where(open_, 0., turns['score']).astype(np.float64)
        count = np.where(open_, 0., 1.)
        steps = turns['steps'].copy()
        survivors = np.arange(k)
        z = np.zeros(width, np.int32)
        for m in self.schedule:
            key, draw = jax.random.split(key)
            cand = survivors[np.arange(width) % m]
            out = self.run(self.reply_lanes, take(ends, cand), jax.random.split(draw, width), me, params,
                           np.zeros(width, bool), z, z, depth=1, after_end=False, changes=1, done=~open_[cand])
            np.add.at(total, cand, np.where(open_[cand], out['score'], 0.))
            np.add.at(count, cand, open_[cand].astype(np.float64))
            np.add.at(steps, cand, out['steps'])
            mean = total / np.maximum(count, 1.)
            alive = np.zeros(k, bool)
            alive[survivors[:m]] = True
            # Next survivors: the greedy turn first, then the best others among this round's survivors.
            rank = np.where(greedy_root, np.inf, np.where(alive, mean, -np.inf))
            survivors = np.argsort(-rank, kind='stable')
        alive = np.zeros(k, bool)
        alive[survivors[:self.schedule[-1]]] = True
        mean = total / np.maximum(count, 1.)
        challenger = int(np.argmax(np.where(greedy_root, -np.inf, np.where(alive, mean, -np.inf))))
        best = challenger if mean[challenger] > mean[0] + self.margin else 0
        return Plan(turns['actions'][best], np.int32(turns['count'][best]), np.int32(best), mean.astype(np.float32),
                    steps, turns['steps'] < self.max_steps)


class TurnSearcher:
    """Plays whole turns for one network: K-lane search, or the raw greedy policy when lanes == 1.

    value_params: parameters of a second network of the same architecture whose value head scores the lanes
    (the policy `params` still generates the candidate turns and the rollouts). Default: the policy's own
    value head. Ignored when lanes == 1 (the raw greedy policy never evaluates a score).
    parallel / workers / fast / vmap are accepted for older callers and have no effect: every phase steps all of
    its lanes in one engine-rs batch."""

    def __init__(self, params, *, size=14, lanes=1, depth=0, temperature=1., rollout_temperature=0.,
                 max_turn_actions=96, fast=True, root='sequence', margin=0., prior_coef=0., after_end=False,
                 replies=0, determinize=False, parallel=False, workers=None, vmap=False, value_params=None):
        if value_params is not None and lanes > 1:
            params = (params, value_params)
        self.params, self.size, self.lanes, self.depth, self.max_turn_actions = params, size, lanes, depth, max_turn_actions
        self._plan = Planner(size=size, lanes=lanes, depth=depth if lanes > 1 else 0, temperature=temperature,
                             rollout_temperature=rollout_temperature, max_turn_actions=max_turn_actions, root=root,
                             margin=margin, prior_coef=prior_coef, after_end=after_end,
                             replies=replies if lanes > 1 else 0)
        self._true = Lanes(size, 1)
        # determinize=True: plan on a world sampled from the searcher's own view (belief.py), replay until it diverges.
        self.determinize, self.fallbacks = determinize, 0
        if determinize:
            from .belief import build_belief
            self._belief = build_belief(size=size)
            self._pair = Lanes(size, 2)

    def compile(self, state, key):
        """Warms every jitted call on this state (the first plan is then as fast as the rest)."""
        self.play_turn(state, key, commit=False)

    def _replay(self, state, actions, count):
        """The plan's first `count` sub-actions on the true game -> (state, worst status)."""
        self._true.load(one(state))
        worst = 0
        for row in actions[:int(count)]:
            worst = max(worst, int(self._true.step(*([v] for v in row))[0]))
        return first(self._true.states()), worst

    def _guarded(self, true, belief, actions, count):
        """Plays the plan on the true game and the sampled world side by side; stops before a sub-action the
        true game rejects and right after the first one that leaves the mover's observation of the two
        different, so the caller re-plans from what was just revealed. -> (true state, applied sub-actions)."""
        me = int(true.current_player)
        pair = jax.tree.map(lambda a, b: np.stack([np.asarray(a), np.asarray(b)]), true, belief)
        self._pair.load(pair)
        applied = 0
        for row in actions[:int(count)]:
            status = self._pair.step(*([v, v] for v in row))
            if status[0] != 0:
                break
            applied += 1
            views = self._pair.views(me)
            if status[1] != 0 or not np.array_equal(views[0], views[1]):
                break
        return first(self._pair.states()), applied

    def rewards(self, state):
        """Final reward of each seat of a single-game state (0 unless the game ended with a winner)."""
        self._true.load(one(state))
        return self._true.rewards()[0]

    def play_turn(self, state, key, commit=True):
        """Plays until the mover changes or the game ends. Returns (state, plans). commit=False plans once and
        returns the state unchanged (warm-up)."""
        mover = int(state.current_player)
        plans, rejected = [], 0
        while int(state.current_player) == mover and not bool(state.terminated | state.truncated):
            key, draw = jax.random.split(key)
            # After 9 rejections in a row, commit one sub-action planned on the true state (counted in self.fallbacks).
            sample = self.determinize and rejected <= 8
            if sample:
                self._true.load(one(state))
                income = int(self._true.batch.income(mover)[0])
                root = jax.tree.map(np.asarray, self._belief(state, jax.random.fold_in(draw, 7), income))
            else:
                root = state
            p = self._plan(self.params, root, draw)
            if not commit:
                return state, [p]
            if sample:
                state, applied = self._guarded(state, root, p.actions, p.count)
                if applied == 0:   # the sampled world allowed an action the true game rejects: resample
                    rejected += 1
                    continue
                p, rejected = p._replace(count=np.int32(applied)), 0
            else:
                if self.determinize:
                    p, rejected = p._replace(count=np.minimum(p.count, 1)), 0
                    self.fallbacks += 1
                state, worst = self._replay(state, p.actions, p.count)
                if worst != 0:
                    raise RuntimeError('search replay produced an illegal action')
            plans.append(p)
        return state, plans
