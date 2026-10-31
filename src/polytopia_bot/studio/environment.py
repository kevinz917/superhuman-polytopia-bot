"""The public game environment Studio plays through, on engine-rs (the real game's rules, polytopia_bot.ruleset).

Same interface as the retired JAX GPUEnvironment (learning/adapter.py): reset / observe / step / snapshot / restore /
close, legal actions as catalog IDs (the flat index of rl.actions.offsets/shapes) with the decoded command fields,
observations serialized from the redacted per-seat view only. `state` is the single-game NumPy State engine-rs
exports (gpu/state.py layout), used for the HUD and for search.
"""
import base64
import json

import numpy as np

from polytopia_bot.environment import LegalAction, Transition
from polytopia_bot.gpu import snapshot
from polytopia_bot.learning.adapter import serialize
from polytopia_bot.learning.interface import canonical
from polytopia_bot.learning.memory import PublicMemory
from polytopia_bot.rl.actions import offsets, shapes

KINDS = ('kind', 'source', 'target', 'parameter')


def decode_command(index, state, size):
    """Catalog index -> the engine action's (kind, source, target, parameter), as engine-rs core::decode."""
    n = size * size
    bounds = offsets(size)
    family = int(np.searchsorted(bounds, index, side='right')) - 1
    at = int(index - bounds[family])
    other = 3 - int(state.current_player)
    if family == 14:
        caps = np.flatnonzero((np.asarray(state.tiles.capital) == other) & (np.asarray(state.cities.owner) == other))
        capital = int(caps[0]) if len(caps) else 0
        return dict(kind=3, source=other, target=(-1, 0, 1, capital, -1, -1)[at], parameter=(18, 19, 19, 20, 21, 22)[at])
    if family in (9, 10, 11, 12):
        return dict(kind=3, source=at, target=-1, parameter=(7, 16, 8, 9)[family - 9])
    table = {0: (1, at // n, at % n, 0), 1: (2, at // n, at % n, 0), 2: (4, at // 128, -1, at % 128),
             3: (6, -1, at // 128, at % 128), 4: (7, -1, -1, at), 5: (5, -1, at, 0), 6: (3, -1, at // 3, at % 3 + 1),
             7: (9, -1, at // 9, at % 9), 8: (8, at, -1, 0), 13: (3, -1, at, 4), 15: (10, -1, -1, 0),
             16: (11, -1, -1, 0)}
    return dict(zip(KINDS, table[family]))


def choice_of(index, size):
    """Catalog index -> the (family, source, target) engine-rs steps."""
    bounds = offsets(size)
    family = int(np.searchsorted(bounds, index, side='right')) - 1
    source, target = divmod(int(index - bounds[family]), shapes(size)[family][1])
    return family, source, target


class RustEnvironment:
    def __init__(self, size=14):
        from polytopia_bot.rl.cpu_engine import unpack_view, view_layout
        from polytopia_bot.rl.rust_engine import RustGames
        self.size = size
        self.games = RustGames(size, 1, no_resign=False)
        self.layout, self._unpack = view_layout(size), unpack_view
        self.state = None
        self.memory = PublicMemory()
        self._cached = {}

    def _sync(self):
        from polytopia_bot.rl.search import first
        self.state = first(self.games.states())

    def view(self, player, state=None):
        """The redacted public view of `player` (a dict like gpu.kernels.observe), optionally of another state."""
        if state is not None:
            from polytopia_bot.rl.search import one
            scratch = self._scratch()
            scratch.load(one(state))
            packed = np.asarray(scratch.batch.observe(int(player)))[0]
            version = int(state.state_version)
        else:
            packed = np.asarray(self.games.batch.observe(int(player)))[0]
            version = int(self.state.state_version)
        out = {k: np.asarray(v) for k, v in self._unpack(packed, self.layout).items()}
        out['state_version'] = np.int32(version)
        return out

    def _scratch(self):
        if not hasattr(self, '_render'):
            from polytopia_bot.rl.rust_engine import RustGames
            self._render = RustGames(self.size, 1, no_resign=False)
        return self._render

    def income(self, player):
        return int(np.asarray(self.games.batch.income(int(player)))[0])

    def reset(self, *, seed, settings):
        if set(settings) - {'size'} or settings.get('size', self.size) != self.size:
            raise ValueError('unsupported settings')
        self.games.reset(np.array([seed], np.int32))
        self._sync()
        self.memory, self._cached = PublicMemory(), {}
        self.observe(2)   # both players observe the start, independently, through the same API
        return self.observe(1)

    def observe(self, player):
        if player not in (1, 2) or self.state is None:
            raise ValueError('invalid seat or closed environment')
        if player in self._cached:
            return self._cached[player]
        legal = []
        if player == int(self.state.current_player) and not bool(self.state.terminated | self.state.truncated):
            mask = np.asarray(self.games.batch.legal_masks())[0]
            for index in np.flatnonzero(mask):
                legal.append(LegalAction(str(int(index)), decode_command(int(index), self.state, self.size)))
        obs = self.memory.observe(serialize(self.view(player), size=self.size, legal=legal))
        self._cached[player] = obs
        return obs

    def step(self, *, action_id, expected_state_version):
        if self.state is None or expected_state_version != int(self.state.state_version):
            raise ValueError('stale_version')
        player = int(self.state.current_player)
        before = self.observe(player)
        catalog = {a.action_id: a for a in before.legal_actions}
        if action_id not in catalog:
            raise ValueError('invalid_action')
        selected = catalog[action_id]
        status = int(np.asarray(self.games.batch.step(*(np.array([v], np.int32)
                                                         for v in choice_of(int(action_id), self.size))))[0])
        if status != 0:
            raise RuntimeError(f'catalog/transition discrepancy: status={status}')
        self.memory.executed(before, selected)
        self._sync()
        self._cached = {}
        terminal, truncated = bool(self.state.terminated), bool(self.state.truncated)
        rewards = {p: float(r) for p, r in enumerate(np.asarray(self.games.batch.terminal_rewards())[0], 1)}
        actor = self.observe(player)   # record every new public observation for both seats, even off-turn
        self.observe(3 - player)
        reason = ('resignation' if selected.command['kind'] == 11 else 'elimination') if terminal else (
            'research_horizon' if truncated else None)
        return Transition(actor, None if terminal or truncated else int(self.state.current_player), rewards, terminal,
                          truncated, reason)

    def preview(self, action_id):
        """Health after `action_id` of every unit the mover can see that it would change, by playing it on a
        scratch copy of the game (combat is deterministic). Keyed by the tile the unit stands on now: a melee
        attacker that wins and advances is reported on its own tile; 0 means the unit dies."""
        from polytopia_bot.rl.search import first, one
        player = int(self.state.current_player)
        obs = self.observe(player)
        action = next((a for a in obs.legal_actions if a.action_id == action_id), None)
        if action is None:
            raise ValueError('invalid_action')
        scratch = self._scratch()
        scratch.load(one(self.state))
        status = int(np.asarray(scratch.batch.step(*(np.array([v], np.int32)
                                                     for v in choice_of(int(action_id), self.size))))[0])
        if status != 0:
            raise RuntimeError(f'catalog/transition discrepancy: status={status}')
        after = first(scratch.states())
        owner, health = np.asarray(after.units.owner), np.asarray(after.units.health)
        source, target = action.command['source'], action.command['target']
        out = []
        for u in obs.visible_state['units']:
            tile, mine = u['tile'], u['owner']
            if owner[tile] == mine:
                hp = float(health[tile])
            elif tile == source and target >= 0 and owner[target] == mine:
                hp = float(health[target])
            else:
                hp = 0.0
            if hp != u['health']:
                out.append(dict(tile=tile, health=u['health'], after=max(0.0, hp)))
        return out

    def load(self, state):
        """Continue from a single-game State (e.g. after a search played the turn on its own copy)."""
        from polytopia_bot.rl.search import one
        self.games.load(one(state))
        self._sync()
        self._cached = {}

    def snapshot(self):
        return canonical(dict(schema='public-runner-v2', size=self.size,
                              engine=base64.b64encode(snapshot.dumps(self.state, profile='native')).decode(),
                              memory=self.memory.snapshot())).encode()

    def restore(self, payload):
        data = json.loads(payload)
        if data.get('schema') != 'public-runner-v2' or data.get('size') != self.size:
            raise ValueError('incompatible runner snapshot (games saved before 2026-10-27 used the retired rules)')
        self.load(snapshot.loads(base64.b64decode(data['engine'], validate=True), profile='native'))
        self.memory = PublicMemory.restore(data['memory'])
        return self.observe(int(self.state.current_player))

    def close(self):
        self.state, self._cached = None, {}
        self.memory = PublicMemory()
