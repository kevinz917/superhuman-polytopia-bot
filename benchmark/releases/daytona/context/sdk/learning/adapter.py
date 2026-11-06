"""Host text transport for the frozen JAX simulator; no host game mechanics.

Only kernels.observe output reaches serialization. Full state is used solely by
compiled legality/transition/decoder functions and private snapshot I/O.
"""
import base64
from functools import lru_cache
import json
import jax
import jax.numpy as jnp
import numpy as np
from polytopia_bot.environment import LegalAction, Observation, Transition
from polytopia_bot.gpu import FULL_GAME_READY, RULESET_ID
from polytopia_bot.gpu import actions, core, kernels, mapgen, snapshot
from polytopia_bot.gpu.rules import load_rules
from .interface import canonical, public_state
from .memory import PublicMemory


@lru_cache(maxsize=2)
def runtime(size):
    rules = load_rules()
    return (
        jax.jit(lambda seed: mapgen.reset(seed, size=size)),
        jax.jit(lambda s: actions.flatten(actions.legal_masks(s, rules, size=size))),
        jax.jit(lambda s, indices: jax.vmap(lambda i: actions.decode(s, i, size=size))(indices)),
        jax.jit(lambda s, a, version: core.step(s, a, version, rules, size=size)),
        jax.jit(kernels.observe),
    )


def serialize(view, *, size, legal):
    """Accept only an already-redacted device view, never an engine State."""
    v = jax.device_get(view)
    player = int(v['player'])
    tiles, units, cities = [], [], []
    for i, explored in enumerate(v['explored']):
        t = dict(tile=i, known=bool(explored))
        if explored:
            t.update({name: int(v[name][i]) for name in ('terrain', 'owner', 'structure', 'resource', 'road', 'capital', 'structure_level')})
            if v['unit_owner'][i] > 0:
                u = dict(tile=i, owner=int(v['unit_owner'][i]), kind=int(v['unit_type'][i]), health=float(v['unit_health'][i]), veteran=bool(v['unit_veteran'][i]))
                if u['owner'] == player:
                    for name in ('moved', 'attacked', 'poison', 'frozen', 'boosted', 'invisible'):
                        u[name] = bool(v['own_unit_' + name][i])
                    for name in ('kills', 'home', 'passenger'):
                        u[name] = int(v['own_unit_' + name][i])
                    u['detect'] = bool(v['detect'][i])
                units.append(u)
            # Structure 1 denotes villages/cities; 2 is a ruin, not a city.
            if v['structure'][i] == 1:
                c = dict(tile=i, owner=int(v['owner'][i]))
                if c['owner'] == player:
                    for name in ('level', 'progress', 'population', 'production'):
                        c[name] = int(v['own_city_' + name][i])
                    for name in ('connected', 'riot'):
                        c[name] = bool(v['own_city_' + name][i])
                    c['rewards'] = v['own_city_rewards'][i].tolist()
                cities.append(c)
        tiles.append(t)
    visible = dict(board_size=size, turn=int(v['turn']), stars=int(v['stars']),
        technology=np.flatnonzero(v['tech']).tolist(), tiles=tiles, units=units, cities=cities,
        relations={k: np.asarray(v[k]).astype(bool).tolist() for k in ('known_players', 'peace', 'embassies_sent', 'embassies_received', 'peace_offers')},
        achievements={k: np.flatnonzero(v['own_' + k]).tolist() for k in ('monuments', 'built')})
    obs = Observation(player, int(v['state_version']), visible, {}, legal)
    public_state(obs)
    return obs


class GPUEnvironment:
    def __init__(self, size=14):
        if size not in (11, 14) or not FULL_GAME_READY:
            raise ValueError('unsupported or unready environment')
        self.size = size
        self._reset, self._mask, self._decode, self._step, self._observe = runtime(size)
        self.state = None
        self.memory = PublicMemory()
        self._cached = {}

    def reset(self, *, seed, settings):
        if set(settings) - {'size', 'ruleset'} or settings.get('size', self.size) != self.size or settings.get('ruleset', RULESET_ID) != RULESET_ID:
            raise ValueError('settings differ from frozen research rules')
        self.state = self._reset(jnp.int32(seed))
        self.memory = PublicMemory()
        self._cached = {}
        # Both players observe the start, independently, through the same API.
        self.observe(2)
        return self.observe(1)

    def observe(self, player):
        if player not in (1, 2) or self.state is None:
            raise ValueError('invalid seat or closed environment')
        if player in self._cached:
            return self._cached[player]
        legal = []
        if player == int(self.state.current_player) and not bool(self.state.terminated | self.state.truncated):
            indices = np.flatnonzero(np.asarray(self._mask(self.state)))
            # Fixed-size decode chunks avoid recompilation for every catalog size.
            for start in range(0, len(indices), 256):
                part = indices[start:start + 256]
                padded = np.pad(part, (0, 256 - len(part)))
                commands = jax.device_get(self._decode(self.state, jnp.asarray(padded, dtype=jnp.int32)))
                for j, index in enumerate(part):
                    legal.append(LegalAction(str(int(index)), {k: int(getattr(commands, k)[j]) for k in core.Action._fields}))
        obs = serialize(self._observe(self.state, jnp.int32(player)), size=self.size, legal=legal)
        obs = self.memory.observe(obs)
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
        a = core.Action(*(jnp.int32(selected.command[k]) for k in core.Action._fields))
        result = self._step(self.state, a, jnp.int32(expected_state_version))
        if int(result.status) != core.OK:
            raise RuntimeError(f'catalog/transition discrepancy: status={int(result.status)}')
        self.memory.executed(before, selected)
        self.state, self._cached = result.state, {}
        terminal, truncated = bool(self.state.terminated), bool(self.state.truncated)
        rewards = {p: float(r) for p, r in enumerate(core.terminal_rewards(self.state), 1)}
        # Record every new public observation for both seats, even off-turn.
        actor = self.observe(player)
        self.observe(3 - player)
        reason = ('resignation' if selected.command['kind'] == 11 else 'elimination') if terminal else ('research_horizon' if truncated else None)
        return Transition(actor, None if terminal or truncated else int(self.state.current_player), rewards, terminal, truncated, reason)

    def snapshot(self):
        return canonical(dict(schema='public-runner-v1', size=self.size,
            engine=base64.b64encode(snapshot.dumps(self.state, profile='native')).decode(), memory=self.memory.snapshot())).encode()

    def restore(self, payload):
        data = json.loads(payload)
        if data.get('schema') != 'public-runner-v1' or data.get('size') != self.size:
            raise ValueError('incompatible runner snapshot')
        self.state = snapshot.loads(base64.b64decode(data['engine'], validate=True), profile='native')
        self.memory, self._cached = PublicMemory.restore(data['memory']), {}
        return self.observe(int(self.state.current_player))

    def close(self):
        self.state, self._cached = None, {}
        self.memory = PublicMemory()
