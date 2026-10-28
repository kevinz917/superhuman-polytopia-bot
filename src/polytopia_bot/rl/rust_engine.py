"""The game engine: engine-rs (Rust), the real game's rules (polytopia_bot.ruleset), behind a batched interface.

RustGames keeps its games inside the extension module (struct-of-arrays, one game stepped at a time) and returns
NumPy arrays; `states()` exports every State field (the field layout of gpu/state.py) and `load(states)` imports
one. Rules come from gpu/rules.json. Observation features (encode_view) are computed in JAX from the packed int16
view, as in the learner. players=4 runs four-seat games (docs/four-player-2026-10-24.md); their exported arrays have
a seat axis of 4.
"""
from typing import NamedTuple

import numpy as np

from polytopia_bot.gpu import ENGINE_REVISION
from polytopia_bot.ruleset import rules_json
from polytopia_bot.gpu.actions import Masks
from polytopia_bot.gpu.state import Cities, Players, State, Tiles, Units
from .actions import shapes
from .cpu_engine import Prepared, Tick, view_features, view_layout

_GROUPS = (("tiles", Tiles), ("units", Units), ("cities", Cities), ("players", Players))
_SCALARS = ("current_player", "turn", "version", "state_version", "actions", "terminated", "truncated", "fog",
            "reference_max_turns", "core_supported", "seed", "initial_seed")


def state_to_dict(state):
    """JAX/NumPy State pytree -> {"tiles.terrain": array, ...} (arrays keep their leading game axis)."""
    out = {}
    for name, cls in _GROUPS:
        group = getattr(state, name)
        for f in cls._fields:
            out[f"{name}.{f}"] = np.asarray(getattr(group, f))
    for f in _SCALARS:
        out[f] = np.asarray(getattr(state, f))
    return out


def dict_to_state(d):
    groups = [cls(*(d[f"{name}.{f}"] for f in cls._fields)) for name, cls in _GROUPS]
    return State(*groups, *(d[f] for f in _SCALARS))


def split_masks(flat, size):
    """Flat [games, mask_width] (uint8/bool) -> Masks of [games, ...] bool arrays shaped like gpu.actions.legal_masks.

    The engine's uint8 masks hold only 0/1, so they are reinterpreted as bool in place (no 131 KB-per-game copy);
    the families are views into that fresh per-tick buffer, as they were into the astype copy before."""
    flat = np.asarray(flat)
    flat = flat.view(bool) if flat.dtype == np.uint8 else flat.astype(bool)
    n = size * size
    target = ((n, n), (n, n), (n, 128), (n, 128), (128,), (n,), (n, 3), (n, 9), *((n,),) * 6, (6,), (), ())
    out, at = [], 0
    for shape in target:
        k = int(np.prod(shape)) if shape else 1
        out.append(flat[:, at:at + k].reshape(flat.shape[0], *shape))
        at += k
    assert at == flat.shape[1]
    return Masks(*out)


class RustGames:
    """A fixed number of independent games stepped by the Rust engine.

    The games live inside the extension module, so `tick` ignores the `states` it is handed and keeps its own;
    the returned Tick.states is a NumPy export of every field (what the actor reads). Policy features are the
    JAX `view_features` of the packed view (the same function the learner uses), so a RustGames tick costs the
    Rust tick plus one jitted feature encode. Every action is fully validated (no trusted step needed), so
    `check` is accepted and ignored; `trusted` / `build_rows` / `fast` are accepted for signature compatibility.
    """

    def __init__(self, size=14, games=32, *, no_resign=True, map_batch=False, max_turns=None, both_views=False,
                 max_actions=10_000, fast=True, trusted=None, build_rows=None, players=2, served=False):
        """map_batch=True: reset/tick take a maps.state.MapBatch of per-lane layouts instead of seeds.
        players: seats per game (2 or 4). With both_views and more than two seats,
        Prepared.opp_view is [games, players - 1, width]: the other seats in turn order after the mover.
        served=True (the policy runs on the node's inference server, rl/inference_server.py): Prepared.features is
        None and Prepared.masks is the engine's [games, ceil(mask_width/8)] uint8 packed bits (numpy.packbits
        order, the server's wire format), so the actor runs no feature encode and no mask split."""
        import jax
        from polytopia_engine_rs import GameBatch
        self.size, self.games, self.both_views, self.map_batch = size, games, both_views, map_batch
        self.players, self.served = players, served
        self.batch = GameBatch(size, games, rules_json(), ENGINE_REVISION, max_turns, no_resign, max_actions, players)
        self.layout = view_layout(size, players)
        names = [k for k, _ in self.layout]
        if names != list(self.batch.view_fields()):
            raise RuntimeError("packed view layout differs between JAX and Rust")
        self.view_width = sum(int(np.prod(s)) for _, s in self.layout)
        if self.view_width != self.batch.view_width:
            raise RuntimeError("packed view width differs between JAX and Rust")
        self._features = jax.jit(jax.vmap(lambda v: view_features(v, self.layout, size)))
        self._features(np.zeros((games, self.view_width), np.int16))   # compile now

    def states(self):
        return dict_to_state(self.batch.export())

    def load(self, states):
        self.batch.import_(state_to_dict(states))

    def _prepare(self, tick=None):
        if tick is None:
            masks = self.batch.legal_masks(packed=self.served)
            view = self.batch.observe()
            opp = self.batch.observe_opponent() if self.both_views else None
            if opp is not None and self.players > 2:
                opp = opp.reshape(self.games, self.players - 1, self.view_width)
        else:
            masks, view, opp = tick["masks"], tick["view"], tick.get("opp_view")
        if self.served:
            return Prepared(None, masks, view, opp)
        return Prepared(self._features(view), split_masks(masks, self.size), view, opp)

    def _layouts(self, seeds):
        if not self.map_batch:
            return np.ascontiguousarray(seeds, np.int32), None
        if not hasattr(seeds, 'terrain'):
            from polytopia_bot.maps.stream import MapSource
            seeds = MapSource('classic', self.size, players=self.players).batch(seeds)[0]
        i32 = lambda a: np.ascontiguousarray(np.asarray(a), np.int32)
        return i32(seeds.seed), (i32(seeds.terrain), i32(seeds.structure), i32(seeds.resource), i32(seeds.capitals))

    def reset(self, seeds):
        """seeds: int seeds, or a MapBatch when map_batch=True."""
        seeds, layouts = self._layouts(seeds)
        if layouts is None:
            self.batch.reset(seeds)
        else:
            self.batch.reset_layouts(seeds, *layouts)
        return self.states(), self._prepare()

    def tick(self, states, choices, seeds, check=False):
        """Returns (Tick, status); `states` and `check` are ignored (see class docstring)."""
        import jax
        f, s, t = (np.ascontiguousarray(np.asarray(a), np.int32) for a in choices)
        seeds, layouts = self._layouts(seeds)
        out = self.batch.tick(f, s, t, seeds, self.both_views, layouts, self.served)
        final_views = out["final_view"]
        seats = self.players
        if np.any(out["truncated"]):
            # the final features of lanes that did not truncate stay zero (not the encode of a zero view)
            keep = np.asarray(out["truncated"]).reshape((self.games, 1))
            flat = self._features(final_views.reshape(self.games * seats, self.view_width))
            final_features = jax.tree.map(
                lambda a: np.where(keep.reshape((self.games, 1) + (1,) * (a.ndim - 1)),
                                   np.asarray(a).reshape((self.games, seats) + a.shape[1:]), 0).astype(np.float32), flat)
        else:
            final_features = None
        tick = Tick(self.states(), self._prepare(out), out["done"], out["terminated"], out["truncated"], out["rewards"],
                    final_features, final_views)
        return tick, out["status"]

    def step(self, choices):
        f, s, t = (np.ascontiguousarray(np.asarray(a), np.int32) for a in choices)
        return self.batch.step(f, s, t)
