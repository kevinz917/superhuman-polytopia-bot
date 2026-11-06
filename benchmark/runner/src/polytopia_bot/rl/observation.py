"""One allowlisted tensor encoder for device views and existing public demos."""
from typing import NamedTuple
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_TABLE = json.loads((Path(__file__).parents[1] / "gpu/rules.json").read_text())
STRUCTURES = tuple(sorted({r["id"] for r in _TABLE["structures"]}))
UNITS = tuple(sorted({r["id"] for r in _TABLE["units"]}))
UNIT_FLAGS = ("moved", "attacked", "poison", "frozen", "boosted", "invisible")
CITY_VALUES = ("level", "progress", "population", "production")
RELATIONS = ("known_players", "peace", "embassies_sent", "embassies_received", "peace_offers")


class Features(NamedTuple):
    board: object
    global_state: object


def encode_view(v, size):
    """Accept observe()'s output, never State. Both actor and critic use this.

    Two seats: the reference encoding. More seats (engine-rs views, len(known_players) == seats): 'enemy' is any
    other seat, and new channels/entries go after the two-seat ones (so a two-seat network extends with zero
    weights): per-tile one-hots of the owner, unit owner and capital by seat offset k = 1..seats-1 in turn order
    after the viewer, then the relation entries of the third and later seats in that order and the seats still in."""
    p = v["player"]
    players = v["known_players"].shape[0]
    known = v["explored"]
    def categorical(name, values):
        return (v[name][:, None] == jnp.asarray(values)[None, :]).astype(jnp.float32) * known[:, None]
    def ownership(name):
        a = v[name]
        return jnp.stack((known & (a == 0), known & (a == p), known & (a > 0) & (a != p)), -1)
    def seat_offsets(name):
        a = v[name]
        return jnp.stack([known & (a > 0) & ((a - p) % players == k) for k in range(1, players)], -1)
    ids = jnp.arange(size * size)
    scalars = [known, ids % size / (size - 1), ids // size / (size - 1),
               jnp.maximum(v["road"], 0), jnp.maximum(v["structure_level"], 0) / 10,
               jnp.maximum(v["unit_health"], 0) / 40, jnp.maximum(v["unit_veteran"], 0), v["detect"]]
    scalars += [v["own_unit_" + k] for k in UNIT_FLAGS]
    scalars += [v["own_unit_kills"] / 10, v["own_unit_home"] / (size * size), v["own_unit_passenger"] / 128]
    scalars += [v["own_city_" + k] / 20 for k in CITY_VALUES]
    scalars += [v["own_city_connected"], v["own_city_riot"]]
    board = jnp.concatenate((categorical("terrain", range(8)), categorical("resource", range(8)),
                             categorical("structure", STRUCTURES), categorical("unit_type", UNITS),
                             ownership("owner"), ownership("unit_owner"), ownership("capital"),
                             jnp.stack(scalars, -1), v["own_city_rewards"] / 10), -1).astype(jnp.float32)
    # seats from the viewer in turn order; with two seats [p - 1, 2 - p]
    order = (p - 1 + jnp.arange(players)) % players
    # seat index: 0/1 with two seats; scaled into [0, 1] with more, the range a two-seat network was trained on
    seat = (p - 1) if players == 2 else (p - 1) / (players - 1)
    g = jnp.concatenate((jnp.array([v["turn"] / 100, jnp.log1p(v["stars"]) / 8, seat]),
                         v["tech"], v["own_monuments"], v["own_built"],
                         *(v[k][order[:2]] for k in RELATIONS))).astype(jnp.float32)
    if players > 2:
        board = jnp.concatenate((board, *(seat_offsets(k) for k in ("owner", "unit_owner", "capital"))),
                                -1).astype(jnp.float32)
        g = jnp.concatenate((g, *(v[k][order[2:]] for k in RELATIONS), v["alive"][order])).astype(jnp.float32)
    return Features(board.reshape(size, size, -1), g)


def view_from_public(obs):
    """Reconstruct only the fields retained by learning.adapter.serialize()."""
    s = obs["visible_state"]
    size, player = s["board_size"], obs["player"]
    n = size * size
    v = dict(player=player, turn=s["turn"], stars=s["stars"], state_version=obs["state_version"])
    for key in ("terrain", "owner", "structure", "resource", "road", "capital", "structure_level",
                "unit_owner", "unit_type", "unit_health", "unit_veteran"):
        v[key] = np.full(n, -1, np.float32 if key == "unit_health" else np.int32)
    v["explored"] = np.zeros(n, bool)
    for key in ("detect", *("own_unit_" + k for k in UNIT_FLAGS), "own_city_connected", "own_city_riot"):
        v[key] = np.zeros(n, bool)
    for key in ("own_unit_kills", "own_unit_home", "own_unit_passenger", *("own_city_" + k for k in CITY_VALUES)):
        v[key] = np.zeros(n, np.int32)
    v["own_city_rewards"] = np.zeros((n, 9), np.int32)
    for t in s["tiles"]:
        i = t["tile"]
        v["explored"][i] = t["known"]
        if t["known"]:
            for k in ("terrain", "owner", "structure", "resource", "road", "capital", "structure_level"):
                v[k][i] = t[k]
            for k in ("unit_owner", "unit_type", "unit_health", "unit_veteran"):
                v[k][i] = 0
    for u in s["units"]:
        i = u["tile"]
        for a, b in (("owner", "owner"), ("type", "kind"), ("health", "health"), ("veteran", "veteran")):
            v["unit_" + a][i] = u[b]
        if u["owner"] == player:
            for k in (*UNIT_FLAGS, "kills", "home", "passenger"):
                v["own_unit_" + k][i] = u[k]
            v["detect"][i] = u["detect"]
    for c in s["cities"]:
        if c["owner"] == player:
            for k in (*CITY_VALUES, "connected", "riot", "rewards"):
                v["own_city_" + k][c["tile"]] = c[k]
    for k, ids in (("tech", s["technology"]), ("own_monuments", s["achievements"]["monuments"]),
                   ("own_built", s["achievements"]["built"])):
        v[k] = np.zeros(128, bool)
        v[k][ids] = True
    for k in RELATIONS:
        v[k] = np.asarray(s["relations"][k], bool)
    return {k: jnp.asarray(value) for k, value in v.items()}
