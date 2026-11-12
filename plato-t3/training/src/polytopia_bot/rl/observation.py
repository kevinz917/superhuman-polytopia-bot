"""One allowlisted tensor encoder for device views and existing public demos."""
from typing import NamedTuple
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_TABLE = json.loads((Path(__file__).parents[1] / "gpu/rules.json").read_text())
# The structure kinds of the board's structure one-hot (observation versions 1-5): rules.json's structures without the
# rows added 2026-11-09, so every existing checkpoint keeps its input width: Elyrion's enchant animal (30), a
# Consumed build that never stands on a tile, and Aquarion's Aquafarm 49 and Atoll 51, which encode as no structure
# until an observation version adds them.
STRUCTURES = tuple(sorted({r["id"] for r in _TABLE["structures"]} - {30, 49, 51}))
# The unit kinds of the board's unit_type one-hot (observation versions 1-5): rules.json's units without the full-game engine
# Aquarion rows added 2026-11-09 (MermaidWarrior 46 .. Aquapult 55). Those kinds encode as no unit type until an
# observation version adds them, so every existing checkpoint keeps its input width.
UNITS = tuple(sorted({r["id"] for r in _TABLE["units"]} - set(range(46, 56))))
UNIT_FLAGS = ("moved", "attacked", "poison", "frozen", "boosted", "invisible")
CITY_VALUES = ("level", "progress", "population", "production")
RELATIONS = ("known_players", "peace", "embassies_sent", "embassies_received", "peace_offers")
# Observation versions (recorded in a checkpoint's config as obs_version; absent = 1):
#   1: the features every network before 2026-10-31 was trained on.
#   2: + the facing of every visible unit (view field unit_direction, grid direction 0..7 one-hot: what the game shows
#      as the sprite's flip, and where a push starts), DIRECTION_CHANNELS board channels right after the two-seat
#      channels of each view (before the multi-seat ones, so a two-seat version-2 network still widens to four
#      seats by appending; network.widen_inputs).
#   3 (2026-11-01): + the seat's observation history (engine-rs history.rs, leak-free: built only from the seat's own
#      past views). HISTORY_CHANNELS board channels at the end of each view group: a visible enemy unit stood here
#      at the end of my turn 1 / 2 / 3 of my turns ago (hist_enemy_unit), an enemy structure, road or level appeared
#      or changed here since my last turn (hist_enemy_built), a visible enemy unit stands on an enemy city tile
#      where none stood at the end of my last turn (hist_enemy_trained); and 128 global entries at the end of the
#      group: the techs the enemy must have, deduced from the enemy units, structures and roads I have seen, with
#      their prerequisites (enemy_tech). A version-2 network widens to version 3 with zero rows (network.widen_inputs).
#   4 (2026-11-09, full-game engine): + Starfish and aqua crop (resource ids 8 and 9), WATER_RESOURCE_CHANNELS board channels at
#      the very end of each view group. Versions 1-3 one-hot resource ids 0-7 only, so on the real game's water presets
#      (10-22 such tiles per board) a network saw those tiles as resource-free. A version-3 network reads version-4
#      features unchanged through fit_board; transformer_v3.widen_obs_v4 gives it zero rows for the new channels
#      (identical outputs on every board until they are trained).
#   5 (2026-11-09, full-game engine tribes): + both seats' tribes, 2 * TRIBE_ENTRIES global entries at the very end of the group:
#      a one-hot of my tribe and a multi-hot of the other seats' tribes (view field "tribe"; another seat's only once
#      I have met it), over the tribes other than Bardur (TRIBES): Bardur is all zeros, so a Bardur-vs-Bardur game
#      has exactly the version-4 features plus zeros, whatever the new weights hold. No new board channels.
#      transformer_v3.widen_obs_v5 gives a version-3/4 network zero rows for them (identical outputs everywhere).
#   6 (2026-11-09, special tribes; section 2): SPECIAL_CHANNELS board channels at the very
#      end of each view group: the unit kinds 46-55 (MermaidWarrior .. Aquapult; versions 1-5 encode them as no unit
#      type), the tile effects Polaris climate, Flooded and Algae, the visible unit's Bubble and Swift, a Grow unit's
#      age (min(turn - created, 3) / 3: DragonEgg, BabyDragon, bug egg, Larva), its chain links (has a leader / has a
#      follower), and two private ones, the seat's own units' double ready and its own Sanctuary / Fungi age
#      (min(turn - structure_turn, 4) / 4); then Decomposing. One global entry at the very end of the group: the
#      Convert task's progress (min(converts, 3) / 3). Every one is zero in a game of two regular tribes, so a
#      version-5 network (transformer_v3.widen_obs_v6: zero rows) gives identical outputs there.
# encode_view always computes the newest features; a network of an older version reads them through fit_board
# (network.trunk does), so every consumer encodes one way and old checkpoints keep their input width.
OBS_VERSION = 6
LEGACY_OBS_VERSION = 1
DIRECTION_CHANNELS = 8
HISTORY_TURNS = 3
HISTORY_CHANNELS = HISTORY_TURNS + 2
WATER_RESOURCES = (8, 9)          # Starfish, aqua crop (version 4)
WATER_RESOURCE_CHANNELS = len(WATER_RESOURCES)
ENEMY_TECH_ENTRIES = 128
# version 5: every tribe type but Bardur (4), in id order (the game's tribe ids: 2 Ai-Mo ... 17 Cymanti)
TRIBES = (2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
TRIBE_ENTRIES = len(TRIBES)
# version 6: the special tribes' unit kinds (the Aquarion rows UNITS leaves out), the Grow units, the aged structures
SPECIAL_UNITS = tuple(range(46, 56))
GROW_UNITS = tuple(sorted(r["id"] for r in _TABLE["units"] if "Grow" in r.get("skills", ())))
AGE_STEPS, STRUCTURE_AGE_STEPS, CONVERT_TASK = 3, 4, 3
SPECIAL_CHANNELS = len(SPECIAL_UNITS) + 11
SPECIAL_GLOBAL = 1
# the global entries every version and seat count starts with: turn, stars, seat, tech, monuments, built, 5 two-seat relations
TWO_SEAT_GLOBAL = 3 + 3 * 128 + 2 * len(RELATIONS)


def two_seat_channels():
    """Board channels of the version-1 two-seat features (the part every version and seat count starts with)."""
    scalars = 8 + len(UNIT_FLAGS) + 3 + len(CITY_VALUES) + 2
    return 8 + 8 + len(STRUCTURES) + len(UNITS) + 9 + scalars + 9


def board_layout(players=2, obs_version=OBS_VERSION):
    """(block, channels) of one view's board channels as encode_view lays them out for `players` seats and a network
    of observation version `obs_version`: the two-seat channels, the unit facing (version 2+), the multi-seat
    one-hots (more than two seats), the observation history (version 3+)."""
    blocks = [('base', two_seat_channels())]
    if obs_version >= 2:
        blocks.append(('facing', DIRECTION_CHANNELS))
    if players > 2:
        blocks.append(('seats', 3 * (players - 1)))
    if obs_version >= 3:
        blocks.append(('history', HISTORY_CHANNELS))
    if obs_version >= 4:
        blocks.append(('water_resources', WATER_RESOURCE_CHANNELS))
    if obs_version >= 6:
        blocks.append(('special', SPECIAL_CHANNELS))
    return tuple(blocks)


def global_layout(players=2, obs_version=OBS_VERSION):
    """(block, entries) of one view's global vector: the two-seat entries, the extra seats' relations and the seats
    still in (more than two seats), the inferred enemy techs (version 3+)."""
    blocks = [('base', TWO_SEAT_GLOBAL)]
    if players > 2:
        blocks.append(('seats', len(RELATIONS) * (players - 2) + players))
    if obs_version >= 3:
        blocks.append(('enemy_tech', ENEMY_TECH_ENTRIES))
    if obs_version >= 5:
        blocks.append(('tribes', 2 * TRIBE_ENTRIES))
    if obs_version >= 6:
        blocks.append(('convert_task', SPECIAL_GLOBAL))
    return tuple(blocks)


def _index(layout, players_net, obs_version_net, players_feat, obs_version_feat):
    offsets, at = {}, 0
    for name, count in layout(players_feat, obs_version_feat):
        offsets[name] = (at, count)
        at += count
    out = []
    for name, count in layout(players_net, obs_version_net):
        if name not in offsets or offsets[name][1] != count:
            raise ValueError(f'a {players_net}-seat version-{obs_version_net} network cannot read {players_feat}-seat '
                             f'version-{obs_version_feat} features')
        out.extend(range(offsets[name][0], offsets[name][0] + count))
    return np.asarray(out, np.int32)


def channel_index(players_net, obs_version_net, players_feat, obs_version_feat=OBS_VERSION):
    """Positions, inside one view's board channels encoded for `players_feat` seats at version `obs_version_feat`, of
    the channels a network for `players_net` seats of version `obs_version_net` reads, in its order."""
    return _index(board_layout, players_net, obs_version_net, players_feat, obs_version_feat)


def global_index(players_net, obs_version_net, players_feat, obs_version_feat=OBS_VERSION):
    """channel_index for the global vector."""
    return _index(global_layout, players_net, obs_version_net, players_feat, obs_version_feat)


def group_width(players=2, obs_version=OBS_VERSION):
    return sum(c for _, c in board_layout(players, obs_version))


def global_width(players=2, obs_version=OBS_VERSION):
    return sum(c for _, c in global_layout(players, obs_version))


def _decode_widths(width, channels, layout=board_layout):
    """(groups, players, network version) of a stem reading `channels` channels of `width`-channel newest features:
    the combination whose widths match. Every (groups, players) product of newest widths is distinct (checked by
    tests); versions that read the same channels (the global vector of versions 1 and 2) are interchangeable, so the
    lowest matching version is returned."""
    size = lambda players, version=OBS_VERSION: sum(c for _, c in layout(players, version))
    found = [(groups, players, version) for groups in range(1, 5) for players in (2, 3, 4) for version in (1, 2, 3, 4, 5, 6)
             if groups * size(players) == width and groups * size(players, version) == channels]
    if not found or len({(g, p) for g, p, _ in found}) != 1:
        raise ValueError(f'features with {width} entries do not fit a network reading {channels}')
    return found[0]


def fit_global(global_state, entries):
    """fit_board for the global vector: `global_state` (newest features, one or more concatenated views) for a layer
    that reads `entries` of them (version 1 and 2 networks read no enemy techs)."""
    width = global_state.shape[-1]
    if width == entries:
        return global_state
    groups, players, version = _decode_widths(width, entries, global_layout)
    per = width // groups
    index = global_index(players, version, players)
    x = global_state.reshape(global_state.shape[:-1] + (groups, per))
    return jnp.take(x, index, axis=-1).reshape(global_state.shape[:-1] + (entries,))


def fit_board(board, channels):
    """`board` (newest features: one or more concatenated views along the last axis) for a stem that reads
    `channels` channels: unchanged when they match, else each view's channels of the older observation version the
    stem was built for (version 1: without the unit facing; version 2: without the history; the omniscient critic
    reads several views)."""
    width = board.shape[-1]
    if width == channels:
        return board
    groups, players, version = _decode_widths(width, channels)
    per = width // groups
    index = channel_index(players, version, players)
    x = board.reshape(board.shape[:-1] + (groups, per))
    return jnp.take(x, index, axis=-1).reshape(board.shape[:-1] + (channels,))


class Features(NamedTuple):
    board: object
    global_state: object


def encode_view(v, size, obs_version=OBS_VERSION):
    """Accept observe()'s output, never State. Both actor and critic use this.

    Two seats: the reference encoding. More seats (engine-rs views, len(known_players) == seats): 'enemy' is any
    other seat, and new channels/entries go after the two-seat ones (so a two-seat network extends with zero
    weights): per-tile one-hots of the owner, unit owner and capital by seat offset k = 1..seats-1 in turn order
    after the viewer, then the relation entries of the third and later seats in that order and the seats still in.
    obs_version >= 2 inserts the unit-facing one-hots right after the two-seat channels (a view without the
    unit_direction field, e.g. a public demonstration, gives zeros). obs_version >= 3 appends the observation history
    planes to the board channels and the inferred enemy techs to the global vector (zeros without the fields)."""
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
    assert board.shape[-1] == two_seat_channels()
    if obs_version >= 2:
        # unit_direction: facing + 1 of a visible unit, 0 without one, -1 unexplored
        facing = v["unit_direction"] if "unit_direction" in v else jnp.zeros(size * size, jnp.int32)
        board = jnp.concatenate((board, ((facing[:, None] == jnp.arange(1, DIRECTION_CHANNELS + 1)[None, :])
                                         * known[:, None]).astype(jnp.float32)), -1)
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
    if obs_version >= 3:
        n = size * size
        zeros = lambda *shape: jnp.zeros(shape, jnp.int32)
        unit = v["hist_enemy_unit"] if "hist_enemy_unit" in v else zeros(HISTORY_TURNS, n)
        built = v["hist_enemy_built"] if "hist_enemy_built" in v else zeros(n)
        trained = v["hist_enemy_trained"] if "hist_enemy_trained" in v else zeros(n)
        tech = v["enemy_tech"] if "enemy_tech" in v else zeros(ENEMY_TECH_ENTRIES)
        history = jnp.concatenate((unit.reshape(HISTORY_TURNS, n).T, built[:, None], trained[:, None]), -1)
        board = jnp.concatenate((board, (history > 0).astype(jnp.float32)), -1)
        g = jnp.concatenate((g, tech.astype(jnp.float32)))
    if obs_version >= 4:
        board = jnp.concatenate((board, categorical("resource", WATER_RESOURCES)), -1).astype(jnp.float32)
    if obs_version >= 5:
        tribe = v["tribe"] if "tribe" in v else jnp.zeros(players, jnp.int32)
        hot = lambda t: (t[..., None] == jnp.asarray(TRIBES)).astype(jnp.float32)
        others = jnp.minimum(hot(tribe[order[1:]]).sum(0), 1.)
        g = jnp.concatenate((g, hot(tribe[order[0]]), others)).astype(jnp.float32)
    if obs_version >= 6:
        n = size * size
        field = lambda name: v[name] if name in v else jnp.zeros(n, jnp.int32)
        flag = lambda name: (field(name) > 0) & known
        kind = v["unit_type"]
        grow = (kind[:, None] == jnp.asarray(GROW_UNITS)[None, :]).any(-1)
        age = jnp.where(grow & known, jnp.clip(field("unit_age"), 0, AGE_STEPS) / AGE_STEPS, 0.)
        special = jnp.concatenate((
            categorical("unit_type", SPECIAL_UNITS),
            jnp.stack((flag("tile_polar"), flag("tile_flooded"), flag("tile_algae"), flag("unit_bubble"),
                       flag("unit_swift")), -1).astype(jnp.float32),
            age[:, None],
            jnp.stack((flag("unit_leader"), flag("unit_follower"), flag("own_unit_double_ready")), -1).astype(jnp.float32),
            (jnp.clip(field("own_structure_age"), 0, STRUCTURE_AGE_STEPS) / STRUCTURE_AGE_STEPS * known)[:, None],
            flag("tile_decomposing")[:, None].astype(jnp.float32)), -1)
        board = jnp.concatenate((board, special), -1).astype(jnp.float32)
        converts = v["converts"].reshape(-1)[0] if "converts" in v else 0
        g = jnp.concatenate((g, jnp.array([jnp.minimum(converts, CONVERT_TASK) / CONVERT_TASK]))).astype(jnp.float32)
    assert board.shape[-1] == group_width(players, obs_version) and g.shape[-1] == sum(c for _, c in global_layout(players, obs_version))
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
    v["unit_direction"] = np.where(v["explored"], 0, -1).astype(np.int32)
    # no observation history in a public demonstration (observation version 3 reads zeros)
    v["hist_enemy_unit"] = np.zeros((HISTORY_TURNS, n), np.int32)
    v["hist_enemy_built"], v["hist_enemy_trained"] = np.zeros(n, np.int32), np.zeros(n, np.int32)
    v["enemy_tech"] = np.zeros(ENEMY_TECH_ENTRIES, np.int32)
    for u in s["units"]:
        i = u["tile"]
        for a, b in (("owner", "owner"), ("type", "kind"), ("health", "health"), ("veteran", "veteran")):
            v["unit_" + a][i] = u[b]
        if "direction" in u:
            v["unit_direction"][i] = u["direction"] + 1
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
