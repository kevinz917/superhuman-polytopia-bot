"""Host-side import for validation and initial-state pools, outside the GPU loop."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .state import State, empty_state


@dataclass(frozen=True)
class Imported:
    state: State
    unsupported: tuple[str, ...]


# grid directions: SW, W, NW, N, NE, E, SE, S as (dx, dy)
_GRID = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def spawn_direction(size, x, y):
    """A new unit's facing toward the map centre (the direction of its distance vector)."""
    vx, vy, best, score = size // 2 - x, size // 2 - y, 0, 0
    for d, (dx, dy) in enumerate(_GRID):
        v = (dy * vy + dx * vx) * (1393 if dx * dx + dy * dy == 1 else 985)
        if score < v:
            best, score = d, v
    return best


def from_polyfish(raw: dict) -> Imported:
    settings = raw["settings"]
    size = int(settings["size"])
    n = size * size
    if set(raw["tiles"]) != {str(i) for i in range(n)}:
        raise ValueError("expected a complete square board")
    if set(raw["tribes"]) != {"1", "2"}:
        raise ValueError("the first GPU core supports exactly players 1 and 2")
    reasons = set()
    if settings["version"] != 115:
        reasons.add("rules_version_other_than_115")
    if raw.get("endOfTurnQueue"):
        reasons.add("deferred_end_turn_effects")
    state = jax.tree.map(lambda a: np.array(a), empty_state(size))
    t, u, c, p = state.tiles, state.units, state.cities, state.players
    for key, tile in raw["tiles"].items():
        i = int(key)
        if tile["coords"]["idx"] != i:
            raise ValueError("tile coordinate index mismatch")
        t.terrain[i], t.owner[i], t.capital[i] = tile["type"], tile.get("owner", 0), tile.get("capitalOf", 0)
        t.road[i] = tile.get("hasRoad", False)
        t.city[i] = tile.get("rulingCityCoords", {}).get("idx", -1)
        if tile["type"] not in (1, 2, 3, 4, 5):
            reasons.add("special_terrain")
        if tile.get("effects"):
            reasons.add("tile_effects")
        for owner in tile.get("explorers", []):
            if owner in (1, 2):
                t.explored[owner - 1, i] = True
    for name, field in (("structures", t.structure), ("resources", t.resource)):
        for key, value in raw.get(name, {}).items():
            i = int(key)
            if not 0 <= i < n:
                raise ValueError(f"{name} index out of range")
            if value is not None:
                field[i] = value["type"]
                if name == "structures":
                    t.structure_level[i] = value.get("level", 0)
                    t.structure_turn[i] = value.get("founded", 0)
    if not np.isin(t.structure, (0, 1, 2, 3, 5, 6, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 47, 50, 8, 48)).all():
        reasons.add("unported_structure_effects")
    if not np.isin(t.resource, (0, 1, 2, 3, 5, 6, 7, 8)).all():
        reasons.add("special_resources")
    for owner_s, tribe in raw["tribes"].items():
        owner = int(owner_s)
        pi = owner - 1
        if tribe["id"] != owner or tribe["type"] != 4:
            reasons.add("non_bardur_tribe")
        if tribe.get("killedTurn", 0) > 0 or tribe.get("resignedTurn", 0) > 0:
            pass  # elimination metadata is represented below
        for field, key in ((p.stars, "stars"), (p.score, "score"), (p.kills, "kills"),
                           (p.casualties, "casualties"), (p.pacifist, "pacifistTurns"), (p.attacked, "attackedThisTurn")):
            field[pi] = tribe.get(key, 0)
        p.killed_turn[pi] = tribe.get("killedTurn", 0)
        p.resigned_turn[pi] = tribe.get("resignedTurn", 0)
        p.killer[pi] = tribe.get("killerId", 0)
        for st in tribe.get("builtUniqueImprovements", []):
            p.built[pi, st] = True
        for other in tribe.get("knownPlayers", []):
            if other in (1, 2):
                p.known[pi, other - 1] = True
        for rel in tribe.get("relations", {}).values():
            if any(rel.get(k, 0) for k in ("state", "embassyLevel", "hasOfferedPeace", "lastPeaceBrokenTurn")):
                reasons.add("diplomacy_state")
        for tech in tribe.get("tech_vanilla", []):
            if isinstance(tech, int):
                tech = {"type": tech, "discovered": True}
            k = tech["type"]
            if not 0 <= k < 128:
                raise ValueError("technology ID out of range")
            p.tech[pi, k] = tech.get("discovered", True)
            p.tech_turn[pi, k] = tech.get("discoveredTurn", 0)
            if not p.tech[pi, k]:
                reasons.add("undiscovered_tech_record")
            if k not in (*range(25), 38, 44, 48, 121):
                reasons.add("special_technology")
        for order, city in enumerate(tribe.get("cities", [])):
            i = city["idx"]
            if not 0 <= i < n or c.owner[i] != 0 or city["owner"] != owner:
                raise ValueError("invalid/duplicate city")
            c.owner[i] = owner
            c.order[i] = order + 1
            c.connected[i] = city.get("connectedToCapital", False)
            for index in city.get("_territory", city.get("territory", [])):
                c.territory[i, index] = True
            for field, key in ((c.level, "level"), (c.population, "population"), (c.progress, "progress"),
                               (c.production, "production"), (c.border, "borderSize")):
                field[i] = city.get(key, 0)
            for reward in city.get("rewards", []):
                if not 0 <= reward < 9:
                    raise ValueError("invalid city reward")
                c.rewards[i, reward] += 1
        for unit in tribe.get("units", []):
            i = unit["coords"]["idx"]
            if not 0 <= i < n or u.owner[i] != 0 or unit["owner"] != owner:
                raise ValueError("invalid/duplicate unit")
            u.owner[i], u.kind[i] = owner, unit["type"]
            u.home[i] = (unit.get("homeCoords") or {}).get("idx", -1) + 1
            u.passenger[i] = unit.get('passengerType') or 0
            u.invisible[i] = 3 in unit.get('effects', [])
            u.created[i] = unit.get("createdTurn", 0)
            u.converted[i] = unit.get("converted", False)
            u.attacks_performed[i] = unit.get("attacksPerformed", 0)
            u.prev_x[i] = unit.get("prevCoords", {}).get("x", -1)
            u.prev_y[i] = unit.get("prevCoords", {}).get("y", -1)
            u.last_attack[i] = (unit.get("lastAttackCoords") or {}).get("idx", -1) + 1
            u.direction[i] = unit.get("direction", spawn_direction(size, i % size, i // size))
            for field, key in ((u.health, "health"), (u.veteran, "veteran"), (u.moved, "moved"),
                               (u.attacked, "attacked"), (u.kills, "kills")):
                field[i] = unit.get(key, 0)
            effects = unit.get("effects", [])
            u.frozen[i], u.poison[i], u.boosted[i] = 0 in effects, 1 in effects, 2 in effects
            if any(e not in (0, 1, 2) for e in effects):
                reasons.add("unported_unit_effect")
            if u.kind[i] not in (2, 3, 4, 5, 8, 9, 10, 11, 12, 41, 42, 43, 44, 45):
                reasons.add("unported_unit_type")
            if unit.get("parentUnitIdx") is not None or unit.get("childUnitIdx") is not None:
                reasons.add("passenger_or_segment")
    if not settings.get("_fow", True) and not t.explored.all():
        reasons.add("uninitialized_no_fog_state")
    active = int(settings["currentPlayerTurnId"])
    if active not in (1, 2):
        raise ValueError("invalid current player")
    dead = any(tribe.get("killedTurn", 0) > 0 or tribe.get("resignedTurn", 0) > 0 for tribe in raw["tribes"].values())
    game_over = bool(settings.get("_gameOver", False))
    state = state._replace(
        current_player=np.int32(active), turn=np.int32(settings["turn"]), version=np.int32(settings["version"]),
        fog=np.bool_(settings.get("_fow", True)), reference_max_turns=np.int32(settings["maxTurns"]),
        terminated=np.bool_(game_over and dead), truncated=np.bool_(game_over and not dead),
        core_supported=np.bool_(not reasons), seed=np.uint32(settings.get("seed", 0) & 0xffffffff), initial_seed=np.uint32(raw.get("initialSeed", 0) & 0xffffffff),
    )
    return Imported(jax.tree.map(jnp.asarray, state), tuple(sorted(reasons)))
