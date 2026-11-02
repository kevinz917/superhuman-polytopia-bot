"""Strict public observation schema and response-local action catalog.

This accepts already-redacted observations, never engine State or snapshots.
The eventual GPU adapter must independently prove that its redaction is correct.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from polytopia_bot.environment import LegalAction, Observation

SCHEMA = "polytopia-public-v2"
SYSTEM = """Play Polytopia using only the supplied public observation and legal catalog.
Unknown tiles have no known terrain or contents. Remembered facts may be stale.
Choose exactly one listed action. Return only {\"action_id\":\"aN\"}, with no prose.
Commands use (kind,source,target,parameter); tiles are row-major indices.
Kinds: 1 move, 2 attack, 3 ability, 4 summon, 5 harvest, 6 build,
7 research, 8 capture, 9 city reward, 10 end turn, 11 resign.
Ability parameters: 1 burn forest, 2 clear forest, 3 grow forest, 4 destroy,
7 recover, 8 disband, 9 heal, 16 promote, 18 break peace, 19 respond to peace,
20 embassy, 21 offer peace, 22 remove embassy. Diplomatic source is a player ID;
peace response target is 0 (reject) or 1 (accept).
Terrain IDs: 1 water, 2 ocean, 3 field, 4 mountain, 5 forest, 6 ice,
7 wetland, 8 mangrove. Resource IDs: 0 none, 1 animal, 2 crop, 3 fish,
5 metal, 6 fruit, 7 spores, 8 starfish, 9 aquatic crop.
City rewards: 1 wall, 2 park, 3 workshop, 4 explorer, 5 border growth,
6 superunit, 7 resources, 8 population. The catalog determines legal choices.
Capture cities and preserve your empire; use income, research and exploration
to support expansion and combat. End turn when no useful action remains."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@lru_cache(maxsize=1)
def rule_table():
    return json.loads((Path(__file__).resolve().parents[1] / "gpu" / "rules.json").read_text())


def rule_glossary(state, actions):
    """Public static definitions for IDs actually present in this observation."""
    v, table = state["visible"], rule_table()
    unit_ids = {u["kind"] for u in v["units"]} | {a.command["parameter"] for a in actions if a.command["kind"] == 4}
    tech_ids = set(v["technology"]) | {a.command["parameter"] for a in actions if a.command["kind"] == 7}
    structure_ids = {t["structure"] for t in v["tiles"] if "structure" in t} | {a.command["parameter"] for a in actions if a.command["kind"] == 6}
    definitions = {}
    for name, ids, fields in (("units", unit_ids, ("id", "name", "cost", "health", "attack", "defense", "movement", "range", "skills")),
                              ("technologies", tech_ids, ("id", "name", "tier", "requires", "structures")),
                              ("structures", structure_ids, ("id", "name", "cost", "population", "stars"))):
        rows = {r["id"]: r for r in table[name]}
        if ids - rows.keys():
            raise ValueError(f"unknown public {name} ID")
        definitions[name] = [{k: rows[i][k] for k in fields} for i in sorted(ids)]
    return definitions


def keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("invalid public schema fields")


def integer(value, lo=0, hi=None):
    if type(value) is not int or value < lo or (hi is not None and value > hi):
        raise ValueError("invalid public integer")
    return value


def command(value):
    keys(value, ("kind", "source", "target", "parameter"))
    integer(value["kind"], 1, 11)
    for k in ("source", "target", "parameter"):
        integer(value[k], -1)
    return dict(value)


def public_state(obs: Observation) -> dict:
    # JSON conversion rejects non-JSON objects, NaNs and nested engine objects.
    v = json.loads(canonical(obs.visible_state))
    # Seats: 2, or one per entry of the relation lists (four-seat games, docs/four-player-2026-10-24.md).
    seats = len(v["relations"]["known_players"]) if isinstance(v.get("relations"), dict) and isinstance(
        v["relations"].get("known_players"), list) else 2
    if seats not in (2, 4):
        raise ValueError("invalid seat count")
    integer(obs.player, 1, seats)
    integer(obs.state_version)
    keys(v, ("board_size", "turn", "stars", "technology", "tiles", "units", "cities"), ("relations", "achievements"))
    n = integer(v["board_size"], 2, 64) ** 2
    integer(v["turn"])
    integer(v["stars"])
    if not isinstance(v["technology"], list):
        raise ValueError("technology must be a list")
    v["technology"] = sorted({integer(t, 0, 127) for t in v["technology"]})
    for name in ("tiles", "units", "cities"):
        if not isinstance(v[name], list):
            raise ValueError("entities must be lists")
        seen = set()
        for row in v[name]:
            if not isinstance(row, dict):
                raise ValueError("entity must be an object")
            tile = integer(row.get("tile"), 0, n - 1)
            if tile in seen:
                raise ValueError("duplicate tile/entity")
            seen.add(tile)
            if name == "tiles":
                keys(row, ("tile", "known"), ("terrain", "owner", "resource", "structure", "road", "capital", "structure_level"))
                if type(row["known"]) is not bool:
                    raise ValueError("known must be boolean")
                if not row["known"] and set(row) != {"tile", "known"}:
                    raise ValueError("unknown tile contains hidden fields")
                for k in set(row) - {"tile", "known"}:
                    integer(row[k])
                if "owner" in row:
                    integer(row["owner"], 0, seats)
            elif name == "units":
                flags = ("veteran", "moved", "attacked", "poison", "frozen", "boosted", "invisible", "detect")
                own_fields = ("kills", "home", "passenger", *flags[1:])
                keys(row, ("tile", "owner", "kind", "health"), (*flags, "kills", "home", "passenger"))
                if row["owner"] != obs.player and set(row) & set(own_fields):
                    raise ValueError("opponent unit internals are not public")
                integer(row["owner"], 1, seats)
                integer(row["kind"], 1, 127)
                if type(row["health"]) not in (int, float) or not math.isfinite(row["health"]) or row["health"] <= 0:
                    raise ValueError("invalid unit health")
                for k in flags:
                    if k in row and type(row[k]) is not bool:
                        raise ValueError("invalid unit flag")
                for k in ("kills", "home", "passenger"):
                    if k in row:
                        integer(row[k])
            else:
                keys(row, ("tile", "owner"), ("level", "population", "production", "progress", "connected", "riot", "rewards"))
                integer(row["owner"], 0, seats)
                if row["owner"] != obs.player and set(row) - {"tile", "owner"}:
                    raise ValueError("opponent city internals are not in this schema")
                for k in set(row) - {"tile", "owner"}:
                    if k in ("connected", "riot"):
                        if type(row[k]) is not bool:
                            raise ValueError("invalid city flag")
                    elif k == "rewards":
                        if not isinstance(row[k], list) or len(row[k]) != 9:
                            raise ValueError("invalid city rewards")
                        for count in row[k]:
                            integer(count)
                    elif k in ("progress", "population"):
                        # Frozen native starvation retains city levels/rewards
                        # while its population progress can become negative.
                        integer(row[k], -(2**31), 2**31 - 1)
                    else:
                        integer(row[k])
        v[name] = sorted(v[name], key=lambda r: r["tile"])
    if len(v["tiles"]) != n:
        raise ValueError("include every tile, with unknown tiles explicitly marked")
    known = {t["tile"] for t in v["tiles"] if t["known"]}
    if any(e["tile"] not in known for name in ("units", "cities") for e in v[name]):
        raise ValueError("entity on unknown tile")
    if "relations" in v:
        keys(v["relations"], ("known_players", "peace", "embassies_sent", "embassies_received", "peace_offers"))
        for value in v["relations"].values():
            if not isinstance(value, list) or len(value) != seats or any(type(x) is not bool for x in value):
                raise ValueError("invalid relations")
    if "achievements" in v:
        keys(v["achievements"], ("monuments", "built"))
        for value in v["achievements"].values():
            if not isinstance(value, list):
                raise ValueError("invalid achievements")
            for item in value:
                integer(item, 0, 127)
    memory = json.loads(canonical(obs.remembered_state))
    keys(memory, (), ("tiles", "schema", "recent_actions", "historical"))
    if set(memory) & {"schema", "recent_actions", "historical"}:
        from .memory import validate_memory
        validate_memory(memory, obs)
    memory.setdefault("tiles", [])
    if not isinstance(memory["tiles"], list):
        raise ValueError("memory tiles must be a list")
    seen = set()
    for row in memory["tiles"]:
        keys(row, ("tile", "last_seen_turn"), ("terrain", "owner", "resource", "structure"))
        tile = integer(row["tile"], 0, n - 1)
        if tile in seen:
            raise ValueError("duplicate memory tile")
        seen.add(tile)
        integer(row["last_seen_turn"], 0, v["turn"])
        for k in set(row) - {"tile", "last_seen_turn"}:
            integer(row[k])
        if "owner" in row:
            integer(row["owner"], 0, 2)
    memory["tiles"].sort(key=lambda r: r["tile"])
    return {"schema": SCHEMA, "player": obs.player, "state_version": obs.state_version,
            "visible": v, "remembered": memory}


@dataclass(frozen=True)
class Prompt:
    messages: list[dict]
    catalog: dict[str, LegalAction]
    state_version: int
    public: dict


def render(obs: Observation) -> Prompt:
    state = public_state(obs)
    actions = []
    ids, commands = set(), set()
    for a in obs.legal_actions:
        if not isinstance(a.action_id, str) or not a.action_id or a.action_id in ids:
            raise ValueError("invalid or duplicate engine action ID")
        c = command(dict(a.command))
        for field in ("source", "target"):
            integer(c[field], -1, state["visible"]["board_size"] ** 2 - 1)
        key = canonical(c)
        if key in commands:
            raise ValueError("duplicate semantic action")
        ids.add(a.action_id)
        commands.add(key)
        # Copy mutable caller data: a later mutation cannot change this prompt.
        actions.append(((c["kind"], c["source"], c["parameter"], c["target"]), LegalAction(a.action_id, c)))
    if not actions:
        raise ValueError("decision requires at least one legal action")
    catalog = {f"a{i}": a for i, (_, a) in enumerate(sorted(actions))}
    state["rules"] = rule_glossary(state, list(catalog.values()))
    state["legal_actions"] = [{"action_id": k, "command": dict(a.command)} for k, a in catalog.items()]
    from .encoding import compact, INSTRUCTIONS
    return Prompt([{"role": "system", "content": SYSTEM + "\n" + INSTRUCTIONS}, {"role": "user", "content": canonical(compact(state))}], catalog, obs.state_version, state)


class ResponseError(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ResponseError("duplicate_key")
        result[k] = v
    return result


def parse_response(text: str, prompt: Prompt, *, current_version: int) -> LegalAction:
    if current_version != prompt.state_version:
        raise ResponseError("stale_version")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError):
        raise ResponseError("invalid_json") from None
    if not isinstance(value, dict) or set(value) != {"action_id"} or not isinstance(value["action_id"], str):
        raise ResponseError("invalid_response_shape")
    if value["action_id"] not in prompt.catalog:
        raise ResponseError("unknown_action_id")
    return prompt.catalog[value["action_id"]]


@dataclass
class Generation:
    text: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]


class PolicyForfeit(RuntimeError):
    def __init__(self, attempts):
        super().__init__("three invalid model responses; no fallback action")
        self.attempts = attempts


def choose_action(obs: Observation, generate: Callable[[list[dict]], Generation]):
    """Return engine action + all attempts; engine.step must recheck the version."""
    prompt = render(obs)
    messages, attempts = list(prompt.messages), []
    for _ in range(3):
        output = generate(messages)
        attempt = {"messages": list(messages), "text": output.text,
                   "prompt_token_ids": output.prompt_token_ids, "response_token_ids": output.response_token_ids}
        attempts.append(attempt)
        try:
            selected = parse_response(output.text, prompt, current_version=obs.state_version)
            attempt["error"] = None
            return selected, attempts
        except ResponseError as e:
            attempt["error"] = str(e)
            messages = messages + [{"role": "assistant", "content": output.text},
                                   {"role": "user", "content": f"Invalid response: {e}. Return one legal action_id JSON object."}]
    raise PolicyForfeit(attempts)
