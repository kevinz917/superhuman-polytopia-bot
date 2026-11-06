"""Small immutable tables exported by the pinned Rust oracle, not hand guessed."""
import json
from importlib.resources import files
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import ENGINE_REVISION


FLOAT, CARRY, SCOUT, CREEP, HIDE, INDEPENDENT, SURPRISE, SPLASH, STOMP, INFILTRATE = range(10)


class StructureRules(NamedTuple):
    cost: jax.Array
    population: jax.Array
    score: jax.Array
    stars: jax.Array
    terrain: jax.Array
    adjacent: jax.Array
    resource: jax.Array
    per_city: jax.Array
    per_tribe: jax.Array
    allowed: jax.Array
    unlock: jax.Array


class Rules(NamedTuple):
    native: jax.Array  # False only for pinned-Polyfish differential validation
    unit_health: jax.Array
    unit_attack: jax.Array
    unit_defense: jax.Array
    unit_movement: jax.Array
    unit_range: jax.Array
    dash: jax.Array
    unit_supported: jax.Array
    tech_tier: jax.Array
    tech_edges: jax.Array
    tech_exists: jax.Array
    unit_cost: jax.Array
    fortify: jax.Array
    stiff: jax.Array
    persist: jax.Array
    escape: jax.Array
    static: jax.Array
    unit_tech: jax.Array
    structures: StructureRules
    free_tech: jax.Array
    free_requires: jax.Array
    skills: jax.Array  # [unit, named skill ABI below]


def load_rules(profile="native") -> Rules:
    if profile not in ("native", "polyfish"):
        raise ValueError("rules profile must be native or polyfish")
    data = json.loads(files(__package__).joinpath("rules.json").read_text())
    if data["engine_revision"] != ENGINE_REVISION:
        raise ValueError("rule table and pinned oracle revision differ")
    attrs = [np.zeros(128, np.float32) for _ in range(3)]
    movement = np.zeros(128, np.int32)
    ranges = np.zeros(128, np.int32)
    dash = np.zeros(128, bool)
    supported = np.zeros(128, bool)
    # Ordinary land movement/recovery only; special skills are not erased.
    core_names = {"Raft", "Rammership", "Scoutship", "Bomber", "Juggernaut", "MindBender", "Warrior", "Rider", "Knight", "Defender", "Catapult", "Archer", "Swordsman", "Giant"}
    if profile == "native":
        core_names |= {"Cloak", "Dagger", "Dinghy", "Pirate"}
    for u in data["units"]:
        i = u["id"]
        for a, name in zip(attrs, ("health", "attack", "defense")):
            a[i] = u[name]
        movement[i], ranges[i] = u["movement"], u["range"]
        dash[i] = "Dash" in u["skills"]
        supported[i] = u["name"] in core_names
    tiers = np.ones(128, np.int32)
    edges = np.zeros((128, 128), bool)
    exists = np.zeros(128, bool)
    for t in data["technologies"]:
        i = t["id"]
        if i < 0:
            continue
        tiers[i], exists[i] = t["tier"], True
        for other in t["next"] + ([] if t["requires"] is None else [t["requires"]]):
            if other >= 0:
                edges[i, other] = True
    # Only the vanilla tech tree is supported in this rule slice.
    exists[25:] = False
    exists[38] = True  # Diplomacy is vanilla; executing diplomacy is unported.
    exists[44] = True  # Ramming is part of the vanilla naval branch.
    skills = np.zeros((128, 10), bool)
    for unit in data['units']:
        for i, name in enumerate(('Float','Carry','Scout','Creep','Hide','Independent','Surprise','Splash','Stomp','Infiltrate')):
            skills[unit['id'], i] = name in unit['skills']
    costs = np.zeros(128, np.int32)
    skill_arrays = [np.zeros(128, bool) for _ in range(5)]
    unit_tech = np.full(128, 127, np.int32)
    for unit, tech in ((2, 0), (3, 1), (4, 3), (5, 7), (8, 17), (9, 18), (10, 22), (11, 24), (38, 38), (41, 14), (42, 13), (44, 44)):
        unit_tech[unit] = tech
    for unit in data["units"]:
        costs[unit["id"]] = unit["cost"]
        for arr, skill in zip(skill_arrays, ("Fortify", "Stiff", "Persist", "Escape", "Static")):
            arr[unit["id"]] = skill in unit["skills"]
    table = {k: np.zeros(128, np.int32) for k in ('cost', 'population', 'score', 'stars', 'resource')}
    table['cost'][:] = -1
    terrain = np.zeros((128, 16), bool)
    adjacent = np.zeros((128, 128), bool)
    per_city, per_tribe, allowed = (np.zeros(128, bool) for _ in range(3))
    unlock = np.zeros((128, 128), bool)
    for st in data['structures']:
        i=st['id']
        for key in table:
            table[key][i] = st[key] if st[key] is not None else (-1 if key == 'cost' else 0)
        terrain[i, st['terrain']] = True
        adjacent[i, st['adjacent']] = True
        per_city[i], per_tribe[i], allowed[i] = st['per_city'], st['per_tribe'], st['tribe'] in (None, 4)
    for tech in data['technologies']:
        if tech['id'] >= 0:
            unlock[tech['id'], tech['structures']] = True
    free_tech=np.zeros(128,bool)
    free_requires=np.full(128,-1,np.int32)
    for tech in data['technologies']:
        i=tech['id']
        if i in (*range(1,25),38,44,48,121):
            free_tech[i]=True
            free_requires[i]=tech['requires'] if tech['requires'] is not None else -1
    if profile == "native":
        free_tech &= exists
    structs = StructureRules(*(jnp.asarray(a) for a in [table['cost'], table['population'], table['score'], table['stars'],
                               terrain, adjacent, table['resource'], per_city, per_tribe, allowed, unlock]))
    return Rules(jnp.bool_(profile == "native"), *(jnp.asarray(a) for a in [*attrs, movement, ranges, dash, supported, tiers, edges, exists,
                                          costs, *skill_arrays, unit_tech]), structs, jnp.asarray(free_tech), jnp.asarray(free_requires), jnp.asarray(skills))
