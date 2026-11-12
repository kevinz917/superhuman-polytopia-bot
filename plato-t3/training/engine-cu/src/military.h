/* military.h - line-by-line port of engine-rs/src/military.rs (gpu/military.py: unit actions, native profile).
 *
 * Every `pub` item of military.rs is `pc_military_<name>`; private helpers keep their Rust name under the same
 * prefix. Argument order follows the Rust. Rust `Option<usize>` -> int with -1 = None; `Option<&[bool]>` -> a
 * `const uint8_t *` that may be NULL; `impl fn mut(usize)` emitters -> an out list + count. */
#ifndef PC_MILITARY_H
#define PC_MILITARY_H

#include "pc.h"
#include "hash.h"         /* military.rs draws no hash (push is deterministic); included so the module set reads alike */
#include "rules.h"        /* pc_rules_bits128 (rules:bits128), pc_cost_le (cost index:le) */
#include "kernels.h"      /* pc_kernels_combat (tuple -> out-params), pc_kernels_max_health, pc_kernels_bridge_blocks */
#include "diplomacy.h"    /* pc_diplomacy_protected, pc_diplomacy_hostile */
#include "economy.h"      /* pc_economy_discover */
#include "structures.h"   /* pc_structures_current_capital (Option<usize> -> tile or -1) */
#include "cymanti.h"      /* spores, step-on-poison, poison splash, the Algae trail (Cymanti step 4) */
#include "polaris.h"      /* pc_polaris_auto_freeze (Polaris) */
#include "aquarion.h"     /* floodable, flood_tile, after_move (Aquarion step 4) */

#ifdef __cplusplus
extern "C" {
#endif

/* forward declarations (same translation unit) */
PC_FN void pc_military_area_damage(PcState *s, int source, int center, const PcRules *r, const PcGeom *g, uint8_t stomp);
PC_FN int32_t pc_military_direction(int size, int from, int to);
PC_FN int32_t pc_military_steps_dx(int k);
PC_FN int32_t pc_military_steps_dy(int k);
PC_FN float pc_military_defense_bonus(const PcState *s, const PcRules *r, int i);
PC_FN void pc_military_convert(PcState *s, int source, int target, const PcRules *r, const PcGeom *g);
PC_FN int pc_military_tentacle_trap(const PcState *s, int t, int32_t mover, const PcRules *r, float *damage);
PC_FN void pc_military_trap_attack(PcState *s, int jelly, int t, float damage, const PcRules *r);
PC_FN int32_t pc_military_spawn_direction(int size, int tile);
PC_FN int pc_military_attack_targets(const PcState *s, int source, const PcRules *r, const PcGeom *g, uint16_t *out);

PC_FN void pc_military_exhaust(PcState *s, int source) {
    s->units.moved[source] = 1;
    s->units.attacked[source] = 1;
    s->units.move_lock[source] = 0;
    s->units.double_ready[source] = 0;   /* make exhausted removes double ready */
}

/* make exhauseted (the city capture, ruins, clear tile effect by a unit): moved and attacked, double ready and
 * Boosted removed */
PC_FN void pc_military_make_exhausted(PcState *s, int source) {
    pc_military_exhaust(s, source);
    s->units.boosted[source] = 0;
    /* then disable followers: every follower down the chain is moved and attacked (Cymanti) */
    int32_t f = s->units.follower[source];
    int guard = 0;
    while (f > 0 && guard < s->n) {
        int i = (int)f - 1;
        s->units.moved[i] = 1;
        s->units.attacked[i] = 1;
        s->units.move_lock[i] = 0;
        f = s->units.follower[i];
        guard++;
    }
}

/* military.remove: killer=None mirrors killer=-1 (index clipped to 0, no credit). `killer` < 0 is None. */
PC_FN void pc_military_remove(PcState *s, int target, const PcRules *r, int killer, uint8_t casualty) {
    const PcUnits *u = &s->units;
    int owner = pc_pclip(s, u->owner[target]);
    int k = killer >= 0 ? killer : 0;
    int kp = pc_pclip(s, u->owner[k]);
    int kind = u->kind[target];
    int32_t passenger = u->passenger[target];
    int32_t value = r->unit_cost[kind] + (passenger > 0 ? r->unit_cost[passenger] : 0);
    int32_t loss = -5 * value;
    s->players.score[owner] = pc_wadd(s->players.score[owner], loss);
    s->players.casualties[owner] += casualty ? 1 : 0;
    s->players.kills[kp] += killer >= 0 ? 1 : 0;
    /* attack action (game version >= 95): no xp for a killer that is a vehicle (Carry: every
     * boat) or Static; the passenger of a boat keeps the xp it had when it embarked */
    uint8_t xp = killer >= 0 && !(r->skills[u->kind[k]][PC_SK_CARRY] || r->stat[u->kind[k]]);
    /* kill unit (Cymanti): a follower becomes a Centipede head, a leader loses the unit (disband unit action
     * and explode unit action unlink first and handle their chains themselves) */
    pc_cymanti_kill_links(s, target, r);
    /* kill unit: a Poisoned unit leaves Spores (spawn spore resource, Cymanti) */
    if (s->units.poison[target]) pc_cymanti_spawn_spores(s, target);
    s->units.kills[k] += xp ? 1 : 0;
    pc_units_clear(&s->units, target);
}

/* military:grow (Elyrion upgrade action of a Grow unit, cost 0): the grows_into type with the old damage (at least
 * 0.1), created this turn, no xp or promotion; score + 5 x cost difference; explores the new sight */
PC_FN void pc_military_grow(PcState *s, int i, const PcRules *r, const PcGeom *g) {
    int n = s->n;
    int old = (int)s->units.kind[i];
    int into = (int)r->grows_into[old];
    float damage = pc_kernels_max_health(s, r, i) - s->units.health[i];
    s->units.kind[i] = (int32_t)into;
    s->units.veteran[i] = 0;
    s->units.kills[i] = 0;
    s->units.created[i] = s->turn;
    s->units.health[i] = pc_fmaxf(r->unit_health[into] - damage, 0.1f);
    int32_t owner = s->units.owner[i];
    int o = pc_pwrap(s, owner);
    s->players.score[o] = pc_wadd(s->players.score[o], 5 * (r->unit_cost[into] - r->unit_cost[old]));
    int radius = (s->tiles.terrain[i] == 4 || r->skills[into][PC_SK_SCOUT]) ? 2 : 1;
    uint8_t reveal[PC_MAXN];
    for (int t = 0; t < n; t++) reveal[t] = (uint8_t)(pc_d(g, i, t) <= radius);
    pc_economy_discover(s, reveal, owner);
    /* upgrade action's train unit: check step on poison */
    pc_cymanti_step_on_poison(s, i, r);
}

/* military:grow_on_landing (disembark action's Grow branch): a Grow passenger older than 2 turns lands grown */
PC_FN void pc_military_grow_on_landing(PcState *s, int i, const PcRules *r, const PcGeom *g) {
    int kind = (int)s->units.kind[i];
    if (r->grow[kind] && r->grows_into[kind] >= 0 && s->turn - s->units.created[i] > 2) pc_military_grow(s, i, r, g);
}

/* relocate; `stomp` false for a move action with move reason 2 (a push), which skips the arrival abilities.
 * `reveal` NULL = None (the unit's own sight is revealed). `sc`: the routes scratch of an AutoFreeze arrival
 * (polaris:auto_freeze runs update routes); NULL only with stomp false. */
PC_FN void pc_military_relocate_one(PcState *s, int source, int target, const PcRules *r, const PcGeom *g, uint8_t involuntary,
                                    const uint8_t *reveal, uint8_t stomp, PcStructScratch *sc) {
    int n = s->n, size = s->size;
    int32_t owner = s->units.owner[source];
    int kind = s->units.kind[source];
    int32_t passenger_src = s->units.passenger[source];
    uint8_t moved_src = s->units.moved[source];
    uint8_t attacked_src = s->units.attacked[source];
    uint8_t invisible_src = s->units.invisible[source];
    int32_t terrain_src = s->tiles.terrain[source];
    uint8_t bubble_src = s->units.bubble[source];
    pc_units_move(&s->units, source, target);
    /* Cymanti: move action's check step on poison, any reason */
    pc_cymanti_step_on_poison(s, target, r);
    /* Aquarion: the Tentacle trap of every move action, any reason (computed now, applied after the arrival actions) */
    float trap_damage = 0.0f;
    int trap = pc_military_tentacle_trap(s, target, owner, r, &trap_damage);
    s->units.moved[target] = 1;
    s->units.attacked[target] = 1;
    s->units.prev_x[target] = (int32_t)(source % size);
    s->units.prev_y[target] = (int32_t)(source / size);
    /* move action: set unit direction from the path's last step (core:move_adjacent refines multi-tile moves) */
    s->units.direction[target] = pc_military_direction(size, source, target);
    uint8_t scout = r->skills[kind][PC_SK_SCOUT];
    uint8_t own_reveal[PC_MAXN];
    if (reveal == NULL) {
        int rs = (terrain_src == 4 || scout) ? 2 : 1;
        int rt = (s->tiles.terrain[target] == 4 || scout) ? 2 : 1;
        for (int t = 0; t < n; t++) own_reveal[t] = (uint8_t)(pc_d(g, source, t) <= rs || pc_d(g, target, t) <= rt);
        reveal = own_reveal;
    }
    pc_economy_discover(s, reveal, owner);
    if (stomp && r->algae_trail[kind]) {
        /* Cymanti: the Island's Algae trail on the path's ends (core adds the tiles between) */
        uint16_t ends[2] = {(uint16_t)target, (uint16_t)source};
        pc_cymanti_algae_trail(s, ends, 2);
    }
    if (stomp && (r->skills[kind][PC_SK_STOMP] || r->tentacles[kind])) {
        /* Stomp, and a Tentacles unit's area damage on arrival (Aquarion) */
        pc_military_area_damage(s, target, target, r, g, 1);
    }
    int32_t tt = s->tiles.terrain[target];
    /* will unit embark (v >= 120): a land unit embarks on a friendly Port on water or on a wetland (Flooded)
     * tile (a Port on plain Ice is land, Polaris), when the player can embark (Fishing); a vehicle disembarks onto a
     * City or onto land that is not wetland (military:relocate_as) */
    uint8_t flooded_t = s->tiles.flooded[target];
    /* a flyer never embarks (will unit embark: a land unit is neither aquatic nor Fly) */
    /* Cymanti: an Algae water tile is entered without the Port: its friendly Port and can player embark checked here */
    uint8_t algae_t = s->tiles.algae[target];
    uint8_t embark = s->tiles.structure[target] == 8 && !r->skills[kind][PC_SK_FLOAT] && !r->fly[kind] && ((tt <= 2 && !algae_t) || ((flooded_t || algae_t) && (
        pc_tribes_can_embark(s->players.tribe[pc_pclip(s, owner)], s->players.tech[pc_pclip(s, owner)]) && (s->tiles.owner[target] == owner
            || (s->tiles.owner[target] > 0 && s->players.peace[pc_pclip(s, owner)][pc_pclip(s, s->tiles.owner[target])])))));
    uint8_t landing = tt > 2 && (!flooded_t || s->tiles.structure[target] == 1);
    uint8_t special_land = (kind == 39 || kind == 40 || kind == 45) && landing;
    uint8_t disembark = r->skills[kind][PC_SK_CARRY] && (passenger_src > 0 || special_land) && landing && !embark;
    int new_kind;
    if (embark) {
        new_kind = kind == 12 ? 45 : (kind == 38 ? 39 : (kind == 37 ? 40 : 43));
    } else if (disembark) {
        new_kind = passenger_src;
    } else {
        new_kind = kind;
    }
    if (special_land) {
        new_kind = kind == 39 ? 38 : (kind == 40 ? 37 : 12);
    }
    int32_t passenger = (embark && kind != 12 && kind != 38 && kind != 37) ? (int32_t)kind : (disembark ? 0 : passenger_src);
    int32_t score_before = r->unit_cost[kind] + (passenger_src > 0 ? r->unit_cost[passenger_src] : 0);
    int32_t score_after = r->unit_cost[new_kind] + (passenger > 0 ? r->unit_cost[passenger] : 0);
    /* embark action / disembark action set moved and attacked on the new unit: no Dash after either
     * move action (v > 101): a Skate unit that moves off the ice for any reason but a push is done
     * for the turn (moved and attacked) */
    uint8_t skate_off = stomp && r->skate[kind] && tt != PC_ICE;
    uint8_t dash = !involuntary && !moved_src && !attacked_src && r->dash[new_kind] && !embark && !disembark && !skate_off;
    s->units.kind[target] = (int32_t)new_kind;
    s->units.passenger[target] = passenger;
    s->units.attacked[target] = dash ? 0 : (uint8_t)(s->units.attacked[target] || disembark);
    if (embark && r->skills[new_kind][PC_SK_STOMP] && !r->skills[kind][PC_SK_STOMP]) {
        /* embark action: a new boat with Stomp (Giant -> Juggernaut) stomps where it embarked */
        pc_military_area_damage(s, target, target, r, g, 1);
    }
    uint8_t hidden = r->skills[new_kind][PC_SK_HIDE] && !involuntary;
    s->units.invisible[target] = (uint8_t)(s->units.invisible[target] || hidden);
    s->units.attacked[target] = (uint8_t)(s->units.attacked[target] || (hidden && !invisible_src));
    /* military.rs: the move alone set `attacked` (Units::move_lock) */
    s->units.move_lock[target] = (uint8_t)(s->units.attacked[target] && !attacked_src && !embark && !disembark && !(hidden && !invisible_src)
                                           && !skate_off);
    int o = pc_pwrap(s, owner);
    s->players.score[o] = pc_wadd(s->players.score[o], 5 * (score_after - score_before));
    if (disembark) pc_military_grow_on_landing(s, target, r, g);
    /* Aquarion: Bubble loss for any move (a push too), AutoFlood (Crab) for any but a push */
    uint8_t flooded_at_move = pc_aquarion_after_move(s, target, bubble_src, owner, r, g, stomp, sc);
    if (stomp) pc_polaris_auto_freeze(s, target, new_kind, r, g, sc);
    if (trap >= 0) pc_military_trap_attack(s, trap, target, trap_damage, r);
    /* the move's own check bubble step, pushed first, runs last */
    pc_aquarion_bubble_on_arrival(s, target, bubble_src, flooded_at_move, owner, r);
}

/* military:relocate_as: relocate_one, then (move action v > 119, Cymanti) the follower, when 2 or more tiles away
 * from its leader's new tile, gets move action([leader's origin, follower], reason Command), whose own move pulls its
 * follower (the Rust recursion as a loop). A chain unit is never an AutoFreeze type: `sc` may stay NULL. */
PC_FN void pc_military_relocate_as(PcState *s, int source, int target, const PcRules *r, const PcGeom *g, uint8_t involuntary,
                                   const uint8_t *reveal, uint8_t stomp, PcStructScratch *sc) {
    pc_military_relocate_one(s, source, target, r, g, involuntary, reveal, stomp, sc);
    int from = source, to = target, guard = 0;
    while (s->units.follower[to] > 0 && guard++ < s->n) {
        int f = (int)s->units.follower[to] - 1;
        if (!(pc_d(g, f, to) >= 2 && s->units.owner[from] == 0)) break;
        pc_military_relocate_one(s, f, from, r, g, 0, NULL, 1, sc);
        to = from;
        from = f;
    }
}

PC_FN void pc_military_relocate(PcState *s, int source, int target, const PcRules *r, const PcGeom *g, uint8_t involuntary,
                                const uint8_t *reveal, PcStructScratch *sc) {
    pc_military_relocate_as(s, source, target, r, g, involuntary, reveal, 1, sc);
}

/* The land unit a boat lands as: its passenger, or the land form our engine keeps implicit for a Dinghy (Cloak),
 * Pirate (Dagger) and Juggernaut (Giant), as relocate's landing does. */
PC_FN int pc_military_landed_kind(int kind, int32_t passenger) {
    switch (kind) {
        case 39: return 38;
        case 40: return 37;
        case 45: return 12;
        default: return (int)passenger;
    }
}

/* can disembark (disembark command, v > 104): a boat with a passenger standing on a Bridge
 * (improvement ability bridge; the other case, a wetland tile, does not occur in these rules). No moved, attacked
 * or owner check of its own: command utils lists unit commands for the mover's units only. */
PC_FN uint8_t pc_military_can_disembark(const PcState *s, const PcRules *r, int i) {
    int kind = pc_clamp(s->units.kind[i], 0, 127);
    uint8_t carries = s->units.passenger[i] > 0 || kind == 39 || kind == 40 || kind == 45;
    return s->units.owner[i] != 0 && r->skills[kind][PC_SK_CARRY] && carries && (s->tiles.structure[i] == 48 || pc_wetland(s, i));
}

/* disembark action (v >= 109) for a boat staying on its tile: the passenger takes the boat's place
 * with the boat's health (no boat of these rules has Protect), its facing (direction and flip copied) and
 * moved = attacked = true; decrease score action takes the boat's part of the score (the amounts of a landing move).
 * No exploration and no arrival ability (no stomp, no hiding); a Grow passenger may land grown. */
PC_FN void pc_military_disembark(PcState *s, int i, const PcRules *r, const PcGeom *g) {
    int kind = s->units.kind[i];
    int32_t passenger = s->units.passenger[i];
    int new_kind = pc_military_landed_kind(kind, passenger);
    int32_t before = r->unit_cost[kind] + (passenger > 0 ? r->unit_cost[passenger] : 0);
    int32_t after = r->unit_cost[new_kind];
    s->units.kind[i] = (int32_t)new_kind;
    s->units.passenger[i] = 0;
    s->units.moved[i] = 1;
    s->units.attacked[i] = 1;
    s->units.move_lock[i] = 0;
    int o = pc_pwrap(s, s->units.owner[i]);
    s->players.score[o] = pc_wadd(s->players.score[o], 5 * (after - before));
    pc_military_grow_on_landing(s, i, r, g);
}

/* The true entries of attack_mask's row for `source`, visiting only the tiles within the unit's range.
 * Rust `emit(t)` -> appended to `out` (may be NULL to count only), in the Rust visiting order (y outer, x inner);
 * returns the count. */
PC_FN int pc_military_attack_targets(const PcState *s, int source, const PcRules *r, const PcGeom *g, uint16_t *out) {
    const PcUnits *u = &s->units;
    int32_t cp = s->current_player;
    int kind = u->kind[source];
    int count = 0;
    (void)g;
    uint8_t own = u->owner[source] == cp && !u->attacked[source] && u->health[source] > 0.0f;
    uint8_t can = own && (!u->moved[source] || r->dash[kind]) && r->unit_attack[kind] >= 0.0f;
    uint8_t infil = r->skills[kind][PC_SK_INFILTRATE];
    /* get attack options: none for a unit with a leader (a Segment; Cymanti) */
    can = can && u->leader[source] == 0;
    /* get attack options: 0 attack needs Convert (the Mind Bender, the Siren) or Infiltrate */
    can = can && (r->unit_attack[kind] > 0.0f || r->convert[kind] || infil);
    int32_t range = r->unit_range[kind];
    if (!can || range < 1) {
        return 0;
    }
    uint8_t drench = r->drench[kind];
    const uint8_t *explored = s->tiles.explored[pc_p(s)];
    int32_t size = s->size;
    int32_t sx = (int32_t)(source % s->size), sy = (int32_t)(source / s->size);
    for (int32_t y = pc_max(sy - range, 0); y <= pc_min(sy + range, size - 1); y++) {
        for (int32_t x = pc_max(sx - range, 0); x <= pc_min(sx + range, size - 1); x++) {
            int t = (int)(y * size + x);
            if (t == source || !explored[t]) {
                continue;
            }
            uint8_t targets;
            if (infil) {
                targets = pc_foreign(s->cities.owner[t], cp) && !pc_diplomacy_protected(s, cp, s->cities.owner[t])
                    && !(u->owner[t] != 0 && u->owner[t] != s->cities.owner[t]) && !s->cities.riot[t];
            } else if (drench && pc_aquarion_floodable(s, t)) {
                /* get attack options at position: every floodable tile is a Drench target, whatever stands there */
                targets = 1;
            } else {
                targets = pc_foreign(u->owner[t], cp) && !u->invisible[t] && !pc_diplomacy_protected(s, cp, u->owner[t]);
            }
            if (targets) {
                if (out) out[count] = (uint16_t)t;
                count++;
            }
        }
    }
    return count;
}

PC_FN void pc_military_attack_mask(const PcState *s, int source, const PcRules *r, const PcGeom *g, uint8_t out[PC_MAXN]) {
    uint16_t list[PC_MAXN];
    for (int t = 0; t < s->n; t++) out[t] = 0;
    int count = pc_military_attack_targets(s, source, r, g, list);
    for (int k = 0; k < count; k++) out[list[k]] = 1;
}

PC_FN float pc_military_terrain_bonus(const PcState *s, const PcRules *r, int i, const uint8_t *tech) {
    const PcTiles *t = &s->tiles;
    const PcUnits *u = &s->units;
    const PcCities *c = &s->cities;
    int32_t terrain = t->terrain[i];
    float city;
    if (c->owner[i] == u->owner[i] && r->fortify[u->kind[i]]) {
        city = pc_reward(s, i, 1) > 0 ? 4.0f : 1.5f;
    } else {
        city = 1.0f;
    }
    float bonus;
    if (terrain == 1 || terrain == 2) {
        bonus = tech[12] ? 1.5f : 1.0f;
    } else if (terrain == 5) {
        bonus = tech[18] ? 1.5f : 1.0f;
    } else if (terrain == 4) {
        bonus = tech[20] ? 1.5f : 1.0f;
    } else {
        bonus = city;
    }
    return bonus * (u->poison[i] ? 0.5f : 1.0f);
}

PC_FN float pc_military_defense_bonus(const PcState *s, const PcRules *r, int i) {
    const uint8_t *tech = s->players.tech[pc_pclip(s, s->units.owner[i])];
    return pc_military_terrain_bonus(s, r, i, tech);
}

/* `sc`: the routes scratch of the advance's AutoFreeze (relocate) */
/* attack command against a Drench target without an attackable defender (no unit, or one with an active peace treaty:
 * own units included; military.rs drench_null): no diplomacy change, zero battle results; the attacker is moved (no
 * Escape) and attacked; the 0-damage attack action turns both units, takes the target's Boosted and Bubble, floods */
PC_FN void pc_military_drench_null(PcState *s, int source, int target, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int size = s->size;
    int kind = s->units.kind[source];
    s->units.direction[source] = pc_military_direction(size, source, target);
    s->units.prev_x[source] = 2 * (int32_t)(source % size) - (int32_t)(target % size);
    s->units.prev_y[source] = 2 * (int32_t)(source / size) - (int32_t)(target / size);
    s->units.last_attack[source] = (int32_t)target + 1;
    uint8_t had_bubble = s->units.owner[target] != 0 && s->units.bubble[target];
    if (s->units.owner[target] != 0) {
        s->units.direction[target] = pc_military_direction(size, target, source);
        s->units.boosted[target] = 0;
    }
    if (r->drench[kind] && pc_aquarion_floodable(s, target)) pc_aquarion_flood_tile(s, target, s->units.owner[source], r, g, sc);
    s->units.swift[target] = 0;   /* perform attack: the defender loses Swift (Cymanti) */
    if (had_bubble) s->units.bubble[target] = 0;
    s->units.attacks_performed[source] += 1;
    s->units.attacked[source] = 1;
    s->units.move_lock[source] = 0;
    s->units.moved[source] = (uint8_t)!r->escape[kind];
}

/* move action's Tentacle trap (military.rs / aquarion:tentacle_trap): among the arrival tile's neighbours in
 * get tile neighbors order the healthiest (first on ties) unit of another player, no active peace treaty with the mover,
 * Tentacles, not Frozen; its tile (-1: none) and its full battle damage against the arrived unit */
PC_FN int pc_military_tentacle_trap(const PcState *s, int t, int32_t mover, const PcRules *r, float *damage) {
    const PcUnits *u = &s->units;
    int32_t size = s->size;
    int32_t x = (int32_t)(t % s->size), y = (int32_t)(t / s->size);
    int best = -1;
    for (int k = 0; k < 8; k++) {
        int32_t nx = x + pc_military_steps_dx(k), ny = y + pc_military_steps_dy(k);
        if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
        int j = (int)(ny * size + nx);
        int32_t o = u->owner[j];
        if (o == 0 || o == mover || pc_diplomacy_protected(s, mover, o)) continue;
        if (!r->tentacles[pc_clamp(u->kind[j], 0, 127)] || u->frozen[j]) continue;
        if (best < 0 || u->health[best] < u->health[j]) best = j;
    }
    if (best < 0) return -1;
    int kj = u->kind[best], kt = u->kind[t];
    float d1, d2;
    pc_kernels_combat(r->unit_attack[kj], u->health[best], pc_kernels_max_health(s, r, best), r->unit_defense[kt], u->health[t],
                      pc_kernels_max_health(s, r, t), pc_military_defense_bonus(s, r, t), damage, &d1, &d2);
    return best;
}

/* The trap's attack action (Passive, no retaliation) -> perform attack */
PC_FN void pc_military_trap_attack(PcState *s, int jelly, int t, float damage, const PcRules *r) {
    if (s->units.owner[t] == 0) return;
    int size = s->size;
    uint8_t present = s->units.owner[jelly] != 0;
    if (present) s->units.direction[jelly] = pc_military_direction(size, jelly, t);
    s->units.direction[t] = pc_military_direction(size, t, jelly);
    s->units.boosted[t] = 0;
    s->units.bubble[t] = 0;
    s->units.swift[t] = 0;
    s->units.health[t] -= damage;
    if (s->units.health[t] <= 0.0f) pc_military_remove(s, t, r, present ? jelly : -1, 1);
}

PC_FN void pc_military_attack(PcState *s, int source, int target, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int size = s->size;
    int32_t cp = s->current_player;
    if (s->units.owner[target] == 0 || pc_diplomacy_protected(s, cp, s->units.owner[target])) {
        pc_military_drench_null(s, source, target, r, g, sc);   /* only a Drench target */
        return;
    }
    uint8_t had_bubble = s->units.bubble[target];
    int kind = s->units.kind[source];
    int defender = s->units.kind[target];
    int32_t terrain = s->tiles.terrain[target];
    const uint8_t *tech = s->players.tech[pc_pwrap(s, s->units.owner[target])];
    float bonus = pc_military_terrain_bonus(s, r, target, tech);
    float hp_s = pc_kernels_max_health(s, r, source);
    float hp_t = pc_kernels_max_health(s, r, target);
    float dd = r->unit_defense[defender];
    float damage, retaliation, half;
    pc_kernels_combat(r->unit_attack[kind], s->units.health[source], hp_s, dd, s->units.health[target], hp_t, bonus,
                      &damage, &retaliation, &half);
    uint8_t dead = s->units.health[target] - damage <= 0.0f;
    /* can defender retaliate: the attacker must also be one of the defender's attack
     * options, i.e. stand on a tile the defender's owner has explored */
    uint8_t seen = s->tiles.explored[pc_pclip(s, s->units.owner[target])][source];
    /* can defender retaliate (v >= 120): never against a Freeze attacker, never by a Frozen defender */
    uint8_t freeze = r->freeze[kind];
    uint8_t retaliate = !r->skills[kind][PC_SK_SURPRISE] && !r->stiff[defender] && pc_d(g, source, target) <= r->unit_range[defender] && seen
        && !freeze && !s->units.frozen[target] && !r->convert[kind] && s->units.leader[target] == 0;
    /* should reset pacifist task: damage dealt to a visible unit; a Freeze attack resets the Pacifist task
     * in freeze unit action, visible or not */
    if ((damage > 0.0f && !s->units.invisible[target]) || freeze) {
        s->players.attacked[pc_p(s)] = 1;
    }
    /* attack action: both units face each other */
    s->units.direction[source] = pc_military_direction(size, source, target);
    s->units.direction[target] = pc_military_direction(size, target, source);
    s->units.health[target] += -damage;
    s->units.boosted[target] = 0;
    s->units.prev_x[source] = 2 * (int32_t)(source % size) - (int32_t)(target % size);
    s->units.prev_y[source] = 2 * (int32_t)(source / size) - (int32_t)(target / size);
    s->units.last_attack[source] = (int32_t)target + 1;
    if (freeze) {
        /* attack command (v > 120) with Freeze: freeze unit action deals the damage and freezes the target (moved,
         * attacked, Frozen) before a kill removes it; no splash, no retaliation, no advance */
        s->units.moved[target] = 1;
        s->units.attacked[target] = 1;
        s->units.move_lock[target] = 0;
        s->units.frozen[target] = 1;
    }
    /* perform attack: a Poison attacker poisons a defender that survives (Cymanti) */
    if (!dead && r->poisons[kind]) s->units.poison[target] = 1;
    if (r->skills[kind][PC_SK_SPLASH] && !freeze && r->poisons[kind]) {
        /* a Poison splasher (Exida): poison unit action only on each non-peace neighbour of the target */
        pc_cymanti_poison_splash(s, source, target, g);
    } else if (r->skills[kind][PC_SK_SPLASH] && !freeze) {
        pc_military_area_damage(s, source, target, r, g, 0);
        /* attack command pushes the retaliation attack action (Target -> Origin, when its damage is >= 1)
         * before the splash ones and the main attack; the action stack is LIFO,
         * so it runs last and the attacker ends facing the defender again */
        if (!dead && retaliate && retaliation > 0.0f) {
            s->units.direction[source] = pc_military_direction(size, source, target);
        }
    }
    /* perform attack: the target loses its Bubble (after a Drench attacker's flood tile, whose bubble check still sees
     * it); a Drench attacker floods a floodable target tile before the kill unit and the retaliation */
    if (r->drench[kind] && pc_aquarion_floodable(s, target)) pc_aquarion_flood_tile(s, target, s->units.owner[source], r, g, sc);
    s->units.swift[target] = 0;   /* perform attack: the defender loses Swift (Cymanti) */
    if (had_bubble) s->units.bubble[target] = 0;
    if (dead) {
        pc_military_remove(s, target, r, source, 1);
    } else {
        s->units.health[source] += -(retaliate ? retaliation : 0.0f);
        /* the retaliation attack action (damage >= 1) is a perform attack on the attacker: it loses its Bubble */
        if (retaliate && retaliation > 0.0f) { s->units.bubble[source] = 0; s->units.swift[source] = 0; }
    }
    uint8_t attacker_dead = s->units.health[source] <= 0.0f;
    /* the retaliation's perform attack: a Poison defender poisons the attacker it hits */
    if (!dead && !attacker_dead && retaliate && retaliation > 0.0f && r->poisons[defender]) s->units.poison[source] = 1;
    if (attacker_dead) {
        pc_military_remove(s, source, r, target, 1);
    }
    int32_t st = s->tiles.structure[target];
    /* get battle results: the advance follows get path with the unit's allowed terrain (water: Fishing,
     * ocean: Sailing or Free Diving, mountain: Climbing; a Bridge always), a land unit onto water only by an own Port */
    const uint8_t *mine = s->players.tech[pc_p(s)];
    uint8_t can_enter;
    if (r->fly[kind]) {
        can_enter = 1;   /* a flyer enters any tile (is tile accessible: Fly before the terrain tests) */
    } else if (r->skills[kind][PC_SK_FLOAT]) {
        /* a Swim unit: Water always, ocean when unlocked, land by its class */
        PcSwim sw = pc_swim_of(s, r, kind, cp);
        can_enter = st == 48 || pc_swim_terrain_ok(&sw, s, mine, target);
    } else {
        uint8_t water_ok = (uint8_t)pc_kernels_terrain_ok(mine, terrain);
        can_enter = terrain == 3 || terrain == 5 || terrain == PC_ICE || (terrain == 4 && mine[20])
            || (terrain <= 2 && (st == 48 || s->tiles.algae[target] || (water_ok && st == 8 && s->tiles.owner[target] == cp
                && pc_tribes_can_embark(s->players.tribe[pc_p(s)], mine))));
    }
    uint8_t advance = dead && !attacker_dead && r->unit_range[kind] < 2 && can_enter && !freeze;
    if (advance) {
        pc_military_relocate(s, source, target, r, g, 1, NULL, sc);
    }
    int at = advance ? target : source;
    /* the advance embarked or disembarked (the unit changed type): the embark action / disembark action the advance's
     * move action pushes runs after attack command's own moved/attacked marks and sets both on the new unit;
     * the advance's move action takes a Skate unit off the ice: done for the turn (move action v > 101, after the marks) */
    uint8_t boarded = advance && (s->units.kind[at] != kind || (r->skate[kind] && s->tiles.terrain[at] != PC_ICE));
    if (!attacker_dead) {
        uint8_t persist = r->persist[kind];
        uint8_t escape = r->escape[kind];
        s->units.attacks_performed[at] += 1;
        /* attack command: a double ready attacker loses it and stays able to attack (Cymanti, the Phychi) */
        uint8_t dbl = s->units.double_ready[at];
        s->units.double_ready[at] = 0;
        s->units.attacked[at] = !dbl && !(dead && persist);
        s->units.move_lock[at] = 0;
        /* attack command: moved = !Escape; a Skate + Escape unit (BattleSled, IceFortress) is moved only when it ends
         * off the ice */
        s->units.moved[at] = (escape && r->skate[kind]) ? (uint8_t)(s->tiles.terrain[at] != PC_ICE) : (uint8_t)!escape;
        if (dead && persist) {
            /* attack command: Persist leaves the unit able to attack only when it has
             * another target from where it now stands
             * (get attack options at position with include hidden when range <= sight: hidden and unexplored enemies count) */
            uint8_t more = pc_military_attack_targets(s, at, r, g, NULL) > 0;
            int32_t sight = 1 + (r->skills[kind][PC_SK_SCOUT] ? 1 : 0);
            if (!more && r->unit_range[kind] <= sight) {
                int32_t owner = s->units.owner[at];
                for (int t = 0; t < s->n; t++) {
                    if (t != at && pc_d(g, at, t) <= r->unit_range[kind] && pc_foreign(s->units.owner[t], owner)
                        && !pc_diplomacy_protected(s, owner, s->units.owner[t])) {
                        more = 1;
                        break;
                    }
                }
            }
            s->units.attacked[at] = !more && !dbl;
        }
        if (boarded) {
            s->units.moved[at] = 1;
            s->units.attacked[at] = 1;
        }
    }
    /* attack command: a Convert attacker with an attack (the Shaman) converts the defender its damage leaves alive */
    if (r->convert[kind] && !dead && !attacker_dead && s->units.owner[target] != 0 && s->units.owner[target] != cp) {
        pc_military_convert(s, source, target, r, g);
    }
    /* attack command: an Eat attacker that kills a defender pushes eat action for the last unit of its follower chain;
     * it runs after the kill and the advance (Cymanti) */
    if (r->eat[kind] && dead && !attacker_dead) {
        int at2 = advance ? target : source;
        if (s->units.owner[at2] == cp) pc_cymanti_eat(s, pc_cymanti_tail(s, at2), r, g);
    }
}

/* convert action; `attack`: the attack command's main attack action ran first (it turns both units and takes the
 * target's Bubble and Swift); otherwise the converter (the chain unit converted before) is marked moved (no Escape)
 * and attacked */
PC_FN void pc_military_convert_one(PcState *s, int source, int target, const PcRules *r, const PcGeom *g, uint8_t attack) {
    int n = s->n;
    int32_t cp = s->current_player;
    int old = pc_pwrap(s, s->units.owner[target]);
    int tk = s->units.kind[target];
    /* convert action: rehomed to the converter's current capital (none: no home) */
    int32_t home;
    if (r->skills[tk][PC_SK_INDEPENDENT]) {
        home = 0;
    } else {
        int cap = pc_structures_current_capital(s, cp);
        home = cap >= 0 ? (int32_t)cap + 1 : 0;
    }
    int32_t passenger = s->units.passenger[target];
    int32_t cost = r->unit_cost[tk] + (passenger > 0 ? r->unit_cost[passenger] : 0);
    /* the full unit score moves from the old owner to the converter, every time */
    s->players.score[old] = pc_wsub(s->players.score[old], 5 * cost);
    int cpi = pc_p(s);
    s->players.score[cpi] = pc_wadd(s->players.score[cpi], 5 * cost);
    /* convert action bumps the Convert task while it is started (Rituals; Cymanti) */
    if (s->players.tech[cpi][49]) s->players.converts[cpi] += 1;
    s->units.owner[target] = cp;
    s->units.converted[target] = 1;
    s->units.home[target] = home;
    s->units.moved[target] = 1;
    s->units.attacked[target] = 1;
    s->units.move_lock[target] = 0;
    /* attack command also pushes the main attack action (Origin -> Target, damage = the converter's
     * attack damage, 0) after the convert action; it runs first and turns the two units
     * toward each other. No retaliation: can defender retaliate is false
     * against Convert. */
    if (attack) {
        s->units.direction[source] = pc_military_direction(s->size, source, target);
        s->units.direction[target] = pc_military_direction(s->size, target, source);
        /* that attack action's perform attack takes the target's Bubble and Swift */
        s->units.bubble[target] = 0;
        s->units.swift[target] = 0;
        pc_military_exhaust(s, source);
    } else {
        s->units.moved[source] = (uint8_t)!r->escape[pc_clamp(s->units.kind[source], 0, 127)];
        s->units.attacked[source] = 1;
        s->units.move_lock[source] = 0;
    }
    int radius = (s->tiles.terrain[target] == 4 || r->skills[tk][PC_SK_SCOUT]) ? 2 : 1;
    uint8_t reveal[PC_MAXN];
    for (int t = 0; t < n; t++) reveal[t] = (uint8_t)(pc_d(g, target, t) <= radius);
    pc_economy_discover(s, reveal, cp);
}

/* military:convert: convert action, then (Cymanti) the target's follower and leader each get their own
 * convert action from the target's tile when another player's: the whole chain changes sides (the Rust recursion as
 * two walks: down the followers, then up the leaders) */
PC_FN void pc_military_convert(PcState *s, int source, int target, const PcRules *r, const PcGeom *g) {
    int32_t cp = s->current_player;
    int32_t follower = s->units.follower[target], leader = s->units.leader[target];
    pc_military_convert_one(s, source, target, r, g, 1);
    for (int pass = 0; pass < 2; pass++) {
        int prev = target, guard = 0;
        int32_t link = pass == 0 ? follower : leader;
        while (link > 0 && guard++ < s->n) {
            int o = (int)link - 1;
            if (s->units.owner[o] == 0 || s->units.owner[o] == cp) break;
            pc_military_convert_one(s, prev, o, r, g, 0);
            link = pass == 0 ? s->units.follower[o] : s->units.leader[o];
            prev = o;
        }
    }
}

/* out[t*128 + kind]
 * The Raft upgrades (Rammership, Scoutship, Bomber): only summonable onto an own Raft.
 * Rust `pub const UPGRADES: u128 = 1 << 41 | 1 << 42 | 1 << 44` (a function: pc_u128 is a struct). */
PC_FN pc_u128 pc_military_upgrades(void) {
    pc_u128 b = pc_u128_zero();
    b.lo = (1ull << 41) | (1ull << 42) | (1ull << 44);
    return b;
}

/* Per-call inputs of summon_mask: units per home city and the unlocked, affordable kinds. */
typedef struct {
    int32_t counts[PC_MAXN];
    pc_u128 ok;
    pc_u128 water;   /* the kinds with movement terrain (Jelly, Shark, Aquapult; PcRules.water) */
} PcSummonCtx;

/* summon ctx:new(n) */
PC_FN void pc_military_summon_ctx_new(PcSummonCtx *ctx) {
    for (int i = 0; i < PC_MAXN; i++) ctx->counts[i] = 0;
    ctx->ok = pc_u128_zero();
    ctx->water = pc_u128_zero();
}

/* summon ctx:prepare over cooperating lanes (pc.h PcLanes): the counts are a scatter-add (atomic with lanes;
 * integer sums do not depend on the order) */
PC_FN void pc_military_summon_prepare_lanes(PcSummonCtx *ctx, const PcState *s, const PcRules *r, PcLanes L) {
    int n = s->n;
    const PcUnits *u = &s->units;
    int p = pc_p(s);
    for (int i = L.lane; i < n; i += L.count) ctx->counts[i] = 0;
    pc_lanes_sync(L);
    for (int i = L.lane; i < n; i += L.count) {
        /* get city unit count: every unit homed at the city counts, whoever owns it now */
        if (u->home[i] >= 1 && u->home[i] <= n) {
            pc_lanes_add(L, &ctx->counts[u->home[i] - 1], 1);
        }
    }
    pc_u128 unlocked = pc_rules_unit_static_of(r, s->players.tribe[p]);
    pc_u128 known = pc_rules_bits128(s->players.tech[p]);
    const pc_u128 *by_tech = pc_rules_units_by_tech_of(r, s->players.tribe[p]);
    while (pc_u128_any(known)) {
        int k = pc_u128_lowest(known);
        unlocked = pc_u128_or(unlocked, by_tech[k]);
        pc_u128_clear(&known, k);
    }
    if (L.lane == 0) {
        ctx->ok = pc_u128_and(unlocked, pc_cost_le(&r->unit_cost_le, s->players.stars[p]));
        ctx->water = pc_rules_bits128(r->water);
    }
    pc_lanes_sync(L);
}

/* has unit terrain for a movement terrain unit (water, ocean, wetland): a neighbour of the city is
 * water or Flooded (military.rs wet_neighbour) */
PC_FN uint8_t pc_military_wet_neighbour(const PcState *s, int t) {
    int32_t size = s->size;
    int32_t x = (int32_t)(t % s->size), y = (int32_t)(t / s->size);
    for (int32_t dy = -1; dy <= 1; dy++) {
        for (int32_t dx = -1; dx <= 1; dx++) {
            int32_t nx = x + dx, ny = y + dy;
            if ((dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
            int k = (int)(ny * size + nx);
            if (s->tiles.terrain[k] == 1 || s->tiles.terrain[k] == 2 || s->tiles.flooded[k]) return 1;
        }
    }
    return 0;
}

/* summon ctx:prepare */
PC_FN void pc_military_summon_prepare(PcSummonCtx *ctx, const PcState *s, const PcRules *r) {
    pc_military_summon_prepare_lanes(ctx, s, r, pc_lanes_one());
}

/* One row of military.summon_mask as a bitset of unit kinds. */
PC_FN pc_u128 pc_military_summon_row(const PcState *s, const PcSummonCtx *ctx, int t) {
    const PcUnits *u = &s->units;
    const PcCities *c = &s->cities;
    int32_t cp = s->current_player;
    uint8_t available = c->owner[t] == cp && ctx->counts[t] <= c->level[t] && u->owner[t] == 0;
    uint8_t friendly = s->tiles.owner[t] == cp || (s->tiles.owner[t] > 0 && s->players.peace[pc_p(s)][pc_pclip(s, s->tiles.owner[t])]);
    uint8_t upgrade_allowed = u->owner[t] == cp && u->kind[t] == 43 && friendly;
    pc_u128 up = pc_military_upgrades();
    /* get trainable units: a unit with movement terrain only where a neighbour fits it (Aquarion) */
    pc_u128 ok = ctx->ok;
    if (available && pc_u128_any(pc_u128_and(ctx->ok, ctx->water)) && !pc_military_wet_neighbour(s, t)) {
        ok = pc_u128_andnot(ctx->ok, ctx->water);
    }
    pc_u128 a = available ? pc_u128_andnot(ok, up) : pc_u128_zero();
    pc_u128 b = upgrade_allowed ? pc_u128_and(ctx->ok, up) : pc_u128_zero();
    return pc_u128_or(a, b);
}

/* out[t * 128 + k] for t < n (the caller provides n * 128 bytes) */
PC_FN void pc_military_summon_mask(const PcState *s, const PcRules *r, uint8_t *out) {
    PcSummonCtx ctx;
    pc_military_summon_ctx_new(&ctx);
    pc_military_summon_prepare(&ctx, s, r);
    for (int t = 0; t < s->n; t++) {
        pc_u128 bits = pc_military_summon_row(s, &ctx, t);
        for (int k = 0; k < 128; k++) {
            out[t * 128 + k] = (uint8_t)pc_u128_test(bits, k);
        }
    }
}

PC_FN void pc_military_summon(PcState *s, int source, int kind, const PcRules *r, const PcGeom *g, uint8_t costs) {
    int n = s->n;
    int32_t cp = s->current_player;
    pc_units_clear(&s->units, source);
    s->units.owner[source] = cp;
    s->units.kind[source] = (int32_t)kind;
    s->units.health[source] = r->unit_health[kind];
    s->units.home[source] = (int32_t)source + 1;
    s->units.moved[source] = 1;
    s->units.attacked[source] = 1;
    s->units.prev_x[source] = -1;
    s->units.prev_y[source] = -1;
    s->units.created[source] = s->turn;
    s->units.direction[source] = pc_military_spawn_direction(s->size, source);
    int p = pc_p(s);
    s->players.stars[p] -= r->unit_cost[kind] * (costs ? 1 : 0);
    s->players.score[p] = pc_wadd(s->players.score[p], 5 * r->unit_cost[kind]);
    /* train action: explore from tile with get sight range (Scout units see 2 tiles) */
    int radius = (s->tiles.terrain[source] == 4 || r->skills[kind][PC_SK_SCOUT]) ? 2 : 1;
    uint8_t reveal[PC_MAXN];
    for (int t = 0; t < n; t++) reveal[t] = (uint8_t)(pc_d(g, source, t) <= radius);
    pc_economy_discover(s, reveal, cp);
    /* Cymanti: train unit's check step on poison */
    pc_cymanti_step_on_poison(s, source, r);
    /* train action: a Tentacles unit's area damage (after the exploration) */
    if (r->tentacles[kind]) pc_military_area_damage(s, source, source, r, g, 1);
}

/* The unit on `i` may be promoted: 3 kills, not yet a veteran, not Static
 * (can be promoted): a boat is judged by its passenger, so a non-Static passenger is promoted at sea. */
PC_FN uint8_t pc_military_promotable(const PcState *s, const PcRules *r, int i) {
    const PcUnits *u = &s->units;
    int judged = u->passenger[i] > 0 ? u->passenger[i] : u->kind[i];
    return !u->veteran[i] && u->kills[i] >= 3 && !r->stat[judged];
}

PC_FN void pc_military_promote(PcState *s, int source, const PcRules *r) {
    float hp = pc_kernels_max_health(s, r, source) + 5.0f;
    s->units.veteran[source] = 1;
    s->units.health[source] = hp;
}

/* Disband the unit on `source`, refunding half its full cost (get full cost: boat plus passenger) to seat
 * index `p` (its owner). */
PC_FN void pc_military_disband_one(PcState *s, int source, const PcRules *r, int p, uint8_t paid) {
    int kind = s->units.kind[source];
    int32_t passenger = s->units.passenger[source];
    int32_t full = r->unit_cost[kind] + (passenger > 0 ? r->unit_cost[passenger] : 0);
    s->units.poison[source] = 0;   /* disband action never spawns spores */
    pc_military_remove(s, source, r, -1, 0);
    if (paid) s->players.stars[p] += pc_div_euclid(full, 2);
}

/* disband unit action: the follower gets its own (paid) disband unit action, the leader loses the unit (Cymanti; the
 * Rust recursion as a loop); `paid`: half the full cost to seat index `p` */
PC_FN void pc_military_disband_chain(PcState *s, int source, const PcRules *r, int p, uint8_t paid) {
    int f = pc_cymanti_unlink(s, source);
    pc_military_disband_one(s, source, r, p, paid);
    int guard = 0;
    while (f >= 0 && s->units.owner[f] != 0 && guard++ < s->n) {
        int next = pc_cymanti_unlink(s, f);
        pc_military_disband_one(s, f, r, p, 1);
        f = next;
    }
}

PC_FN void pc_military_disband_of(PcState *s, int source, const PcRules *r, int p) {
    pc_military_disband_chain(s, source, r, p, 1);
}

PC_FN void pc_military_disband(PcState *s, int source, const PcRules *r) {
    int p = pc_p(s);
    pc_military_disband_of(s, source, r, p);
}

PC_FN void pc_military_heal_others(PcState *s, int source, const PcRules *r, const PcGeom *g) {
    int32_t cp = s->current_player;
    for (int i = 0; i < s->n; i++) {
        if (pc_d(g, source, i) == 1 && s->units.owner[i] == cp) {
            if (!s->units.poison[i]) {
                float mh = pc_kernels_max_health(s, r, i);
                s->units.health[i] = pc_fminf(s->units.health[i] + 4.0f, mh);
            }
            s->units.poison[i] = 0;
        }
    }
    pc_military_exhaust(s, source);
}

/* to coordinates: SW, W, NW, N, NE, E, SE, S as (dx, dy) in game coordinates (y is the row).
 * Rust `pub const GRID: [(i32, i32); 8]`; here two accessors over function-local tables (device-safe). */
PC_FN int32_t pc_military_grid_dx(int d) {
    static const int32_t GX[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
    return GX[d];
}
PC_FN int32_t pc_military_grid_dy(int d) {
    static const int32_t GY[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
    return GY[d];
}
/* get tile neighbors order (x outer). */
PC_FN int32_t pc_military_steps_dx(int k) {
    static const int32_t SX[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    return SX[k];
}
PC_FN int32_t pc_military_steps_dy(int k) {
    static const int32_t SY[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    return SY[k];
}

/* get direction from distance vector: the direction with the largest weighted dot product with the
 * vector (axial 1393, diagonal 985); SW when none is positive. */
PC_FN int32_t pc_military_vector_direction(int32_t vx, int32_t vy) {
    int32_t best = 0, score = 0;
    for (int d = 0; d < 8; d++) {
        int32_t x = pc_military_grid_dx(d), y = pc_military_grid_dy(d);
        int32_t v = (y * vy + x * vx) * ((x * x + y * y == 1) ? 1393 : 985);
        if (score < v) {
            best = (int32_t)d;
            score = v;
        }
    }
    return best;
}

/* get direction on coordinates that may be off the board: the game's
 * null coordinates (-1, -1) (world coordinates..cctor) are the origin of the attack in
 * infiltration reward action. Rust tuples (from, to) -> (fx, fy, tx, ty). */
PC_FN int32_t pc_military_direction_xy(int32_t fx, int32_t fy, int32_t tx, int32_t ty) {
    int32_t dx = tx - fx, dy = ty - fy;
    if (pc_max(pc_abs(dx), pc_abs(dy)) > 1) {
        return pc_military_vector_direction(dx, dy);
    }
    for (int d = 0; d < 8; d++) {
        if (pc_military_grid_dx(d) == dx && pc_military_grid_dy(d) == dy) return (int32_t)d;
    }
    return 8;
}

/* get direction(from, to): a neighbour's direction (8, none, for the same tile), farther tiles by
 * vector_direction. */
PC_FN int32_t pc_military_direction(int size, int from, int to) {
    return pc_military_direction_xy((int32_t)(from % size), (int32_t)(from / size), (int32_t)(to % size), (int32_t)(to / size));
}

/* unit state create: a new unit faces the map centre (width / 2, height / 2). */
PC_FN int32_t pc_military_spawn_direction(int size, int tile) {
    int32_t c = (int32_t)(size / 2);
    return pc_military_vector_direction(c - (int32_t)(tile % size), c - (int32_t)(tile / size));
}

/* get path(source, dest, 1, unit) for a neighbour `dest` of the unit on `source`: is allowed to finish on tile
 * and is tile accessible under create for unit (enemy territory allowed, occupied tiles not). */
/* military:push_step: is tile accessible of `dest` coming from `from` for the unit on `source` */
PC_FN uint8_t pc_military_push_step(const PcState *s, int source, int from, int dest, const PcRules *r) {
    const PcTiles *t = &s->tiles;
    const PcUnits *u = &s->units;
    int32_t owner = u->owner[source];
    int p = pc_pclip(s, owner);
    int kind = u->kind[source];
    if (u->owner[dest] != 0 || !t->explored[p][dest]) {
        return 0;
    }
    int32_t terrain = t->terrain[dest], structure = t->structure[dest];
    if (structure == 1 && pc_foreign(t->owner[dest], owner)) {
        /* not a peace (or cool-down) partner's city, and no Infiltrate unit into an opponent's city */
        int q = pc_pclip(s, t->owner[dest]);
        if (s->players.peace[p][q] || s->players.truce_breaker[p][q] != 0 || r->skills[kind][PC_SK_INFILTRATE]) {
            return 0;
        }
    }
    if (structure == 48) {
        return r->skills[kind][PC_SK_FLOAT] || r->fly[kind] || !pc_kernels_bridge_blocks(s, owner, from, dest);
    }
    if (structure == 1) {
        return 1;
    }
    /* Cymanti: an Algae water tile is wetland, in every unit's allowed terrain */
    if (t->algae[dest] && (terrain == 1 || terrain == 2)) {
        return 1;
    }
    const uint8_t *tech = s->players.tech[p];
    PcSwim sw = pc_swim_of(s, r, kind, owner);
    uint8_t terrain_ok = (uint8_t)pc_swim_terrain_ok(&sw, s, tech, dest);
    uint8_t land_access = (terrain != 1 && terrain != 2)
        || (structure == 8 && t->owner[dest] == owner && pc_tribes_can_embark(s->players.tribe[p], tech));
    return terrain_ok && (land_access || r->skills[kind][PC_SK_FLOAT] || r->fly[kind]);
}

/* military::push_reachable road tile / zone-of-control closures */
PC_FN uint8_t pc_military_push_road(const PcState *s, int32_t owner, int p, int i) {
    const PcTiles *t = &s->tiles;
    uint8_t friendly = t->owner[i] == 0 || t->owner[i] == owner || s->players.peace[p][pc_pclip(s, t->owner[i])];
    return (t->road[i] || t->structure[i] == 1 || s->cities.owner[i] != 0 || t->structure[i] == 48) && friendly;
}
PC_FN uint8_t pc_military_push_zoc(const PcState *s, const PcGeom *g, int32_t owner, int p, int i) {
    const PcUnits *u = &s->units;
    for (int k = 0; k < g->adj_n[i]; k++) {
        int j = g->adj[i * 8 + k];
        /* get movement cost: no zone of control from a unit with a leader (Cymanti) */
        if (u->owner[j] != 0 && u->owner[j] != owner && !u->invisible[j] && !s->players.peace[p][pc_pclip(s, u->owner[j])]
            && u->leader[j] == 0) return 1;
    }
    return 0;
}

/* military:push_reachable: try push unit default's get path(max cost 1): the direct step, or onto a Bridge that refuses
 * it, two road steps through a free road tile next to both (no enemy zone of control) */
PC_FN uint8_t pc_military_push_reachable(const PcState *s, int source, int dest, const PcRules *r, const PcGeom *g) {
    if (pc_military_push_step(s, source, source, dest, r)) return 1;
    const PcTiles *t = &s->tiles;
    const PcUnits *u = &s->units;
    int32_t owner = u->owner[source];
    if (t->structure[dest] != 48 || u->owner[dest] != 0 || !pc_kernels_bridge_blocks(s, owner, source, dest)) return 0;
    int p = pc_pclip(s, owner);
    if (!pc_military_push_road(s, owner, p, source)) return 0;
    for (int k = 0; k < g->adj_n[source]; k++) {
        int m = g->adj[source * 8 + k];
        if (m != dest && pc_d(g, m, dest) == 1 && u->owner[m] == 0 && pc_military_push_road(s, owner, p, m)
            && !pc_military_push_zoc(s, g, owner, p, m) && pc_military_push_step(s, source, source, m, r)
            && pc_military_push_step(s, source, m, dest, r)) return 1;
    }
    return 0;
}

/* water(t) closure of push */
PC_FN uint8_t pc_military_push_water(const PcState *s, int t) {
    return s->tiles.terrain[t] == 1 || s->tiles.terrain[t] == 2;
}

/* try push unit default, with train unit on occupied space's kill unit action when it fails: moves the unit on
 * `source` off it for a unit the current player trains there. The occupant's facing (reversed for another player's
 * unit) is tried first, then the directions -1, +1, -2, +2, -3, +3, -4 from it; a boat on water keeps the first land
 * tile as its last resort. The push is a move action with move reason 2: no stomp, and moved/attacked stay as they were
 * unless the unit embarks or disembarks (those actions exhaust it). Deterministic, no hash draw. */
PC_FN void pc_military_push(PcState *s, int source, const PcRules *r, const PcGeom *g) {
    int32_t size = s->size;
    int32_t owner = s->units.owner[source];
    if (owner <= 0) {
        return;
    }
    /* train unit on occupied space (game version > 105): an occupant hidden from the
     * trainer (is invisible: another player's, no peace with them) gets a reveal action without an origin
     * (no turn, no meeting). The action stack runs it before the push's move action, so the unit stays revealed. */
    int32_t cp = s->current_player;
    if (s->units.invisible[source] && owner != cp && !s->players.peace[pc_p(s)][pc_pclip(s, owner)]) {
        s->units.invisible[source] = 0;
    }
    int32_t facing = s->units.direction[source];
    if (owner != s->current_player) {
        facing = facing < 4 ? facing + 4 : facing - 4;
    }
    /* try push unit default: a chain unit starts from its leader / follower (their average with both; Cymanti) */
    int32_t link_dir = pc_cymanti_link_direction(s, source);
    if (link_dir >= 0) facing = link_dir;
    int32_t kind = s->units.kind[source];
    uint8_t vehicle = r->skills[kind][PC_SK_CARRY];
    int32_t x = (int32_t)source % size, y = (int32_t)source / size;
    int target = -1, fallback = -1;
    for (int32_t k = 1; k <= 8; k++) {
        int32_t offset = ((k - 1) % 2 == 0) ? k / 2 : -(k >> 1);
        int d = (int)pc_rem_euclid(offset + facing + 8, 8);
        int32_t dx = pc_military_grid_dx(d), dy = pc_military_grid_dy(d);
        int32_t tx = x + dx, ty = y + dy;
        if (tx < 0 || ty < 0 || tx >= size || ty >= size) {
            continue;
        }
        int t = (int)(ty * size + tx);
        if (s->units.owner[t] != 0) {
            continue;
        }
        if (pc_military_push_water(s, source) && !pc_military_push_water(s, t) && vehicle && fallback < 0) {
            fallback = t;
            continue;
        }
        if (pc_military_push_reachable(s, source, t, r, g)) {
            target = t;
            break;
        }
    }
    /* target.or(fallback.filter(push_reachable)) */
    int chosen = target;
    if (chosen < 0 && fallback >= 0 && pc_military_push_reachable(s, source, fallback, r, g)) {
        chosen = fallback;
    }
    if (chosen >= 0) {
        int t = chosen;
        uint8_t moved = s->units.moved[source], attacked = s->units.attacked[source], lock = s->units.move_lock[source];
        pc_military_relocate_as(s, source, t, r, g, 1, NULL, 0, NULL);
        if (s->units.kind[t] == kind) {
            s->units.moved[t] = moved;
            s->units.attacked[t] = attacked;
            s->units.move_lock[t] = lock;
        }
    } else {
        pc_military_remove(s, source, r, -1, 1);
    }
}

PC_FN void pc_military_superunit(PcState *s, int source, const PcRules *r, const PcGeom *g) {
    pc_military_push(s, source, r, g);
    /* train unit on occupied space(GIANT) trains the tribe's override (Polaris: Gaami) */
    int kind = (int)pc_tribes_unit_override(s->players.tribe[pc_p(s)], 12);
    pc_military_summon(s, source, kind, r, g, 0);
}

PC_FN void pc_military_area_damage(PcState *s, int source, int center, const PcRules *r, const PcGeom *g, uint8_t stomp) {
    int n = s->n;
    int32_t src_owner = s->units.owner[source];
    int src_kind = s->units.kind[source];
    float hp_s = pc_kernels_max_health(s, r, source);
    float ah = s->units.health[source];
    float aa = r->unit_attack[src_kind];
    float damage[PC_MAXN];
    uint8_t dead[PC_MAXN];
    /* the seats hit (modify diplomacy relations from attack per damaged unit's owner) */
    uint8_t hit_seats[PC_MAXP + 1];
    uint8_t pacifist_reset = 0;
    for (int i = 0; i < n; i++) { damage[i] = 0.0f; dead[i] = 0; }
    for (int k = 0; k <= PC_MAXP; k++) hit_seats[k] = 0;
    for (int i = 0; i < n; i++) {
        const PcUnits *u = &s->units;
        uint8_t targets = pc_d(g, center, i) == 1 && u->owner[i] != 0 && u->owner[i] != src_owner && !pc_diplomacy_protected(s, src_owner, u->owner[i]);
        if (targets) {
            float dd = r->unit_defense[u->kind[i]];
            float dmg, d0, d1;
            pc_kernels_combat(aa, ah, hp_s, dd, u->health[i], pc_kernels_max_health(s, r, i), pc_military_defense_bonus(s, r, i), &d0, &d1, &dmg);
            (void)stomp;
            /* area damage: the full battle damage for a Tentacles unit (Passive), else the floored half (Splash) */
            dmg = r->tentacles[src_kind] ? d0 : floorf(dmg);
            damage[i] = dmg;
            dead[i] = u->health[i] - dmg <= 0.0f;
            if (dmg > 0.0f) {
                hit_seats[pc_clamp(u->owner[i], 0, PC_MAXP)] = 1;
            }
            pacifist_reset = (uint8_t)(pacifist_reset || (dmg > 0.0f && !u->invisible[i]));
        }
    }
    /* One attack action per target, pushed in get tile neighbors (x-outer) order, from the stomper's tile for a stomp
     * (area damage default, origin = the centre) and from the attacker's tile for
     * a splash (attack command, attack action(Origin, neighbour)). Each
     * attack action turns its target toward the origin and the origin's unit toward the target (
     * 753); the stack is LIFO, so the origin ends facing the first target in neighbour order. */
    int size = s->size;
    int origin = stomp ? center : source;
    int first = -1;
    for (int k = 0; k < 8; k++) {
        int32_t dx = pc_military_steps_dx(k), dy = pc_military_steps_dy(k);
        int32_t x = (int32_t)(center % size) + dx, y = (int32_t)(center / size) + dy;
        if (x >= 0 && y >= 0 && x < size && y < size) {
            int i = (int)y * size + (int)x;
            const PcUnits *u = &s->units;
            if (u->owner[i] != 0 && u->owner[i] != src_owner && !pc_diplomacy_protected(s, src_owner, u->owner[i])) {
                if (first < 0) first = i;
                s->units.direction[i] = pc_military_direction(size, i, origin);
            }
        }
    }
    if (first >= 0) {
        s->units.direction[origin] = pc_military_direction(size, origin, first);
    }
    for (int i = 0; i < n; i++) {
        s->units.health[i] -= damage[i];
    }
    /* every attack action's perform attack takes its target's Bubble, whatever the damage */
    for (int i = 0; i < n; i++) {
        const PcUnits *u = &s->units;
        if (pc_d(g, center, i) == 1 && u->owner[i] != 0 && u->owner[i] != src_owner && !pc_diplomacy_protected(s, src_owner, u->owner[i])) {
            s->units.bubble[i] = 0;
            s->units.swift[i] = 0;
        }
    }
    for (int i = 0; i < n; i++) {
        if (dead[i]) {
            pc_military_remove(s, i, r, source, 1);
        } else if (r->poisons[src_kind] && pc_d(g, center, i) == 1 && s->units.owner[i] != 0 && s->units.owner[i] != src_owner
                   && !pc_diplomacy_protected(s, src_owner, s->units.owner[i])) {
            /* Cymanti: every attack action's perform attack from a Poison origin (the Island's Stomp) poisons its
             * survivor, whatever the damage */
            s->units.poison[i] = 1;
        }
    }
    for (int victim = 0; victim <= PC_MAXP; victim++) {
        if (hit_seats[victim]) {
            pc_diplomacy_hostile(s, (int32_t)victim);
        }
    }
    if (pacifist_reset) {
        s->players.attacked[pc_p(s)] = 1;
    }
}

PC_FN void pc_military_upgrade(PcState *s, int source, int kind, const PcRules *r, const PcGeom *g) {
    int n = s->n;
    float before = pc_kernels_max_health(s, r, source);
    int old = s->units.kind[source];
    s->units.kind[source] = (int32_t)kind;
    /* military.rs upgrade: can attack reads the new type; a move-only lock lifts on a Dash type */
    if (s->units.move_lock[source] && r->dash[kind]) s->units.attacked[source] = 0;
    s->units.move_lock[source] = 0;
    float hp = pc_fmaxf(pc_kernels_max_health(s, r, source) - (before - s->units.health[source]), 1.0f);
    s->units.health[source] = hp;
    int p = pc_p(s);
    s->players.stars[p] -= r->unit_cost[kind];
    s->players.score[p] = pc_wadd(s->players.score[p], 5 * (r->unit_cost[kind] - r->unit_cost[old]));
    int radius = r->skills[kind][PC_SK_SCOUT] ? 2 : 1;
    uint8_t reveal[PC_MAXN];
    for (int t = 0; t < n; t++) reveal[t] = (uint8_t)(pc_d(g, source, t) <= radius);
    pc_economy_discover(s, reveal, s->current_player);
    /* upgrade action's train unit: check step on poison */
    pc_cymanti_step_on_poison(s, source, r);
}

#ifdef __cplusplus
}
#endif
#endif /* PC_MILITARY_H */
