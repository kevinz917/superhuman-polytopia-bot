/* military.h - line-by-line port of engine-rs/src/military.rs (gpu/military.py: unit actions, native profile).
 *
 * Every `pub` item of military.rs is `pc_military_<name>`; private helpers keep their Rust name under the same
 * prefix. Argument order follows the Rust. Rust `Option<usize>` -> int with -1 = None; `Option<&[bool]>` -> a
 * `const uint8_t *` that may be NULL; `impl FnMut(usize)` emitters -> an out list + count. */
#ifndef PC_MILITARY_H
#define PC_MILITARY_H

#include "pc.h"
#include "hash.h"         /* military.rs draws no hash (push is deterministic); included so the module set reads alike */
#include "rules.h"        /* pc_rules_bits128 (rules::bits128), pc_cost_le (CostIndex::le) */
#include "kernels.h"      /* pc_kernels_combat (tuple -> out-params), pc_kernels_max_health, pc_kernels_bridge_blocks */
#include "diplomacy.h"    /* pc_diplomacy_protected, pc_diplomacy_hostile */
#include "economy.h"      /* pc_economy_discover */
#include "structures.h"   /* pc_structures_current_capital (Option<usize> -> tile or -1) */

#ifdef __cplusplus
extern "C" {
#endif

/* forward declarations (same translation unit) */
PC_FN void pc_military_area_damage(PcState *s, int source, int center, const PcRules *r, const PcGeom *g, uint8_t stomp);
PC_FN int32_t pc_military_direction(int size, int from, int to);
PC_FN int32_t pc_military_spawn_direction(int size, int tile);
PC_FN int pc_military_attack_targets(const PcState *s, int source, const PcRules *r, const PcGeom *g, uint16_t *out);

PC_FN void pc_military_exhaust(PcState *s, int source) {
    s->units.moved[source] = 1;
    s->units.attacked[source] = 1;
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
    if (s->units.poison[target] && s->tiles.structure[target] == 0) {
        s->tiles.resource[target] = 7;
    }
    /* AttackAction.Execute (game version >= 95): no xp for a killer that is a vehicle (Carry: every
     * boat) or Static; the passenger of a boat keeps the xp it had when it embarked */
    uint8_t xp = killer >= 0 && !(r->skills[u->kind[k]][PC_SK_CARRY] || r->stat[u->kind[k]]);
    s->units.kills[k] += xp ? 1 : 0;
    pc_units_clear(&s->units, target);
}

/* relocate; `stomp` false for a MoveAction with MoveReason 2 (a push), which skips the arrival abilities.
 * `reveal` NULL = None (the unit's own sight is revealed). */
PC_FN void pc_military_relocate_as(PcState *s, int source, int target, const PcRules *r, const PcGeom *g, uint8_t involuntary,
                                   const uint8_t *reveal, uint8_t stomp) {
    int n = s->n, size = s->size;
    int32_t owner = s->units.owner[source];
    int kind = s->units.kind[source];
    int32_t passenger_src = s->units.passenger[source];
    uint8_t moved_src = s->units.moved[source];
    uint8_t attacked_src = s->units.attacked[source];
    uint8_t invisible_src = s->units.invisible[source];
    int32_t terrain_src = s->tiles.terrain[source];
    pc_units_move(&s->units, source, target);
    s->units.moved[target] = 1;
    s->units.attacked[target] = 1;
    s->units.prev_x[target] = (int32_t)(source % size);
    s->units.prev_y[target] = (int32_t)(source / size);
    /* MoveAction: SetUnitDirection from the path's last step (core::move_adjacent refines multi-tile moves) */
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
    if (stomp && r->skills[kind][PC_SK_STOMP]) {
        pc_military_area_damage(s, target, target, r, g, 1);
    }
    int32_t tt = s->tiles.terrain[target];
    uint8_t embark = s->tiles.structure[target] == 8 && !r->skills[kind][PC_SK_FLOAT];
    uint8_t special_land = (kind == 39 || kind == 40 || kind == 45) && tt > 2;
    uint8_t disembark = r->skills[kind][PC_SK_CARRY] && (passenger_src > 0 || special_land) && tt > 2 && !embark;
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
    /* EmbarkAction / DisembarkAction.ExecuteDefault set moved and attacked on the new unit: no Dash after either */
    uint8_t dash = !involuntary && !moved_src && !attacked_src && r->dash[new_kind] && !embark && !disembark;
    s->units.kind[target] = (int32_t)new_kind;
    s->units.passenger[target] = passenger;
    s->units.attacked[target] = dash ? 0 : (uint8_t)(s->units.attacked[target] || disembark);
    if (embark && r->skills[new_kind][PC_SK_STOMP] && !r->skills[kind][PC_SK_STOMP]) {
        /* EmbarkAction.ExecuteDefault: a new boat with Stomp (Giant -> Juggernaut) stomps where it embarked */
        pc_military_area_damage(s, target, target, r, g, 1);
    }
    uint8_t hidden = r->skills[new_kind][PC_SK_HIDE] && !involuntary;
    s->units.invisible[target] = (uint8_t)(s->units.invisible[target] || hidden);
    s->units.attacked[target] = (uint8_t)(s->units.attacked[target] || (hidden && !invisible_src));
    int o = pc_pwrap(s, owner);
    s->players.score[o] = pc_wadd(s->players.score[o], 5 * (score_after - score_before));
}

PC_FN void pc_military_relocate(PcState *s, int source, int target, const PcRules *r, const PcGeom *g, uint8_t involuntary,
                                const uint8_t *reveal) {
    pc_military_relocate_as(s, source, target, r, g, involuntary, reveal, 1);
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

/* UnitDataExtensions.CanDisembark (DisembarkCommand, v > 104): a boat with a passenger standing on a Bridge
 * (ImprovementAbility.Bridge; the other case, a wetland tile, does not occur in these rules). No moved, attacked
 * or owner check of its own: CommandUtils lists unit commands for the mover's units only. */
PC_FN uint8_t pc_military_can_disembark(const PcState *s, const PcRules *r, int i) {
    int kind = pc_clamp(s->units.kind[i], 0, 127);
    uint8_t carries = s->units.passenger[i] > 0 || kind == 39 || kind == 40 || kind == 45;
    return s->units.owner[i] != 0 && r->skills[kind][PC_SK_CARRY] && carries && s->tiles.structure[i] == 48;
}

/* DisembarkAction.ExecuteDefault (v >= 109) for a boat staying on its tile: the passenger takes the boat's place
 * with the boat's health (no boat of these rules has Protect), its facing (direction and flip copied) and
 * moved = attacked = true; DecreaseScoreAction takes the boat's part of the score (the amounts of a landing move).
 * No exploration and no arrival ability (no stomp, no hiding). The Grow upgrade branch needs a Grow passenger,
 * which these rules cannot carry. */
PC_FN void pc_military_disembark(PcState *s, int i, const PcRules *r) {
    int kind = s->units.kind[i];
    int32_t passenger = s->units.passenger[i];
    int new_kind = pc_military_landed_kind(kind, passenger);
    int32_t before = r->unit_cost[kind] + (passenger > 0 ? r->unit_cost[passenger] : 0);
    int32_t after = r->unit_cost[new_kind];
    s->units.kind[i] = (int32_t)new_kind;
    s->units.passenger[i] = 0;
    s->units.moved[i] = 1;
    s->units.attacked[i] = 1;
    int o = pc_pwrap(s, s->units.owner[i]);
    s->players.score[o] = pc_wadd(s->players.score[o], 5 * (after - before));
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
    can = can && (r->unit_attack[kind] > 0.0f || kind == 10 || infil);
    int32_t range = r->unit_range[kind];
    if (!can || range < 1) {
        return 0;
    }
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

PC_FN void pc_military_attack(PcState *s, int source, int target, const PcRules *r, const PcGeom *g) {
    int size = s->size;
    int32_t cp = s->current_player;
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
    /* BattleHelpers.CanDefenderRetaliate: the attacker must also be one of the defender's attack
     * options, i.e. stand on a tile the defender's owner has explored */
    uint8_t seen = s->tiles.explored[pc_pclip(s, s->units.owner[target])][source];
    uint8_t retaliate = !r->skills[kind][PC_SK_SURPRISE] && !r->stiff[defender] && pc_d(g, source, target) <= r->unit_range[defender] && seen;
    /* BattleResults.shouldResetPacifistTask: damage dealt to a visible unit */
    if (damage > 0.0f && !s->units.invisible[target]) {
        s->players.attacked[pc_p(s)] = 1;
    }
    /* AttackAction: both units face each other */
    s->units.direction[source] = pc_military_direction(size, source, target);
    s->units.direction[target] = pc_military_direction(size, target, source);
    s->units.health[target] += -damage;
    s->units.boosted[target] = 0;
    s->units.prev_x[source] = 2 * (int32_t)(source % size) - (int32_t)(target % size);
    s->units.prev_y[source] = 2 * (int32_t)(source / size) - (int32_t)(target / size);
    s->units.last_attack[source] = (int32_t)target + 1;
    if (r->skills[kind][PC_SK_SPLASH]) {
        pc_military_area_damage(s, source, target, r, g, 0);
        /* AttackCommand.ExecuteDefault pushes the retaliation AttackAction (Target -> Origin, when its damage is >= 1;
         * AttackCommand.c:8029) before the splash ones (8177) and the main attack (8210); the action stack is LIFO,
         * so it runs last and the attacker ends facing the defender again */
        if (!dead && retaliate && retaliation > 0.0f) {
            s->units.direction[source] = pc_military_direction(size, source, target);
        }
    }
    if (dead) {
        pc_military_remove(s, target, r, source, 1);
    } else {
        s->units.health[source] += -(retaliate ? retaliation : 0.0f);
    }
    uint8_t attacker_dead = s->units.health[source] <= 0.0f;
    if (attacker_dead) {
        pc_military_remove(s, source, r, target, 1);
    }
    int32_t st = s->tiles.structure[target];
    uint8_t can_enter = terrain == 3 || terrain == 5 || (terrain == 4 && s->players.tech[pc_p(s)][20])
        || (terrain <= 2 && (r->skills[kind][PC_SK_FLOAT] || st == 48 || (st == 8 && s->tiles.owner[target] == cp)));
    uint8_t advance = dead && !attacker_dead && r->unit_range[kind] < 2 && can_enter;
    if (advance) {
        pc_military_relocate(s, source, target, r, g, 1, NULL);
    }
    int at = advance ? target : source;
    if (!attacker_dead) {
        uint8_t persist = r->persist[kind];
        uint8_t escape = r->escape[kind];
        s->units.attacks_performed[at] += 1;
        s->units.attacked[at] = !(dead && persist);
        s->units.moved[at] = !escape;
        if (dead && persist) {
            /* AttackCommand.ExecuteV104: Persist leaves the unit able to attack only when it has
             * another target from where it now stands
             * (GetAttackOptionsAtPosition with includeHidden when range <= sight: hidden and unexplored enemies count) */
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
            s->units.attacked[at] = !more;
        }
    }
}

PC_FN void pc_military_convert(PcState *s, int source, int target, const PcRules *r, const PcGeom *g) {
    int n = s->n;
    int32_t cp = s->current_player;
    int old = pc_pwrap(s, s->units.owner[target]);
    int tk = s->units.kind[target];
    /* ConvertAction.ExecuteDefault: rehomed to the converter's current capital (none: no home) */
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
    s->units.owner[target] = cp;
    s->units.converted[target] = 1;
    s->units.home[target] = home;
    s->units.moved[target] = 1;
    s->units.attacked[target] = 1;
    /* AttackCommand.ExecuteDefault also pushes the main AttackAction (Origin -> Target, damage = the converter's
     * attack damage, 0; AttackCommand.c:8210) after the ConvertAction (7800); it runs first and turns the two units
     * toward each other (AttackAction.c:726, 753). No retaliation: BattleHelpers.CanDefenderRetaliate is false
     * against Convert (BattleHelpers.c:460). */
    s->units.direction[source] = pc_military_direction(s->size, source, target);
    s->units.direction[target] = pc_military_direction(s->size, target, source);
    pc_military_exhaust(s, source);
    int radius = (s->tiles.terrain[target] == 4 || r->skills[tk][PC_SK_SCOUT]) ? 2 : 1;
    uint8_t reveal[PC_MAXN];
    for (int t = 0; t < n; t++) reveal[t] = (uint8_t)(pc_d(g, target, t) <= radius);
    pc_economy_discover(s, reveal, cp);
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
} PcSummonCtx;

/* SummonCtx::new(n) */
PC_FN void pc_military_summon_ctx_new(PcSummonCtx *ctx) {
    for (int i = 0; i < PC_MAXN; i++) ctx->counts[i] = 0;
    ctx->ok = pc_u128_zero();
}

/* SummonCtx::prepare over cooperating lanes (pc.h PcLanes): the counts are a scatter-add (atomic with lanes;
 * integer sums do not depend on the order) */
PC_FN void pc_military_summon_prepare_lanes(PcSummonCtx *ctx, const PcState *s, const PcRules *r, PcLanes L) {
    int n = s->n;
    const PcUnits *u = &s->units;
    int p = pc_p(s);
    for (int i = L.lane; i < n; i += L.count) ctx->counts[i] = 0;
    pc_lanes_sync(L);
    for (int i = L.lane; i < n; i += L.count) {
        /* MapDataExtensions.GetCityUnitCount: every unit homed at the city counts, whoever owns it now */
        if (u->home[i] >= 1 && u->home[i] <= n) {
            pc_lanes_add(L, &ctx->counts[u->home[i] - 1], 1);
        }
    }
    pc_u128 unlocked = r->unit_static;
    pc_u128 known = pc_rules_bits128(s->players.tech[p]);
    while (pc_u128_any(known)) {
        int k = pc_u128_lowest(known);
        unlocked = pc_u128_or(unlocked, r->units_by_tech[k]);
        pc_u128_clear(&known, k);
    }
    if (L.lane == 0) ctx->ok = pc_u128_and(unlocked, pc_cost_le(&r->unit_cost_le, s->players.stars[p]));
    pc_lanes_sync(L);
}

/* SummonCtx::prepare */
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
    pc_u128 a = available ? pc_u128_andnot(ctx->ok, up) : pc_u128_zero();
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
    /* TrainAction: ExploreFromTile with UnitDataExtensions.GetSightRange (Scout units see 2 tiles) */
    int radius = (s->tiles.terrain[source] == 4 || r->skills[kind][PC_SK_SCOUT]) ? 2 : 1;
    uint8_t reveal[PC_MAXN];
    for (int t = 0; t < n; t++) reveal[t] = (uint8_t)(pc_d(g, source, t) <= radius);
    pc_economy_discover(s, reveal, cp);
}

/* The unit on `i` may be promoted: 3 kills, not yet a veteran, not Static
 * (UnitDataExtensions.CanBePromoted): a boat is judged by its passenger, so a non-Static passenger is promoted at sea. */
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

/* Disband the unit on `source`, refunding half its full cost (UnitState.GetFullCost: boat plus passenger) to seat
 * index `p` (its owner). */
PC_FN void pc_military_disband_of(PcState *s, int source, const PcRules *r, int p) {
    int kind = s->units.kind[source];
    int32_t passenger = s->units.passenger[source];
    int32_t full = r->unit_cost[kind] + (passenger > 0 ? r->unit_cost[passenger] : 0);
    pc_military_remove(s, source, r, -1, 0);
    s->players.stars[p] += pc_div_euclid(full, 2);
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

/* GridDirections.ToCoordinates: SW, W, NW, N, NE, E, SE, S as (dx, dy) in game coordinates (y is the row).
 * Rust `pub const GRID: [(i32, i32); 8]`; here two accessors over function-local tables (device-safe). */
PC_FN int32_t pc_military_grid_dx(int d) {
    static const int32_t GX[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
    return GX[d];
}
PC_FN int32_t pc_military_grid_dy(int d) {
    static const int32_t GY[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
    return GY[d];
}
/* MapDataExtensions.GetTileNeighbors order (x outer). */
PC_FN int32_t pc_military_steps_dx(int k) {
    static const int32_t SX[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    return SX[k];
}
PC_FN int32_t pc_military_steps_dy(int k) {
    static const int32_t SY[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    return SY[k];
}

/* WorldCoordinates.GetDirectionFromDistanceVector: the direction with the largest weighted dot product with the
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

/* WorldCoordinates.GetDirection (WorldCoordinates.c:10-153) on coordinates that may be off the board: the game's
 * null coordinates (-1, -1) (WorldCoordinates..cctor, WorldCoordinates.c:833-844) are the origin of the attack in
 * InfiltrationRewardAction.ExecuteDefault. Rust tuples (from, to) -> (fx, fy, tx, ty). */
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

/* WorldCoordinates.GetDirection(from, to): a neighbour's direction (8, none, for the same tile), farther tiles by
 * vector_direction. */
PC_FN int32_t pc_military_direction(int size, int from, int to) {
    return pc_military_direction_xy((int32_t)(from % size), (int32_t)(from / size), (int32_t)(to % size), (int32_t)(to / size));
}

/* UnitState.Create: a new unit faces the map centre (width / 2, height / 2). */
PC_FN int32_t pc_military_spawn_direction(int size, int tile) {
    int32_t c = (int32_t)(size / 2);
    return pc_military_vector_direction(c - (int32_t)(tile % size), c - (int32_t)(tile / size));
}

/* PathFinder.GetPath(source, dest, 1, unit) for a neighbour `dest` of the unit on `source`: IsAllowedToFinishOnTile
 * and IsTileAccessible under PathFinderSettings.CreateForUnit (enemy territory allowed, occupied tiles not). */
PC_FN uint8_t pc_military_push_reachable(const PcState *s, int source, int dest, const PcRules *r) {
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
        return !pc_kernels_bridge_blocks(s, owner, source, dest);
    }
    if (structure == 1) {
        return 1;
    }
    const uint8_t *tech = s->players.tech[p];
    uint8_t terrain_ok = (terrain != 1 || tech[10]) && (terrain != 2 || tech[13] || tech[25]) && (terrain != 4 || tech[20]);
    uint8_t land_access = (terrain != 1 && terrain != 2) || (structure == 8 && t->owner[dest] == owner);
    return terrain_ok && (land_access || r->skills[kind][PC_SK_FLOAT]);
}

/* water(t) closure of push */
PC_FN uint8_t pc_military_push_water(const PcState *s, int t) {
    return s->tiles.terrain[t] == 1 || s->tiles.terrain[t] == 2;
}

/* ActionUtils.TryPushUnitDefault, with TrainUnitOnOccupiedSpace's KillUnitAction when it fails: moves the unit on
 * `source` off it for a unit the current player trains there. The occupant's facing (reversed for another player's
 * unit) is tried first, then the directions -1, +1, -2, +2, -3, +3, -4 from it; a boat on water keeps the first land
 * tile as its last resort. The push is a MoveAction with MoveReason 2: no stomp, and moved/attacked stay as they were
 * unless the unit embarks or disembarks (those actions exhaust it). Deterministic, no hash draw. */
PC_FN void pc_military_push(PcState *s, int source, const PcRules *r, const PcGeom *g) {
    int32_t size = s->size;
    int32_t owner = s->units.owner[source];
    if (owner <= 0) {
        return;
    }
    /* ActionUtils.TrainUnitOnOccupiedSpace (ActionUtils.c:1071-1081, game version > 0x69): an occupant hidden from the
     * trainer (UnitState.IsInvisible: another player's, no peace with them) gets a RevealAction without an origin
     * (no turn, no meeting). The action stack runs it before the push's MoveAction, so the unit stays revealed. */
    int32_t cp = s->current_player;
    if (s->units.invisible[source] && owner != cp && !s->players.peace[pc_p(s)][pc_pclip(s, owner)]) {
        s->units.invisible[source] = 0;
    }
    int32_t facing = s->units.direction[source];
    if (owner != s->current_player) {
        facing = facing < 4 ? facing + 4 : facing - 4;
    }
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
        if (pc_military_push_reachable(s, source, t, r)) {
            target = t;
            break;
        }
    }
    /* target.or(fallback.filter(push_reachable)) */
    int chosen = target;
    if (chosen < 0 && fallback >= 0 && pc_military_push_reachable(s, source, fallback, r)) {
        chosen = fallback;
    }
    if (chosen >= 0) {
        int t = chosen;
        uint8_t moved = s->units.moved[source], attacked = s->units.attacked[source];
        pc_military_relocate_as(s, source, t, r, g, 1, NULL, 0);
        if (s->units.kind[t] == kind) {
            s->units.moved[t] = moved;
            s->units.attacked[t] = attacked;
        }
    } else {
        pc_military_remove(s, source, r, -1, 1);
    }
}

PC_FN void pc_military_superunit(PcState *s, int source, const PcRules *r, const PcGeom *g) {
    pc_military_push(s, source, r, g);
    pc_military_summon(s, source, 12, r, g, 0);
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
    /* the seats hit (ModifyDiplomacyRelationsFromAttack per damaged unit's owner) */
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
            dmg = floorf(dmg);
            damage[i] = dmg;
            dead[i] = u->health[i] - dmg <= 0.0f;
            if (dmg > 0.0f) {
                hit_seats[pc_clamp(u->owner[i], 0, PC_MAXP)] = 1;
            }
            pacifist_reset = (uint8_t)(pacifist_reset || (dmg > 0.0f && !u->invisible[i]));
        }
    }
    /* One AttackAction per target, pushed in GetTileNeighbors (x-outer) order, from the stomper's tile for a stomp
     * (ActionUtils.AreaDamageDefault, ActionUtils.c:2878-2971: origin = the centre) and from the attacker's tile for
     * a splash (AttackCommand.ExecuteDefault, AttackCommand.c:8112-8177: AttackAction(Origin, neighbour)). Each
     * AttackAction turns its target toward the origin and the origin's unit toward the target (AttackAction.c:726,
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
    for (int i = 0; i < n; i++) {
        if (dead[i]) {
            pc_military_remove(s, i, r, source, 1);
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
    float hp = pc_fmaxf(pc_kernels_max_health(s, r, source) - (before - s->units.health[source]), 1.0f);
    s->units.health[source] = hp;
    int p = pc_p(s);
    s->players.stars[p] -= r->unit_cost[kind];
    s->players.score[p] = pc_wadd(s->players.score[p], 5 * (r->unit_cost[kind] - r->unit_cost[old]));
    int radius = r->skills[kind][PC_SK_SCOUT] ? 2 : 1;
    uint8_t reveal[PC_MAXN];
    for (int t = 0; t < n; t++) reveal[t] = (uint8_t)(pc_d(g, source, t) <= radius);
    pc_economy_discover(s, reveal, s->current_player);
}

#ifdef __cplusplus
}
#endif
#endif /* PC_MILITARY_H */
