/* cymanti.h - line-by-line port of engine-rs/src/cymanti.rs (Cymanti, increment S4): Algae (step 3). Rules from
 *cited in `tiles.algae` is
 * tile effect algae. The Algae build itself is pc_structures_build_algae (structures.h, cymanti:build_algae) and
 * resource_visible pc_cymanti_resource_visible (tribes.h). */
#ifndef PC_CYMANTI_H
#define PC_CYMANTI_H
#define PC_CYMANTI_MYCELIUM 39   /* cymanti::MYCELIUM: the Network router with Heal */

#include "pc.h"
#include "tribes.h"
#include "structures.h"   /* pc_structures_update_routes, pc_structures_connections, PcStructScratch */
#include "diplomacy.h"    /* pc_diplomacy_protected, pc_diplomacy_hostile, pc_military_remove (declared) */
#include "kernels.h"      /* pc_kernels_combat, pc_kernels_max_health */


#ifdef __cplusplus
extern "C" {
#endif

/* defined in military.h / aquarion.h, which include this file; declared inside the extern "C" block so nvcc
 * (C++) gives them the same linkage as the definitions */
PC_FN float pc_military_defense_bonus(const PcState *s, const PcRules *r, int i);
PC_FN int pc_aquarion_floodable(const PcState *s, int t);
PC_FN void pc_aquarion_flood_tile(PcState *s, int t, int32_t player, const PcRules *r, const PcGeom *g, PcStructScratch *sc);
/* defined in military.h (Cymanti step 6: the Eat Segment, the chain facing) */
PC_FN void pc_military_summon(PcState *s, int source, int kind, const PcRules *r, const PcGeom *g, uint8_t costs);
PC_FN int32_t pc_military_grid_dx(int d);
PC_FN int32_t pc_military_grid_dy(int d);
PC_FN int32_t pc_military_vector_direction(int32_t vx, int32_t vy);
PC_FN int32_t pc_military_direction(int size, int from, int to);

/* get unit actions' clear tile effect Algae: the mover's unit on Algae, able to attack (not
 * attacked, not Frozen) or to move (not moved, not Frozen), with Amphibious, Swim or Fly; any tribe */
PC_FN int pc_cymanti_can_clear_algae(const PcState *s, int i, const PcRules *r) {
    const PcUnits *u = &s->units;
    int kind = (int)pc_clamp(u->kind[i], 0, 127);
    return u->owner[i] == s->current_player && u->owner[i] != 0 && s->tiles.algae[i] && !u->frozen[i]
        && (!u->attacked[i] || !u->moved[i])
        && (r->amphibious[kind] || r->skills[kind][PC_SK_FLOAT] || r->fly[kind]);
}

/* clear tile effect action(Algae, by unit): the effect goes, the unit is exhausted, then update routes and
 * update city connections(player) */
PC_FN void pc_cymanti_clear_algae_command(PcState *s, int i, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int32_t player = s->current_player;
    s->tiles.algae[i] = 0;
    s->units.moved[i] = 1;
    s->units.attacked[i] = 1;
    s->units.move_lock[i] = 0;
    s->units.boosted[i] = 0;
    s->units.double_ready[i] = 0;
    pc_structures_update_routes(s, r, sc);
    pc_structures_connections(s, player, g, 1);
}

/* spawn spore resource (cymanti:spawn_spores): Spores overwrite the resource; on water the Algae too, its
 * routes and connections deferred to the command's end (algae_dirty) */
PC_FN void pc_cymanti_spawn_spores(PcState *s, int t) {
    if (s->tiles.terrain[t] == 1 || s->tiles.terrain[t] == 2) {
        s->tiles.algae[t] = 1;
        s->algae_dirty = 1;
    }
    s->tiles.resource[t] = 7;
}

/* move action's Algae ability (cymanti:algae_trail): each path tile that is water without Algae gets it */
PC_FN void pc_cymanti_algae_trail(PcState *s, const uint16_t *tiles, int count) {
    for (int k = 0; k < count; k++) {
        int t = tiles[k];
        if ((s->tiles.terrain[t] == 1 || s->tiles.terrain[t] == 2) && !s->tiles.algae[t]) {
            s->tiles.algae[t] = 1;
            s->algae_dirty = 1;
        }
    }
}

/* check step on poison (cymanti:step_on_poison): a unit on Fungi (37) or Clathrus (41) is Poisoned unless
 * its owner is Cymanti or it flies */
PC_FN void pc_cymanti_step_on_poison(PcState *s, int t, const PcRules *r) {
    int32_t st = s->tiles.structure[t];
    int32_t owner = s->units.owner[t];
    if ((st == 37 || st == 41) && owner > 0) {
        int kind = (int)pc_clamp(s->units.kind[t], 0, 127);
        if (s->players.tribe[pc_pclip(s, owner)] != PC_CYMANTI && !r->fly[kind]) s->units.poison[t] = 1;
    }
}

/* A Poison splasher's splash (cymanti::poison_splash): every non-peace neighbour unit of the target is Poisoned */
PC_FN void pc_cymanti_poison_splash(PcState *s, int source, int target, const PcGeom *g) {
    int32_t owner = s->units.owner[source];
    for (int k = 0; k < g->adj_n[target]; k++) {
        int j = (int)g->adj[target * 8 + k];
        int32_t o = s->units.owner[j];
        if (o != 0 && o != owner && !pc_diplomacy_protected(s, owner, o)) s->units.poison[j] = 1;
    }
}

/* can explode (cymanti:can_explode): own, Explode, not Frozen, not created this turn */
PC_FN int pc_cymanti_can_explode(const PcState *s, int i, const PcRules *r) {
    const PcUnits *u = &s->units;
    int kind = (int)pc_clamp(u->kind[i], 0, 127);
    /* not on its creation turn unless it has a leader (a Segment eaten this turn) */
    return u->owner[i] == s->current_player && u->owner[i] != 0 && r->explode[kind] && !u->frozen[i]
        && (u->created[i] != s->turn || u->leader[i] != 0);
}

/* explode unit action (cymanti:explode_command): Spores on the tile; each non-peace unit in radius 1 takes the floored
 * half of the exploder's battle damage from its own tile (kill to the player; Boosted, Swift, Bubble lost; a Drench
 * victim floods its tile), relations and Pacifist as for an attack, a survivor is Poisoned; the exploder goes
 * without a kill unit */
PC_FN int pc_cymanti_unlink(PcState *s, int i);

/* one explode unit action; returns the unit's follower (unlinked first), -1 none */
PC_FN int pc_cymanti_explode_one(PcState *s, int src, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int n = s->n;
    int32_t cp = s->current_player;
    int p = pc_p(s);
    int kind = (int)pc_clamp(s->units.kind[src], 0, 127);
    int follower = pc_cymanti_unlink(s, src);
    pc_cymanti_spawn_spores(s, src);
    float aa = r->unit_attack[kind], ah = s->units.health[src], am = pc_kernels_max_health(s, r, src);
    uint16_t hit_tile[9];
    float hit_dmg[9];
    int hits = 0;
    for (int t = 0; t < n; t++) {
        int32_t o = s->units.owner[t];
        if (pc_d(g, src, t) > 1 || o == 0 || pc_diplomacy_protected(s, cp, o)) continue;
        float dd = r->unit_defense[pc_clamp(s->units.kind[t], 0, 127)];
        float damage, retaliation, half;
        pc_kernels_combat(aa, ah, am, dd, s->units.health[t], pc_kernels_max_health(s, r, t), pc_military_defense_bonus(s, r, t),
                          &damage, &retaliation, &half);
        hit_tile[hits] = (uint16_t)t;
        hit_dmg[hits] = floorf(half);
        hits++;
        pc_diplomacy_hostile(s, o);
        s->players.attacked[p] = 1;
    }
    pc_military_remove(s, src, r, -1, 0);
    for (int k = 0; k < hits; k++) {
        int t = (int)hit_tile[k];
        int vk = (int)pc_clamp(s->units.kind[t], 0, 127);
        s->units.boosted[t] = 0;
        s->units.swift[t] = 0;
        s->units.bubble[t] = 0;
        s->units.health[t] -= pc_fminf(hit_dmg[k], s->units.health[t]);
        uint8_t dead = s->units.health[t] <= 0.0f;
        if (r->drench[vk] && pc_aquarion_floodable(s, t)) pc_aquarion_flood_tile(s, t, cp, r, g, sc);
        if (dead) {
            s->players.kills[p] += 1;
            pc_military_remove(s, t, r, -1, 1);
        } else {
            s->units.poison[t] = 1;
        }
    }
    return follower;
}

/* explode unit action (cymanti:explode_command): a follower gets its own explode unit action, run after this one (the chain
 * explodes tailwards; the Rust recursion as a loop) */
PC_FN void pc_cymanti_explode_command(PcState *s, int src, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int at = src, guard = 0;
    while (at >= 0 && guard++ < s->n) {
        int f = pc_cymanti_explode_one(s, at, r, g, sc);
        at = (f >= 0 && s->units.owner[f] != 0) ? f : -1;
    }
}

/* get boost options (cymanti:boost_options): neighbours with a unit of the player, not Boosted, not Swift */
PC_FN int pc_cymanti_boost_option(const PcState *s, int i, int j) {
    return s->units.owner[j] == s->units.owner[i] && !s->units.boosted[j] && !s->units.swift[j] && s->units.leader[j] == 0;
}

/* Swarm (cymanti:can_swarm): own, Swarm, can attack, a boost option */
PC_FN int pc_cymanti_can_swarm(const PcState *s, int i, const PcRules *r, const PcGeom *g) {
    const PcUnits *u = &s->units;
    int kind = (int)pc_clamp(u->kind[i], 0, 127);
    if (!(u->owner[i] == s->current_player && u->owner[i] != 0 && r->swarm[kind] && !u->attacked[i] && !u->frozen[i])) return 0;
    for (int k = 0; k < g->adj_n[i]; k++) {
        if (pc_cymanti_boost_option(s, i, (int)g->adj[i * 8 + k])) return 1;
    }
    return 0;
}

/* swarm command (cymanti:swarm_command): the unit moved and attacked, every boost option Swift */
PC_FN void pc_cymanti_swarm_command(PcState *s, int i, const PcGeom *g) {
    uint8_t opt[8];
    int m = g->adj_n[i];
    for (int k = 0; k < m; k++) opt[k] = (uint8_t)pc_cymanti_boost_option(s, i, (int)g->adj[i * 8 + k]);
    s->units.moved[i] = 1;
    s->units.attacked[i] = 1;
    s->units.move_lock[i] = 0;
    for (int k = 0; k < m; k++) {
        if (opt[k]) s->units.swift[g->adj[i * 8 + k]] = 1;
    }
}

/* ---- step 6: Centipede chains (cymanti.rs) ---- */
#define PC_CYMANTI_CENTIPEDE 33
#define PC_CYMANTI_SEGMENT 34

/* cymanti::unlink: the unit on `i` out of its chain (its leader loses it, its follower its leader); returns the
 * follower, -1 none */
PC_FN int pc_cymanti_unlink(PcState *s, int i) {
    if (s->units.leader[i] > 0) s->units.follower[s->units.leader[i] - 1] = 0;
    int f = s->units.follower[i] > 0 ? (int)s->units.follower[i] - 1 : -1;
    if (f >= 0) s->units.leader[f] = 0;
    s->units.leader[i] = 0;
    s->units.follower[i] = 0;
    return f;
}

/* cymanti:upgrade_head: upgrade action(Centipede, cost 0) on a split-off Segment: created this turn, no xp or
 * promotion, health = new max - old damage (at least 0.1), a Dash type lifts the move-only lock, score + 5 x the cost
 * difference, explore from tile, check step on poison */
PC_FN void pc_cymanti_upgrade_head(PcState *s, int i, const PcRules *r) {
    const int c = PC_CYMANTI_CENTIPEDE;
    int old = (int)pc_clamp(s->units.kind[i], 0, 127);
    float damage = pc_kernels_max_health(s, r, i) - s->units.health[i];
    s->units.kind[i] = c;
    s->units.veteran[i] = 0;
    s->units.kills[i] = 0;
    s->units.created[i] = s->turn;
    s->units.health[i] = pc_fmaxf(r->unit_health[c] - damage, 0.1f);
    if (s->units.move_lock[i] && r->dash[c]) {
        s->units.attacked[i] = 0;
        s->units.move_lock[i] = 0;
    }
    int32_t owner = s->units.owner[i];
    int o = pc_pwrap(s, owner);
    s->players.score[o] = pc_wadd(s->players.score[o], 5 * (r->unit_cost[c] - r->unit_cost[old]));
    int n = s->n, size = s->size;
    int radius = (s->tiles.terrain[i] == 4 || r->skills[c][PC_SK_SCOUT]) ? 2 : 1;
    int x = i % size, y = i / size;
    uint8_t reveal[PC_MAXN];
    for (int t = 0; t < n; t++) reveal[t] = (uint8_t)(pc_max(pc_abs(t % size - x), pc_abs(t / size - y)) <= radius);
    pc_economy_discover(s, reveal, owner);
    pc_cymanti_step_on_poison(s, i, r);
}

/* cymanti:kill_links (kill unit's chain part): the follower takes the dying unit's moved and attacked and
 * becomes a Centipede head; the leader loses the unit */
PC_FN void pc_cymanti_kill_links(PcState *s, int i, const PcRules *r) {
    uint8_t moved = s->units.moved[i], attacked = s->units.attacked[i], lock = s->units.move_lock[i];
    int f = pc_cymanti_unlink(s, i);
    if (f >= 0 && s->units.owner[f] != 0) {
        s->units.moved[f] = moved;
        s->units.attacked[f] = attacked;
        s->units.move_lock[f] = lock;
        pc_cymanti_upgrade_head(s, f, r);
    }
}

/* cymanti::tail: the last unit of the follower chain from `i` */
PC_FN int pc_cymanti_tail(const PcState *s, int i) {
    int at = i, guard = 0;
    while (s->units.follower[at] > 0 && guard++ <= s->n) at = (int)s->units.follower[at] - 1;
    return at;
}

/* cymanti:eat (eat action): the first neighbour of the eater (the chain's tail) that is empty and accessible to a land
 * Segment led by it, trying the direction opposite its facing, then -1, +1, -2, +2, -3, +3, -4, gets train unit(Segment)
 * (moved and attacked, created this turn), led by the eater; the home is the tile's city, if it is one */
PC_FN void pc_cymanti_eat(PcState *s, int eater, const PcRules *r, const PcGeom *g) {
    int32_t size = s->size;
    int32_t owner = s->units.owner[eater];
    int32_t dir = pc_clamp(s->units.direction[eater], 0, 7);
    int32_t opp = dir < 4 ? dir + 4 : dir - 4;
    uint8_t flags[PC_MAXN];
    pc_kernels_movement_flags(s, owner, g, flags);
    int32_t x = eater % size, y = eater / size;
    int found = -1;
    for (int32_t k = 1; k <= 8 && found < 0; k++) {
        int32_t off = ((k - 1) & 1) == 0 ? k / 2 : -(k >> 1);
        int d = (int)pc_rem_euclid(off + opp + 8, 8);
        int32_t tx = x + pc_military_grid_dx(d), ty = y + pc_military_grid_dy(d);
        if (tx < 0 || ty < 0 || tx >= size || ty >= size) continue;
        int t = (int)(ty * size + tx);
        if (s->units.owner[t] != 0) continue;
        /* is tile accessible for a land Segment: explored, its terrain, land access (a unit with a leader never takes a
         * Port into water), no Bridge span across the step */
        uint8_t f = flags[t];
        int port_only = (f & MV_OWN_PORT) != 0 && (s->tiles.terrain[t] <= 2 || s->tiles.flooded[t]) && !s->tiles.algae[t]
            && s->tiles.structure[t] != 48;
        if ((f & (MV_PASS | MV_LAND)) == (MV_PASS | MV_LAND) && !port_only && !pc_kernels_bridge_blocks(s, owner, eater, t)) found = t;
    }
    if (found < 0) return;
    int32_t cp = s->current_player;
    s->current_player = owner;
    pc_military_summon(s, found, PC_CYMANTI_SEGMENT, r, g, 0);
    s->current_player = cp;
    s->units.home[found] = s->tiles.structure[found] == 1 ? (int32_t)found + 1 : 0;
    s->units.leader[found] = (int32_t)eater + 1;
    s->units.follower[eater] = (int32_t)found + 1;
}

/* cymanti:link_direction (try push unit default's chain facing): toward the leader / follower, or the average of the two
 * (grid directions average: opposite directions give first + 2); -1 for a unit outside a chain */
PC_FN int32_t pc_cymanti_link_direction(const PcState *s, int i) {
    int32_t size = s->size;
    int l = s->units.leader[i] > 0 ? (int)s->units.leader[i] - 1 : -1;
    int f = s->units.follower[i] > 0 ? (int)s->units.follower[i] - 1 : -1;
    if (l < 0 && f < 0) return -1;
    int32_t a = l >= 0 ? pc_military_direction(size, i, l) : 0;
    int32_t b = f >= 0 ? pc_military_direction(size, i, f) : 0;
    if (a > 7) a = 0;
    if (b > 7) b = 0;
    if (l >= 0 && f >= 0) {
        int32_t sx = pc_military_grid_dx(a) + pc_military_grid_dx(b), sy = pc_military_grid_dy(a) + pc_military_grid_dy(b);
        return (sx == 0 && sy == 0) ? (2 + a) % 8 : pc_military_vector_direction(sx, sy);
    }
    return f >= 0 ? b : a;
}

#ifdef __cplusplus
}
#endif
#endif /* PC_CYMANTI_H */
