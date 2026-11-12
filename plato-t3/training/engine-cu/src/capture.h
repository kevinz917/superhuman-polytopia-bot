/* capture.h - line-by-line port of engine-rs/src/capture.rs (gpu/capture.py: village/city capture, starfish and
 * ruins, native profile). Every `pub` item is `pc_capture_<name>`; `Vec<usize>` returns -> out list + count,
 * `Vec<bool>` -> uint8_t[PC_MAXN] locals. */
#ifndef PC_CAPTURE_H
#define PC_CAPTURE_H

#include "pc.h"
#include "hash.h"         /* pc_hash_range (ruins draws) */
#include "diplomacy.h"    /* pc_diplomacy_protected, pc_diplomacy_hostile */
#include "discovery.h"    /* pc_discovery_explorer (Vec<bool> -> out[PC_MAXN]), pc_discovery_movable */
#include "economy.h"      /* add_population, claim, discover, city_score, add_population_vector */
#include "structures.h"   /* adjust_building_population, capital_index, connections, population_contribution, update_routes,
                             reachable, current_capital; PcStructScratch (update_routes needs one) */
#include "military.h"     /* exhaust, promote, push, summon, disband_of */
#include "polaris.h"      /* city_area_climate (Polaris) */
#include "tribes.h"       /* starfishing_tech */

#ifdef __cplusplus
extern "C" {
#endif

/* provided by core.h (core.rs leave: seat k has just left a multi-seat game); core.h includes this header, so a
 * prototype here instead of an include */
PC_FN void pc_core_leave(PcState *s, int k);

/* capture_mask's tile condition (a foreign unprotected village/city, a ruin, or a starfish with Navigation). */
PC_FN uint8_t pc_capture_capturable(const PcState *s, int i) {
    const PcTiles *t = &s->tiles;
    int32_t cp = s->current_player;
    /* Star Fishing is a build (can build), so not on a tile that has an improvement */
    return (t->structure[i] == 1 && t->owner[i] != cp && !pc_diplomacy_protected(s, cp, t->owner[i])) || t->structure[i] == 2
        || (t->resource[i] == 8 && s->players.tech[pc_p(s)][pc_tribes_starfishing_tech(s->players.tribe[pc_p(s)])] && t->structure[i] == 0);
}

PC_FN void pc_capture_capture_mask(const PcState *s, uint8_t out[PC_MAXN]) {
    const PcUnits *u = &s->units;
    int32_t cp = s->current_player;
    for (int i = 0; i < s->n; i++) {
        /* can capture / can examine ruins: no leader (Cymanti) */
        uint8_t idle = u->owner[i] == cp && !u->moved[i] && !u->attacked[i] && u->leader[i] == 0;
        out[i] = idle && pc_capture_capturable(s, i);
    }
}

/* capture::city; `sc` is the PcStructScratch pc_structures_update_routes needs (too large for a device stack), as
 * economy.h's pc_economy_reward takes it. */
PC_FN void pc_capture_city(PcState *s, int source, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int n = s->n;
    int32_t before[PC_MAXN];
    pc_structures_population_contribution(s, r, g, before);
    int32_t owner = s->current_player;
    int32_t old = s->tiles.owner[source];
    uint8_t enemy = old > 0;
    int32_t old_order = s->cities.order[source];
    /* (0..n).filter(owner == owner).map(order).max().unwrap_or(0).max(0) + 1 */
    int32_t order_max = 0;
    for (int i = 0; i < n; i++) {
        if (s->cities.owner[i] == owner && s->cities.order[i] > order_max) order_max = s->cities.order[i];
    }
    int32_t order = order_max + 1;
    uint8_t area[PC_MAXN];
    for (int t = 0; t < n; t++) area[t] = (uint8_t)(pc_d(g, source, t) <= 1);
    /* capture city action / get city area: a captured city explores the square of its border radius and rules
     * the tiles there that no other city rules */
    int32_t radius = pc_max(s->cities.border[source], 1);
    uint8_t city_area[PC_MAXN];
    for (int t = 0; t < n; t++) city_area[t] = (uint8_t)(pc_d(g, source, t) <= radius);
    uint8_t membership[PC_MAXN];
    if (enemy) {
        for (int t = 0; t < n; t++)
            membership[t] = (uint8_t)(city_area[t] && (s->tiles.city[t] < 0 || s->tiles.city[t] == (int32_t)source));
    } else {
        for (int t = 0; t < n; t++) membership[t] = area[t];
    }
    uint8_t was_connected = enemy && s->cities.connected[source];
    /* capture city action: the city's improvement score (calculate improvement score) changes hands */
    int32_t transfer = pc_economy_city_score(s, source);
    int32_t score = enemy ? transfer : 100;
    int32_t old_owners[PC_MAXN];
    for (int i = 0; i < n; i++) old_owners[i] = s->cities.owner[i];
    {
        PcCities *c = &s->cities;
        c->owner[source] = owner;
        if (!enemy) {
            c->level[source] = 1;
            c->population[source] = 0;
            c->progress[source] = 0;
            c->production[source] = 1;
            c->border[source] = 1;
            for (int k = 0; k < 9; k++) {
                c->rewards[source * 9 + k] = 0;
            }
        } else {
            /* capture.rs: capture city action adds no city level up trigger, so unclaimed rewards are not the capturer's;
             * kind 0 makes up the count (no reward pending, the next level-up offers its own level's pair) */
            int32_t count = 0;
            for (int k = 0; k < 9; k++) count += c->rewards[source * 9 + k];
            int32_t owed = c->level[source] - 1 - count;
            if (owed > 0) c->rewards[source * 9] += owed;
        }
        c->connected[source] = (uint8_t)(enemy && c->connected[source]);
        for (int t = 0; t < n; t++) pc_set_territory(s, source, t, membership[t]);
        for (int i = 0; i < n; i++) {
            if (enemy && old_owners[i] == old && c->order[i] > old_order) {
                c->order[i] -= 1;
            }
        }
        c->order[source] = order;
    }
    int po = (int)(owner - 1);
    s->players.score[po] = pc_wadd(s->players.score[po], score);
    int oo = pc_pclip(s, old);
    if (enemy) {
        s->players.score[oo] = pc_wsub(s->players.score[oo], score);
    }
    uint8_t capital_here = s->tiles.capital[source] > 0;
    if (enemy && capital_here) {
        for (int k = 0; k < s->np; k++) {
            s->players.embassy[k][oo] = 0;
        }
    }
    uint8_t any_old = 0;
    for (int i = 0; i < n; i++) {
        if (s->cities.owner[i] == old) { any_old = 1; break; }
    }
    uint8_t eliminated = enemy && !any_old;
    if (eliminated) {
        int oi = (int)(old - 1);
        s->players.killed_turn[oi] = pc_max(s->turn, 1);
        s->players.killer[oi] = owner;
        pc_core_leave(s, oi);
        for (int i = 0; i < n; i++) {
            if (s->units.owner[i] == old) {
                /* wipe player action: every unit of the wiped player is disbanded, refund included */
                pc_military_disband_of(s, i, r, oi);
            }
        }
    }
    /* two seats: the reference's end at the first elimination; more: when one seat is left */
    s->terminated = (uint8_t)(s->terminated || (eliminated && (s->np == 2 || pc_alive_count(s) <= 1)));
    uint8_t claimed[PC_MAXN];
    for (int t = 0; t < n; t++) {
        claimed[t] = enemy ? (uint8_t)(membership[t] || t == source) : (uint8_t)(area[t] && s->tiles.owner[t] == 0);
    }
    pc_economy_claim(s, source, claimed, r);
    pc_structures_adjust_building_population(before, s, r, g);
    pc_economy_discover(s, enemy ? city_area : area, owner);
    /* capture city action: after the explore actions, Polaris freezes the city area and any
     * other player restores the climate there */
    pc_polaris_city_area_climate(s, source, owner);
    /* capture city action: the city tile becomes Field; Aquarion floods it, any other
     * capturer drains it */
    s->tiles.terrain[source] = 3;
    s->tiles.flooded[source] = (uint8_t)(s->players.tribe[pc_pclip(s, owner)] == PC_AQUARION);
    if (enemy) {
        /* capture city action: capturing a player's city counts as an attack on them */
        pc_diplomacy_hostile(s, old);
    }
    /* capture command queues update city connections action for [the
     * capturer, the tile's previous owner] under update routes action and capture city action: once the capture has run,
     * the routes are updated, then both players' connections */
    pc_structures_update_routes(s, r, sc);
    uint8_t hub_loss = 0;
    if (was_connected) {
        /* update city connections for a city connected to its old owner's capital: the old owner's start
         * tile loses its +1; if the new owner's network reaches the city, their start tile gains it and the city keeps
         * its own (change city connection action), else the city loses its +1 (disconnect city action) */
        int old_cap;
        uint8_t old_holds;
        pc_structures_capital_index(s, old, &old_cap, &old_holds);
        if (old_holds) {
            int32_t v[PC_MAXN];
            for (int i = 0; i < n; i++) v[i] = 0;
            v[old_cap] = -1;
            pc_economy_add_population_vector(s, v, old, 1);
        }
        uint8_t reached[PC_MAXN];
        int cap;
        uint8_t holds;
        pc_structures_reachable(s, owner, g, 1, reached, &cap, &holds);
        if (holds && reached[source] && source != cap) {
            int32_t v[PC_MAXN];
            for (int i = 0; i < n; i++) v[i] = 0;
            v[cap] = 1;
            pc_economy_add_population_vector(s, v, owner, 1);
        } else {
            /* a recaptured start tile is the network's hub, not a connected city: it disconnects after the new
             * owner's cities reconnect to it (recorded games: the hub levels up on the way) */
            s->cities.connected[source] = 0;
            hub_loss = 1;
        }
    }
    pc_structures_connections(s, owner, g, 1);
    if (hub_loss) {
        int32_t v[PC_MAXN];
        for (int i = 0; i < s->n; i++) v[i] = 0;
        v[source] = -1;
        pc_economy_add_population_vector(s, v, owner, 1);
    }
    if (enemy) {
        pc_structures_connections(s, old, g, 1);
    }
    pc_military_make_exhausted(s, source);
    s->units.invisible[source] = 0;
    for (int i = 0; i < n; i++) {
        if (s->units.owner[i] != owner && s->units.home[i] == (int32_t)source + 1) {
            s->units.home[i] = 0;
        }
    }
    int k = s->units.kind[source];
    s->units.home[source] = r->skills[k][PC_SK_INDEPENDENT] ? 0 : (int32_t)source + 1;
}

/* build action(star fishing): a Manual improvement, so the unit is only marked moved and attacked (double ready and
 * Boosted stay, unlike make exhausted) */
PC_FN void pc_capture_starfish(PcState *s, int source) {
    s->units.moved[source] = 1;
    s->units.attacked[source] = 1;
    s->units.move_lock[source] = 0;
    s->units.invisible[source] = 0;
    s->tiles.resource[source] = 0;
    int p = pc_p(s);
    s->players.stars[p] += 8;
}

/* get unlockable tect default: walking get all tech for tribe (the tribe's tree, Rules:tech_tree_of), the
 * unknown `next` techs of every known tech, then any unknown tech with a known `next` tech, each once, in that order.
 * Rust `Vec<usize>` -> `out[128]` (each tech at most once) in the same order; returns the count. */
PC_FN int pc_capture_unlockable_tech(const PcRules *r, int32_t tribe, const uint8_t *tech, int32_t out[128]) {
    int count = 0;
    int32_t tree_n;
    const int32_t *tree_tech, *tree_off, *tree_next;
    pc_rules_tech_tree_of(r, tribe, &tree_n, &tree_tech, &tree_off, &tree_next);
    pc_u128 exists = pc_rules_tech_exists_of(r, tribe);
    for (int e = 0; e < tree_n; e++) {
        int32_t k = tree_tech[e];
        int32_t lo = tree_off[e], hi = tree_off[e + 1];
        if (tech[k]) {
            for (int32_t j = lo; j < hi; j++) {
                int32_t c = tree_next[j];
                if (pc_u128_test(exists, c) && !tech[c]) {
                    int present = 0;
                    for (int m = 0; m < count; m++) if (out[m] == c) { present = 1; break; }
                    if (!present) out[count++] = c;
                }
            }
        } else {
            int any_known = 0;
            for (int32_t j = lo; j < hi; j++) if (tech[tree_next[j]]) { any_known = 1; break; }
            if (any_known) {
                int present = 0;
                for (int m = 0; m < count; m++) if (out[m] == k) { present = 1; break; }
                if (!present) out[count++] = k;
            }
        }
    }
    return count;
}

/* ruins reward */
#define PC_RUIN_RESOURCES 1u
#define PC_RUIN_POPULATION 2u
#define PC_RUIN_EXPLORER 3u
#define PC_RUIN_FREE_TECH 4u
#define PC_RUIN_BATTLESHIP 6u
#define PC_RUIN_SWORDSMAN 8u
#define PC_RUIN_CITY 10u

/* capture.rs:lost_city_site (valid lost city location): not on the map edge, an unowned Ocean
 * tile, no city (village) or ruin within distance 2 besides the examined ruin. */
PC_FN uint8_t pc_capture_lost_city_site(const PcState *s, int source, const PcGeom *g) {
    int size = s->size, x = source % size, y = source / size;
    if (x == 0 || y == 0 || x + 1 >= size || y + 1 >= size || s->tiles.owner[source] != 0 || s->tiles.terrain[source] != 2) return 0;
    int x0, x1, y0, y1;
    pc_ball(g, source, 2, &x0, &x1, &y0, &y1);
    for (int yy = y0; yy <= y1; yy++) {
        for (int xx = x0; xx <= x1; xx++) {
            int t = yy * size + xx;
            if (t != source && (s->tiles.structure[t] == 1 || s->tiles.structure[t] == 2)) return 0;
        }
    }
    return 1;
}

/* capture.rs:continent_resource_mod: get modifier(resource) of a `tribe` continent (ctor defaults,
 * set player's aquacrop 0.3, the tribe's resource modifier; a positive default is scaled, a zero one replaced). */
PC_FN float pc_capture_continent_resource_mod(int32_t tribe, int resource) {
    const float base[10] = {0.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.8f, 0.5f, 0.0f, 0.0f, 0.3f};
    int k[4] = {-1, -1, -1, -1};
    float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    switch (tribe) {
    case 2: k[0] = 2; f[0] = 0.1f; break;
    case 3: k[0] = 2; f[0] = 0.5f; k[1] = 9; f[1] = 1.5f; break;
    case 4: k[0] = 2; f[0] = 0.0f; break;
    case 5: k[0] = 2; f[0] = 1.5f; break;
    case 7: k[0] = 6; f[0] = 2.0f; k[1] = 1; f[1] = 0.5f; break;
    case 8: k[0] = 3; f[0] = 1.5f; break;
    case 10: k[0] = 1; f[0] = 0.2f; break;
    case 11: k[0] = 2; f[0] = 0.1f; k[1] = 6; f[1] = 2.0f; break;
    case 12: k[0] = 6; f[0] = 0.1f; k[1] = 3; f[1] = 0.1f; k[2] = 1; f[2] = 0.1f; k[3] = 5; f[3] = 2.0f; break;
    case 13: k[0] = 5; f[0] = 1.5f; break;
    case 14: k[0] = 6; f[0] = 1.5f; break;
    case 15: k[0] = 6; f[0] = 0.5f; break;
    case 17: k[0] = 7; f[0] = 0.3f; k[1] = 2; f[1] = 0.0f; break;
    default: break;
    }
    float v = base[resource];
    for (int q = 0; q < 4; q++) {
        if (k[q] == resource) v = 0.0f < v ? v * f[q] : f[q];
    }
    return v;
}

/* capture.rs:pick_resource (add resource's pick with random hash). */
PC_FN int32_t pc_capture_pick_resource(uint32_t seed, int32_t x, int32_t y, int32_t terrain, int32_t continent) {
    int32_t cand[10];
    int found = 0;
    for (int i = 0; i < 10; i++) {
        float m = pc_capture_continent_resource_mod(continent, i);
        uint8_t fits;
        switch (i) {
        case 1: fits = terrain == 5; break;
        case 2: case 6: fits = terrain == 3; break;
        case 3: case 8: case 9: fits = terrain == 1 || terrain == 2; break;
        case 4: fits = terrain == 2; break;
        case 5: fits = terrain == 4; break;
        case 7: fits = terrain == 3 || terrain == 5; break;
        default: fits = 0; break;
        }
        if (m != 0.0f && pc_hash_value3(seed, x, y, i) <= m && fits) cand[found++] = i;
    }
    return found == 0 ? 0 : cand[pc_hash_range3(seed, 0, found, x, y, 10)];
}

/* capture.rs:lost_city (examine ruins action's City reward). */
PC_FN void pc_capture_lost_city(PcState *s, int source, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int size = s->size, x = source % size, y = source / size;
    const int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};
    for (int q = 0; q < 4; q++) {
        int nx = x + dx[q], ny = y + dy[q];
        if (nx >= 0 && ny >= 0 && nx < size && ny < size) {
            int t = ny * size + nx;
            if (s->tiles.terrain[t] != 6) s->tiles.terrain[t] = 1;
        }
    }
    s->tiles.structure[source] = 1;
    s->tiles.terrain[source] = 3;
    s->tiles.flooded[source] = 1;
    uint32_t seed = s->initial_seed;
    int32_t tribe = s->players.tribe[pc_p(s)];
    int x0, x1, y0, y1;
    pc_ball(g, source, 2, &x0, &x1, &y0, &y1);
    for (int yy = y0; yy <= y1; yy++) {
        for (int xx = x0; xx <= x1; xx++) {
            int t = yy * size + xx;
            if (t == source) continue;
            uint8_t near = (xx - x <= 1 && x - xx <= 1) && (yy - y <= 1 && y - yy <= 1);
            uint32_t words[2];
            words[0] = (uint32_t)xx; words[1] = (uint32_t)yy;
            float value = (float)pc_hash_words(seed, words, 2) * 2.3283064e-10f;
            uint8_t free_ = s->tiles.resource[t] == 0 && s->tiles.structure[t] == 0 && s->tiles.owner[t] == 0;
            if (value <= (near ? 1.0f : 0.33333334f) && free_) {
                if (s->tiles.continent[t] == 0) s->tiles.continent[t] = tribe;
                s->tiles.resource[t] = pc_capture_pick_resource(seed, xx, yy, s->tiles.terrain[t], s->tiles.continent[t]);
            }
        }
    }
    pc_capture_city(s, source, r, g, sc);
}

/* get ruins reward default: [Resources, population growth, Swordsman, Battleship, Explorer, free tech]
 * shuffled by random hash range(i, count, x, y, i) (Fisher-Yates); the first valid entry wins, so the reward is a
 * fixed function of the tile, the game seed and the examining player's position. */
PC_FN uint32_t pc_capture_ruin_reward(const PcState *s, int source, const PcRules *r, const PcGeom *g) {
    int p = pc_p(s);
    const uint8_t *tech = s->players.tech[p];
    uint32_t x = (uint32_t)(source % s->size), y = (uint32_t)(source / s->size);
    /* version >= 120: a Flooded-tribe (Aquarion) player on a valid lost-city location gets a city */
    if (s->players.tribe[p] == PC_AQUARION && pc_capture_lost_city_site(s, source, g)) return PC_RUIN_CITY;
    uint32_t list[6] = {PC_RUIN_RESOURCES, PC_RUIN_POPULATION, PC_RUIN_SWORDSMAN, PC_RUIN_BATTLESHIP, PC_RUIN_EXPLORER, PC_RUIN_FREE_TECH};
    for (int i = 0; i < 6 - 1; i++) {
        uint32_t words[3];
        words[0] = x; words[1] = y; words[2] = (uint32_t)i;
        int j = (int)pc_hash_range(s->initial_seed, (uint32_t)i, 6u, words, 3);
        uint32_t tmp = list[i]; list[i] = list[j]; list[j] = tmp;
    }
    uint8_t water = s->tiles.terrain[source] == 1 || s->tiles.terrain[source] == 2;
    for (int li = 0; li < 6; li++) {
        uint32_t reward = list[li];
        uint8_t valid;
        switch (reward) {
            /* a neighbour the player can enter (can be accessed by player) and an unexplored tile within 2 */
            case PC_RUIN_EXPLORER: {
                uint8_t enter = 0;
                int x0, x1, y0, y1;
                pc_ball(g, source, 1, &x0, &x1, &y0, &y1);
                for (int yy = y0; yy <= y1; yy++) {
                    for (int xx = x0; xx <= x1; xx++) {
                        int t = yy * s->size + xx;
                        enter = (uint8_t)(enter || (t != source && (pc_discovery_movable(tech, s->tiles.terrain[t]) || s->tiles.structure[t] == 48)));
                    }
                }
                uint8_t fog = 0;
                pc_ball(g, source, 2, &x0, &x1, &y0, &y1);
                for (int yy = y0; yy <= y1; yy++) {
                    for (int xx = x0; xx <= x1; xx++) {
                        int t = yy * s->size + xx;
                        fog = (uint8_t)(fog || (t != source && !s->tiles.explored[p][t]));
                    }
                }
                valid = enter && fog;
                break;
            }
            case PC_RUIN_FREE_TECH: {
                int32_t techs[128];
                valid = pc_capture_unlockable_tech(r, s->players.tribe[p], tech, techs) > 0;
                break;
            }
            /* has unlocked movement on terrain(Ocean): Sailing (Aquarion: marine life) */
            case PC_RUIN_BATTLESHIP:
                valid = water && pc_kernels_ocean_move(tech);
                break;
            /* another living player: the game's nature player (id 255) always counts as one */
            case PC_RUIN_SWORDSMAN:
                valid = !water;
                break;
            default:
                valid = 1;
                break;
        }
        if (valid) {
            return reward;
        }
    }
    return PC_RUIN_RESOURCES;
}

/* examine ruins command + examine ruins action. */
PC_FN void pc_capture_ruin(PcState *s, int source, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int p = pc_p(s);
    int32_t cp = s->current_player;
    uint32_t reward = pc_capture_ruin_reward(s, source, r, g);
    pc_military_make_exhausted(s, source);
    s->units.invisible[source] = 0;
    s->tiles.structure[source] = 0;
    s->tiles.structure_level[source] = 0;
    s->tiles.structure_turn[source] = 0;
    uint32_t x = (uint32_t)(source % s->size), y = (uint32_t)(source / s->size);
    switch (reward) {
        case PC_RUIN_CITY:
            pc_capture_lost_city(s, source, r, g, sc);
            break;
        case PC_RUIN_RESOURCES:
            s->players.stars[p] += 10;
            break;
        case PC_RUIN_FREE_TECH: {
            /* a research action at no cost for list[random hash range(0, count, x, y)] */
            int32_t list[128];
            int len = pc_capture_unlockable_tech(r, s->players.tribe[p], s->players.tech[p], list);
            uint32_t words[2];
            words[0] = x; words[1] = y;
            int kind = (int)list[pc_hash_range(s->initial_seed, 0u, (uint32_t)len, words, 2)];
            s->players.tech[p][kind] = 1;
            s->players.tech_turn[p][kind] = s->turn;
            s->players.score[p] = pc_wadd(s->players.score[p], 100 * r->tech_tier[kind]);
            break;
        }
        case PC_RUIN_POPULATION: {
            /* three increase population actions on get current capital coordinates (the start tile, else the first own
             * city in tile order); nothing without a city */
            int cap = pc_structures_current_capital(s, cp);
            if (cap >= 0) {
                pc_economy_add_population(s, cap, 3);
            }
            break;
        }
        case PC_RUIN_EXPLORER: {
            uint8_t reveal[PC_MAXN];
            pc_discovery_explorer(s, source, g, reveal);
            pc_economy_discover(s, reveal, cp);
            break;
        }
        default: {
            /* train unit on occupied space (the examining unit is pushed off first), then promote action: a veteran unit 44
             * (our Rammership carrying a Warrior) on water, a veteran Swordsman on land; no home (not a city tile).
             * train unit on occupied space trains the tribe's override (Aquarion: a Shark, which carries nothing) */
            uint8_t water = reward == PC_RUIN_BATTLESHIP;
            int32_t kind = pc_tribes_unit_override(s->players.tribe[p], water ? 44 : 11);
            pc_military_push(s, source, r, g);
            pc_military_summon(s, source, (int)kind, r, g, 0);
            pc_military_promote(s, source, r);
            uint8_t carry = r->skills[kind][PC_SK_CARRY];
            s->units.passenger[source] = carry ? 2 : 0;
            s->units.home[source] = 0;
            /* train unit gives a vehicle a Warrior passenger, and train action scores get unit score = 5 x get full cost,
             * passenger included */
            if (carry) s->players.score[p] = pc_wadd(s->players.score[p], 5 * r->unit_cost[2]);
            break;
        }
    }
}

#ifdef __cplusplus
}
#endif
#endif /* PC_CAPTURE_H */
