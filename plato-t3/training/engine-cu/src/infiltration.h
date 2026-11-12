/* infiltration.h - line-by-line port of engine-rs/src/infiltration.rs (gpu/infiltration.py: native Cloak
 * infiltration). Every `pub` item is `pc_infiltration_<name>`; `Vec<usize>` -> out list + count. */
#ifndef PC_INFILTRATION_H
#define PC_INFILTRATION_H

#include "pc.h"
#include "military.h"   /* defense_bonus, direction, direction_xy, remove, summon; hash.h include / stub;
                           pc_kernels_combat / pc_kernels_max_health prototypes */

#ifdef __cplusplus
extern "C" {
#endif

/* Spawn tiles for `target`'s city in priority order (eligible only): infiltration reward action's
 * Evaluate: tie noise + city +2, water -0.1, a resource the player sees +0.5,
 * defence bonus +1, a Poison improvement -0.5. Sorted best first.
 * Rust `Vec<usize>` -> `out[PC_MAXN]`, returns the count. */
PC_FN int pc_infiltration_spawn_order(const PcState *s, int target, const PcGeom *g, uint8_t bug, int32_t out[PC_MAXN]) {
    int n = s->n, size = s->size;
    const PcTiles *t = &s->tiles;
    const uint8_t *tech = s->players.tech[pc_p(s)];
    int32_t tribe = s->players.tribe[pc_p(s)];
    float key[PC_MAXN];
    int32_t idx[PC_MAXN];
    int count = 0;
    for (int i = 0; i < n; i++) {
        uint8_t water = t->terrain[i] <= 2;
        uint8_t eligible = t->city[i] == (int32_t)target && s->units.owner[i] == 0;
        /* a Bridge lets the Dagger stand on its water tile whatever the techs (infiltration.rs) */
        uint8_t bridge = (uint8_t)(t->structure[i] == 48);
        eligible = (uint8_t)(eligible && (t->terrain[i] != 4 || tech[20])
                             && (bridge || (bug ? (!water || t->algae[i])
                                                : ((t->terrain[i] != 1 || pc_kernels_water_move(tech))
                                                   && (t->terrain[i] != 2 || tech[13] || tech[46])))));
        /* is tile accessible: only tiles the infiltrator's owner has explored */
        eligible = (uint8_t)(eligible && t->explored[pc_p(s)][i]);
        if (!eligible) {
            continue;
        }
        uint8_t defense = (t->terrain[i] == 4 && tech[20]) || (t->terrain[i] == 5 && tech[18]) || (water && tech[12]);
        /* get resource: no resource on a tile with an improvement (a farm, a mine, the
         * city), whatever lies under it */
        uint8_t known_resource = t->structure[i] == 0 && pc_cymanti_resource_visible(tribe, tech, t->resource[i]);   /* (Cymanti) */
        /* an improvement with improvement ability Poison (Fungi 37, Clathrus 41) for a tribe without poison resist
         * (Bardur): -0.5 */
        uint8_t poison = (t->structure[i] == 37 || t->structure[i] == 41) && tribe != PC_CYMANTI;
        /* tie noise: the game's random hash of (x, y, current turn), an integer hash over the three
         * values in that order, mapped to the range [0, 0.1):
         * 0.1 * XXHash(int) * 2^-32.
         * Our `turn` is the game's current turn. */
        uint32_t words[3];
        words[0] = (uint32_t)(i % size); words[1] = (uint32_t)(i / size); words[2] = (uint32_t)s->turn;
        float noise = ((float)pc_hash_word(s->initial_seed, pc_hash_words(s->initial_seed, words, 3))) * 0.1f / 4294967296.0f;
        float priority = noise + (float)(int32_t)(t->structure[i] == 1) * 2.0f - (float)(int32_t)water * 0.1f + (float)(int32_t)known_resource * 0.5f
            + (float)(int32_t)defense - (float)(int32_t)poison * 0.5f;
        key[count] = -priority;
        idx[count] = (int32_t)i;
        count++;
    }
    /* stable argsort on the float key, ties by index (sort_by(partial_cmp(key).then(cmp(index)))): insertion sort */
    for (int a = 1; a < count; a++) {
        float ka = key[a];
        int32_t ia = idx[a];
        int b = a - 1;
        while (b >= 0 && (key[b] > ka || (key[b] == ka && idx[b] > ia))) {
            key[b + 1] = key[b];
            idx[b + 1] = idx[b];
            b--;
        }
        key[b + 1] = ka;
        idx[b + 1] = ia;
    }
    (void)g;
    for (int a = 0; a < count; a++) out[a] = idx[a];
    return count;
}

PC_FN void pc_infiltration_execute(PcState *s, int source, int target, const PcRules *r, const PcGeom *g) {
    int32_t owner = s->current_player;
    int po = pc_p(s);
    const PcCities *c = &s->cities;
    /* infiltration reward action: calculate raw production, no negative-progress term */
    int32_t innate = pc_max(c->production[target] + (s->tiles.capital[target] == c->owner[target] ? 1 : 0), 0);
    int count = (int)pc_min(c->level[target], 5);
    /* a Moth brings the Poison reward (Cymanti): no currency, the Moth's battle, a survivor Poisoned, bug eggs */
    uint8_t moth = s->units.kind[source] == 61;
    int fake = moth ? 61 : 37;
    pc_military_remove(s, source, r, -1, 1);
    uint8_t enemy = s->units.owner[target] > 0 && s->units.owner[target] != owner;
    int tk = s->units.kind[target];
    float damage, retaliation, half;
    pc_kernels_combat(r->unit_attack[fake], r->unit_health[fake], r->unit_health[fake], r->unit_defense[tk], s->units.health[target],
                      pc_kernels_max_health(s, r, target), pc_military_defense_bonus(s, r, target), &damage, &retaliation, &half);
    s->units.health[target] += enemy ? -damage : 0.0f;
    if (enemy) s->units.bubble[target] = 0;   /* perform attack on the city's unit takes its Bubble (Aquarion) */
    if (enemy) s->units.swift[target] = 0;    /* and its Swift (Cymanti) */
    uint8_t dead = enemy && s->units.health[target] <= 0.0f;
    if (dead) {
        pc_military_remove(s, target, r, -1, 1);
    } else if (enemy) {
        /* perform attack(state, player, world coordinates null, city, damage): no unit on the
         * null origin (-1, -1), and the city's unit turns toward it (perform attack default) */
        int32_t size = s->size;
        s->units.direction[target] = pc_military_direction_xy((int32_t)target % size, (int32_t)target / size, -1, -1);
        if (moth) s->units.poison[target] = 1;
    }
    s->players.kills[po] += dead ? 1 : 0;
    if (!moth) s->players.stars[po] += innate;
    s->cities.infiltrated[target] = s->turn * s->np + owner;
    s->cities.riot[target] = 1;
    int32_t order[PC_MAXN];
    int len = pc_infiltration_spawn_order(s, target, g, moth, order);
    /* infiltration reward action: the spawn is the tribe's override of the Dagger (Aquarion: the MermaidDagger, a Swim
     * unit that stands on water itself; infiltration.rs) */
    int dagger = (int)pc_tribes_unit_override(s->players.tribe[pc_p(s)], moth ? 60 : 37);
    for (int i = 0; i < len; i++) {
        int at = (int)order[i];
        if (i >= pc_min(count, 5)) {
            break;
        }
        /* not on a Bridge (infiltration.rs) */
        uint8_t water = !moth && s->tiles.terrain[at] <= 2 && s->tiles.structure[at] != 48 && !r->skills[dagger][PC_SK_FLOAT];
        pc_military_summon(s, at, water ? 40 : dagger, r, g, 0);
        s->units.home[at] = 0;
        /* train action(player, Dagger, tile, 0, look coordinates = the city): train action turns
         * the new unit toward look coordinates (the same tile gives 8, none); on water the
         * queued embark action copies the facing to the Pirate */
        s->units.direction[at] = pc_military_direction(s->size, at, target);
    }
}

#ifdef __cplusplus
}
#endif
#endif /* PC_INFILTRATION_H */
