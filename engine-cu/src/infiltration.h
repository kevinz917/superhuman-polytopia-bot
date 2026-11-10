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

/* Spawn tiles for `target`'s city in priority order (eligible only): InfiltrationRewardAction.ExecuteDefault's
 * Evaluate (InfiltrationRewardAction.c:8306-8403): tie noise + city +2, water -0.1, a resource the player sees +0.5,
 * defence bonus +1, a Poison improvement -0.5. Sorted best first (the comparison b__1, c:8406-8423).
 * Rust `Vec<usize>` -> `out[PC_MAXN]`, returns the count. */
PC_FN int pc_infiltration_spawn_order(const PcState *s, int target, const PcGeom *g, int32_t out[PC_MAXN]) {
    int n = s->n, size = s->size;
    const PcTiles *t = &s->tiles;
    const uint8_t *tech = s->players.tech[pc_p(s)];
    float key[PC_MAXN];
    int32_t idx[PC_MAXN];
    int count = 0;
    for (int i = 0; i < n; i++) {
        uint8_t water = t->terrain[i] <= 2;
        uint8_t eligible = t->city[i] == (int32_t)target && s->units.owner[i] == 0;
        eligible = (uint8_t)(eligible && (t->terrain[i] != 4 || tech[20]) && (t->terrain[i] != 1 || tech[10]) && (t->terrain[i] != 2 || tech[13]));
        /* PathFinder.IsTileAccessible: only tiles the infiltrator's owner has explored */
        eligible = (uint8_t)(eligible && t->explored[pc_p(s)][i]);
        if (!eligible) {
            continue;
        }
        uint8_t defense = (t->terrain[i] == 4 && tech[20]) || (t->terrain[i] == 5 && tech[18]) || (water && tech[12]);
        /* TileData.GetResource (TileData.c:1650-1656): no resource on a tile with an improvement (a farm, a mine, the
         * city), whatever lies under it */
        uint8_t known_resource = t->resource[i] > 0 && t->structure[i] == 0 && (t->resource[i] != 2 || tech[6] || tech[8] || tech[9])
            && (t->resource[i] != 5 || tech[20] || tech[23] || tech[24]);
        /* an improvement with ImprovementAbility Poison (Fungi 37, Clathrus 41) for a tribe without PoisonResist
         * (Bardur): -0.5 (c:8371-8394) */
        uint8_t poison = t->structure[i] == 37 || t->structure[i] == 41;
        /* c:8335-8342: GameState.RandomHash.GetHash(x, y, CurrentTurn) (vtable slot 0x1A8 = HashFunction.GetHash(int,
         * int, int), ISIL: `mov esi,[r14+10h]; mov edx,[r14+14h]; mov ecx,[rcx+1Ch]` = x, y, GameState+0x1C
         * CurrentTurn), then HashFunction.Range(0f, 0.1f, that hash) (0x6E8A30, HashFunction.c:260-270):
         * 0.1 * XXHash.GetHash(int) * 2^-32. Our `turn` is CurrentTurn (tools/il2cpp-host/to_polyfish.py). */
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
    /* InfiltrationRewardAction: TileDataExtensions.CalculateRawProduction, no negative-progress term */
    int32_t innate = pc_max(c->production[target] + (s->tiles.capital[target] == c->owner[target] ? 1 : 0), 0);
    int count = (int)pc_min(c->level[target], 5);
    pc_military_remove(s, source, r, -1, 1);
    uint8_t enemy = s->units.owner[target] > 0 && s->units.owner[target] != owner;
    int tk = s->units.kind[target];
    float damage, retaliation, half;
    pc_kernels_combat(r->unit_attack[37], r->unit_health[37], r->unit_health[37], r->unit_defense[tk], s->units.health[target],
                      pc_kernels_max_health(s, r, target), pc_military_defense_bonus(s, r, target), &damage, &retaliation, &half);
    s->units.health[target] += enemy ? -damage : 0.0f;
    uint8_t dead = enemy && s->units.health[target] <= 0.0f;
    if (dead) {
        pc_military_remove(s, target, r, -1, 1);
    } else if (enemy) {
        /* ActionUtils.PerformAttack(state, player, WorldCoordinates.NULL, city, damage) (c:7994-8004): no unit on the
         * null origin (-1, -1), and the city's unit turns toward it (ActionUtils.PerformAttackDefault, ActionUtils.c:4346) */
        int32_t size = s->size;
        s->units.direction[target] = pc_military_direction_xy((int32_t)target % size, (int32_t)target / size, -1, -1);
    }
    s->players.kills[po] += dead ? 1 : 0;
    s->players.stars[po] += innate;
    s->cities.infiltrated[target] = s->turn * s->np + owner;
    s->cities.riot[target] = 1;
    int32_t order[PC_MAXN];
    int len = pc_infiltration_spawn_order(s, target, g, order);
    for (int i = 0; i < len; i++) {
        int at = (int)order[i];
        if (i >= pc_min(count, 5)) {
            break;
        }
        uint8_t water = s->tiles.terrain[at] <= 2;
        pc_military_summon(s, at, water ? 40 : 37, r, g, 0);
        s->units.home[at] = 0;
        /* TrainAction(player, Dagger, tile, 0, LookCoordinates = the city) (c:7965): TrainAction.ExecuteDefault turns
         * the new unit toward LookCoordinates (TrainAction.c:651-660; the same tile gives 8, none); on water the
         * queued EmbarkAction (c:7936) copies the facing to the Pirate (EmbarkAction.c:96-97) */
        s->units.direction[at] = pc_military_direction(s->size, at, target);
    }
}

#ifdef __cplusplus
}
#endif
#endif /* PC_INFILTRATION_H */
