/* economy.h - line-by-line port of engine-rs/src/economy.rs (gpu/economy.py: population, territory, harvest/forest
 * actions, discovery and city rewards). Every `pub fn f` of the Rust is `pc_economy_f` with the same argument order;
 * Vec<bool> / mask results are `uint8_t out[]` parameters (forest_masks: out[t*3 + k], reward_mask: out[t*9 + k]).
 *
 * Includes structures.h (it calls current_capital, score_of, update_routes, connections); structures.h declares the
 * economy prototypes it needs, so the include is one-directional. `reward` takes the PcStructScratch that
 * update_routes needs (kind 5, Border Growth). */
#ifndef PC_ECONOMY_H
#define PC_ECONOMY_H

#include "pc.h"
#include "structures.h"

#ifdef __cplusplus
extern "C" {
#endif

/* provided by discovery.h (discovery.rs explorer: Vec<bool> -> out[PC_MAXN]) */
PC_FN void pc_discovery_explorer(const PcState *s, int source, const PcGeom *g, uint8_t out[PC_MAXN]);
/* provided by military.h (military.rs superunit) */
PC_FN void pc_military_superunit(PcState *s, int source, const PcRules *r, const PcGeom *g);

/* ScoreExtensions.CalculateImprovementScore of city `i`: 50 per level + 50, 5 per point of progress (xp), the Park's
 * 250; with negative progress (version > 114) 100 more per point, floored at 0. */
PC_FN int32_t pc_economy_city_score(const PcState *s, int i) {
    const PcCities *c = &s->cities;
    int32_t v = 5 * c->progress[i] + 50 * (c->level[i] + 1) + 250 * pc_reward(s, i, 2);
    if (c->progress[i] < 0) return pc_max(v + 100 * c->progress[i], 0);
    return v;
}

/* ImprovementExtensions.LevelUp while ShouldLevelUp (CityLevelUpAction): progress pays for the next level. */
PC_FN void pc_economy_level_up(PcState *s, int i) {
    PcCities *c = &s->cities;
    while (c->progress[i] >= c->level[i] + 1) {
        c->progress[i] -= c->level[i] + 1;
        c->level[i] += 1;
        c->production[i] += 1;
        if (s->tiles.structure[i] > 0) {
            s->tiles.structure_level[i] += 1;
        }
    }
}

/* IncreasePopulationAction per city of `owner`: progress and score (the change of city_score). A city levels up
 * only on its owner's turn; otherwise at the owner's next turn start (StartTurnAction, start_level_ups). Cities
 * never lose levels (negative progress stays). */
PC_FN void pc_economy_add_population_vector(PcState *s, const int32_t amount[PC_MAXN], int32_t owner, uint8_t native) {
    int n = s->n;
    int32_t score = 0;
    (void)native;
    for (int i = 0; i < n; i++) {
        if (s->cities.owner[i] != owner || amount[i] == 0) {
            continue;
        }
        int32_t before = pc_economy_city_score(s, i);
        s->cities.progress[i] += amount[i];
        s->cities.population[i] += amount[i];
        if (owner == s->current_player) {
            pc_economy_level_up(s, i);
        }
        score += pc_economy_city_score(s, i) - before;
    }
    int o = owner - 1;
    s->players.score[o] = pc_wadd(s->players.score[o], score);
}

PC_FN void pc_economy_add_population(PcState *s, int index, int32_t amount) {
    int n = s->n;
    int32_t cp = s->current_player;
    int32_t v[PC_MAXN];
    for (int i = 0; i < n; i++) v[i] = 0;
    if (index < n && s->cities.owner[index] == cp) {
        v[index] = amount;
    }
    pc_economy_add_population_vector(s, v, cp, 0);
}

/* StartTurnAction.ExecuteDefault: the mover's cities that gained enough progress off-turn level up now (after
 * the turn's income, which reads the old level). */
PC_FN void pc_economy_start_level_ups(PcState *s, int32_t owner) {
    int o = owner - 1;
    for (int i = 0; i < s->n; i++) {
        if (s->cities.owner[i] == owner && s->cities.progress[i] >= s->cities.level[i] + 1) {
            int32_t before = pc_economy_city_score(s, i);
            pc_economy_level_up(s, i);
            s->players.score[o] = pc_wadd(s->players.score[o], pc_economy_city_score(s, i) - before);
        }
    }
}

PC_FN void pc_economy_in_territory(const PcState *s, uint8_t out[PC_MAXN]) {
    int n = s->n;
    for (int t = 0; t < n; t++) out[t] = 0;
    for (int c = 0; c < n; c++) {
        if (s->cities.owner[c] == s->current_player) {
            for (int t = 0; t < n; t++) {
                out[t] |= (uint8_t)pc_territory(s, c, t);
            }
        }
    }
}

/* No unit blocks work on tile `t`: none, the mover's, a hidden one, or (GameLogicData.CanBuild /
 * TileData.CanDestroy) a unit of a player the mover has peace with. */
PC_FN uint8_t pc_economy_unit_allows_work(const PcState *s, int t) {
    int32_t o = s->units.owner[t];
    return o == 0 || o == s->current_player || s->units.invisible[t] || s->players.peace[pc_p(s)][pc_pclip(s, o)];
}

PC_FN void pc_economy_economic_tiles(const PcState *s, uint8_t out[PC_MAXN]) {
    pc_economy_in_territory(s, out);
    for (int t = 0; t < s->n; t++) {
        out[t] = out[t] && pc_economy_unit_allows_work(s, t) && s->tiles.owner[t] == s->current_player;
    }
}

/* harvest_mask given economic_tiles(s). */
PC_FN void pc_economy_harvest_mask_with(const PcState *s, const uint8_t *econ, uint8_t out[PC_MAXN]) {
    int n = s->n;
    const uint8_t *tech = s->players.tech[pc_p(s)];
    int32_t stars = s->players.stars[pc_p(s)];
    for (int t = 0; t < n; t++) {
        int32_t r = s->tiles.resource[t];
        uint8_t unlocked = (r == 1 && tech[15]) || (r == 3 && tech[10]) || (r == 6 && tech[6]);
        out[t] = econ[t] && s->tiles.structure[t] == 0 && unlocked && stars >= 2;
    }
}

PC_FN void pc_economy_harvest_mask(const PcState *s, uint8_t out[PC_MAXN]) {
    uint8_t econ[PC_MAXN];
    pc_economy_economic_tiles(s, econ);
    pc_economy_harvest_mask_with(s, econ, out);
}

PC_FN void pc_economy_harvest(PcState *s, int target) {
    int city = pc_clamp(s->tiles.city[target], 0, s->n - 1);
    s->tiles.resource[target] = 0;
    int p = pc_p(s);
    s->players.stars[p] -= 2;
    pc_economy_add_population(s, city, 1);
}

/* forest_masks given economic_tiles(s); out[t*3 + k] */
PC_FN void pc_economy_forest_masks_with(const PcState *s, const uint8_t *econ, uint8_t *out) {
    int n = s->n;
    const uint8_t *tech = s->players.tech[pc_p(s)];
    int32_t stars = s->players.stars[pc_p(s)];
    for (int t = 0; t < n; t++) {
        uint8_t free_ = econ[t] && (s->tiles.structure[t] == 0 || s->tiles.structure[t] == 3);
        uint8_t forest = s->tiles.terrain[t] == 5;
        uint8_t field = s->tiles.terrain[t] == 3;
        out[t * 3] = free_ && forest && tech[9] && stars >= 3;
        out[t * 3 + 1] = free_ && forest && tech[16];
        out[t * 3 + 2] = free_ && field && tech[19] && stars >= 5;
    }
}

/* out[t*3 + k] */
PC_FN void pc_economy_forest_masks(const PcState *s, uint8_t *out) {
    uint8_t econ[PC_MAXN];
    pc_economy_economic_tiles(s, econ);
    pc_economy_forest_masks_with(s, econ, out);
}

PC_FN void pc_economy_forest(PcState *s, int target, int32_t ability) {
    int32_t terrain = ability == 3 ? 5 : 3;
    /* BuildAction: a `creates` terrain clears the resource (Consumed improvements destroy it), so Grow
     * Forest leaves no resource behind */
    int32_t resource = ability == 1 ? 2 : 0;
    int32_t stars = ability == 1 ? -3 : (ability == 2 ? 1 : -5);
    s->tiles.terrain[target] = terrain;
    s->tiles.resource[target] = resource;
    int p = pc_p(s);
    s->players.stars[p] += stars;
}

/* economy.discover(s, reveal, player): exploration gain for `player`; lighthouse population to the
 * preferred city of `player` (applied only if that city belongs to the player to move). */
PC_FN void pc_economy_discover(PcState *s, const uint8_t *reveal, int32_t player) {
    int n = s->n;
    int p = player - 1;
    int32_t new_tiles = 0;
    int32_t count = 0;
    for (int t = 0; t < n; t++) {
        if (reveal[t] && !s->tiles.explored[p][t]) {
            new_tiles += 1;
            if (s->tiles.structure[t] == 47) {
                count += 1;
            }
            s->tiles.explored[p][t] = 1;
        }
    }
    s->players.score[p] = pc_wadd(s->players.score[p], new_tiles * 5);
    if (count == 0) {
        return; /* add_population(.., 0) below changes nothing */
    }
    /* ActionUtils.OnRevealLighthouseTile: +1 to GetCurrentCapitalCoordinates, nothing without a city */
    int index = pc_structures_current_capital(s, player);
    if (index >= 0) {
        pc_economy_add_population(s, index, count);
    }
}

/* out[t*9 + k] */
PC_FN void pc_economy_reward_mask(const PcState *s, uint8_t *out) {
    int n = s->n;
    for (int t = 0; t < n; t++) {
        int32_t count = pc_reward_count(s, t);
        uint8_t pending = s->cities.owner[t] == s->current_player && count < s->cities.level[t] - 1;
        /* the game stops at each level-up's reward trigger before the next population point (recorded games:
         * a city that gains two levels is offered the next level's pair, then the one after) */
        int32_t slot = count + 1;
        for (int32_t k = 0; k < 9; k++) {
            uint8_t allowed;
            switch (slot) {
            case 1: allowed = k == 3 || k == 4; break;
            case 2: allowed = k == 1 || k == 7; break;
            case 3: allowed = k == 5 || k == 8; break;
            default: allowed = k == 2 || k == 6; break;
            }
            out[t * 9 + k] = pending && allowed;
        }
    }
}

PC_FN void pc_economy_claim(PcState *s, int city, const uint8_t *mask, const PcRules *r) {
    int n = s->n;
    int32_t cp = s->current_player;
    int32_t gain = 0;
    int32_t loss[PC_MAXP];
    for (int k = 0; k < PC_MAXP; k++) loss[k] = 0;
    for (int t = 0; t < n; t++) {
        if (mask[t] && s->tiles.owner[t] != cp) {
            int32_t value = 20 + pc_structures_score_of(s, r, t);
            gain += value;
            if (pc_foreign(s->tiles.owner[t], cp)) {
                loss[s->tiles.owner[t] - 1] += value;
            }
        }
    }
    for (int t = 0; t < n; t++) {
        if (mask[t]) {
            s->tiles.owner[t] = cp;
            s->tiles.city[t] = (int32_t)city;
            pc_set_territory(s, city, t, 1);
        }
    }
    int p = pc_p(s);
    s->players.score[p] = pc_wadd(s->players.score[p], gain);
    /* with two seats exactly the reference's single subtraction from the opponent (also when its loss is 0) */
    for (int q = 0; q < s->np; q++) {
        if (q != p) s->players.score[q] = pc_wsub(s->players.score[q], loss[q]);
    }
}

PC_FN void pc_economy_reward(PcState *s, int target, int32_t kind, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int n = s->n;
    s->cities.rewards[target * 9 + kind] += 1;
    switch (kind) {
    case 2:
    case 3: {
        s->cities.production[target] += 1;
        if (kind == 2) {
            int p = pc_p(s);
            s->players.score[p] += 250;
        }
        break;
    }
    case 7: {
        int p = pc_p(s);
        s->players.stars[p] += 5;
        break;
    }
    case 8:
        pc_economy_add_population(s, target, 3);
        break;
    case 5: {
        uint8_t area[PC_MAXN];
        uint8_t claimed[PC_MAXN];
        for (int t = 0; t < n; t++) {
            area[t] = pc_d(g, target, t) <= 2 && t != target;
        }
        s->cities.border[target] += 1;
        for (int t = 0; t < n; t++) claimed[t] = area[t] && s->tiles.owner[t] == 0;
        pc_economy_claim(s, target, claimed, r);
        pc_economy_discover(s, area, s->current_player);
        /* CityRewardAction (BorderGrowth, CityRewardAction.c 240-320): ExpandCityAction, UpdateRoutesAction, then
         * UpdateCityConnectionsAction for every other player (the grower's own come with the deferred update) */
        pc_structures_update_routes(s, r, sc);
        int32_t cp = s->current_player;
        for (int32_t q = 1; q <= s->np; q++) {
            if (q != cp) pc_structures_connections(s, q, g, 1);
        }
        break;
    }
    case 4: {
        uint8_t reveal[PC_MAXN];
        pc_discovery_explorer(s, target, g, reveal);
        pc_economy_discover(s, reveal, s->current_player);
        break;
    }
    case 6:
        pc_military_superunit(s, target, r, g);
        break;
    default:
        break;
    }
}

#ifdef __cplusplus
}
#endif
#endif /* PC_ECONOMY_H */
