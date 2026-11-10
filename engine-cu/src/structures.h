/* structures.h - line-by-line port of engine-rs/src/structures.rs (gpu/structures.py: buildings, monuments, temples,
 * road networks, income). Every `pub fn f` of the Rust is `pc_structures_f` with the same argument order; Vec<bool>
 * / Vec<i32> results are `out[PC_MAXN]` parameters, tuples are out-params in order, Option<usize> is -1.
 *
 * Scratch: the Rust allocates per call (RouterScratch, the level queue of refresh_levels). Those buffers are too
 * large for a device thread's stack, so the caller passes a `PcStructScratch` (one per game, in core's Scratch) to
 * update_routes, refresh_levels, build and destroy (and economy's reward, which calls update_routes). Everything
 * else lives on the stack (arrays of PC_MAXN ints at most).
 *
 * Cross-module calls are declared, not defined, under "provided by <module>.h". economy.h includes this header and
 * defines the economy prototypes below later in the translation unit (rule 9 of the design doc). */
#ifndef PC_STRUCTURES_H
#define PC_STRUCTURES_H

#include "pc.h"
#include "rules.h"   /* pc_bits128 (rules.rs bits128), pc_cost_le (CostIndex::le) */

#ifdef __cplusplus
extern "C" {
#endif

/* provided by diplomacy.h (diplomacy.rs at_war) */
PC_FN uint8_t pc_diplomacy_at_war(const PcState *s, int p, int q);

/* provided by economy.h (economy.rs); economy.h includes structures.h, so these are forward prototypes */
PC_FN void pc_economy_add_population(PcState *s, int index, int32_t amount);
PC_FN void pc_economy_add_population_vector(PcState *s, const int32_t amount[PC_MAXN], int32_t owner, uint8_t native);
PC_FN void pc_economy_economic_tiles(const PcState *s, uint8_t out[PC_MAXN]);
PC_FN void pc_economy_in_territory(const PcState *s, uint8_t out[PC_MAXN]);
PC_FN uint8_t pc_economy_unit_allows_work(const PcState *s, int t);

/* ---- scratch (RouterScratch + the refresh_levels queue) ---- */

/* pathfinding::OpenList as router_path uses it: a sorted sequence, pop takes the lowest priority, equal priorities in
 * insertion order. Entries are consumed from `head` (VecDeque::pop_front).
 *
 * Capacity (2026-11-06), the smaller of two bounds on the pushes of one router_path:
 *   (a) 1 + 8 x 1000: the start, then at most 8 per pop of the 1000-pop loop;
 *   (b) PC_ROUTER_OPEN_PER_TILE x n + 1: a tile is pushed when unvisited (cost 0) or when its cost strictly improves,
 *       so at most once per distinct cost it can take. A step costs 5 or 10 (pc_structures_rp_cost) and a tile is
 *       expanded only while its cost < range x 10 <= 50 (the largest router range is 5, pc_structures_route_data), so a
 *       pushed cost is a multiple of 5 in [5, 55]: 11 values per tile, plus the origin's first push.
 * 12 x 324 + 1 = 3889 entries (23 KB at 6 bytes) against the 8001 x 6 = 48 KB of the first port; 2353 (14 KB) in the
 * PC_MAXSIZE=14 build. The push compacts the live entries to the front rather than write past the arrays (invisible to
 * the semantics), so a bound miss could only cost time. (pathfinding.h has its own copy of OpenList for the unit
 * search; the semantics are identical.) */
#define PC_ROUTER_OPEN_PER_TILE 12
#define PC_ROUTER_OPEN_BOUND (PC_ROUTER_OPEN_PER_TILE * PC_MAXN + 1)
#define PC_ROUTER_OPEN_CAP (PC_ROUTER_OPEN_BOUND < 8 * 1000 + 1 ? PC_ROUTER_OPEN_BOUND : 8 * 1000 + 1)
typedef struct {
    float prio[PC_ROUTER_OPEN_CAP];
    uint16_t tile[PC_ROUTER_OPEN_CAP];
    int head, count;
} PcRouterOpen;

/* refresh_levels pushes |CalculateImprovementLevel - stored| entries per owned tile; both are neighbour counts (<= 8)
 * for every leveled kind the game has, so 8 x n is the real bound and 16 x PC_MAXN leaves room for imported states. */
#define PC_LEVEL_QUEUE_CAP (16 * PC_MAXN)

typedef struct {
    int32_t cost[PC_MAXN];
    int32_t came_from[PC_MAXN];
    PcRouterOpen open;
    uint16_t path[PC_MAXN + 2];   /* ReconstructPath: dest, then <= n + 1 steps */
    int path_n;
    uint16_t queue_tile[PC_LEVEL_QUEUE_CAP];
    int8_t queue_sign[PC_LEVEL_QUEUE_CAP];
} PcStructScratch;

PC_FN void pc_structures_open_clear(PcRouterOpen *o) {
    o->head = 0;
    o->count = 0;
}

/* OpenList::push: insert at partition_point(|&(p, _)| p <= prio), i.e. after every entry whose priority is <= prio */
PC_FN void pc_structures_open_push(PcRouterOpen *o, float prio, int i) {
    int left = 0, right = o->count;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (o->prio[o->head + mid] <= prio) left = mid + 1; else right = mid;
    }
    if (o->head + o->count >= PC_ROUTER_OPEN_CAP) {
        /* unreachable by the push bound above; compact (or drop) rather than write past the arrays */
        if (o->head == 0) return;
        for (int k = 0; k < o->count; k++) {
            o->prio[k] = o->prio[o->head + k];
            o->tile[k] = o->tile[o->head + k];
        }
        o->head = 0;
    }
    for (int k = o->count; k > left; k--) {
        o->prio[o->head + k] = o->prio[o->head + k - 1];
        o->tile[o->head + k] = o->tile[o->head + k - 1];
    }
    o->prio[o->head + left] = prio;
    o->tile[o->head + left] = (uint16_t)i;
    o->count += 1;
}

/* OpenList::pop: the front entry's tile, -1 when empty */
PC_FN int pc_structures_open_pop(PcRouterOpen *o) {
    if (o->count == 0) return -1;
    int i = (int)o->tile[o->head];
    o->head += 1;
    o->count -= 1;
    return i;
}

PC_FN uint8_t pc_structures_open_is_empty(const PcRouterOpen *o) { return o->count == 0; }

/* ---- structures.rs ---- */

#define PC_ROAD_BITS pc_u128_bit(3)
#define PC_BRIDGE_BITS pc_u128_bit(48)

PC_FN uint8_t pc_structures_temple(int32_t kind) { return kind >= 17 && kind <= 20; }

PC_FN int32_t pc_structures_score_of(const PcState *s, const PcRules *r, int t) {
    int32_t k = s->tiles.structure[t];
    if (pc_structures_temple(k)) return 100 * s->tiles.structure_level[t];
    return r->structures.score[k];
}

/* ActionUtils.CalculateImprovementLevel of a Market at `i` given its neighbours' levels: GetAdjacencyBonusAt
 * (ActionUtils.c 59-235) adds, for every same-owner neighbour with a Windmill, Forge or Sawmill, that neighbour's
 * level raised to at least 1 (<GetAdjacencyBonusAt>g__addLevel, ActionUtils.c 7167-7215: `if (level < 2) level = 1`;
 * the Market is neither Picky nor Simple), capped at the Market's maxLevel 8. */
PC_FN int32_t pc_structures_market_level(const PcState *s, const PcRules *r, const PcGeom *g, int i, const int32_t *levels) {
    const PcTiles *t = &s->tiles;
    int32_t sum = 0;
    for (int a = 0; a < g->adj_n[i]; a++) {
        int j = (int)g->adj[i * 8 + a];
        if (t->owner[i] == t->owner[j] && t->owner[i] > 0 && r->structures.adjacent[50][t->structure[j]]) {
            sum += pc_max(levels[j], 1);
        }
    }
    return pc_min(sum, 8);
}

/* Native multiplier levels from adjacent owned compatible improvements. */
PC_FN void pc_structures_production_levels(const PcState *s, const PcRules *r, const PcGeom *g, int32_t out[PC_MAXN]) {
    int n = s->n;
    const PcTiles *t = &s->tiles;
    int32_t levels[PC_MAXN];
    uint8_t has_market = 0;
    for (int i = 0; i < n; i++) {
        int k = t->structure[i];
        uint8_t multiplier = r->structures.any_adjacent[k] && t->structure[i] != 50;
        if (multiplier) {
            int32_t count = 0;
            for (int a = 0; a < g->adj_n[i]; a++) {
                int j = (int)g->adj[i * 8 + a];
                if (t->owner[i] == t->owner[j] && t->owner[i] > 0 && r->structures.adjacent[k][t->structure[j]]) {
                    count += 1;
                }
            }
            levels[i] = count;
        } else {
            levels[i] = t->structure_level[i];
        }
    }
    for (int i = 0; i < n; i++) has_market |= t->structure[i] == 50;
    if (!has_market) {
        for (int i = 0; i < n; i++) out[i] = levels[i];
        return;
    }
    for (int i = 0; i < n; i++) out[i] = levels[i];
    for (int i = 0; i < n; i++) {
        if (t->structure[i] == 50) {
            out[i] = pc_structures_market_level(s, r, g, i, levels);
        }
    }
}

/* population_contribution given production_levels(s). */
PC_FN void pc_structures_contribution_from(const PcState *s, const PcRules *r, const int32_t *levels, int32_t out[PC_MAXN]) {
    for (int i = 0; i < s->n; i++) {
        int k = s->tiles.structure[i];
        out[i] = r->structures.population[k] * (r->structures.any_adjacent[k] ? levels[i] : 1);
    }
}

PC_FN void pc_structures_population_contribution(const PcState *s, const PcRules *r, const PcGeom *g, int32_t out[PC_MAXN]) {
    int32_t levels[PC_MAXN];
    pc_structures_production_levels(s, r, g, levels);
    pc_structures_contribution_from(s, r, levels, out);
}

/* structures.adjust_building_population(before, after): `before_contribution` is population_contribution of the
 * pre-action state. */
PC_FN void pc_structures_adjust_building_population(const int32_t *before_contribution, PcState *s, const PcRules *r, const PcGeom *g) {
    int n = s->n;
    int32_t levels[PC_MAXN];
    int32_t after[PC_MAXN];
    int32_t amount[PC_MAXN];
    uint8_t any = 0;
    pc_structures_production_levels(s, r, g, levels);   /* s is unchanged until structure_level is set below */
    pc_structures_contribution_from(s, r, levels, after);
    for (int i = 0; i < n; i++) amount[i] = 0;
    for (int tile = 0; tile < n; tile++) {
        int32_t city = s->tiles.city[tile];
        if (city >= 0 && city < n) {
            if (s->cities.owner[city] == s->tiles.owner[tile]) {
                amount[city] += after[tile] - before_contribution[tile];
            }
        }
    }
    for (int i = 0; i < n; i++) s->tiles.structure_level[i] = levels[i];
    for (int i = 0; i < n; i++) any |= amount[i] != 0;
    if (any) {   /* all-zero amounts change nothing */
        for (int32_t owner = 1; owner <= s->np; owner++) {
            pc_economy_add_population_vector(s, amount, owner, 1);
        }
    }
}

/* (idx, exists) */
PC_FN void pc_structures_capital_index(const PcState *s, int32_t owner, int *idx, uint8_t *exists) {
    int32_t best = INT32_MAX;
    int at = 0;
    uint8_t ex = 0;
    for (int i = 0; i < s->n; i++) {
        if (s->cities.owner[i] == owner && s->tiles.capital[i] == owner) {
            ex = 1;
            if (s->cities.order[i] < best) {
                best = s->cities.order[i];
                at = i;
            }
        }
    }
    *idx = at;
    *exists = ex;
}

/* PlayerExtensions.GetCurrentCapitalCoordinates: the player's start tile while they own it, else their first city
 * in tile order; -1 (None) without cities. */
PC_FN int pc_structures_current_capital(const PcState *s, int32_t owner) {
    int cap;
    uint8_t exists;
    pc_structures_capital_index(s, owner, &cap, &exists);
    if (exists) return cap;
    for (int i = 0; i < s->n; i++) {
        if (s->cities.owner[i] == owner) return i;
    }
    return -1;
}

/* GameLogicData v28 route data of the improvements that open routes (ImprovementDataExtensions.IsRouteOpener:
 * `routes.Count > 0`): the terrain bitset of `routes`, `range`, and whether it has the Network ability (15). Port
 * (routes ocean, water, wetland; range 5), Outpost (ice; 5), Mycelium (field, forest, wetland, ice; 3; Network) and
 * Bridge (water; 1). Only Port and Bridge exist for Bardur. Returns 1 (Some) and fills the non-NULL out-params. */
PC_FN uint8_t pc_structures_route_data(int32_t kind, uint16_t *routes, int32_t *range, uint8_t *network) {
    uint16_t b;
    int32_t rg;
    uint8_t nw;
    switch (kind) {
    case 8: b = (uint16_t)((1u << 2) | (1u << 1) | (1u << 7)); rg = 5; nw = 0; break;
    case 33: b = (uint16_t)(1u << 6); rg = 5; nw = 0; break;
    case 39: b = (uint16_t)((1u << 3) | (1u << 5) | (1u << 7) | (1u << 6)); rg = 3; nw = 1; break;
    case 48: b = (uint16_t)(1u << 1); rg = 1; nw = 0; break;
    default: return 0;
    }
    if (routes) *routes = b;
    if (range) *range = rg;
    if (network) *network = nw;
    return 1;
}

/* GameLogicData.GetUnlockedMovements as a terrain bitset: the `movementUnlocks` keys of every known tech (GLD v28:
 * tech 0 field/forest/ice/wetland, 10 and 27/40/45 water, 13/25 and 43/46 ocean, 20 mountain), plus Water when
 * Ocean is in without it. */
PC_FN uint16_t pc_structures_unlocked_movements(const uint8_t tech[128]) {
    static const int UNLOCK_TECH[10] = {0, 10, 13, 20, 25, 27, 40, 43, 45, 46};
    static const uint16_t UNLOCK_BITS[10] = {
        (1u << 3) | (1u << 5) | (1u << 6) | (1u << 7), 1u << 1, 1u << 2, 1u << 4, 1u << 2, 1u << 1, 1u << 1, 1u << 2, 1u << 1, 1u << 2};
    uint16_t out = 0;
    for (int k = 0; k < 10; k++) {
        if (tech[UNLOCK_TECH[k]]) out |= UNLOCK_BITS[k];
    }
    if (out & (1u << 2)) out |= 1u << 1;
    return out;
}

/* TileData.IsConnectable: a road (the raw TileData.hasRoad), a City improvement (any city or village, whoever owns
 * it) or a route opener. */
PC_FN uint8_t pc_structures_connectable(const PcState *s, int i) {
    return s->tiles.road[i] || s->tiles.structure[i] == 1 || pc_structures_route_data(s->tiles.structure[i], NULL, NULL, NULL);
}

/* TileData.HasMatchingTransportPath(cur, other) (TileData.c 2488-2525): refused when `other` belongs to a player
 * who is neither `cur`'s owner nor at peace with that owner; ownership is checked only between the two tiles, and a
 * neutral `cur` passes (GameState.TryGetPlayer(0) finds no player, so the peace test is skipped). Otherwise two
 * connectable tiles (IsConnectable, 2534-2600) or two route tiles (hasRoute) link. */
PC_FN uint8_t pc_structures_transport_step(const PcState *s, int cur, int other, uint8_t native) {
    const PcTiles *t = &s->tiles;
    int32_t a = t->owner[cur], b = t->owner[other];
    if (b != 0 && b != a && a > 0 && !(native && s->players.peace[pc_pclip(s, a)][pc_pclip(s, b)])) {
        return 0;
    }
    return (pc_structures_connectable(s, cur) && pc_structures_connectable(s, other)) || (t->route[cur] && t->route[other]);
}

/* Tiles reached from `owner`'s start tile by MapDataExtensions.FindConnectedCities (MapDataExtensions.c 2740-2970):
 * a flood over the 8 neighbours (GetAreaPreallocated radius 1) along HasMatchingTransportPath, starting only if the
 * player owns their start tile. Returns the reached tiles and the start tile with whether they hold it. */
PC_FN void pc_structures_reachable(const PcState *s, int32_t owner, const PcGeom *g, uint8_t native, uint8_t out[PC_MAXN], int *cap, uint8_t *exists) {
    int n = s->n;
    uint16_t stack[PC_MAXN];   /* a tile is pushed once (marked reached first) */
    int sp = 0;
    pc_structures_capital_index(s, owner, cap, exists);
    for (int i = 0; i < n; i++) out[i] = 0;
    if (*exists) {
        out[*cap] = 1;
        stack[sp++] = (uint16_t)*cap;
        while (sp > 0) {
            int x = (int)stack[--sp];
            for (int a = 0; a < g->adj_n[x]; a++) {
                int y = (int)g->adj[x * 8 + a];
                if (!out[y] && pc_structures_transport_step(s, x, y, native)) {
                    out[y] = 1;
                    stack[sp++] = (uint16_t)y;
                }
            }
        }
    }
}

/* TileData.GetPossibleBridgeDirection (from the true terrain): 1 for a north-south span (land north and
 * south), 2 for east-west (land east and west), 0 for none; north-south wins when both hold. */
PC_FN uint8_t pc_structures_bridge_span(const PcState *s, int tile) {
    int size = s->size;
    int x = tile % size, y = tile / size;
    const int32_t *terr = s->tiles.terrain;
    if (y > 0 && y + 1 < size && (terr[tile - size] != 1 && terr[tile - size] != 2) && (terr[tile + size] != 1 && terr[tile + size] != 2)) {
        return 1;
    } else if (x > 0 && x + 1 < size && (terr[tile - 1] != 1 && terr[tile - 1] != 2) && (terr[tile + 1] != 1 && terr[tile + 1] != 2)) {
        return 2;
    } else {
        return 0;
    }
}

/* router_path's closures, with their captures as arguments */
PC_FN uint8_t pc_structures_rp_peace(const PcState *s, int32_t player, int p, int32_t o) {
    return o > 0 && o != player && s->players.peace[p][pc_pclip(s, o)];
}
PC_FN uint8_t pc_structures_rp_water(const PcState *s, int i) { return s->tiles.terrain[i] == 1 || s->tiles.terrain[i] == 2; }
PC_FN uint8_t pc_structures_rp_has(uint16_t allowed, int32_t terrain) {
    return terrain >= 0 && terrain < 16 && ((allowed >> terrain) & 1) == 1;
}
PC_FN uint8_t pc_structures_rp_accessible(const PcState *s, int32_t player, int p, uint8_t embark, uint16_t allowed, int tile, int from) {
    const PcTiles *t = &s->tiles;
    int32_t o = t->owner[tile];
    if (o != 0 && o != player && !pc_structures_rp_peace(s, player, p, o)) {
        return 0;
    }
    if (!t->explored[p][tile]) {
        return 0;
    }
    if (t->structure[tile] == 48) {
        if (!pc_structures_rp_has(allowed, t->terrain[tile])) {
            /* GetNonAllowedBridgeDirection: not straight across the span */
            int32_t dx = (int32_t)(from % s->size) - (int32_t)(tile % s->size);
            int32_t dy = (int32_t)(from / s->size) - (int32_t)(tile / s->size);
            uint8_t span = pc_structures_bridge_span(s, tile);
            if (span == 1 && dy == 0) return 0;
            if (span == 2 && dx == 0) return 0;
        }
        return 1;
    }
    if (pc_structures_rp_water(s, tile) && !pc_structures_rp_water(s, from)) {
        uint8_t friendly = o == player || pc_structures_rp_peace(s, player, p, o);
        if (!(t->structure[tile] == 8 && friendly) || !embark) {
            return 0;
        }
    }
    return t->structure[tile] == 1 || pc_structures_rp_has(allowed, t->terrain[tile]);
}
PC_FN int32_t pc_structures_rp_cost(const PcState *s, const pc_tilebits *route, int tile) {
    const PcTiles *t = &s->tiles;
    return (t->road[tile] || t->structure[tile] == 48 || t->structure[tile] == 1 || pc_tb_get(route, tile)) ? 5 : 10;
}

/* PathFinder.GetPath(map, origin, dest, maxCost = the origin's range, PathFinderSettings.CreateRouterSettings(player,
 * allowed)) as MapDataExtensions.<UpdateRoutes>g__FindPathBetweenRouters calls it (version 123). Router settings:
 * no unit, occupied tiles allowed, enemy tiles not, peace partners' tiles allowed (version > 59), port required to
 * enter water from land, transport paths followed. PathFinder.IsTileAccessible (PathFinder.c 1-195): the tile must
 * be the player's, neutral or a peace partner's, and explored by the player; a Bridge is entered unless its terrain
 * is not allowed and the step crosses its span; a water tile entered from land must be an own or peace partner's
 * port (Embark) and the player able to embark (ActionUtils.CanPlayerEmbark); then a City tile or allowed terrain.
 * TileData.GetMovementCost (TileData.c 2712-2745, no unit, shouldFollowTransportPaths): 5 onto a road, a Bridge,
 * a City or a tile already on a route (hasRoute, set by the paths found earlier in the same update), else 10.
 * A* as the unit search (pathfinding::search): priority Distance(tile, dest) + cost, the start at its Manhattan
 * distance, no closed set, expansion while cost < range x 10, at most 1000 pops. Fills `sc->path` (dest to origin). */
PC_FN uint8_t pc_structures_router_path(const PcState *s, int32_t player, uint8_t embark, int origin, int dest, int32_t range, uint16_t allowed,
                                        const pc_tilebits *route, PcStructScratch *sc) {
    int n = s->n;
    int32_t size = s->size;
    const PcTiles *t = &s->tiles;
    int p = player - 1;
    /* PathFinder.IsAllowedToFinishOnTile: occupied tiles allowed; a City of a player at peace or in cool-down is not */
    int32_t o = t->owner[dest];
    if (t->structure[dest] == 1 && o > 0 && o != player
        && (pc_structures_rp_peace(s, player, p, o) || s->players.truce_breaker[p][pc_pclip(s, o)] != 0)) {
        return 0;
    }
    int32_t tx = (int32_t)(dest % s->size), ty = (int32_t)(dest / s->size);
    for (int i = 0; i < n; i++) sc->cost[i] = 0;
    for (int i = 0; i < n; i++) sc->came_from[i] = -1;
    pc_structures_open_clear(&sc->open);
    int32_t sx = (int32_t)(origin % s->size), sy = (int32_t)(origin / s->size);
    pc_structures_open_push(&sc->open, (float)(pc_abs(tx - sx) + pc_abs(ty - sy)), origin);
    int32_t max_cost = range * 10;
    uint8_t found = 0;
    for (int iter = 0; iter < 1000; iter++) {
        int at = pc_structures_open_pop(&sc->open);
        if (at < 0) break;
        if (at == dest) {
            found = 1;
            break;
        }
        int32_t here = sc->cost[at];
        if (here < max_cost) {
            int32_t ax = (int32_t)(at % s->size), ay = (int32_t)(at / s->size);
            for (int32_t dx = -1; dx <= 1; dx++) {
                for (int32_t dy = -1; dy <= 1; dy++) {
                    int32_t x = ax + dx, y = ay + dy;
                    if ((dx == 0 && dy == 0) || x < 0 || y < 0 || x >= size || y >= size) {
                        continue;
                    }
                    int next = (int)(y * size + x);
                    if (!pc_structures_rp_accessible(s, player, p, embark, allowed, next, at)) {
                        continue;
                    }
                    int32_t c = pc_structures_rp_cost(s, route, next) + here;
                    int32_t old = sc->cost[next];
                    if (old < 1 || c < old) {
                        sc->cost[next] = c;
                        sc->came_from[next] = at;
                        {
                            int32_t ddx = tx - (int32_t)(next % s->size), ddy = ty - (int32_t)(next / s->size);
                            float dist = sqrtf((float)(ddy * ddy + ddx * ddx));
                            pc_structures_open_push(&sc->open, dist + (float)c, next);
                        }
                    }
                }
            }
        }
        if (pc_structures_open_is_empty(&sc->open)) {
            break;
        }
    }
    sc->path_n = 0;
    if (!found) {
        return 0;
    }
    /* PathFinder.ReconstructPath: destination back to the start, both included */
    int at = dest;
    sc->path[sc->path_n++] = (uint16_t)at;
    for (int iter = 0; iter <= n; iter++) {
        int32_t c = sc->came_from[at];
        if (c == -1) return 0;
        at = c;
        sc->path[sc->path_n++] = (uint16_t)at;
        if (at == origin) {
            return 1;
        }
    }
    return 0;
}

/* MapDataExtensions.UpdateRoutes (MapDataExtensions.c 1916-2345; UpdateRoutesAction.Execute): ResetRoutes clears
 * every tile's hasRoute, then for each player in turn order, over their empire tiles (GetPlayerEmpireTiles: tile
 * owner = the player, tile order) whose improvement opens routes, every pair i < j is tried with
 * FindPathBetweenRouters(router i, router j) (MapDataExtensions.c 2423-2735): no route beyond router i's range
 * (Chebyshev); the allowed terrain is each of router i's route terrains that router j's routes and the player's
 * movements (GetUnlockedMovements) both contain, or every one when j is a City; a path found marks all its tiles
 * hasRoute. A Network router (Mycelium) also tries each of the player's City tiles. hasRoute is one flag per tile
 * for all players, so later paths (any player) pay 5 on earlier route tiles. */
PC_FN void pc_structures_update_routes(PcState *s, const PcRules *r, PcStructScratch *sc) {
    int n = s->n;
    pc_tilebits route;
    uint16_t routers[PC_MAXN];
    uint16_t cities[PC_MAXN];
    pc_tb_clear(&route);
    for (int32_t player = 1; player <= s->np; player++) {
        int p = player - 1;
        const PcTiles *t = &s->tiles;
        int nr = 0, nc = 0;
        for (int i = 0; i < n; i++) {
            if (t->owner[i] == player && pc_structures_route_data(t->structure[i], NULL, NULL, NULL)) routers[nr++] = (uint16_t)i;
        }
        if (nr == 0) {
            continue;
        }
        for (int i = 0; i < n; i++) {
            if (t->owner[i] == player && t->structure[i] == 1) cities[nc++] = (uint16_t)i;
        }
        uint16_t moves = pc_structures_unlocked_movements(s->players.tech[p]);
        /* ActionUtils.CanPlayerEmbark: a known tech unlocks an improvement with the Embark ability (the Port) */
        uint8_t embark = 0;
        for (int k = 0; k < 128; k++) {
            if (s->players.tech[p][k] && pc_u128_test(r->structures.tech_unlocks[k], 8)) { embark = 1; break; }
        }
        for (int a = 0; a < nr; a++) {
            int origin = (int)routers[a];
            uint16_t from_routes;
            int32_t range;
            uint8_t network;
            pc_structures_route_data(s->tiles.structure[origin], &from_routes, &range, &network);
            /* routers[a + 1..] then, for a Network router, the player's City tiles */
            int total = (nr - (a + 1)) + (network ? nc : 0);
            for (int q = 0; q < total; q++) {
                int dest = q < nr - (a + 1) ? (int)routers[a + 1 + q] : (int)cities[q - (nr - (a + 1))];
                int ox = origin % s->size, oy = origin / s->size, dx = dest % s->size, dy = dest / s->size;
                if (pc_max(pc_abs(ox - dx), pc_abs(oy - dy)) > range) {
                    continue;
                }
                uint8_t city = s->tiles.structure[dest] == 1;
                uint16_t to_routes = 0;
                if (!pc_structures_route_data(s->tiles.structure[dest], &to_routes, NULL, NULL)) to_routes = 0;
                uint16_t allowed = city ? from_routes : (uint16_t)(from_routes & to_routes & moves);
                if (allowed == 0) {
                    continue;
                }
                if (pc_structures_router_path(s, player, embark, origin, dest, range, allowed, &route, sc)) {
                    for (int k = 0; k < sc->path_n; k++) {
                        pc_tb_set(&route, (int)sc->path[k], 1);
                    }
                }
            }
        }
    }
    for (int i = 0; i < n; i++) s->tiles.route[i] = (uint8_t)pc_tb_get(&route, i);
}

PC_FN void pc_structures_connections(PcState *s, int32_t owner, const PcGeom *g, uint8_t native) {
    int n = s->n;
    uint8_t reached[PC_MAXN];
    int cap;
    uint8_t exists;
    int32_t amount[PC_MAXN];
    int32_t total = 0;
    pc_structures_reachable(s, owner, g, native, reached, &cap, &exists);
    for (int i = 0; i < n; i++) amount[i] = 0;
    if (!exists) {
        /* the player no longer holds their start tile: every connected city of theirs disconnects
         * (DisconnectCityAction): -1 to it and -1 to the start tile, whoever holds it now */
        for (int c = 0; c < n; c++) {
            if (s->cities.owner[c] == owner && s->cities.connected[c]) {
                amount[c] = -1;
                total -= 1;
                s->cities.connected[c] = 0;
            }
        }
        if (total == 0) {
            return;
        }
        pc_economy_add_population_vector(s, amount, owner, native);
        {
            int start = -1;
            for (int i = 0; i < n; i++) {
                if (s->tiles.capital[i] == owner && s->cities.owner[i] > 0) { start = i; break; }
            }
            if (start >= 0) {
                int32_t v[PC_MAXN];
                for (int i = 0; i < n; i++) v[i] = 0;
                v[start] = total;
                pc_economy_add_population_vector(s, v, s->cities.owner[start], native);
            }
        }
        return;
    }
    for (int c = 0; c < n; c++) {
        uint8_t own = s->cities.owner[c] == owner && c != cap && exists;
        if (own) {
            int32_t delta = (int32_t)reached[c] - (int32_t)s->cities.connected[c];
            amount[c] = delta;
            total += delta;
            s->cities.connected[c] = reached[c];
        }
    }
    amount[cap] += total;
    int32_t pop_owner = native ? owner : s->current_player;
    pc_economy_add_population_vector(s, amount, pop_owner, native);
}

/* monument_conditions as a bitset, given the owner's connected-city count and whether a city of theirs has level 5. */
PC_FN pc_u128 pc_structures_monument_bits(const PcState *s, int32_t owner, const PcRules *r, const PcGeom *g, int connected, uint8_t level5) {
    const PcPlayers *p = &s->players;
    int pi = owner - 1;
    const uint8_t *tech = p->tech[pi];
    pc_u128 all_ones;
    all_ones.lo = ~0ull;
    all_ones.hi = ~0ull;
    uint8_t all_tech = pc_u128_eq(pc_u128_or(pc_bits128(tech), pc_u128_not(r->tech_exists_bits)), all_ones);
    uint8_t explorer = 1;
    for (int k = 0; k < 4; k++) explorer = explorer && s->tiles.explored[pi][g->corners[k]];
    explorer = explorer && s->fog;
    pc_u128 bits = pc_u128_zero();
    /* the Network task (Grand Bazaar) is enabled by the first city connection, not by the Roads tech */
    if (tech[21] && p->pacifist[pi] >= 5) pc_u128_set(&bits, 23);
    if (all_tech) pc_u128_set(&bits, 24);
    if (connected >= 5) pc_u128_set(&bits, 25);
    if (tech[5] && p->stars[pi] >= 100) pc_u128_set(&bits, 26);
    if (p->kills[pi] >= 10) pc_u128_set(&bits, 27);
    if (level5) pc_u128_set(&bits, 28);
    if (explorer) pc_u128_set(&bits, 29);
    return bits;
}

PC_FN void pc_structures_monument_conditions(const PcState *s, int32_t owner, const PcRules *r, const PcGeom *g, uint8_t out[128]) {
    const PcCities *c = &s->cities;
    int connected = 0;
    uint8_t level5 = 0;
    for (int i = 0; i < s->n; i++) {
        if (c->owner[i] == owner) {
            connected += c->connected[i];
            level5 |= c->level[i] >= 5;
        }
    }
    pc_u128 bits = pc_structures_monument_bits(s, owner, r, g, connected, level5);
    for (int k = 0; k < 128; k++) {
        out[k] = (uint8_t)pc_u128_test(bits, k);
    }
}

PC_FN void pc_structures_remember_monuments(PcState *s, const PcRules *r, const PcGeom *g) {
    /* one pass over the cities for every seat's connected counts and level-5 flags */
    const PcCities *c = &s->cities;
    int32_t np = s->np;
    int connected[PC_MAXP];
    uint8_t level5[PC_MAXP];
    for (int k = 0; k < PC_MAXP; k++) { connected[k] = 0; level5[k] = 0; }
    for (int i = 0; i < s->n; i++) {
        int32_t o = c->owner[i];
        if (o >= 1 && o <= np) {
            connected[o - 1] += c->connected[i];
            level5[o - 1] |= c->level[i] >= 5;
        }
    }
    for (int32_t owner = 1; owner <= np; owner++) {
        int pi = owner - 1;
        pc_u128 earned = pc_structures_monument_bits(s, owner, r, g, connected[pi], level5[pi]);
        uint8_t *m = s->players.monuments[pi];
        for (int k = 0; k < 128; k++) {
            m[k] |= (uint8_t)pc_u128_test(earned, k);
        }
    }
}

/* Per-call inputs of structures.build_mask shared by every tile (BuildCtx::prepare). */
typedef struct {
    /* economic_tiles(s), filled by prepare(), and the same tiles as a list */
    uint8_t econ[PC_MAXN];
    uint16_t econ_list[PC_MAXN];
    int econ_n;
    uint16_t own_cities[PC_MAXN];
    int own_n;
    /* structures present in the territory of each own city (bit = structure kind), parallel to own_cities */
    pc_u128 present[PC_MAXN];
    pc_u128 unlocked;
    pc_u128 monuments;
} PcBuildCtx;

PC_FN uint8_t pc_structures_build_ctx_road_unlocked(const PcBuildCtx *ctx) { return pc_u128_any(pc_u128_and(ctx->unlocked, PC_ROAD_BITS)); }

/* BuildCtx::new */
PC_FN void pc_structures_build_ctx_init(PcBuildCtx *ctx) {
    for (int i = 0; i < PC_MAXN; i++) ctx->econ[i] = 0;
    ctx->econ_n = 0;
    ctx->own_n = 0;
    ctx->unlocked = pc_u128_zero();
    ctx->monuments = pc_u128_zero();
}

/* Everything of build_mask that does not depend on the tile, including economic_tiles(s) into `econ` (one
 * pass over each own city's territory row gives both the economy and the structures present per city).
 * With lanes (pc.h PcLanes): tile t is always handled by lane t % count, so econ[t] has one writer; the own cities
 * are visited in index order (a ballot per chunk of candidate cities), each city's territory row strided over the
 * lanes and its present-structure bits OR-reduced; econ_list is compacted in tile order. Ends with a lane sync. */
PC_FN void pc_structures_build_ctx_prepare_lanes(PcBuildCtx *ctx, const PcState *s, const PcRules *r, const PcGeom *g, PcLanes L) {
    int n = s->n;
    const PcTiles *t = &s->tiles;
    const PcCities *c = &s->cities;
    const PcPlayers *p = &s->players;
    int32_t cp = s->current_player;
    int pi = pc_p(s);
    const PcStructureRules *rs = &r->structures;
    uint8_t *econ = ctx->econ;
    for (int i = L.lane; i < n; i += L.count) econ[i] = 0;
    pc_u128 tribe_has = pc_u128_zero();
    int connected = 0;
    uint8_t level5 = 0;
    int own_n = 0;
    for (int base = 0; base < n; base += L.count) {
        int cand = base + L.lane;
        uint32_t mine = pc_lanes_ballot(L, cand < n && c->owner[cand] == cp);
        while (mine) {
            int city = base + pc_lowest32(mine);
            mine &= mine - 1u;
            connected += c->connected[city];
            level5 |= c->level[city] >= 5;
            /* the city's structures: those on the tiles it rules (a Limited kind is one per ruling city) */
            pc_u128 pr = pc_u128_zero();
            for (int tile = L.lane; tile < n; tile += L.count) {
                if (pc_territory(s, city, tile)) {
                    econ[tile] = 1;
                }
                if (t->city[tile] == city) {
                    pc_u128_set(&pr, t->structure[tile]);
                }
            }
            pr = pc_lanes_or128(L, pr);
            tribe_has = pc_u128_or(tribe_has, pr);
            if (L.lane == 0) {
                ctx->own_cities[own_n] = (uint16_t)city;
                ctx->present[own_n] = pr;
            }
            own_n += 1;
        }
    }
    int econ_n = 0;
    for (int base = 0; base < n; base += L.count) {
        int tile = base + L.lane;
        int e = 0;
        if (tile < n) {
            /* GameLogicData.CanBuild: the tile itself must be the player's, whatever the city rows say */
            e = econ[tile] && pc_economy_unit_allows_work(s, tile) && t->owner[tile] == cp;
            econ[tile] = (uint8_t)e;
        }
        int at = pc_lanes_compact(L, e, &econ_n);
        if (e) {
            ctx->econ_list[at] = (uint16_t)tile;
        }
    }
    pc_u128 tech_ok = pc_u128_zero();
    pc_u128 known = pc_bits128(p->tech[pi]);
    while (pc_u128_any(known)) {
        int k = pc_u128_lowest(known);
        tech_ok = pc_u128_or(tech_ok, rs->tech_unlocks[k]);
        pc_u128_clear(&known, k);
    }
    pc_u128 earned = pc_u128_or(pc_structures_monument_bits(s, cp, r, g, connected, level5), pc_bits128(p->monuments[pi]));
    if (L.lane == 0) {
        ctx->own_n = own_n;
        ctx->econ_n = econ_n;
        ctx->unlocked = pc_u128_andnot(pc_u128_and(pc_u128_and(tech_ok, rs->static_ok), pc_cost_le(&rs->cost_le, p->stars[pi])),
                                       pc_u128_and(rs->per_tribe_bits, tribe_has));
        /* ImprovementAbility Unique (PlayerState.builtUniqueImprovements): a monument is blocked only by the player
         * having built it, not by one standing in a captured city's territory */
        ctx->monuments = pc_u128_andnot(earned, pc_bits128(p->built[pi]));
    }
    pc_lanes_sync(L);
}

PC_FN void pc_structures_build_ctx_prepare(PcBuildCtx *ctx, const PcState *s, const PcRules *r, const PcGeom *g) {
    pc_structures_build_ctx_prepare_lanes(ctx, s, r, g, pc_lanes_one());
}

/* GameLogicData.CanBuild (a Bridge is Freelance like a road): an unlocked bridge on a neutral, explored,
 * structure-less water tile with a span, free of other players' visible units. */
PC_FN uint8_t pc_structures_neutral_bridge(const PcState *s, const PcRules *r, const PcBuildCtx *ctx, int tile) {
    const PcTiles *t = &s->tiles;
    return pc_u128_any(pc_u128_and(ctx->unlocked, PC_BRIDGE_BITS)) && t->owner[tile] == 0 && t->explored[pc_p(s)][tile] && t->structure[tile] == 0
        && pc_economy_unit_allows_work(s, tile)
        && pc_u128_any(pc_u128_and(r->structures.terrain_bits[pc_clamp(t->terrain[tile], 0, 15)], PC_BRIDGE_BITS))
        && pc_structures_bridge_span(s, tile) != 0;
}

/* Term (b) of build_row: an unlocked road on a neutral, explored, road-less tile whose terrain allows roads. */
PC_FN uint8_t pc_structures_neutral_road(const PcState *s, const PcRules *r, const PcBuildCtx *ctx, int tile) {
    const PcTiles *t = &s->tiles;
    return pc_u128_any(pc_u128_and(ctx->unlocked, PC_ROAD_BITS)) && t->owner[tile] == 0 && t->explored[pc_p(s)][tile] && !t->road[tile]
        && t->structure[tile] != 1
        && pc_economy_unit_allows_work(s, tile)
        && pc_u128_any(pc_u128_and(r->structures.terrain_bits[pc_clamp(t->terrain[tile], 0, 15)], PC_ROAD_BITS));
}

/* Buildable structure kinds on `tile` as a bitset: one row of structures.build_mask (native profile), with each
 * of its four terms evaluated for all 128 kinds at once as bitset algebra:
 *   (a) unlocked & terrain & empty & adjacency & city room   (b) neutral road   (c) the tile's resource building
 *   (d) earned monuments on free land; then the bridge span test. */
PC_FN pc_u128 pc_structures_build_row(const PcState *s, const PcRules *r, const PcGeom *g, const PcBuildCtx *ctx, int tile) {
    const PcTiles *t = &s->tiles;
    const PcCities *c = &s->cities;
    const PcPlayers *p = &s->players;
    int32_t cp = s->current_player;
    int pi = pc_p(s);
    const PcStructureRules *rs = &r->structures;
    pc_u128 terrain = rs->terrain_bits[pc_clamp(t->terrain[tile], 0, 15)];
    pc_u128 b = pc_u128_zero();
    if (pc_structures_neutral_road(s, r, ctx, tile)) b = pc_u128_or(b, PC_ROAD_BITS);
    if (pc_structures_neutral_bridge(s, r, ctx, tile)) b = pc_u128_or(b, PC_BRIDGE_BITS);
    if (!ctx->econ[tile]) {
        return b; /* every other term carries econ[tile] */
    }
    uint8_t free_ = t->structure[tile] == 0;
    pc_u128 adjacency = rs->no_adjacent_bits;
    for (int a = 0; a < g->adj_n[tile]; a++) {
        int j = (int)g->adj[tile * 8 + a];
        if (t->owner[j] == cp) {
            adjacency = pc_u128_or(adjacency, rs->adjacent_rev[t->structure[j]]);
        }
    }
    /* a per-city (Limited) kind has room iff the city ruling this tile (tile.rulingCityCoordinates) lacks it;
     * territory rows can overlap, the ruler is one city */
    int32_t ruler = t->city[tile];
    pc_u128 room = pc_u128_zero();
    {
        int k = -1;
        for (int q = 0; q < ctx->own_n; q++) {
            if ((int32_t)ctx->own_cities[q] == ruler) { k = q; break; }
        }
        if (k >= 0) room = pc_u128_or(pc_u128_not(rs->per_city_bits), pc_u128_not(ctx->present[k]));
    }
    /* TileData.HasRoad is true on every City tile, villages included, so no road goes there */
    uint8_t road_ok = !t->road[tile] && c->owner[tile] == 0 && t->structure[tile] != 1;
    pc_u128 empty = pc_u128_zero();
    if (free_) empty = pc_u128_or(empty, pc_u128_not(PC_ROAD_BITS));
    if (road_ok) empty = pc_u128_or(empty, PC_ROAD_BITS);
    pc_u128 a = pc_u128_and(pc_u128_and(pc_u128_and(pc_u128_and(ctx->unlocked, terrain), empty), adjacency), room);
    int res_tech, res_build;
    switch (t->resource[tile]) {
    case 2: res_tech = 8; res_build = 5; break;
    case 5: res_tech = 23; res_build = 21; break;
    case 7: res_tech = 0; res_build = 37; break;
    default: res_tech = 0; res_build = 0; break;
    }
    uint8_t resource = free_ && res_build > 0 && p->tech[pi][res_tech] && p->stars[pi] >= rs->cost[res_build];
    pc_u128 d = free_ ? pc_u128_and(terrain, ctx->monuments) : pc_u128_zero();
    pc_u128 v = pc_u128_or(pc_u128_or(a, b), d);
    if (resource) pc_u128_set(&v, res_build);
    if (pc_u128_any(pc_u128_and(v, PC_BRIDGE_BITS)) && pc_structures_bridge_span(s, tile) == 0) {
        v = pc_u128_andnot(v, PC_BRIDGE_BITS);
    }
    return v;
}

/* build_mask into out[t*128 + kind] (structures.build_mask, native profile). */
PC_FN void pc_structures_build_mask(const PcState *s, const PcRules *r, const PcGeom *g, PcBuildCtx *ctx, uint8_t *out) {
    pc_structures_build_ctx_init(ctx);
    pc_structures_build_ctx_prepare(ctx, s, r, g);
    for (int tile = 0; tile < s->n; tile++) {
        pc_u128 bits = pc_structures_build_row(s, r, g, ctx, tile);
        for (int kind = 0; kind < 128; kind++) {
            out[tile * 128 + kind] = (uint8_t)pc_u128_test(bits, kind);
        }
    }
}

/* One entry of build_mask (step validation). */
PC_FN uint8_t pc_structures_build_legal(const PcState *s, const PcRules *r, const PcGeom *g, PcBuildCtx *ctx, int tile, int kind) {
    pc_structures_build_ctx_prepare(ctx, s, r, g);
    return (uint8_t)pc_u128_test(pc_structures_build_row(s, r, g, ctx, tile), kind);
}

PC_FN void pc_structures_build(PcState *s, int target, int kind, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int32_t before[PC_MAXN];
    pc_structures_population_contribution(s, r, g, before);
    const PcStructureRules *rs = &r->structures;
    uint8_t road = kind == 3;
    int pi = pc_p(s);
    if (!road) {
        s->tiles.structure[target] = (int32_t)kind;
        s->tiles.structure_level[target] = 1;
        s->tiles.structure_turn[target] = s->turn;
    }
    s->tiles.road[target] |= road;
    s->players.stars[pi] -= pc_max(rs->cost[kind], 0);
    /* BuildAction.Execute (BuildAction.c 680-720): a route opener or Bridge adds UpdateRoutesAction, then (a road
     * too) UpdateCityConnectionsAction for the builder */
    uint8_t opener = pc_structures_route_data((int32_t)kind, NULL, NULL, NULL);
    if (opener) {
        pc_structures_update_routes(s, r, sc);
    }
    if (kind == 3 || opener) {
        pc_structures_connections(s, s->current_player, g, 1);
    }
    int32_t score = pc_structures_temple((int32_t)kind) ? 100 : rs->score[kind];
    int32_t owner = s->tiles.owner[target];
    if (owner > 0) {
        int o = pc_pclip(s, owner);
        s->players.score[o] = pc_wadd(s->players.score[o], score);
    }
    s->players.built[pi][kind] |= score > 0;
    pc_structures_adjust_building_population(before, s, r, g);
}

/* Structures Destroy removes (ImprovementDataExtensions.IsDestructable): not cities, ruins or lighthouses. */
PC_FN uint8_t pc_structures_destructible(int32_t k) { return k > 0 && k != 1 && k != 2 && k != 47; }

/* destroy_mask given economic_tiles(s). */
PC_FN void pc_structures_destroy_mask_with(const PcState *s, const uint8_t *econ, uint8_t out[PC_MAXN]) {
    uint8_t tech3 = s->players.tech[pc_p(s)][3];
    for (int t = 0; t < s->n; t++) {
        out[t] = econ[t] && pc_structures_destructible(s->tiles.structure[t]) && tech3;
    }
}

PC_FN void pc_structures_destroy_mask(const PcState *s, uint8_t out[PC_MAXN]) {
    uint8_t econ[PC_MAXN];
    pc_economy_economic_tiles(s, out);
    for (int t = 0; t < s->n; t++) econ[t] = out[t];
    pc_structures_destroy_mask_with(s, econ, out);
}

PC_FN void pc_structures_destroy(PcState *s, int target, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int32_t before[PC_MAXN];
    pc_structures_population_contribution(s, r, g, before);
    int32_t kind = s->tiles.structure[target];
    int32_t score = pc_structures_score_of(s, r, target);
    int32_t owner = s->tiles.owner[target];
    s->tiles.structure[target] = 0;
    s->tiles.structure_level[target] = 0;
    s->tiles.structure_turn[target] = 0;
    s->tiles.road[target] = s->tiles.road[target] && kind != 3 && kind != 48;
    if (owner > 0) {
        int o = pc_pclip(s, owner);
        s->players.score[o] = pc_wsub(s->players.score[o], score);
    }
    pc_structures_adjust_building_population(before, s, r, g);
    /* DestroyImprovementAction.ExecuteDefault (DestroyImprovementAction.c 196-240): a route opener or Bridge adds
     * UpdateRoutesAction and UpdateCityConnectionsAction for the destroyer */
    uint8_t opener = pc_structures_route_data(kind, NULL, NULL, NULL);
    if (opener) {
        pc_structures_update_routes(s, r, sc);
    }
    if (kind == 3 || opener) {
        pc_structures_connections(s, s->current_player, g, 1);
    }
}

PC_FN void pc_structures_start_turn(PcState *s, int32_t owner) {
    int n = s->n;
    int32_t gain = 0;
    for (int i = 0; i < n; i++) {
        PcTiles *t = &s->tiles;
        if (pc_structures_temple(t->structure[i]) && t->owner[i] == owner) {
            int32_t expected = pc_clamp(1 + pc_div_euclid(s->turn - t->structure_turn[i], 2), 1, 5);
            int32_t new_level = pc_max(expected, t->structure_level[i]);
            gain += new_level - t->structure_level[i];
            t->structure_level[i] = new_level;
        }
    }
    int o = owner - 1;
    s->players.score[o] = pc_wadd(s->players.score[o], gain * 100);
}

/* refresh_levels' `adjacency` closure: CalculateImprovementLevel of tile i against the stored neighbour levels */
PC_FN int32_t pc_structures_refresh_adjacency(const PcState *s, const PcRules *r, const PcGeom *g, int i) {
    int k = s->tiles.structure[i];
    if (k == 50) {
        return pc_structures_market_level(s, r, g, i, s->tiles.structure_level);
    }
    /* a neighbour without levels (farm, mine, lumber hut) counts 1 */
    int32_t count = 0;
    for (int a = 0; a < g->adj_n[i]; a++) {
        int j = (int)g->adj[i * 8 + a];
        if (s->tiles.owner[i] == s->tiles.owner[j] && s->tiles.owner[i] > 0 && r->structures.adjacent[k][s->tiles.structure[j]]) count += 1;
    }
    return count;
}

/* The ImprovementLevelUp/LevelDown actions of StartTurnAction.ExecuteDefault: ActionUtils.UpdateImprovementLevel
 * (ActionUtils.c 6777-6875), for each tile of `owner` in tile order, queues |CalculateImprovementLevel - stored|
 * actions computed from the stored levels; the action stack then runs them last queued first, each only while it is
 * still valid against the levels at that moment (ImprovementLevelUpAction.IsValid: stored < CalculateImprovementLevel;
 * ImprovementLevelDownAction.IsValid: stored > it), one level each, with the improvement's growth population to its
 * ruling city. Windmills, Forges, Sawmills and Markets (temples level by age in start_turn). */
PC_FN void pc_structures_refresh_levels(PcState *s, int32_t owner, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int n = s->n;
    const PcStructureRules *rs = &r->structures;
    int qn = 0;
    for (int i = 0; i < n; i++) {
        int32_t k = s->tiles.structure[i];
        if (s->tiles.owner[i] == owner && rs->any_adjacent[k] && !pc_structures_temple(k)) {
            int32_t d = pc_structures_refresh_adjacency(s, r, g, i) - s->tiles.structure_level[i];
            int8_t sign = (int8_t)(d > 0 ? 1 : (d < 0 ? -1 : 0));
            for (int32_t c = 0; c < pc_abs(d); c++) {
                if (qn >= PC_LEVEL_QUEUE_CAP) break;   /* unreachable for game-made levels (|d| <= 8 per tile) */
                sc->queue_tile[qn] = (uint16_t)i;
                sc->queue_sign[qn] = sign;
                qn += 1;
            }
        }
    }
    if (qn == 0) {
        return;
    }
    int32_t amount[PC_MAXN];
    uint8_t any = 0;
    for (int i = 0; i < n; i++) amount[i] = 0;
    while (qn > 0) {
        qn -= 1;
        int i = (int)sc->queue_tile[qn];
        int32_t sign = sc->queue_sign[qn];
        int32_t now = pc_structures_refresh_adjacency(s, r, g, i);
        int32_t stored = s->tiles.structure_level[i];
        if ((sign > 0 && stored < now) || (sign < 0 && stored > now && stored > 0)) {
            s->tiles.structure_level[i] += sign;
            int32_t city = s->tiles.city[i];
            if (city >= 0 && city < n && s->cities.owner[city] == s->tiles.owner[i]) {
                amount[city] += sign * rs->population[s->tiles.structure[i]];
            }
        }
    }
    for (int i = 0; i < n; i++) any |= amount[i] != 0;
    if (any) {
        pc_economy_add_population_vector(s, amount, owner, 1);
    }
}

/* income's `embassy_pay` closure. PlayerDiplomacyExtensions.GetIncomeFromEmbassy: an embassy pays 2 (4 at peace)
 * unless the two are at war */
PC_FN int32_t pc_structures_embassy_pay(const PcState *s, int builder, int host) {
    if (s->players.embassy[builder][host] && !pc_diplomacy_at_war(s, builder, host)) {
        return 2 * (s->players.peace[builder][host] ? 2 : 1);
    } else {
        return 0;
    }
}

PC_FN int32_t pc_structures_income(const PcState *s, int32_t owner, const PcRules *r, const PcGeom *g) {
    int n = s->n;
    const PcTiles *t = &s->tiles;
    const PcCities *c = &s->cities;
    const PcUnits *u = &s->units;
    int32_t total = 0;
    uint8_t has_market = 0;
    for (int i = 0; i < n; i++) has_market |= t->structure[i] == 50;
    if (has_market) {
        /* StartTurnAction.ExecuteDefault (StartTurnAction.c 1022-1150) pays TileDataExtensions.CalculateWork(tile,
         * ActionUtils.UpdateImprovementLevel(tile)) for every owned tile, in tile order: each owned market pays on its
         * own, whatever happens to its city (a siege or negative progress stops only the city). The level is
         * CalculateImprovementLevel against the neighbours' *stored* levels: the ImprovementLevelUp/Down actions
         * UpdateImprovementLevel queues run only after the whole loop (refresh_levels). */
        for (int i = 0; i < n; i++) {
            if (t->owner[i] == owner && t->structure[i] == 50) {
                total += pc_structures_market_level(s, r, g, i, t->structure_level) * r->structures.stars[50];
            }
        }
    }
    int p = owner - 1;
    for (int city = 0; city < n; city++) {
        if (c->owner[city] != owner) {
            /* the builder's share: paid from the host's capital while no other player's unit stands on it */
            int32_t host = t->capital[city];
            if (host > 0 && c->owner[city] == host && !(u->owner[city] != 0 && u->owner[city] != host)) {
                total += pc_structures_embassy_pay(s, p, host - 1);
            }
            continue;
        }
        /* any unit of another player on the city stops its work, a hidden one included; so does a riot */
        uint8_t siege = (u->owner[city] != 0 && u->owner[city] != c->owner[city]) || c->riot[city];
        if (siege) {
            continue;
        }
        uint8_t start = t->capital[city] == c->owner[city] && t->capital[city] > 0;
        int32_t production = c->production[city] + (int32_t)start;
        if (start) {
            /* the host's share is part of its capital's work (TileDataExtensions.CalculateWork) */
            for (int q = 0; q < s->np; q++) {
                if (q != p) production += pc_structures_embassy_pay(s, q, p);
            }
        }
        if (c->progress[city] < 0) {
            production = pc_max(production + c->progress[city], 0);
        }
        total += production;
    }
    return total;
}

/* structures.rs `_unused` (keeps in_territory / add_population referenced) has no port: nothing to compute. */

#ifdef __cplusplus
}
#endif
#endif /* PC_STRUCTURES_H */
