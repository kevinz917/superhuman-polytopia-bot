/* pathfinding.h - line-by-line port of engine-rs/src/pathfinding.rs.
 *
 * The path a commanded move takes (PathFinder.GetPath, game version 123), which decides the tiles it explores;
 * gpu/pathfinding.py's reference-ordered Dijkstra remains as a fallback. */
#ifndef PC_PATHFINDING_H
#define PC_PATHFINDING_H

#include "pc.h"
#include "kernels.h"  /* PC_INF, MV_* flag bits, pc_kernels_movement_flags / hide_passable / peace_unit / forest_step / bridge_blocks */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Heap (search_ref's reference heap): ids/costs of 8 * n + 1 entries in the Rust; ids are tile indices ---- */
#define PC_PATH_HEAP_CAP (8 * PC_MAXN + 1)
typedef struct {
    uint16_t ids[PC_PATH_HEAP_CAP];
    int32_t costs[PC_PATH_HEAP_CAP];
    int count;
} PcPathHeap;

PC_FN void pc_pathfinding_heap_push(PcPathHeap *h, int32_t index, int32_t cost) {
    int at = h->count;
    h->ids[at] = (uint16_t)index;
    h->costs[at] = cost;
    while (at > 0 && h->costs[at] < h->costs[(at - 1) / 2]) {
        int parent = (at - 1) / 2;
        uint16_t ti = h->ids[at]; h->ids[at] = h->ids[parent]; h->ids[parent] = ti;
        int32_t tc = h->costs[at]; h->costs[at] = h->costs[parent]; h->costs[parent] = tc;
        at = parent;
    }
    h->count += 1;
}

/* Rust (index, cost) -> (*index, *cost) */
PC_FN void pc_pathfinding_heap_pop(PcPathHeap *h, int32_t *index, int32_t *cost) {
    *index = (int32_t)h->ids[0];
    *cost = h->costs[0];
    h->count -= 1;
    int count = h->count;
    uint16_t last_id = h->ids[count];
    int32_t last_cost = h->costs[count];
    int at = 0;
    while (2 * at + 1 < count) {
        int left = 2 * at + 1;
        int right = left + 1;
        int child = (right < count && h->costs[right] <= h->costs[left]) ? right : left;
        h->ids[at] = h->ids[child];
        h->costs[at] = h->costs[child];
        at = child;
    }
    h->ids[at] = last_id;
    h->costs[at] = last_cost;
    while (at > 0 && h->costs[at] < h->costs[(at - 1) / 2]) {
        int p = (at - 1) / 2;
        uint16_t ti = h->ids[at]; h->ids[at] = h->ids[p]; h->ids[p] = ti;
        int32_t tc = h->costs[at]; h->costs[at] = h->costs[p]; h->costs[p] = tc;
        at = p;
    }
}

/* ---- OpenList (pub; structures.rs RouterScratch uses it too): a sorted open list, pop takes the lowest priority,
 * equal priorities in insertion order. The Rust VecDeque never shrinks its reach: here items live in [head, head +
 * count) of a flat array, pop_front advances head, and insert shifts the tail.
 *
 * Capacity (2026-11-06). Two bounds hold on the pushes of one `search`, the capacity is the smaller:
 *   (a) 1 + 8 x 1000: the start, then at most 8 per pop of the 1000-pop loop.
 *   (b) PC_PATH_OPEN_PER_TILE x n + 1: a tile is pushed when it is unvisited (or is the source, whose cost 0 reads as
 *       unvisited once) or when its cost strictly improves, so its pushes are at most one per distinct cost value it
 *       can take. Every step costs 5, 10 or 1000 (pc_pathfinding_cost), so costs are multiples of 5; a tile is expanded
 *       only while its cost `here` < max_cost = 10 x (movement + boost) <= 40 (the rules' largest movement is 3, boost
 *       adds 1; pc_pathfinding_open_fits checks this at pc_set_rules), so a pushed cost is here + {5, 10} in
 *       (0, max_cost + 10) (max_cost / 5 + 1 values) or here + 1000 in [1000, 1000 + max_cost) (max_cost / 5 values):
 *       2 x max_cost / 5 + 1 = 17 distinct values, 18 with the source's extra push, per tile.
 * 18 x 324 + 1 = 5833 (35 KB at 6 bytes per entry) against the 8192 x 8 = 64 KB of the first port; 3529 (21 KB) in the
 * PC_MAXSIZE=14 build. The push keeps the compaction fallback (semantically invisible: the live entries move to the
 * front of the array) so a bound miss can only cost time, never an entry. ---- */
#define PC_PATH_OPEN_PER_TILE 18
#define PC_PATH_OPEN_MAX_MOVEMENT 3   /* largest unit movement the per-tile bound (b) assumes; + 1 for the boost */
#define PC_PATH_OPEN_BOUND (PC_PATH_OPEN_PER_TILE * PC_MAXN + 1)
#define PC_PATH_OPEN_CAP (PC_PATH_OPEN_BOUND < 8001 ? PC_PATH_OPEN_BOUND : 8001)
typedef struct {
    float prio[PC_PATH_OPEN_CAP];
    uint16_t tile[PC_PATH_OPEN_CAP];   /* tile index < PC_MAXN (the Rust usize) */
    int head, count;
} PcPathOpenList;

/* The movement bound the open-list capacity rests on: every supported unit kind moves at most
 * PC_PATH_OPEN_MAX_MOVEMENT (pc_set_rules refuses a table that breaks it rather than risk a dropped push). */
PC_FN int pc_pathfinding_open_fits(const PcRules *r) {
    for (int k = 0; k < 128; k++) {
        if (r->unit_supported[k] && r->unit_movement[k] > PC_PATH_OPEN_MAX_MOVEMENT) return 0;
    }
    return 1;
}

PC_FN void pc_pathfinding_open_clear(PcPathOpenList *o) {
    o->head = 0;
    o->count = 0;
}
PC_FN void pc_pathfinding_open_push(PcPathOpenList *o, float prio, int i) {
    if (o->head + o->count >= PC_PATH_OPEN_CAP) {
        /* unreachable by the bounds above; compact (the VecDeque's contents are unchanged) rather than write past
         * the arrays, and drop only when the live entries alone fill the capacity (never in a conforming search) */
        if (o->head == 0) return;
        for (int k = 0; k < o->count; k++) {
            o->prio[k] = o->prio[o->head + k];
            o->tile[k] = o->tile[o->head + k];
        }
        o->head = 0;
    }
    /* partition_point(|&(p, _)| p <= prio): first position whose priority is greater */
    int lo = 0, hi = o->count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (o->prio[o->head + mid] <= prio) lo = mid + 1; else hi = mid;
    }
    for (int k = o->count; k > lo; k--) {
        o->prio[o->head + k] = o->prio[o->head + k - 1];
        o->tile[o->head + k] = o->tile[o->head + k - 1];
    }
    o->prio[o->head + lo] = prio;
    o->tile[o->head + lo] = (uint16_t)i;
    o->count += 1;
}
/* -1 when empty (Rust Option<usize>) */
PC_FN int pc_pathfinding_open_pop(PcPathOpenList *o) {
    if (o->count == 0) return -1;
    int i = (int)o->tile[o->head];
    o->head += 1;
    o->count -= 1;
    return i;
}
PC_FN uint8_t pc_pathfinding_open_is_empty(const PcPathOpenList *o) {
    return o->count == 0;
}

/* Reusable buffers for path search and reveal (one per game, inside the core scratch). */
typedef struct {
    PcPathHeap heap;
    uint8_t flags[PC_MAXN];
    int32_t dist[PC_MAXN];
    int32_t parent[PC_MAXN];
    PcPathOpenList open;
    /* path tiles from the target back to (excluding) the source: path[k] has rank k; path_len of them
     * (search's reconstruction loop runs n + 1 times, hence the + 1); tile indices, so u16 */
    uint16_t path[PC_MAXN + 1];
    int path_len;
    uint8_t reveal[PC_MAXN];
} PcPathScratch;

/* PathScratch::new */
PC_FN void pc_pathfinding_scratch_init(PcPathScratch *ps, int n) {
    for (int k = 0; k < PC_PATH_HEAP_CAP; k++) { ps->heap.ids[k] = 0; ps->heap.costs[k] = PC_INF; }
    ps->heap.count = 0;
    for (int i = 0; i < PC_MAXN; i++) { ps->flags[i] = 0; ps->dist[i] = PC_INF; ps->parent[i] = -1; ps->reveal[i] = 0; }
    pc_pathfinding_open_clear(&ps->open);
    ps->path_len = 0;
    (void)n;
}

/* `accessible` closure of search */
PC_FN uint8_t pc_pathfinding_accessible(const PcState *s, const uint8_t *flags, int source, int i, uint8_t pass_need,
                                        int32_t cp, int32_t owner, uint8_t hide, uint8_t aquatic) {
    const PcUnits *u = &s->units;
    return i == source || ((flags[i] & pass_need) == pass_need && (u->owner[i] == 0 || u->owner[i] == cp || pc_hidden_from(s, i, cp)))
        || ((hide || pc_kernels_peace_unit(s, owner, i)) && (aquatic || (flags[i] & MV_LAND) != 0)
            && pc_kernels_hide_passable(s, owner, i));
}

/* `cost` closure of search: TileData.GetMovementCost (version > 114) for the unit entering `to` from `from` */
PC_FN int32_t pc_pathfinding_cost(const uint8_t *flags, int from, int to, uint8_t aquatic, uint8_t creep, uint8_t no_zoc) {
    uint8_t f = flags[to];
    if (!aquatic && (f & MV_OWN_PORT) != 0) {
        return 1000; /* WillUnitEmbark */
    }
    if (!no_zoc && (f & MV_ZOC) != 0) {
        return 1000;
    }
    if ((f & MV_WATER) != 0) {
        if (creep) {
            return 10;
        }
    } else {
        if (aquatic) {
            return 1000; /* Swim on land */
        }
        if (creep && (f & MV_MOUNTAIN) == 0) {
            return 10;
        }
    }
    if ((flags[from] & MV_ROAD) != 0 && (f & MV_ROAD) != 0 && !creep && !aquatic) {
        return 5;
    }
    if ((f & (MV_MOUNTAIN | MV_FOREST)) != 0) {
        return 1000;
    }
    return 10;
}

/* The reference-ordered Dijkstra (the JAX engine's path), used only if GetPath's search fails. */
PC_FN uint8_t pc_pathfinding_search_ref(const PcState *s, int source, int target, const PcRules *r, const PcGeom *g,
                                        PcPathScratch *ps) {
    int n = s->n;
    const PcUnits *u = &s->units;
    int32_t cp = s->current_player;
    int32_t owner = u->owner[source];
    int kind = (int)u->kind[source];
    const uint8_t *sk = r->skills[kind];
    uint8_t aquatic = sk[PC_SK_FLOAT], creep = sk[PC_SK_CREEP], carry = sk[PC_SK_CARRY];
    uint8_t no_zoc = creep || sk[PC_SK_INFILTRATE] || u->invisible[source];
    int32_t budget = (r->unit_movement[kind] + (int32_t)u->boosted[source]) * 2;
    pc_kernels_movement_flags(s, owner, g, ps->flags);
    const uint8_t *flags = ps->flags;
    uint8_t term_bits = (uint8_t)((no_zoc ? 0 : MV_ZOC) | MV_MOUNTAIN | (aquatic ? 0 : MV_OWN_PORT));
    uint8_t pass_need = aquatic ? MV_PASS : (MV_PASS | MV_LAND);
    uint8_t hide = sk[PC_SK_HIDE];
    PcPathHeap *heap = &ps->heap;
    int32_t *dist = ps->dist;
    int32_t *parent = ps->parent;
    for (int i = 0; i < n; i++) { dist[i] = PC_INF; parent[i] = -1; }
    heap->ids[0] = (uint16_t)source;
    heap->costs[0] = 0;
    heap->count = 1;
    dist[source] = 0;
    uint8_t found = 0;
    while (heap->count > 0 && !found) {
        int32_t at_i, cost;
        pc_pathfinding_heap_pop(heap, &at_i, &cost);
        int at = (int)at_i;
        found = at == target;
        /* terminal(flags[at]) */
        uint8_t f_at = flags[at];
        uint8_t terminal = (f_at & term_bits) != 0 || ((f_at & (MV_FOREST | MV_ROAD)) == MV_FOREST && !creep)
            || (carry && (f_at & MV_WATER) == 0);
        uint8_t expand = !found && cost <= dist[at] && (at == source || !terminal) && cost < budget;
        if (expand) {
            uint8_t road_at = (flags[at] & MV_ROAD) != 0 && !creep && !aquatic;
            for (int k = 0; k < 8; k++) {
                if (!g->nb_valid[at * 8 + k]) {
                    continue;
                }
                int dest = (int)g->nb[at * 8 + k];
                uint8_t road_edge = road_at && (flags[dest] & MV_ROAD) != 0;
                int32_t newcost = pc_kernels_forest_step(creep, road_edge, flags[dest], cost + (road_edge ? 1 : 2), budget);
                /* passable(dest) */
                uint8_t passable = ((flags[dest] & pass_need) == pass_need
                                    && (u->owner[dest] == 0 || u->owner[dest] == cp || pc_hidden_from(s, dest, cp)))
                    || ((hide || pc_kernels_peace_unit(s, owner, dest)) && (aquatic || (flags[dest] & MV_LAND) != 0)
                        && pc_kernels_hide_passable(s, owner, dest));
                if (passable && dest != source && newcost < dist[dest] && !pc_kernels_bridge_blocks(s, owner, at, dest)) {
                    pc_pathfinding_heap_push(heap, (int32_t)dest, newcost);
                    dist[dest] = newcost;
                    parent[dest] = (int32_t)at;
                }
            }
        }
    }
    ps->path_len = 0;
    if (found) {
        int32_t at = (int32_t)target;
        while (at >= 0 && (int)at != source) {
            ps->path[ps->path_len++] = (uint16_t)at;
            at = parent[at];
        }
    }
    return found;
}

/* pathfinding.path_details: the reference-ordered shortest path from `source` to `target` (Dijkstra with the
 * reference heap and its tie behaviour, neighbours in kernels' fixed order, movement data's passable/terminal/
 * road/budget rules). Fills `ps.path` (target first, source excluded) and returns whether the target was found;
 * the reference's mask is {source, target} plus the path and its rank is the index in `ps.path` (source: len).
 * PathFinder.GetPath with PathFinderSettings.CreateForUnit for the unit on `source`: A* over a sorted open list (ties
 * first in, first out), priority WorldCoordinates.Distance(tile, target) + cost as float32 (the start: Manhattan
 * distance), TileData.GetMovementCost in the game's units (10 a step, 5 road to road, 1000 for a step that ends the
 * move), no closed set (a tile is re-queued whenever its cost improves, or when unvisited), expansion while cost <
 * movement x 10, at most 1000 pops; neighbours in MapDataExtensions.GetArea order (x offset outer, y inner). Fills
 * `ps.path` (target first, source excluded) and returns whether the target was found; falls back to `search_ref`. */
PC_FN uint8_t pc_pathfinding_search(const PcState *s, int source, int target, const PcRules *r, const PcGeom *g,
                                    PcPathScratch *ps) {
    if (source == target) {
        ps->path_len = 0;
        return 1;
    }
    int n = s->n;
    int32_t size = s->size;
    const PcUnits *u = &s->units;
    int32_t cp = s->current_player;
    int32_t owner = u->owner[source];
    int kind = (int)u->kind[source];
    const uint8_t *sk = r->skills[kind];
    uint8_t aquatic = sk[PC_SK_FLOAT], creep = sk[PC_SK_CREEP];
    uint8_t no_zoc = creep || sk[PC_SK_INFILTRATE] || u->invisible[source];
    int32_t max_cost = (r->unit_movement[kind] + (int32_t)u->boosted[source]) * 10;
    pc_kernels_movement_flags(s, owner, g, ps->flags);
    const uint8_t *flags = ps->flags;
    uint8_t pass_need = aquatic ? MV_PASS : (MV_PASS | MV_LAND);
    uint8_t hide = sk[PC_SK_HIDE];
    int32_t tx = (int32_t)(target % s->size), ty = (int32_t)(target / s->size);
    int32_t *cost_so_far = ps->dist;
    int32_t *came_from = ps->parent;
    for (int i = 0; i < n; i++) { cost_so_far[i] = 0; came_from[i] = -1; }
    PcPathOpenList *open = &ps->open;
    pc_pathfinding_open_clear(open);
    int32_t sx = (int32_t)(source % s->size), sy = (int32_t)(source / s->size);
    pc_pathfinding_open_push(open, (float)(pc_abs(tx - sx) + pc_abs(ty - sy)), source);
    uint8_t found = 0;
    for (int it = 0; it < 1000; it++) {
        int at = pc_pathfinding_open_pop(open);
        if (at < 0) break;
        if (at == target) {
            found = 1;
            break;
        }
        int32_t here = cost_so_far[at];
        if (here < max_cost) {
            int32_t ax = (int32_t)(at % s->size), ay = (int32_t)(at / s->size);
            for (int32_t dx = -1; dx <= 1; dx++) {
                for (int32_t dy = -1; dy <= 1; dy++) {
                    int32_t x = ax + dx, y = ay + dy;
                    if ((dx == 0 && dy == 0) || x < 0 || y < 0 || x >= size || y >= size) {
                        continue;
                    }
                    int dest = (int)(y * size + x);
                    if (!pc_pathfinding_accessible(s, flags, source, dest, pass_need, cp, owner, hide, aquatic)
                        || pc_kernels_bridge_blocks(s, owner, at, dest)) {
                        continue;
                    }
                    int32_t next = pc_pathfinding_cost(flags, at, dest, aquatic, creep, no_zoc) + here;
                    int32_t old = cost_so_far[dest];
                    if (old < 1 || next < old) {
                        cost_so_far[dest] = next;
                        came_from[dest] = (int32_t)at;
                        /* dist(dest): sqrt((dy*dy + dx*dx) as f32), then + next as f32, all f32 */
                        int32_t ddx = tx - (int32_t)(dest % s->size), ddy = ty - (int32_t)(dest / s->size);
                        float d = sqrtf((float)(ddy * ddy + ddx * ddx));
                        pc_pathfinding_open_push(open, d + (float)next, dest);
                    }
                }
            }
        }
        if (pc_pathfinding_open_is_empty(open)) {
            break;
        }
    }
    ps->path_len = 0;
    if (found) {
        int at = target;
        for (int k = 0; k <= n; k++) {
            if (at == source) {
                return 1;
            }
            ps->path[ps->path_len++] = (uint16_t)at;
            int32_t c = came_from[at];
            if (c == -1) break;
            at = (int)c;
        }
    }
    return pc_pathfinding_search_ref(s, source, target, r, g, ps);
}

/* pathfinding.reveal_path into `ps.reveal` for the traveled tiles {source} + ps.path[from..] + extra: every tile
 * within 1 (2 from a mountain, or for scouts) of one of them, visiting only the boxes around them.
 * `extra`: -1 = None. */
PC_FN void pc_pathfinding_reveal_one(const PcState *s, int m, uint8_t scout, uint8_t *out) {
    int32_t size = s->size;
    int32_t radius = (s->tiles.terrain[m] == 4 || scout) ? 2 : 1;
    int32_t mx = (int32_t)(m % s->size), my = (int32_t)(m / s->size);
    for (int32_t y = pc_max(my - radius, 0); y <= pc_min(my + radius, size - 1); y++) {
        for (int32_t x = pc_max(mx - radius, 0); x <= pc_min(mx + radius, size - 1); x++) {
            out[y * size + x] = 1;
        }
    }
}
PC_FN void pc_pathfinding_reveal_tiles(const PcState *s, int source, int from, int extra, const PcRules *r,
                                       PcPathScratch *ps) {
    uint8_t scout = r->skills[(int)s->units.kind[source]][PC_SK_SCOUT];
    uint8_t *out = ps->reveal;
    for (int i = 0; i < s->n; i++) out[i] = 0;
    pc_pathfinding_reveal_one(s, source, scout, out);
    for (int k = from; k < ps->path_len; k++) pc_pathfinding_reveal_one(s, (int)ps->path[k], scout, out);
    if (extra >= 0) pc_pathfinding_reveal_one(s, extra, scout, out);
}

#ifdef __cplusplus
}
#endif
#endif /* PC_PATHFINDING_H */
