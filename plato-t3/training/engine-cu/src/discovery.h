/* discovery.h - line-by-line port of engine-rs/src/discovery.rs.
 *
 * The Explorer walk of the ruins and city-level Explorer rewards (game version 123): scout move action with the
 * fast path fog-distance search. 12 steps; each step scores an 11x11 window around the explorer, moves to the best
 * of its 8 neighbours (ties drawn from the step's hash) and explores radius 1 around it (2 on a mountain). */
#ifndef PC_DISCOVERY_H
#define PC_DISCOVERY_H

#include "pc.h"
#include "hash.h"     /* pc_hash_words, pc_hash_word (hash.rs hash_words / hash_word) */
#include "kernels.h"  /* pc_kernels_bridge_blocks */

#ifdef __cplusplus
extern "C" {
#endif

/* get number of moves for game version >= 120. */
#define PC_DISCOVERY_MOVES 12
/* fast path scan radius around the explorer (an 11x11 window). */
#define PC_DISCOVERY_RADIUS 5
#define PC_DISCOVERY_WIDTH (2 * PC_DISCOVERY_RADIUS + 1)
#define PC_DISCOVERY_CELLS (PC_DISCOVERY_WIDTH * PC_DISCOVERY_WIDTH)

/* lookup flags */
#define PC_DISCOVERY_VALID 1
#define PC_DISCOVERY_CAN_MOVE_OVER 2
#define PC_DISCOVERY_CAN_END_MOVE_ON 4
#define PC_DISCOVERY_FOG 8
#define PC_DISCOVERY_CHECK_FROM_TO 16
#define PC_DISCOVERY_LIGHTHOUSE 32
#define PC_DISCOVERY_MOUNTAIN 64
#define PC_DISCOVERY_READ_ONLY 128

/* Last queue index the fog-distance relaxation processes (fast path caps it). */
#define PC_DISCOVERY_QUEUE_LIMIT 0x1f2
/* The relaxation reads queue[at] only for at <= QUEUE_LIMIT, so entries pushed past that index are never read:
 * the fixed array stores the first QUEUE_LIMIT + 1 pushes and only counts the rest (the Rust Vec, hinted at
 * 4 * CELLS, keeps them all; the push order and the `at < queue.len()` test are unchanged). */
#define PC_DISCOVERY_QUEUE_CAP (PC_DISCOVERY_QUEUE_LIMIT + 1)

/* get unlocked movements: field, forest (and the unused 6/7) always, water with Fishing (Aquarion:
 * aqua base), ocean with Sailing (Aquarion: marine life), mountain with Climbing. `tech` is one seat's row (128 flags). */
PC_FN uint8_t pc_discovery_movable(const uint8_t *tech, int32_t terrain) {
    switch (terrain) {
        case 1: return (uint8_t)pc_kernels_water_move(tech);
        case 2: return (uint8_t)pc_kernels_ocean_move(tech);
        case 4: return tech[20];
        case 3: case 5: case 6: case 7: return 1;
        default: return 0;
    }
}

/* get score from surrounding fog tiles: fog tiles in the (clipped) square of radius `r` around `index`, capped
 * at 4; a fogged lighthouse in it lowers the score. Radius 2 with neither gives 9999. */
PC_FN int32_t pc_discovery_fog_score(const uint8_t *look, int index, int32_t r) {
    const int32_t W = PC_DISCOVERY_WIDTH;
    int32_t cx = (int32_t)index % W, cy = (int32_t)index / W;
    int32_t x0 = pc_max(cx - r, 0), x1 = pc_min(cx + r, W - 1);
    int32_t y0 = pc_max(cy - r, 0), y1 = pc_min(cy + r, W - 1);
    int32_t count = 0;
    uint8_t light = 0;
    for (int32_t y = y0; y <= y1; y++) {
        for (int32_t x = x0; x <= x1; x++) {
            uint8_t f = look[y * W + x];
            count += (f & PC_DISCOVERY_FOG) != 0;
            light |= (f & (PC_DISCOVERY_FOG | PC_DISCOVERY_LIGHTHOUSE)) == (PC_DISCOVERY_FOG | PC_DISCOVERY_LIGHTHOUSE);
        }
    }
    if (r == 2 && count == 0 && !light) {
        return 9999;
    }
    int32_t v = pc_min(count, 4);
    return light ? 40 - 10 * v : 73 - 10 * v;
}

/* Map tile of window cell `k` around (fx, fy), if it is on the board (-1 otherwise; Rust Option<usize>). */
PC_FN int pc_discovery_tile_of(int32_t size, int32_t fx, int32_t fy, int k) {
    int32_t x = fx - PC_DISCOVERY_RADIUS + (int32_t)k % PC_DISCOVERY_WIDTH;
    int32_t y = fy - PC_DISCOVERY_RADIUS + (int32_t)k / PC_DISCOVERY_WIDTH;
    return (x >= 0 && x < size && y >= 0 && y < size) ? (int)(y * size + x) : -1;
}

/* get next target default: the explorer's next tile from `from`, or -1 (None: the walk stops). `path`
 * is every tile walked so far (`path_len` of them), the start included. */
PC_FN int pc_discovery_next_step(const PcState *s, const uint8_t *explored, int from, const int *path, int path_len,
                                 uint32_t seed) {
    /* Candidate steps in the game's order, (dx, dy). */
    static const int32_t STEPS_DX[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static const int32_t STEPS_DY[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int32_t W = PC_DISCOVERY_WIDTH;
    const int32_t R = PC_DISCOVERY_RADIUS;
    int32_t size = s->size;
    int32_t fx = (int32_t)(from % s->size), fy = (int32_t)(from / s->size);
    int32_t cp = s->current_player;
    const uint8_t *tech = s->players.tech[pc_p(s)];
    /* populate lookup default */
    uint8_t look[PC_DISCOVERY_CELLS];
    for (int k = 0; k < PC_DISCOVERY_CELLS; k++) {
        look[k] = 0;
        int t = pc_discovery_tile_of(size, fx, fy, k);
        if (t < 0) continue;
        uint8_t seen = explored[t];
        int32_t structure = s->tiles.structure[t];
        int32_t terrain = s->tiles.terrain[t];
        uint8_t bridge = structure == 48, city = structure == 1;
        uint8_t enter = (bridge || city) ? city : pc_discovery_movable(tech, terrain);
        look[k] = (uint8_t)(PC_DISCOVERY_VALID
            | (bridge ? PC_DISCOVERY_CHECK_FROM_TO : 0)
            | (enter ? (PC_DISCOVERY_CAN_MOVE_OVER | PC_DISCOVERY_CAN_END_MOVE_ON) : 0)
            | (seen ? 0 : PC_DISCOVERY_FOG)
            | (structure == 47 ? PC_DISCOVERY_LIGHTHOUSE : 0)
            | ((terrain == 4 && seen) ? PC_DISCOVERY_MOUNTAIN : 0));
    }
    /* populate fog distance map default (check mountains: version != 115) */
    int32_t dist[PC_DISCOVERY_CELLS];
    int32_t queue[PC_DISCOVERY_QUEUE_CAP];
    int qlen = 0; /* queue.len(): every push counts, only the first QUEUE_CAP are stored (see QUEUE_CAP) */
    for (int k = 0; k < PC_DISCOVERY_CELLS; k++) {
        dist[k] = (look[k] & PC_DISCOVERY_CAN_MOVE_OVER) != 0 ? 1000 : 9999;
    }
    for (int k = 0; k < PC_DISCOVERY_CELLS; k++) {
        if ((look[k] & PC_DISCOVERY_FOG) != 0) {
            dist[k] = 0;
            if (qlen < PC_DISCOVERY_QUEUE_CAP) queue[qlen] = k;
            qlen++;
        }
    }
    for (int k = 0; k < PC_DISCOVERY_CELLS; k++) {
        if ((look[k] & (PC_DISCOVERY_MOUNTAIN | PC_DISCOVERY_FOG | PC_DISCOVERY_CAN_MOVE_OVER))
            == (PC_DISCOVERY_MOUNTAIN | PC_DISCOVERY_CAN_MOVE_OVER)) {
            int32_t score = pc_discovery_fog_score(look, k, 2);
            if (score != 9999) {
                dist[k] = score + 100;
                if (qlen < PC_DISCOVERY_QUEUE_CAP) queue[qlen] = k;
                qlen++;
                look[k] |= PC_DISCOVERY_READ_ONLY;
            }
        }
    }
    int at = 0;
    while (at < qlen) {
        int32_t u = queue[at];
        int32_t right = (u % W < W - 1) ? 1 : 9999;
        int32_t left = (u % W > 0) ? -1 : -9999;
        int32_t step = dist[u] + 100;
        int32_t cand[8];
        cand[0] = u + W; cand[1] = u - W; cand[2] = u + right; cand[3] = u + left;
        cand[4] = u + left + W; cand[5] = u + right + W; cand[6] = u + left - W; cand[7] = u + right - W;
        for (int c = 0; c < 8; c++) {
            int32_t index = cand[c];
            /* try update neighbour and enqueue */
            if (index < 0 || index >= PC_DISCOVERY_CELLS) {
                continue;
            }
            int i = (int)index;
            if (dist[i] <= step || (look[i] & PC_DISCOVERY_READ_ONLY) != 0) {
                continue;
            }
            /* can move from to (bridges): is tile accessible for a scout */
            uint8_t crosses = 0;
            if ((look[i] & PC_DISCOVERY_CHECK_FROM_TO) != 0) {
                int to = pc_discovery_tile_of(size, fx, fy, i);
                int fr = pc_discovery_tile_of(size, fx, fy, (int)u);
                crosses = (to >= 0 && fr >= 0) ? (explored[to] && !pc_kernels_bridge_blocks(s, cp, fr, to)) : 0;
            }
            if (!crosses && (look[i] & (PC_DISCOVERY_VALID | PC_DISCOVERY_CAN_MOVE_OVER))
                != (PC_DISCOVERY_VALID | PC_DISCOVERY_CAN_MOVE_OVER)) {
                continue;
            }
            dist[i] = (step == 100 && (look[i] & PC_DISCOVERY_FOG) == 0) ? pc_discovery_fog_score(look, i, 1) + 100 : step;
            if (qlen < PC_DISCOVERY_QUEUE_CAP) queue[qlen] = i;
            qlen++;
        }
        if (at == PC_DISCOVERY_QUEUE_LIMIT) {
            break;
        }
        at += 1;
    }
    /* apply backtracking penalty: 1, 2, 3, ... along the path, or 1, 1, 2, 2, ... once it has 9 tiles */
    uint8_t is_long = path_len > 8;
    int32_t penalty = 1;
    uint32_t count = 0;
    for (int k = 0; k < path_len; k++) {
        int t = path[k];
        int32_t lx = (int32_t)(t % s->size) - (fx - R), ly = (int32_t)(t / s->size) - (fy - R);
        if (lx >= 0 && lx < W && ly >= 0 && ly < W) {
            dist[ly * W + lx] += penalty;
        }
        penalty += is_long ? (int32_t)(count & 1u) : 1;
        count += (uint32_t)is_long;
    }
    int32_t best = INT32_MAX;
    int ties[8];
    int n = 0;
    for (int j = 0; j < 8; j++) {
        int32_t dx = STEPS_DX[j], dy = STEPS_DY[j];
        int32_t v = dist[(dy + R) * W + dx + R];
        if (v < best) {
            best = v;
            n = 0;
        }
        if (v == best) {
            ties[n] = j;
            n += 1;
        }
    }
    if (best > 9998) {
        return -1;
    }
    /* hash function range(0, count, seed) over random hash */
    int pick = n == 1 ? ties[0] : ties[(int)(pc_hash_word(s->initial_seed, seed) % (uint32_t)n)];
    int32_t dx = STEPS_DX[pick], dy = STEPS_DY[pick];
    int32_t x = fx + dx, y = fy + dy;
    return (x >= 0 && x < size && y >= 0 && y < size) ? (int)(y * size + x) : -1;
}

/* The tiles the current player newly explores by an Explorer reward on `source` (ruins: examine ruins action,
 * city: city reward action; both seed the walk with get hash(x, y) of the reward tile). game state seed is
 * our `initial_seed`. Rust Vec<bool> -> out[0..n). */
PC_FN void pc_discovery_explorer(const PcState *s, int source, const PcGeom *g, uint8_t *out) {
    int n = s->n;
    const uint8_t *initial = s->tiles.explored[pc_p(s)];
    uint8_t explored[PC_MAXN];
    for (int t = 0; t < n; t++) explored[t] = initial[t];
    uint32_t words[2];
    words[0] = (uint32_t)(source % s->size);
    words[1] = (uint32_t)(source / s->size);
    uint32_t seed = pc_hash_words(s->initial_seed, words, 2);
    int path[PC_DISCOVERY_MOVES + 1];
    int path_len = 0;
    path[path_len++] = source;
    int from = source;
    for (int m = 0; m < PC_DISCOVERY_MOVES; m++) {
        int target = pc_discovery_next_step(s, explored, from, path, path_len, seed);
        if (target < 0) break;
        path[path_len++] = target;
        /* explore from tile: radius 1, 2 from a mountain (Geom:for_ball, row-major over the box) */
        int radius = s->tiles.terrain[target] == 4 ? 2 : 1;
        int x0, x1, y0, y1;
        pc_ball(g, target, radius, &x0, &x1, &y0, &y1);
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) explored[y * g->size + x] = 1;
        }
        /* the next scout move action's seed: XXHash.get hash([seed]) */
        seed = pc_hash_word(s->initial_seed, seed);
        from = target;
    }
    for (int t = 0; t < n; t++) out[t] = explored[t] && !initial[t];
}

#ifdef __cplusplus
}
#endif
#endif /* PC_DISCOVERY_H */
