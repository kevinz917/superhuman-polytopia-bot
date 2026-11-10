/* pc.h - shared types of the single-source C/CUDA Polytopia engine (engine-cu).
 *
 * One header set, compiled twice: by a C compiler (CPU oracle driver, pc_cpu.c) and by nvcc (pc_cuda.cu). Every
 * engine function is `PC_FN` (static inline, __host__ __device__ under nvcc), allocates nothing, recurses never, and
 * works on fixed-size per-game storage. The reference is engine-rs (Rust, the real game's rules, 2026-11): every
 * module here is a line-by-line port of the Rust module of the same name, and the gate is bitwise equality of
 * step status, legal masks, packed views, rewards and every State field in lockstep (tests/test_gpu_engine.py), then
 * CPU == CUDA bitwise.
 *
 * Numeric discipline (both builds): f32 only, no fast-math, no FMA contraction (-ffp-contract=off / --fmad=false),
 * int32 arithmetic (wrap where Rust wraps explicitly), Rust cast semantics through the helpers below. */
#ifndef PC_H
#define PC_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#if defined(PC_PROFILE) && !defined(__CUDACC__)
#include <time.h>   /* clock_gettime (pc_cpu.c defines _POSIX_C_SOURCE) */
#endif

#ifdef __CUDACC__
#define PC_FN static inline __host__ __device__
#define PC_HD __host__ __device__
#else
#define PC_FN static inline
#define PC_HD
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- fixed shapes ---- */
/* Largest board side this build serves (11, 14, 18 are generated). The default build takes every board; a
 * `-DPC_MAXSIZE=14` build (Makefile SIZE=14 -> libpc_cuda_14.so) shrinks every per-game array to 196 tiles for the
 * 14x14 trainer (pc_create rejects larger boards). Results never depend on PC_MAXSIZE: every loop runs to the
 * runtime n, the fixed arrays only bound the storage (gated: the 14 variant against the same harness). */
#ifndef PC_MAXSIZE
#define PC_MAXSIZE 18
#endif
#define PC_MAXN (PC_MAXSIZE * PC_MAXSIZE)   /* 324 tiles (196 in the SIZE=14 build) */
#define PC_MAXP 4                           /* seats */
#define PC_NW ((PC_MAXN + 31) / 32)         /* u32 words of a per-tile bitset */
#define PC_HIST_TURNS 3                     /* state.rs HIST_TURNS */
#define PC_NO_ROUND_LIMIT INT32_MAX         /* mapgen.rs NO_ROUND_LIMIT = i32::MAX (exported as reference_max_turns) */

/* ---- 128-bit bitsets (Rust u128) ---- */
typedef struct { uint64_t lo, hi; } pc_u128;

PC_FN pc_u128 pc_u128_zero(void) { pc_u128 z; z.lo = 0; z.hi = 0; return z; }
PC_FN int pc_u128_test(pc_u128 b, int k) { return k < 64 ? (int)((b.lo >> k) & 1u) : (int)((b.hi >> (k - 64)) & 1u); }
PC_FN pc_u128 pc_u128_bit(int k) { pc_u128 b = pc_u128_zero(); if (k < 64) b.lo = 1ull << k; else b.hi = 1ull << (k - 64); return b; }
PC_FN pc_u128 pc_u128_or(pc_u128 a, pc_u128 b) { a.lo |= b.lo; a.hi |= b.hi; return a; }
PC_FN pc_u128 pc_u128_and(pc_u128 a, pc_u128 b) { a.lo &= b.lo; a.hi &= b.hi; return a; }
PC_FN pc_u128 pc_u128_not(pc_u128 a) { a.lo = ~a.lo; a.hi = ~a.hi; return a; }
PC_FN pc_u128 pc_u128_andnot(pc_u128 a, pc_u128 b) { a.lo &= ~b.lo; a.hi &= ~b.hi; return a; }
PC_FN int pc_u128_any(pc_u128 a) { return (a.lo | a.hi) != 0; }
PC_FN int pc_u128_eq(pc_u128 a, pc_u128 b) { return a.lo == b.lo && a.hi == b.hi; }
PC_FN void pc_u128_set(pc_u128 *b, int k) { if (k < 64) b->lo |= 1ull << k; else b->hi |= 1ull << (k - 64); }
PC_FN void pc_u128_clear(pc_u128 *b, int k) { if (k < 64) b->lo &= ~(1ull << k); else b->hi &= ~(1ull << (k - 64)); }
PC_FN int pc_popcount64(uint64_t x) {
#ifdef __CUDA_ARCH__
    return __popcll(x);
#else
    return __builtin_popcountll(x);
#endif
}
PC_FN int pc_u128_count(pc_u128 a) { return pc_popcount64(a.lo) + pc_popcount64(a.hi); }
/* index of the lowest set bit (128 when empty): Rust trailing_zeros */
PC_FN int pc_u128_lowest(pc_u128 a) {
    if (a.lo) {
#ifdef __CUDA_ARCH__
        return __ffsll((long long)a.lo) - 1;
#else
        return __builtin_ctzll(a.lo);
#endif
    }
    if (a.hi) {
#ifdef __CUDA_ARCH__
        return 64 + __ffsll((long long)a.hi) - 1;
#else
        return 64 + __builtin_ctzll(a.hi);
#endif
    }
    return 128;
}

/* ---- per-tile bitsets (Rust Vec<bool> of n entries where a bitset is cheaper) ---- */
typedef struct { uint32_t w[PC_NW]; } pc_tilebits;
PC_FN int pc_tb_get(const pc_tilebits *b, int i) { return (int)((b->w[i >> 5] >> (i & 31)) & 1u); }
PC_FN void pc_tb_set(pc_tilebits *b, int i, int v) {
    uint32_t m = 1u << (i & 31);
    if (v) b->w[i >> 5] |= m; else b->w[i >> 5] &= ~m;
}
PC_FN void pc_tb_clear(pc_tilebits *b) { for (int k = 0; k < PC_NW; k++) b->w[k] = 0; }

/* ---- Rust cast / arithmetic semantics ---- */
/* `x as i32` on f32: saturating, NaN -> 0 */
PC_FN int32_t pc_f2i(float x) {
    if (x != x) return 0;
    if (x >= 2147483648.0f) return INT32_MAX;
    if (x <= -2147483648.0f) return INT32_MIN;
    return (int32_t)x;
}
/* `x as i16` on f32: saturating, NaN -> 0 */
PC_FN int16_t pc_f2i16(float x) {
    if (x != x) return 0;
    if (x >= 32767.0f) return 32767;
    if (x <= -32768.0f) return -32768;
    return (int16_t)x;
}
PC_FN int32_t pc_min(int32_t a, int32_t b) { return a < b ? a : b; }
PC_FN int32_t pc_max(int32_t a, int32_t b) { return a > b ? a : b; }
PC_FN int32_t pc_clamp(int32_t x, int32_t lo, int32_t hi) { return x < lo ? lo : (x > hi ? hi : x); }
PC_FN float pc_fminf(float a, float b) { return a < b ? a : b; }
PC_FN float pc_fmaxf(float a, float b) { return a > b ? a : b; }
PC_FN float pc_fclamp(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
/* Rust i32::div_euclid / rem_euclid (b > 0) */
PC_FN int32_t pc_div_euclid(int32_t a, int32_t b) { int32_t q = a / b; if ((a % b) < 0) q -= 1; return q; }
PC_FN int32_t pc_rem_euclid(int32_t a, int32_t b) { int32_t r = a % b; return r < 0 ? r + b : r; }
PC_FN int32_t pc_wadd(int32_t a, int32_t b) { return (int32_t)((uint32_t)a + (uint32_t)b); }
PC_FN int32_t pc_wsub(int32_t a, int32_t b) { return (int32_t)((uint32_t)a - (uint32_t)b); }
PC_FN int32_t pc_wmul(int32_t a, int32_t b) { return (int32_t)((uint32_t)a * (uint32_t)b); }
PC_FN int32_t pc_abs(int32_t a) { return a < 0 ? -a : a; }

/* ---- board geometry (geom.rs) ---- */
typedef struct {
    int size, n;
    uint8_t dist[PC_MAXN * PC_MAXN];   /* Chebyshev distance */
    uint16_t nb[PC_MAXN * 8];          /* neighbour index, clipped into the board like the JAX table */
    uint8_t nb_valid[PC_MAXN * 8];
    uint16_t adj[PC_MAXN * 8];         /* in-board neighbours, compact; adj_n[i] of them, in k order */
    uint8_t adj_n[PC_MAXN];
    int corners[4];
} PcGeom;

/* ---- rules (rules.rs) ---- */
#define PC_SK_FLOAT 0
#define PC_SK_CARRY 1
#define PC_SK_SCOUT 2
#define PC_SK_CREEP 3
#define PC_SK_HIDE 4
#define PC_SK_INDEPENDENT 5
#define PC_SK_SURPRISE 6
#define PC_SK_SPLASH 7
#define PC_SK_STOMP 8
#define PC_SK_INFILTRATE 9

/* CostIndex: kinds whose cost <= stars, as a bitset; table[0] is for stars == min */
#define PC_COST_TABLE_MAX 160
typedef struct {
    int32_t min, len;
    pc_u128 table[PC_COST_TABLE_MAX];
} PcCostIndex;

typedef struct {
    int32_t cost[128];
    int32_t population[128];
    int32_t score[128];
    int32_t stars[128];
    uint8_t terrain[128][16];
    uint8_t adjacent[128][128];
    uint8_t any_adjacent[128];
    int32_t resource[128];
    uint8_t per_city[128];
    uint8_t per_tribe[128];
    uint8_t allowed[128];
    uint8_t unlock[128][128];          /* [tech][structure] */
    pc_u128 unlock_by[128];
    pc_u128 adjacent_rev[128];
    pc_u128 terrain_bits[16];
    pc_u128 no_adjacent_bits;
    pc_u128 per_city_bits;
    pc_u128 tech_unlocks[128];
    pc_u128 static_ok;
    pc_u128 per_tribe_bits;
    PcCostIndex cost_le;
} PcStructureRules;

typedef struct {
    float unit_health[128];
    float unit_attack[128];
    float unit_defense[128];
    int32_t unit_movement[128];
    int32_t unit_range[128];
    uint8_t dash[128];
    uint8_t unit_supported[128];
    int32_t tech_tier[128];
    uint8_t tech_edges[128][128];
    uint8_t tech_exists[128];
    int32_t unit_cost[128];
    uint8_t fortify[128];
    uint8_t stiff[128];
    uint8_t persist[128];
    uint8_t escape[128];
    uint8_t stat[128];
    int32_t unit_tech[128];
    PcStructureRules structures;
    uint8_t free_tech[128];
    int32_t free_requires[128];
    uint8_t skills[128][10];
    pc_u128 tech_pred[128];
    pc_u128 tech_closure[128];
    pc_u128 supported_bits;
    pc_u128 tech_exists_bits;
    pc_u128 tech_succ[128];
    pc_u128 units_by_tech[128];
    pc_u128 unit_static;
    PcCostIndex unit_cost_le;
    /* rules.rs tech_tree: Vec<(tech, Vec<structures>)> in table order; flattened */
    int32_t tech_tree_n;
    int32_t tech_tree_tech[128];
    int32_t tech_tree_struct_off[129];   /* structures of tree entry i: tech_tree_structs[off[i]..off[i+1]] */
    int32_t tech_tree_structs[1024];
} PcRules;

/* ---- one game's state (state.rs), fixed storage, runtime size n <= PC_MAXN and np <= PC_MAXP ---- */
typedef struct {
    int32_t terrain[PC_MAXN];
    int32_t owner[PC_MAXN];
    int32_t capital[PC_MAXN];
    int32_t city[PC_MAXN];
    uint8_t road[PC_MAXN];
    int32_t structure[PC_MAXN];
    int32_t resource[PC_MAXN];
    uint8_t explored[PC_MAXP][PC_MAXN];
    int32_t structure_level[PC_MAXN];
    int32_t structure_turn[PC_MAXN];
    uint8_t route[PC_MAXN];
} PcTiles;

typedef struct {
    int32_t owner[PC_MAXN];
    int32_t kind[PC_MAXN];
    float health[PC_MAXN];
    uint8_t veteran[PC_MAXN];
    uint8_t moved[PC_MAXN];
    uint8_t attacked[PC_MAXN];
    int32_t kills[PC_MAXN];
    uint8_t poison[PC_MAXN];
    uint8_t frozen[PC_MAXN];
    uint8_t boosted[PC_MAXN];
    int32_t home[PC_MAXN];
    int32_t created[PC_MAXN];
    uint8_t converted[PC_MAXN];
    int32_t attacks_performed[PC_MAXN];
    int32_t prev_x[PC_MAXN];
    int32_t prev_y[PC_MAXN];
    int32_t last_attack[PC_MAXN];
    int32_t passenger[PC_MAXN];
    uint8_t invisible[PC_MAXN];
    int32_t direction[PC_MAXN];
} PcUnits;

typedef struct {
    int32_t owner[PC_MAXN];
    int32_t level[PC_MAXN];
    int32_t population[PC_MAXN];
    int32_t progress[PC_MAXN];
    int32_t production[PC_MAXN];
    int32_t border[PC_MAXN];
    int32_t rewards[PC_MAXN * 9];
    pc_tilebits territory[PC_MAXN];    /* territory[city].bit(tile) == Rust territory[city * n + tile] */
    uint8_t connected[PC_MAXN];
    int32_t order[PC_MAXN];
    int32_t infiltrated[PC_MAXN];
    uint8_t riot[PC_MAXN];
} PcCities;

typedef struct {
    int32_t stars[PC_MAXP];
    int32_t score[PC_MAXP];
    uint8_t tech[PC_MAXP][128];
    int32_t tech_turn[PC_MAXP][128];
    uint8_t known[PC_MAXP][PC_MAXP];
    int32_t kills[PC_MAXP];
    int32_t casualties[PC_MAXP];
    int32_t pacifist[PC_MAXP];
    uint8_t attacked[PC_MAXP];
    uint8_t built[PC_MAXP][128];
    int32_t killed_turn[PC_MAXP];
    int32_t resigned_turn[PC_MAXP];
    int32_t killer[PC_MAXP];
    uint8_t peace[PC_MAXP][PC_MAXP];
    uint8_t embassy[PC_MAXP][PC_MAXP];
    uint8_t peace_offer[PC_MAXP][PC_MAXP];
    int32_t truce_breaker[PC_MAXP][PC_MAXP];
    int32_t hostile_until[PC_MAXP][PC_MAXP];
    int32_t peace_broken[PC_MAXP][PC_MAXP];
    uint8_t monuments[PC_MAXP][128];
    int32_t out_order[PC_MAXP];
} PcPlayers;

typedef struct {
    uint8_t seen_unit[PC_HIST_TURNS][PC_MAXN];
    int32_t prev_code[PC_MAXN];
    uint8_t built[PC_MAXN];
    uint8_t trained[PC_MAXN];
    pc_u128 enemy_tech;
} PcHistory;

typedef struct {
    int32_t size, n, np;
    PcTiles tiles;
    PcUnits units;
    PcCities cities;
    PcPlayers players;
    PcHistory history[PC_MAXP];
    int32_t current_player;
    int32_t turn;
    int32_t version;
    int32_t state_version;
    int32_t actions;
    uint8_t terminated;
    uint8_t truncated;
    uint8_t fog;
    int32_t reference_max_turns;
    uint8_t core_supported;
    uint32_t seed;
    uint32_t initial_seed;
    uint8_t monuments_settled;
} PcState;

/* ---- State helpers (state.rs impl State) ---- */
PC_FN int pc_p(const PcState *s) { return s->current_player - 1; }
PC_FN int pc_pclip(const PcState *s, int32_t owner) { return pc_clamp(owner - 1, 0, s->np - 1); }
/* JAX gather semantics for tech[owner - 1]: a negative index wraps to the last row */
PC_FN int pc_pwrap(const PcState *s, int32_t owner) {
    int32_t i = owner - 1;
    if (i < 0) i += s->np;
    return pc_clamp(i, 0, s->np - 1);
}
PC_FN int pc_alive(const PcState *s, int k) { return s->players.killed_turn[k] == 0 && s->players.resigned_turn[k] == 0; }
PC_FN int pc_alive_count(const PcState *s) { int c = 0; for (int k = 0; k < s->np; k++) c += pc_alive(s, k); return c; }
PC_FN int pc_foreign(int32_t owner, int32_t me) { return owner > 0 && owner != me; }
PC_FN int pc_hidden_from(const PcState *s, int i, int32_t viewer) {
    int32_t o = s->units.owner[i];
    return s->units.invisible[i] && pc_foreign(o, viewer) && !s->players.peace[pc_pclip(s, viewer)][pc_pclip(s, o)];
}
PC_FN int pc_territory(const PcState *s, int city, int tile) { return pc_tb_get(&s->cities.territory[city], tile); }
PC_FN void pc_set_territory(PcState *s, int city, int tile, int v) { pc_tb_set(&s->cities.territory[city], tile, v); }
PC_FN int32_t pc_reward(const PcState *s, int city, int k) { return s->cities.rewards[city * 9 + k]; }
PC_FN int32_t pc_reward_count(const PcState *s, int city) {
    int32_t t = 0;
    for (int k = 0; k < 9; k++) t += s->cities.rewards[city * 9 + k];
    return t;
}
PC_FN int pc_pending_reward(const PcState *s) {
    int32_t cp = s->current_player;
    for (int i = 0; i < s->n; i++)
        if (s->cities.owner[i] == cp && pc_reward_count(s, i) < s->cities.level[i] - 1) return 1;
    return 0;
}
/* Units::clear: every field of unit slot i to zero */
PC_FN void pc_units_clear(PcUnits *u, int i) {
    u->owner[i] = 0; u->kind[i] = 0; u->health[i] = 0.0f; u->veteran[i] = 0; u->moved[i] = 0; u->attacked[i] = 0;
    u->kills[i] = 0; u->poison[i] = 0; u->frozen[i] = 0; u->boosted[i] = 0; u->home[i] = 0; u->created[i] = 0;
    u->converted[i] = 0; u->attacks_performed[i] = 0; u->prev_x[i] = 0; u->prev_y[i] = 0; u->last_attack[i] = 0;
    u->passenger[i] = 0; u->invisible[i] = 0; u->direction[i] = 0;
}
/* Units::move_unit: slot s -> slot t, then clear s */
PC_FN void pc_units_move(PcUnits *u, int s, int t) {
    u->owner[t] = u->owner[s]; u->kind[t] = u->kind[s]; u->health[t] = u->health[s]; u->veteran[t] = u->veteran[s];
    u->moved[t] = u->moved[s]; u->attacked[t] = u->attacked[s]; u->kills[t] = u->kills[s]; u->poison[t] = u->poison[s];
    u->frozen[t] = u->frozen[s]; u->boosted[t] = u->boosted[s]; u->home[t] = u->home[s]; u->created[t] = u->created[s];
    u->converted[t] = u->converted[s]; u->attacks_performed[t] = u->attacks_performed[s]; u->prev_x[t] = u->prev_x[s];
    u->prev_y[t] = u->prev_y[s]; u->last_attack[t] = u->last_attack[s]; u->passenger[t] = u->passenger[s];
    u->invisible[t] = u->invisible[s]; u->direction[t] = u->direction[s];
    pc_units_clear(u, s);
}
PC_FN void pc_history_clear(PcHistory *h, int n) {
    for (int k = 0; k < PC_HIST_TURNS; k++) memset(h->seen_unit[k], 0, PC_MAXN);
    for (int i = 0; i < PC_MAXN; i++) h->prev_code[i] = -1;
    memset(h->built, 0, PC_MAXN);
    memset(h->trained, 0, PC_MAXN);
    h->enemy_tech = pc_u128_zero();
    (void)n;
}
/* State::empty_players */
PC_FN void pc_state_empty(PcState *s, int size, int np) {
    memset(s, 0, sizeof(*s));
    s->size = size;
    s->n = size * size;
    s->np = np;
    for (int i = 0; i < PC_MAXN; i++) s->tiles.city[i] = -1;
    for (int k = 0; k < PC_MAXP; k++) pc_history_clear(&s->history[k], s->n);
    s->current_player = 1;
    s->turn = 0;
    s->version = 115;
    s->state_version = 0;
    s->actions = 0;
    s->terminated = 0;
    s->truncated = 0;
    s->fog = 1;
    s->reference_max_turns = 45;
    s->core_supported = 1;
    s->seed = 0;
    s->initial_seed = 0;
    s->monuments_settled = 0;
}

/* ---- action (core.rs) ---- */
typedef struct { int32_t kind, source, target, parameter; } PcAction;

#define PC_OK 0
#define PC_INVALID_ACTION 1
#define PC_UNSUPPORTED_RULE 2
#define PC_STALE_VERSION 3
#define PC_EPISODE_DONE 4

/* ---- geometry construction (geom.rs Geom::new) ---- */
PC_FN void pc_geom_init(PcGeom *g, int size) {
    static const int DX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static const int DY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int n = size * size;
    g->size = size;
    g->n = n;
    for (int i = 0; i < n; i++) {
        int xi = i % size, yi = i / size;
        for (int j = 0; j < n; j++) {
            int xj = j % size, yj = j / size;
            int dx = pc_abs(xi - xj), dy = pc_abs(yi - yj);
            g->dist[i * n + j] = (uint8_t)(dx > dy ? dx : dy);
        }
    }
    for (int i = 0; i < n; i++) {
        int x = i % size, y = i / size, c = 0;
        for (int k = 0; k < 8; k++) {
            int nx = x + DX[k], ny = y + DY[k];
            int valid = nx >= 0 && nx < size && ny >= 0 && ny < size;
            int idx = pc_clamp(ny * size + nx, 0, n - 1);
            g->nb[i * 8 + k] = (uint16_t)idx;
            g->nb_valid[i * 8 + k] = (uint8_t)valid;
            if (valid) g->adj[i * 8 + c++] = (uint16_t)idx;
        }
        g->adj_n[i] = (uint8_t)c;
    }
    g->corners[0] = 0; g->corners[1] = size - 1; g->corners[2] = n - size; g->corners[3] = n - 1;
}
PC_FN int pc_d(const PcGeom *g, int i, int j) { return (int)g->dist[i * g->n + j]; }
/* Geom::for_ball bounds: tiles with d(center, t) <= radius are the rows y0..=y1, columns x0..=x1 (row-major) */
PC_FN void pc_ball(const PcGeom *g, int center, int radius, int *x0, int *x1, int *y0, int *y1) {
    int cx = center % g->size, cy = center / g->size;
    *x0 = pc_max(cx - radius, 0); *x1 = pc_min(cx + radius, g->size - 1);
    *y0 = pc_max(cy - radius, 0); *y1 = pc_min(cy + radius, g->size - 1);
    if (radius < 0) { *x1 = -1; *y1 = -1; *x0 = 0; *y0 = 0; }
}

/* ---- packed legal masks (masks.rs Packed sink: numpy.packbits order, buffer zeroed by the caller) ---- */
PC_FN void pc_mask_set(uint8_t *bits, int i) { bits[i >> 3] |= (uint8_t)(0x80u >> (i & 7)); }
PC_FN int pc_mask_get(const uint8_t *bits, int i) { return (bits[i >> 3] >> (7 - (i & 7))) & 1; }

/* ---- phase profile (compile-time: -DPC_PROFILE; Makefile PROFILE=1 -> libpc_*_prof.so) ----
 * The tick's phases are timed per game by the thread (lane 0 of the warp) that runs them: clock64 cycles on the
 * device, CLOCK_MONOTONIC nanoseconds on the host, accumulated into a per-game array of PC_PROFILE_PHASES counters
 * and reduced on the host by pc_profile_read (pc_api.h). Without PC_PROFILE every lap is a no-op and the structs
 * carry no profile pointer, so the production kernels are untouched. */
#define PC_PH_STEP 0            /* decode + step_with */
#define PC_PH_OUTPUTS 1         /* flags, terminal rewards, scores */
#define PC_PH_FINAL_VIEWS 2     /* every seat's view of a truncated final state */
#define PC_PH_RESET 3           /* initial_state of a finished game */
#define PC_PH_MASK_PRE 4        /* masks: active / pending peace / reward rows */
#define PC_PH_MASK_MOVE 5       /* masks: move rows (movement_flags + reach per unit) */
#define PC_PH_MASK_ATTACK 6
#define PC_PH_MASK_SUMMON 7
#define PC_PH_MASK_BUILD 8      /* masks: BuildCtx::prepare + build_row per economy tile */
#define PC_PH_MASK_RESEARCH 9
#define PC_PH_MASK_ECONOMY 10   /* harvest / forest / destroy rows */
#define PC_PH_MASK_UNIT 11      /* capture / recover / promote / disband / heal rows */
#define PC_PH_MASK_DIPLO 12     /* diplomacy blocks, end, resign */
#define PC_PH_LEGAL_COPY 13     /* the packed row into the legal cache */
#define PC_PH_OBSERVE_MOVER 14
#define PC_PH_OBSERVE_OTHERS 15
#define PC_PROFILE_PHASES 16
typedef unsigned long long pc_prof_t;   /* one phase counter: device clock64 cycles, host nanoseconds */
#ifdef PC_PROFILE
PC_FN pc_prof_t pc_prof_now(void) {
#if defined(__CUDA_ARCH__)
    return (pc_prof_t)clock64();
#elif defined(__CUDACC__)
    return 0;   /* the host side of the CUDA build never times a phase */
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (pc_prof_t)ts.tv_sec * 1000000000ull + (pc_prof_t)ts.tv_nsec;
#endif
}
/* acc[k] += now - t; t = now (acc may be NULL: no profile sink for this call) */
#define PC_PROF_LAP(acc, t, k) do { pc_prof_t pc_prof_n_ = pc_prof_now(); if (acc) (acc)[k] += pc_prof_n_ - (t); (t) = pc_prof_n_; } while (0)
#define PC_PROF_START(t) pc_prof_t t = pc_prof_now()
#else
#define PC_PROF_LAP(acc, t, k) ((void)0)
#define PC_PROF_START(t) ((void)0)
#endif

/* ---- lanes: one game run by `count` cooperating threads (2026-11-06) ----
 * The serial engine runs one game per thread (count = 1: lane 0 of 1, every helper below is the plain serial
 * operation). The warp-per-game tick (pc_cuda.cu k_tick_warp, PC_CUDA_WARP_TICK=1) runs one game per warp (count =
 * 32): code outside the per-tile loops runs redundantly on every lane (uniform: same inputs, same values) with its
 * writes made by lane 0, and a per-tile loop `for (i = 0; i < n; i++)` becomes `for (i = L.lane; i < n; i +=
 * L.count)`. A loop with a warp collective inside (any / ballot / reductions) walks chunks of `count` tiles so every
 * lane makes the same number of calls. Mask bits are a set, so the order lanes set them in is free; ordered lists
 * (BuildCtx's econ_list, own_cities) are compacted with a ballot so they keep the serial order exactly. */
typedef struct { int lane, count; } PcLanes;
PC_FN PcLanes pc_lanes_one(void) { PcLanes L; L.lane = 0; L.count = 1; return L; }
PC_FN void pc_lanes_sync(PcLanes L) {
#ifdef __CUDA_ARCH__
    if (L.count > 1) __syncwarp();
#endif
    (void)L;
}
PC_FN int pc_lanes_any(PcLanes L, int x) {
#ifdef __CUDA_ARCH__
    if (L.count > 1) return __any_sync(0xffffffffu, x) != 0;
#endif
    (void)L;
    return x != 0;
}
/* bit k = lane k's predicate (count = 1: bit 0) */
PC_FN uint32_t pc_lanes_ballot(PcLanes L, int x) {
#ifdef __CUDA_ARCH__
    if (L.count > 1) return __ballot_sync(0xffffffffu, x);
#endif
    (void)L;
    return x ? 1u : 0u;
}
/* index of the lowest set bit of a nonzero word */
PC_FN int pc_lowest32(uint32_t m) {
#ifdef __CUDA_ARCH__
    return __ffs((int)m) - 1;
#else
    return __builtin_ctz(m);
#endif
}
/* every lane's x summed (wrapping int32: the same value as the serial sum in any order) */
PC_FN int32_t pc_lanes_sum(PcLanes L, int32_t x) {
#ifdef __CUDA_ARCH__
    if (L.count > 1) {
        for (int off = 16; off > 0; off >>= 1) x = pc_wadd(x, __shfl_xor_sync(0xffffffffu, x, off));
    }
#endif
    (void)L;
    return x;
}
PC_FN pc_u128 pc_lanes_or128(PcLanes L, pc_u128 v) {
#ifdef __CUDA_ARCH__
    if (L.count > 1) {
        for (int off = 16; off > 0; off >>= 1) {
            v.lo |= __shfl_xor_sync(0xffffffffu, v.lo, off);
            v.hi |= __shfl_xor_sync(0xffffffffu, v.hi, off);
        }
    }
#endif
    (void)L;
    return v;
}
/* Ordered stream compaction over one chunk: every lane calls it with its predicate; returns this lane's slot (valid
 * when pred) = *count + the number of lower lanes with pred, then advances *count past the chunk's entries. */
PC_FN int pc_lanes_compact(PcLanes L, int pred, int *count) {
    uint32_t m = pc_lanes_ballot(L, pred);
    int pos = *count + pc_popcount64((uint64_t)(m & ((1u << L.lane) - 1u)));
    *count += pc_popcount64((uint64_t)m);
    return pos;
}
/* *p += v from any lane (count = 1: plain add) */
PC_FN void pc_lanes_add(PcLanes L, int32_t *p, int32_t v) {
#ifdef __CUDA_ARCH__
    if (L.count > 1) { atomicAdd((int *)p, (int)v); return; }
#endif
    (void)L;
    *p += v;
}
/* pc_mask_set from any lane: with count > 1 an atomic OR on the 32-bit word holding the byte (little-endian), so
 * `bits` must be 4-byte aligned and its words this game's alone (pc_cuda.cu's legal cache rows are). */
PC_FN void pc_lanes_mask_set(PcLanes L, uint8_t *bits, int i) {
#ifdef __CUDA_ARCH__
    if (L.count > 1) {
        int b = i >> 3;
        atomicOr((unsigned int *)(bits + (b & ~3)), (0x80u >> (i & 7)) << (8 * (b & 3)));
        return;
    }
#endif
    (void)L;
    pc_mask_set(bits, i);
}

#ifdef __cplusplus
}
#endif
#endif /* PC_H */
