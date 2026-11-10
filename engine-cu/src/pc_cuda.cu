/* pc_cuda.cu - CUDA driver of the single-source Polytopia engine: the C ABI of pc_api.h over engine.h with one thread
 * per game (Netherite's k_tick shape: the serial game logic runs unchanged on a thread; 32 threads per block spread the
 * latency-bound warps across the SMs). Gated bitwise against pc_cpu.c on the same harness.
 *
 * Two tick kernels, chosen at pc_create (PC_CUDA_WARP_TICK=1 in the environment selects the second):
 *   k_tick       one thread per game (the serial engine on each thread);
 *   k_tick_warp  one warp per game (2026-11-06): lane 0 runs the serial step and reset, then the 32 lanes run the
 *                masks and the views together (pc.h PcLanes: per-tile loops strided over the lanes, per-unit move and
 *                attack rows one unit per lane, mask bits set with atomic ORs into the word-aligned legal cache row,
 *                then copied to the output row). Same outputs bit for bit (the gate runs both).
 * A -DPC_PROFILE build (Makefile PROFILE=1) times the phases per game with clock64 (pc_profile_read).
 *
 * Memory: PcState, PcScratch and PcMaskScratch per game on the device, PcRules and PcGeom once, the legal cache
 * (last tick's packed masks) per game. PcTickOut.device_ptrs = 1 means every pointer the caller passes to pc_tick
 * (inputs AND outputs) is device memory and nothing is staged; 0 means host memory, staged through pinned buffers
 * (the harness). Every API call synchronizes the batch's stream before returning. */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "pc_api.h"
#include "pc_state_io.h"

#define PC_TPB 32
#define PC_WARP_TPB_DEFAULT 128   /* k_tick_warp: 4 games (warps) per block; PC_CUDA_WARP_TPB overrides */
/* -DPC_WARP_MIN_BLOCKS=k (NVEXTRA): __launch_bounds__(128, k) on k_tick_warp, i.e. a register cap so k blocks of 4 games
 * fit an SM (ptxas then spills instead); PC_CUDA_WARP_TPB is held to <= 128 in such a build */
#ifdef PC_WARP_MIN_BLOCKS
#define PC_WARP_BOUNDS __launch_bounds__(PC_WARP_TPB_DEFAULT, PC_WARP_MIN_BLOCKS)
#define PC_WARP_TPB_MAX PC_WARP_TPB_DEFAULT
#else
#define PC_WARP_BOUNDS
#define PC_WARP_TPB_MAX 1024
#endif

struct PcBatch {
    int size, n, np, games;
    int32_t max_turns;
    int no_resign, no_peace;
    int32_t max_actions;
    int device;
    int rules_set;
    int mask_bytes, vw;
    cudaStream_t stream;
    PcRules *d_rules;
    PcGeom *d_geom;
    PcState *d_states;         /* [G] */
    PcScratch *d_scratch;      /* [G] */
    PcMaskScratch *d_mscratch; /* [G] */
    uint8_t *d_legal;          /* [G, legal_stride]: rows rounded up to whole 32-bit words (the warp tick ORs words) */
    uint8_t *d_legal_valid;    /* [G] */
    /* host-mode staging (allocated lazily, pinned) */
    int32_t *d_in;             /* family, source, target, seeds: [4, G] */
    int32_t *d_layout;         /* terrain, structure, resource [3, G, n] then caps [G, np] */
    uint8_t *d_masks;          /* [G, mask_bytes] */
    int16_t *d_view;           /* [G, vw] */
    int16_t *d_opp;            /* [G, np-1, vw] */
    int16_t *d_final;          /* [G, np, vw] */
    uint8_t *d_flags;          /* done, terminated, truncated: [3, G] */
    float *d_rewards;          /* [G, np] */
    int32_t *d_status;         /* [G] */
    int32_t *d_scores;         /* [G, np] */
    PcState *h_state;          /* one host PcState for export / import */
    int legal_stride;          /* bytes per legal-cache row: mask_bytes rounded up to a multiple of 4 */
    int warp_tick;             /* 1: k_tick_warp (one warp per game), 0: k_tick (one thread per game) */
    int warp_tpb;              /* threads per block of k_tick_warp (a multiple of 32) */
    pc_prof_t *d_prof;         /* [G, PC_PROFILE_PHASES] phase cycles (profile builds only) */
    uint64_t prof_ticks, prof_ns;
    int clock_khz;
    cudaEvent_t ev0, ev1;      /* profile builds: k_tick bracket */
};

static int cu_ok(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return 1;
    fprintf(stderr, "pc_cuda: %s: %s\n", what, cudaGetErrorString(e));
    return 0;
}

/* ---- kernels ---- */
__global__ void k_init(PcState *states, PcScratch *scratch, PcMaskScratch *mscratch, int games, int size, int np, int n) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= games) return;
    pc_state_empty(&states[g], size, np);
    PcScratch *sc = &scratch[g];
    memset(sc, 0, sizeof(PcScratch));
    pc_kernels_move_scratch_init(&sc->mv);
    pc_structures_build_ctx_init(&sc->build);
    pc_military_summon_ctx_new(&sc->summon);
    pc_pathfinding_scratch_init(&sc->path, n);
    PcMaskScratch *ms = &mscratch[g];
    memset(ms, 0, sizeof(PcMaskScratch));
    pc_kernels_move_scratch_init(&ms->mv);
    pc_structures_build_ctx_init(&ms->build);
    pc_military_summon_ctx_new(&ms->summon);
}

__global__ void k_reset(PcState *states, int games, const int32_t *seeds, const int32_t *terrain, const int32_t *structure,
                        const int32_t *resource, const int32_t *caps, int n, int np, const PcGeom *geom, int32_t max_turns,
                        uint8_t *legal_valid) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= games) return;
    pc_core_initial_state(&states[g], (uint32_t)seeds[g], terrain + (size_t)g * n, structure + (size_t)g * n,
                          resource + (size_t)g * n, caps + (size_t)g * np, np, geom, max_turns);
    legal_valid[g] = 0;
}

__device__ static int other_seat(const PcState *s, int k) { return (s->current_player - 1 + k) % s->np + 1; }

/* lib.rs::tick per game; masks and final_view rows are zeroed by the caller (cudaMemsetAsync). ls: legal-cache row
 * stride (bytes); prof: [G, PC_PROFILE_PHASES] counters (profile builds) or NULL. */
__global__ void k_tick(PcState *states, PcScratch *scratch, PcMaskScratch *mscratch, int games, const PcRules *rules,
                       const PcGeom *geom, int32_t max_actions, int no_resign, int no_peace, int32_t max_turns,
                       uint8_t *legal, uint8_t *legal_valid, int mb, int ls, int vw,
                       const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
                       const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
                       int both_views, uint8_t *masks, int16_t *view, int16_t *opp_view, int16_t *final_view,
                       uint8_t *done, uint8_t *terminated, uint8_t *truncated, float *rewards, int32_t *status, int32_t *scores,
                       pc_prof_t *prof) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= games) return;
    pc_prof_t *acc = prof ? prof + (size_t)g * PC_PROFILE_PHASES : NULL;
    (void)acc;
    PC_PROF_START(t_prof);
    const int np = states[g].np;
    const int n = states[g].n;
    const int others = np - 1;
    PcState *st = &states[g];
    PcAction a = pc_core_decode(st, family[g], source[g], target[g]);
    int32_t v = st->state_version;
    const uint8_t *lg = legal_valid[g] ? legal + (size_t)g * ls : NULL;
    status[g] = pc_core_step_with(st, a, v, rules, geom, max_actions, &scratch[g], lg);
    PC_PROF_LAP(acc, t_prof, PC_PH_STEP);
    legal_valid[g] = 0;
    terminated[g] = st->terminated;
    truncated[g] = st->truncated;
    done[g] = (uint8_t)(st->terminated || st->truncated);
    float rw[PC_MAXP];
    pc_core_terminal_rewards(st, rw);
    for (int k = 0; k < np; k++) rewards[(size_t)g * np + k] = rw[k];
    if (scores) for (int k = 0; k < np; k++) scores[(size_t)g * np + k] = st->players.score[k];
    PC_PROF_LAP(acc, t_prof, PC_PH_OUTPUTS);
    if (st->truncated) {
        for (int seat = 0; seat < np; seat++)
            pc_kernels_observe(st, seat + 1, geom, final_view + ((size_t)g * np + seat) * vw);
    }
    PC_PROF_LAP(acc, t_prof, PC_PH_FINAL_VIEWS);
    if (done[g]) {
        pc_core_initial_state(st, (uint32_t)seeds[g], terrain + (size_t)g * n, structure + (size_t)g * n,
                              resource + (size_t)g * n, caps + (size_t)g * np, np, geom, max_turns);
    }
    PC_PROF_LAP(acc, t_prof, PC_PH_RESET);
    uint8_t *row = masks + (size_t)g * mb;
    pc_masks_into_lanes(st, rules, geom, (uint8_t)no_resign, (uint8_t)no_peace, &mscratch[g], row, pc_lanes_one(), acc);
    PC_PROF_START(t_copy);
    memcpy(legal + (size_t)g * ls, row, (size_t)mb);
    legal_valid[g] = 1;
    PC_PROF_LAP(acc, t_copy, PC_PH_LEGAL_COPY);
    pc_kernels_observe(st, st->current_player, geom, view + (size_t)g * vw);
    PC_PROF_LAP(acc, t_copy, PC_PH_OBSERVE_MOVER);
    if (both_views) {
        for (int k = 1; k <= others; k++)
            pc_kernels_observe(st, other_seat(st, k), geom, opp_view + ((size_t)g * others + (k - 1)) * vw);
    }
    PC_PROF_LAP(acc, t_copy, PC_PH_OBSERVE_OTHERS);
}

/* k_tick with one warp per game (game = global warp index): the same outputs as k_tick, bit for bit. Lane 0 runs
 * decode, step_with, the outputs and the reset (the serial game logic, unchanged); every lane then runs the masks
 * (into the word-aligned legal-cache row, which the warp zeroes first: pc_lanes_mask_set ORs whole words) and the
 * views (pc_kernels_observe_lanes), and the legal row is copied to the output row byte for byte. Lane syncs order
 * lane 0's state writes before the other lanes read the state. Profile laps are lane 0's, taken after a sync. */
__global__ void PC_WARP_BOUNDS k_tick_warp(PcState *states, PcScratch *scratch, PcMaskScratch *mscratch, int games, const PcRules *rules,
                            const PcGeom *geom, int32_t max_actions, int no_resign, int no_peace, int32_t max_turns,
                            uint8_t *legal, uint8_t *legal_valid, int mb, int ls, int vw,
                            const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
                            const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
                            int both_views, uint8_t *masks, int16_t *view, int16_t *opp_view, int16_t *final_view,
                            uint8_t *done, uint8_t *terminated, uint8_t *truncated, float *rewards, int32_t *status,
                            int32_t *scores, pc_prof_t *prof) {
    int g = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (g >= games) return;   /* whole warps only: the game index is the warp's */
    PcLanes L;
    L.lane = (int)(threadIdx.x & 31u);
    L.count = 32;
    pc_prof_t *acc = (prof && L.lane == 0) ? prof + (size_t)g * PC_PROFILE_PHASES : NULL;
    (void)acc;
    PC_PROF_START(t_prof);
    PcState *st = &states[g];
    const int np = st->np;
    const int n = st->n;
    const int others = np - 1;
    if (L.lane == 0) {
        PcAction a = pc_core_decode(st, family[g], source[g], target[g]);
        int32_t v = st->state_version;
        const uint8_t *lg = legal_valid[g] ? legal + (size_t)g * ls : NULL;
        status[g] = pc_core_step_with(st, a, v, rules, geom, max_actions, &scratch[g], lg);
        PC_PROF_LAP(acc, t_prof, PC_PH_STEP);
        legal_valid[g] = 0;
        terminated[g] = st->terminated;
        truncated[g] = st->truncated;
        done[g] = (uint8_t)(st->terminated || st->truncated);
        float rw[PC_MAXP];
        pc_core_terminal_rewards(st, rw);
        for (int k = 0; k < np; k++) rewards[(size_t)g * np + k] = rw[k];
        if (scores) for (int k = 0; k < np; k++) scores[(size_t)g * np + k] = st->players.score[k];
    }
    __syncwarp();
    if (L.lane == 0) PC_PROF_LAP(acc, t_prof, PC_PH_OUTPUTS);
    const int was_truncated = st->truncated;
    const int finished = st->terminated || st->truncated;
    if (was_truncated) {
        for (int seat = 0; seat < np; seat++)
            pc_kernels_observe_lanes(st, seat + 1, geom, final_view + ((size_t)g * np + seat) * vw, L);
    }
    __syncwarp();   /* every lane has read the final state before lane 0 resets it */
    if (L.lane == 0) PC_PROF_LAP(acc, t_prof, PC_PH_FINAL_VIEWS);
    if (finished && L.lane == 0) {
        pc_core_initial_state(st, (uint32_t)seeds[g], terrain + (size_t)g * n, structure + (size_t)g * n,
                              resource + (size_t)g * n, caps + (size_t)g * np, np, geom, max_turns);
    }
    uint8_t *lrow = legal + (size_t)g * ls;
    for (int w = L.lane; w < ls / 4; w += 32) ((uint32_t *)lrow)[w] = 0u;
    __syncwarp();
    if (L.lane == 0) PC_PROF_LAP(acc, t_prof, PC_PH_RESET);
    pc_masks_into_lanes(st, rules, geom, (uint8_t)no_resign, (uint8_t)no_peace, &mscratch[g], lrow, L, acc);
    __syncwarp();
    PC_PROF_START(t_copy);
    uint8_t *row = masks + (size_t)g * mb;
    for (int k = L.lane; k < mb; k += 32) row[k] = lrow[k];
    if (L.lane == 0) legal_valid[g] = 1;
    __syncwarp();
    if (L.lane == 0) PC_PROF_LAP(acc, t_copy, PC_PH_LEGAL_COPY);
    pc_kernels_observe_lanes(st, st->current_player, geom, view + (size_t)g * vw, L);
    __syncwarp();
    if (L.lane == 0) PC_PROF_LAP(acc, t_copy, PC_PH_OBSERVE_MOVER);
    if (both_views) {
        for (int k = 1; k <= others; k++)
            pc_kernels_observe_lanes(st, other_seat(st, k), geom, opp_view + ((size_t)g * others + (k - 1)) * vw, L);
    }
    __syncwarp();
    if (L.lane == 0) PC_PROF_LAP(acc, t_copy, PC_PH_OBSERVE_OTHERS);
}

__global__ void k_masks(const PcState *states, PcMaskScratch *mscratch, int games, const PcRules *rules, const PcGeom *geom,
                        int no_resign, int no_peace, int mb, uint8_t *packed) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= games) return;
    pc_masks_into(&states[g], rules, geom, (uint8_t)no_resign, (uint8_t)no_peace, &mscratch[g], packed + (size_t)g * mb);
}

__global__ void k_observe(const PcState *states, int games, const PcGeom *geom, int player, int vw, int16_t *view) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= games) return;
    const PcState *st = &states[g];
    pc_kernels_observe(st, player == 0 ? st->current_player : player, geom, view + (size_t)g * vw);
}

/* ---- host driver ---- */
static unsigned blocks(int games) { return (unsigned)((games + PC_TPB - 1) / PC_TPB); }

/* Several batches on several GPUs in one process (the data-parallel trainer): a kernel launch needs the stream's device
 * to be the thread's current device, so every API call makes the batch's device current and restores the caller's
 * (torch and XLA keep their own notion of the current device; a changed one would misplace their allocations). */
static int enter_device(const PcBatch *b, int *prev) {
    int dev = b->device < 0 ? 0 : b->device;
    if (cudaGetDevice(prev) != cudaSuccess) { *prev = dev; return PC_E_DEVICE; }
    if (*prev != dev && !cu_ok(cudaSetDevice(dev), "cudaSetDevice")) return PC_E_DEVICE;
    return 0;
}

static void leave_device(int prev) {
    int cur;
    if (cudaGetDevice(&cur) == cudaSuccess && cur != prev) cudaSetDevice(prev);
}

static int sync_check(PcBatch *b, const char *what) {
    cudaError_t e = cudaGetLastError();
    if (!cu_ok(e, what)) return PC_E_DEVICE;
    e = cudaStreamSynchronize(b->stream);
    if (!cu_ok(e, what)) return PC_E_DEVICE;
    return 0;
}

static PcBatch *pc_create_on(int size, int games, int players, int max_turns, int no_resign, int no_peace, int max_actions,
                             int device);

extern "C" PcBatch *pc_create(int size, int games, int players, int max_turns, int no_resign, int no_peace, int max_actions,
                              int device) {
    if (!(size == 11 || size == 14 || size == 18) || games < 1 || players < 2 || players > PC_MAXP) return NULL;
    int prev_device = 0;
    cudaGetDevice(&prev_device);
    PcBatch *b = pc_create_on(size, games, players, max_turns, no_resign, no_peace, max_actions, device);
    leave_device(prev_device);
    return b;
}

static PcBatch *pc_create_on(int size, int games, int players, int max_turns, int no_resign, int no_peace, int max_actions,
                             int device) {
    if (size > PC_MAXSIZE) return NULL;   /* a PC_MAXSIZE=14 build (libpc_cuda_14.so) takes 11 and 14 only */
    if (!cu_ok(cudaSetDevice(device < 0 ? 0 : device), "cudaSetDevice")) return NULL;
    /* the serial game logic is deep: raise the per-thread stack (default 1 KB); PC_CUDA_STACK_KB overrides */
    const char *env = getenv("PC_CUDA_STACK_KB");
    size_t stack_kb = env ? (size_t)atoi(env) : 32;   /* ptxas: k_tick frame 26,624 B (2026-11-06) */
    if (!cu_ok(cudaDeviceSetLimit(cudaLimitStackSize, stack_kb * 1024), "cudaDeviceSetLimit(stack)")) return NULL;
    PcBatch *b = (PcBatch *)calloc(1, sizeof(PcBatch));
    if (!b) return NULL;
    b->size = size; b->n = size * size; b->np = players; b->games = games;
    b->max_turns = max_turns <= 0 ? PC_NO_ROUND_LIMIT : max_turns;
    b->no_resign = no_resign != 0; b->no_peace = no_peace != 0;
    b->max_actions = max_actions; b->device = device;
    b->mask_bytes = pc_masks_packed_width(b->n);
    b->legal_stride = (b->mask_bytes + 3) & ~3;
    b->vw = pc_kernels_view_width_np(b->n, b->np);
    {
        const char *w = getenv("PC_CUDA_WARP_TICK");
        b->warp_tick = w && atoi(w) != 0;
        const char *tpb = getenv("PC_CUDA_WARP_TPB");
        int t = tpb ? atoi(tpb) : PC_WARP_TPB_DEFAULT;
        b->warp_tpb = (t >= 32 && t <= PC_WARP_TPB_MAX && t % 32 == 0) ? t : PC_WARP_TPB_DEFAULT;
    }
    size_t G = (size_t)games;
    int ok = cu_ok(cudaStreamCreateWithFlags(&b->stream, cudaStreamNonBlocking), "stream")   /* no implicit sync with the legacy default stream (torch, XLA) */
        && cu_ok(cudaMalloc(&b->d_rules, sizeof(PcRules)), "rules")
        && cu_ok(cudaMalloc(&b->d_geom, sizeof(PcGeom)), "geom")
        && cu_ok(cudaMalloc(&b->d_states, G * sizeof(PcState)), "states")
        && cu_ok(cudaMalloc(&b->d_scratch, G * sizeof(PcScratch)), "scratch")
        && cu_ok(cudaMalloc(&b->d_mscratch, G * sizeof(PcMaskScratch)), "mscratch")
        && cu_ok(cudaMalloc(&b->d_legal, G * (size_t)b->legal_stride), "legal")
        && cu_ok(cudaMalloc(&b->d_legal_valid, G), "legal_valid")
        && cu_ok(cudaMalloc(&b->d_in, 4 * G * sizeof(int32_t)), "in")
        && cu_ok(cudaMalloc(&b->d_layout, (3 * G * (size_t)b->n + G * (size_t)b->np) * sizeof(int32_t)), "layout")
        && cu_ok(cudaMalloc(&b->d_masks, G * (size_t)b->mask_bytes), "masks")
        && cu_ok(cudaMalloc(&b->d_view, G * (size_t)b->vw * sizeof(int16_t)), "view")
        && cu_ok(cudaMalloc(&b->d_opp, G * (size_t)(b->np - 1) * (size_t)b->vw * sizeof(int16_t)), "opp")
        && cu_ok(cudaMalloc(&b->d_final, G * (size_t)b->np * (size_t)b->vw * sizeof(int16_t)), "final")
        && cu_ok(cudaMalloc(&b->d_flags, 3 * G), "flags")
        && cu_ok(cudaMalloc(&b->d_rewards, G * (size_t)b->np * sizeof(float)), "rewards")
        && cu_ok(cudaMalloc(&b->d_status, G * sizeof(int32_t)), "status")
        && cu_ok(cudaMalloc(&b->d_scores, G * (size_t)b->np * sizeof(int32_t)), "scores")
        && cu_ok(cudaMallocHost((void **)&b->h_state, sizeof(PcState)), "h_state");
    if (!ok) { pc_destroy(b); return NULL; }
#ifdef PC_PROFILE
    ok = cu_ok(cudaMalloc(&b->d_prof, G * PC_PROFILE_PHASES * sizeof(pc_prof_t)), "prof")
        && cu_ok(cudaMemset(b->d_prof, 0, G * PC_PROFILE_PHASES * sizeof(pc_prof_t)), "prof zero")
        && cu_ok(cudaEventCreate(&b->ev0), "event") && cu_ok(cudaEventCreate(&b->ev1), "event")
        && cu_ok(cudaDeviceGetAttribute(&b->clock_khz, cudaDevAttrClockRate, device < 0 ? 0 : device), "clock rate");
    if (!ok) { pc_destroy(b); return NULL; }
#endif
    PcGeom *geom = (PcGeom *)calloc(1, sizeof(PcGeom));
    if (!geom) { pc_destroy(b); return NULL; }
    pc_geom_init(geom, size);
    ok = cu_ok(cudaMemcpy(b->d_geom, geom, sizeof(PcGeom), cudaMemcpyHostToDevice), "geom upload");
    free(geom);
    if (!ok) { pc_destroy(b); return NULL; }
    cudaMemsetAsync(b->d_legal_valid, 0, G, b->stream);
    cudaMemsetAsync(b->d_legal, 0, G * (size_t)b->legal_stride, b->stream);
    k_init<<<blocks(games), PC_TPB, 0, b->stream>>>(b->d_states, b->d_scratch, b->d_mscratch, games, size, players, b->n);
    if (sync_check(b, "k_init") != 0) { pc_destroy(b); return NULL; }
    return b;
}

static void pc_destroy_impl(PcBatch *b) {
    if (!b) return;
    cudaFree(b->d_rules); cudaFree(b->d_geom); cudaFree(b->d_states); cudaFree(b->d_scratch); cudaFree(b->d_mscratch);
    cudaFree(b->d_legal); cudaFree(b->d_legal_valid); cudaFree(b->d_in); cudaFree(b->d_layout); cudaFree(b->d_masks);
    cudaFree(b->d_view); cudaFree(b->d_opp); cudaFree(b->d_final); cudaFree(b->d_flags); cudaFree(b->d_rewards);
    cudaFree(b->d_status); cudaFree(b->d_scores); cudaFree(b->d_prof);
    if (b->ev0) cudaEventDestroy(b->ev0);
    if (b->ev1) cudaEventDestroy(b->ev1);
    if (b->h_state) cudaFreeHost(b->h_state);
    if (b->stream) cudaStreamDestroy(b->stream);
    free(b);
}

static int pc_set_rules_impl(PcBatch *b, const PcRulesInput *in) {
    if (!b || !in) return PC_E_ARGS;
    PcRules *r = (PcRules *)calloc(1, sizeof(PcRules));
    if (!r) return PC_E_MEMORY;
    int rc = pc_rules_finalize(in, r);
    if (rc != PC_RULES_OK) { free(r); b->rules_set = 0; return rc > 0 ? rc : PC_E_RULES; }
    if (!pc_pathfinding_open_fits(r)) { free(r); b->rules_set = 0; return PC_E_RULES; }   /* the open-list bound's premise */
    int ok = cu_ok(cudaMemcpy(b->d_rules, r, sizeof(PcRules), cudaMemcpyHostToDevice), "rules upload");
    free(r);
    if (!ok) return PC_E_DEVICE;
    b->rules_set = 1;
    return 0;
}

extern "C" int pc_mask_bytes(const PcBatch *b) { return b ? b->mask_bytes : PC_E_ARGS; }
extern "C" int pc_view_width(const PcBatch *b) { return b ? b->vw : PC_E_ARGS; }
extern "C" int pc_sizes(const PcBatch *b, int *n, int *np, int *games) {
    if (!b) return PC_E_ARGS;
    if (n) *n = b->n;
    if (np) *np = b->np;
    if (games) *games = b->games;
    return 0;
}

/* stage the four [G] int arrays and the layouts from host memory; returns the device pointers */
static int stage_inputs(PcBatch *b, const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
                        const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
                        const int32_t **df, const int32_t **ds, const int32_t **dt, const int32_t **dseed,
                        const int32_t **dterrain, const int32_t **dstructure, const int32_t **dresource, const int32_t **dcaps) {
    size_t G = (size_t)b->games, L = G * (size_t)b->n;
    const int32_t *src[4] = { family, source, target, seeds };
    for (int i = 0; i < 4; i++) {
        if (src[i] && !cu_ok(cudaMemcpyAsync(b->d_in + i * G, src[i], G * sizeof(int32_t), cudaMemcpyHostToDevice, b->stream), "in upload"))
            return PC_E_DEVICE;
    }
    const int32_t *lay[3] = { terrain, structure, resource };
    for (int i = 0; i < 3; i++) {
        if (!cu_ok(cudaMemcpyAsync(b->d_layout + i * L, lay[i], L * sizeof(int32_t), cudaMemcpyHostToDevice, b->stream), "layout upload"))
            return PC_E_DEVICE;
    }
    if (!cu_ok(cudaMemcpyAsync(b->d_layout + 3 * L, caps, G * (size_t)b->np * sizeof(int32_t), cudaMemcpyHostToDevice, b->stream), "caps upload"))
        return PC_E_DEVICE;
    *df = b->d_in; *ds = b->d_in + G; *dt = b->d_in + 2 * G; *dseed = b->d_in + 3 * G;
    *dterrain = b->d_layout; *dstructure = b->d_layout + L; *dresource = b->d_layout + 2 * L; *dcaps = b->d_layout + 3 * L;
    return 0;
}

static int pc_reset_impl(PcBatch *b, const int32_t *seeds, const int32_t *terrain, const int32_t *structure,
                        const int32_t *resource, const int32_t *caps) {
    if (!b || !seeds || !terrain || !structure || !resource || !caps) return PC_E_ARGS;
    if (!b->rules_set) return PC_E_NO_RULES;
    const int32_t *df, *ds, *dt, *dseed, *dterrain, *dstructure, *dresource, *dcaps;
    int rc = stage_inputs(b, NULL, NULL, NULL, seeds, terrain, structure, resource, caps, &df, &ds, &dt, &dseed, &dterrain,
                          &dstructure, &dresource, &dcaps);
    if (rc) return rc;
    k_reset<<<blocks(b->games), PC_TPB, 0, b->stream>>>(b->d_states, b->games, dseed, dterrain, dstructure, dresource, dcaps,
                                                        b->n, b->np, b->d_geom, b->max_turns, b->d_legal_valid);
    return sync_check(b, "k_reset");
}

static int tick_launch(PcBatch *b, const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
                       const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
                       int both_views, PcTickOut *out) {
    if (!b || !family || !source || !target || !seeds || !terrain || !structure || !resource || !caps || !out) return PC_E_ARGS;
    if (!out->masks || !out->view || !out->final_view || !out->done || !out->terminated || !out->truncated || !out->rewards
        || !out->status || (both_views && !out->opp_view)) return PC_E_ARGS;
    if (!b->rules_set) return PC_E_NO_RULES;
    size_t G = (size_t)b->games;
    const int np = b->np, vw = b->vw, mb = b->mask_bytes;
    const int32_t *df, *ds, *dt, *dseed, *dterrain, *dstructure, *dresource, *dcaps;
    uint8_t *masks, *done, *terminated, *truncated;
    int16_t *view, *opp, *final_view;
    float *rewards;
    int32_t *status, *scores;
    if (out->device_ptrs) {
        df = family; ds = source; dt = target; dseed = seeds;
        dterrain = terrain; dstructure = structure; dresource = resource; dcaps = caps;
        masks = out->masks; view = out->view; opp = out->opp_view; final_view = out->final_view;
        done = out->done; terminated = out->terminated; truncated = out->truncated; rewards = out->rewards; status = out->status; scores = out->scores;
    } else {
        int rc = stage_inputs(b, family, source, target, seeds, terrain, structure, resource, caps, &df, &ds, &dt, &dseed,
                              &dterrain, &dstructure, &dresource, &dcaps);
        if (rc) return rc;
        masks = b->d_masks; view = b->d_view; opp = b->d_opp; final_view = b->d_final;
        done = b->d_flags; terminated = b->d_flags + G; truncated = b->d_flags + 2 * G; rewards = b->d_rewards; status = b->d_status; scores = out->scores ? b->d_scores : NULL;
    }
    if (!b->warp_tick) cudaMemsetAsync(masks, 0, G * (size_t)mb, b->stream);   /* k_tick_warp writes every mask byte */
    cudaMemsetAsync(final_view, 0, G * (size_t)np * (size_t)vw * sizeof(int16_t), b->stream);
    pc_prof_t *prof = NULL;
#ifdef PC_PROFILE
    prof = b->d_prof;
    cudaEventRecord(b->ev0, b->stream);
#endif
    if (b->warp_tick) {
        unsigned wblocks = (unsigned)(((size_t)b->games * 32 + (size_t)b->warp_tpb - 1) / (size_t)b->warp_tpb);
        k_tick_warp<<<wblocks, b->warp_tpb, 0, b->stream>>>(b->d_states, b->d_scratch, b->d_mscratch, b->games, b->d_rules,
                                                            b->d_geom, b->max_actions, b->no_resign, b->no_peace, b->max_turns,
                                                            b->d_legal, b->d_legal_valid, mb, b->legal_stride, vw, df, ds, dt,
                                                            dseed, dterrain, dstructure, dresource, dcaps, both_views, masks, view,
                                                            opp, final_view, done, terminated, truncated, rewards, status, scores,
                                                            prof);
    } else {
        k_tick<<<blocks(b->games), PC_TPB, 0, b->stream>>>(b->d_states, b->d_scratch, b->d_mscratch, b->games, b->d_rules, b->d_geom,
                                                           b->max_actions, b->no_resign, b->no_peace, b->max_turns, b->d_legal,
                                                           b->d_legal_valid, mb, b->legal_stride, vw, df, ds, dt, dseed, dterrain,
                                                           dstructure, dresource, dcaps, both_views, masks, view, opp, final_view,
                                                           done, terminated, truncated, rewards, status, scores, prof);
    }
#ifdef PC_PROFILE
    cudaEventRecord(b->ev1, b->stream);
    if (cudaEventSynchronize(b->ev1) == cudaSuccess) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, b->ev0, b->ev1) == cudaSuccess) b->prof_ns += (uint64_t)((double)ms * 1.0e6);
        b->prof_ticks += 1;
    }
#endif
    if (!out->device_ptrs) {
        cudaMemcpyAsync(out->masks, masks, G * (size_t)mb, cudaMemcpyDeviceToHost, b->stream);
        cudaMemcpyAsync(out->view, view, G * (size_t)vw * sizeof(int16_t), cudaMemcpyDeviceToHost, b->stream);
        if (both_views)
            cudaMemcpyAsync(out->opp_view, opp, G * (size_t)(np - 1) * (size_t)vw * sizeof(int16_t), cudaMemcpyDeviceToHost, b->stream);
        cudaMemcpyAsync(out->final_view, final_view, G * (size_t)np * (size_t)vw * sizeof(int16_t), cudaMemcpyDeviceToHost, b->stream);
        cudaMemcpyAsync(out->done, done, G, cudaMemcpyDeviceToHost, b->stream);
        cudaMemcpyAsync(out->terminated, terminated, G, cudaMemcpyDeviceToHost, b->stream);
        cudaMemcpyAsync(out->truncated, truncated, G, cudaMemcpyDeviceToHost, b->stream);
        cudaMemcpyAsync(out->rewards, rewards, G * (size_t)np * sizeof(float), cudaMemcpyDeviceToHost, b->stream);
        cudaMemcpyAsync(out->status, status, G * sizeof(int32_t), cudaMemcpyDeviceToHost, b->stream);
        if (out->scores) cudaMemcpyAsync(out->scores, scores, G * (size_t)np * sizeof(int32_t), cudaMemcpyDeviceToHost, b->stream);
    }
    return cu_ok(cudaGetLastError(), "k_tick launch") ? 0 : PC_E_DEVICE;
}

static int pc_tick_impl(PcBatch *b, const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
                       const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
                       int both_views, PcTickOut *out) {
    int rc = tick_launch(b, family, source, target, seeds, terrain, structure, resource, caps, both_views, out);
    if (rc) return rc;
    return sync_check(b, "k_tick");
}

/* queued on the batch's stream; complete after pc_sync. Device pointers only (host staging needs the sync). */
static int pc_tick_async_impl(PcBatch *b, const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
                             const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
                             int both_views, PcTickOut *out) {
    if (!out || !out->device_ptrs) return PC_E_DEVICE;
    return tick_launch(b, family, source, target, seeds, terrain, structure, resource, caps, both_views, out);
}

static int pc_sync_impl(PcBatch *b) {
    if (!b) return PC_E_ARGS;
    return sync_check(b, "pc_sync");
}

static int pc_legal_masks_impl(PcBatch *b, uint8_t *packed) {
    if (!b || !packed) return PC_E_ARGS;
    if (!b->rules_set) return PC_E_NO_RULES;
    size_t G = (size_t)b->games;
    cudaMemsetAsync(b->d_masks, 0, G * (size_t)b->mask_bytes, b->stream);
    k_masks<<<blocks(b->games), PC_TPB, 0, b->stream>>>(b->d_states, b->d_mscratch, b->games, b->d_rules, b->d_geom, b->no_resign,
                                                        b->no_peace, b->mask_bytes, b->d_masks);
    cudaMemcpyAsync(packed, b->d_masks, G * (size_t)b->mask_bytes, cudaMemcpyDeviceToHost, b->stream);
    return sync_check(b, "k_masks");
}

static int pc_observe_impl(PcBatch *b, int player, int16_t *view) {
    if (!b || !view) return PC_E_ARGS;
    size_t G = (size_t)b->games;
    k_observe<<<blocks(b->games), PC_TPB, 0, b->stream>>>(b->d_states, b->games, b->d_geom, player, b->vw, b->d_view);
    cudaMemcpyAsync(view, b->d_view, G * (size_t)b->vw * sizeof(int16_t), cudaMemcpyDeviceToHost, b->stream);
    return sync_check(b, "k_observe");
}

static int pc_export_impl(PcBatch *b, int game, void **field_ptrs) {
    if (!b || !field_ptrs || game < 0 || game >= b->games) return PC_E_ARGS;
    if (!cu_ok(cudaMemcpy(b->h_state, b->d_states + game, sizeof(PcState), cudaMemcpyDeviceToHost), "state download")) return PC_E_DEVICE;
    PcIo io = { field_ptrs, 0, 0 };
    state_io(&io, b->h_state, b->n, b->np);
    return io.k == PC_EXPORT_FIELD_COUNT(b->np) ? 0 : PC_E_ARGS;
}

static int pc_import_impl(PcBatch *b, int game, void **field_ptrs) {
    if (!b || !field_ptrs || game < 0 || game >= b->games) return PC_E_ARGS;
    if (!cu_ok(cudaMemcpy(b->h_state, b->d_states + game, sizeof(PcState), cudaMemcpyDeviceToHost), "state download")) return PC_E_DEVICE;
    PcIo io = { field_ptrs, 0, 1 };
    state_io(&io, b->h_state, b->n, b->np);
    b->h_state->monuments_settled = 0;
    if (!cu_ok(cudaMemcpy(b->d_states + game, b->h_state, sizeof(PcState), cudaMemcpyHostToDevice), "state upload")) return PC_E_DEVICE;
    if (!cu_ok(cudaMemset(b->d_legal_valid + game, 0, 1), "legal_valid")) return PC_E_DEVICE;
    return io.k == PC_EXPORT_FIELD_COUNT(b->np) ? 0 : PC_E_ARGS;
}

extern "C" void pc_destroy(PcBatch *b) {
    if (!b) return;
    int prev;
    enter_device(b, &prev);
    pc_destroy_impl(b);
    leave_device(prev);
}

extern "C" int pc_set_rules(PcBatch *b, const PcRulesInput *in) {
    if (!b) return PC_E_ARGS;
    int prev, rc = enter_device(b, &prev);
    if (rc) return rc;
    rc = pc_set_rules_impl(b, in);
    leave_device(prev);
    return rc;
}

extern "C" int pc_reset(PcBatch *b, const int32_t *seeds, const int32_t *terrain, const int32_t *structure,
                        const int32_t *resource, const int32_t *caps) {
    if (!b) return PC_E_ARGS;
    int prev, rc = enter_device(b, &prev);
    if (rc) return rc;
    rc = pc_reset_impl(b, seeds, terrain, structure, resource, caps);
    leave_device(prev);
    return rc;
}

extern "C" int pc_tick(PcBatch *b, const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
                       const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
                       int both_views, PcTickOut *out) {
    if (!b) return PC_E_ARGS;
    int prev, rc = enter_device(b, &prev);
    if (rc) return rc;
    rc = pc_tick_impl(b, family, source, target, seeds, terrain, structure, resource, caps, both_views, out);
    leave_device(prev);
    return rc;
}

extern "C" int pc_tick_async(PcBatch *b, const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
                             const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
                             int both_views, PcTickOut *out) {
    if (!b) return PC_E_ARGS;
    int prev, rc = enter_device(b, &prev);
    if (rc) return rc;
    rc = pc_tick_async_impl(b, family, source, target, seeds, terrain, structure, resource, caps, both_views, out);
    leave_device(prev);
    return rc;
}

extern "C" int pc_sync(PcBatch *b) {
    if (!b) return PC_E_ARGS;
    int prev, rc = enter_device(b, &prev);
    if (rc) return rc;
    rc = pc_sync_impl(b);
    leave_device(prev);
    return rc;
}

extern "C" int pc_legal_masks(PcBatch *b, uint8_t *packed) {
    if (!b) return PC_E_ARGS;
    int prev, rc = enter_device(b, &prev);
    if (rc) return rc;
    rc = pc_legal_masks_impl(b, packed);
    leave_device(prev);
    return rc;
}

extern "C" int pc_observe(PcBatch *b, int player, int16_t *view) {
    if (!b) return PC_E_ARGS;
    int prev, rc = enter_device(b, &prev);
    if (rc) return rc;
    rc = pc_observe_impl(b, player, view);
    leave_device(prev);
    return rc;
}

extern "C" int pc_export(PcBatch *b, int game, void **field_ptrs) {
    if (!b) return PC_E_ARGS;
    int prev, rc = enter_device(b, &prev);
    if (rc) return rc;
    rc = pc_export_impl(b, game, field_ptrs);
    leave_device(prev);
    return rc;
}

extern "C" int pc_import(PcBatch *b, int game, void **field_ptrs) {
    if (!b) return PC_E_ARGS;
    int prev, rc = enter_device(b, &prev);
    if (rc) return rc;
    rc = pc_import_impl(b, game, field_ptrs);
    leave_device(prev);
    return rc;
}

/* ---- build information and the phase profile (pc_api.h, 2026-11-06) ---- */
static int pc_build_info_impl(const PcBatch *b, int32_t *out) {
    if (!b || !out) return PC_E_ARGS;
    out[0] = PC_MAXSIZE;
#ifdef PC_PROFILE
    out[1] = 1;
#else
    out[1] = 0;
#endif
    out[2] = b->warp_tick;
    out[3] = b->warp_tick ? b->warp_tpb : PC_TPB;
    return 0;
}

static int pc_profile_read_impl(PcBatch *b, uint64_t *out) {
    if (!b || !out) return PC_E_ARGS;
#ifdef PC_PROFILE
    size_t G = (size_t)b->games;
    pc_prof_t *h = (pc_prof_t *)malloc(G * PC_PROFILE_PHASES * sizeof(pc_prof_t));
    if (!h) return PC_E_MEMORY;
    int ok = cu_ok(cudaStreamSynchronize(b->stream), "profile sync")
        && cu_ok(cudaMemcpy(h, b->d_prof, G * PC_PROFILE_PHASES * sizeof(pc_prof_t), cudaMemcpyDeviceToHost), "profile download");
    if (!ok) { free(h); return PC_E_DEVICE; }
    for (int k = 0; k < PC_PROFILE_OUT; k++) out[k] = 0;
    for (size_t g = 0; g < G; g++)
        for (int k = 0; k < PC_PROFILE_PHASES; k++) out[k] += h[g * PC_PROFILE_PHASES + k];
    free(h);
    out[16] = b->prof_ticks;
    out[17] = G;
    out[18] = (uint64_t)b->clock_khz;
    out[19] = b->prof_ns;
    out[20] = (uint64_t)b->warp_tick;
    out[21] = (uint64_t)(b->warp_tick ? b->warp_tpb : PC_TPB);
    out[22] = PC_MAXSIZE;
    return 0;
#else
    return PC_E_NO_PROFILE;
#endif
}

static int pc_profile_reset_impl(PcBatch *b) {
    if (!b) return PC_E_ARGS;
#ifdef PC_PROFILE
    if (!cu_ok(cudaStreamSynchronize(b->stream), "profile sync")
        || !cu_ok(cudaMemset(b->d_prof, 0, (size_t)b->games * PC_PROFILE_PHASES * sizeof(pc_prof_t)), "profile reset"))
        return PC_E_DEVICE;
    b->prof_ticks = 0;
    b->prof_ns = 0;
    return 0;
#else
    return PC_E_NO_PROFILE;
#endif
}

extern "C" int pc_build_info(const PcBatch *b, int32_t *out) { return pc_build_info_impl(b, out); }

extern "C" int pc_profile_read(PcBatch *b, uint64_t *out) {
    if (!b) return PC_E_ARGS;
    int prev, rc = enter_device(b, &prev);
    if (rc) return rc;
    rc = pc_profile_read_impl(b, out);
    leave_device(prev);
    return rc;
}

extern "C" int pc_profile_reset(PcBatch *b) {
    if (!b) return PC_E_ARGS;
    int prev, rc = enter_device(b, &prev);
    if (rc) return rc;
    rc = pc_profile_reset_impl(b);
    leave_device(prev);
    return rc;
}
