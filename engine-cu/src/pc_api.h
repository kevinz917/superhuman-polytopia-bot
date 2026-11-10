/* pc_api.h - the C ABI of the single-source Polytopia engine (engine-cu), exported by both drivers:
 * pc_cpu.c (serial loop over games, the CPU oracle) and pc_cuda.cu (one thread per game). The contract is
 * engine-rs's GameBatch (engine-rs/src/lib.rs): reset_layouts, tick(packed_masks=True), legal_masks(packed=True),
 * observe, export, import_. src/polytopia_bot/rl/gpu_engine.py wraps it with ctypes.
 *
 * Conventions: every function returns 0 on success and a negative PC_E_* code otherwise (the engine's own step
 * status codes, PC_OK.. PC_EPISODE_DONE, travel in PcTickOut.status). Array arguments are host pointers (C order,
 * contiguous) unless PcTickOut.device_ptrs says otherwise. Game count G, tiles n = size * size, seats np. */
#ifndef PC_API_H
#define PC_API_H

#include <stdint.h>
#include "engine.h"   /* PcRulesInput (rules.h) and the engine entry points the drivers call */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- driver error codes (negative; the engine's step status codes are >= 0) ---- */
#define PC_E_ARGS (-1)        /* NULL pointer, size not 11/14/18, players outside 2..PC_MAXP, games < 1, game index */
#define PC_E_NO_RULES (-2)    /* pc_set_rules has not succeeded on this batch */
#define PC_E_RULES (-3)       /* pc_rules_finalize rejected the table */
#define PC_E_DEVICE (-4)      /* device_ptrs requested of a driver that cannot serve it, or the CUDA device failed */
#define PC_E_MEMORY (-5)
/* PC_E_NO_PROFILE (-6): see pc_profile_read below */

typedef struct PcBatch PcBatch;

/* A batch of `games` independent games on `size` x `size` boards with `players` seats. max_turns <= 0: no round
 * limit (the real game); else the harness truncation after that many rounds. no_resign / no_peace: mask options
 * (GameBatch's). max_actions: per-turn action cap (core.rs). device: -1 = CPU driver, else the CUDA device ordinal
 * (the CPU driver records and ignores it). Returns NULL on bad arguments (a size above the build's PC_MAXSIZE among
 * them: the libpc_*_14.so builds take 11 and 14) or allocation failure. CUDA: PC_CUDA_WARP_TICK=1 in the environment
 * selects the warp-per-game tick kernel (PC_CUDA_WARP_TPB threads per block, default 128), PC_CUDA_STACK_KB the
 * per-thread stack. */
PcBatch *pc_create(int size, int games, int players, int max_turns, int no_resign, int no_peace, int max_actions,
                   int device);
void pc_destroy(PcBatch *b);

/* Finalize the primary rule tables (gpu_engine.py::pack_rules) into the batch's PcRules: 0 ok, 1..6 the
 * pc_rules_finalize PC_RULES_ERR_* code (rules.h), negative a driver error (PC_E_RULES too when a supported unit
 * moves more than the unit search's open-list bound assumes, pathfinding.h). Required before reset / tick / legal_masks. */
int pc_set_rules(PcBatch *b, const PcRulesInput *in);

int pc_mask_bytes(const PcBatch *b);     /* packed legal-mask bytes per game = ceil(mask_width / 8) */
int pc_view_width(const PcBatch *b);     /* int16 entries of one seat's packed view */
int pc_sizes(const PcBatch *b, int *n, int *np, int *games);

/* Fresh games from explicit layouts (engine-rs GameBatch.layout(seed) / reset_layouts): seeds [G] (cast to u32),
 * terrain / structure / resource [G, n], caps [G, np] capital tiles per seat. Invalidates the cached masks. */
int pc_reset(PcBatch *b, const int32_t *seeds, const int32_t *terrain, const int32_t *structure,
             const int32_t *resource, const int32_t *caps);

typedef struct {
    uint8_t *masks;        /* [G, mask_bytes] packed legal masks of the post-tick states, numpy.packbits order */
    int16_t *view;         /* [G, vw] the mover's view */
    int16_t *opp_view;     /* [G, np-1, vw] the other seats in turn order after the mover (when both_views) */
    int16_t *final_view;   /* [G, np, vw] every seat's view of a truncated final state (zero otherwise) */
    uint8_t *done, *terminated, *truncated;   /* [G] 0/1 */
    float *rewards;        /* [G, np] terminal rewards of the stepped state (zero while a game runs) */
    int32_t *status;       /* [G] step status (PC_OK, PC_INVALID_ACTION, ...) */
    int32_t *scores;       /* [G, np] players.score of the stepped state before any reset (cap-score rewards); NULL = skip */
    int device_ptrs;       /* 1: every pointer above is device memory (CUDA driver only; no staging copies) */
} PcTickOut;

/* One actor tick, GameBatch.tick(packed_masks=True) exactly: per game decode (family, source, target) in the current
 * state, step_with validated by the masks the previous tick returned for that state (if still valid), record
 * terminated / truncated / done / terminal rewards, write every seat's final view when truncated, reset finished
 * games from the layouts (seeds [G], terrain / structure / resource [G, n], caps [G, np]), then the new state's packed
 * masks (cached for the next tick's validation), the mover's view and, with both_views, the other seats' views. */
int pc_tick(PcBatch *b, const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
            const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
            int both_views, PcTickOut *out);

/* pc_tick without the final stream synchronization (CUDA driver, device_ptrs only): the kernels are queued on the
 * batch's stream and the call returns at once; the outputs are complete after pc_sync. The CPU driver runs the tick
 * synchronously and pc_sync is a no-op. Inputs must be complete when called (the caller synchronizes its own streams). */
int  pc_tick_async(PcBatch *b, const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
                   const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
                   int both_views, PcTickOut *out);
int  pc_sync(PcBatch *b);

/* Every State field of one game to / from host buffers, in PC_EXPORT_FIELDS order (field_ptrs[i] is the buffer of
 * field i, dtype and shape as listed). pc_import overwrites the game, resets its monuments_settled flag and
 * invalidates its cached masks (lib.rs import_ with every field present). */
int pc_export(PcBatch *b, int game, void **field_ptrs);
int pc_import(PcBatch *b, int game, void **field_ptrs);

/* Packed legal masks of the current states into packed [G, mask_bytes] (zeroed here); the cache is untouched. */
int pc_legal_masks(PcBatch *b, uint8_t *packed);
/* Packed view of seat `player` (1-based; 0 = the player to move) of every game into view [G, vw]. */
int pc_observe(PcBatch *b, int player, int16_t *view);

/* ---- build information and the phase profile (2026-11-06 additions) ----
 * pc_build_info: out[PC_BUILD_INFO_OUT] = { PC_MAXSIZE of this build, 1 if built with -DPC_PROFILE, tick kernel
 * (0 = one thread per game, 1 = one warp per game: CUDA driver with PC_CUDA_WARP_TICK=1 at pc_create), threads per
 * block of the tick kernel (1 for the CPU driver) }.
 * pc_profile_read (profile builds; PC_E_NO_PROFILE otherwise): out[PC_PROFILE_OUT] = phase counters summed over the
 * batch's games since pc_create / pc_profile_reset, out[0..PC_PROFILE_PHASES) in PC_PH_* order (pc.h: step, outputs,
 * final views, reset, mask prelude, mask move / attack / summon / build / research / economy / unit / diplomacy,
 * legal copy, observe mover, observe others), then out[16] ticks, out[17] games, out[18] counter unit (CUDA: the
 * SM clock in kHz, the counters being clock64 cycles of the thread (lane 0 of the warp) running the game; 0 = CPU
 * driver, the counters being nanoseconds), out[19] tick time in ns summed over the ticks (CUDA: k_tick alone, CUDA
 * events; CPU: the whole pc_tick loop), out[20] tick kernel (as pc_build_info), out[21] threads per block, out[22]
 * PC_MAXSIZE, out[23] reserved 0. A profile build synchronizes the stream after each tick (pc_tick_async included).
 * pc_profile_reset zeroes the counters. */
#define PC_E_NO_PROFILE (-6)
#define PC_BUILD_INFO_OUT 4
#define PC_PROFILE_OUT 24
int pc_build_info(const PcBatch *b, int32_t *out);
int pc_profile_read(PcBatch *b, uint64_t *out);
int pc_profile_reset(PcBatch *b);

/* PC_EXPORT_FIELDS: the export / import field order, engine-rs lib.rs::export's dict keys exactly (gpu_engine.py
 * EXPORT_FIELDS mirrors it). Shapes are per game; "bool" is one uint8 0/1 per entry (numpy bool); n tiles, np seats.
 *
 *   tiles.terrain i32 [n]            tiles.owner i32 [n]            tiles.capital i32 [n]        tiles.city i32 [n]
 *   tiles.road bool [n]              tiles.structure i32 [n]        tiles.resource i32 [n]
 *   tiles.explored bool [np, n]      tiles.structure_level i32 [n]  tiles.structure_turn i32 [n] tiles.route bool [n]
 *   units.owner i32 [n]              units.kind i32 [n]             units.health f32 [n]         units.veteran bool [n]
 *   units.moved bool [n]             units.attacked bool [n]        units.kills i32 [n]          units.poison bool [n]
 *   units.frozen bool [n]            units.boosted bool [n]         units.home i32 [n]           units.created i32 [n]
 *   units.converted bool [n]         units.attacks_performed i32 [n] units.prev_x i32 [n]        units.prev_y i32 [n]
 *   units.last_attack i32 [n]        units.passenger i32 [n]        units.invisible bool [n]     units.direction i32 [n]
 *   cities.owner i32 [n]             cities.level i32 [n]           cities.population i32 [n]    cities.progress i32 [n]
 *   cities.production i32 [n]        cities.border i32 [n]          cities.rewards i32 [n, 9]
 *   cities.territory bool [n, n]     (dense: [city, tile], the bitset unpacked)
 *   cities.connected bool [n]        cities.order i32 [n]           cities.infiltrated i32 [n]   cities.riot bool [n]
 *   players.stars i32 [np]           players.score i32 [np]         players.tech bool [np, 128]
 *   players.tech_turn i32 [np, 128]  players.known bool [np, np]    players.kills i32 [np]       players.casualties i32 [np]
 *   players.pacifist i32 [np]        players.attacked bool [np]     players.built bool [np, 128] players.killed_turn i32 [np]
 *   players.resigned_turn i32 [np]   players.killer i32 [np]        players.peace bool [np, np]  players.embassy bool [np, np]
 *   players.peace_offer bool [np, np] players.truce_breaker i32 [np, np] players.hostile_until i32 [np, np]
 *   players.peace_broken i32 [np, np] players.monuments bool [np, 128]
 *   players.out_order i32 [np]       (only when np > 2)
 *   history.seen_unit bool [np, 3, n] history.prev_code i32 [np, n]  history.built bool [np, n]  history.trained bool [np, n]
 *   history.enemy_tech bool [np, 128] (bit i of the u128)
 *   current_player i32 []            turn i32 []                    version i32 []               state_version i32 []
 *   actions i32 []                   terminated bool []             truncated bool []            fog bool []
 *   reference_max_turns i32 []       core_supported bool []         seed u32 []                  initial_seed u32 []
 */
#define PC_EXPORT_FIELDS_2P 80   /* field count with two seats */
#define PC_EXPORT_FIELDS_NP 81   /* with players.out_order (np > 2) */
/* number of export fields of a batch with np seats */
#define PC_EXPORT_FIELD_COUNT(np) ((np) > 2 ? PC_EXPORT_FIELDS_NP : PC_EXPORT_FIELDS_2P)

#ifdef __cplusplus
}
#endif
#endif /* PC_API_H */
