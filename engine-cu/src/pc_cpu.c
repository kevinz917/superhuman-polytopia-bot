/* pc_cpu.c - CPU driver of the single-source Polytopia engine: the C ABI of pc_api.h over engine.h, one serial loop
 * over the games. This is the oracle the CUDA driver is gated against, and itself gated bitwise against engine-rs
 * (tests/test_gpu_engine.py). The driver allocates (malloc) and the engine never does.
 *
 * pc_tick is engine-rs lib.rs::GameBatch::tick(packed_masks=True) line for line; see the comments there.
 * A -DPC_PROFILE build (Makefile PROFILE=1) times the tick's phases per game in nanoseconds (pc_profile_read). */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime under -std=c11 (profile builds) */
#include <stdlib.h>
#include <string.h>
#include "pc_api.h"
#include "pc_state_io.h"

struct PcBatch {
    int size, n, np, games;
    int32_t max_turns;        /* PC_NO_ROUND_LIMIT when the caller passed <= 0 (lib.rs: max_turns.unwrap_or(NO_ROUND_LIMIT)) */
    int no_resign, no_peace;
    int32_t max_actions;
    int device;
    int rules_set;
    int mask_bytes, vw;
    PcRules *rules;
    PcGeom *geom;
    PcState *states;          /* [G] */
    PcScratch *scratch;       /* one: the CPU steps games one at a time, like GameBatch */
    PcMaskScratch *mscratch;
    /* lib.rs LegalCache per game: the packed masks tick() last returned, while they describe the game's state */
    uint8_t *legal;           /* [G, mask_bytes] */
    uint8_t *legal_valid;     /* [G] */
    pc_prof_t *prof;          /* [G, PC_PROFILE_PHASES] phase nanoseconds (profile builds only) */
    uint64_t prof_ticks, prof_ns;
};

static void forget_masks(PcBatch *b) {
    /* GameBatch::forget_masks: the states changed outside tick(), the cached masks no longer describe them */
    memset(b->legal_valid, 0, (size_t)b->games);
}

PcBatch *pc_create(int size, int games, int players, int max_turns, int no_resign, int no_peace, int max_actions,
                   int device) {
    if (!(size == 11 || size == 14 || size == 18) || games < 1 || players < 2 || players > PC_MAXP) return NULL;
    if (size > PC_MAXSIZE) return NULL;   /* a PC_MAXSIZE=14 build takes 11 and 14 only */
    PcBatch *b = (PcBatch *)calloc(1, sizeof(PcBatch));
    if (!b) return NULL;
    b->size = size;
    b->n = size * size;
    b->np = players;
    b->games = games;
    b->max_turns = max_turns <= 0 ? PC_NO_ROUND_LIMIT : max_turns;
    b->no_resign = no_resign != 0;
    b->no_peace = no_peace != 0;
    b->max_actions = max_actions;
    b->device = device;
    b->mask_bytes = pc_masks_packed_width(b->n);
    b->vw = pc_kernels_view_width_np(b->n, b->np);
    b->rules = (PcRules *)calloc(1, sizeof(PcRules));
    b->geom = (PcGeom *)calloc(1, sizeof(PcGeom));
    b->states = (PcState *)calloc((size_t)games, sizeof(PcState));
    b->scratch = (PcScratch *)calloc(1, sizeof(PcScratch));
    b->mscratch = (PcMaskScratch *)calloc(1, sizeof(PcMaskScratch));
    b->legal = (uint8_t *)calloc((size_t)games * (size_t)b->mask_bytes, 1);
    b->legal_valid = (uint8_t *)calloc((size_t)games, 1);
#ifdef PC_PROFILE
    b->prof = (pc_prof_t *)calloc((size_t)games * PC_PROFILE_PHASES, sizeof(pc_prof_t));
    if (!b->prof) { pc_destroy(b); return NULL; }
#endif
    if (!b->rules || !b->geom || !b->states || !b->scratch || !b->mscratch || !b->legal || !b->legal_valid) {
        pc_destroy(b);
        return NULL;
    }
    pc_geom_init(b->geom, size);
    /* Scratch::new(n) / MaskScratch::new(n): the component constructors (dist = INF, parent = -1 ...), not zeros */
    pc_kernels_move_scratch_init(&b->scratch->mv);
    pc_structures_build_ctx_init(&b->scratch->build);
    pc_military_summon_ctx_new(&b->scratch->summon);
    pc_pathfinding_scratch_init(&b->scratch->path, b->n);
    pc_kernels_move_scratch_init(&b->mscratch->mv);
    pc_structures_build_ctx_init(&b->mscratch->build);
    pc_military_summon_ctx_new(&b->mscratch->summon);
    for (int g = 0; g < games; g++) pc_state_empty(&b->states[g], size, players);   /* State::empty_players */
    return b;
}

void pc_destroy(PcBatch *b) {
    if (!b) return;
    free(b->rules); free(b->geom); free(b->states); free(b->scratch); free(b->mscratch); free(b->legal); free(b->legal_valid);
    free(b->prof);
    free(b);
}

int pc_set_rules(PcBatch *b, const PcRulesInput *in) {
    if (!b || !in) return PC_E_ARGS;
    memset(b->rules, 0, sizeof(PcRules));
    int rc = pc_rules_finalize(in, b->rules);
    if (rc != PC_RULES_OK) {
        b->rules_set = 0;
        return rc > 0 ? rc : PC_E_RULES;   /* the PC_RULES_ERR_* code itself (1..6) */
    }
    if (!pc_pathfinding_open_fits(b->rules)) { b->rules_set = 0; return PC_E_RULES; }
    b->rules_set = 1;
    return 0;
}

int pc_mask_bytes(const PcBatch *b) { return b ? b->mask_bytes : PC_E_ARGS; }
int pc_view_width(const PcBatch *b) { return b ? b->vw : PC_E_ARGS; }
int pc_sizes(const PcBatch *b, int *n, int *np, int *games) {
    if (!b) return PC_E_ARGS;
    if (n) *n = b->n;
    if (np) *np = b->np;
    if (games) *games = b->games;
    return 0;
}

static void reset_game(PcBatch *b, int g, const int32_t *seeds, const int32_t *terrain, const int32_t *structure,
                       const int32_t *resource, const int32_t *caps) {
    /* mapgen::initial_state(seeds[g] as u32, layout row g, caps row g, geom, max_turns) */
    pc_core_initial_state(&b->states[g], (uint32_t)seeds[g], terrain + (size_t)g * b->n, structure + (size_t)g * b->n,
                          resource + (size_t)g * b->n, caps + (size_t)g * b->np, b->np, b->geom, b->max_turns);
}

int pc_reset(PcBatch *b, const int32_t *seeds, const int32_t *terrain, const int32_t *structure,
             const int32_t *resource, const int32_t *caps) {
    if (!b || !seeds || !terrain || !structure || !resource || !caps) return PC_E_ARGS;
    if (!b->rules_set) return PC_E_NO_RULES;
    for (int g = 0; g < b->games; g++) reset_game(b, g, seeds, terrain, structure, resource, caps);
    forget_masks(b);
    return 0;
}

/* Seat ids (1-based) other than the mover, in turn order after it (lib.rs other_seats): k = 1..np-1 ->
 * (current_player - 1 + k) % np + 1. */
static int other_seat(const PcState *s, int k) { return (s->current_player - 1 + k) % s->np + 1; }

int pc_tick(PcBatch *b, const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
            const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
            int both_views, PcTickOut *out) {
    if (!b || !family || !source || !target || !seeds || !terrain || !structure || !resource || !caps || !out) return PC_E_ARGS;
    if (!out->masks || !out->view || !out->final_view || !out->done || !out->terminated || !out->truncated || !out->rewards
        || !out->status || (both_views && !out->opp_view)) return PC_E_ARGS;
    if (!b->rules_set) return PC_E_NO_RULES;
    if (out->device_ptrs) return PC_E_DEVICE;   /* the CPU driver only writes host memory */
    const int np = b->np, vw = b->vw, mb = b->mask_bytes, others = np - 1;
    /* the masks only write their true entries and the final views only for truncated games: those start zeroed
     * (the views are written in full for every game) */
    memset(out->masks, 0, (size_t)b->games * (size_t)mb);
    memset(out->final_view, 0, (size_t)b->games * (size_t)np * (size_t)vw * sizeof(int16_t));
#ifdef PC_PROFILE
    pc_prof_t t_tick = pc_prof_now();
#endif
    for (int g = 0; g < b->games; g++) {
        PcState *st = &b->states[g];
        pc_prof_t *acc = NULL;
#ifdef PC_PROFILE
        acc = b->prof + (size_t)g * PC_PROFILE_PHASES;
#endif
        PC_PROF_START(t_prof);
        (void)acc;
        PcAction a = pc_core_decode(st, family[g], source[g], target[g]);
        int32_t v = st->state_version;
        /* the masks the previous tick returned describe exactly this state (until it is stepped below) */
        const uint8_t *legal = b->legal_valid[g] ? b->legal + (size_t)g * mb : NULL;
        out->status[g] = pc_core_step_with(st, a, v, b->rules, b->geom, b->max_actions, b->scratch, legal);
        PC_PROF_LAP(acc, t_prof, PC_PH_STEP);
        b->legal_valid[g] = 0;   /* cache.clear() */
        out->terminated[g] = st->terminated;
        out->truncated[g] = st->truncated;
        out->done[g] = (uint8_t)(st->terminated || st->truncated);
        float rw[PC_MAXP];
        pc_core_terminal_rewards(st, rw);
        for (int k = 0; k < np; k++) out->rewards[(size_t)g * np + k] = rw[k];
        if (out->scores) for (int k = 0; k < np; k++) out->scores[(size_t)g * np + k] = st->players.score[k];
        PC_PROF_LAP(acc, t_prof, PC_PH_OUTPUTS);
        if (st->truncated) {
            for (int seat = 0; seat < np; seat++)
                pc_kernels_observe(st, seat + 1, b->geom, out->final_view + ((size_t)g * np + seat) * vw);
        }
        PC_PROF_LAP(acc, t_prof, PC_PH_FINAL_VIEWS);
        if (out->done[g]) reset_game(b, g, seeds, terrain, structure, resource, caps);
        PC_PROF_LAP(acc, t_prof, PC_PH_RESET);
        /* masks of the (possibly fresh) state: into the output row and the cache (lib.rs Tee) */
        uint8_t *row = out->masks + (size_t)g * mb;
        pc_masks_into_lanes(st, b->rules, b->geom, (uint8_t)b->no_resign, (uint8_t)b->no_peace, b->mscratch, row, pc_lanes_one(), acc);
        PC_PROF_START(t_copy);
        memcpy(b->legal + (size_t)g * mb, row, (size_t)mb);
        b->legal_valid[g] = 1;
        PC_PROF_LAP(acc, t_copy, PC_PH_LEGAL_COPY);
        pc_kernels_observe(st, st->current_player, b->geom, out->view + (size_t)g * vw);
        PC_PROF_LAP(acc, t_copy, PC_PH_OBSERVE_MOVER);
        if (both_views) {
            for (int k = 1; k <= others; k++)
                pc_kernels_observe(st, other_seat(st, k), b->geom, out->opp_view + ((size_t)g * others + (k - 1)) * vw);
        }
        PC_PROF_LAP(acc, t_copy, PC_PH_OBSERVE_OTHERS);
    }
#ifdef PC_PROFILE
    b->prof_ns += pc_prof_now() - t_tick;
    b->prof_ticks += 1;
#endif
    return 0;
}

int pc_tick_async(PcBatch *b, const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
                  const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
                  int both_views, PcTickOut *out) {
    return pc_tick(b, family, source, target, seeds, terrain, structure, resource, caps, both_views, out);
}
int pc_sync(PcBatch *b) { return b ? 0 : PC_E_ARGS; }

int pc_legal_masks(PcBatch *b, uint8_t *packed) {
    if (!b || !packed) return PC_E_ARGS;
    if (!b->rules_set) return PC_E_NO_RULES;
    memset(packed, 0, (size_t)b->games * (size_t)b->mask_bytes);
    for (int g = 0; g < b->games; g++)
        pc_masks_into(&b->states[g], b->rules, b->geom, b->no_resign, b->no_peace, b->mscratch, packed + (size_t)g * b->mask_bytes);
    return 0;
}

int pc_observe(PcBatch *b, int player, int16_t *view) {
    if (!b || !view) return PC_E_ARGS;
    for (int g = 0; g < b->games; g++) {
        const PcState *st = &b->states[g];
        pc_kernels_observe(st, player == 0 ? st->current_player : player, b->geom, view + (size_t)g * b->vw);
    }
    return 0;
}

int pc_export(PcBatch *b, int game, void **field_ptrs) {
    if (!b || !field_ptrs || game < 0 || game >= b->games) return PC_E_ARGS;
    PcIo io = { field_ptrs, 0, 0 };
    state_io(&io, &b->states[game], b->n, b->np);
    return io.k == PC_EXPORT_FIELD_COUNT(b->np) ? 0 : PC_E_ARGS;
}

int pc_import(PcBatch *b, int game, void **field_ptrs) {
    if (!b || !field_ptrs || game < 0 || game >= b->games) return PC_E_ARGS;
    PcIo io = { field_ptrs, 0, 1 };
    PcState *s = &b->states[game];
    state_io(&io, s, b->n, b->np);
    /* lib.rs import_: monuments_settled = false (any way of changing a state other than step), masks forgotten */
    s->monuments_settled = 0;
    b->legal_valid[game] = 0;
    return io.k == PC_EXPORT_FIELD_COUNT(b->np) ? 0 : PC_E_ARGS;
}

int pc_build_info(const PcBatch *b, int32_t *out) {
    if (!b || !out) return PC_E_ARGS;
    out[0] = PC_MAXSIZE;
#ifdef PC_PROFILE
    out[1] = 1;
#else
    out[1] = 0;
#endif
    out[2] = 0;   /* serial loop over games */
    out[3] = 1;
    return 0;
}

int pc_profile_read(PcBatch *b, uint64_t *out) {
    if (!b || !out) return PC_E_ARGS;
#ifdef PC_PROFILE
    for (int k = 0; k < PC_PROFILE_OUT; k++) out[k] = 0;
    for (int g = 0; g < b->games; g++)
        for (int k = 0; k < PC_PROFILE_PHASES; k++) out[k] += b->prof[(size_t)g * PC_PROFILE_PHASES + k];
    out[16] = b->prof_ticks;
    out[17] = (uint64_t)b->games;
    out[18] = 0;   /* nanoseconds */
    out[19] = b->prof_ns;
    out[20] = 0;
    out[21] = 1;
    out[22] = PC_MAXSIZE;
    return 0;
#else
    return PC_E_NO_PROFILE;
#endif
}

int pc_profile_reset(PcBatch *b) {
    if (!b) return PC_E_ARGS;
#ifdef PC_PROFILE
    memset(b->prof, 0, (size_t)b->games * PC_PROFILE_PHASES * sizeof(pc_prof_t));
    b->prof_ticks = 0;
    b->prof_ns = 0;
    return 0;
#else
    return PC_E_NO_PROFILE;
#endif
}
