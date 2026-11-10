/* masks.h - port of engine-rs/src/masks.rs: gpu/actions.py::legal_masks written as the true entries of one flat
 * boolean buffer in catalog order (move, attack, summon, build, research, harvest, forest, reward, capture, recover,
 * promote, disband, heal, destroy, diplomatic, end, resign).
 *
 * Only the Packed sink exists here (pc_mask_set in pc.h: bits in numpy.packbits order into a buffer the caller
 * zeroed). The packed output of a tick IS the legal cache for the next step (core::step_with's `legal`), so the
 * Dense sink, Tee, LegalCache and Count of the Rust are not ported. Nothing is written for false entries.
 *
 * Emission order: every family is visited in the Rust order and every entry of a family in the Rust order; where the
 * Rust emits through a closure (kernels::reach, military::attack_targets) the callee returns the tiles as a list in
 * its emit order and this file sets them in that order. */
#ifndef PC_MASKS_H
#define PC_MASKS_H

#include "pc.h"
#include "kernels.h"      /* PcMoveScratch */
#include "military.h"     /* PcSummonCtx */
#include "structures.h"   /* PcBuildCtx */
#include "capture.h"
#include "diplomacy.h"

#ifdef __cplusplus
extern "C" {
#endif

/* core.rs constants this module needs before core.h is read (same values; core.h re-defines them under guards) */
#ifndef PC_UNIT_ROW
#define PC_UNIT_ROW 2
#endif
#ifndef PC_DIPLOMACY_BLOCK
#define PC_DIPLOMACY_BLOCK 6
#endif
#ifndef PC_DIPLOMACY_BLOCKS
#define PC_DIPLOMACY_BLOCKS (PC_MAXP - 1)
#endif
#ifndef PC_DIPLOMATIC_WIDTH
#define PC_DIPLOMATIC_WIDTH (PC_DIPLOMACY_BLOCK * PC_DIPLOMACY_BLOCKS)
#endif

/* provided by core.h */
PC_FN void pc_core_family_widths(int n, int out[17]);

/* provided by kernels.h */
PC_FN float pc_kernels_max_health(const PcState *s, const PcRules *r, int i);
PC_FN void pc_kernels_movement_flags(const PcState *s, int32_t owner, const PcGeom *g, uint8_t *flags);
PC_FN void pc_kernels_movement_flags_lanes(const PcState *s, int32_t owner, const PcGeom *g, uint8_t *flags, PcLanes L);
PC_FN void pc_kernels_move_scratch_init(PcMoveScratch *ms);
/* reach's `emit` closure as a list: the destinations of the unit on `source` in emit order; returns how many */
PC_FN int pc_kernels_reach(const PcState *s, int source, const PcRules *r, const PcGeom *g, PcMoveScratch *ms,
                           uint16_t reached[PC_MAXN]);
PC_FN pc_u128 pc_kernels_research_bits(const PcState *s, const PcRules *r);
PC_FN pc_u128 pc_kernels_research_bits_lanes(const PcState *s, const PcRules *r, PcLanes L);

/* provided by military.h */
/* attack_targets' `emit` closure as a list of tiles in emit order; returns how many */
PC_FN int pc_military_attack_targets(const PcState *s, int source, const PcRules *r, const PcGeom *g, uint16_t *out);
PC_FN void pc_military_summon_prepare(PcSummonCtx *ctx, const PcState *s, const PcRules *r);   /* SummonCtx::prepare */
PC_FN void pc_military_summon_prepare_lanes(PcSummonCtx *ctx, const PcState *s, const PcRules *r, PcLanes L);
PC_FN pc_u128 pc_military_summon_row(const PcState *s, const PcSummonCtx *ctx, int t);
PC_FN uint8_t pc_military_can_disembark(const PcState *s, const PcRules *r, int i);
PC_FN uint8_t pc_military_promotable(const PcState *s, const PcRules *r, int i);

/* provided by structures.h */
PC_FN void pc_structures_build_ctx_prepare(PcBuildCtx *ctx, const PcState *s, const PcRules *r, const PcGeom *g); /* BuildCtx::prepare */
PC_FN void pc_structures_build_ctx_prepare_lanes(PcBuildCtx *ctx, const PcState *s, const PcRules *r, const PcGeom *g, PcLanes L);
PC_FN pc_u128 pc_structures_build_row(const PcState *s, const PcRules *r, const PcGeom *g, const PcBuildCtx *ctx, int tile);
PC_FN uint8_t pc_structures_neutral_road(const PcState *s, const PcRules *r, const PcBuildCtx *ctx, int tile);
PC_FN uint8_t pc_structures_neutral_bridge(const PcState *s, const PcRules *r, const PcBuildCtx *ctx, int tile);
PC_FN uint8_t pc_structures_destructible(int32_t k);

/* provided by capture.h */
PC_FN uint8_t pc_capture_capturable(const PcState *s, int i);

/* provided by diplomacy.h */
PC_FN int32_t pc_diplomacy_block_seat(const PcState *s, int b);
PC_FN int pc_diplomacy_seat_block(const PcState *s, int32_t q);
PC_FN void pc_diplomacy_capital_of(const PcState *s, int32_t q, int *capital, uint8_t *has_capital);
PC_FN uint8_t pc_diplomacy_legal_with(const PcState *s, int32_t source, int32_t target, int32_t ability, int capital,
                                      uint8_t has_capital);
PC_FN int32_t pc_diplomacy_pending_sender(const PcState *s);   /* Option<i32>: -1 when none */
PC_FN int32_t pc_diplomacy_ability(int k);                     /* diplomacy::ABILITIES[k] */

PC_FN int pc_masks_mask_width(int n) {
    int w[17];
    int t = 0;
    pc_core_family_widths(n, w);
    for (int i = 0; i < 17; i++) t += w[i];
    return t;
}

/* Bytes per game of the packed masks: ceil(mask_width / 8). */
PC_FN int pc_masks_packed_width(int n) { return (pc_masks_mask_width(n) + 7) / 8; }

PC_FN void pc_masks_family_offsets(int n, int out[18]) {
    int w[17];
    pc_core_family_widths(n, w);
    out[0] = 0;
    for (int i = 0; i < 17; i++) out[i + 1] = out[i] + w[i];
}

/* Buffers reused across legal_masks calls (one per game). The move and attack rows keep the reach / attack_targets
 * emit lists on the stack (this port's addition; one list per lane in the warp tick). */
typedef struct {
    PcMoveScratch mv;
    PcBuildCtx build;
    PcSummonCtx summon;
} PcMaskScratch;

/* Every family below takes the lanes it runs on (pc.h PcLanes; pc_lanes_one() for the serial engine) and sets its
 * bits with pc_lanes_mask_set. Each returns with the lanes in step (uniform control flow, no lane sync needed by
 * its caller beyond the ones it makes). */

/* City rewards: the two kinds of the next slot of every own city with one pending. Returns whether any is
 * pending (State::pending_reward), on every lane. */
PC_FN uint8_t pc_masks_reward_rows(const PcState *s, PcLanes L, uint8_t *out) {
    int o[18];
    const PcCities *c = &s->cities;
    int32_t cp = s->current_player;
    uint8_t pending = 0;
    pc_masks_family_offsets(s->n, o);
    for (int i = L.lane; i < s->n; i += L.count) {
        if (c->owner[i] == cp) {
            int32_t count = pc_reward_count(s, i);
            if (count < c->level[i] - 1) {
                int a, b;
                pending = 1;
                switch (count + 1) {
                    case 1: a = 3; b = 4; break;
                    case 2: a = 1; b = 7; break;
                    case 3: a = 5; b = 8; break;
                    default: a = 2; b = 6; break;
                }
                pc_lanes_mask_set(L, out, o[7] + i * 9 + a);
                pc_lanes_mask_set(L, out, o[7] + i * 9 + b);
            }
        }
    }
    return (uint8_t)pc_lanes_any(L, pending);
}

PC_FN int pc_masks_movable(const PcState *s, const PcRules *r, int i) {
    const PcUnits *u = &s->units;
    return u->owner[i] == s->current_player && !u->moved[i] && r->unit_supported[u->kind[i]];
}

/* Move rows of the mover's unmoved units (kernels.movement_targets' remaining gate: a supported kind). One lane:
 * movement_flags into mv->flags, then reach per unit on `mv` (which reach leaves clean). More lanes: the flags strided
 * over the lanes, then one unit per lane, each lane's reach on its own stack scratch holding a copy of the flags
 * (reach is a function of the state, the flags and a clean scratch, so every row is the serial row). */
PC_FN void pc_masks_move_rows(const PcState *s, const PcRules *r, const PcGeom *g, PcMoveScratch *mv, PcLanes L, uint8_t *out) {
    int n = s->n;
    int32_t cp = s->current_player;
    int o[18];
    int any = 0;
    pc_masks_family_offsets(n, o);
    for (int i = L.lane; i < n && !any; i += L.count) any = pc_masks_movable(s, r, i);
    if (!pc_lanes_any(L, any)) return;
    uint16_t reached[PC_MAXN];
    if (L.count == 1) {
        pc_kernels_movement_flags(s, cp, g, mv->flags);
        for (int i = 0; i < n; i++) {
            if (pc_masks_movable(s, r, i)) {
                int base = o[0] + i * n;
                int count = pc_kernels_reach(s, i, r, g, mv, reached);
                for (int k = 0; k < count; k++) pc_mask_set(out, base + (int)reached[k]);
            }
        }
        return;
    }
    pc_kernels_movement_flags_lanes(s, cp, g, mv->flags, L);
    PcMoveScratch local;
    pc_kernels_move_scratch_init(&local);
    for (int i = 0; i < n; i++) local.flags[i] = mv->flags[i];
    for (int i = L.lane; i < n; i += L.count) {
        if (pc_masks_movable(s, r, i)) {
            int base = o[0] + i * n;
            int count = pc_kernels_reach(s, i, r, g, &local, reached);
            for (int k = 0; k < count; k++) pc_lanes_mask_set(L, out, base + (int)reached[k]);
        }
    }
}

/* Attack rows, one unit per lane (attack_targets is a pure function of the state). */
PC_FN void pc_masks_attack_rows(const PcState *s, const PcRules *r, const PcGeom *g, PcLanes L, uint8_t *out) {
    int n = s->n;
    const PcUnits *u = &s->units;
    int32_t cp = s->current_player;
    int o[18];
    uint16_t targets[PC_MAXN];
    pc_masks_family_offsets(n, o);
    for (int i = L.lane; i < n; i += L.count) {
        if (u->owner[i] == cp && !u->attacked[i]) {
            int base = o[1] + i * n;
            int count = pc_military_attack_targets(s, i, r, g, targets);
            for (int k = 0; k < count; k++) pc_lanes_mask_set(L, out, base + (int)targets[k]);
        }
    }
}

PC_FN void pc_masks_summon_rows(const PcState *s, const PcRules *r, PcSummonCtx *ctx, PcLanes L, uint8_t *out) {
    int n = s->n;
    const PcUnits *u = &s->units;
    int32_t cp = s->current_player;
    int o[18];
    pc_masks_family_offsets(n, o);
    pc_military_summon_prepare_lanes(ctx, s, r, L);
    for (int i = L.lane; i < n; i += L.count) {
        if (s->cities.owner[i] == cp || u->owner[i] == cp) {
            pc_u128 bits = pc_u128_and(pc_military_summon_row(s, ctx, i), r->supported_bits);
            while (pc_u128_any(bits)) {
                int k = pc_u128_lowest(bits);
                pc_lanes_mask_set(L, out, o[2] + i * 128 + k);
                pc_u128_clear(&bits, k);
            }
        }
    }
}

/* Build rows; leaves economic_tiles(s) in `ctx->econ` (and its list) for the economy rows. */
PC_FN void pc_masks_build_rows(const PcState *s, const PcRules *r, const PcGeom *g, PcBuildCtx *ctx, PcLanes L, uint8_t *out) {
    int o[18];
    pc_masks_family_offsets(s->n, o);
    pc_structures_build_ctx_prepare_lanes(ctx, s, r, g, L);
    /* outside the economy only the neutral road and the neutral bridge can be built */
    for (int i = L.lane; i < s->n; i += L.count) {
        if (!ctx->econ[i] && pc_structures_neutral_road(s, r, ctx, i)) pc_lanes_mask_set(L, out, o[3] + i * 128 + 3);
        if (!ctx->econ[i] && pc_structures_neutral_bridge(s, r, ctx, i)) pc_lanes_mask_set(L, out, o[3] + i * 128 + 48);
    }
    for (int e = L.lane; e < ctx->econ_n; e += L.count) {
        int i = (int)ctx->econ_list[e];
        pc_u128 bits = pc_u128_andnot(pc_structures_build_row(s, r, g, ctx, i), pc_u128_bit(37));
        while (pc_u128_any(bits)) {
            int k = pc_u128_lowest(bits);
            pc_lanes_mask_set(L, out, o[3] + i * 128 + k);
            pc_u128_clear(&bits, k);
        }
    }
}

/* Harvest, forest and destroy (the economy tiles, as listed by BuildCtx::prepare). */
PC_FN void pc_masks_economy_rows(const PcState *s, const uint16_t *econ, int econ_n, PcLanes L, uint8_t *out) {
    int o[18];
    const PcTiles *t = &s->tiles;
    int pi = pc_p(s);
    const uint8_t *tech = s->players.tech[pi];
    int32_t stars = s->players.stars[pi];
    pc_masks_family_offsets(s->n, o);
    for (int e = L.lane; e < econ_n; e += L.count) {
        int i = (int)econ[e];
        int32_t resource = t->resource[i], structure = t->structure[i], terrain = t->terrain[i];
        int unlocked = (resource == 1 && tech[15]) || (resource == 3 && tech[10]) || (resource == 6 && tech[6]);
        if (structure == 0 && unlocked && stars >= 2) pc_lanes_mask_set(L, out, o[5] + i);
        /* forest: clear (3 stars), burn, grow */
        if (structure == 0 || structure == 3) {
            if (terrain == 5 && tech[9] && stars >= 3) pc_lanes_mask_set(L, out, o[6] + i * 3);
            if (terrain == 5 && tech[16]) pc_lanes_mask_set(L, out, o[6] + i * 3 + 1);
            if (terrain == 3 && tech[19] && stars >= 5) pc_lanes_mask_set(L, out, o[6] + i * 3 + 2);
        }
        if (pc_structures_destructible(structure) && tech[3]) pc_lanes_mask_set(L, out, o[13] + i);
    }
}

/* Capture, recover, promote, disband, heal (the mover's units only). */
PC_FN void pc_masks_unit_rows(const PcState *s, const PcRules *r, const PcGeom *g, PcLanes L, uint8_t *out) {
    int o[18];
    const PcUnits *u = &s->units;
    int32_t cp = s->current_player;
    uint8_t tech2 = s->players.tech[pc_p(s)][2];
    pc_masks_family_offsets(s->n, o);
    for (int i = L.lane; i < s->n; i += L.count) {
        if (u->owner[i] != cp) continue;
        int idle = !u->moved[i] && !u->attacked[i];
        if (idle && pc_capture_capturable(s, i)) pc_lanes_mask_set(L, out, o[8] + i);
        /* recover row: 0 recover, 1 disembark in place (DisembarkCommand) */
        if (idle && u->health[i] < pc_kernels_max_health(s, r, i)) pc_lanes_mask_set(L, out, o[9] + i * PC_UNIT_ROW);
        if (pc_military_can_disembark(s, r, i)) pc_lanes_mask_set(L, out, o[9] + i * PC_UNIT_ROW + 1);
        if (pc_military_promotable(s, r, i)) pc_lanes_mask_set(L, out, o[10] + i);
        if (idle && tech2) pc_lanes_mask_set(L, out, o[11] + i);
        if (idle && u->kind[i] == 10) {
            int any = 0;
            for (int k = 0; k < g->adj_n[i] && !any; k++) {
                int j = (int)g->adj[i * 8 + k];
                any = u->owner[j] == cp && u->health[j] < pc_kernels_max_health(s, r, j);
            }
            if (any) pc_lanes_mask_set(L, out, o[12] + i);
        }
    }
}

/* Emit every true entry of actions.legal_masks (CpuGames: resign forced False when no_resign) into the zeroed packed
 * buffer `out` (pc_masks_packed_width(n) bytes) on lanes L (pc_lanes_one() for the serial engine; with more lanes
 * `out` must be 4-byte aligned and its words this game's alone, see pc_lanes_mask_set). no_resign: never resign.
 * no_peace: never propose or accept peace; a pending request can only be declined. acc: per-phase profile counters
 * (PC_PH_MASK_*, pc.h) or NULL; only a -DPC_PROFILE build reads the clock. Returns with the lanes synchronized.
 * Lane 0's own writes (diplomacy, research, end, resign) also go through pc_lanes_mask_set: a family's first byte can
 * share a 32-bit word with the previous family's last, which other lanes may still be OR-ing atomically. */
PC_FN void pc_masks_into_lanes(const PcState *s, const PcRules *r, const PcGeom *g, uint8_t no_resign, uint8_t no_peace,
                               PcMaskScratch *ms, uint8_t *out, PcLanes L, pc_prof_t *acc) {
    int o[18];
    int active = s->core_supported && !s->terminated && !s->truncated;
    (void)acc;
#ifdef PC_PROFILE
    pc_prof_t t_prof = pc_prof_now();
    /* a lap closes a phase for the whole warp: sync, then lane 0 reads the clock */
#define PC_MASK_LAP(k) do { pc_lanes_sync(L); if (L.lane == 0) PC_PROF_LAP(acc, t_prof, (k)); } while (0)
#else
#define PC_MASK_LAP(k) ((void)0)
#endif
    if (!active) {
        PC_MASK_LAP(PC_PH_MASK_PRE);
        return;
    }
    pc_masks_family_offsets(s->n, o);
    {
        int32_t sender = pc_diplomacy_pending_sender(s);
        if (sender >= 0) {
            /* a pending peace request is answered before anything else (accept or decline; resigning stays) */
            if (L.lane == 0) {
                int base = o[14] + pc_diplomacy_seat_block(s, sender) * PC_DIPLOMACY_BLOCK;
                pc_lanes_mask_set(L, out, base + 1);              /* decline */
                if (!no_peace) pc_lanes_mask_set(L, out, base + 2);   /* accept */
                if (!no_resign && s->np == 2) pc_lanes_mask_set(L, out, o[16]);
            }
            pc_lanes_sync(L);
            PC_MASK_LAP(PC_PH_MASK_PRE);
            return;
        }
    }
    /* reward is gated by `active` only; any reward entry is also State::pending_reward, which gates the rest */
    if (pc_masks_reward_rows(s, L, out)) {
        pc_lanes_sync(L);
        PC_MASK_LAP(PC_PH_MASK_PRE);
        return;
    }
    PC_MASK_LAP(PC_PH_MASK_PRE);
    pc_masks_move_rows(s, r, g, &ms->mv, L, out);
    PC_MASK_LAP(PC_PH_MASK_MOVE);
    pc_masks_attack_rows(s, r, g, L, out);
    PC_MASK_LAP(PC_PH_MASK_ATTACK);
    pc_masks_summon_rows(s, r, &ms->summon, L, out);
    PC_MASK_LAP(PC_PH_MASK_SUMMON);
    pc_masks_build_rows(s, r, g, &ms->build, L, out);
    PC_MASK_LAP(PC_PH_MASK_BUILD);
    {
        pc_u128 bits = pc_kernels_research_bits_lanes(s, r, L);
        if (L.lane == 0) {
            while (pc_u128_any(bits)) {
                int k = pc_u128_lowest(bits);
                pc_lanes_mask_set(L, out, o[4] + k);
                pc_u128_clear(&bits, k);
            }
        }
    }
    PC_MASK_LAP(PC_PH_MASK_RESEARCH);
    pc_masks_economy_rows(s, ms->build.econ_list, ms->build.econ_n, L, out);
    PC_MASK_LAP(PC_PH_MASK_ECONOMY);
    pc_masks_unit_rows(s, r, g, L, out);
    PC_MASK_LAP(PC_PH_MASK_UNIT);
    if (L.lane == 0) {
        /* one block per other seat, by turn-order offset from the mover (a two-seat game: block 0 only) */
        for (int b = 0; b < s->np - 1; b++) {
            int32_t seat = pc_diplomacy_block_seat(s, b);
            int capital = 0;
            uint8_t has_capital = 0;
            int32_t targets[PC_DIPLOMACY_BLOCK];
            pc_diplomacy_capital_of(s, seat, &capital, &has_capital);
            targets[0] = -1; targets[1] = 0; targets[2] = 1; targets[3] = (int32_t)capital; targets[4] = -1; targets[5] = -1;
            for (int k = 0; k < PC_DIPLOMACY_BLOCK; k++) {
                if (no_peace && k == 4) continue;   /* propose peace */
                if (pc_diplomacy_legal_with(s, seat, targets[k], pc_diplomacy_ability(k), capital, has_capital))
                    pc_lanes_mask_set(L, out, o[14] + b * PC_DIPLOMACY_BLOCK + k);
            }
        }
        pc_lanes_mask_set(L, out, o[15]);
        /* resigning exists only in two-seat games */
        if (!no_resign && s->np == 2) pc_lanes_mask_set(L, out, o[16]);
    }
    pc_lanes_sync(L);
    PC_MASK_LAP(PC_PH_MASK_DIPLO);
#undef PC_MASK_LAP
}

PC_FN void pc_masks_into(const PcState *s, const PcRules *r, const PcGeom *g, uint8_t no_resign, uint8_t no_peace,
                         PcMaskScratch *ms, uint8_t *out) {
    pc_masks_into_lanes(s, r, g, no_resign, no_peace, ms, out, pc_lanes_one(), (pc_prof_t *)0);
}

#ifdef __cplusplus
}
#endif
#endif /* PC_MASKS_H */
