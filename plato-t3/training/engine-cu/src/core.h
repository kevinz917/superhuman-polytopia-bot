/* core.h - port of engine-rs/src/core.rs (gpu/core.py: the atomic action dispatcher, end of turn, terminal rewards;
 * gpu/actions.py::decode) plus mapgen::initial_state (engine-rs/src/mapgen.rs; the map generator itself stays in Rust
 * on the CPU and hands over terrain/structure/resource/capitals layouts).
 *
 * Status codes PC_OK.. and PcAction are in pc.h. Everything here is a line-by-line port; prof:: and cov:: are dropped. */
#ifndef PC_CORE_H
#define PC_CORE_H

#include "pc.h"
#include "kernels.h"      /* PcMoveScratch */
#include "pathfinding.h"  /* PcPathScratch */
#include "diplomacy.h"
#include "economy.h"
#include "structures.h"   /* PcBuildCtx */
#include "capture.h"
#include "infiltration.h"
#include "military.h"     /* PcSummonCtx */
#include "history.h"
#include "masks.h"
#include "polaris.h"      /* FreezeArea / break ice commands, restore_climate, sync_ice_banks (Polaris) */
#include "aquarion.h"     /* Canal / clear tile effect commands (Aquarion step 4) */
#include "cymanti.h"      /* clear tile effect Algae (Cymanti step 3) */
#include "tribes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Entries per row of the recover family: 0 recover command, 1 disembark command (added 2026-10-31, appended so the
 * recover entry keeps its target index; docs/rules-parity-2-actions.md), 2 freeze area command, 3 break ice command
 * (action layout 3, Polaris 2026-11-09; appended the same way), 4 build command(Canal) on the unit's tile, 5
 * clear tile effect command(Flooded) (action layout 4, Aquarion 2026-11-09), 6 clear tile effect command(Algae), 7
 * explode command, 8 swarm command (action layout 5, Cymanti 2026-11-09). */
#ifndef PC_UNIT_ROW
#define PC_UNIT_ROW 9
#endif
/* Action parameters of freeze area command, break ice command, the Canal and clear tile effect Flooded (Action(3, tile,
 * -1, p)). */
#define PC_FREEZE_AREA 24
#define PC_BREAK_ICE 25
#define PC_CANAL 26
#define PC_CLEAR_FLOOD 27
/* Cymanti: clear tile effect Algae, Explode, Swarm */
#define PC_CLEAR_ALGAE 28
#define PC_EXPLODE 29
#define PC_SWARM 30
/* Width of the diplomatic family: one block of diplomacy::BLOCK entries per other seat (two-seat games use the
 * first block only, which is the earlier six-entry layout). */
#ifndef PC_DIPLOMACY_BLOCK
#define PC_DIPLOMACY_BLOCK 6
#endif
#ifndef PC_DIPLOMACY_BLOCKS
#define PC_DIPLOMACY_BLOCKS (PC_MAXP - 1)
#endif
#ifndef PC_DIPLOMATIC_WIDTH
#define PC_DIPLOMATIC_WIDTH (PC_DIPLOMACY_BLOCK * PC_DIPLOMACY_BLOCKS)
#endif
/* Action parameter of disembark command (Action(3, tile, -1, 23)): a boat on a bridge lands where it stands. */
#define PC_DISEMBARK 23

/* provided by kernels.h */
PC_FN float pc_kernels_max_health(const PcState *s, const PcRules *r, int i);
PC_FN int32_t pc_kernels_research_cost(const PcState *s, const PcRules *r, int k);
PC_FN void pc_kernels_research_mask(const PcState *s, const PcRules *r, uint8_t out[128]);
PC_FN void pc_kernels_movement_targets(const PcState *s, int source, const PcRules *r, const PcGeom *g,
                                       PcMoveScratch *ms, uint8_t *out);
PC_FN void pc_kernels_update_exploration(PcState *s, int32_t player, const PcGeom *g);

/* provided by pathfinding.h (PcPathScratch: int path[PC_MAXN], int path_len, uint8_t reveal[PC_MAXN] assumed) */
PC_FN uint8_t pc_pathfinding_search(const PcState *s, int source, int target, const PcRules *r, const PcGeom *g,
                                    PcPathScratch *ps);
PC_FN void pc_pathfinding_reveal_tiles(const PcState *s, int source, int from, int extra /* -1 none */, const PcRules *r,
                                       PcPathScratch *ps);

/* provided by military.h */
PC_FN void pc_military_relocate(PcState *s, int source, int target, const PcRules *r, const PcGeom *g,
                                uint8_t involuntary, const uint8_t *reveal /* NULL none */, PcStructScratch *sc);
PC_FN int32_t pc_military_direction(int size, int from, int to);
PC_FN int32_t pc_military_spawn_direction(int size, int tile);
PC_FN void pc_military_attack_mask(const PcState *s, int source, const PcRules *r, const PcGeom *g, uint8_t *out);
PC_FN void pc_military_summon_prepare(PcSummonCtx *ctx, const PcState *s, const PcRules *r);   /* summon ctx:prepare */
PC_FN pc_u128 pc_military_summon_row(const PcState *s, const PcSummonCtx *ctx, int t);
PC_FN uint8_t pc_military_promotable(const PcState *s, const PcRules *r, int i);
PC_FN uint8_t pc_military_can_disembark(const PcState *s, const PcRules *r, int i);
PC_FN void pc_military_convert(PcState *s, int source, int target, const PcRules *r, const PcGeom *g);
PC_FN void pc_military_attack(PcState *s, int source, int target, const PcRules *r, const PcGeom *g, PcStructScratch *sc);
PC_FN void pc_military_upgrade(PcState *s, int source, int kind, const PcRules *r, const PcGeom *g);
PC_FN void pc_military_summon(PcState *s, int source, int kind, const PcRules *r, const PcGeom *g, uint8_t costs);
PC_FN void pc_military_promote(PcState *s, int source, const PcRules *r);
PC_FN void pc_military_disband(PcState *s, int source, const PcRules *r);
PC_FN void pc_military_heal_others(PcState *s, int source, const PcRules *r, const PcGeom *g);
PC_FN void pc_military_disembark(PcState *s, int i, const PcRules *r, const PcGeom *g);
PC_FN void pc_military_grow(PcState *s, int i, const PcRules *r, const PcGeom *g);

/* provided by economy.h */
PC_FN void pc_economy_harvest_mask(const PcState *s, uint8_t *out);
PC_FN void pc_economy_forest_masks(const PcState *s, uint8_t *out);      /* out[t*3 + k] */
PC_FN void pc_economy_reward_mask(const PcState *s, uint8_t *out);       /* out[t*9 + k] */
PC_FN void pc_economy_harvest(PcState *s, int target);
PC_FN void pc_economy_forest(PcState *s, int target, int32_t ability);
PC_FN void pc_economy_reward(PcState *s, int target, int32_t kind, const PcRules *r, const PcGeom *g, PcStructScratch *sc);
PC_FN void pc_economy_start_level_ups(PcState *s, int32_t owner);

/* provided by structures.h (the route/level functions take the module's PcStructScratch: the port's replacement of
 * the Vecs the Rust allocates inside them) */
PC_FN uint8_t pc_structures_build_legal(const PcState *s, const PcRules *r, const PcGeom *g, PcBuildCtx *ctx, int tile,
                                        int kind);
PC_FN void pc_structures_destroy_mask(const PcState *s, uint8_t *out);
PC_FN void pc_structures_build(PcState *s, int target, int kind, const PcRules *r, const PcGeom *g, PcStructScratch *sc);
PC_FN void pc_structures_destroy(PcState *s, int target, const PcRules *r, const PcGeom *g, PcStructScratch *sc);
PC_FN void pc_structures_remember_monuments(PcState *s, const PcRules *r, const PcGeom *g);
PC_FN void pc_structures_update_routes(PcState *s, const PcRules *r, PcStructScratch *sc);
PC_FN void pc_structures_connections(PcState *s, int32_t owner, const PcGeom *g, uint8_t native);
PC_FN void pc_structures_start_turn(PcState *s, int32_t owner);
PC_FN void pc_structures_attract(PcState *s, int32_t owner, const PcRules *r);
PC_FN int32_t pc_structures_income(const PcState *s, int32_t owner, const PcRules *r, const PcGeom *g);
PC_FN void pc_structures_refresh_levels(PcState *s, int32_t owner, const PcRules *r, const PcGeom *g, PcStructScratch *sc);

/* provided by capture.h */
PC_FN void pc_capture_capture_mask(const PcState *s, uint8_t *out);
PC_FN void pc_capture_city(PcState *s, int source, const PcRules *r, const PcGeom *g, PcStructScratch *sc);
PC_FN void pc_capture_ruin(PcState *s, int source, const PcRules *r, const PcGeom *g, PcStructScratch *sc);
PC_FN void pc_capture_starfish(PcState *s, int source);

/* provided by infiltration.h */
PC_FN void pc_infiltration_execute(PcState *s, int source, int target, const PcRules *r, const PcGeom *g);

/* provided by diplomacy.h */
PC_FN int32_t pc_diplomacy_block_seat(const PcState *s, int b);
PC_FN void pc_diplomacy_capital_of(const PcState *s, int32_t q, int *capital, uint8_t *has_capital);
PC_FN uint8_t pc_diplomacy_legal(const PcState *s, int32_t source, int32_t target, int32_t ability);
PC_FN uint8_t pc_diplomacy_peace_pending(const PcState *s);
PC_FN void pc_diplomacy_hostile(PcState *s, int32_t victim);
PC_FN void pc_diplomacy_capital_vision(PcState *s);
PC_FN void pc_diplomacy_start_turn(PcState *s, int32_t owner);
PC_FN void pc_diplomacy_execute(PcState *s, int32_t source, int32_t target, int32_t ability, const PcRules *r,
                                const PcGeom *g, PcStructScratch *sc);

PC_FN int32_t pc_diplomacy_ability(int k);                     /* diplomacy::ABILITIES[k] */

/* provided by history.h */
PC_FN void pc_history_end_of_turn(PcState *s, int32_t player, const PcRules *r);
PC_FN void pc_history_start_of_turn(PcState *s, int32_t player, const PcRules *r);

/* Catalog widths of the 17 mask families (rl/actions.py::shapes) for board tiles n. */
PC_FN void pc_core_family_widths(int n, int out[17]) {
    out[0] = n * n; out[1] = n * n; out[2] = n * 128; out[3] = n * 128; out[4] = 128; out[5] = n; out[6] = n * 3;
    out[7] = n * 9; out[8] = n; out[9] = n * PC_UNIT_ROW; out[10] = n; out[11] = n; out[12] = n; out[13] = n;
    out[14] = PC_DIPLOMATIC_WIDTH; out[15] = 1; out[16] = 1;
}

/* Row widths used by rl/actions.py::flatten_choice (second dim of `shapes`). */
PC_FN void pc_core_family_row_widths(int out[17]) {
    out[0] = 0; out[1] = 0; out[2] = 128; out[3] = 128; out[4] = 128; out[5] = 1; out[6] = 3; out[7] = 9; out[8] = 1;
    out[9] = PC_UNIT_ROW; out[10] = 1; out[11] = 1; out[12] = 1; out[13] = 1; out[14] = PC_DIPLOMATIC_WIDTH;
    out[15] = 1; out[16] = 1;
}

/* Rust `x as i32` on an i64: the low 32 bits */
PC_FN int32_t pc_core_i64_to_i32(int64_t x) { return (int32_t)(uint32_t)(uint64_t)x; }

PC_FN PcAction pc_core_act(int32_t kind, int64_t source, int64_t target, int64_t parameter) {
    PcAction a;
    a.kind = kind;
    a.source = pc_core_i64_to_i32(source);
    a.target = pc_core_i64_to_i32(target);
    a.parameter = pc_core_i64_to_i32(parameter);
    return a;
}

/* rl.actions.flatten_choice followed by gpu.actions.decode. */
PC_FN PcAction pc_core_decode(const PcState *s, int32_t family, int32_t source, int32_t target) {
    int64_t n = (int64_t)s->n;
    int widths[17], rows[17];
    int64_t offsets[18];
    int f, group;
    int64_t flat, at;
    pc_core_family_widths(s->n, widths);
    pc_core_family_row_widths(rows);
    rows[0] = s->n;
    rows[1] = s->n;
    offsets[0] = 0;
    for (int i = 0; i < 17; i++) offsets[i + 1] = offsets[i] + (int64_t)widths[i];
    f = (int)pc_clamp(family, 0, 16);
    flat = offsets[f] + (int64_t)source * (int64_t)rows[f] + (int64_t)target;
    group = 0;
    for (int g = 1; g <= 17; g++)
        if (flat >= offsets[g]) group++;
    if (group > 16) group = 16;
    at = flat - offsets[group];
    switch (group) {
        case 0: return pc_core_act(1, at / n, at % n, 0);
        case 1: return pc_core_act(2, at / n, at % n, 0);
        case 2: return pc_core_act(4, at / 128, -1, at % 128);
        case 3: return pc_core_act(6, -1, at / 128, at % 128);
        case 4: return pc_core_act(7, -1, -1, at);
        case 5: return pc_core_act(5, -1, at, 0);
        case 6: return pc_core_act(3, -1, at / 3, at % 3 + 1);
        case 7: return pc_core_act(9, -1, at / 9, at % 9);
        case 8: return pc_core_act(8, at, -1, 0);
        /* recover row: 0 recover, 1 disembark in place (ability 23), 2 freeze area, 3 break ice, 4 canal, 5 clear flood,
         * 6 clear algae, 7 explode, 8 swarm */
        case 9: {
            static const int32_t params[9] = {7, PC_DISEMBARK, PC_FREEZE_AREA, PC_BREAK_ICE, PC_CANAL, PC_CLEAR_FLOOD,
                                              PC_CLEAR_ALGAE, PC_EXPLODE, PC_SWARM};
            return pc_core_act(3, at / PC_UNIT_ROW, -1, params[at % PC_UNIT_ROW]);
        }
        case 10: return pc_core_act(3, at, -1, 16);
        case 11: return pc_core_act(3, at, -1, 8);
        case 12: return pc_core_act(3, at, -1, 9);
        case 13: return pc_core_act(3, -1, at, 4);
        case 14: {
            /* block b addresses the seat b + 1 places after the mover (0 when there is none: never legal) */
            int64_t targets[PC_DIPLOMACY_BLOCK];
            int capital = 0;
            uint8_t has_capital = 0;
            int ac, b, k;
            int32_t seat;
            if (at < 0) at = 0;
            if (at > PC_DIPLOMATIC_WIDTH - 1) at = PC_DIPLOMATIC_WIDTH - 1;
            ac = (int)at;
            b = ac / PC_DIPLOMACY_BLOCK;
            k = ac % PC_DIPLOMACY_BLOCK;
            seat = pc_diplomacy_block_seat(s, b);
            pc_diplomacy_capital_of(s, seat, &capital, &has_capital);
            targets[0] = -1; targets[1] = 0; targets[2] = 1; targets[3] = (int64_t)capital; targets[4] = -1; targets[5] = -1;
            return pc_core_act(3, (int64_t)seat, targets[k], (int64_t)pc_diplomacy_ability(k));
        }
        case 15: return pc_core_act(10, -1, -1, 0);
        default: return pc_core_act(11, -1, -1, 0);
    }
}

/* get reward: the star gift for meeting `other` (and the v123 "away stars" with two seats). */
PC_FN int32_t pc_core_meet_gift(const PcState *s, int other) {
    int32_t score = s->players.score[other];
    int32_t gift = 1 + 2 * (score > 0 ? pc_min(pc_div_euclid(score - 1, 1000) + 1, 5) : 0);
    /* game version >= 123 (should add away stars): in a two-player game, meeting player 1 pays 2 more */
    if (s->np == 2 && other == 0) gift += 2;
    return gift;
}

/* meet action for seat `p` meeting seat `other`: the star gift and `known`, once. */
PC_FN void pc_core_meet_pair(PcState *s, int p, int other) {
    if (p == other || s->players.known[p][other]) return;
    s->players.stars[p] += pc_core_meet_gift(s, other);
    s->players.known[p][other] = 1;
}

/* The mover meets every seat it can see: a visible unit or an owned city on an explored tile
 * (explore action, check tribe connection default). */
PC_FN void pc_core_meet(PcState *s) {
    int p, n;
    if (!s->fog) return;   /* `new` below is false: nothing changes */
    p = pc_p(s);
    n = s->n;
    /* every other seat in order; with two seats exactly the reference's single opponent */
    for (int other = 0; other < s->np; other++) {
        int32_t o;
        int sees = 0;
        if (other == p) continue;
        if (s->players.known[p][other]) continue;
        o = (int32_t)other + 1;
        for (int t = 0; t < n && !sees; t++) {
            sees = s->tiles.explored[p][t]
                && ((s->units.owner[t] == o && !s->units.invisible[t]) || (s->tiles.structure[t] == 1 && s->cities.owner[t] == o));
        }
        if (sees) pc_core_meet_pair(s, p, other);
    }
}

PC_FN void pc_core_research(PcState *s, PcAction a, const PcRules *r) {
    int p = pc_p(s);
    int k = (int)pc_clamp(a.parameter, 0, 127);
    int32_t cost = pc_kernels_research_cost(s, r, k);
    s->players.stars[p] -= cost;
    s->players.score[p] = pc_wadd(s->players.score[p], r->tech_tier[k] * 100);
    s->players.tech[p][k] = 1;
    s->players.tech_turn[p][k] = s->turn;
    pc_core_meet(s);
}

PC_FN void pc_core_recover(PcState *s, int src, const PcRules *r) {
    int32_t cp = s->current_player;
    int32_t territory_owner = s->tiles.owner[src];
    int friendly = territory_owner == cp || (territory_owner > 0 && s->players.peace[pc_p(s)][pc_pclip(s, territory_owner)]);
    float amount = friendly ? 4.0f : 2.0f;
    float mh = pc_kernels_max_health(s, r, src);
    PcUnits *u = &s->units;
    float hp = u->poison[src] ? u->health[src] : pc_fminf(u->health[src] + amount, mh);
    u->health[src] = hp;
    u->poison[src] = 0;
    u->moved[src] = 1;
    u->attacked[src] = 1;
    u->move_lock[src] = 0;
    pc_core_meet(s);
}

/* `st`: the routes scratch of an AutoFreeze arrival (relocate) */
PC_FN void pc_core_move_adjacent(PcState *s, int source, int target, const PcRules *r, const PcGeom *g, PcPathScratch *ps,
                                 PcStructScratch *st) {
    int32_t cp = s->current_player;
    uint8_t found;
    int extra;
    if (s->units.owner[target] != 0 && s->units.owner[target] != cp) {
        /* move command: a (hidden) unit on the target is revealed (reveal action, with its meet action); the
         * mover stays where it is and keeps its move */
        s->units.invisible[target] = 0;
        /* the reveal action has Origin = the mover's tile; reveal action turns
         * the unit on Origin toward the revealed one (set unit direction) */
        s->units.direction[source] = pc_military_direction(s->size, source, target);
        pc_core_meet(s);
        return;
    }
    /* move action: the unit goes along get path's path, which treats hidden units as free tiles (they
     * stay hidden), and explores from every tile of it */
    found = pc_pathfinding_search(s, source, target, r, g, ps);
    extra = found ? -1 : target;
    pc_pathfinding_reveal_tiles(s, source, 0, extra, r, ps);
    uint8_t trail = r->algae_trail[pc_clamp(s->units.kind[source], 0, 127)];
    pc_military_relocate(s, source, target, r, g, 0, ps->reveal, st);
    /* Cymanti: the Island's Algae trail on the path's inner tiles (relocate did both ends) */
    if (trail && found && ps->path_len > 1) pc_cymanti_algae_trail(s, ps->path + 1, ps->path_len - 1);
    /* move action faces the unit along the path's last step */
    if (ps->path_len > 1) s->units.direction[target] = pc_military_direction(s->size, ps->path[1], target);
    /* move action: ending on another player's city counts as an attack on them */
    if (s->tiles.structure[target] == 1 && pc_foreign(s->cities.owner[target], cp)) {
        int32_t victim = s->cities.owner[target];
        pc_diplomacy_hostile(s, victim);
    }
    pc_core_meet(s);
}

/* `st` is the structures scratch (refresh_levels); the Rust signature is (s, r, g). */
/* end_turn's get heal options test: an own unit that is damaged or Poisoned */
PC_FN int pc_core_heal_option(const PcState *s, const PcRules *r, int32_t owner, int j) {
    return s->units.owner[j] == owner && (s->units.health[j] < pc_kernels_max_health(s, r, j) || s->units.poison[j]);
}

/* heal action: a Poisoned unit is cured instead; else health + amount, capped */
PC_FN void pc_core_heal_unit(PcState *s, const PcRules *r, int i, float amount) {
    if (s->units.owner[i] == 0) return;
    if (s->units.poison[i]) {
        s->units.poison[i] = 0;
    } else {
        s->units.health[i] = pc_fminf(s->units.health[i] + amount, pc_kernels_max_health(s, r, i));
    }
}

PC_FN void pc_core_end_turn(PcState *s, const PcRules *r, const PcGeom *g, PcStructScratch *st) {
    int n = s->n;
    int32_t owner = s->current_player;
    int old = pc_p(s);
    int32_t new_owner, turn;
    int native, limit, p, no;
    int32_t pacifist, income;
    uint8_t frozen[PC_MAXN];
    /* observation history: what the seat sees as it leaves its turn (history.rs; observation version 3) */
    pc_history_end_of_turn(s, owner, r);
    /* end turn command's pushes, tile by tile (core.rs end_turn): rest heal (0), (Cymanti) a Mycelium's
     * heal action(40) on its unit (1) and heal others action (2), a Decomposing refund + destroy (3); run last first */
    {
        uint8_t ev_kind[4 * PC_MAXN];
        uint16_t ev_tile[4 * PC_MAXN];
        int ne = 0;
        for (int i = 0; i < n; i++) {
            const PcUnits *u = &s->units;
            /* can recover: idle, no leader */
            if (u->owner[i] == owner && !u->moved[i] && !u->attacked[i] && u->leader[i] == 0) { ev_kind[ne] = 0; ev_tile[ne] = (uint16_t)i; ne++; }
            if (s->tiles.owner[i] == owner && s->tiles.structure[i] == PC_CYMANTI_MYCELIUM) {
                int any = pc_core_heal_option(s, r, owner, i);
                if (any) { ev_kind[ne] = 1; ev_tile[ne] = (uint16_t)i; ne++; }
                for (int k = 0; k < g->adj_n[i] && !any; k++) any = pc_core_heal_option(s, r, owner, (int)g->adj[i * 8 + k]);
                if (any) { ev_kind[ne] = 2; ev_tile[ne] = (uint16_t)i; ne++; }
            }
            if (s->tiles.owner[i] == owner && s->tiles.decomposing[i] && s->tiles.structure[i] != 0
                && !(s->tiles.structure[i] == 48 && s->units.owner[i] != 0)) { ev_kind[ne] = 3; ev_tile[ne] = (uint16_t)i; ne++; }
        }
        for (int e = ne - 1; e >= 0; e--) {
            int i = (int)ev_tile[e];
            if (ev_kind[e] == 0) {
                int32_t to = s->tiles.owner[i];
                int friendly = to == owner || (to > 0 && s->players.peace[old][pc_pclip(s, to)]);
                pc_core_heal_unit(s, r, i, friendly ? 4.0f : 2.0f);
            } else if (ev_kind[e] == 1) {
                pc_core_heal_unit(s, r, i, 4.0f);
            } else if (ev_kind[e] == 2) {
                uint8_t opt[8];
                int m = g->adj_n[i];
                for (int k = 0; k < m; k++) opt[k] = (uint8_t)pc_core_heal_option(s, r, owner, (int)g->adj[i * 8 + k]);
                for (int k = m - 1; k >= 0; k--) {
                    if (opt[k]) pc_core_heal_unit(s, r, (int)g->adj[i * 8 + k], 4.0f);
                }
            } else {
                int p0 = pc_p(s);
                s->players.stars[p0] += pc_max(r->structures.cost[pc_clamp(s->tiles.structure[i], 0, 127)], 0);
                pc_structures_destroy(s, i, r, g, st);
            }
        }
    }
    /* end turn command (Elyrion): every unit of the owner with Grow and age >= growth rate grows, frozen or
     * not (engine-rs core.rs end_turn) */
    for (int i = 0; i < n; i++) {
        int kind = (int)s->units.kind[i];
        if (s->units.owner[i] == owner && r->grow[kind] && r->grows_into[kind] >= 0
            && s->turn - s->units.created[i] >= r->growth_rate[kind]) pc_military_grow(s, i, r, g);
    }
    /* end turn command: the owner's units lose Frozen after the heal test (a frozen unit, moved and attacked,
     * does not heal) */
    for (int i = 0; i < n; i++) {
        if (s->units.owner[i] == owner) {
            s->units.frozen[i] = 0;
            /* Cymanti: a DoubleAttack unit gets double ready */
            if (r->double_attack[pc_clamp(s->units.kind[i], 0, 127)]) s->units.double_ready[i] = 1;
        }
    }
    /* end turn action: the riot of an infiltrated city ends at the end of its owner's turn */
    for (int i = 0; i < n; i++) {
        if (s->cities.riot[i] && s->cities.owner[i] == owner) s->cities.riot[i] = 0;
    }
    if (s->np == 2) {
        const PcPlayers *pl = &s->players;
        int dead[2];
        dead[0] = pl->killed_turn[0] > 0 || pl->resigned_turn[0] > 0;
        dead[1] = pl->killed_turn[1] > 0 || pl->resigned_turn[1] > 0;
        new_owner = dead[2 - owner] ? owner : 3 - owner;
        turn = s->turn + (new_owner == 1 ? 1 : 0);
        native = dead[0] || dead[1];
    } else {
        /* the next seat still in the game, in seat order; a round ends when the order wraps */
        int32_t np = s->np;
        new_owner = owner;
        for (int32_t k = 1; k <= np; k++) {
            int32_t o = (owner - 1 + k) % np + 1;
            if (pc_alive(s, o - 1)) { new_owner = o; break; }
        }
        turn = s->turn + (new_owner <= owner ? 1 : 0);
        native = pc_alive_count(s) <= 1;
    }
    /* pacifist task (engine-rs core.rs): the attack resets to 0, the end-turn bump still adds 1 */
    pacifist = (s->players.attacked[old] ? 0 : s->players.pacifist[old]) + (int32_t)s->players.tech[old][21];
    s->players.pacifist[old] = pacifist;
    s->players.attacked[old] = 0;
    /* end turn action: every cool down relation of the player ending its turn goes back to Neutral (the breaker of a
     * peace is free again from its next turn; the victim's own cool down lasts through the victim's turn) */
    for (int q = 0; q < s->np; q++) {
        s->players.truce_breaker[old][q] = 0;
    }
    s->current_player = new_owner;
    s->turn = turn;
    /* harness truncation after exactly `reference_max_turns` rounds (turns 0..max-1); NO_ROUND_LIMIT never fires */
    limit = turn >= s->reference_max_turns;
    if (limit || native) {
        s->truncated = (uint8_t)(limit && !native);
        s->terminated = (uint8_t)native;
        return;
    }
    for (int i = 0; i < n; i++) frozen[i] = s->units.frozen[i];
    pc_diplomacy_start_turn(s, new_owner);
    /* check diplomacy: receiving a peace request meets its sender (every request of a seat still in) */
    p = pc_p(s);
    for (int q = 0; q < s->np; q++) {
        if (q != p && s->players.peace_offer[p][q] && pc_alive(s, q)) pc_core_meet_pair(s, p, q);
    }
    /* start turn action explores nothing: tiles are explored when units move, train, upgrade or convert and when
     * borders grow */
    pc_core_meet(s);
    pc_structures_start_turn(s, new_owner);
    income = pc_structures_income(s, new_owner, r, g);
    if (!(turn > 0 && income > 0)) income = 0;
    no = (int)(new_owner - 1);
    s->players.stars[no] += income;
    /* start turn action's Attract spawns (Elyrion Sanctuary), after the income they do not count in */
    if (turn > 0) pc_structures_attract(s, new_owner, r);
    /* start turn action (v > 111): after the income, restore climate on every tile the
     * player owns; the Ice Banks' levels */
    for (int t = 0; t < n; t++) {
        if (s->tiles.owner[t] == new_owner) pc_polaris_restore_climate(s, t, new_owner);
    }
    pc_polaris_sync_ice_banks(s);
    if (turn > 0) pc_structures_refresh_levels(s, new_owner, r, g, st);
    pc_economy_start_level_ups(s, new_owner);
    for (int i = 0; i < n; i++) {
        if (s->units.owner[i] == new_owner) {
            s->units.moved[i] = frozen[i];
            s->units.attacked[i] = frozen[i];
            s->units.move_lock[i] = 0;
            if (!frozen[i]) s->units.attacks_performed[i] = 0;
        }
    }
    /* observation history: what changed for the new seat since the end of its last turn */
    pc_history_start_of_turn(s, new_owner, r);
}

/* Record that seat `k` has just left a multi-seat game (killed or resigned): its place in the leaving order. */
PC_FN void pc_core_leave(PcState *s, int k) {
    if (s->np > 2 && s->players.out_order[k] == 0) {
        int32_t c = 0;
        for (int q = 0; q < PC_MAXP; q++) c += s->players.out_order[q] > 0;
        s->players.out_order[k] = c + 1;
    }
}

PC_FN void pc_core_resign(PcState *s) {
    int p = pc_p(s);
    s->players.resigned_turn[p] = pc_max(s->turn, 1);
    if (s->np == 2) {
        s->terminated = 1;
        s->players.killer[p] = 3 - s->current_player;
    } else {
        /* the seat leaves (its pieces stay as obstacles; the turn order skips it); the game ends at one seat left */
        pc_core_leave(s, p);
        s->terminated = (uint8_t)(pc_alive_count(s) <= 1);
    }
}

/* Reward of finishing place `place` (0 = first) among `np` seats: evenly spaced from +1 down to -1, zero-sum. */
PC_FN float pc_core_place_reward(int place, int np) {
    return 1.0f - 2.0f * (float)place / (float)(np - 1);
}

/* Seat-indexed final rewards (rows >= np are 0). Two seats: the reference's win +1 / loss -1, 0 unless terminated.
 * More seats: rank rewards (place_reward), set at termination and at truncation. Seats that left take the
 * places from last upward in the order they left; the seats still in share the top places equally (a seat
 * still in at the round cap ties with every other one still in). */
PC_FN void pc_core_terminal_rewards(const PcState *s, float out[PC_MAXP]) {
    const PcPlayers *p = &s->players;
    int np, alive;
    float shared;
    for (int k = 0; k < PC_MAXP; k++) out[k] = 0.0f;
    if (s->np == 2) {
        int dead[2];
        dead[0] = p->killed_turn[0] > 0 || p->resigned_turn[0] > 0;
        dead[1] = p->killed_turn[1] > 0 || p->resigned_turn[1] > 0;
        if (s->terminated) {
            for (int k = 0; k < 2; k++) {
                int winner = !dead[k] && dead[1 - k];
                out[k] = winner ? 1.0f : (dead[k] ? -1.0f : 0.0f);
            }
        }
        return;
    }
    if (!(s->terminated || s->truncated)) return;
    np = s->np;
    alive = pc_alive_count(s);
    shared = 0.0f;
    for (int k = 0; k < alive; k++) shared += pc_core_place_reward(k, np);
    shared = shared / (float)pc_max(alive, 1);
    for (int k = 0; k < np; k++) {
        out[k] = pc_alive(s, k) ? shared : pc_core_place_reward(np - (int)pc_clamp(p->out_order[k], 1, np), np);
    }
}

/* core.rs Kind, same order */
typedef enum {
    PC_KIND_RESEARCH = 0, PC_KIND_RECOVER, PC_KIND_MOVE, PC_KIND_END, PC_KIND_ATTACK, PC_KIND_SUMMON, PC_KIND_HARVEST,
    PC_KIND_FOREST, PC_KIND_PROMOTE, PC_KIND_DISBAND, PC_KIND_REWARD, PC_KIND_BUILD, PC_KIND_DESTROY, PC_KIND_CAP,
    PC_KIND_HEAL, PC_KIND_DIPLOMATIC, PC_KIND_RESIGN, PC_KIND_DISEMBARK, PC_KIND_FREEZE_AREA, PC_KIND_BREAK_ICE,
    PC_KIND_CANAL, PC_KIND_CLEAR_FLOOD, PC_KIND_CLEAR_ALGAE, PC_KIND_EXPLODE, PC_KIND_SWARM, PC_KIND_UNSUPPORTED
} PcKind;

PC_FN PcKind pc_core_classify(PcAction a) {
    switch (a.kind) {
        case 7: return PC_KIND_RESEARCH;
        case 1: return PC_KIND_MOVE;
        case 10: return PC_KIND_END;
        case 2: return PC_KIND_ATTACK;
        case 4: return PC_KIND_SUMMON;
        case 5: return PC_KIND_HARVEST;
        case 9: return PC_KIND_REWARD;
        case 6: return PC_KIND_BUILD;
        case 8: return PC_KIND_CAP;
        case 11: return PC_KIND_RESIGN;
        case 3:
            switch (a.parameter) {
                case 7: return PC_KIND_RECOVER;
                case 1: case 2: case 3: return PC_KIND_FOREST;
                case 16: return PC_KIND_PROMOTE;
                case 8: return PC_KIND_DISBAND;
                case 4: return PC_KIND_DESTROY;
                case 9: return PC_KIND_HEAL;
                case 18: case 19: case 20: case 21: case 22: return PC_KIND_DIPLOMATIC;
                case PC_DISEMBARK: return PC_KIND_DISEMBARK;
                case PC_FREEZE_AREA: return PC_KIND_FREEZE_AREA;
                case PC_BREAK_ICE: return PC_KIND_BREAK_ICE;
                case PC_CANAL: return PC_KIND_CANAL;
                case PC_CLEAR_FLOOD: return PC_KIND_CLEAR_FLOOD;
                case PC_CLEAR_ALGAE: return PC_KIND_CLEAR_ALGAE;
                case PC_EXPLODE: return PC_KIND_EXPLODE;
                case PC_SWARM: return PC_KIND_SWARM;
                default: return PC_KIND_UNSUPPORTED;
            }
        default: return PC_KIND_UNSUPPORTED;
    }
}

/* Tiles explored, summed over the seats (a new exploration raises it). */
PC_FN int pc_core_explored_count(const PcState *s) {
    int c = 0;
    for (int k = 0; k < s->np; k++)
        for (int t = 0; t < s->n; t++) c += s->tiles.explored[k][t];
    return c;
}

/* Scratch buffers reused across steps. `structs` is this port's addition: the structures module's buffers (routes,
 * level queue) that the Rust allocates inside its functions. */
typedef struct {
    uint8_t row[PC_MAXN * 9];
    PcMoveScratch mv;
    PcBuildCtx build;
    PcSummonCtx summon;
    PcPathScratch path;
    PcStructScratch structs;
} PcScratch;

/* step, given optionally the packed legal masks (masks::Packed layout) computed for exactly this state: Move,
 * Build and Summon are then validated by one mask lookup. The status is the same either way: where a mask entry
 * and step's own predicate can differ (a pending city reward, an unsupported unit kind, a
 * core_supported=False state) the action is unsupported regardless of validity. `legal` is NULL when absent. */
PC_FN int32_t pc_core_step_with(PcState *s, PcAction a, int32_t expected_version, const PcRules *r, const PcGeom *g,
                                int32_t max_actions, PcScratch *sc, const uint8_t *legal) {
    int32_t n = s->n;
    int o[18];
    int src, dst, param;
    int source_valid, target_valid, param_valid;
    int32_t cp;
    PcKind kind;
    int own, idle, pending, reward, reward_implemented, peace, supported, valid;
    int32_t status, victim, count;
    int explored_before;
    if (!s->core_supported) legal = NULL;
    pc_masks_family_offsets(s->n, o);
    src = (int)pc_clamp(a.source, 0, n - 1);
    dst = (int)pc_clamp(a.target, 0, n - 1);
    param = (int)pc_clamp(a.parameter, 0, 127);
    source_valid = a.source >= 0 && a.source < n;
    target_valid = a.target >= 0 && a.target < n;
    param_valid = a.parameter >= 0 && a.parameter < 128;
    cp = s->current_player;
    kind = pc_core_classify(a);
    own = source_valid && s->units.owner[src] == cp;
    idle = own && !s->units.moved[src] && !s->units.attacked[src];
    pending = pc_pending_reward(s);
    reward = kind == PC_KIND_REWARD;
    reward_implemented = a.parameter >= 1 && a.parameter <= 8;
    /* pending command triggers: a peace request (answered first) or a city reward block every other command */
    peace = pc_diplomacy_peace_pending(s);
    supported = s->core_supported && kind != PC_KIND_UNSUPPORTED
        && (peace ? (kind == PC_KIND_DIPLOMATIC || kind == PC_KIND_RESIGN) : (!pending || reward || kind == PC_KIND_RESIGN))
        && (!reward || reward_implemented);
    if (kind == PC_KIND_SUMMON) supported = supported && r->unit_supported[param];
    switch (kind) {
        case PC_KIND_END:
        case PC_KIND_RESIGN:
            valid = 1;
            break;
        case PC_KIND_RESEARCH: {
            uint8_t m[128];
            pc_kernels_research_mask(s, r, m);
            valid = param_valid && m[param];
            break;
        }
        case PC_KIND_RECOVER:
            valid = idle && s->units.leader[src] == 0 && (s->units.health[src] < pc_kernels_max_health(s, r, src) || s->units.poison[src]);
            break;
        case PC_KIND_MOVE:
            if (!(source_valid && target_valid)) {
                valid = 0;
            } else if (legal) {
                valid = pc_mask_get(legal, o[0] + src * s->n + dst);
            } else {
                pc_kernels_movement_targets(s, src, r, g, &sc->mv, sc->row);
                valid = sc->row[dst];
            }
            break;
        case PC_KIND_ATTACK:
            pc_military_attack_mask(s, src, r, g, sc->row);
            valid = source_valid && target_valid && sc->row[dst];
            break;
        case PC_KIND_SUMMON:
            if (!(source_valid && param_valid)) {
                valid = 0;
            } else if (legal) {
                valid = pc_mask_get(legal, o[2] + src * 128 + param);
            } else {
                pc_military_summon_prepare(&sc->summon, s, r);
                valid = pc_u128_test(pc_military_summon_row(s, &sc->summon, src), param);
            }
            break;
        case PC_KIND_HARVEST:
            pc_economy_harvest_mask(s, sc->row);
            valid = target_valid && sc->row[dst];
            break;
        case PC_KIND_FOREST:
            pc_economy_forest_masks(s, sc->row);
            valid = target_valid && sc->row[dst * 3 + (int)pc_clamp(a.parameter - 1, 0, 2)];
            break;
        case PC_KIND_PROMOTE:
            valid = own && pc_military_promotable(s, r, src);
            break;
        case PC_KIND_DISBAND:
            valid = idle && s->units.leader[src] == 0 && s->players.tech[pc_p(s)][2];
            break;
        case PC_KIND_BUILD:
            if (!(target_valid && param_valid)) {
                valid = 0;
            } else if (legal) {
                valid = pc_mask_get(legal, o[3] + dst * 128 + param);
            } else {
                valid = pc_structures_build_legal(s, r, g, &sc->build, dst, param);
            }
            break;
        case PC_KIND_DESTROY:
            pc_structures_destroy_mask(s, sc->row);
            valid = target_valid && sc->row[dst];
            break;
        case PC_KIND_HEAL: {
            int any = 0;
            int heals = r->heal_others[s->units.kind[src]];   /* Heal by ability (Mind Bender, Siren) */
            if (idle && heals) {
                for (int i = 0; i < s->n && !any; i++)
                    any = pc_d(g, src, i) == 1 && s->units.owner[i] == cp
                        && (s->units.health[i] < pc_kernels_max_health(s, r, i) || s->units.poison[i]);
            }
            valid = idle && heals && any;
            break;
        }
        case PC_KIND_CAP:
            pc_capture_capture_mask(s, sc->row);
            valid = source_valid && sc->row[src];
            break;
        case PC_KIND_DIPLOMATIC:
            valid = pc_diplomacy_legal(s, a.source, a.target, a.parameter);
            break;
        case PC_KIND_DISEMBARK:
            valid = own && pc_military_can_disembark(s, r, src);
            break;
        case PC_KIND_FREEZE_AREA:
            valid = own && pc_polaris_can_freeze_area(s, src, r, g);
            break;
        case PC_KIND_BREAK_ICE:
            valid = own && pc_polaris_can_break_ice(s, src, r, g);
            break;
        case PC_KIND_CANAL:
            valid = own && pc_aquarion_can_canal(s, src);
            break;
        case PC_KIND_CLEAR_FLOOD:
            valid = own && pc_aquarion_can_clear_flood(s, src, r);
            break;
        case PC_KIND_CLEAR_ALGAE:
            valid = own && pc_cymanti_can_clear_algae(s, src, r);
            break;
        case PC_KIND_EXPLODE:
            valid = own && pc_cymanti_can_explode(s, src, r);
            break;
        case PC_KIND_SWARM:
            valid = own && pc_cymanti_can_swarm(s, src, r, g);
            break;
        case PC_KIND_REWARD:
            pc_economy_reward_mask(s, sc->row);
            valid = target_valid && a.parameter >= 0 && a.parameter < 9 && sc->row[dst * 9 + (int)pc_clamp(a.parameter, 0, 8)];
            break;
        case PC_KIND_UNSUPPORTED:
        default:
            valid = 0;
            break;
    }
    status = !supported ? PC_UNSUPPORTED_RULE : (valid ? PC_OK : PC_INVALID_ACTION);
    if (expected_version != s->state_version) status = PC_STALE_VERSION;
    if (s->terminated || s->truncated) status = PC_EPISODE_DONE;
    if (status != PC_OK) return status;
    if (!s->monuments_settled) pc_structures_remember_monuments(s, r, g);
    /* modify diplomacy relations from attack's victim (attack command): the target unit's owner, or for an
     * infiltration the target tile's owner; read before the action changes the tile */
    victim = 0;
    if (kind == PC_KIND_ATTACK) {
        /* (a Drench attack on an own or peace-protected unit changes no relation: no attackable defender) */
        victim = r->skills[s->units.kind[src]][PC_SK_INFILTRATE] ? s->tiles.owner[dst]
            : pc_diplomacy_protected(s, cp, s->units.owner[dst]) ? 0 : s->units.owner[dst];
    }
    /* explore action (version >= 98) only flags has flagged need for update routes; command base queues
     * run deferred action under every command's actions, so once they have all run it queues update routes action and
     * update city connections action for the command's player (the action stack is last in,
     * first out, so the routes are updated first) */
    explored_before = pc_core_explored_count(s);
    switch (kind) {
        case PC_KIND_RESEARCH: pc_core_research(s, a, r); break;
        case PC_KIND_RECOVER: pc_core_recover(s, src, r); break;
        case PC_KIND_MOVE: pc_core_move_adjacent(s, src, dst, r, g, &sc->path, &sc->structs); break;
        case PC_KIND_END: pc_core_end_turn(s, r, g, &sc->structs); break;
        case PC_KIND_ATTACK: {
            int uk = (int)s->units.kind[src];
            if (r->skills[uk][PC_SK_INFILTRATE]) {
                pc_infiltration_execute(s, src, dst, r, g);
            } else if (r->convert[uk] && r->unit_attack[uk] <= 0.0f) {   /* the Mind Bender; the Shaman attacks */
                pc_military_convert(s, src, dst, r, g);
            } else {
                pc_military_attack(s, src, dst, r, g, &sc->structs);
            }
            break;
        }
        case PC_KIND_SUMMON:
            if (s->units.kind[src] == 43) {
                pc_military_upgrade(s, src, param, r, g);
            } else {
                pc_military_summon(s, src, param, r, g, 1);
            }
            break;
        case PC_KIND_HARVEST: pc_economy_harvest(s, dst); break;
        case PC_KIND_FOREST: pc_economy_forest(s, dst, a.parameter); break;
        case PC_KIND_PROMOTE: pc_military_promote(s, src, r); break;
        case PC_KIND_DISBAND: pc_military_disband(s, src, r); break;
        case PC_KIND_REWARD: pc_economy_reward(s, dst, a.parameter, r, g, &sc->structs); break;
        case PC_KIND_BUILD: pc_structures_build(s, dst, param, r, g, &sc->structs); break;
        case PC_KIND_DESTROY:
            if (s->players.tech[pc_p(s)][3] || s->players.tech[pc_p(s)][26]) {
                pc_structures_destroy(s, dst, r, g, &sc->structs);
            } else if (s->tiles.owner[dst] == cp) {
                s->tiles.decomposing[dst] = 1;   /* Cymanti: decompose action */
            }
            break;
        case PC_KIND_CAP: {
            int32_t st = s->tiles.structure[src];
            if (st == 1) {
                pc_capture_city(s, src, r, g, &sc->structs);
            } else if (st == 2) {
                pc_capture_ruin(s, src, r, g, &sc->structs);
            } else {
                pc_capture_starfish(s, src);
            }
            break;
        }
        case PC_KIND_HEAL: pc_military_heal_others(s, src, r, g); break;
        case PC_KIND_DIPLOMATIC: pc_diplomacy_execute(s, a.source, a.target, a.parameter, r, g, &sc->structs); break;
        case PC_KIND_DISEMBARK: pc_military_disembark(s, src, r, g); break;
        case PC_KIND_FREEZE_AREA: pc_polaris_freeze_area_command(s, src, r, g, &sc->structs); break;
        case PC_KIND_BREAK_ICE: pc_polaris_break_ice_command(s, src, r, g, &sc->structs); break;
        case PC_KIND_CANAL: pc_aquarion_canal_command(s, src, r, g, &sc->structs); break;
        case PC_KIND_CLEAR_FLOOD: pc_aquarion_clear_flood_command(s, src, r, g, &sc->structs); break;
        case PC_KIND_CLEAR_ALGAE: pc_cymanti_clear_algae_command(s, src, r, g, &sc->structs); break;
        case PC_KIND_EXPLODE: pc_cymanti_explode_command(s, src, r, g, &sc->structs); break;
        case PC_KIND_SWARM: pc_cymanti_swarm_command(s, src, g); break;
        case PC_KIND_RESIGN: pc_core_resign(s); break;
        case PC_KIND_UNSUPPORTED:
        default:
            break;   /* unreachable: status was UNSUPPORTED_RULE */
    }
    if (a.kind == 2) pc_diplomacy_hostile(s, victim);
    pc_core_meet(s);
    pc_diplomacy_capital_vision(s);
    uint8_t algae = s->algae_dirty;
    s->algae_dirty = 0;
    if (pc_core_explored_count(s) != explored_before || algae) {
        pc_structures_update_routes(s, r, &sc->structs);
        pc_structures_connections(s, cp, g, 1);
    }
    pc_structures_remember_monuments(s, r, g);
    s->monuments_settled = 1;
    count = s->actions + 1;
    s->state_version += 1;
    s->actions = count;
    s->truncated = (uint8_t)((s->truncated || count >= max_actions) && !s->terminated);
    return PC_OK;
}

/* core.step: apply one atomic action; rejection leaves the state untouched. Returns the status code. */
PC_FN int32_t pc_core_step(PcState *s, PcAction a, int32_t expected_version, const PcRules *r, const PcGeom *g,
                           int32_t max_actions, PcScratch *sc) {
    return pc_core_step_with(s, a, expected_version, r, g, max_actions, sc, NULL);
}

/* engine-rs tribes.rs: what a tribe starts with (game logic data 28 tribe data): the twelve regular
 * tribes, Aquarion (S2), Polaris (S1) and Elyrion (S3) (work in progress: refused unless opted in, pc_caps_ok). Returns 0 for a tribe the
 * engine does not have (the other special tribes): the caller refuses it. */
typedef struct { int32_t tech, unit, stars, capital_level; float health; } PcTribeStart;
PC_FN int pc_tribe_start(int32_t tribe, PcTribeStart *o) {
    PcTribeStart v;
    switch (tribe) {
    case 2: v = (PcTribeStart){22, 10, 5, 1, 10.0f}; break;   /* Ai-Mo: Philosophy, MindBender */
    case 4: v = (PcTribeStart){15, 2, 5, 1, 10.0f}; break;    /* Bardur: Hunting, Warrior */
    case 6: v = (PcTribeStart){18, 9, 7, 1, 10.0f}; break;    /* Hoodrick: Archery, Archer */
    case 7: v = (PcTribeStart){6, 2, 5, 1, 10.0f}; break;     /* Imperius: Organization */
    case 8: v = (PcTribeStart){10, 2, 5, 1, 10.0f}; break;    /* Kickoo: Fishing */
    case 9: v = (PcTribeStart){-1, 2, 2, 3, 10.0f}; break;    /* Luxidoor: Basic only, level-3 capital */
    case 10: v = (PcTribeStart){1, 3, 6, 1, 10.0f}; break;    /* Oumaji: Riding, Rider */
    case 11: v = (PcTribeStart){7, 5, 7, 1, 15.0f}; break;    /* Quetzali: Strategy, Defender */
    case 12: v = (PcTribeStart){24, 11, 5, 1, 15.0f}; break;  /* Vengir: Smithery, Swordsman */
    case 13: v = (PcTribeStart){20, 2, 7, 1, 10.0f}; break;   /* Xin-xi: Climbing */
    case 14: v = (PcTribeStart){4, 2, 7, 1, 10.0f}; break;    /* Yadakk: Roads */
    case 15: v = (PcTribeStart){8, 2, 5, 1, 10.0f}; break;    /* Zebasi: Farming */
    case 3: v = (PcTribeStart){45, 46, 5, 1, 10.0f}; break;   /* Aquarion: aqua base, MermaidWarrior */
    case 16: v = (PcTribeStart){30, 22, 5, 1, 10.0f}; break;  /* Polaris: Frostwork, Mooni */
    case 5: v = (PcTribeStart){28, 2, 5, 1, 10.0f}; break;    /* Elyrion: forest magic, Warrior (S3, work in progress) */
    case 17: v = (PcTribeStart){8, 36, 5, 1, 10.0f}; break;   /* Cymanti: Farming, Shaman (S4, work in progress) */
    default: return 0;
    }
    if (o) *o = v;
    return 1;
}
PC_FN int32_t pc_tribe_tier(int32_t tech) {
    switch (tech) {
    case 1: case 6: case 10: case 15: case 20: case 28: case 30: return 1;
    case 4: case 7: case 8: case 18: return 2;
    case 22: case 24: return 3;
    default: return 0;
    }
}
PC_FN int32_t pc_tribe_unit_cost(int32_t unit) {
    switch (unit) {
    case 2: case 46: return 2;
    case 3: case 5: case 9: return 3;
    case 10: case 11: case 22: case 36: return 5;
    default: return 0;
    }
}
/* tribes::unpack: a packed capital is tile | tribe << 16, tribe bits 0 = Bardur */
PC_FN int pc_cap_tile(int32_t cap) { return (int)(cap & 0xffff); }
PC_FN int32_t pc_cap_tribe(int32_t cap) { int32_t t = cap >> 16; return t == 0 ? 4 : t; }

/* Host-side check of packed capitals (lib.rs check_capitals): every tile on the board, every tribe one the engine has
 * and offers (tribes::offered: a work-in-progress tribe only when its bit of `wip_ok` is set, the caller's
 * POLYTOPIA_GEN_WIP_TRIBES opt-in, pc_api.h pc_wip_opted_in; Aquarion 3, Elyrion 5, Polaris 16 and Cymanti 17). */
PC_FN int pc_caps_ok(const int32_t *caps, size_t count, int n, uint32_t wip_ok) {
    for (size_t i = 0; i < count; i++) {
        if (caps[i] < 0 || pc_cap_tile(caps[i]) >= n) return 0;
        int32_t tribe = pc_cap_tribe(caps[i]);
        if (!pc_tribe_start(tribe, NULL) || (pc_tribes_wip(tribe) && !((wip_ok >> (tribe & 31)) & 1u))) return 0;
    }
    return 1;
}

/* mapgen::initial_state (maps/state.py::initial_state, bitwise mapgen.reset after the layout step): the fresh game
 * from a layout (terrain/structure/resource [n], the np packed capitals) for `np` seats. */
PC_FN void pc_core_initial_state(PcState *s, uint32_t seed, const int32_t *terrain, const int32_t *structure,
                                 const int32_t *resource, const int32_t *packed_caps, int np, const PcGeom *g, int32_t max_turns) {
    int size = g->size, n = g->n;
    int32_t caps[PC_MAXP];
    PcTribeStart st[PC_MAXP];
    pc_state_empty(s, size, np);
    for (int k = 0; k < np; k++) {
        caps[k] = pc_cap_tile(packed_caps[k]);
        s->players.tribe[k] = pc_cap_tribe(packed_caps[k]);
        if (!pc_tribe_start(s->players.tribe[k], &st[k])) pc_tribe_start(4, &st[k]);   /* callers validate first */
    }
    s->seed = seed;
    s->initial_seed = seed;
    s->reference_max_turns = max_turns;
    for (int t = 0; t < n; t++) {
        s->tiles.terrain[t] = terrain[t] & 0xff;            /* bit 8 = mapgen Flooded (mapgen::FLOODED_BIT) */
        s->tiles.flooded[t] = (terrain[t] & (1 << 8)) != 0;
        s->tiles.continent[t] = (terrain[t] >> 9) & 31;    /* bits 9-13 = mapgen continent (mapgen::CONTINENT_SHIFT) */
        s->tiles.structure[t] = structure[t];
        s->tiles.resource[t] = resource[t];
    }
    for (int t = 0; t < n; t++) {
        /* owner = max over seats of (area ? seat+1 : 0) */
        int32_t owner = 0;
        for (int k = 0; k < np; k++) {
            if (pc_d(g, (int)caps[k], t) <= 1) owner = pc_max(owner, (int32_t)k + 1);
        }
        s->tiles.owner[t] = owner;
        s->tiles.city[t] = owner > 0 ? caps[owner - 1] : -1;
    }
    for (int k = 0; k < np; k++) {
        int c = (int)caps[k];
        int32_t owner = (int32_t)k + 1;
        s->tiles.capital[c] = owner;
        s->cities.owner[c] = owner;
        s->cities.level[c] = st[k].capital_level;
        s->cities.rewards[c * 9] = st[k].capital_level - 1;     /* levels without a city reward (tribes.rs) */
        s->cities.production[c] = st[k].capital_level;          /* calculate work base: level + production - 1 */
        s->cities.border[c] = 1;
        for (int t = 0; t < n; t++) pc_set_territory(s, c, t, pc_d(g, c, t) <= 1);
        s->cities.order[c] = 1;
        s->units.owner[c] = owner;
        s->units.kind[c] = st[k].unit;
        s->units.health[c] = st[k].health;
        s->units.home[c] = (int32_t)c + 1;
        s->units.prev_x[c] = (int32_t)(c % size);
        s->units.prev_y[c] = (int32_t)(c / size);
        s->units.direction[c] = pc_military_spawn_direction(size, c);
        /* Polaris: the start ring and the capital have the Polaris climate (its continent never
         * grows); prepare alien climates already made the water there Ice (the layout's terrain) */
        if (s->players.tribe[k] == PC_POLARIS) {
            for (int t = 0; t < n; t++) {
                if (pc_d(g, c, t) <= 1 && !(s->tiles.terrain[t] == 1 || s->tiles.terrain[t] == 2)) s->tiles.polar[t] = 1;
            }
        }
    }
    for (int k = 0; k < np; k++) {
        s->players.stars[k] = st[k].stars;
        s->players.tech[k][0] = 1;
        if (st[k].tech >= 0) s->players.tech[k][st[k].tech] = 1;
        /* tribes::start_score */
        s->players.score[k] = 390 - 100 * pc_tribe_tier(15) - 5 * pc_tribe_unit_cost(2)
            + (st[k].tech >= 0 ? 100 * pc_tribe_tier(st[k].tech) : 0) + 5 * pc_tribe_unit_cost(st[k].unit)
            + 50 * (st[k].capital_level - 1);
    }
    for (int k = 0; k < np; k++) pc_kernels_update_exploration(s, (int32_t)k + 1, g);
}

#ifdef __cplusplus
}
#endif
#endif /* PC_CORE_H */
