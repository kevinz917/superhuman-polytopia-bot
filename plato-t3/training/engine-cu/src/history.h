/* history.h - line-by-line port of engine-rs/src/history.rs: observation history (observation version 3).
 *
 * Everything here is a function of the seat's own observations, never of hidden state, so the policy gets no
 * information a human player could not have: a snapshot of what the seat sees is taken at the end of each of its
 * turns (end_of_turn), and at the start of its next turn (start_of_turn) the differences to that snapshot
 * are the "since my last turn" planes. Views then carry (kernels::observe):
 *   hist_enemy_unit[k]   a visible enemy unit stood here at the end of my turn k + 1 of my turns ago (k = 0, 1, 2)
 *   hist_enemy_built     an enemy structure, road or level appeared or changed here since my last turn (tile explored then)
 *   hist_enemy_trained   a visible enemy unit stands on an enemy city tile where none stood at the end of my last turn
 *   enemy_tech           techs the enemy must have: the tech of every enemy unit kind, structure and road seen, a unit
 *                        seen on a mountain (Climbing) or on water (Fishing), with every prerequisite (Rules::tech_closure)
 * In a game with more than two seats "enemy" is any other seat (one union over them). */
#ifndef PC_HISTORY_H
#define PC_HISTORY_H

#include "pc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tile code compared across turns: what the view shows of a tile's improvement. */
PC_FN int32_t pc_history_code(const PcState *s, int t) {
    return s->tiles.structure[t] * 64 + pc_clamp(s->tiles.structure_level[t], 0, 31) * 2 + (int32_t)s->tiles.road[t];
}

/* A visible, non-hidden enemy unit stands on t for `player`. */
PC_FN int pc_history_enemy_unit_visible(const PcState *s, int32_t player, int t, int vis) {
    return vis && pc_foreign(s->units.owner[t], player) && !pc_hidden_from(s, t, player);
}

/* Fold what `player` can see right now into its inferred enemy techs. */
PC_FN void pc_history_fold_tech(PcState *s, int32_t player, const PcRules *r) {
    int p = (int)(player - 1);
    pc_u128 bits = s->history[p].enemy_tech;
    pc_u128 raw = pc_u128_zero();
    for (int t = 0; t < s->n; t++) {
        int vis = s->tiles.explored[p][t] || !s->fog;
        if (!vis) continue;
        if (pc_history_enemy_unit_visible(s, player, t, 1)) {
            int kind = (int)pc_clamp(s->units.kind[t], 0, 127);
            int32_t tech = r->unit_tech[kind];
            if (tech < 127) pc_u128_set(&raw, (int)tech);
            if (kind == 37) pc_u128_set(&raw, 38);   /* a Dagger comes from a Cloak: Diplomacy */
            switch (s->tiles.terrain[t]) {
                case 4: pc_u128_set(&raw, 20); break;   /* a unit on a mountain: Climbing */
                case 1: pc_u128_set(&raw, 10); break;   /* a unit on shallow water: Fishing */
                default: break;
            }
        }
        if (pc_foreign(s->tiles.owner[t], player)) {
            int st = (int)pc_clamp(s->tiles.structure[t], 0, 127);
            pc_u128 by = r->structures.unlock_by[st];
            /* by != 0 && by & (by - 1) == 0: exactly one tech unlocks this structure */
            if (pc_u128_any(by) && pc_u128_count(by) == 1) raw = pc_u128_or(raw, by);
            if (s->tiles.road[t]) {
                pc_u128 by3 = r->structures.unlock_by[3];
                if (pc_u128_any(by3) && pc_u128_count(by3) == 1) raw = pc_u128_or(raw, by3);
            }
        }
    }
    while (pc_u128_any(raw)) {
        int k = pc_u128_lowest(raw);
        pc_u128_clear(&raw, k);
        bits = pc_u128_or(bits, r->tech_closure[k]);
    }
    s->history[p].enemy_tech = pc_u128_and(bits, pc_u128_or(r->tech_exists_bits, r->polaris.tech_exists_bits));
}

/* `player` ends its turn: remember what it sees now (shift the unit ring; record the tile codes). */
PC_FN void pc_history_end_of_turn(PcState *s, int32_t player, const PcRules *r) {
    pc_history_fold_tech(s, player, r);
    int p = (int)(player - 1);
    int n = s->n;
    /* seen_unit.rotate_right(1): [0] becomes the oldest slot, overwritten below */
    {
        PcHistory *h = &s->history[p];
        uint8_t last[PC_MAXN];
        memcpy(last, h->seen_unit[PC_HIST_TURNS - 1], PC_MAXN);
        for (int k = PC_HIST_TURNS - 1; k > 0; k--) memcpy(h->seen_unit[k], h->seen_unit[k - 1], PC_MAXN);
        memcpy(h->seen_unit[0], last, PC_MAXN);
    }
    for (int t = 0; t < n; t++) {
        int vis = s->tiles.explored[p][t] || !s->fog;
        int seen = pc_history_enemy_unit_visible(s, player, t, vis);
        int32_t c = vis ? pc_history_code(s, t) : -1;
        PcHistory *h = &s->history[p];
        h->seen_unit[0][t] = (uint8_t)seen;
        h->prev_code[t] = c;
    }
}

/* `player` starts its turn: what changed since the end of its last turn, on tiles it could see then. */
PC_FN void pc_history_start_of_turn(PcState *s, int32_t player, const PcRules *r) {
    int p = (int)(player - 1);
    int n = s->n;
    for (int t = 0; t < n; t++) {
        int vis = s->tiles.explored[p][t] || !s->fog;
        int32_t prev = s->history[p].prev_code[t];
        int enemy_tile = pc_foreign(s->tiles.owner[t], player);
        int built = vis && prev >= 0 && enemy_tile && pc_history_code(s, t) != prev;
        int unit_now = pc_history_enemy_unit_visible(s, player, t, vis);
        int trained = unit_now && !s->history[p].seen_unit[0][t] && pc_foreign(s->cities.owner[t], player);
        PcHistory *h = &s->history[p];
        h->built[t] = (uint8_t)built;
        h->trained[t] = (uint8_t)trained;
    }
    pc_history_fold_tech(s, player, r);
}

#ifdef __cplusplus
}
#endif
#endif /* PC_HISTORY_H */
