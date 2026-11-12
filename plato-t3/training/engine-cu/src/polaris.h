/* polaris.h - line-by-line port of engine-rs/src/polaris.rs (Polaris, increment S1): freezing. Rules as in the
 * project's full-game engine port.
 *
 * Representation: a tile is frozen (is frozen) when its terrain is Ice or it is polar land (`tiles.polar`:
 * climate 16). Every `pub fn f` is `pc_polaris_f`, same argument order; the functions that run update routes take the
 * PcStructScratch pc_structures_update_routes needs (the Rust allocates inside it). Geom::for_ball is pc_ball's
 * row-major box. */
#ifndef PC_POLARIS_H
#define PC_POLARIS_H

#include "pc.h"
#include "tribes.h"       /* PC_POLARIS */
#include "kernels.h"      /* PC_ICE */
#include "structures.h"   /* pc_structures_update_routes, pc_structures_connections, PcStructScratch (before diplomacy.h) */
#include "diplomacy.h"    /* pc_diplomacy_protected, pc_diplomacy_hostile */

#ifdef __cplusplus
extern "C" {
#endif

PC_FN int pc_polaris_water(int32_t terrain) { return terrain == 1 || terrain == 2; }

/* is frozen */
PC_FN int pc_polaris_frozen_tile(const PcState *s, int t) {
    return s->tiles.terrain[t] == PC_ICE || (s->tiles.polar[t] && !pc_polaris_water(s->tiles.terrain[t]));
}

/* is freezable for `player`: a unit with no active peace treaty with the player (own units, peace and
 * cool-down count as one) and not already Frozen. */
PC_FN int pc_polaris_unit_freezable(const PcState *s, int t, int32_t player) {
    int32_t o = s->units.owner[t];
    return o > 0 && !pc_diplomacy_protected(s, player, o) && !s->units.frozen[t];
}

/* is freezable(state, player, only owned): explored by the player (and owned, with `only_owned`), and water,
 * or land not of the player's climate, or a visible freezable unit on it. */
PC_FN int pc_polaris_tile_freezable(const PcState *s, int t, int32_t player, int only_owned) {
    int p = pc_pclip(s, player);
    if ((only_owned && s->tiles.owner[t] != player) || !s->tiles.explored[p][t]) return 0;
    /* a Flooded tile stays freezable (freeze tile removes Swamped only; polaris.rs) */
    return pc_polaris_water(s->tiles.terrain[t]) || s->tiles.flooded[t] || !s->tiles.polar[t]
        || (!pc_hidden_from(s, t, player) && pc_polaris_unit_freezable(s, t, player));
}

/* freeze tile (v >= 120) as freeze tile action runs it: re-checks is freezable (only owned false); water
 * becomes Ice; the tile takes the player's climate. Returns whether the tile changed. */
PC_FN int pc_polaris_freeze_tile(PcState *s, int t, int32_t player) {
    if (!pc_polaris_tile_freezable(s, t, player, 0)) return 0;
    if (pc_polaris_water(s->tiles.terrain[t])) s->tiles.terrain[t] = PC_ICE;
    s->tiles.polar[t] = 1;
    return 1;
}

/* freeze unit action with damage 0 and no origin (FreezeArea, AutoFreeze): modify diplomacy relations from attack, the
 * unit is moved, attacked and Frozen, and the freezing player's Pacifist task is reset. */
PC_FN void pc_polaris_freeze_unit(PcState *s, int t, int32_t player) {
    int32_t victim = s->units.owner[t];
    if (victim <= 0) return;
    pc_diplomacy_hostile(s, victim);
    s->units.moved[t] = 1;
    s->units.attacked[t] = 1;
    s->units.move_lock[t] = 0;
    s->units.frozen[t] = 1;
    s->players.attacked[pc_pclip(s, player)] = 1;
}

/* get ice bank level: min(count ice tiles / 20, max level 30), count ice tiles over the whole map. */
PC_FN int32_t pc_polaris_ice_bank_level(const PcState *s) {
    int32_t c = 0;
    for (int t = 0; t < s->n; t++) c += pc_polaris_frozen_tile(s, t);
    return pc_min(c / 20, 30);
}

/* check ice bank levels (and the owner's start turn update improvement level): every Ice Bank's stored level becomes the
 * calculated one. */
PC_FN void pc_polaris_sync_ice_banks(PcState *s) {
    int any = 0;
    for (int t = 0; t < s->n && !any; t++) any = s->tiles.structure[t] == 34;
    if (!any) return;
    int32_t level = pc_polaris_ice_bank_level(s);
    for (int t = 0; t < s->n; t++) {
        if (s->tiles.structure[t] == 34) s->tiles.structure_level[t] = level;
    }
}

/* start turn action's Ice Bank income for `owner`: calculate work = work 2 x the calculated level, per owned bank. */
PC_FN int32_t pc_polaris_ice_bank_income(const PcState *s, int32_t owner) {
    int32_t banks = 0;
    for (int t = 0; t < s->n; t++) banks += s->tiles.structure[t] == 34 && s->tiles.owner[t] == owner;
    return banks == 0 ? 0 : banks * 2 * pc_polaris_ice_bank_level(s);
}

/* freeze area action: over the radius area, centre included, each freezable raw unit is
 * frozen, then each freezable tile; then update routes and update city connections of the player run last. */
PC_FN void pc_polaris_freeze_area(PcState *s, int centre, int32_t radius, int freeze_units, int only_owned, const PcRules *r,
                                  const PcGeom *g, PcStructScratch *sc) {
    int32_t player = s->current_player;
    int n = s->n;
    uint8_t pushed[PC_MAXN];
    int any = 0;
    /* pushed before the freeze tile per tile (LIFO): the push-time test decides which tiles get a freeze tile */
    for (int t = 0; t < n; t++) {
        uint8_t in = pc_d(g, centre, t) <= radius;
        any |= in;
        pushed[t] = (uint8_t)(in && pc_polaris_tile_freezable(s, t, player, only_owned));
    }
    if (!any) return;
    for (int t = 0; t < n; t++) {
        if (pc_d(g, centre, t) > radius) continue;
        if (freeze_units && pc_polaris_unit_freezable(s, t, player)) pc_polaris_freeze_unit(s, t, player);
        if (pushed[t]) pc_polaris_freeze_tile(s, t, player);
    }
    pc_polaris_sync_ice_banks(s);
    pc_structures_update_routes(s, r, sc);
    pc_structures_connections(s, player, g, 1);
}

/* move action (v > 101, not a push): an AutoFreeze unit (Mooni, Gaami) freezes radius 1 around where
 * it arrives, units included. */
PC_FN void pc_polaris_auto_freeze(PcState *s, int at, int kind, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    if (r->autofreeze[kind]) pc_polaris_freeze_area(s, at, 1, 1, 0, r, g, sc);
}

/* restore climate (v > 111) on a tile of the Polaris climate, for a player whose climate does not spread
 * (any non-Polaris player): polar land other than Ice goes back to the player's climate. */
PC_FN void pc_polaris_restore_climate(PcState *s, int t, int32_t player) {
    if (s->players.tribe[pc_pclip(s, player)] != PC_POLARIS && s->tiles.polar[t]
        && !pc_polaris_water(s->tiles.terrain[t]) && s->tiles.terrain[t] != PC_ICE) {
        s->tiles.polar[t] = 0;
    }
}

/* The city area's climate after an expand city / capture city of `player`: Polaris freezes every tile ruled by the
 * city (freeze tile), any other player restores the climate there. */
PC_FN void pc_polaris_city_area_climate(PcState *s, int city, int32_t player) {
    int polaris = s->players.tribe[pc_pclip(s, player)] == PC_POLARIS;
    for (int t = 0; t < s->n; t++) {
        if (s->tiles.city[t] == (int32_t)city) {
            if (polaris) {
                pc_polaris_freeze_tile(s, t, player);
            } else {
                pc_polaris_restore_climate(s, t, player);
            }
        }
    }
    if (polaris) pc_polaris_sync_ice_banks(s);
}

/* can freeze area(unit, 1): not moved, not attacked, the FreezeArea ability, and a freezable tile
 * (is freezable, only owned false) in the radius-1 area around it, centre excluded. */
PC_FN int pc_polaris_can_freeze_area(const PcState *s, int i, const PcRules *r, const PcGeom *g) {
    const PcUnits *u = &s->units;
    int kind = (int)u->kind[i];
    int32_t owner = u->owner[i];
    if (owner != s->current_player || u->moved[i] || u->attacked[i] || !r->freeze_area[kind]) return 0;
    int x0, x1, y0, y1;
    pc_ball(g, i, 1, &x0, &x1, &y0, &y1);
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            int t = y * g->size + x;
            if (t != i && pc_polaris_tile_freezable(s, t, owner, 0)) return 1;
        }
    }
    return 0;
}

/* freeze area command: the unit is moved and attacked, then freeze area action(radius 1, units, any tile). */
PC_FN void pc_polaris_freeze_area_command(PcState *s, int i, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    s->units.moved[i] = 1;
    s->units.attacked[i] = 1;
    s->units.move_lock[i] = 0;
    pc_polaris_freeze_area(s, i, 1, 1, 0, r, g, sc);
}

/* has breakable ice(tile, player) (v >= 116): Ice, and a Bridge on it, or no unit, or a Swim / Fly unit, or a
 * unit the player cannot see. */
PC_FN int pc_polaris_has_breakable_ice(const PcState *s, int t, int32_t player, const PcRules *r) {
    if (s->tiles.terrain[t] != PC_ICE) return 0;
    int32_t o = s->units.owner[t];
    int k = (int)s->units.kind[t];
    return s->tiles.structure[t] == 48 || s->tiles.algae[t] || o == 0 || r->skills[k][PC_SK_FLOAT] || r->fly[k] || pc_hidden_from(s, t, player);
}

/* can break ice (v >= 99): any tribe's unit, not attacked, not moved unless Dash, no AutoFreeze, and
 * breakable ice in the radius-1 area around it, centre excluded. */
PC_FN int pc_polaris_can_break_ice(const PcState *s, int i, const PcRules *r, const PcGeom *g) {
    const PcUnits *u = &s->units;
    int kind = (int)u->kind[i];
    int32_t owner = u->owner[i];
    /* can break ice: no leader (Cymanti) */
    if (owner != s->current_player || u->attacked[i] || (u->moved[i] && !r->dash[kind]) || r->autofreeze[kind] || u->leader[i] != 0) return 0;
    int x0, x1, y0, y1;
    pc_ball(g, i, 1, &x0, &x1, &y0, &y1);
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            int t = y * g->size + x;
            if (t != i && pc_polaris_has_breakable_ice(s, t, owner, r)) return 1;
        }
    }
    return 0;
}

/* break ice command: the unit is moved and attacked; an invisible breaker is revealed (after the breaking);
 * break ice area action over radius 1, centre included, nearest first: each break ice action on breakable ice reveals an
 * unseen unit there and stops, or turns the Ice into Water (an orthogonal neighbour is land other than Ice) or
 * Ocean; then check ice bank levels; update routes and update city connections of the player run last. */
PC_FN void pc_polaris_break_ice_command(PcState *s, int i, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    static const int DX[4] = {0, -1, 1, 0};
    static const int DY[4] = {-1, 0, 0, 1};
    int32_t player = s->current_player;
    s->units.moved[i] = 1;
    s->units.attacked[i] = 1;
    s->units.move_lock[i] = 0;
    uint8_t reveal_self = s->units.invisible[i];
    int size = s->size;
    int pushed[9];
    int np = 0;
    int x0, x1, y0, y1;
    pc_ball(g, i, 1, &x0, &x1, &y0, &y1);
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            int t = y * size + x;
            if (pc_polaris_has_breakable_ice(s, t, player, r)) pushed[np++] = t;
        }
    }
    for (int k = 0; k < np; k++) {
        int t = pushed[k];
        if (s->units.owner[t] != 0 && pc_hidden_from(s, t, player)) {
            s->units.invisible[t] = 0;
            continue;
        }
        if (!pc_polaris_has_breakable_ice(s, t, player, r)) continue;
        int tx = t % size, ty = t / size;
        int land_next = 0;
        for (int d = 0; d < 4 && !land_next; d++) {
            int nx = tx + DX[d], ny = ty + DY[d];
            if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
            int32_t k2 = s->tiles.terrain[ny * size + nx];
            /* map data is_land (3..=6) other than Ice */
            land_next = k2 >= 3 && k2 <= 5;
        }
        s->tiles.terrain[t] = land_next ? 1 : 2;
    }
    pc_polaris_sync_ice_banks(s);
    if (reveal_self) s->units.invisible[i] = 0;
    pc_structures_update_routes(s, r, sc);
    pc_structures_connections(s, player, g, 1);
}

#ifdef __cplusplus
}
#endif
#endif /* PC_POLARIS_H */
