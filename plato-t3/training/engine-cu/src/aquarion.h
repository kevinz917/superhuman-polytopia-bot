/* aquarion.h - line-by-line port of engine-rs/src/aquarion.rs (Aquarion, increment S2, step 4): flooding and the
 * Bubble effect. Rules as in the
 * project's full-game engine port.
 *
 * `tiles.flooded` is tile effect flooded, `units.bubble` unit effect bubble; the Bubble player ability is Waterways'
 * (48). Every `pub fn f` is `pc_aquarion_f`, same argument order; the functions that run update routes take the
 * PcStructScratch pc_structures_update_routes needs (the Rust allocates inside it). */
#ifndef PC_AQUARION_H
#define PC_AQUARION_H

#include "pc.h"
#include "tribes.h"       /* PC_AQUARION */
#include "structures.h"   /* pc_structures_update_routes, pc_structures_connections, PcStructScratch */

#ifdef __cplusplus
extern "C" {
#endif

#define PC_WATERWAYS 48

/* is floodable: not water, not already Flooded, not Mountain or Ice */
PC_FN int pc_aquarion_floodable(const PcState *s, int t) {
    int32_t k = s->tiles.terrain[t];
    return !(k == 1 || k == 2) && !s->tiles.flooded[t] && (k & ~2) != 4;
}

/* player state has the Bubble ability: Waterways known */
PC_FN int pc_aquarion_bubble_ability(const PcState *s, int32_t player) {
    return player > 0 && s->players.tech[pc_pclip(s, player)][PC_WATERWAYS];
}

/* check bubble step as its bubble action runs: on a Flooded tile an Amphibious unit gains Bubble */
PC_FN void pc_aquarion_check_bubble_step(PcState *s, int t, const PcRules *r) {
    int kind = (int)pc_clamp(s->units.kind[t], 0, 127);
    if (s->tiles.flooded[t] && s->units.owner[t] != 0 && r->amphibious[kind]) s->units.bubble[t] = 1;
}

/* flood tile action for `player`: skipped unless floodable; Flooded; update routes
 * then update city connections(player); a unit on the tile is bubbled when the player has the Bubble ability */
PC_FN void pc_aquarion_flood_tile(PcState *s, int t, int32_t player, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    if (!pc_aquarion_floodable(s, t)) return;
    s->tiles.flooded[t] = 1;
    pc_structures_update_routes(s, r, sc);
    pc_structures_connections(s, player, g, 1);
    if (s->units.owner[t] != 0 && pc_aquarion_bubble_ability(s, player)) pc_aquarion_check_bubble_step(s, t, r);
}

/* move action's Bubble loss and AutoFlood (not for a push) for the unit that
 * arrived on `t`, `had` its Bubble before the move; returns whether `t` was Flooded on arrival */
PC_FN uint8_t pc_aquarion_after_move(PcState *s, int t, uint8_t had, int32_t mover, const PcRules *r, const PcGeom *g,
                                     uint8_t autoflood, PcStructScratch *sc) {
    int kind = (int)pc_clamp(s->units.kind[t], 0, 127);
    uint8_t flooded_before = s->tiles.flooded[t];
    int32_t terrain = s->tiles.terrain[t];
    if (had && terrain >= 3 && terrain <= 6 && !flooded_before && r->amphibious[kind]) s->units.bubble[t] = 0;
    if (autoflood && r->autoflood[kind]) pc_aquarion_flood_tile(s, t, mover, r, g, sc);
    return flooded_before;
}

/* move action's check bubble step, decided on arrival and run last */
PC_FN void pc_aquarion_bubble_on_arrival(PcState *s, int t, uint8_t had, uint8_t flooded_at_move, int32_t mover,
                                         const PcRules *r) {
    if (flooded_at_move && !had && s->tiles.flooded[t] && pc_aquarion_bubble_ability(s, mover)) pc_aquarion_check_bubble_step(s, t, r);
}

/* get unit actions' Build of the Manual Canal (52): the mover's unit, not moved, not attacked, not
 * Frozen, Waterways, on a Field or Forest that is not Flooded (any owner, any improvement) */
PC_FN int pc_aquarion_can_canal(const PcState *s, int i) {
    const PcUnits *u = &s->units;
    return u->owner[i] == s->current_player && u->owner[i] != 0 && !u->moved[i] && !u->attacked[i] && !u->frozen[i]
        && s->players.tech[pc_p(s)][PC_WATERWAYS] && pc_aquarion_floodable(s, i);
}

/* build command(Canal) -> build action: no improvement, cost 0; flood tile action; the unit is moved and attacked, an
 * invisible one revealed */
PC_FN void pc_aquarion_canal_command(PcState *s, int i, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int32_t player = s->current_player;
    s->units.moved[i] = 1;
    s->units.attacked[i] = 1;
    s->units.move_lock[i] = 0;
    pc_aquarion_flood_tile(s, i, player, r, g, sc);
    s->units.invisible[i] = 0;
}

/* get unit actions' clear tile effect Flooded: the mover's unit can attack and move, its
 * tile is Flooded, the player is not Aquarion, the unit is Amphibious or has no Swim */
PC_FN int pc_aquarion_can_clear_flood(const PcState *s, int i, const PcRules *r) {
    const PcUnits *u = &s->units;
    int kind = (int)pc_clamp(u->kind[i], 0, 127);
    return u->owner[i] == s->current_player && u->owner[i] != 0 && !u->moved[i] && !u->attacked[i] && !u->frozen[i]
        && s->tiles.flooded[i] && s->players.tribe[pc_p(s)] != PC_AQUARION
        && (r->amphibious[kind] || !r->skills[kind][PC_SK_FLOAT]);
}

/* clear tile effect action(Flooded, by unit): the effect goes; update routes then update city connections(player); the unit
 * is exhausted (moved, attacked, Boosted lost) */
PC_FN void pc_aquarion_clear_flood_command(PcState *s, int i, const PcRules *r, const PcGeom *g, PcStructScratch *sc) {
    int32_t player = s->current_player;
    s->tiles.flooded[i] = 0;
    pc_structures_update_routes(s, r, sc);
    pc_structures_connections(s, player, g, 1);
    s->units.moved[i] = 1;
    s->units.attacked[i] = 1;
    s->units.move_lock[i] = 0;
    s->units.boosted[i] = 0;
    s->units.double_ready[i] = 0;    /* make exhausted */
}

#ifdef __cplusplus
}
#endif
#endif /* PC_AQUARION_H */
