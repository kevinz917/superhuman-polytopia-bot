/* kernels.h - line-by-line port of engine-rs/src/kernels.rs: combat formula, unit health, research, movement
 * reachability, exploration and the observation. See docs/gpu-engine-design-2026-11-06.md "Porting rules". */
#ifndef PC_KERNELS_H
#define PC_KERNELS_H

#include "pc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* provided by structures.h */
/* structures::bridge_span(s, tile) -> u8: 1 north-south span, 2 east-west span, 0 none */
PC_FN uint8_t pc_structures_bridge_span(const PcState *s, int tile);

/* provided by rules.h */
/* rules::bits128(row): bit k set for every true entry of a 128-wide flag row */
PC_FN pc_u128 pc_rules_bits128(const uint8_t row[128]);

#define PC_INF 1000000   /* kernels::INF */

/* ---- local cast helpers (kernels.rs only) ---- */
/* `x as i64` on f32: saturating, NaN -> 0 */
PC_FN int64_t pc_kernels_f2i64(float x) {
    if (x != x) return 0;
    if (x >= 9223372036854775808.0f) return INT64_MAX;
    if (x <= -9223372036854775808.0f) return INT64_MIN;
    return (int64_t)x;
}
/* `x as i16` on i32: truncation to the low 16 bits */
PC_FN int16_t pc_kernels_i16(int32_t x) { return (int16_t)(uint16_t)(uint32_t)x; }
/* round10 closure of combat */
PC_FN int64_t pc_kernels_round10(int64_t v) {
    if (v % 10 < 10 - v % 10) return v / 10 * 10;
    return v / 10 * 10 + 10;
}

/* BattleHelpers.GetBattleResults in the game's integer units (health in tenths, attack and defence in hundredths,
 * defence bonus in tenths): forces are truncated, then the damage is rounded half-up to whole HP. Inputs are our
 * f32 HP/stat values and the defender's bonus (poison already halves it, as GetDefenceBonus's `>> 1`); returns
 * (attack damage, retaliation, floored half attack damage) in HP; retaliation is 0 when the attack kills. */
PC_FN void pc_kernels_combat(float aa, float ah, float am, float dd, float dh, float dm, float db,
                             float *out_damage, float *out_retaliation, float *out_half) {
    /* tenths = |x| (x * 10.0).round() as i64 */
    int64_t attack = pc_kernels_f2i64(roundf(aa * 100.0f));
    int64_t bonus = pc_kernels_f2i64(floorf((db * 10.0f) + 0.001f));
    int64_t defence = pc_kernels_f2i64(roundf(dd * 10.0f)) * bonus;
    int64_t ahi = pc_kernels_f2i64(roundf(ah * 10.0f));
    int64_t ami = pc_kernels_f2i64(roundf(am * 10.0f));
    int64_t dhi = pc_kernels_f2i64(roundf(dh * 10.0f));
    int64_t dmi = pc_kernels_f2i64(roundf(dm * 10.0f));
    if (ami < 1) ami = 1;
    if (dmi < 1) dmi = 1;
    int64_t a_force = ahi * attack * 100 / ami;
    int64_t d_force = defence * dhi * 100 / dmi;
    int64_t total = a_force + d_force;
    if (total <= 0) {
        *out_damage = 0.0f; *out_retaliation = 0.0f; *out_half = 0.0f;
        return;
    }
    int64_t damage = pc_kernels_round10(attack * a_force * 450 / (total * 1000));
    int64_t retaliation = (damage < dhi && bonus > 0) ? pc_kernels_round10(d_force * defence * 4500 / (total * 1000 * bonus)) : 0;
    *out_damage = (float)damage / 10.0f;
    *out_retaliation = (float)retaliation / 10.0f;
    *out_half = (float)(damage / 20 * 10) / 10.0f;
}

PC_FN float pc_kernels_max_health(const PcState *s, const PcRules *r, int i) {
    const PcUnits *u = &s->units;
    int32_t kind = u->passenger[i] > 0 ? u->passenger[i] : u->kind[i];
    /* (kind as usize).min(127): a negative kind wraps to a huge usize, so it clamps to 127 too */
    int k = (kind < 0 || kind > 127) ? 127 : (int)kind;
    return r->unit_health[k] + (float)((int32_t)u->veteran[i] * 5);
}

/* research_cost given the mover's city count. */
PC_FN int32_t pc_kernels_research_cost_with(const PcState *s, const PcRules *r, int k, int32_t cities) {
    int32_t cost = 4 + cities * r->tech_tier[k];
    /* GameLogicData.GetTechPrice: Literacy multiplies by 0.66666f */
    if (s->players.tech[pc_p(s)][22]) return pc_f2i(ceilf((float)cost * 0.66666f));
    return cost;
}

PC_FN int32_t pc_kernels_research_cost(const PcState *s, const PcRules *r, int k) {
    int32_t cp = s->current_player;
    int32_t cities = 0;
    for (int i = 0; i < s->n; i++) cities += (s->cities.owner[i] == cp);
    return pc_kernels_research_cost_with(s, r, k, cities);
}

/* research_mask without its `active` gate, as a bitset: techs reachable from a discovered one, not discovered,
 * existing and affordable (one city count, predecessor bitsets). With lanes (pc.h PcLanes) the city count is a
 * warp sum; every lane returns the same bitset. */
PC_FN pc_u128 pc_kernels_research_bits_lanes(const PcState *s, const PcRules *r, PcLanes L) {
    int p = pc_p(s);
    const uint8_t *discovered = s->players.tech[p];
    int32_t stars = s->players.stars[p];
    pc_u128 known = pc_rules_bits128(discovered);
    pc_u128 cand = pc_u128_zero();
    pc_u128 rest = known;
    while (pc_u128_any(rest)) {
        int k = pc_u128_lowest(rest);
        cand = pc_u128_or(cand, r->tech_succ[k]);
        pc_u128_clear(&rest, k);
    }
    cand = pc_u128_and(cand, pc_u128_and(pc_u128_not(known), r->tech_exists_bits));
    if (!pc_u128_any(cand)) return pc_u128_zero();
    int32_t cp = s->current_player;
    int32_t cities = 0;
    for (int i = L.lane; i < s->n; i += L.count) cities += (s->cities.owner[i] == cp);
    cities = pc_lanes_sum(L, cities);
    pc_u128 out = pc_u128_zero();
    while (pc_u128_any(cand)) {
        int k = pc_u128_lowest(cand);
        pc_u128_clear(&cand, k);
        if (pc_kernels_research_cost_with(s, r, k, cities) <= stars) pc_u128_set(&out, k);
    }
    return out;
}

PC_FN pc_u128 pc_kernels_research_bits(const PcState *s, const PcRules *r) {
    return pc_kernels_research_bits_lanes(s, r, pc_lanes_one());
}

/* out[128] 0/1 */
PC_FN void pc_kernels_research_mask(const PcState *s, const PcRules *r, uint8_t out[128]) {
    int active = !pc_pending_reward(s) && !s->terminated && !s->truncated;
    pc_u128 bits = active ? pc_kernels_research_bits(s, r) : pc_u128_zero();
    for (int k = 0; k < 128; k++) out[k] = (uint8_t)pc_u128_test(bits, k);
}

typedef struct {
    uint8_t passable[PC_MAXN];
    uint8_t terminal[PC_MAXN];
    uint8_t road[PC_MAXN];
    uint8_t enemy[PC_MAXN];
    int32_t budget;
} PcMovementData;

/* kernels.movement_data into reusable buffers. */
PC_FN void pc_kernels_movement_data_into(const PcState *s, int source, const PcRules *r, const PcGeom *g, PcMovementData *md) {
    int n = s->n;
    const PcTiles *t = &s->tiles;
    const PcUnits *u = &s->units;
    int32_t owner = u->owner[source];
    int p = pc_pclip(s, owner);
    const uint8_t *tech = s->players.tech[p];
    int kind = (int)u->kind[source];
    int aquatic = r->skills[kind][PC_SK_FLOAT];
    int creep = r->skills[kind][PC_SK_CREEP];
    uint8_t *passable = md->passable, *enemy = md->enemy, *road = md->road, *terminal = md->terminal;
    for (int i = 0; i < n; i++) {
        int bridge = t->structure[i] == 48;
        int water = t->terrain[i] == 1 || t->terrain[i] == 2;
        int terrain_ok = ((t->terrain[i] != 1 || tech[10]) && (t->terrain[i] != 2 || tech[13] || tech[25]) && (t->terrain[i] != 4 || tech[20])) || bridge;
        int own_port = t->structure[i] == 8 && (t->owner[i] == owner || (t->owner[i] > 0 && s->players.peace[p][pc_pclip(s, t->owner[i])]));
        int land_access = !water || bridge || own_port;
        int hidden_enemy = pc_hidden_from(s, i, owner);
        passable[i] = (uint8_t)(t->explored[p][i] && terrain_ok && (aquatic || land_access) && (u->owner[i] == 0 || u->owner[i] == owner || hidden_enemy));
        int peaceful = s->players.peace[p][pc_pclip(s, u->owner[i])];
        enemy[i] = (uint8_t)(u->owner[i] != 0 && u->owner[i] != owner && !u->invisible[i] && !peaceful);
        int friendly_road = t->owner[i] == 0 || t->owner[i] == owner || s->players.peace[p][pc_pclip(s, t->owner[i])];
        road[i] = (uint8_t)((t->road[i] || t->structure[i] == 1 || s->cities.owner[i] != 0 || bridge) && friendly_road);
    }
    int no_zoc = creep || r->skills[kind][PC_SK_INFILTRATE] || u->invisible[source];
    for (int i = 0; i < n; i++) {
        int any_enemy = 0;
        for (int a = 0; a < g->adj_n[i]; a++) {
            if (enemy[g->adj[i * 8 + a]]) { any_enemy = 1; break; }
        }
        int zoc = !no_zoc && any_enemy;
        int water = t->terrain[i] == 1 || t->terrain[i] == 2;
        int own_port = t->structure[i] == 8 && (t->owner[i] == owner || (t->owner[i] > 0 && s->players.peace[p][pc_pclip(s, t->owner[i])]));
        terminal[i] = (uint8_t)(zoc || t->terrain[i] == 4 || (t->terrain[i] == 5 && !road[i] && !creep) || (own_port && !aquatic) || (r->skills[kind][PC_SK_CARRY] && !water));
    }
    md->budget = (r->unit_movement[kind] + (int32_t)u->boosted[source]) * 2;
}

/* Unit-independent movement terms of one owner, one byte per tile (movement_flags). */
#define MV_PASS 1       /* explored, terrain allowed by tech (or bridge), no blocking unit */
#define MV_LAND 2       /* land access: not water, or a bridge, or an own port */
#define MV_ROAD 4       /* friendly road network tile */
#define MV_ZOC 8        /* next to a visible, non-peaceful, explored enemy unit */
#define MV_MOUNTAIN 16
#define MV_FOREST 32
#define MV_OWN_PORT 64
#define MV_WATER 128

/* movement_flags' first-pass byte of tile i (every bit but MV_ZOC) */
PC_FN uint8_t pc_kernels_movement_base(const PcState *s, int32_t owner, const uint8_t *tech, const uint8_t *peace,
                                       const uint8_t *explored, int i) {
    const PcTiles *t = &s->tiles;
    const PcUnits *u = &s->units;
    int32_t terrain = t->terrain[i], structure = t->structure[i];
    int bridge = structure == 48;
    int water = terrain == 1 || terrain == 2;
    int terrain_ok = ((terrain != 1 || tech[10]) && (terrain != 2 || tech[13] || tech[25]) && (terrain != 4 || tech[20])) || bridge;
    /* PathFinder.IsTileAccessible / ActionUtils.WillUnitEmbark: a peace partner's port serves like an own one */
    int own_port = structure == 8 && (t->owner[i] == owner || (t->owner[i] > 0 && peace[pc_pclip(s, t->owner[i])]));
    int32_t uo = u->owner[i];
    int hidden_enemy = pc_hidden_from(s, i, owner);
    int pass = explored[i] && terrain_ok && (uo == 0 || uo == owner || hidden_enemy);
    int friendly_road = t->owner[i] == 0 || t->owner[i] == owner || peace[pc_pclip(s, t->owner[i])];
    int road = (t->road[i] || structure == 1 || s->cities.owner[i] != 0 || bridge) && friendly_road;
    return (uint8_t)((pass ? MV_PASS : 0) | ((!water || bridge || own_port) ? MV_LAND : 0) | (road ? MV_ROAD : 0)
        | (terrain == 4 ? MV_MOUNTAIN : 0) | (terrain == 5 ? MV_FOREST : 0) | (own_port ? MV_OWN_PORT : 0)
        | (water ? MV_WATER : 0));
}

/* the unit on tile i exerts zone of control on `owner`'s units */
PC_FN int pc_kernels_zoc_unit(const PcState *s, int32_t owner, const uint8_t *peace, int i) {
    int32_t uo = s->units.owner[i];
    /* TileData.GetMovementCost reads the raw neighbours: unseen enemies exert zone of control too */
    return uo != 0 && uo != owner && !s->units.invisible[i] && !peace[pc_pclip(s, uo)];
}

/* kernels.movement_data's per-tile terms for units of `owner`, which only the unit's skills (float, creep,
 * carry, zoc immunity) and budget specialise: passable = PASS && (float || LAND); terminal = (ZOC && !no_zoc)
 * || MOUNTAIN || (FOREST && !ROAD && !creep) || (OWN_PORT && !float) || (carry && !WATER). flags[0..n) written. */
PC_FN void pc_kernels_movement_flags(const PcState *s, int32_t owner, const PcGeom *g, uint8_t flags[PC_MAXN]) {
    int n = s->n;
    int p = pc_pclip(s, owner);
    const uint8_t *tech = s->players.tech[p];
    const uint8_t *peace = s->players.peace[p];
    const uint8_t *explored = s->tiles.explored[p];
    for (int i = 0; i < n; i++) flags[i] = pc_kernels_movement_base(s, owner, tech, peace, explored, i);
    for (int i = 0; i < n; i++) {
        if (pc_kernels_zoc_unit(s, owner, peace, i)) {
            for (int a = 0; a < g->adj_n[i]; a++) flags[g->adj[i * 8 + a]] |= MV_ZOC;
        }
    }
}

/* movement_flags over cooperating lanes: with one lane the scatter above; with more, the same bytes as a gather (a
 * tile gets MV_ZOC iff one of its neighbours holds a ZOC unit: the in-board neighbour relation is symmetric), one
 * tile per lane and no write conflicts. Ends with a lane sync. */
PC_FN void pc_kernels_movement_flags_lanes(const PcState *s, int32_t owner, const PcGeom *g, uint8_t flags[PC_MAXN], PcLanes L) {
    if (L.count == 1) {
        pc_kernels_movement_flags(s, owner, g, flags);
        return;
    }
    int n = s->n;
    int p = pc_pclip(s, owner);
    const uint8_t *tech = s->players.tech[p];
    const uint8_t *peace = s->players.peace[p];
    const uint8_t *explored = s->tiles.explored[p];
    for (int i = L.lane; i < n; i += L.count) {
        uint8_t f = pc_kernels_movement_base(s, owner, tech, peace, explored, i);
        for (int a = 0; a < g->adj_n[i]; a++) {
            if (pc_kernels_zoc_unit(s, owner, peace, (int)g->adj[i * 8 + a])) { f |= MV_ZOC; break; }
        }
        flags[i] = f;
    }
    pc_lanes_sync(L);
}

/* PathFinder.IsTileAccessible: a Hide unit may pass a tile held by another player's unit (it never
 * ends there). True when tile `i` stops `owner`'s units only because of such a unit: explored, terrain allowed. */
PC_FN int pc_kernels_hide_passable(const PcState *s, int32_t owner, int i) {
    const PcTiles *t = &s->tiles;
    const PcUnits *u = &s->units;
    const uint8_t *tech = s->players.tech[pc_pclip(s, owner)];
    int32_t terrain = t->terrain[i];
    int terrain_ok = ((terrain != 1 || tech[10]) && (terrain != 2 || tech[13] || tech[25]) && (terrain != 4 || tech[20]))
        || t->structure[i] == 48;
    return u->owner[i] != 0 && u->owner[i] != owner && t->explored[pc_pclip(s, owner)][i] && terrain_ok;
}

/* PathFinder.IsTileAccessible: any unit may pass a tile held by a peace partner's unit. */
PC_FN int pc_kernels_peace_unit(const PcState *s, int32_t owner, int i) {
    int32_t o = s->units.owner[i];
    return o != 0 && o != owner && s->players.peace[pc_pclip(s, owner)][pc_pclip(s, o)];
}

/* TileData.GetMovementCost: entering a forest ends the move unless the step runs road to road. `cand` is the step's cost so far. */
PC_FN int32_t pc_kernels_forest_step(int creep, int road_edge, uint8_t dest_flags, int32_t cand, int32_t budget) {
    if (!creep && !road_edge && (dest_flags & MV_FOREST) != 0) return pc_max(cand, budget);
    return cand;
}

/* PathFinder.IsTileAccessible: without water movement (Fishing), a bridge cannot be entered across its
 * span: not straight from the east or west onto a north-south bridge, nor straight from north or south onto an
 * east-west one (diagonal steps are allowed). */
PC_FN int pc_kernels_bridge_blocks(const PcState *s, int32_t owner, int from, int to) {
    if (s->tiles.structure[to] != 48 || s->players.tech[pc_pclip(s, owner)][10]) return 0;
    int size = s->size;
    int32_t dx = (int32_t)(to % size) - (int32_t)(from % size);
    int32_t dy = (int32_t)(to / size) - (int32_t)(from / size);
    switch (pc_structures_bridge_span(s, to)) {
        case 1: return dy == 0;
        case 2: return dx == 0;
        default: return 0;
    }
}

/* Reusable buffers for movement_targets (kernels::MoveScratch). Initialise with pc_kernels_move_scratch_init once;
 * reach leaves dist back at PC_INF and the lists empty, as the Rust does. The Vec<(u16, i32)> frontier is two
 * parallel arrays; each round's `next` holds distinct tiles (stamp), so PC_MAXN entries suffice. */
typedef struct {
    uint8_t flags[PC_MAXN];
    int32_t dist[PC_MAXN];
    uint32_t stamp[PC_MAXN];
    uint32_t epoch;
    uint16_t frontier_tile[PC_MAXN];
    int32_t frontier_dist[PC_MAXN];
    int frontier_n;
    uint16_t next[PC_MAXN];
    int next_n;
    uint16_t touched[PC_MAXN];
    int touched_n;
} PcMoveScratch;

/* MoveScratch::new */
PC_FN void pc_kernels_move_scratch_init(PcMoveScratch *ms) {
    for (int i = 0; i < PC_MAXN; i++) { ms->flags[i] = 0; ms->dist[i] = PC_INF; ms->stamp[i] = 0; }
    ms->epoch = 0;
    ms->frontier_n = 0;
    ms->next_n = 0;
    ms->touched_n = 0;
}

/* Destination tiles of the unit on `source` (kernels.movement_targets without its `active` gate), given
 * `ms->flags` = movement_flags of the unit's owner. kernels.movement_targets runs 8 synchronous relaxation sweeps
 * over every tile; this runs the same 8 synchronous rounds (Jacobi: every relaxation in round k reads the
 * distances of round k-1) over the frontier only, i.e. the tiles whose distance changed in the previous round,
 * which is exact: a tile that did not change contributes nothing new.
 * The Rust `emit(tile)` callback becomes the list `reached[0..count)`: the destinations in exactly the order the
 * Rust emits them (its `touched` order = discovery order); returns count. masks.rs move_rows sets bit
 * base + reached[k] for each k; movement_targets sets out[reached[k]] = 1. */
PC_FN int pc_kernels_reach(const PcState *s, int source, const PcRules *r, const PcGeom *g, PcMoveScratch *ms,
                           uint16_t reached[PC_MAXN]) {
    const PcUnits *u = &s->units;
    int32_t owner = u->owner[source];
    int kind = (int)u->kind[source];
    const uint8_t *sk = r->skills[kind];
    int aquatic = sk[PC_SK_FLOAT], creep = sk[PC_SK_CREEP], carry = sk[PC_SK_CARRY];
    int no_zoc = creep || sk[PC_SK_INFILTRATE] || u->invisible[source];
    int32_t budget = (r->unit_movement[kind] + (int32_t)u->boosted[source]) * 2;
    int hide = sk[PC_SK_HIDE];
    /* terminal: any of these bits set (plus FOREST without ROAD for non-creepers, handled below) */
    uint8_t term_bits = (uint8_t)((no_zoc ? 0 : MV_ZOC) | MV_MOUNTAIN | (aquatic ? 0 : MV_OWN_PORT));
    uint8_t pass_need = (uint8_t)(aquatic ? MV_PASS : (MV_PASS | MV_LAND));
    uint8_t *flags = ms->flags;
    int32_t *dist = ms->dist;
    uint32_t *stamp = ms->stamp;
    ms->touched_n = 0;
    ms->frontier_n = 0;
    dist[source] = 0;
    ms->touched[ms->touched_n++] = (uint16_t)source;
    ms->frontier_tile[0] = (uint16_t)source;
    ms->frontier_dist[0] = 0;
    ms->frontier_n = 1;
    for (int round = 0; round < 8; round++) {
        if (ms->frontier_n == 0) break;
        ms->epoch = ms->epoch + 1u;   /* wrapping_add */
        if (ms->epoch == 0) {
            for (int i = 0; i < PC_MAXN; i++) stamp[i] = 0;
            ms->epoch = 1;
        }
        uint32_t epoch = ms->epoch;
        ms->next_n = 0;
        for (int f = 0; f < ms->frontier_n; f++) {
            int j = (int)ms->frontier_tile[f];
            int32_t dj = ms->frontier_dist[f];
            uint8_t fj = flags[j];
            int terminal_j = (fj & term_bits) != 0 || ((fj & (MV_FOREST | MV_ROAD)) == MV_FOREST && !creep) || (carry && (fj & MV_WATER) == 0);
            if (dj >= budget || (j != source && terminal_j)) continue;
            /* TileData.GetMovementCost: no road speed for Swim units (boats) or Creep units */
            int road_j = (fj & MV_ROAD) != 0 && !creep && !aquatic;
            for (int a = 0; a < g->adj_n[j]; a++) {
                int i = (int)g->adj[j * 8 + a];
                uint8_t fi = flags[i];
                if ((fi & pass_need) != pass_need && i != source
                    && !((hide || pc_kernels_peace_unit(s, owner, i)) && (aquatic || (fi & MV_LAND) != 0) && pc_kernels_hide_passable(s, owner, i))) {
                    continue;
                }
                if (pc_kernels_bridge_blocks(s, owner, j, i)) continue;
                int road_edge = road_j && (fi & MV_ROAD) != 0;
                int32_t cand = pc_kernels_forest_step(creep, road_edge, fi, dj + (road_edge ? 1 : 2), budget);
                if (cand < dist[i]) {
                    if (dist[i] == PC_INF) ms->touched[ms->touched_n++] = (uint16_t)i;
                    dist[i] = cand;
                    if (stamp[i] != epoch) {
                        stamp[i] = epoch;
                        ms->next[ms->next_n++] = (uint16_t)i;
                    }
                }
            }
        }
        ms->frontier_n = 0;
        for (int k = 0; k < ms->next_n; k++) {
            int i = (int)ms->next[k];
            ms->frontier_tile[ms->frontier_n] = (uint16_t)i;
            ms->frontier_dist[ms->frontier_n] = dist[i];
            ms->frontier_n++;
        }
    }
    /* PathFinder.IsAllowedToFinishOnTile: an Infiltrate unit never ends on an opponent's city, and no
     * unit ends on the city of a player it has peace or a cool-down with */
    int po = pc_pclip(s, owner);
    int count = 0;
    for (int k = 0; k < ms->touched_n; k++) {
        int i = (int)ms->touched[k];
        dist[i] = PC_INF;
        int empty = u->owner[i] == 0 || pc_hidden_from(s, i, owner);
        if (empty && i != source) {
            int32_t co = s->cities.owner[i];
            int finish_ok = co == 0 || co == owner
                || (!sk[PC_SK_INFILTRATE] && !s->players.peace[po][pc_pclip(s, co)] && s->players.truce_breaker[po][pc_pclip(s, co)] == 0);
            if (finish_ok) reached[count++] = (uint16_t)i;
        }
    }
    return count;
}

/* Destination set of the unit on `source` (kernels.movement_targets). out[0..n) 0/1. */
PC_FN void pc_kernels_movement_targets(const PcState *s, int source, const PcRules *r, const PcGeom *g, PcMoveScratch *ms,
                                       uint8_t out[PC_MAXN]) {
    int n = s->n;
    const PcUnits *u = &s->units;
    int kind = (int)u->kind[source];
    for (int i = 0; i < n; i++) out[i] = 0;
    int active = u->owner[source] == s->current_player && !u->moved[source] && r->unit_supported[kind] && !pc_pending_reward(s)
        && !s->terminated && !s->truncated;
    if (!active) return;
    pc_kernels_movement_flags(s, u->owner[source], g, ms->flags);
    uint16_t reached[PC_MAXN];
    int count = pc_kernels_reach(s, source, r, g, ms, reached);
    for (int k = 0; k < count; k++) out[reached[k]] = 1;
}

PC_FN void pc_kernels_update_exploration(PcState *s, int32_t player, const PcGeom *g) {
    int n = s->n;
    int p = (int)(player - 1);
    uint8_t explored[PC_MAXN];
    for (int t = 0; t < n; t++) explored[t] = (uint8_t)(s->tiles.explored[p][t] || !s->fog);
    for (int c = 0; c < n; c++) {
        if (s->cities.owner[c] == player) {
            int32_t range = s->tiles.capital[c] > 0 ? 2 : s->cities.border[c];
            int x0, x1, y0, y1;
            pc_ball(g, c, range, &x0, &x1, &y0, &y1);
            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    int t = y * g->size + x;
                    int corner = t == g->corners[0] || t == g->corners[1] || t == g->corners[2] || t == g->corners[3];
                    if (!corner || s->cities.owner[t] == player) explored[t] = 1;
                }
            }
        }
    }
    for (int i = 0; i < n; i++) {
        if (s->units.owner[i] == player) {
            int32_t k = s->units.kind[i];
            int32_t range = (s->tiles.terrain[i] == 4 || k == 42 || k == 38 || k == 39) ? 2 : 1;
            int x0, x1, y0, y1;
            pc_ball(g, i, range, &x0, &x1, &y0, &y1);
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) explored[y * g->size + x] = 1;
        }
    }
    int32_t gain = 0;
    for (int t = 0; t < n; t++) {
        if (explored[t] && !s->tiles.explored[p][t]) gain += 5;
        s->tiles.explored[p][t] = explored[t];
    }
    s->players.score[p] = pc_wadd(s->players.score[p], gain);
}

/* Packed public view: kernels.observe(state, player) fields in sorted-key order, int16 (cpu_engine.pack_view).
 * Games with more than two seats add "alive" (first in sorted order) and give the five relation fields one entry
 * per seat (view_fields / view_width).
 * "unit_direction" (2026-10-31) is the facing of each visible unit plus one (0: no visible unit, -1: unexplored);
 * observation.encode_view gives it to networks of observation version 2 only.
 * "enemy_tech" [128] and "hist_enemy_built" [n], "hist_enemy_trained" [n], "hist_enemy_unit" [HIST_TURNS, n]
 * (2026-11-01) are the seat's observation history (history.rs), for networks of observation version 3.
 * Field order (VIEW_FIELDS): capital, detect, embassies_received, embassies_sent, enemy_tech, explored,
 * hist_enemy_built, hist_enemy_trained, hist_enemy_unit, known_players, own_built, own_city_connected,
 * own_city_level, own_city_population, own_city_production, own_city_progress, own_city_rewards, own_city_riot,
 * own_monuments, own_unit_attacked, own_unit_boosted, own_unit_frozen, own_unit_home, own_unit_invisible,
 * own_unit_kills, own_unit_moved, own_unit_passenger, own_unit_poison, owner, peace, peace_offers, player,
 * resource, road, stars, structure, structure_level, tech, terrain, turn, unit_direction, unit_health,
 * unit_owner, unit_type, unit_veteran. */
#define PC_VIEW_FIELDS 45

PC_FN int pc_kernels_view_width(int n) { return (40 + PC_HIST_TURNS) * n + 397 + 128; }

/* Packed view width for `np` seats: relations take np entries each, plus "alive" when np > 2. */
PC_FN int pc_kernels_view_width_np(int n, int np) { return pc_kernels_view_width(n) + (np > 2 ? 6 * np - 10 : 0); }

/* kernels.observe packed in VIEW_FIELDS order: one tight loop per field. Per tile: vis = explored || !fog, the
 * seat's unit = units.owner == player, uvis = vis && (own unit || not hidden from the player); the seat's city and
 * unit fields ("own" fields) are the value on the seat's own tiles and zero elsewhere. Scalars and the 128-wide rows
 * are written by lane 0 / strided; with lanes (pc.h PcLanes) every tile loop is strided over the lanes, every entry
 * written exactly once. out holds pc_kernels_view_width_np(n, np) entries. */
PC_FN void pc_kernels_observe_lanes(const PcState *s, int32_t player, const PcGeom *g, int16_t *out, PcLanes L) {
    int n = s->n;
    int np = s->np;
    int p = (int)(player - 1);
    const PcTiles *t = &s->tiles;
    const PcUnits *u = &s->units;
    const PcCities *c = &s->cities;
    const PcPlayers *pl = &s->players;
    const uint8_t *tech = pl->tech[p];
    const uint8_t *explored = t->explored[p];
    const int nofog = !s->fog;
    const int l0 = L.lane, lc = L.count;
#define PC_OBS_VIS(i) (explored[i] || nofog)
#define PC_OBS_UVIS(i) (PC_OBS_VIS(i) && (u->owner[i] == player || !pc_hidden_from(s, (i), player)))
    int at = 0;
    if (np > 2) {
        for (int k = l0; k < np; k += lc) out[at + k] = (int16_t)pc_alive(s, k);
        at += np;
    }
    for (int i = l0; i < n; i += lc) out[at + i] = PC_OBS_VIS(i) ? pc_kernels_i16(t->capital[i]) : -1;
    at += n;
    /* detect: on the seat's unit tiles, a hidden unit next to it */
    for (int i = l0; i < n; i += lc) {
        int any = 0;
        if (u->owner[i] == player) {
            for (int a = 0; a < g->adj_n[i]; a++) {
                int j = (int)g->adj[i * 8 + a];
                if (pc_hidden_from(s, j, player)) { any = 1; break; }
            }
        }
        out[at + i] = (int16_t)any;
    }
    at += n;
    for (int k = l0; k < np; k += lc) out[at + k] = (int16_t)pl->embassy[k][p];
    at += np;
    for (int k = l0; k < np; k += lc) out[at + k] = (int16_t)pl->embassy[p][k];
    at += np;
    const PcHistory *hist = &s->history[p];
    for (int k = l0; k < 128; k += lc) out[at + k] = (int16_t)pc_u128_test(hist->enemy_tech, k);
    at += 128;
    for (int i = l0; i < n; i += lc) out[at + i] = (int16_t)PC_OBS_VIS(i);
    at += n;
    for (int i = l0; i < n; i += lc) out[at + i] = (int16_t)hist->built[i];
    at += n;
    for (int i = l0; i < n; i += lc) out[at + i] = (int16_t)hist->trained[i];
    at += n;
    for (int k = 0; k < PC_HIST_TURNS; k++) {
        for (int i = l0; i < n; i += lc) out[at + i] = (int16_t)hist->seen_unit[k][i];
        at += n;
    }
    for (int k = l0; k < np; k += lc) out[at + k] = (int16_t)pl->known[p][k];
    at += np;
    for (int k = l0; k < 128; k += lc) out[at + k] = (int16_t)pl->built[p][k];
    at += 128;
    /* own city fields: own_city_connected, level, population, production, progress, rewards [n, 9], riot */
#define PC_OBS_CITY_BOOL(field) do { for (int i = l0; i < n; i += lc) out[at + i] = c->owner[i] == player ? (int16_t)c->field[i] : 0; at += n; } while (0)
#define PC_OBS_CITY_I32(field) do { for (int i = l0; i < n; i += lc) out[at + i] = c->owner[i] == player ? pc_kernels_i16(c->field[i]) : 0; at += n; } while (0)
    PC_OBS_CITY_BOOL(connected);
    PC_OBS_CITY_I32(level);
    PC_OBS_CITY_I32(population);
    PC_OBS_CITY_I32(production);
    PC_OBS_CITY_I32(progress);
    for (int i = l0; i < n; i += lc) {
        int own = c->owner[i] == player;
        for (int q = 0; q < 9; q++) out[at + i * 9 + q] = own ? pc_kernels_i16(c->rewards[i * 9 + q]) : 0;
    }
    at += n * 9;
    PC_OBS_CITY_BOOL(riot);
#undef PC_OBS_CITY_BOOL
#undef PC_OBS_CITY_I32
    for (int k = l0; k < 128; k += lc) out[at + k] = (int16_t)pl->monuments[p][k];
    at += 128;
    /* own unit fields */
#define PC_OBS_UNIT_BOOL(field) do { for (int i = l0; i < n; i += lc) out[at + i] = u->owner[i] == player ? (int16_t)u->field[i] : 0; at += n; } while (0)
#define PC_OBS_UNIT_I32(field) do { for (int i = l0; i < n; i += lc) out[at + i] = u->owner[i] == player ? pc_kernels_i16(u->field[i]) : 0; at += n; } while (0)
    PC_OBS_UNIT_BOOL(attacked);
    PC_OBS_UNIT_BOOL(boosted);
    PC_OBS_UNIT_BOOL(frozen);
    PC_OBS_UNIT_I32(home);
    PC_OBS_UNIT_BOOL(invisible);
    PC_OBS_UNIT_I32(kills);
    PC_OBS_UNIT_BOOL(moved);
    PC_OBS_UNIT_I32(passenger);
    PC_OBS_UNIT_BOOL(poison);
#undef PC_OBS_UNIT_BOOL
#undef PC_OBS_UNIT_I32
    for (int i = l0; i < n; i += lc) out[at + i] = PC_OBS_VIS(i) ? pc_kernels_i16(t->owner[i]) : -1;
    at += n;
    for (int k = l0; k < np; k += lc) out[at + k] = (int16_t)pl->peace[p][k];
    at += np;
    for (int k = l0; k < np; k += lc) out[at + k] = (int16_t)pl->peace_offer[p][k];
    at += np;
    if (l0 == 0) out[at] = pc_kernels_i16(player);
    at += 1;
    int hide2 = !(tech[6] || tech[8] || tech[9]);
    int hide5 = !(tech[20] || tech[23] || tech[24]);
    /* ResourceDataExtensions.IsResourceVisibleToPlayer: starfish only once Star Fishing (Navigation) is unlockable */
    int hide8 = !(tech[13] || tech[14]);
    for (int i = l0; i < n; i += lc) {
        int32_t rs = t->resource[i];
        int hidden = (rs == 2 && hide2) || (rs == 5 && hide5) || (rs == 8 && hide8);
        out[at + i] = PC_OBS_VIS(i) ? (hidden ? 0 : pc_kernels_i16(rs)) : -1;
    }
    at += n;
    for (int i = l0; i < n; i += lc) out[at + i] = PC_OBS_VIS(i) ? (int16_t)t->road[i] : -1;
    at += n;
    if (l0 == 0) out[at] = pc_kernels_i16(pl->stars[p]);
    at += 1;
    for (int i = l0; i < n; i += lc) out[at + i] = PC_OBS_VIS(i) ? pc_kernels_i16(t->structure[i]) : -1;
    at += n;
    for (int i = l0; i < n; i += lc) out[at + i] = PC_OBS_VIS(i) ? pc_kernels_i16(t->structure_level[i]) : -1;
    at += n;
    for (int k = l0; k < 128; k += lc) out[at + k] = (int16_t)tech[k];
    at += 128;
    for (int i = l0; i < n; i += lc) out[at + i] = PC_OBS_VIS(i) ? pc_kernels_i16(t->terrain[i]) : -1;
    at += n;
    if (l0 == 0) out[at] = pc_kernels_i16(s->turn);
    at += 1;
    /* UnitState.direction of every visible unit, + 1 (the sprite's facing; TryPushUnit starts from it); then
     * unit_health (float32 in the view; the pack casts it to int16 toward zero), unit_owner, unit_type, unit_veteran:
     * visible ? (unit visible ? v : 0) : -1 */
    for (int i = l0; i < n; i += lc) {
        int vis = PC_OBS_VIS(i), uvis = PC_OBS_UVIS(i);
        out[at + i] = vis ? ((uvis && u->owner[i] != 0) ? pc_kernels_i16(u->direction[i] + 1) : 0) : -1;
        float h = vis ? (uvis ? u->health[i] : 0.0f) : -1.0f;
        out[at + n + i] = pc_f2i16(h);
        out[at + 2 * n + i] = vis ? (uvis ? pc_kernels_i16(u->owner[i]) : 0) : -1;
        out[at + 3 * n + i] = vis ? (uvis ? pc_kernels_i16(u->kind[i]) : 0) : -1;
        out[at + 4 * n + i] = vis ? (uvis ? (int16_t)u->veteran[i] : 0) : -1;
    }
    at += 5 * n;
#undef PC_OBS_VIS
#undef PC_OBS_UVIS
    (void)at;   /* == pc_kernels_view_width_np(n, np) */
}

PC_FN void pc_kernels_observe(const PcState *s, int32_t player, const PcGeom *g, int16_t *out) {
    pc_kernels_observe_lanes(s, player, g, out, pc_lanes_one());
}

#ifdef __cplusplus
}
#endif
#endif /* PC_KERNELS_H */
