/* rules.h - port of engine-rs/src/rules.rs: the primary rule tables of gpu/rules.json (PcRulesInput, packed by
 * gpu_engine.py::pack_rules) and the derivations of `Rules::from_json` (pc_rules_finalize -> PcRules in pc.h).
 *
 * The JSON parsing itself stays in Python; this header only ports the code that runs after serde in from_json, in the
 * same order, with the same hand-written ids (kind 37, techs 2, 6, 8 ... exactly as rules.rs).
 *
 * Nulls: the packer writes -1 for a JSON null in every Option<i32>/Option<i64> field. from_json maps them with
 * unwrap_or: cost -> -1 (so -1 stays -1), population/score/stars/resource -> 0, requires -> -1 (free_requires) /
 * dropped (edges), tribe -> "none" (allowed). Nothing in rules.json has a negative population/score/stars/resource,
 * so the -1 sentinel is lossless there; a structure cost of -1 and null both become -1 in the Rust too. */
#ifndef PC_RULES_H
#define PC_RULES_H

#include "pc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* rules.rs `pub const` ids: FLOAT..INFILTRATE are PC_SK_FLOAT..PC_SK_INFILTRATE in pc.h (no others exist). */

/* Extra skill strings from_json tests besides the 10 PC_SK_* flags; index into PcRulesInput.unit_flags[kind][*]. */
#define PC_UF_DASH 0
#define PC_UF_FORTIFY 1
#define PC_UF_STIFF 2
#define PC_UF_PERSIST 3
#define PC_UF_ESCAPE 4
#define PC_UF_STATIC 5
#define PC_UNIT_NAME_MAX 32
#define PC_TECH_NEXT_MAX 128

typedef struct {
    /* units, indexed by unit id 0..127; *_exists marks ids present in the JSON */
    float unit_health[128], unit_attack[128], unit_defense[128];
    int32_t unit_movement[128], unit_range[128], unit_cost[128];
    uint8_t unit_skills[128][10];      /* skill flags in rules.rs const order FLOAT=0 .. INFILTRATE=9 */
    uint8_t unit_exists[128];
    int32_t unit_order[128], unit_count;   /* unit ids in JSON table order */
    /* technologies, indexed by tech id 0..127 */
    int32_t tech_tier[128], tech_requires[128];   /* requires: -1 = JSON null */
    uint8_t tech_next[128][128];       /* [tech][next tech] */
    uint8_t tech_structures[128][128]; /* [tech][structure] */
    uint8_t tech_exists[128];
    int32_t tech_order[128], tech_count;   /* tech ids in JSON table order */
    /* structures, indexed by structure id 0..127; -1 = JSON null for the Option fields */
    int32_t struct_cost[128], struct_population[128], struct_score[128], struct_stars[128], struct_resource[128], struct_tribe[128];
    uint8_t struct_terrain[128][16], struct_adjacent[128][128], struct_per_city[128], struct_per_tribe[128], struct_exists[128];
    int32_t struct_order[128], struct_count;
    char engine_revision[64];
    /* ---- appended after the design text: from_json inputs the fields above cannot carry (the Python mirror must
     * add these four fields, in this order, after engine_revision) ---- */
    char unit_name[128][PC_UNIT_NAME_MAX];    /* JSON "name", NUL-terminated: core_names membership -> unit_supported */
    uint8_t unit_flags[128][6];               /* skill strings in PC_UF_* order: Dash, Fortify, Stiff, Persist, Escape, Static */
    int32_t tech_next_list[128][PC_TECH_NEXT_MAX];   /* JSON "next" verbatim, in JSON order (order drives tech_tree) */
    int32_t tech_next_n[128];                 /* entries of tech_next_list[tech]; its in-range set must equal tech_next[tech] */
} PcRulesInput;

/* pc_rules_finalize return codes */
#define PC_RULES_OK 0
#define PC_RULES_ERR_UNIT_ID 1      /* unit id out of range / unit_order inconsistent with unit_exists (Rust: Err) */
#define PC_RULES_ERR_TECH_ID 2      /* tech id out of range / tech_order inconsistent with tech_exists (Rust: Err) */
#define PC_RULES_ERR_STRUCT_ID 3    /* structure id out of range / struct_order inconsistent with struct_exists */
#define PC_RULES_ERR_COST_TABLE 4   /* a CostIndex spans more than PC_COST_TABLE_MAX stars */
#define PC_RULES_ERR_TECH_TREE 5    /* tech_tree flattening overflows tech_tree_tech / tech_tree_structs */
#define PC_RULES_ERR_NEXT_ORDER 6   /* tech_next_list[t] (in-range entries) disagrees with tech_next[t] */

/* rules.rs bits128: bit k set for every true entry of a 128-wide flag row */
PC_FN pc_u128 pc_bits128(const uint8_t *row) {
    pc_u128 b = pc_u128_zero();
    for (int k = 0; k < 128; k++) {
        if (row[k]) pc_u128_set(&b, k);
    }
    return b;
}
/* module-qualified alias (rules::bits128) used by kernels.h */
PC_FN pc_u128 pc_rules_bits128(const uint8_t *row) { return pc_bits128(row); }

/* CostIndex::new: table[s - min] = kinds whose cost <= s, for s in min..=max. 0 ok, nonzero if the span overflows. */
PC_FN int pc_cost_index_new(PcCostIndex *ci, const int32_t *cost) {
    int32_t min = cost[0], max = cost[0];
    for (int k = 1; k < 128; k++) {
        if (cost[k] < min) min = cost[k];
        if (cost[k] > max) max = cost[k];
    }
    int64_t span = (int64_t)max - (int64_t)min + 1;
    if (span > PC_COST_TABLE_MAX) return PC_RULES_ERR_COST_TABLE;
    ci->min = min;
    ci->len = (int32_t)span;
    for (int32_t j = 0; j < ci->len; j++) {
        int32_t s = min + j;
        pc_u128 b = pc_u128_zero();
        for (int k = 0; k < 128; k++) {
            if (cost[k] <= s) pc_u128_set(&b, k);
        }
        ci->table[j] = b;
    }
    for (int32_t j = ci->len; j < PC_COST_TABLE_MAX; j++) ci->table[j] = pc_u128_zero();
    return PC_RULES_OK;
}

/* CostIndex::le(stars) */
PC_FN pc_u128 pc_cost_le(const PcCostIndex *ci, int32_t stars) {
    if (stars < ci->min) return pc_u128_zero();
    int32_t idx = stars - ci->min;
    if (idx > ci->len - 1) idx = ci->len - 1;
    return ci->table[idx];
}

/* NUL-terminated string equality (no libc strcmp in device code) */
PC_FN int pc_streq(const char *a, const char *b, int max) {
    for (int i = 0; i < max; i++) {
        if (a[i] != b[i]) return 0;
        if (a[i] == 0) return 1;
    }
    return 1;
}

/* the derivation half of Rules::from_json, after serde and the Rules literal with its defaults */
PC_FN int pc_rules_finalize(const PcRulesInput *in, PcRules *out) {
    static const char core_names[18][12] = {
        "Raft", "Rammership", "Scoutship", "Bomber", "Juggernaut", "MindBender", "Warrior", "Rider", "Knight",
        "Defender", "Catapult", "Archer", "Swordsman", "Giant", "Cloak", "Dagger", "Dinghy", "Pirate",
    };
    static const int32_t unit_tech_table[12][2] = {
        {2, 0}, {3, 1}, {4, 3}, {5, 7}, {8, 17}, {9, 18}, {10, 22}, {11, 24}, {38, 38}, {41, 14}, {42, 13}, {44, 44},
    };
    PcRules *r = out;
    PcStructureRules *rs = &out->structures;

    /* the Rules literal: every table zero/false except these defaults */
    memset(r, 0, sizeof(*r));
    for (int k = 0; k < 128; k++) {
        r->tech_tier[k] = 1;
        r->unit_tech[k] = 127;
        rs->cost[k] = -1;
        r->free_requires[k] = -1;
    }
    rs->cost_le.min = 0; rs->cost_le.len = 1;     /* CostIndex { min: 0, table: vec![0] } */
    r->unit_cost_le.min = 0; r->unit_cost_le.len = 1;
    r->tech_tree_n = 0;
    r->tech_tree_struct_off[0] = 0;

    /* for u in &data.units */
    if (in->unit_count < 0 || in->unit_count > 128) return PC_RULES_ERR_UNIT_ID;
    for (int ui = 0; ui < in->unit_count; ui++) {
        int32_t i = in->unit_order[ui];
        if (i < 0 || i >= 128) return PC_RULES_ERR_UNIT_ID;   /* Err("unit id out of range") */
        if (!in->unit_exists[i]) return PC_RULES_ERR_UNIT_ID;
        r->unit_health[i] = in->unit_health[i];
        r->unit_attack[i] = in->unit_attack[i];
        r->unit_defense[i] = in->unit_defense[i];
        r->unit_movement[i] = in->unit_movement[i];
        r->unit_range[i] = in->unit_range[i];
        r->dash[i] = in->unit_flags[i][PC_UF_DASH] != 0;
        r->unit_supported[i] = 0;
        for (int c = 0; c < 18; c++) {
            if (pc_streq(in->unit_name[i], core_names[c], PC_UNIT_NAME_MAX)) { r->unit_supported[i] = 1; break; }
        }
        for (int k = 0; k < 10; k++) r->skills[i][k] = in->unit_skills[i][k] != 0;
        r->unit_cost[i] = in->unit_cost[i];
        r->fortify[i] = in->unit_flags[i][PC_UF_FORTIFY] != 0;
        r->stiff[i] = in->unit_flags[i][PC_UF_STIFF] != 0;
        r->persist[i] = in->unit_flags[i][PC_UF_PERSIST] != 0;
        r->escape[i] = in->unit_flags[i][PC_UF_ESCAPE] != 0;
        r->stat[i] = in->unit_flags[i][PC_UF_STATIC] != 0;
    }

    /* for t in &data.technologies (rows with id < 0 are skipped by the Rust and absent from tech_order) */
    if (in->tech_count < 0 || in->tech_count > 128) return PC_RULES_ERR_TECH_ID;
    for (int ti = 0; ti < in->tech_count; ti++) {
        int32_t i = in->tech_order[ti];
        if (i < 0 || i >= 128) return PC_RULES_ERR_TECH_ID;    /* Err("tech id out of range") */
        if (!in->tech_exists[i]) return PC_RULES_ERR_TECH_ID;
        r->tech_tier[i] = in->tech_tier[i];
        r->tech_exists[i] = 1;
        /* others = next + [requires]; edge for every o in 0..128 */
        pc_u128 from_list = pc_u128_zero();
        int32_t nn = in->tech_next_n[i];
        if (nn < 0 || nn > PC_TECH_NEXT_MAX) return PC_RULES_ERR_NEXT_ORDER;
        for (int32_t j = 0; j < nn; j++) {
            int32_t o = in->tech_next_list[i][j];
            if (o >= 0 && o < 128) pc_u128_set(&from_list, o);
        }
        for (int o = 0; o < 128; o++) {
            if (in->tech_next[i][o]) r->tech_edges[i][o] = 1;
        }
        if (!pc_u128_eq(from_list, pc_bits128(in->tech_next[i]))) return PC_RULES_ERR_NEXT_ORDER;
        if (in->tech_requires[i] != -1) {
            int32_t o = in->tech_requires[i];
            if (o >= 0 && o < 128) r->tech_edges[i][o] = 1;
        }
    }
    for (int i = 25; i < 128; i++) r->tech_exists[i] = 0;
    r->tech_exists[38] = 1;
    r->tech_exists[44] = 1;

    /* next[t.id] = t.next filtered to 0..128, in JSON order; then GetAllTechForTribe: breadth-first from Basic, at most
     * 200 dequeues, as the game. Only the first 200 queue slots can ever be dequeued, so only those are stored;
     * qlen still counts every push (queue.len()). */
    {
        int32_t queue[200];
        int32_t qlen = 1, at = 0, nstructs = 0;
        queue[0] = 0;
        while (at < qlen && at < 200) {
            int32_t k = queue[at];
            at += 1;
            if (!r->tech_exists[k]) continue;
            int seen = 0;
            for (int32_t e = 0; e < r->tech_tree_n; e++) {
                if (r->tech_tree_tech[e] == k) { seen = 1; break; }
            }
            if (seen) continue;
            if (r->tech_tree_n >= 128) return PC_RULES_ERR_TECH_TREE;
            r->tech_tree_tech[r->tech_tree_n] = k;
            int32_t nn = in->tech_exists[k] ? in->tech_next_n[k] : 0;
            for (int32_t j = 0; j < nn; j++) {
                int32_t c = in->tech_next_list[k][j];
                if (c < 0 || c >= 128) continue;
                if (nstructs >= 1024) return PC_RULES_ERR_TECH_TREE;
                r->tech_tree_structs[nstructs++] = c;
                if (qlen < 200) queue[qlen] = c;
                qlen += 1;
            }
            r->tech_tree_n += 1;
            r->tech_tree_struct_off[r->tech_tree_n] = nstructs;
        }
    }

    for (int e = 0; e < 12; e++) r->unit_tech[unit_tech_table[e][0]] = unit_tech_table[e][1];

    /* for st in &data.structures */
    if (in->struct_count < 0 || in->struct_count > 128) return PC_RULES_ERR_STRUCT_ID;
    for (int si = 0; si < in->struct_count; si++) {
        int32_t i = in->struct_order[si];
        if (i < 0 || i >= 128) return PC_RULES_ERR_STRUCT_ID;   /* Err("structure id out of range") */
        if (!in->struct_exists[i]) return PC_RULES_ERR_STRUCT_ID;
        rs->cost[i] = in->struct_cost[i];                                               /* unwrap_or(-1) */
        rs->population[i] = in->struct_population[i] == -1 ? 0 : in->struct_population[i];   /* unwrap_or(0) */
        rs->score[i] = in->struct_score[i] == -1 ? 0 : in->struct_score[i];
        rs->stars[i] = in->struct_stars[i] == -1 ? 0 : in->struct_stars[i];
        rs->resource[i] = in->struct_resource[i] == -1 ? 0 : in->struct_resource[i];
        for (int t = 0; t < 16; t++) {
            if (in->struct_terrain[i][t]) rs->terrain[i][t] = 1;
        }
        for (int a = 0; a < 128; a++) {
            if (in->struct_adjacent[i][a]) {
                rs->adjacent[i][a] = 1;
                rs->any_adjacent[i] = 1;
            }
        }
        rs->per_city[i] = in->struct_per_city[i] != 0;
        rs->per_tribe[i] = in->struct_per_tribe[i] != 0;
        rs->allowed[i] = in->struct_tribe[i] == -1 || in->struct_tribe[i] == 4;   /* tribe.is_none() || tribe == Some(4) */
    }

    /* unlock[tech][structure] for every tech row with id >= 0 */
    for (int ti = 0; ti < in->tech_count; ti++) {
        int32_t i = in->tech_order[ti];
        for (int st = 0; st < 128; st++) {
            if (in->tech_structures[i][st]) rs->unlock[i][st] = 1;
        }
    }

    /* free techs: ids 1..=24, 38, 44, 48, 121 */
    for (int ti = 0; ti < in->tech_count; ti++) {
        int32_t i = in->tech_order[ti];
        if ((i >= 1 && i <= 24) || i == 38 || i == 44 || i == 48 || i == 121) {
            r->free_tech[i] = 1;
            r->free_requires[i] = in->tech_requires[i];   /* requires.map(|v| v as i32).unwrap_or(-1) */
        }
    }
    for (int i = 0; i < 128; i++) r->free_tech[i] = r->free_tech[i] && r->tech_exists[i];

    /* derived bitsets */
    for (int k = 0; k < 128; k++) {
        for (int i = 0; i < 128; i++) {
            if (r->tech_edges[i][k]) pc_u128_set(&r->tech_pred[k], i);
            if (rs->unlock[i][k]) pc_u128_set(&rs->unlock_by[k], i);
            if (rs->adjacent[k][i]) pc_u128_set(&rs->adjacent_rev[i], k);
        }
        for (int t = 0; t < 16; t++) {
            if (rs->terrain[k][t]) pc_u128_set(&rs->terrain_bits[t], k);
        }
        if (!rs->any_adjacent[k]) pc_u128_set(&rs->no_adjacent_bits, k);
        if (rs->per_city[k]) pc_u128_set(&rs->per_city_bits, k);
    }
    for (int k = 0; k < 128; k++) r->tech_closure[k] = pc_u128_bit(k);
    for (;;) {
        int changed = 0;
        for (int k = 0; k < 128; k++) {
            pc_u128 c = r->tech_closure[k];
            pc_u128 pred = r->tech_pred[k];
            while (pc_u128_any(pred)) {
                int i = pc_u128_lowest(pred);
                pc_u128_clear(&pred, i);
                c = pc_u128_or(c, r->tech_closure[i]);
            }
            if (!pc_u128_eq(c, r->tech_closure[k])) {
                r->tech_closure[k] = c;
                changed = 1;
            }
        }
        if (!changed) break;
    }
    r->supported_bits = pc_bits128(r->unit_supported);
    r->tech_exists_bits = pc_bits128(r->tech_exists);
    for (int i = 0; i < 128; i++) {
        r->tech_succ[i] = pc_bits128(r->tech_edges[i]);
        rs->tech_unlocks[i] = pc_bits128(rs->unlock[i]);
    }
    for (int k = 0; k < 128; k++) {
        if (rs->allowed[k] && rs->cost[k] >= 0 && rs->resource[k] == 0) pc_u128_set(&rs->static_ok, k);
        if (rs->per_tribe[k]) pc_u128_set(&rs->per_tribe_bits, k);
        int summonable = r->unit_cost[k] >= 1 && r->unit_tech[k] < 127;
        if (summonable) {
            pc_u128_set(&r->units_by_tech[r->unit_tech[k]], k);
            if (k == 2) pc_u128_set(&r->unit_static, k);
        }
    }
    if (pc_cost_index_new(&rs->cost_le, rs->cost) != PC_RULES_OK) return PC_RULES_ERR_COST_TABLE;
    if (pc_cost_index_new(&r->unit_cost_le, r->unit_cost) != PC_RULES_OK) return PC_RULES_ERR_COST_TABLE;
    /* engine_revision: PcRules (pc.h) has no string field; the driver keeps in->engine_revision itself. */
    return PC_RULES_OK;
}

#ifdef __cplusplus
}
#endif
#endif /* PC_RULES_H */
