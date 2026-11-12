/* tribes.h - port of the tribe overrides of engine-rs/src/tribes.rs (Polaris S1, Aquarion S2, Elyrion S3):
 * game logic data 28 tribe data's tech / unit / improvement overrides and the techs that unlock Fishing and Star Fishing.
 *
 * The tribe starts (tribes::start, tier, unit_cost, unpack) are in core.h next to initial_state, which reads them;
 * tribes::offered (the POLYTOPIA_GEN_WIP_TRIBES opt-in) is host code: pc_api.h pc_tribe_offered. */
#ifndef PC_TRIBES_H
#define PC_TRIBES_H

#include "pc.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PC_POLARIS 16
#define PC_AQUARION 3
#define PC_ELYRION 5
#define PC_CYMANTI 17

/* tribes::wip: a tribe whose increment is still being built (a batch refuses it unless the caller opts in) */
PC_FN int pc_tribes_wip(int32_t tribe) { return tribe == PC_POLARIS || tribe == PC_AQUARION || tribe == PC_ELYRION || tribe == PC_CYMANTI; }

/* tribes:tech_override (get override(tech data, tribe)). Polaris: Fishing -> Frostwork, Sailing -> ice fishing,
 * Navigation -> polar warfare, Aquatism -> Polarism, Aquaculture -> Sledding. Aquarion: Chivalry -> Spearing, Sailing ->
 * marine life, Ramming -> Aquaculture, Fishing -> aqua fishing, Roads -> Waterways. Elyrion: Hunting -> forest magic,
 * Whaling -> water magic. Cymanti: Construction -> Recycling, Chivalry -> Shock, Meditation -> Rituals, Philosophy ->
 * Mysticism. Other tribes: none. */
PC_FN int pc_tribes_tech_override(int32_t tribe, int tech) {
    if (tribe == PC_ELYRION) return tech == 15 ? 28 : tech == 11 ? 29 : tech;
    if (tribe == PC_CYMANTI) return tech == 9 ? 36 : tech == 3 ? 35 : tech == 21 ? 49 : tech == 22 ? 50 : tech;
    if (tribe == PC_POLARIS) {
        switch (tech) {
            case 10: return 30;
            case 13: return 42;
            case 14: return 31;
            case 12: return 32;
            case 39: return 41;
            default: return tech;
        }
    }
    if (tribe == PC_AQUARION) {
        switch (tech) {
            case 3: return 26;
            case 13: return 46;
            case 44: return 39;
            case 10: return 47;
            case 4: return 48;
            default: return tech;
        }
    }
    return tech;
}

/* tribes::unit_override. Polaris: archer -> icearcher, giant -> gaami. Aquarion: giant -> crab, warrior ->
 * mermaidwarrior, archer -> mermaidarcher, swordsman -> mermaidswordsman, rider -> amphibian, cloak -> mermaidcloak,
 * dagger -> mermaiddagger, defender -> mermaiddefender, bombership -> jelly, rammership -> shark, mindbender -> siren,
 * catapult -> aquapult. Elyrion: giant -> dragonegg. Cymanti: rider -> hexapod, giant -> centipede, knight -> doomux,
 * archer -> phychi, swordsman -> mantis, catapult -> exida, defender -> kiton, mindbender -> shaman, cloak -> moth,
 * dagger and pirate -> bugegg, rammership -> boomchi, scoutship -> raychi, bombership -> island. */
PC_FN int32_t pc_tribes_unit_override(int32_t tribe, int32_t unit) {
    if (tribe == PC_POLARIS) {
        if (unit == 9) return 25;
        if (unit == 12) return 27;
        return unit;
    }
    if (tribe == PC_ELYRION) return unit == 12 ? 17 : unit;
    if (tribe == PC_CYMANTI) {
        switch (unit) {
            case 3: return 28;
            case 12: return 33;
            case 4: return 29;
            case 9: return 30;
            case 11: return 59;
            case 8: return 32;
            case 5: return 31;
            case 10: return 36;
            case 38: return 61;
            case 37: return 60;
            case 40: return 60;
            case 44: return 56;
            case 42: return 35;
            case 41: return 57;
            default: return unit;
        }
    }
    if (tribe == PC_AQUARION) {
        switch (unit) {
            case 12: return 26;
            case 2: return 46;
            case 9: return 47;
            case 11: return 48;
            case 3: return 20;
            case 38: return 50;
            case 37: return 51;
            case 5: return 49;
            case 41: return 52;
            case 44: return 53;
            case 10: return 54;
            case 8: return 55;
            default: return unit;
        }
    }
    return unit;
}

/* tribes::structure_override (Polaris: market -> icebank; Aquarion: burnforest -> fertilize; Elyrion: clearforest ->
 * sanctuary, burnforest -> none; Cymanti: market -> clathrus, farm -> fungi, port -> algae, road -> mycelium, bridge ->
 * none). None is 0, the cost-less None structure, never buildable. */
PC_FN int pc_tribes_structure_override(int32_t tribe, int st) {
    if (tribe == PC_CYMANTI) return st == 50 ? 41 : st == 5 ? 37 : st == 8 ? 38 : st == 3 ? 39 : st == 48 ? 0 : st;
    if (tribe == PC_POLARIS && st == 50) return 34;
    if (tribe == PC_AQUARION && st == 11) return 53;
    if (tribe == PC_ELYRION) return st == 10 ? 32 : st == 11 ? 0 : st;
    return st;
}

/* tribes:forest_slot_ok: forest family slot k (0 burn forest, 1 clear forest, 2 grow forest) exists for the tribe;
 * Elyrion's improvement overrides remove slots 0 and 1 */
PC_FN int pc_tribes_forest_slot_ok(int32_t tribe, int k) { return !(tribe == PC_ELYRION && k < 2); }

/* tribes:tech_units: tech data unit unlocks of a special tribe's own techs, as (tech, unit) pairs. Polaris: Frostwork:
 * Mooni, Sledding: BattleSled, polar warfare: IceFortress (POLARIS_TECH_UNITS). Aquarion: Spearing: Tridention,
 * marine life: Shark (AQUARION_TECH_UNITS). Cymanti: Rituals: Shaman, Shock: Doomux (CYMANTI_TECH_UNITS). Returns the
 * pair count (0 for the regular tribes). */
PC_FN int pc_tribes_tech_units_n(int32_t tribe) {
    return tribe == PC_POLARIS ? 3 : tribe == PC_AQUARION || tribe == PC_CYMANTI ? 2 : 0;
}
PC_FN void pc_tribes_tech_unit(int32_t tribe, int k, int *tech, int32_t *unit) {
    static const int32_t polaris[3][2] = {{30, 22}, {41, 23}, {31, 24}};
    static const int32_t aquarion[2][2] = {{26, 21}, {46, 53}};
    static const int32_t cymanti[2][2] = {{49, 36}, {35, 29}};
    const int32_t *e = tribe == PC_POLARIS ? polaris[k] : tribe == PC_CYMANTI ? cymanti[k] : aquarion[k];
    *tech = (int)e[0];
    *unit = e[1];
}

/* tribes:can_embark: can player embark, the Port through Fishing (10); never for Cymanti (port -> algae) */
PC_FN int pc_tribes_can_embark(int32_t tribe, const uint8_t *tech) { return tech[10] && tribe != PC_CYMANTI; }

/* tribes:literacy: get tech price's Literacy, Philosophy or (Cymanti) Mysticism */
PC_FN int pc_tribes_literacy(const uint8_t *tech) { return tech[22] || tech[50]; }

/* tribes::resource_build: the tile's resource building and its tech (Farm on crop, never for Cymanti; Mine on metal;
 * Fungi on Spores, Cymanti only; enchant animal on game); res_build 0 = none */
PC_FN void pc_tribes_resource_build(int32_t tribe, int32_t resource, int *res_tech, int *res_build) {
    *res_tech = 0;
    *res_build = 0;
    if (resource == 2 && tribe != PC_CYMANTI) { *res_tech = 8; *res_build = 5; }
    else if (resource == 5) { *res_tech = 23; *res_build = 21; }
    else if (resource == 7 && tribe == PC_CYMANTI) { *res_tech = 8; *res_build = 37; }
    else if (resource == 1) { *res_tech = 28; *res_build = 30; }
    else if (resource == 9) { *res_tech = 39; *res_build = 49; }
}

/* cymanti:resource_visible: is resource visible to player for the player's tribe and techs (the resource's
 * improvement unlocked or unlockable), as the observation's hides (defined below the starfishing helpers) */
PC_FN int pc_cymanti_resource_visible(int32_t tribe, const uint8_t *tech, int32_t resource);

/* tribes:fishing_tech: the tech that unlocks the Fishing improvement (Polaris: ice fishing, Aquarion: aqua fishing) */
PC_FN int pc_tribes_fishing_tech(int32_t tribe) { return tribe == PC_POLARIS ? 42 : tribe == PC_AQUARION ? 47 : 10; }

/* tribes:starfishing_tech (Polaris: polar warfare) and its parent (starfish show once it is unlockable; Aquarion:
 * Navigation under marine life) */
PC_FN int pc_tribes_starfishing_tech(int32_t tribe) { return tribe == PC_POLARIS ? 31 : 14; }
PC_FN int pc_tribes_starfishing_parent(int32_t tribe) { return tribe == PC_POLARIS ? 41 : tribe == PC_AQUARION ? 46 : 13; }

PC_FN int pc_cymanti_resource_visible(int32_t tribe, const uint8_t *tech, int32_t resource) {
    int cymanti = tribe == PC_CYMANTI;
    switch (resource) {
    case 0: return 0;
    case 2: return !cymanti && (tech[6] || tech[8] || tech[9]);
    case 5: return tech[20] || tech[23] || tech[24];
    case 7: return cymanti;
    case 8: return tech[pc_tribes_starfishing_parent(tribe)] || tech[pc_tribes_starfishing_tech(tribe)];
    case 9: return tech[39];
    default: return 1;
    }
}

#ifdef __cplusplus
}
#endif
#endif /* PC_TRIBES_H */
