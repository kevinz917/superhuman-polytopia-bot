/* diplomacy.h - line-by-line port of engine-rs/src/diplomacy.rs.
 *
 * Diplomacy between every pair of seats, two seats or four (the real game's PlayerDiplomacyExtensions,
 * CommandUtils.GetDiplomacyCommandsForOpponent and the diplomacy commands; docs/rules-parity-2-actions.md).
 *
 * The diplomatic action family has one block of BLOCK entries per other seat, addressed by turn-order offset from
 * the mover: block b is the seat b + 1 places after the mover (block_seat). A two-seat game only ever uses block 0,
 * the sole opponent, so its entries keep the earlier one-opponent layout. Within a block: 0 break peace,
 * 1 decline / 2 accept a peace request, 3 embassy, 4 offer peace, 5 destroy embassy. */
#ifndef PC_DIPLOMACY_H
#define PC_DIPLOMACY_H

#include "pc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* provided by economy.h */
PC_FN void pc_economy_discover(PcState *s, const uint8_t *reveal, int32_t player);
/* provided by military.h (killer: -1 = None) */
PC_FN void pc_military_remove(PcState *s, int target, const PcRules *r, int killer, uint8_t casualty);
/* provided by structures.h */
PC_FN void pc_structures_update_routes(PcState *s, const PcRules *r, PcStructScratch *sc);
PC_FN void pc_structures_connections(PcState *s, int32_t owner, const PcGeom *g, uint8_t native);
/* provided by core.h */
PC_FN void pc_core_meet_pair(PcState *s, int p, int other);

/* Entries per opponent block of the diplomatic family. */
#define PC_DIPLOMACY_BLOCK 6
/* Blocks in the family (one per possible other seat). */
#define PC_DIPLOMACY_BLOCKS (PC_MAXP - 1)
/* Action parameter (ability) of each entry of a block (diplomacy::ABILITIES). */
#define PC_DIPLOMACY_ABILITIES_INIT {18, 19, 19, 20, 21, 22}
PC_FN int32_t pc_diplomacy_ability(int k) {
    static const int32_t ABILITIES[PC_DIPLOMACY_BLOCK] = PC_DIPLOMACY_ABILITIES_INIT;
    return ABILITIES[k];
}

PC_FN uint8_t pc_diplomacy_protected(const PcState *s, int32_t owner, int32_t other) {
    int p = pc_pclip(s, owner), q = pc_pclip(s, other);
    return other > 0 && (owner == other || s->players.peace[p][q] || s->players.truce_breaker[p][q] != 0);
}

/* Seat id of block `b`: the seat b + 1 places after the mover in turn order; 0 when the game has no such seat. */
PC_FN int32_t pc_diplomacy_block_seat(const PcState *s, int b) {
    if (b + 1 >= s->np) {
        return 0;
    }
    return pc_rem_euclid(s->current_player - 1 + (int32_t)b + 1, s->np) + 1;
}

/* Block of seat `q` (1-based id, not the mover) relative to the mover. */
PC_FN int pc_diplomacy_seat_block(const PcState *s, int32_t q) {
    return (int)(pc_rem_euclid(q - s->current_player, s->np) - 1);
}

/* Seat `q`'s capital tile (its original capital, still its own city; first match) and whether it has one
 * (PlayerExtensions.GetCurrentCapitalCoordinates / OwnsTheirCapital). Rust (usize, bool) -> (*capital, *has_capital). */
PC_FN void pc_diplomacy_capital_of(const PcState *s, int32_t q, int *capital, uint8_t *has_capital) {
    if (q <= 0) {
        *capital = 0;
        *has_capital = 0;
        return;
    }
    for (int i = 0; i < s->n; i++) {
        if (s->tiles.capital[i] == q && s->cities.owner[i] == q) {
            *capital = i;
            *has_capital = 1;
            return;
        }
    }
    *capital = 0;
    *has_capital = 0;
}

/* The two-seat opponent's capital (block 0's embassy target). */
PC_FN void pc_diplomacy_opponent_capital(const PcState *s, int *capital, uint8_t *has_capital) {
    pc_diplomacy_capital_of(s, pc_diplomacy_block_seat(s, 0), capital, has_capital);
}

/* DiplomacyData.GetEmbassyCostForPlayer (v > 113): 4 + 2 per embassy the player already has. */
PC_FN int32_t pc_diplomacy_embassy_cost(const PcState *s, int p) {
    int32_t count = 0;
    for (int q = 0; q < s->np; q++) {
        if (q != p && s->players.embassy[p][q]) count += 1;
    }
    return 4 + 2 * count;
}

/* The seat whose peace request the mover must answer now, if any (`peace_offer[p][q]`: p was offered peace by q).
 * StartTurnAction.CheckDiplomacy pushes one ReceiveDiplomacyMessageAction per message onto the LIFO action stack,
 * from the last message to the first, and the stack stops while a trigger is pending: the requests are answered
 * one at a time, oldest first. All of them were sent since the mover's previous turn, by seats moving in turn
 * order after it, so the oldest is the nearest such seat in turn order. Requests of seats no longer in the game
 * are dropped (v >= 82).
 * Rust Option<i32>: returns the seat id (1-based), -1 when there is none. */
PC_FN int32_t pc_diplomacy_pending_sender(const PcState *s) {
    int p = pc_p(s);
    for (int b = 0; b < s->np - 1; b++) {
        int32_t q = pc_diplomacy_block_seat(s, b);
        int qi = (int)(q - 1);
        if (s->players.peace_offer[p][qi] && pc_alive(s, qi)) {
            return q;
        }
    }
    return -1;
}

/* The mover has a peace request to answer. */
PC_FN uint8_t pc_diplomacy_peace_pending(const PcState *s) {
    return pc_diplomacy_pending_sender(s) >= 0;
}

/* diplomacy.legal given capital_of(s, source). */
PC_FN uint8_t pc_diplomacy_legal_with(const PcState *s, int32_t source, int32_t target, int32_t ability, int capital,
                                      uint8_t has_capital) {
    int32_t np = s->np;
    if (source < 1 || source > np || source == s->current_player) {
        return 0;
    }
    /* a pending peace request is a command trigger: it must be answered before anything else
     * (PeaceRequestResponseCommand; StartTurnAction.CheckDiplomacy makes it pending on the receiver's turn) */
    {
        int32_t sender = pc_diplomacy_pending_sender(s);
        if (sender >= 0) {
            return source == sender && ability == 19 && (target == 0 || target == 1);
        }
    }
    int p = pc_p(s);
    int q = (int)(source - 1);
    const PcPlayers *pl = &s->players;
    /* GetDiplomacyCommandsForOpponent: only known players still in the game */
    uint8_t eligible = pc_alive(s, q) && pl->known[p][q];
    return eligible
        /* BreakPeaceCommand.IsValid */
        && ((ability == 18 && pl->peace[p][q])
            /* EstablishEmbassyCommand.IsValid (v >= 80): no war check; DiplomacyData.GetEmbassyCostForPlayer */
            || (ability == 20 && pl->tech[p][38] && !pl->embassy[p][q] && has_capital && target == (int32_t)capital
                && pl->stars[p] >= pc_diplomacy_embassy_cost(s, p))
            /* PeaceTreatyCommand.IsValid: no cool-down check; no request of ours already waiting at theirs */
            || (ability == 21 && pl->tech[p][7] && !pl->peace[p][q] && !pl->peace_offer[q][p] && !pl->peace_offer[p][q])
            /* DestroyEmbassyCommand.IsValid: HasEmbassyWith */
            || (ability == 22 && pl->embassy[p][q]));
}

/* diplomacy.legal for Action(3, source = the other seat's id, target, ability). */
PC_FN uint8_t pc_diplomacy_legal(const PcState *s, int32_t source, int32_t target, int32_t ability) {
    int capital;
    uint8_t has_capital;
    pc_diplomacy_capital_of(s, source, &capital, &has_capital);
    return pc_diplomacy_legal_with(s, source, target, ability, capital, has_capital);
}

/* Seat `p` is at war with seat `q` (PlayerDiplomacyExtensions.HasWarWith): not at peace, and either attacked the
 * other within the last round (from that attack up to, not including, the attacker's next turn). */
PC_FN uint8_t pc_diplomacy_at_war(const PcState *s, int p, int q) {
    int32_t now = s->turn * s->np + (int32_t)pc_p(s);
    const int32_t (*h)[PC_MAXP] = s->players.hostile_until;
    return !s->players.peace[p][q] && (now < h[p][q] || now < h[q][p]);
}

/* PlayerDiplomacyExtensions.ModifyDiplomacyRelationsFromAttack(mover, victim): stamps the mover's attack on
 * `victim` (war until the mover's next turn) and destroys the embassies between the two (DestroyEmbassyAction both
 * ways). Other pairs and pending peace requests are untouched. No-op for neutral or the mover itself. */
PC_FN void pc_diplomacy_hostile(PcState *s, int32_t victim) {
    if (victim <= 0 || victim > s->np || victim == s->current_player) {
        return;
    }
    int p = pc_p(s);
    int q = (int)(victim - 1);
    s->players.embassy[p][q] = 0;
    s->players.embassy[q][p] = 0;
    s->players.hostile_until[p][q] = s->turn * s->np + (int32_t)p + s->np;
}

PC_FN void pc_diplomacy_capital_vision(PcState *s) {
    int n = s->n;
    int p = pc_p(s);
    if (!s->players.tech[p][38]) {
        return; /* discover of an empty reveal changes nothing */
    }
    /* the capitals of every met seat (with two seats: the opponent's, once met) */
    int32_t met[PC_MAXP];
    int met_n = 0;
    for (int q = 0; q < s->np; q++) {
        if (q != p && s->players.known[p][q]) met[met_n++] = (int32_t)q + 1;
    }
    if (met_n == 0) {
        return;
    }
    uint8_t reveal[PC_MAXN];
    for (int t = 0; t < n; t++) {
        uint8_t hit = 0;
        for (int k = 0; k < met_n; k++) {
            if (met[k] == s->tiles.capital[t]) { hit = 1; break; }
        }
        reveal[t] = hit;
    }
    pc_economy_discover(s, reveal, s->current_player);
}

/* StartTurnAction.CheckDiplomacy for seat `owner` (after the turn counter moved on): requests from seats no longer
 * in the game are dropped, and each of the seat's CoolDown relations ends on its own clock: towards a lower-id seat
 * once the break is in an earlier turn, towards a higher-id seat once it is two turns back (State -> Neutral). */
PC_FN void pc_diplomacy_start_turn(PcState *s, int32_t owner) {
    int p = (int)(owner - 1);
    for (int q = 0; q < s->np; q++) {
        if (q == p) {
            continue;
        }
        if (s->players.peace_offer[p][q] && !pc_alive(s, q)) {
            s->players.peace_offer[p][q] = 0;
        }
        if (s->players.truce_breaker[p][q] != 0) {
            int32_t broken = s->players.peace_broken[p][q];
            if ((q < p && broken < s->turn) || (q > p && broken + 1 < s->turn)) {
                s->players.truce_breaker[p][q] = 0;
            }
        }
    }
}

/* Execute a legal diplomatic action of the mover towards seat `source` (1-based id). */
PC_FN void pc_diplomacy_execute(PcState *s, int32_t source, int32_t target, int32_t ability, const PcRules *r,
                                const PcGeom *g, PcStructScratch *sc) {
    int n = s->n;
    int32_t cp = s->current_player;
    int p = pc_p(s);
    int q = (int)(source - 1);
    switch (ability) {
        case 18: {
            /* BreakPeaceAction (v >= 80): both relations CoolDown, both LastPeaceBrokenTurn; the breaker's units in
             * the victim's territory are disbanded, every other unit of the breaker is exhausted */
            s->players.peace[p][q] = 0;
            s->players.peace[q][p] = 0;
            s->players.truce_breaker[p][q] = cp;
            s->players.truce_breaker[q][p] = cp;
            s->players.peace_broken[p][q] = s->turn;
            s->players.peace_broken[q][p] = s->turn;
            /* `doomed` is collected in full before the first removal, like the Rust Vec */
            int doomed[PC_MAXN];
            int doomed_n = 0;
            for (int i = 0; i < n; i++) {
                if (s->units.owner[i] == cp && s->tiles.owner[i] == source) doomed[doomed_n++] = i;
            }
            for (int k = 0; k < doomed_n; k++) {
                pc_military_remove(s, doomed[k], r, -1, 1);
            }
            for (int i = 0; i < n; i++) {
                if (s->units.owner[i] == cp) {
                    s->units.moved[i] = 1;
                    s->units.attacked[i] = 1;
                }
            }
            break;
        }
        case 19: {
            /* PeaceRequestResponseCommand pops the pending trigger; PeaceRequestResponseAction sets both relations to
             * Peace when accepted (which ends a CoolDown) */
            if (target == 1) {
                s->players.peace[p][q] = 1;
                s->players.peace[q][p] = 1;
                s->players.truce_breaker[p][q] = 0;
                s->players.truce_breaker[q][p] = 0;
            }
            s->players.peace_offer[p][q] = 0;
            break;
        }
        case 20: {
            s->players.stars[p] -= pc_diplomacy_embassy_cost(s, p);
            s->players.embassy[p][q] = 1;
            /* EstablishEmbassyCommand.ExecuteDefault: the host meets the builder (MeetAction, star gift included) */
            pc_core_meet_pair(s, q, p);
            int t = (int)pc_clamp(target, 0, (int32_t)n - 1);
            uint8_t reveal[PC_MAXN];
            for (int i = 0; i < n; i++) reveal[i] = pc_d(g, t, i) <= 1;
            pc_economy_discover(s, reveal, cp);
            break;
        }
        case 21: {
            /* PeaceTreatyAction: a PeaceRequest message to the other seat, received at its next turn start */
            s->players.peace_offer[q][p] = 1;
            break;
        }
        default: {
            s->players.embassy[p][q] = 0;
            break;
        }
    }
    /* BreakPeaceCommand and PeaceRequestResponseCommand (accepted only) queue UpdateCityConnectionsAction for
     * [the mover, the other player] under UpdateRoutesAction for each (BreakPeaceCommand.c 200-300,
     * PeaceRequestResponseCommand.c 180-275; last in, first out: routes first). The other diplomatic commands update
     * no connections (an embassy's exploration does, through the deferred update in core::step_with). `source` is the
     * other seat of this pair (any seat count). */
    if (ability == 18 || (ability == 19 && target == 1)) {
        pc_structures_update_routes(s, r, sc);
        pc_structures_connections(s, cp, g, 1);
        pc_structures_connections(s, source, g, 1);
    }
}

#ifdef __cplusplus
}
#endif
#endif /* PC_DIPLOMACY_H */
