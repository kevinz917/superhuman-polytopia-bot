//! Acting-seat public candidate projection, as used by the site and by T3's evaluation.
//! Own units/cities/territory/relations and own history are known facts (including own action bookkeeping).
//! Foreign facts are restricted to visible tiles, visible unit sprites/effects, met tribes and public relations.
//! Unseen tiles/units, foreign cities' private state, chain coordinates, economy, maps and seeds are canonicalized.
//! Unknown terrain is 0 (the proposal convention); this is a candidate set, not a physical legality oracle.
//! `masks(public_state)` must never be used to validate actions against the real game.

use crate::state::{State, MAXP};

pub fn public_state(s: &State, viewer: i32) -> State {
    let n = s.n;
    let np = s.np;
    let p = (viewer - 1).clamp(0, np as i32 - 1) as usize;
    let mut o = s.clone();
    let visible: Vec<bool> = (0..n).map(|i| s.tiles.explored[p][i] || !s.fog).collect();

    // ---- tiles: unexplored tiles are blank
    {
        let t = &mut o.tiles;
        for i in 0..n {
            if visible[i] {
                continue;
            }
            t.terrain[i] = 0;
            t.owner[i] = 0;
            t.capital[i] = 0;
            t.city[i] = -1;
            t.road[i] = false;
            t.structure[i] = 0;
            t.resource[i] = 0;
            t.structure_level[i] = 0;
            t.structure_turn[i] = 0;
            t.route[i] = false;
        }
        // other seats' maps are theirs
        for k in 0..MAXP {
            if k != p {
                t.explored[k].fill(false);
            }
        }
        // structure_turn is not shown for foreign tiles either
        for i in 0..n {
            if visible[i] && t.owner[i] != viewer {
                t.structure_turn[i] = 0;
            }
        }
    }

    // ---- units: unseen ones are absent; visible foreign ones show only what their piece shows
    for i in 0..n {
        let owner = s.units.owner[i];
        if owner == viewer {
            continue;
        }
        if owner == 0 || !visible[i] || s.hidden_from(i, viewer) {
            o.units.clear(i);
            continue;
        }
        let u = &mut o.units;
        u.moved[i] = false;
        u.attacked[i] = false;
        u.kills[i] = 0;
        u.poison[i] = false;
        u.frozen[i] = false;
        u.boosted[i] = false;
        u.home[i] = 0;
        u.created[i] = 0;
        u.converted[i] = false;
        u.attacks_performed[i] = 0;
        u.prev_x[i] = 0;
        u.prev_y[i] = 0;
        u.last_attack[i] = 0;
        u.passenger[i] = 0;
        u.move_lock[i] = false;
    }

    // ---- cities: foreign cities show their tile only
    {
        let c = &mut o.cities;
        for i in 0..n {
            let owner = s.cities.owner[i];
            if owner == viewer {
                continue;
            }
            if owner == 0 || !visible[i] {
                c.owner[i] = 0;
                c.level[i] = 0;
                c.population[i] = 0;
                c.riot[i] = false;
            }
            c.progress[i] = 0;
            c.production[i] = 0;
            c.border[i] = 0;
            c.connected[i] = false;
            c.order[i] = 0;
            c.infiltrated[i] = 0;
            c.reward_seq[i] = 0;
            for k in 0..9 {
                c.rewards[i * 9 + k] = 0;
            }
            let keep = c.owner[i] != 0;
            for j in 0..n {
                let cell = &mut c.territory[i * n + j];
                *cell = keep && *cell && visible[j];
            }
        }
    }

    // ---- players: the viewer's own facts and its relations; nothing private of anyone else
    {
        let pl = &mut o.players;
        for k in 0..MAXP {
            if k == p {
                continue;
            }
            pl.killed_turn[k] = if np > 2 && !s.alive(k) { 1 } else { 0 };
            pl.resigned_turn[k] = 0;
            pl.killer[k] = 0;
            pl.out_order[k] = 0;
            pl.known[k][p] = false;
            pl.peace[k][p] = s.players.peace[p][k];
            // Our own already-sent offer is remembered public diplomatic history.
            pl.truce_breaker[k][p] = 0;
            pl.hostile_until[k][p] = 0;
            pl.peace_broken[k][p] = 0;
            pl.stars[k] = 0;
            pl.score[k] = 0;
            pl.tech[k] = [false; 128];
            pl.tech_turn[k] = [0; 128];
            pl.kills[k] = 0;
            pl.casualties[k] = 0;
            pl.pacifist[k] = 0;
            pl.attacked[k] = false;
            pl.built[k] = [false; 128];
            pl.monuments[k] = [false; 128];
            // relations between two other seats are theirs (four-seat games)
            for q in 0..MAXP {
                if q == p {
                    continue;
                }
                pl.known[k][q] = false;
                pl.peace[k][q] = false;
                pl.embassy[k][q] = false;
                pl.peace_offer[k][q] = false;
                pl.truce_breaker[k][q] = 0;
                pl.hostile_until[k][q] = 0;
                pl.peace_broken[k][q] = 0;
            }
            // another seat's tribe shows once the viewer has met it (as in the packed view); before that, the default
            if !s.players.known[p][k] {
                pl.tribe[k] = crate::tribes::BARDUR;
            }
        }
    }

    // ---- other seats' memories and the random state
    for k in 0..MAXP {
        if k != p {
            o.history[k].clear();
        }
    }
    // T3 additions and stricter packed-view boundary: neither hidden climate effects nor raw chain coordinates
    // are public. Foreign cities expose a city tile/level, not their progress/territory partition.
    for i in 0..n {
        let city = s.tiles.city[i];
        o.tiles.city[i] = if city >= 0 && (city as usize) < n && s.cities.owner[city as usize] == viewer { city } else { -1 };
        o.tiles.explored[p][i] = visible[i];
    }
    for i in 0..n {
        if !visible[i] {
            o.tiles.polar[i] = false;
            o.tiles.flooded[i] = false;
            o.tiles.algae[i] = false;
            o.tiles.decomposing[i] = false;
        }
        o.tiles.continent[i] = 0;
        o.tiles.route[i] = false;
        if visible[i] {
            let r = o.tiles.resource[i];
            let tech = &s.players.tech[p];
            let tribe = s.players.tribe[p];
            let cymanti = tribe == crate::tribes::CYMANTI;
            if (r == 2 && (cymanti || !(tech[6] || tech[8] || tech[9])))
                || (r == 5 && !(tech[20] || tech[23] || tech[24]))
                || (r == 7 && !cymanti)
                || (r == 8 && !(tech[crate::tribes::starfishing_parent(tribe)] || tech[crate::tribes::starfishing_tech(tribe)]))
                || (r == 9 && !tech[39]) { o.tiles.resource[i] = 0; }
        }
        if s.cities.owner[i] != viewer {
            o.cities.owner[i] = if visible[i] && s.tiles.structure[i] == 1 { s.tiles.owner[i] } else { 0 };
            o.cities.population[i] = 0;
            o.cities.riot[i] = false;
            o.cities.territory[i*n..(i+1)*n].fill(false);
            o.cities.level[i] = if visible[i] { s.tiles.structure_level[i] } else { 0 };
        }
        if s.units.owner[i] != viewer {
            o.units.double_ready[i] = false;
            o.units.leader[i] = 0;
            o.units.follower[i] = 0;
            if o.units.owner[i] != 0 {
                o.units.invisible[i] = false;
                o.units.health[i] = o.units.health[i].trunc();
                // Creation age is not revealed by seeing a foreign Grow sprite.
                o.units.created[i] = s.turn;
            }
        }
    }
    for k in 0..MAXP { if k != p { o.players.converts[k] = 0; } }
    o.metrics = crate::telemetry::Metrics { values: [[0; crate::telemetry::WIDTH]; MAXP] };
    o.algae_dirty = false;
    o.monuments_settled = false;
    o.seed = 0;
    o.initial_seed = 0;
    o
}


/// One public proposal; physical validation always ignores public cache bits. A confirmed forced end turn is status5.
pub fn step(s: &mut State, a: crate::core::Action, version: i32, r: &crate::rules::Rules, g: &crate::geom::Geom,
            max_actions: i32, scratch: &mut crate::core::Scratch, legal: Option<&[u8]>, enabled: bool) -> i32 {
    let status = crate::core::step_with(s, a, version, r, g, max_actions, scratch, if enabled { None } else { legal });
    if enabled && status == crate::core::INVALID_ACTION {
        let end = crate::core::decode(s, 15, 0, 0);
        let ended = crate::core::step_with(s, end, s.state_version, r, g, max_actions, scratch, None);
        if ended == crate::core::OK { 5 } else { ended }
    } else { status }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_foreign_creation_turn_never_enters_public_projection() {
        let mut s = State::empty_players(11, 2);
        s.turn = 5;
        s.tiles.explored[0][5] = true;
        s.units.owner[5] = 2;
        s.units.kind[5] = 17;
        s.units.created[5] = 0;
        let a = public_state(&s, 1);
        s.units.created[5] = 4;
        let b = public_state(&s, 1);
        assert_eq!(a.units.created[5], 5);
        assert_eq!(a.units.created[5], b.units.created[5]);
    }

    #[test]
    fn diagnostic_counters_never_enter_observation_or_public_projection() {
        let mut s = State::empty_players(11, 2);
        let geom = crate::geom::Geom::new(11);
        let width = crate::kernels::view_width_np(s.n, 2);
        let mut before = vec![0; width];
        let mut after = vec![0; width];
        crate::kernels::observe(&s, 1, &geom, &mut before);
        for row in &mut s.metrics.values { row.fill(123456); }
        crate::kernels::observe(&s, 1, &geom, &mut after);
        assert_eq!(before, after);
        assert!(public_state(&s, 1).metrics.values.iter().flatten().all(|&x| x == 0));
    }

    #[test]
    fn own_rulership_is_distinct_from_overlapping_territory() {
        let mut s = State::empty_players(11, 2);
        s.cities.owner[10] = 1;
        s.cities.owner[20] = 1;
        s.cities.territory[10 * s.n + 12] = true;
        s.cities.territory[20 * s.n + 12] = true;
        s.tiles.city[12] = 10;
        let public = public_state(&s, 1);
        assert_eq!(public.tiles.city[12], 10);
        assert!(public.cities.territory[20 * s.n + 12]);
    }

    #[test]
    fn hidden_special_effects_and_foreign_riot_are_not_candidates_input() {
        let mut s = State::empty_players(11, 2);
        s.tiles.algae[5] = true;
        s.tiles.flooded[5] = true;
        s.tiles.polar[5] = true;
        s.tiles.decomposing[5] = true;
        s.tiles.explored[0][6] = true;
        s.tiles.structure[6] = 1;
        s.tiles.owner[6] = 2;
        s.cities.owner[6] = 2;
        s.cities.riot[6] = true;
        let public = public_state(&s, 1);
        assert!(!public.tiles.algae[5] && !public.tiles.flooded[5] && !public.tiles.polar[5] && !public.tiles.decomposing[5]);
        assert_eq!(public.cities.owner[6], 2);
        assert!(!public.cities.riot[6]);
    }

    #[test]
    fn foreign_chain_coordinates_and_technology_hidden_resources_are_removed() {
        let mut s = State::empty_players(11, 2);
        s.tiles.explored[0][5] = true;
        s.units.owner[5] = 2;
        s.units.kind[5] = 45;
        s.units.leader[5] = 99;
        s.units.follower[5] = 89;
        s.tiles.resource[5] = 2;
        let public = public_state(&s, 1);
        assert_eq!((public.units.leader[5], public.units.follower[5]), (0, 0));
        assert_eq!(public.tiles.resource[5], 0);
    }
}
