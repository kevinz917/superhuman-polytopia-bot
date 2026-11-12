//! The bot's public candidate set (from the project's hidden-information audit).
//!
//! `public_state(s, viewer)` is a copy of `s` that holds only what `viewer` can see; every other fact is a canonical
//! default, never a guess. The policy's legal-candidate mask is computed on it, so the candidates depend only on the
//! viewer's public view: an unseen enemy (on an unexplored tile, or a hidden Cloak) can neither remove a destination
//! through its zone of control nor block a tile. The true state stays the referee: the site checks the one proposed
//! action against it (engine-rs legality and physics are unchanged).
//!
//! Visible means what the policy's own packed view (engine-rs kernels::observe_impl) shows: a tile is visible when
//! the viewer explored it (or the game has no fog); a unit is visible on a visible tile unless it is hidden from the
//! viewer (State::hidden_from). Visible enemy units keep owner, kind, health (whole points, as the view packs it),
//! veteran and direction; their other fields are zero. A foreign city keeps only what its tile shows: that it is a
//! city, its owner, and a level read from the tile; its population, riot, progress, production, border, rewards,
//! territory and every other internal are zero (: a foreign city's riot reached the Cloak-infiltration
//! mask). Resources the viewer's technology cannot see are absent, as in the view. Tile route flags and foreign
//! ruling-city links are not shown and are canonical. Opponents' economy, technology, relations among themselves,
//! private relation clocks, elimination stamps, histories, maps and the random seeds are zeroed.
//!
//! Every State field is covered by the `fields` test below: a perturbation of any foreign or hidden instance that
//! leaves the viewer's packed view unchanged must leave this projection byte-identical; a new State field fails that
//! test until it is classified there.

use polytopia_engine_rs::state::{State, MAXP};
use polytopia_engine_rs::tribes;

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
            t.polar[i] = false;
            t.flooded[i] = false;
            t.algae[i] = false;
            t.decomposing[i] = false;
        }
        // the continent partition (the examining tribe's new continent) is not in the view (public.rs)
        t.continent.fill(0);
        // has route is one shared flag per tile that the view does not show (it only feeds city connections)
        t.route.fill(false);
        // the ruling-city link of a tile the viewer does not own is not shown
        for i in 0..n {
            if t.owner[i] != viewer {
                t.city[i] = -1;
            }
        }
        // resources the viewer's technology does not reveal (kernels:observe_impl, is resource visible to player)
        // (public.rs: tribe-specific rows; the same as before for the 12 regular tribes the site offers)
        let tech = &s.players.tech[p];
        let tribe = s.players.tribe[p];
        let cymanti = tribe == tribes::CYMANTI;
        let (hide2, hide5) = (cymanti || !(tech[6] || tech[8] || tech[9]), !(tech[20] || tech[23] || tech[24]));
        let hide8 = !(tech[tribes::starfishing_parent(tribe)] || tech[tribes::starfishing_tech(tribe)]);
        for i in 0..n {
            let r = t.resource[i];
            if (r == 2 && hide2) || (r == 5 && hide5) || (r == 7 && !cymanti) || (r == 8 && hide8) || (r == 9 && !tech[39]) {
                t.resource[i] = 0;
            }
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
        if owner == 0 {
            o.units.clear(i); // an empty slot is canonical
            continue;
        }
        if !visible[i] || s.hidden_from(i, viewer) {
            o.units.clear(i);
            continue;
        }
        let u = &mut o.units;
        u.health[i] = (s.units.health[i] as i16) as f32; // the view packs health as whole points (toward zero)
        u.invisible[i] = false; // shown only for the viewer's own units; a visible foreign one is not hidden from it
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
        // a foreign unit's double-attack readiness and its chain links are not shown
        u.double_ready[i] = false;
        u.leader[i] = 0;
        u.follower[i] = 0;
    }

    // ---- cities: a foreign city shows only its tile: a city structure (1) on a visible tile owned by another seat,
    // its owner and the tile's structure level. Everything else about it is canonical (: riot).
    {
        let (c, t) = (&mut o.cities, &s.tiles);
        for i in 0..n {
            if s.cities.owner[i] == viewer {
                continue;
            }
            let shown = visible[i] && t.structure[i] == 1 && t.owner[i] != 0 && t.owner[i] != viewer;
            c.owner[i] = if shown { t.owner[i] } else { 0 };
            c.level[i] = if shown { t.structure_level[i] } else { 0 };
            c.population[i] = 0;
            c.riot[i] = false;
            c.progress[i] = 0;
            c.production[i] = 0;
            c.border[i] = 0;
            c.connected[i] = false;
            c.order[i] = 0;
            c.infiltrated[i] = 0;
            c.reward_seq[i] = 0;
            c.rewards[i * 9..i * 9 + 9].fill(0);
            c.territory[i * n..i * n + n].fill(false);
        }
    }

    // ---- players: the viewer's own facts and its relations; nothing private of anyone else
    {
        let pl = &mut o.players;
        for k in 0..MAXP {
            if k == p {
                continue;
            }
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
            // elimination: only whether the seat is still in (the view's alive flags), no stamps or order
            let alive = s.alive(k);
            pl.killed_turn[k] = if alive { 0 } else { 1 };
            pl.resigned_turn[k] = 0;
            pl.killer[k] = 0;
            pl.out_order[k] = 0;
            pl.converts[k] = 0;
            // k's relation to the viewer: the shared facts are the viewer's own row; k's private clocks are not shown.
            // peace_offer[k][p] (the viewer offered k peace) is the viewer's own act and stays; embassy[k][p] is shown.
            pl.known[k][p] = s.players.known[p][k];
            pl.peace[k][p] = s.players.peace[p][k];
            pl.truce_breaker[k][p] = s.players.truce_breaker[p][k];
            pl.peace_broken[k][p] = s.players.peace_broken[p][k];
            pl.hostile_until[k][p] = 0;
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
                pl.tribe[k] = polytopia_engine_rs::tribes::BARDUR;
            }
        }
    }

    // ---- other seats' memories and the random state
    for k in 0..MAXP {
        if k != p {
            o.history[k].clear();
        }
    }
    // per-seat telemetry counters and the transient algae flag (public.rs)
    o.metrics = polytopia_engine_rs::telemetry::Metrics { values: [[0; polytopia_engine_rs::telemetry::WIDTH]; MAXP] };
    o.algae_dirty = false;
    o.seed = 0;
    o.initial_seed = 0;
    o
}

/// Bits of `public_variant`: each sets a fact the viewer cannot see to one extreme, the same way everywhere.
pub const VAR_FOREIGN_TECH: u32 = 1; // every other seat knows every technology (else none)
pub const VAR_FOREIGN_WALLS: u32 = 2; // every foreign city has walls (else none)
pub const VAR_FOREIGN_POISON: u32 = 4; // every visible foreign unit is poisoned (else none)
pub const VAR_FOREIGN_HEALTH_TOP: u32 = 8; // every visible foreign unit's health is at the top of its whole point
pub const VAR_FOREIGN_EXPLORED_ALL: u32 = 16; // every other seat has explored the whole map (else the lower bound)
pub const VAR_ALL: u32 = 31;

/// The viewer's public projection with the unseen facts that decide a forecast set to one extreme (`mask`, the VAR_
/// bits). Every variant is a function of the projection only, so a forecast over all of them is public; the true
/// outcome is one of the outcomes those worlds give for each fact taken separately (r13: an attack forecast must not
/// be falsely safe, r12b-M1: retaliation needs the defender's map to hold the attacker's tile, which
/// `public_state` clears). Without VAR_FOREIGN_EXPLORED_ALL another seat's map is the part the viewer can prove: the
/// tiles of and next to its visible units, and the visible tiles it owns (`invariant` test below).
pub fn public_variant(s: &State, r: &polytopia_engine_rs::rules::Rules, viewer: i32, mask: u32) -> State {
    let mut p = public_state(s, viewer);
    let n = p.n;
    for k in 0..p.np {
        let seat = k as i32 + 1;
        if seat == viewer {
            continue;
        }
        if mask & VAR_FOREIGN_EXPLORED_ALL != 0 {
            p.tiles.explored[k].fill(true);
        } else {
            for i in 0..n {
                if p.units.owner[i] == seat {
                    p.tiles.explored[k][i] = true;
                    let adj: Vec<usize> = adjacent(p.size, i);
                    for j in adj { p.tiles.explored[k][j] = true; }
                }
                if p.tiles.owner[i] == seat {
                    p.tiles.explored[k][i] = true;
                }
            }
        }
        if mask & VAR_FOREIGN_TECH != 0 {
            p.players.tech[k] = [true; 128];
        }
    }
    for i in 0..n {
        let o = p.units.owner[i];
        if o != 0 && o != viewer {
            if mask & VAR_FOREIGN_POISON != 0 { p.units.poison[i] = true; }
            // the view packs whole points toward zero: the true value is below the next whole point and at most full
            if mask & VAR_FOREIGN_HEALTH_TOP != 0 {
                let top = polytopia_engine_rs::kernels::max_health(&p, r, i);
                p.units.health[i] = (p.units.health[i] + 0.999).min(top).max(p.units.health[i]);
            }
        }
        let c = p.cities.owner[i];
        if mask & VAR_FOREIGN_WALLS != 0 && c != 0 && c != viewer {
            p.cities.rewards[i * 9 + 1] = 1;
        }
    }
    p
}

/// r14: the passengers a visible foreign boat may hold, as the view allows. A boat's maximum health is
/// its passenger's (kernels.rs max_health), and the projection clears foreign passengers. For each foreign unit of a
/// carrying kind (not the Juggernaut/Dinghy/Pirate forms, which carry no passenger kind), one representative land kind
/// per distinct maximum health (unit health + 5 for a veteran) at least the boat's shown whole health; the forecasts
/// try each in turn ("lockstep": an outcome depends on one boat's passenger only). Others get [their own passenger].
pub fn passenger_options(p: &State, r: &polytopia_engine_rs::rules::Rules, viewer: i32) -> Vec<(usize, Vec<i32>)> {
    use polytopia_engine_rs::rules::{CARRY, FLOAT};
    let land: Vec<usize> = (1..128).filter(|&k| r.unit_health[k] > 0.0 && !r.skills[k][FLOAT] && !r.skills[k][CARRY] && !matches!(k, 12 | 37 | 38)).collect();
    let mut out = Vec::new();
    for i in 0..p.n {
        let (o, k) = (p.units.owner[i], p.units.kind[i] as usize);
        if o == 0 || o == viewer || !r.skills[k.min(127)][CARRY] || matches!(k, 39 | 40 | 45) { continue; }
        let vet = if p.units.veteran[i] { 5.0 } else { 0.0 };
        let mut seen: Vec<f32> = Vec::new();
        let mut opts = Vec::new();
        for &l in &land {
            let top = r.unit_health[l] + vet;
            if top >= p.units.health[i] && !seen.contains(&top) { seen.push(top); opts.push(l as i32); }
        }
        if !opts.is_empty() { out.push((i, opts)); }
    }
    out
}

/// World `k` of the lockstep passenger enumeration (passenger_options): boat b holds its option k mod its count.
pub fn set_passengers(q: &mut State, opts: &[(usize, Vec<i32>)], k: usize) {
    for (i, o) in opts { q.units.passenger[*i] = o[k % o.len()]; }
}

fn adjacent(size: usize, i: usize) -> Vec<usize> {
    let (x, y) = ((i % size) as i64, (i / size) as i64);
    let mut out = Vec::with_capacity(8);
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (a, b) = (x + dx, y + dy);
            if (dx, dy) != (0, 0) && a >= 0 && b >= 0 && a < size as i64 && b < size as i64 {
                out.push(b as usize * size + a as usize);
            }
        }
    }
    out
}

#[cfg(test)]
mod privacy {
    //! Privacy fixture: natural positions from the site's own bank boards,
    //! played forward by a seeded random player, each mutated in hidden state only. The mover's packed view must be
    //! unchanged (else the mutation was not hidden and is not counted), and then the public candidate mask (and, with
    //! the weights, the bot's decision at a fixed seed) must be identical. Positive control: the old full-state mask
    //! changes under the same mutations.
    //!     GEN_WASM_BANK=<dir with bank-N.json> [GEN_WASM_WEIGHTS=<.bin>] cargo test --release --target aarch64-apple-darwin --lib privacy
    use super::public_state;
    use crate::policy::Policy;
    use crate::search::Rng;
    use crate::view;
    use crate::{net::Net, Engine};
    use polytopia_engine_rs::{core, mapgen, state::State};
    use serde_json::Value;

    const WARRIOR: i32 = 2;
    const CLOAK: i32 = 38;

    struct Rand(u64);
    impl Rand {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    fn visible(s: &State, p: usize, i: usize) -> bool {
        s.tiles.explored[p][i] || !s.fog
    }

    /// The hidden-state mutations, each independent of the others. None: not applicable to this position.
    fn mutate(s: &State, e: &Engine, k: usize, rand: &mut Rand) -> Option<State> {
        let (n, cp) = (s.n, s.current_player);
        let p = s.p();
        let opp = if cp == 1 { 2 } else { 1 };
        let mut m = s.clone();
        let hidden: Vec<usize> = (0..n).filter(|&i| !visible(s, p, i)).collect();
        match k {
            // an unseen enemy warrior on an unexplored tile next to explored land (its ZOC is the audit's leak)
            0 => {
                let frontier: Vec<usize> = hidden.iter().copied()
                    .filter(|&i| s.units.owner[i] == 0 && e.geom.adj[i].iter().any(|&j| visible(s, p, j as usize)))
                    .collect();
                if frontier.is_empty() { return None; }
                for _ in 0..3 {
                    let i = frontier[rand.below(frontier.len())];
                    if m.units.owner[i] != 0 { continue; }
                    m.units.owner[i] = opp;
                    m.units.kind[i] = WARRIOR;
                    m.units.health[i] = 10.0;
                }
            }
            // every unseen enemy unit removed
            1 => {
                let any = hidden.iter().any(|&i| s.units.owner[i] != 0 && s.units.owner[i] != cp);
                if !any { return None; }
                for &i in &hidden { if m.units.owner[i] != 0 && m.units.owner[i] != cp { m.units.clear(i); } }
            }
            // a hidden enemy Cloak on an empty explored land tile two steps from one of the mover's units (within a
            // move, outside the adjacency the view's public "detect" plane reports)
            2 => {
                let spots: Vec<usize> = (0..n).filter(|&i| visible(s, p, i) && s.units.owner[i] == 0 && s.tiles.terrain[i] >= 3
                    && (0..n).any(|j| s.units.owner[j] == cp && e.geom.d(i, j) == 2)
                    && !(0..n).any(|j| s.units.owner[j] == cp && e.geom.d(i, j) <= 1)).collect();
                if spots.is_empty() || s.players.peace[p][(opp - 1) as usize] { return None; }
                let i = spots[rand.below(spots.len())];
                m.units.owner[i] = opp;
                m.units.kind[i] = CLOAK;
                m.units.health[i] = 5.0;
                m.units.invisible[i] = true;
            }
            // unexplored terrain, resources, structures and roads redrawn
            3 => {
                if hidden.is_empty() { return None; }
                for &i in &hidden {
                    m.tiles.terrain[i] = 1 + rand.below(5) as i32;
                    m.tiles.resource[i] = rand.below(3) as i32;
                    m.tiles.structure[i] = if rand.below(4) == 0 { 6 } else { 0 };
                    m.tiles.road[i] = rand.below(2) == 0;
                }
            }
            // the opponent's stars, technology and score
            4 => {
                let q = (opp - 1) as usize;
                m.players.stars[q] += 37;
                m.players.score[q] += 1000;
                for t in m.players.tech[q].iter_mut() { *t = true; }
            }
            // the opponent's map and memory
            5 => {
                let q = (opp - 1) as usize;
                m.tiles.explored[q].fill(true);
                m.history[q].built.fill(true);
            }
            // the random state
            _ => {
                m.seed ^= 0x9e37_79b9;
                m.initial_seed ^= 0x7f4a_7c15;
            }
        }
        Some(m)
    }

    #[test]
    fn hidden_mutations_never_change_the_bot_candidates_or_decision() {
        let Ok(dir) = std::env::var("GEN_WASM_BANK") else { eprintln!("GEN_WASM_BANK not set: skipped"); return };
        let mut policy = std::env::var("GEN_WASM_WEIGHTS").ok()
            .map(|w| Policy::new(Net::parse(&std::fs::read(w).unwrap()).unwrap(), 11));
        let mut rand = Rand(0x2545_f491_4f6c_dd1d);
        let names = ["unseen-warrior", "unseen-removed", "hidden-cloak", "unexplored-tiles", "opponent-economy", "opponent-map", "rng"];
        let (mut applied, mut skipped, mut full_changed, mut decisions) = ([0usize; 7], [0usize; 7], [0usize; 7], 0usize);
        let mut positions = 0;
        for size in [11usize, 14, 16, 18, 20] {
            let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{dir}/bank-{size}.json")).unwrap()).unwrap();
            if let Some(p) = policy.as_mut() { p.resize(size); }
            for row in rows.iter().take(6) {
                let mut e = Engine::new(size);
                let ints = |k: &str| row[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect::<Vec<_>>();
                let tribes = ints("tribes");
                let caps: Vec<usize> = ints("capitals").iter().zip(&tribes).map(|(&c, &t)| c as usize | (t as usize) << 16).collect();
                let mut s = mapgen::initial_state(row["seed"].as_u64().unwrap() as u32, &ints("terrain"), &ints("structure"),
                                                  &ints("resource"), &caps, &e.geom, 200);
                for step in 0..260 {
                    if s.terminated || s.truncated { break; }
                    if step % 4 == 0 {
                        positions += 1;
                        let cp = s.current_player;
                        let base_view = view::packed(&s, cp, &e.geom);
                        let base_public = e.legal(&public_state(&s, cp), true).to_vec();
                        let base_full = e.legal(&s, true).to_vec();
                        let base_decision = policy.as_mut().map(|pol| {
                            let d = pol.act(&mut e, &s, false, &mut Rng::new(991)).unwrap();
                            let j = pol.act_mode(&mut e, &s, 2, &mut Rng::new(991)).unwrap();
                            (d.index, d.logp.to_bits(), j.index, j.logp.to_bits())
                        });
                        if base_decision.is_some() { decisions += 1; }
                        for k in 0..7 {
                            let Some(m) = mutate(&s, &e, k, &mut rand) else { continue };
                            if view::packed(&m, cp, &e.geom) != base_view { skipped[k] += 1; continue; }
                            applied[k] += 1;
                            let public = e.legal(&public_state(&m, cp), true).to_vec();
                            assert!(public == base_public, "{} changed the public candidates: size {size} step {step}", names[k]);
                            if e.legal(&m, true) != &base_full[..] { full_changed[k] += 1; }
                            if let (Some(pol), Some(b)) = (policy.as_mut(), base_decision) {
                                let d = pol.act(&mut e, &m, false, &mut Rng::new(991)).unwrap();
                                let j = pol.act_mode(&mut e, &m, 2, &mut Rng::new(991)).unwrap();
                                assert_eq!((d.index, d.logp.to_bits(), j.index, j.logp.to_bits()), b,
                                           "{} changed the decision (sampled or joint top-1): size {size} step {step}", names[k]);
                            }
                        }
                    }
                    // a seeded random player (both seats) on the true state, resignation excluded
                    let legal: Vec<usize> = e.legal(&s, true).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
                    let a = legal[rand.below(legal.len())];
                    assert_eq!(e.apply(&mut s, a), core::OK);
                }
            }
        }
        for k in 0..7 {
            eprintln!("{:>17}: applied {:>4}, not hidden (skipped) {:>3}, old full-state mask changed {:>3}", names[k], applied[k], skipped[k], full_changed[k]);
        }
        eprintln!("positions {positions}, bot decisions compared {decisions}");
        assert!(applied.iter().all(|&a| a > 0), "every mutation family must apply somewhere: {applied:?}");
        // positive control: the leak this fixes is real in these positions
        assert!(full_changed[0] + full_changed[2] > 0, "the old full-state mask never reacted: the fixture is too weak");
    }

    // ---------------------------------------------------------------------------------------------------------------
    //  and every other field: `public_state` must be a function of the viewer's public view (and its own
    // state). For every field of State, a perturbation of a foreign or hidden instance that leaves the viewer's packed
    // view byte-identical must leave the projection byte-identical (so the candidate mask cannot change either).

    /// Every field of State, in a fixed order (the projection's fingerprint). `fields_cover_the_whole_state` checks
    /// this list against state.rs, so a new field fails until it is classified.
    fn fp(s: &State) -> Vec<u8> {
        let mut b: Vec<u8> = Vec::new();
        macro_rules! v { ($x:expr) => { for y in $x.iter() { b.extend_from_slice(&(*y as i64).to_le_bytes()); } } }
        macro_rules! f { ($x:expr) => { for y in $x.iter() { b.extend_from_slice(&y.to_bits().to_le_bytes()); } } }
        macro_rules! m { ($x:expr) => { for row in $x.iter() { v!(row); } } }
        let (t, u, c, pl) = (&s.tiles, &s.units, &s.cities, &s.players);
        v!(t.terrain); v!(t.owner); v!(t.capital); v!(t.city); v!(t.road); v!(t.structure); v!(t.resource);
        m!(t.explored); v!(t.structure_level); v!(t.structure_turn); v!(t.route);
        v!(t.polar); v!(t.flooded); v!(t.continent); v!(t.algae); v!(t.decomposing);
        v!(u.bubble); v!(u.swift); v!(u.double_ready); v!(u.leader); v!(u.follower); v!(pl.converts);
        m!(s.metrics.values); v!([s.algae_dirty as i64]);
        v!(u.owner); v!(u.kind); f!(u.health); v!(u.veteran); v!(u.moved); v!(u.attacked); v!(u.kills); v!(u.poison);
        v!(u.frozen); v!(u.boosted); v!(u.home); v!(u.created); v!(u.converted); v!(u.attacks_performed); v!(u.prev_x);
        v!(u.prev_y); v!(u.last_attack); v!(u.passenger); v!(u.invisible); v!(u.direction); v!(u.move_lock);
        v!(c.owner); v!(c.level); v!(c.population); v!(c.progress); v!(c.production); v!(c.border); v!(c.rewards);
        v!(c.territory); v!(c.connected); v!(c.order); v!(c.infiltrated); v!(c.riot); v!(c.reward_seq);
        v!(pl.stars); v!(pl.score); m!(pl.tech); m!(pl.tech_turn); m!(pl.known); v!(pl.kills); v!(pl.casualties);
        v!(pl.pacifist); v!(pl.attacked); m!(pl.built); v!(pl.killed_turn); v!(pl.resigned_turn); v!(pl.killer);
        m!(pl.peace); m!(pl.embassy); m!(pl.peace_offer); m!(pl.truce_breaker); m!(pl.hostile_until); m!(pl.peace_broken);
        m!(pl.monuments); v!(pl.out_order); v!(pl.tribe);
        for h in s.history.iter() {
            m!(h.seen_unit); v!(h.prev_code); v!(h.built); v!(h.trained); b.extend_from_slice(&h.enemy_tech.to_le_bytes());
        }
        v!([s.size as i64, s.n as i64, s.np as i64, s.current_player as i64, s.turn as i64, s.version as i64,
            s.state_version as i64, s.actions as i64, s.terminated as i64, s.truncated as i64, s.fog as i64,
            s.reference_max_turns as i64, s.core_supported as i64, s.seed as i64, s.initial_seed as i64,
            s.monuments_settled as i64]);
        b
    }

    /// Tile, unit, city, seat and history fields this test perturbs, and the global ones it does not (with why).
    const PERTURBED: &[&str] = &[
        "Tiles.terrain", "Tiles.owner", "Tiles.capital", "Tiles.city", "Tiles.road", "Tiles.structure", "Tiles.resource",
        "Tiles.explored", "Tiles.structure_level", "Tiles.structure_turn", "Tiles.route",
        "Tiles.polar", "Tiles.flooded", "Tiles.continent", "Tiles.algae", "Tiles.decomposing",
        "Units.owner", "Units.kind", "Units.health", "Units.veteran", "Units.moved", "Units.attacked", "Units.kills",
        "Units.poison", "Units.frozen", "Units.boosted", "Units.home", "Units.created", "Units.converted",
        "Units.attacks_performed", "Units.prev_x", "Units.prev_y", "Units.last_attack", "Units.passenger",
        "Units.invisible", "Units.direction", "Units.move_lock",
        "Units.bubble", "Units.swift", "Units.double_ready", "Units.leader", "Units.follower",
        "Cities.owner", "Cities.level", "Cities.population", "Cities.progress", "Cities.production", "Cities.border",
        "Cities.rewards", "Cities.territory", "Cities.connected", "Cities.order", "Cities.infiltrated", "Cities.riot",
        "Cities.reward_seq",
        "Players.stars", "Players.score", "Players.tech", "Players.tech_turn", "Players.known", "Players.kills",
        "Players.casualties", "Players.pacifist", "Players.attacked", "Players.built", "Players.killed_turn",
        "Players.resigned_turn", "Players.killer", "Players.peace", "Players.embassy", "Players.peace_offer",
        "Players.truce_breaker", "Players.hostile_until", "Players.peace_broken", "Players.monuments",
        "Players.out_order", "Players.tribe", "Players.converts", "State.metrics",
        "History.seen_unit", "History.prev_code", "History.built", "History.trained", "History.enemy_tech",
        "State.seed", "State.initial_seed",
    ];
    /// Global game facts, the same for every seat (board size, whose turn, turn number, the shared action counter and
    /// versions, end flags, fog, the round cap, engine-internal flags). They are not hidden information; aggregates
    /// of the other fields (tiles, units, cities, players, history) are perturbed above.
    const GLOBAL: &[&str] = &[
        "State.size", "State.n", "State.np", "State.tiles", "State.units", "State.cities", "State.players",
        "State.history", "State.current_player", "State.turn", "State.version", "State.state_version", "State.actions",
        "State.terminated", "State.truncated", "State.fog", "State.reference_max_turns", "State.core_supported",
        "State.monuments_settled", "State.algae_dirty",
    ];

    #[test]
    fn fields_cover_the_whole_state() {
        let src = include_str!("../../engine-rs/src/state.rs");
        let mut found = Vec::new();
        let mut current: Option<&str> = None;
        for line in src.lines() {
            if let Some(rest) = line.strip_prefix("pub struct ") {
                current = rest.split_whitespace().next().filter(|n| ["Tiles", "Units", "Cities", "Players", "History", "State"].contains(n));
                continue;
            }
            if line.starts_with('}') { current = None; continue; }
            if let (Some(st), Some(rest)) = (current, line.strip_prefix("    pub ")) {
                if let Some(name) = rest.split(':').next().filter(|n| !n.contains('(') && !n.contains(' ')) {
                    found.push(format!("{st}.{name}"));
                }
            }
        }
        let known: Vec<&str> = PERTURBED.iter().chain(GLOBAL.iter()).copied().collect();
        let missing: Vec<&String> = found.iter().filter(|f| !known.contains(&f.as_str())).collect();
        assert!(missing.is_empty(), "State fields not classified for the privacy projection: {missing:?}");
        assert!(found.len() >= 90, "state.rs parse found only {} fields", found.len());
    }

    /// One perturbation of `field` at instance `i` (tile/unit/city slot) and seat `k`; false if it does not apply.
    fn perturb(m: &mut State, field: &str, i: usize, k: usize, q: usize, r: usize) -> bool {
        let n = m.n;
        // an owner stays foreign: a hidden or foreign slot toggles between empty and the other seat (never the viewer)
        let opp = 3 - m.current_player;
        let (t, u, c, pl) = (&mut m.tiles, &mut m.units, &mut m.cities, &mut m.players);
        macro_rules! int { ($x:expr) => {{ $x = $x.wrapping_add(1 + (r % 3) as i32); true }} }
        macro_rules! flip { ($x:expr) => {{ $x = !$x; true }} }
        match field {
            "Tiles.terrain" => int!(t.terrain[i]), "Tiles.owner" => int!(t.owner[i]), "Tiles.capital" => int!(t.capital[i]),
            "Tiles.city" => int!(t.city[i]), "Tiles.road" => flip!(t.road[i]), "Tiles.structure" => int!(t.structure[i]),
            "Tiles.resource" => { t.resource[i] = [2, 5, 8, 9, 1, 3][r % 6]; true }
            "Tiles.explored" => flip!(t.explored[k][i]), "Tiles.structure_level" => int!(t.structure_level[i]),
            "Tiles.structure_turn" => int!(t.structure_turn[i]), "Tiles.route" => flip!(t.route[i]),
            "Units.owner" => { u.owner[i] = if u.owner[i] == 0 { opp } else { 0 }; true } "Units.kind" => int!(u.kind[i]),
            "Units.health" => { u.health[i] += if r % 2 == 0 { 0.5 } else { 1.0 }; true }
            "Units.veteran" => flip!(u.veteran[i]), "Units.moved" => flip!(u.moved[i]), "Units.attacked" => flip!(u.attacked[i]),
            "Units.kills" => int!(u.kills[i]), "Units.poison" => flip!(u.poison[i]), "Units.frozen" => flip!(u.frozen[i]),
            "Units.boosted" => flip!(u.boosted[i]), "Units.home" => int!(u.home[i]), "Units.created" => int!(u.created[i]),
            "Units.converted" => flip!(u.converted[i]), "Units.attacks_performed" => int!(u.attacks_performed[i]),
            "Units.prev_x" => int!(u.prev_x[i]), "Units.prev_y" => int!(u.prev_y[i]), "Units.last_attack" => int!(u.last_attack[i]),
            "Units.passenger" => int!(u.passenger[i]), "Units.invisible" => flip!(u.invisible[i]),
            "Units.direction" => int!(u.direction[i]), "Units.move_lock" => flip!(u.move_lock[i]),
            "Tiles.polar" => flip!(t.polar[i]), "Tiles.flooded" => flip!(t.flooded[i]), "Tiles.continent" => int!(t.continent[i]),
            "Tiles.algae" => flip!(t.algae[i]), "Tiles.decomposing" => flip!(t.decomposing[i]),
            "Units.bubble" => flip!(u.bubble[i]), "Units.swift" => flip!(u.swift[i]), "Units.double_ready" => flip!(u.double_ready[i]),
            "Units.leader" => int!(u.leader[i]), "Units.follower" => int!(u.follower[i]),
            "Players.converts" => int!(pl.converts[k]),
            "State.metrics" => int!(m.metrics.values[k][r % polytopia_engine_rs::telemetry::WIDTH]),
            "Cities.owner" => { c.owner[i] = if c.owner[i] == 0 { opp } else { 0 }; true } "Cities.level" => int!(c.level[i]), "Cities.population" => int!(c.population[i]),
            "Cities.progress" => int!(c.progress[i]), "Cities.production" => int!(c.production[i]), "Cities.border" => int!(c.border[i]),
            "Cities.rewards" => int!(c.rewards[i * 9 + r % 9]), "Cities.territory" => flip!(c.territory[i * n + (r * 7919) % n]),
            "Cities.connected" => flip!(c.connected[i]), "Cities.order" => int!(c.order[i]), "Cities.infiltrated" => int!(c.infiltrated[i]),
            "Cities.riot" => flip!(c.riot[i]), "Cities.reward_seq" => int!(c.reward_seq[i]),
            "Players.stars" => int!(pl.stars[k]), "Players.score" => int!(pl.score[k]), "Players.tech" => flip!(pl.tech[k][r % 128]),
            "Players.tech_turn" => int!(pl.tech_turn[k][r % 128]), "Players.known" => flip!(pl.known[k][q]),
            "Players.kills" => int!(pl.kills[k]), "Players.casualties" => int!(pl.casualties[k]), "Players.pacifist" => int!(pl.pacifist[k]),
            "Players.attacked" => flip!(pl.attacked[k]), "Players.built" => flip!(pl.built[k][r % 128]),
            "Players.killed_turn" => { pl.killed_turn[k] = if pl.killed_turn[k] == 0 { 0 } else { pl.killed_turn[k] + 1 }; pl.killed_turn[k] != 0 }
            "Players.resigned_turn" => { pl.resigned_turn[k] = if pl.resigned_turn[k] == 0 { 0 } else { pl.resigned_turn[k] + 1 }; pl.resigned_turn[k] != 0 }
            "Players.killer" => int!(pl.killer[k]), "Players.peace" => flip!(pl.peace[k][q]), "Players.embassy" => flip!(pl.embassy[k][q]),
            "Players.peace_offer" => flip!(pl.peace_offer[k][q]), "Players.truce_breaker" => int!(pl.truce_breaker[k][q]),
            "Players.hostile_until" => int!(pl.hostile_until[k][q]), "Players.peace_broken" => int!(pl.peace_broken[k][q]),
            "Players.monuments" => flip!(pl.monuments[k][r % 128]), "Players.out_order" => int!(pl.out_order[k]),
            "Players.tribe" => int!(pl.tribe[k]),
            "History.seen_unit" => flip!(m.history[k].seen_unit[r % 3][i]), "History.prev_code" => int!(m.history[k].prev_code[i]),
            "History.built" => flip!(m.history[k].built[i]), "History.trained" => flip!(m.history[k].trained[i]),
            "History.enemy_tech" => { m.history[k].enemy_tech ^= 1u128 << (r % 128); true }
            "State.seed" => { m.seed ^= 1 << (r % 32); true }
            "State.initial_seed" => { m.initial_seed ^= 1 << (r % 32); true }
            _ => panic!("unhandled field {field}"),
        }
    }

    #[test]
    fn every_hidden_field_leaves_the_projection_unchanged() {
        let Ok(dir) = std::env::var("GEN_WASM_BANK") else { eprintln!("GEN_WASM_BANK not set: skipped"); return };
        let mut rand = Rand(0x9e37_79b9_7f4a_7c15);
        let (mut applied, mut shown, mut failed) = (vec![0usize; PERTURBED.len()], vec![0usize; PERTURBED.len()], Vec::<String>::new());
        let (mut positions, mut true_changed) = (0usize, 0usize);
        for size in [11usize, 14, 16, 18, 20] {
            let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{dir}/bank-{size}.json")).unwrap()).unwrap();
            for row in rows.iter().take(3) {
                let mut e = Engine::new(size);
                let ints = |k: &str| row[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect::<Vec<_>>();
                let tribes = ints("tribes");
                let caps: Vec<usize> = ints("capitals").iter().zip(&tribes).map(|(&c, &t)| c as usize | (t as usize) << 16).collect();
                let mut s = mapgen::initial_state(row["seed"].as_u64().unwrap() as u32, &ints("terrain"), &ints("structure"),
                                                  &ints("resource"), &caps, &e.geom, 200);
                for step in 0..240 {
                    if s.terminated || s.truncated { break; }
                    if step % 12 == 0 {
                        positions += 1;
                        let cp = s.current_player;
                        let p = s.p();
                        let n = s.n;
                        let base_view = view::packed(&s, cp, &e.geom);
                        let base = fp(&public_state(&s, cp));
                        let pick = |v: &Vec<usize>, rand: &mut Rand| if v.is_empty() { None } else { Some(v[rand.below(v.len())]) };
                        let vis = |i: usize| visible(&s, p, i);
                        let hidden: Vec<usize> = (0..n).filter(|&i| !vis(i)).collect();
                        let foreign_tiles: Vec<usize> = (0..n).filter(|&i| vis(i) && s.tiles.owner[i] != cp).collect();
                        let foreign_units: Vec<usize> = (0..n).filter(|&i| s.units.owner[i] != 0 && s.units.owner[i] != cp).collect();
                        let empty_units: Vec<usize> = (0..n).filter(|&i| s.units.owner[i] == 0).collect();
                        let foreign_cities: Vec<usize> = (0..n).filter(|&i| s.cities.owner[i] != 0 && s.cities.owner[i] != cp).collect();
                        let empty_cities: Vec<usize> = (0..n).filter(|&i| s.cities.owner[i] == 0).collect();
                        let others: Vec<usize> = (0..s.np).filter(|&k| k != p).collect();
                        for (fi, field) in PERTURBED.iter().enumerate() {
                            let pools: Vec<&Vec<usize>> = match field.split('.').next().unwrap() {
                                "Tiles" => vec![&hidden, &foreign_tiles],
                                "Units" => vec![&foreign_units, &empty_units],
                                "Cities" => vec![&foreign_cities, &empty_cities],
                                _ => vec![&hidden, &foreign_tiles],
                            };
                            for pool in pools {
                                for _ in 0..2 {
                                    let Some(i) = pick(pool, &mut rand) else { continue };
                                    let k = if *field == "Tiles.explored" || *field == "State.metrics" || field.starts_with("Players") || field.starts_with("History") {
                                        others[rand.below(others.len())]
                                    } else { p };
                                    let q = rand.below(s.np);
                                    // the viewer's own sent peace offer (peace_offer[k][viewer]) is its own act
                                    if *field == "Players.peace_offer" && q == p { continue; }
                                    let mut m = s.clone();
                                    if !perturb(&mut m, field, i, k, q, rand.below(1 << 20)) { continue; }
                                    if view::packed(&m, cp, &e.geom) != base_view { shown[fi] += 1; continue; }
                                    applied[fi] += 1;
                                    if fp(&public_state(&m, cp)) != base && failed.len() < 40 {
                                        failed.push(format!("{field} size {size} step {step} slot {i} seat {k}/{q}"));
                                    }
                                    // positive control: these hidden perturbations reach the true (referee) mask
                                    let truth = e.legal(&s, true).to_vec();
                                    if e.legal(&m, true) != &truth[..] { true_changed += 1; }
                                }
                            }
                        }
                    }
                    let legal: Vec<usize> = e.legal(&s, true).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
                    let a = legal[rand.below(legal.len())];
                    assert_eq!(e.apply(&mut s, a), core::OK);
                }
            }
        }
        for (fi, f) in PERTURBED.iter().enumerate() {
            eprintln!("{f:>26}: hidden perturbations {:>5}, shown in the view (not counted) {:>5}", applied[fi], shown[fi]);
        }
        eprintln!("positions {positions}; hidden perturbations {}; projection differences {}; true mask changed (positive control) {true_changed}",
                  applied.iter().sum::<usize>(), failed.len());
        assert!(failed.is_empty(), "hidden fields reached the projection: {failed:#?}");
        // a seat that is out ends a two-seat game (terminated, public): its elimination stamps never apply live
        let never: Vec<&&str> = PERTURBED.iter().enumerate()
            .filter(|(i, f)| applied[*i] == 0 && !["Players.killed_turn", "Players.resigned_turn"].contains(f)).map(|(_, f)| f).collect();
        assert!(never.is_empty(), "fields never perturbed while hidden: {never:?}");
        assert!(true_changed > 0, "positive control: no hidden perturbation reached the true mask");
    }

    // ---------------------------------------------------------------------------------------------------------------
    // 11 Nov (r12): the whole UI payload a seat receives in a live game must be a function of its public
    // view. Every hidden perturbation (as above: the viewer's packed view byte-identical) must leave each part
    // byte-identical: the seat view (board, HUD), the viewer's legal actions (targets, menus, aria), the attack
    // forecasts and the enemy threat view. `ui_parts` is what the site's worker builds from the engine.

    /// [seat view, legal actions, attack forecasts, enemy threat], as JSON text, for `viewer` (the player to move).
    pub(crate) fn ui_parts(e: &mut Engine, s: &State, viewer: i32) -> [String; 4] {
        crate::ui::parts(e, s, viewer)
    }

    #[test]
    fn every_hidden_field_leaves_the_ui_payload_unchanged() {
        let Ok(dir) = std::env::var("GEN_WASM_BANK") else { eprintln!("GEN_WASM_BANK not set: skipped"); return };
        const PARTS: [&str; 4] = ["view", "legal", "forecast", "threat"];
        let mut rand = Rand(0x5eed_0012_c0ff_ee00);
        let mut applied = vec![0usize; PERTURBED.len()];
        let mut diff = vec![[0usize; 4]; PERTURBED.len()];
        let mut examples: Vec<String> = Vec::new();
        let (mut positions, mut forecasts, mut threats) = (0usize, 0usize, 0usize);
        let mut control = [0usize; 4];
        let sizes: Vec<usize> = std::env::var("UI_SIZES").map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![11, 14, 16, 18, 20]);
        let every: usize = std::env::var("UI_EVERY").map(|v| v.parse().unwrap()).unwrap_or(12);
        for &size in &sizes {
            let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{dir}/bank-{size}.json")).unwrap()).unwrap();
            for row in rows.iter().take(3) {
                let mut e = Engine::new(size);
                let ints = |k: &str| row[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect::<Vec<_>>();
                let tribes = ints("tribes");
                let caps: Vec<usize> = ints("capitals").iter().zip(&tribes).map(|(&c, &t)| c as usize | (t as usize) << 16).collect();
                let mut s = mapgen::initial_state(row["seed"].as_u64().unwrap() as u32, &ints("terrain"), &ints("structure"),
                                                  &ints("resource"), &caps, &e.geom, 200);
                for step in 0..240 {
                    if s.terminated || s.truncated { break; }
                    // Both seats: the mover (its whole turn payload) and the other seat (its view while it waits).
                    for cp in if step % every == 0 { vec![s.current_player, 3 - s.current_player] } else { vec![] } {
                        positions += 1;
                        let p = (cp - 1) as usize;
                        let n = s.n;
                        let base_view = view::packed(&s, cp, &e.geom);
                        let base = ui_parts(&mut e, &s, cp);
                        let base_true = crate::ui::parts_from(&mut e, &s, cp, false);
                        if base[2] != "[]" { forecasts += 1; }
                        if base[3] != "[]" { threats += 1; }
                        let pick = |v: &Vec<usize>, rand: &mut Rand| if v.is_empty() { None } else { Some(v[rand.below(v.len())]) };
                        let vis = |i: usize| visible(&s, p, i);
                        let hidden: Vec<usize> = (0..n).filter(|&i| !vis(i)).collect();
                        let foreign_tiles: Vec<usize> = (0..n).filter(|&i| vis(i) && s.tiles.owner[i] != cp).collect();
                        let foreign_units: Vec<usize> = (0..n).filter(|&i| s.units.owner[i] != 0 && s.units.owner[i] != cp).collect();
                        let empty_units: Vec<usize> = (0..n).filter(|&i| s.units.owner[i] == 0).collect();
                        let foreign_cities: Vec<usize> = (0..n).filter(|&i| s.cities.owner[i] != 0 && s.cities.owner[i] != cp).collect();
                        let empty_cities: Vec<usize> = (0..n).filter(|&i| s.cities.owner[i] == 0).collect();
                        let others: Vec<usize> = (0..s.np).filter(|&k| k != p).collect();
                        for (fi, field) in PERTURBED.iter().enumerate() {
                            let pools: Vec<&Vec<usize>> = match field.split('.').next().unwrap() {
                                "Tiles" => vec![&hidden, &foreign_tiles],
                                "Units" => vec![&foreign_units, &empty_units],
                                "Cities" => vec![&foreign_cities, &empty_cities],
                                _ => vec![&hidden, &foreign_tiles],
                            };
                            for pool in pools {
                                for _ in 0..2 {
                                    let Some(i) = pick(pool, &mut rand) else { continue };
                                    let k = if *field == "Tiles.explored" || *field == "State.metrics" || field.starts_with("Players") || field.starts_with("History") {
                                        others[rand.below(others.len())]
                                    } else { p };
                                    let q = rand.below(s.np);
                                    if *field == "Players.peace_offer" && q == p { continue; }
                                    let mut m = s.clone();
                                    // perturb keeps a foreign owner foreign to the player to move: make that the viewer
                                    m.current_player = cp;
                                    let ok = perturb(&mut m, field, i, k, q, rand.below(1 << 20));
                                    m.current_player = s.current_player;
                                    if !ok { continue; }
                                    if view::packed(&m, cp, &e.geom) != base_view { continue; }
                                    applied[fi] += 1;
                                    let got = ui_parts(&mut e, &m, cp);
                                    // positive control: the same payload from the true state (site r12 5a017773)
                                    let truth = crate::ui::parts_from(&mut e, &m, cp, false);
                                    for j in 1..4 { if truth[j] != base_true[j] { control[j] += 1; } }
                                    for j in 0..4 {
                                        if got[j] != base[j] {
                                            diff[fi][j] += 1;
                                            if examples.len() < 60 {
                                                examples.push(format!("{field} -> {} (size {size} step {step} slot {i} seat {k}/{q})", PARTS[j]));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let legal: Vec<usize> = e.legal(&s, true).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
                    let a = legal[rand.below(legal.len())];
                    assert_eq!(e.apply(&mut s, a), core::OK);
                }
            }
        }
        let mut total = [0usize; 4];
        for (fi, f) in PERTURBED.iter().enumerate() {
            for j in 0..4 { total[j] += diff[fi][j]; }
            if diff[fi].iter().any(|&d| d > 0) || std::env::var("UI_VERBOSE").is_ok() {
                eprintln!("{f:>26}: hidden {:>5}  view {:>4} legal {:>4} forecast {:>4} threat {:>4}", applied[fi], diff[fi][0], diff[fi][1], diff[fi][2], diff[fi][3]);
            }
        }
        eprintln!("ui payload: positions {positions} (with forecasts {forecasts}, with threats {threats}); hidden perturbations {}; differences view {} legal {} forecast {} threat {}",
                  applied.iter().sum::<usize>(), total[0], total[1], total[2], total[3]);
        eprintln!("positive control (legal, forecast, threat from the true state): differences legal {} forecast {} threat {}", control[1], control[2], control[3]);
        for x in &examples { eprintln!("  {x}"); }
        assert_eq!(total, [0; 4], "hidden fields reached the UI payload");
        assert!(control[1] > 0 && control[2] > 0 && control[3] > 0, "positive control: {control:?}");
        let never: Vec<&&str> = PERTURBED.iter().enumerate()
            .filter(|(i, f)| applied[*i] == 0 && !["Players.killed_turn", "Players.resigned_turn"].contains(f)).map(|(_, f)| f).collect();
        assert!(never.is_empty(), "fields never perturbed while hidden: {never:?}");
        assert!(forecasts > 0 && threats > 0, "premise: positions with forecasts {forecasts}, threats {threats}");
    }

    /// r13 chain (web tests/endpoint-privacy.test.ts swaps a live game for `game_public`): on natural positions, both
    /// seats, the projection is idempotent and the seat view and every UI payload part are the same on the projection
    /// as on the true state. With the every-field tests, the worker's endpoints then cannot see hidden state.
    #[test]
    fn the_projection_is_idempotent_and_carries_the_whole_seat_payload() {
        let Ok(dir) = std::env::var("GEN_WASM_BANK") else { eprintln!("GEN_WASM_BANK not set: skipped"); return };
        let mut rand = Rand(0x0b5e_55ed_1337_0013);
        let mut checked = 0usize;
        for size in [11usize, 14, 16, 18, 20] {
            let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{dir}/bank-{size}.json")).unwrap()).unwrap();
            for row in rows.iter().take(3) {
                let mut e = Engine::new(size);
                let ints = |k: &str| row[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect::<Vec<_>>();
                let tribes = ints("tribes");
                let caps: Vec<usize> = ints("capitals").iter().zip(&tribes).map(|(&c, &t)| c as usize | (t as usize) << 16).collect();
                let mut s = mapgen::initial_state(row["seed"].as_u64().unwrap() as u32, &ints("terrain"), &ints("structure"),
                                                  &ints("resource"), &caps, &e.geom, 200);
                for step in 0..240 {
                    if s.terminated || s.truncated { break; }
                    if step % 6 == 0 {
                        for v in [1, 2] {
                            let p = public_state(&s, v);
                            assert_eq!(fp(&public_state(&p, v)), fp(&p), "idempotent: size {size} step {step} seat {v}");
                            assert_eq!(crate::ui::parts(&mut e, &p, v), crate::ui::parts(&mut e, &s, v), "payload on the projection: size {size} step {step} seat {v}");
                            checked += 1;
                        }
                    }
                    let legal: Vec<usize> = e.legal(&s, true).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
                    let a = legal[rand.below(legal.len())];
                    assert_eq!(e.apply(&mut s, a), core::OK);
                }
            }
        }
        eprintln!("projection chain: {checked} seat positions, idempotent, seat payload identical");
        assert!(checked > 500);
    }

    /// an own Cloak next to a visible enemy city; the city's riot flag is hidden. The public candidates
    /// (and the bot's decision) must not change with it; positive control: the true mask does (Cloak infiltration).
    #[test]
    fn foreign_city_riot_does_not_reach_the_candidates() {
        let Ok(dir) = std::env::var("GEN_WASM_BANK") else { eprintln!("GEN_WASM_BANK not set: skipped"); return };
        let mut policy = std::env::var("GEN_WASM_WEIGHTS").ok()
            .map(|w| Policy::new(Net::parse(&std::fs::read(w).unwrap()).unwrap(), 11));
        let (mut cases, mut true_changed, mut decisions) = (0usize, 0usize, 0usize);
        let mut rand = Rand(0x51ed_2701_a4c3_9e1f);
        for size in [11usize, 14, 16, 18, 20] {
            let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{dir}/bank-{size}.json")).unwrap()).unwrap();
            if let Some(p) = policy.as_mut() { p.resize(size); }
            for row in rows.iter().take(6) {
                let mut e = Engine::new(size);
                let ints = |k: &str| row[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect::<Vec<_>>();
                let tribes = ints("tribes");
                let caps: Vec<usize> = ints("capitals").iter().zip(&tribes).map(|(&c, &t)| c as usize | (t as usize) << 16).collect();
                let mut s = mapgen::initial_state(row["seed"].as_u64().unwrap() as u32, &ints("terrain"), &ints("structure"),
                                                  &ints("resource"), &caps, &e.geom, 200);
                for step in 0..200 {
                    if s.terminated || s.truncated { break; }
                    let cp = s.current_player;
                    let p = s.p();
                    // every visible enemy city with a free land neighbour: put a fresh own Cloak there
                    for c in 0..s.n {
                        if !(s.cities.owner[c] != 0 && s.cities.owner[c] != cp && visible(&s, p, c)) { continue; }
                        let Some(&a) = e.geom.adj[c].iter().find(|&&a| {
                            let a = a as usize;
                            visible(&s, p, a) && s.units.owner[a] == 0 && s.tiles.terrain[a] >= 3 && s.tiles.terrain[a] <= 5
                        }) else { continue };
                        let a = a as usize;
                        let mut base = s.clone();
                        base.units.owner[a] = cp;
                        base.units.kind[a] = CLOAK;
                        base.units.health[a] = 5.0;
                        base.units.invisible[a] = true;
                        base.units.direction[a] = 8;
                        for riot in [false, true] {
                            let mut b = base.clone();
                            b.cities.riot[c] = riot;
                            let mut m = b.clone();
                            m.cities.riot[c] = !riot;
                            assert!(view::packed(&b, cp, &e.geom) == view::packed(&m, cp, &e.geom), "riot of a foreign city is not in the view");
                            cases += 1;
                            let (pb, pm) = (e.legal(&public_state(&b, cp), true).to_vec(), e.legal(&public_state(&m, cp), true).to_vec());
                            assert!(pb == pm, "foreign riot changed the public candidates: size {size} step {step} city {c}");
                            let tm = e.legal(&m, true).to_vec();
                            if e.legal(&b, true) != &tm[..] { true_changed += 1; }
                            if let Some(pol) = policy.as_mut() {
                                for seed in 0..4u64 {
                                    let x = pol.act(&mut e, &b, false, &mut Rng::new(seed)).unwrap();
                                    let y = pol.act(&mut e, &m, false, &mut Rng::new(seed)).unwrap();
                                    assert_eq!((x.index, x.logp.to_bits()), (y.index, y.logp.to_bits()), "foreign riot changed the decision");
                                    let x = pol.act_mode(&mut e, &b, 2, &mut Rng::new(seed)).unwrap();
                                    let y = pol.act_mode(&mut e, &m, 2, &mut Rng::new(seed)).unwrap();
                                    assert_eq!((x.index, x.logp.to_bits()), (y.index, y.logp.to_bits()), "foreign riot changed the joint top-1");
                                    decisions += 1;
                                }
                            }
                        }
                    }
                    let legal: Vec<usize> = e.legal(&s, true).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
                    let a = legal[rand.below(legal.len())];
                    assert_eq!(e.apply(&mut s, a), core::OK);
                }
            }
        }
        eprintln!("riot cases {cases}; true mask changed (Cloak infiltration, positive control) {true_changed}; decisions compared {decisions}");
        assert!(cases > 0 && true_changed > 0, "the riot fixture never exercised Cloak infiltration");
    }
}

#[cfg(test)]
mod invariant {
    //! The explored lower bound in `public_variant`: on natural games every unit's owner has explored the unit's tile
    //! and its neighbours, and every tile it owns (units explore around themselves when they appear or move; borders
    //! explore what they claim).
    use crate::Engine;
    use polytopia_engine_rs::{core, mapgen};
    use serde_json::Value;
    #[test]
    fn units_owner_explored_their_neighbours() {
        let Ok(dir) = std::env::var("GEN_WASM_BANK") else { return };
        let (mut checked, mut bad, mut bad_owned) = (0usize, 0usize, 0usize);
        let (mut healths, mut fractional, mut poisoned) = (0usize, 0usize, 0usize);
        let mut x: u64 = 0x1234_5678_9abc_def1;
        for size in [11usize, 14, 16, 18, 20] {
            let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{dir}/bank-{size}.json")).unwrap()).unwrap();
            for row in rows.iter().take(3) {
                let mut e = Engine::new(size);
                let ints = |k: &str| row[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect::<Vec<_>>();
                let tribes = ints("tribes");
                let caps: Vec<usize> = ints("capitals").iter().zip(&tribes).map(|(&c, &t)| c as usize | (t as usize) << 16).collect();
                let mut s = mapgen::initial_state(row["seed"].as_u64().unwrap() as u32, &ints("terrain"), &ints("structure"), &ints("resource"), &caps, &e.geom, 200);
                for _ in 0..600 {
                    if s.terminated || s.truncated { break; }
                    for i in 0..s.n {
                        let o = s.units.owner[i];
                        if o > 0 {
                            for j in std::iter::once(i).chain(e.geom.adj[i].iter().map(|&j| j as usize)) {
                                checked += 1;
                                if !s.tiles.explored[(o - 1) as usize][j] { bad += 1; }
                            }
                        }
                        if s.tiles.owner[i] > 0 && !s.tiles.explored[(s.tiles.owner[i] - 1) as usize][i] { bad_owned += 1; }
                        // r13: health is always whole (kernels.rs combat rounds damage and retaliation to whole points,
                        // splash floors, healing adds 2 or 4, max health is whole), so a seat's whole-point view of an
                        // enemy unit's health is exact and the forecasts need no sub-point world
                        if o > 0 { healths += 1; if s.units.health[i].fract() != 0.0 { fractional += 1; } }
                        // and no unit is ever poisoned (engine-rs only clears Units.poison: no rule sets it), so the
                        // forecasts need no poisoned-enemy world either
                        if o > 0 && s.units.poison[i] { poisoned += 1; }
                    }
                    let legal: Vec<usize> = e.legal(&s, true).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                    assert_eq!(e.apply(&mut s, legal[(x % legal.len() as u64) as usize]), core::OK);
                }
            }
        }
        eprintln!("invariant: {checked} unit-neighbour checks, {bad} unexplored; owned tiles unexplored {bad_owned}; unit healths {healths}, fractional {fractional}, poisoned {poisoned}");
        assert_eq!((fractional, poisoned), (0, 0), "ui.rs forecasts assume whole health and no poison");
        assert!(healths > 50_000);
        assert_eq!((bad, bad_owned), (0, 0), "public_variant's explored lower bound would be wrong");
        assert!(checked > 100_000);
    }
}
