//! What the site's worker shows a seat during its own turn, built from the engine (r12 systematic payload test):
//! the seat view, the legal actions, the attack forecasts (each legal attack played on a copy: every unit of the
//! seat's view, after) and the enemy threat view (end the turn on a copy, settle the opponent's rewards, then each
//! visible enemy unit's moves and attacks, also after each move, on tiles the seat knows). The worker builds the
//! last three on the seat's public projection (`game_public`), exactly as `parts` does here, so none can depend on
//! hidden state; the true game referees what the seat then does.
use polytopia_engine_rs::core::{decode, Action};
use polytopia_engine_rs::state::State;
use serde_json::{json, Value};

use crate::view;
use crate::Engine;

const MOVE: i32 = 1;
const ATTACK: i32 = 2;
const REWARD: i32 = 9;
const END: i32 = 10;

/// The site's action id of an engine catalog entry the 12 regular tribes can be offered (Engine::site_id).
fn site(e: &Engine, i: usize) -> usize {
    e.site_id(i).expect("a special-tribe unit ability has no site action id")
}

/// Legal entries of the player to move, as ENGINE catalog ids (site ids only where the JSON leaves this module).
pub fn commands(e: &mut Engine, s: &State) -> Vec<(usize, Action)> {
    if s.terminated || s.truncated {
        return Vec::new();
    }
    let ids: Vec<usize> = e.legal(s, false).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
    ids.into_iter().map(|i| {
        let (f, src, t) = e.catalog.unflatten(i);
        (i, decode(s, f as i32, src as i32, t as i32))
    }).collect()
}

fn seat(e: &Engine, s: &State, viewer: i32) -> Value {
    view::seat_view(s, &e.rules, &e.geom, &e.layout, viewer)
}

/// The value the seat's board shows for a unit: its own exact health, another seat's in whole points (toward zero).
fn shown(s: &State, viewer: i32, i: usize) -> f32 {
    if s.units.owner[i] == viewer { s.units.health[i] } else { s.units.health[i].trunc() }
}

struct Range { lo: f32, hi: f32 }
impl Range {
    fn new() -> Range { Range { lo: f32::INFINITY, hi: f32::NEG_INFINITY } }
    fn add(&mut self, v: f32) { self.lo = self.lo.min(v); self.hi = self.hi.max(v); }
}

/// Every legal attack of the seat (`viewer`, to move), played in each public variant of its projection
/// (public::public_variant: the unseen defence technology, walls and retaliation map at each extreme; whole health and
/// no poison are invariants, see below). Each field is the range over those worlds; the site shows a number only when they agree and
/// otherwise the range (r13, r12b-M1: never "safe" when a world consistent with the view kills the
/// attacker). [{action_id, source, target, target_before, target_after (highest), target_after_min, attacker_before,
/// attacker_after (lowest), attacker_after_max, advances (in every world), advances_maybe, converts, converts_maybe,
/// collateral [{tile, before, after (highest), after_min}], certain}]
pub fn forecasts(e: &mut Engine, s: &State, viewer: i32) -> Value {
    // No sub-point health or poisoned-enemy world: health is always whole and no rule of engine-rs poisons a unit
    // (public.rs invariant test), so neither is a world consistent with the view.
    use crate::public::{passenger_options, public_state, public_variant, set_passengers, VAR_ALL, VAR_FOREIGN_HEALTH_TOP, VAR_FOREIGN_POISON};
    if s.current_player != viewer || s.terminated || s.truncated {
        return json!([]);
    }
    let base = public_state(s, viewer);
    // r14: every passenger a visible foreign boat may hold (its maximum health is the passenger's)
    let popts = passenger_options(&base, &e.rules, viewer);
    let worlds_p = popts.iter().map(|(_, o)| o.len()).max().unwrap_or(1).max(1);
    let attacks: Vec<(usize, Action)> = commands(e, &base).into_iter().filter(|(_, a)| a.kind == ATTACK).collect();
    let mut out = Vec::new();
    for (id, a) in attacks {
        let (src, tgt) = (a.source as usize, a.target as usize);
        let (me, foe) = (base.units.owner[src], base.units.owner[tgt]);
        let others: Vec<usize> = (0..base.n).filter(|&i| i != tgt && base.units.owner[i] != 0 && base.units.owner[i] != me).collect();
        let (mut ta, mut aa) = (Range::new(), Range::new());
        let mut coll: Vec<Range> = others.iter().map(|_| Range::new()).collect();
        let (mut adv, mut conv, mut worlds) = ([0usize; 2], [0usize; 2], 0usize);
        for mask in (0..=VAR_ALL).filter(|m| m & (VAR_FOREIGN_HEALTH_TOP | VAR_FOREIGN_POISON) == 0) {
          for pk in 0..worlds_p {
            let mut q = public_variant(s, &e.rules, viewer, mask);
            set_passengers(&mut q, &popts, pk);
            if e.legal(&q, false).get(id) != Some(&1) { continue; }
            worlds += 1;
            e.apply(&mut q, id);
            let at = |i: usize| (q.units.owner[i] != 0).then(|| q.units.owner[i]);
            let target_after = if at(tgt) == Some(foe) { shown(&q, viewer, tgt) } else { 0.0 };
            let attacker_after = if at(src) == Some(me) { shown(&q, viewer, src) } else if at(tgt) == Some(me) { shown(&q, viewer, tgt) } else { 0.0 };
            ta.add(target_after);
            aa.add(attacker_after);
            let advances = at(src).is_none() && at(tgt) == Some(me);
            let converts = at(src) == Some(me) && at(tgt) == Some(me) && foe != me;
            adv[advances as usize] += 1;
            conv[converts as usize] += 1;
            for (k, &i) in others.iter().enumerate() {
                coll[k].add(if q.units.owner[i] == base.units.owner[i] { shown(&q, viewer, i) } else { 0.0 });
            }
          }
        }
        if worlds == 0 { continue; }
        let collateral: Vec<Value> = others.iter().zip(&coll).filter_map(|(&i, r)| {
            let before = shown(&base, viewer, i);
            (r.lo < before).then(|| json!({"tile": i, "before": before, "after": r.hi, "after_min": r.lo}))
        }).collect();
        let certain = ta.lo == ta.hi && aa.lo == aa.hi && adv[0] * adv[1] == 0 && conv[0] * conv[1] == 0
            && collateral.iter().all(|c| c["after"] == c["after_min"]);
        out.push(json!({
            "action_id": site(e, id).to_string(), "source": src, "target": tgt,
            "target_before": if foe != 0 { shown(&base, viewer, tgt) } else { 0.0 }, "target_after": ta.hi, "target_after_min": ta.lo,
            "attacker_before": shown(&base, viewer, src), "attacker_after": aa.lo, "attacker_after_max": aa.hi,
            "advances": adv[0] == 0, "advances_maybe": adv[1] > 0, "converts": conv[0] == 0, "converts_maybe": conv[1] > 0,
            "collateral": collateral, "certain": certain,
        }));
    }
    Value::Array(out)
}

/// The threat view in two public worlds, joined: the enemy's map and technology at the provable minimum and at
/// everything (public::public_variant). For each visible enemy unit, [tile, known move targets, known attack
/// targets] (attacks also after each move); first the danger row [-1, [], every known tile any of them can attack].
/// "Could" is the union, so a public-consistent move is never left out.
pub fn threats(e: &mut Engine, s: &State, viewer: i32) -> Value {
    use crate::public::{passenger_options, public_state, public_variant, set_passengers, VAR_ALL};
    if s.current_player != viewer || s.terminated || s.truncated {
        return json!([]);
    }
    let base = public_state(s, viewer);
    let mine = seat(e, &base, viewer);
    let known: Vec<bool> = mine["visible_state"]["tiles"].as_array().unwrap().iter().map(|t| t["known"].as_bool() == Some(true)).collect();
    let sources: Vec<i32> = mine["visible_state"]["units"].as_array().unwrap().iter()
        .filter(|u| u["owner"].as_i64() != Some(viewer as i64)).map(|u| u["tile"].as_i64().unwrap() as i32).collect();
    if sources.is_empty() {
        return json!([]);
    }
    let mut moves: Vec<Vec<i32>> = vec![Vec::new(); sources.len()];
    let mut hits: Vec<Vec<i32>> = vec![Vec::new(); sources.len()];
    // r14: and each passenger a visible foreign boat may hold (a boat with a passenger can also land)
    let popts = passenger_options(&base, &e.rules, viewer);
    let worlds_p = popts.iter().map(|(_, o)| o.len()).max().unwrap_or(1).max(1);
    for (mask, pk) in [0, VAR_ALL].into_iter().flat_map(|m| (0..worlds_p).map(move |k| (m, k))) {
        let mut c = public_variant(s, &e.rules, viewer, mask);
        set_passengers(&mut c, &popts, pk);
        let Some((end, _)) = commands(e, &c).into_iter().find(|(_, a)| a.kind == END) else { continue };
        e.apply(&mut c, end);
        for _ in 0..8 {
            let Some((r, _)) = commands(e, &c).into_iter().find(|(_, a)| a.kind == REWARD) else { break };
            e.apply(&mut c, r);
        }
        if c.current_player == viewer || c.terminated || c.truncated {
            continue;
        }
        let legal = commands(e, &c);
        for (k, &src) in sources.iter().enumerate() {
            for (id, m) in legal.iter().filter(|(_, a)| a.kind == MOVE && a.source == src) {
                moves[k].push(m.target);
                let mut after = c.clone();
                e.apply(&mut after, *id);
                for (_, a) in commands(e, &after) {
                    if a.kind == ATTACK && a.source == m.target { hits[k].push(a.target); }
                }
            }
            hits[k].extend(legal.iter().filter(|(_, a)| a.kind == ATTACK && a.source == src).map(|(_, a)| a.target));
        }
    }
    let mut rows = Vec::new();
    let mut danger: Vec<i32> = Vec::new();
    for (k, &src) in sources.iter().enumerate() {
        let tidy = |v: &mut Vec<i32>| { v.retain(|&t| known[t as usize]); v.sort(); v.dedup(); };
        tidy(&mut moves[k]);
        tidy(&mut hits[k]);
        danger.extend(&hits[k]);
        rows.push(json!([src, moves[k], hits[k]]));
    }
    danger.sort();
    danger.dedup();
    rows.insert(0, json!([-1, [], danger]));
    Value::Array(rows)
}

/// [seat view, legal actions, attack forecasts, enemy threat] as JSON text, for the player to move: the view from
/// the game, the rest from the seat's public projection (what the worker does with `game_public`).
pub fn parts(e: &mut Engine, s: &State, viewer: i32) -> [String; 4] {
    parts_from(e, s, viewer, true)
}

/// `parts` with the legal actions, forecasts and threats from the true state when `public` is false (the site up to
/// r12 5a017773): the positive control of the every-field test.
pub fn parts_from(e: &mut Engine, s: &State, viewer: i32, public: bool) -> [String; 4] {
    let view = seat(e, s, viewer).to_string();
    if !public {
        let legal: Vec<Value> = if s.current_player == viewer {
            commands(e, s).into_iter().map(|(i, a)| json!([site(e, i), a.kind, a.source, a.target, a.parameter])).collect()
        } else { Vec::new() };
        return [view, Value::Array(legal).to_string(), true_forecasts(e, s, viewer).to_string(), true_threats(e, s, viewer).to_string()];
    }
    let truth = s;
    let projected = crate::public::public_state(s, viewer);
    let s = &projected;
    let legal: Vec<Value> = if s.current_player == viewer {
        commands(e, s).into_iter().map(|(i, a)| json!([site(e, i), a.kind, a.source, a.target, a.parameter])).collect()
    } else {
        Vec::new()
    };
    [view, Value::Array(legal).to_string(), forecasts(e, truth, viewer).to_string(), threats(e, truth, viewer).to_string()]
}

/// The site up to r12 (positive control only): one world, the true state.
fn true_forecasts(e: &mut Engine, s: &State, viewer: i32) -> Value {
    if s.current_player != viewer || s.terminated || s.truncated { return json!([]); }
    let mut out = Vec::new();
    for (id, a) in commands(e, s).into_iter().filter(|(_, a)| a.kind == ATTACK) {
        let mut c = s.clone();
        e.apply(&mut c, id);
        let row: Vec<Value> = (0..c.n).filter(|&i| c.units.owner[i] != 0).map(|i| json!([i, c.units.owner[i], c.units.health[i]])).collect();
        out.push(json!([site(e, id), a.target, row]));
    }
    Value::Array(out)
}

fn true_threats(e: &mut Engine, s: &State, viewer: i32) -> Value {
    if s.current_player != viewer || s.terminated || s.truncated { return json!([]); }
    let mut c = s.clone();
    let Some((end, _)) = commands(e, &c).into_iter().find(|(_, a)| a.kind == END) else { return json!([]) };
    e.apply(&mut c, end);
    for _ in 0..8 {
        let Some((r, _)) = commands(e, &c).into_iter().find(|(_, a)| a.kind == REWARD) else { break };
        e.apply(&mut c, r);
    }
    let legal: Vec<Value> = commands(e, &c).into_iter().filter(|(_, a)| a.kind == MOVE || a.kind == ATTACK)
        .map(|(_, a)| json!([a.kind, a.source, a.target])).collect();
    Value::Array(legal)
}

#[cfg(test)]
mod tests {
    //! 11 Nov: a seat is offered the orders its public view allows; when one meets hidden
    //! state the true game resolves it as the real game does. Recorded cases (the hidden zone-of-control refusal is
    //! web tests/carry-paths.test.ts, natural pair).
    use super::*;
    use polytopia_engine_rs::mapgen;
    use serde_json::Value;

    const CLOAK: i32 = 38;

    /// move command (engine-rs core.rs move_adjacent): a move onto a tile
    /// holding a hidden enemy unit reveals it (reveal action), and the mover stays where it is and keeps its move.
    #[test]
    fn a_move_onto_a_hidden_cloak_reveals_it_and_the_mover_stays() {
        let mut e = Engine::new(14);
        let mut s = mapgen::reset(7, &e.geom, None);
        let cp = s.current_player;
        let p = (cp - 1) as usize;
        let opp = 3 - cp;
        let (src, t) = (0..s.n).filter(|&i| s.units.owner[i] == cp)
            .find_map(|i| e.geom.adj[i].iter().map(|&j| j as usize)
                .find(|&j| s.tiles.terrain[j] == 3 && s.units.owner[j] == 0 && s.tiles.explored[p][j] && s.tiles.structure[j] == 0)
                .map(|j| (i, j)))
            .expect("an own unit next to an empty explored field");
        s.units.owner[t] = opp;
        s.units.kind[t] = CLOAK;
        s.units.health[t] = 5.0;
        s.units.invisible[t] = true;
        // The seat cannot see it, and is offered the move (from its public projection, as the site builds it) ...
        fn seen(e: &Engine, s: &State, cp: i32, t: usize) -> bool {
            view::seat_view(s, &e.rules, &e.geom, &e.layout, cp)["visible_state"]["units"].as_array().unwrap()
                .iter().any(|u| u["tile"].as_u64() == Some(t as u64))
        }
        assert!(!seen(&e, &s, cp, t));
        let public = crate::public::public_state(&s, cp);
        let offered = commands(&mut e, &public).into_iter().find(|(_, a)| a.kind == MOVE && a.source == src as i32 && a.target == t as i32);
        let (id, _) = offered.expect("the public view offers the move");
        // ... the true game accepts it (a hidden unit's tile is passable, is tile accessible) ...
        assert_eq!(e.legal(&s, false)[id], 1);
        assert_eq!(e.apply(&mut s, id), polytopia_engine_rs::core::OK);
        // ... and resolves it: the Cloak is revealed, the mover stays on its tile with its move left.
        assert_eq!(s.units.owner[src], cp);
        assert_eq!(s.units.owner[t], opp);
        assert!(!s.units.invisible[t]);
        assert!(seen(&e, &s, cp, t));
        assert!(commands(&mut e, &s).iter().any(|(_, a)| a.kind == MOVE && a.source == src as i32), "the mover keeps its move");
    }

    /// r13 (a projection-only forecast said the Archer survives 10->10 when the real
    /// attack kills it). On natural positions every forecast range must hold the true outcome: target, attacker,
    /// collateral, advance and conversion. Control: the r12b single-world forecast (the projection) misses it.
    #[test]
    fn every_forecast_range_holds_the_true_outcome() {
        let Ok(dir) = std::env::var("GEN_WASM_BANK") else { return };
        let (mut checked, mut outside, mut uncertain, mut refused, mut single_wrong) = (0usize, Vec::new(), 0usize, 0usize, 0usize);
        let mut x: u64 = 0x2545_f491_4f6c_dd1d;
        for size in [11usize, 14, 16, 18] {
            let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{dir}/bank-{size}.json")).unwrap()).unwrap();
            for row in rows.iter().take(6) {
                let mut e = Engine::new(size);
                let ints = |k: &str| row[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect::<Vec<_>>();
                let tribes = ints("tribes");
                let caps: Vec<usize> = ints("capitals").iter().zip(&tribes).map(|(&c, &t)| c as usize | (t as usize) << 16).collect();
                let mut s = mapgen::initial_state(row["seed"].as_u64().unwrap() as u32, &ints("terrain"), &ints("structure"), &ints("resource"), &caps, &e.geom, 200);
                for step in 0..700 {
                    if s.terminated || s.truncated { break; }
                    let viewer = s.current_player;
                    if true {
                        let f = forecasts(&mut e, &s, viewer);
                        let public = crate::public::public_state(&s, viewer);
                        for p in f.as_array().unwrap() {
                            let id: usize = e.engine_of(p["action_id"].as_str().unwrap()).unwrap();
                            let (src, tgt) = (p["source"].as_u64().unwrap() as usize, p["target"].as_u64().unwrap() as usize);
                            if e.legal(&s, false).get(id) != Some(&1) { refused += 1; continue; }
                            let (me, foe) = (s.units.owner[src], s.units.owner[tgt]);
                            let mut t = s.clone();
                            e.apply(&mut t, id);
                            let at = |q: &State, i: usize| (q.units.owner[i] != 0).then(|| q.units.owner[i]);
                            let ta = if at(&t, tgt) == Some(foe) { shown(&t, viewer, tgt) } else { 0.0 };
                            let aa = if at(&t, src) == Some(me) { shown(&t, viewer, src) } else if at(&t, tgt) == Some(me) { shown(&t, viewer, tgt) } else { 0.0 };
                            let adv = at(&t, src).is_none() && at(&t, tgt) == Some(me);
                            let conv = at(&t, src) == Some(me) && at(&t, tgt) == Some(me) && foe != me;
                            let f32v = |k: &str| p[k].as_f64().unwrap() as f32;
                            let mut ok = f32v("target_after_min") <= ta && ta <= f32v("target_after")
                                && f32v("attacker_after") <= aa && aa <= f32v("attacker_after_max")
                                && (!p["advances"].as_bool().unwrap() || adv) && (p["advances_maybe"].as_bool().unwrap() || !adv)
                                && (!p["converts"].as_bool().unwrap() || conv) && (p["converts_maybe"].as_bool().unwrap() || !conv);
                            // collateral: every other foreign unit the truth hurts is listed, within its range
                            for i in 0..t.n {
                                if i == tgt || s.units.owner[i] == 0 || s.units.owner[i] == me { continue; }
                                let before = shown(&s, viewer, i);
                                let after = if t.units.owner[i] == s.units.owner[i] { shown(&t, viewer, i) } else { 0.0 };
                                let listed = p["collateral"].as_array().unwrap().iter().find(|c| c["tile"].as_u64() == Some(i as u64));
                                match listed {
                                    Some(c) => ok &= c["after_min"].as_f64().unwrap() as f32 <= after && after <= c["after"].as_f64().unwrap() as f32,
                                    None => ok &= after >= before || public.units.owner[i] == 0,
                                }
                            }
                            if !ok { outside.push(format!("size {size} seed {} step {step} {p} truth target {ta} attacker {aa} adv {adv} conv {conv}", row["seed"])); }
                            if !p["certain"].as_bool().unwrap() {
                                uncertain += 1;
                                if std::env::var("DIAG").is_ok() {
                                    let out = |e: &mut Engine, m: u32| { let mut q = crate::public::public_variant(&s, &e.rules, viewer, m); if e.legal(&q, false).get(id) != Some(&1) { return (-1.0, -1.0) }; e.apply(&mut q, id);
                                        (if at(&q, tgt) == Some(foe) { shown(&q, viewer, tgt) } else { 0.0 }, if at(&q, src) == Some(me) { shown(&q, viewer, src) } else if at(&q, tgt) == Some(me) { shown(&q, viewer, tgt) } else { 0.0 }) };
                                    let o0 = out(&mut e, 0);
                                    let bits: Vec<u32> = [1u32, 2, 4, 16].into_iter().filter(|&b| out(&mut e, b) != o0).collect();
                                    let ranged = e.rules.unit_range[s.units.kind[src] as usize];
                                    eprintln!("DIAG bits {bits:?} range {ranged} truth ({ta},{aa}) m0 {o0:?} explored_src {}", s.tiles.explored[(foe - 1) as usize][src]);
                                }
                            }
                            // control: the r12b forecast, one world (the projection)
                            let mut q = public.clone();
                            if e.legal(&q, false).get(id) == Some(&1) {
                                e.apply(&mut q, id);
                                let qa = if at(&q, src) == Some(me) { shown(&q, viewer, src) } else if at(&q, tgt) == Some(me) { shown(&q, viewer, tgt) } else { 0.0 };
                                let qt = if at(&q, tgt) == Some(foe) { shown(&q, viewer, tgt) } else { 0.0 };
                                if qa != aa || qt != ta { single_wrong += 1; }
                            }
                            checked += 1;
                        }
                    }
                    let legal: Vec<usize> = e.legal(&s, true).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
                    // natural-ish play: prefer attacks so fights happen
                    let attacks: Vec<usize> = legal.iter().copied().filter(|&i| e.catalog.unflatten(i).0 as i32 == ATTACK).collect();
                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                    let pick = if !attacks.is_empty() && x % 3 == 0 { attacks[(x as usize / 3) % attacks.len()] } else { legal[(x % legal.len() as u64) as usize] };
                    assert_eq!(e.apply(&mut s, pick), polytopia_engine_rs::core::OK);
                }
            }
        }
        eprintln!("forecast ranges: {checked} attacks against the truth, {} outside, {uncertain} uncertain, {refused} refused by the referee; control (r12b single-world forecast) wrong {single_wrong}", outside.len());
        for o in outside.iter().take(5) { eprintln!("  {o}"); }
        assert!(outside.is_empty());
        assert!(checked > 1000 && single_wrong > 0, "premise: enough attacks, and the control must miss");
    }

    /// r14: a visible foreign boat's passenger is hidden, and the boat's maximum health is the
    /// passenger's. On natural positions, every attack whose target or neighbours hold a foreign boat is replayed on
    /// the truth with every passenger the boat may hold (land kinds whose maximum health is at least its health): every
    /// outcome must lie in the forecast range (the forecast is the same in all of them: the passenger is hidden).
    /// Control: without the passenger worlds (r13b) some fall outside.
    #[test]
    fn forecast_ranges_hold_for_every_hidden_passenger() {
        use polytopia_engine_rs::rules::{CARRY, FLOAT};
        let Ok(dir) = std::env::var("GEN_WASM_BANK") else { return };
        let (mut cases, mut outside, mut r13b_outside) = (0usize, Vec::new(), 0usize);
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        for size in [11usize, 14, 16, 18, 20] {
            let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{dir}/bank-{size}.json")).unwrap()).unwrap();
            for row in rows.iter().take(8) {
                let mut e = Engine::new(size);
                let ints = |k: &str| row[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect::<Vec<_>>();
                let tribes = ints("tribes");
                let caps: Vec<usize> = ints("capitals").iter().zip(&tribes).map(|(&c, &t)| c as usize | (t as usize) << 16).collect();
                let mut s = mapgen::initial_state(row["seed"].as_u64().unwrap() as u32, &ints("terrain"), &ints("structure"), &ints("resource"), &caps, &e.geom, 200);
                let land: Vec<i32> = (1..128).filter(|&k| e.rules.unit_health[k] > 0.0 && !e.rules.skills[k][FLOAT] && !e.rules.skills[k][CARRY] && !matches!(k, 12 | 37 | 38)).map(|k| k as i32).collect();
                for _ in 0..900 {
                    if s.terminated || s.truncated { break; }
                    let viewer = s.current_player;
                    let boats: Vec<usize> = (0..s.n).filter(|&i| s.units.owner[i] != 0 && s.units.owner[i] != viewer && s.units.passenger[i] > 0).collect();
                    if !boats.is_empty() {
                        let f = forecasts(&mut e, &s, viewer);
                        for p in f.as_array().unwrap() {
                            let id: usize = e.engine_of(p["action_id"].as_str().unwrap()).unwrap();
                            let (src, tgt) = (p["source"].as_u64().unwrap() as usize, p["target"].as_u64().unwrap() as usize);
                            let near: Vec<usize> = boats.iter().copied().filter(|&b| b == tgt || e.geom.adj[tgt].iter().any(|&j| j as usize == b)).collect();
                            for &b in &near {
                                for &l in &land {
                                    let vet = if s.units.veteran[b] { 5.0 } else { 0.0 };
                                    if e.rules.unit_health[l as usize] + vet < s.units.health[b] { continue; }
                                    let mut t = s.clone();
                                    t.units.passenger[b] = l;
                                    if e.legal(&t, false).get(id) != Some(&1) { continue; }
                                    let (me, foe) = (t.units.owner[src], t.units.owner[tgt]);
                                    let before = t.clone();
                                    e.apply(&mut t, id);
                                    let at = |q: &State, i: usize| (q.units.owner[i] != 0).then(|| q.units.owner[i]);
                                    let ta = if at(&t, tgt) == Some(foe) { shown(&t, viewer, tgt) } else { 0.0 };
                                    let aa = if at(&t, src) == Some(me) { shown(&t, viewer, src) } else if at(&t, tgt) == Some(me) { shown(&t, viewer, tgt) } else { 0.0 };
                                    let fv = |k: &str| p[k].as_f64().unwrap() as f32;
                                    let mut ok = fv("target_after_min") <= ta && ta <= fv("target_after") && fv("attacker_after") <= aa && aa <= fv("attacker_after_max");
                                    if b != tgt {
                                        let after = if t.units.owner[b] == before.units.owner[b] { shown(&t, viewer, b) } else { 0.0 };
                                        match p["collateral"].as_array().unwrap().iter().find(|c| c["tile"].as_u64() == Some(b as u64)) {
                                            Some(c) => ok &= c["after_min"].as_f64().unwrap() as f32 <= after && after <= c["after"].as_f64().unwrap() as f32,
                                            None => ok &= after >= shown(&before, viewer, b),
                                        }
                                    }
                                    cases += 1;
                                    if !ok { outside.push(format!("size {size} boat {b} passenger {l} {p} truth target {ta} attacker {aa}")); }
                                    // control: the r13b worlds (passenger 0) alone
                                    let mut q = crate::public::public_state(&before, viewer);
                                    if e.legal(&q, false).get(id) == Some(&1) {
                                        e.apply(&mut q, id);
                                        let qt = if at(&q, tgt) == Some(foe) { shown(&q, viewer, tgt) } else { 0.0 };
                                        if b == tgt && qt != ta { r13b_outside += 1; }
                                    }
                                }
                            }
                        }
                    }
                    let legal: Vec<usize> = e.legal(&s, true).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
                    let attacks: Vec<usize> = legal.iter().copied().filter(|&i| e.catalog.unflatten(i).0 as i32 == ATTACK).collect();
                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                    let pick = if !attacks.is_empty() && x % 3 == 0 { attacks[(x as usize / 3) % attacks.len()] } else { legal[(x % legal.len() as u64) as usize] };
                    assert_eq!(e.apply(&mut s, pick), polytopia_engine_rs::core::OK);
                }
            }
        }
        eprintln!("hidden passengers: {cases} attack x passenger cases, {} outside the forecast; control (projection, passenger 0) differs from the truth in {r13b_outside}", outside.len());
        for o in outside.iter().take(5) { eprintln!("  {o}"); }
        assert!(outside.is_empty());
        assert!(cases >= 50 && r13b_outside > 0, "premise: boats attacked, and the control must miss");
    }
}
