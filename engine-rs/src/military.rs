//! gpu/military.py: unit actions (native profile).

use crate::diplomacy::{hostile, protected};
use crate::economy::discover;
use crate::geom::Geom;
use crate::kernels::{combat, max_health};
use crate::rules::{Rules, CARRY, FLOAT, HIDE, INDEPENDENT, SCOUT, SPLASH, STOMP, SURPRISE};
use crate::state::State;
use crate::structures::capital_index;

pub fn exhaust(s: &mut State, source: usize) {
    s.units.moved[source] = true;
    s.units.attacked[source] = true;
}

/// military.remove: killer=None mirrors killer=-1 (index clipped to 0, no credit).
pub fn remove(s: &mut State, target: usize, r: &Rules, killer: Option<usize>, casualty: bool) {
    let u = &s.units;
    let owner = s.pclip(u.owner[target]);
    let k = killer.unwrap_or(0);
    let kp = s.pclip(u.owner[k]);
    let kind = u.kind[target] as usize;
    let passenger = u.passenger[target];
    let value = r.unit_cost[kind] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    let loss = -5 * value * (!u.converted[target]) as i32;
    s.players.score[owner] = s.players.score[owner].wrapping_add(loss);
    s.players.casualties[owner] += casualty as i32;
    s.players.kills[kp] += killer.is_some() as i32;
    if s.units.poison[target] && s.tiles.structure[target] == 0 {
        s.tiles.resource[target] = 7;
    }
    // game version >= 95: no xp for a killer that is a vehicle (Carry: every
    // boat) or Static; the passenger of a boat keeps the xp it had when it embarked
    let xp = killer.is_some() && !(r.skills[u.kind[k] as usize][CARRY] || r.stat[u.kind[k] as usize]);
    s.units.kills[k] += xp as i32;
    s.units.clear(target);
}

pub fn relocate(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom, involuntary: bool, reveal: Option<&[bool]>) {
    let (n, size) = (s.n, s.size);
    let owner = s.units.owner[source];
    let kind = s.units.kind[source] as usize;
    let passenger_src = s.units.passenger[source];
    let moved_src = s.units.moved[source];
    let attacked_src = s.units.attacked[source];
    let invisible_src = s.units.invisible[source];
    let converted_src = s.units.converted[source];
    let terrain_src = s.tiles.terrain[source];
    s.units.move_unit(source, target);
    s.units.moved[target] = true;
    s.units.attacked[target] = true;
    s.units.prev_x[target] = (source % size) as i32;
    s.units.prev_y[target] = (source / size) as i32;
    let scout = r.skills[kind][SCOUT];
    let own_reveal: Vec<bool>;
    let reveal: &[bool] = match reveal {
        Some(v) => v,
        None => {
            let rs = if terrain_src == 4 || scout { 2 } else { 1 };
            let rt = if s.tiles.terrain[target] == 4 || scout { 2 } else { 1 };
            own_reveal = (0..n).map(|t| g.d(source, t) <= rs || g.d(target, t) <= rt).collect();
            &own_reveal
        }
    };
    discover(s, reveal, owner);
    if r.skills[kind][STOMP] {
        crate::cov::hit(28);
        area_damage(s, target, target, r, g, true);
    }
    let tt = s.tiles.terrain[target];
    let embark = s.tiles.structure[target] == 8 && !r.skills[kind][FLOAT];
    let special_land = (kind == 39 || kind == 40 || kind == 45) && tt > 2;
    let disembark = r.skills[kind][CARRY] && (passenger_src > 0 || special_land) && tt > 2 && !embark;
    let mut new_kind = if embark {
        if kind == 12 { 45 } else if kind == 38 { 39 } else if kind == 37 { 40 } else { 43 }
    } else if disembark {
        passenger_src as usize
    } else {
        kind
    };
    if special_land {
        new_kind = if kind == 39 { 38 } else if kind == 40 { 37 } else { 12 };
    }
    let passenger = if embark && kind != 12 && kind != 38 && kind != 37 { kind as i32 } else if disembark { 0 } else { passenger_src };
    let score_before = r.unit_cost[kind] + if passenger_src > 0 { r.unit_cost[passenger_src as usize] } else { 0 };
    let score_after = r.unit_cost[new_kind] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    if embark || disembark { crate::cov::hit(36); }
    let dash = !involuntary && !moved_src && !attacked_src && r.dash[new_kind];
    s.units.kind[target] = new_kind as i32;
    s.units.passenger[target] = passenger;
    s.units.attacked[target] = if dash { false } else { s.units.attacked[target] || disembark };
    let hidden = r.skills[new_kind][HIDE] && !involuntary;
    s.units.invisible[target] |= hidden;
    s.units.attacked[target] |= hidden && !invisible_src;
    let o = s.pwrap(owner);
    s.players.score[o] = s.players.score[o].wrapping_add(5 * (score_after - score_before) * (!converted_src) as i32);
}

pub fn attack_mask(s: &State, source: usize, r: &Rules, g: &Geom, out: &mut [bool]) {
    out[..s.n].fill(false);
    attack_targets(s, source, r, g, |t| out[t] = true);
}

/// The true entries of attack_mask's row for `source`, visiting only the tiles within the unit's range.
pub fn attack_targets(s: &State, source: usize, r: &Rules, g: &Geom, mut emit: impl FnMut(usize)) {
    let u = &s.units;
    let cp = s.current_player;
    let kind = u.kind[source] as usize;
    let own = u.owner[source] == cp && !u.attacked[source] && u.health[source] > 0.0;
    let mut can = own && (!u.moved[source] || r.dash[kind]) && r.unit_attack[kind] >= 0.0;
    let infil = r.skills[kind][crate::rules::INFILTRATE];
    can = can && (r.unit_attack[kind] > 0.0 || kind == 10 || infil);
    let range = r.unit_range[kind];
    if !can || range < 1 {
        return;
    }
    let stamp = s.turn * s.np as i32 + cp;
    let explored = &s.tiles.explored[s.p()];
    let size = s.size as i32;
    let (sx, sy) = ((source % s.size) as i32, (source / s.size) as i32);
    for y in (sy - range).max(0)..=(sy + range).min(size - 1) {
        for x in (sx - range).max(0)..=(sx + range).min(size - 1) {
            let t = (y * size + x) as usize;
            if t == source || !explored[t] {
                continue;
            }
            let targets = if infil {
                State::foreign(s.cities.owner[t], cp) && !protected(s, cp, s.cities.owner[t])
                    && !(u.owner[t] != 0 && u.owner[t] != s.cities.owner[t]) && s.cities.infiltrated[t] != stamp
            } else {
                State::foreign(u.owner[t], cp) && !u.invisible[t] && !protected(s, cp, u.owner[t])
            };
            if targets {
                debug_assert!(g.d(source, t) > 0 && g.d(source, t) <= range);
                emit(t);
            }
        }
    }
}

#[inline]
fn terrain_bonus(s: &State, r: &Rules, i: usize, tech: &[bool; 128]) -> f32 {
    let (t, u, c) = (&s.tiles, &s.units, &s.cities);
    let terrain = t.terrain[i];
    let city = if c.owner[i] == u.owner[i] && r.fortify[u.kind[i] as usize] {
        if s.reward(i, 1) > 0 { 4.0 } else { 1.5 }
    } else {
        1.0
    };
    let bonus = if terrain == 1 || terrain == 2 {
        if tech[12] { 1.5 } else { 1.0 }
    } else if terrain == 5 {
        if tech[18] { 1.5 } else { 1.0 }
    } else if terrain == 4 {
        if tech[20] { 1.5 } else { 1.0 }
    } else {
        city
    };
    bonus * if u.poison[i] { 0.5 } else { 1.0 }
}

pub fn defense_bonus(s: &State, r: &Rules, i: usize) -> f32 {
    let tech = &s.players.tech[s.pclip(s.units.owner[i])];
    terrain_bonus(s, r, i, tech)
}

pub fn attack(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom) {
    let size = s.size;
    let cp = s.current_player;
    let kind = s.units.kind[source] as usize;
    let defender = s.units.kind[target] as usize;
    let terrain = s.tiles.terrain[target];
    let tech = s.players.tech[s.pwrap(s.units.owner[target])];
    let bonus = terrain_bonus(s, r, target, &tech);
    let hp_s = max_health(s, r, source);
    let hp_t = max_health(s, r, target);
    let dd = r.unit_defense[defender] * if s.units.poison[target] { 0.7 } else { 1.0 };
    let (damage, retaliation, _) = combat(r.unit_attack[kind], s.units.health[source], hp_s, dd, s.units.health[target], hp_t, bonus);
    let dead = s.units.health[target] - damage <= 0.0;
    // retaliation: the attacker must also be one of the defender's attack
    // options, i.e. stand on a tile the defender's owner has explored
    let seen = s.tiles.explored[s.pclip(s.units.owner[target])][source];
    let retaliate = !r.skills[kind][SURPRISE] && !r.stiff[defender] && g.d(source, target) <= r.unit_range[defender] && seen;
    s.players.attacked[s.p()] = true;
    s.units.health[target] += -damage;
    s.units.boosted[target] = false;
    s.units.prev_x[source] = 2 * (source % size) as i32 - (target % size) as i32;
    s.units.prev_y[source] = 2 * (source / size) as i32 - (target / size) as i32;
    s.units.last_attack[source] = target as i32 + 1;
    if r.skills[kind][SPLASH] {
        crate::cov::hit(27);
        area_damage(s, source, target, r, g, false);
    }
    if dead {
        remove(s, target, r, Some(source), true);
    } else {
        s.units.health[source] += -(if retaliate { retaliation } else { 0.0 });
    }
    let attacker_dead = s.units.health[source] <= 0.0;
    if attacker_dead {
        crate::cov::hit(34);
        remove(s, source, r, Some(target), true);
    }
    let st = s.tiles.structure[target];
    let can_enter = terrain == 3 || terrain == 5 || (terrain == 4 && s.players.tech[s.p()][20])
        || (terrain <= 2 && (r.skills[kind][FLOAT] || st == 48 || (st == 8 && s.tiles.owner[target] == cp)));
    let advance = dead && !attacker_dead && r.unit_range[kind] < 2 && can_enter;
    if advance {
        crate::cov::hit(35);
        relocate(s, source, target, r, g, true, None);
    }
    let at = if advance { target } else { source };
    if !attacker_dead {
        let persist = r.persist[kind];
        let escape = r.escape[kind];
        s.units.attacks_performed[at] += 1;
        s.units.attacked[at] = !(dead && persist);
        s.units.moved[at] = !escape;
        if dead && persist {
            // Persist leaves the unit able to attack only when it has
            // another target from where it now stands
            let mut more = false;
            attack_targets(s, at, r, g, |_| more = true);
            s.units.attacked[at] = !more;
        }
    }
}

pub fn convert(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let cp = s.current_player;
    let old = s.pwrap(s.units.owner[target]);
    let (cap, exists) = capital_index(s, cp);
    let mut fallback = 0usize;
    let mut best = i32::MAX;
    for i in 0..n {
        if s.cities.owner[i] == cp && s.cities.order[i] < best {
            best = s.cities.order[i];
            fallback = i;
        }
    }
    let tk = s.units.kind[target] as usize;
    let home = if r.skills[tk][INDEPENDENT] { 0 } else { (if exists { cap } else { fallback }) as i32 + 1 };
    let passenger = s.units.passenger[target];
    let cost = r.unit_cost[tk] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    s.players.score[old] = s.players.score[old].wrapping_sub(5 * cost * (!s.units.converted[target]) as i32);
    s.players.attacked[s.p()] = true;
    s.units.owner[target] = cp;
    s.units.converted[target] = true;
    s.units.home[target] = home;
    s.units.moved[target] = true;
    s.units.attacked[target] = true;
    exhaust(s, source);
    let radius = if s.tiles.terrain[target] == 4 || r.skills[tk][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(target, t) <= radius).collect();
    discover(s, &reveal, cp);
}

/// out[t*128 + kind]
/// The Raft upgrades (Rammership, Scoutship, Bomber): only summonable onto an own Raft.
pub const UPGRADES: u128 = 1 << 41 | 1 << 42 | 1 << 44;

/// Per-call inputs of summon_mask: units per home city and the unlocked, affordable kinds.
pub struct SummonCtx {
    counts: Vec<i32>,
    ok: u128,
}

impl SummonCtx {
    pub fn new(n: usize) -> SummonCtx {
        SummonCtx { counts: vec![0; n], ok: 0 }
    }

    pub fn prepare(&mut self, s: &State, r: &Rules) {
        let n = s.n;
        let u = &s.units;
        let cp = s.current_player;
        let p = s.p();
        self.counts.fill(0);
        for i in 0..n {
            if u.owner[i] == cp && u.home[i] >= 1 && (u.home[i] as usize) <= n {
                self.counts[(u.home[i] - 1) as usize] += 1;
            }
        }
        let mut unlocked = r.unit_static;
        let mut known = crate::rules::bits128(&s.players.tech[p]);
        while known != 0 {
            unlocked |= r.units_by_tech[known.trailing_zeros() as usize];
            known &= known - 1;
        }
        self.ok = unlocked & r.unit_cost_le.le(s.players.stars[p]);
    }
}

/// One row of military.summon_mask as a bitset of unit kinds.
#[inline]
pub fn summon_row(s: &State, ctx: &SummonCtx, t: usize) -> u128 {
    let (u, c) = (&s.units, &s.cities);
    let cp = s.current_player;
    let available = c.owner[t] == cp && ctx.counts[t] <= c.level[t] && u.owner[t] == 0;
    let friendly = s.tiles.owner[t] == cp || (s.tiles.owner[t] > 0 && s.players.peace[s.p()][s.pclip(s.tiles.owner[t])]);
    let upgrade_allowed = u.owner[t] == cp && u.kind[t] == 43 && friendly;
    (if available { ctx.ok & !UPGRADES } else { 0 }) | (if upgrade_allowed { ctx.ok & UPGRADES } else { 0 })
}

pub fn summon_mask(s: &State, r: &Rules, out: &mut [bool]) {
    let mut ctx = SummonCtx::new(s.n);
    ctx.prepare(s, r);
    for t in 0..s.n {
        let bits = summon_row(s, &ctx, t);
        for k in 0..128 {
            out[t * 128 + k] = bits >> k & 1 == 1;
        }
    }
}

pub fn summon(s: &mut State, source: usize, kind: usize, r: &Rules, g: &Geom, costs: bool) {
    let n = s.n;
    let cp = s.current_player;
    s.units.clear(source);
    s.units.owner[source] = cp;
    s.units.kind[source] = kind as i32;
    s.units.health[source] = r.unit_health[kind];
    s.units.home[source] = source as i32 + 1;
    s.units.moved[source] = true;
    s.units.attacked[source] = true;
    s.units.prev_x[source] = -1;
    s.units.prev_y[source] = -1;
    s.units.created[source] = s.turn;
    let p = s.p();
    s.players.stars[p] -= r.unit_cost[kind] * costs as i32;
    s.players.score[p] = s.players.score[p].wrapping_add(5 * r.unit_cost[kind]);
    // training explores around the new unit with its sight range (Scout units see 2 tiles)
    let radius = if s.tiles.terrain[source] == 4 || r.skills[kind][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(source, t) <= radius).collect();
    discover(s, &reveal, cp);
}

/// The unit on `i` may be promoted: 3 kills, not yet a veteran, not Static;
/// a boat is judged by its passenger, so a non-Static passenger is promoted at sea.
#[inline]
pub fn promotable(s: &State, r: &Rules, i: usize) -> bool {
    let u = &s.units;
    let judged = if u.passenger[i] > 0 { u.passenger[i] } else { u.kind[i] } as usize;
    !u.veteran[i] && u.kills[i] >= 3 && !r.stat[judged]
}

pub fn promote(s: &mut State, source: usize, r: &Rules) {
    let hp = max_health(s, r, source) + 5.0;
    s.units.veteran[source] = true;
    s.units.health[source] = hp;
}

pub fn disband(s: &mut State, source: usize, r: &Rules) {
    let p = s.p();
    disband_of(s, source, r, p);
}

/// Disband the unit on `source`, refunding half its cost to seat index `p` (its owner).
pub fn disband_of(s: &mut State, source: usize, r: &Rules, p: usize) {
    let kind = s.units.kind[source];
    let passenger = s.units.passenger[source];
    let underlying = if passenger > 0 {
        passenger
    } else if kind == 39 {
        38
    } else if kind == 40 {
        37
    } else if kind == 45 {
        12
    } else {
        kind
    };
    let refund = r.unit_cost[underlying as usize].div_euclid(2);
    remove(s, source, r, None, false);
    s.players.stars[p] += refund;
}

pub fn heal_others(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let cp = s.current_player;
    for i in 0..s.n {
        if g.d(source, i) == 1 && s.units.owner[i] == cp {
            if !s.units.poison[i] {
                let mh = max_health(s, r, i);
                s.units.health[i] = (s.units.health[i] + 4.0).min(mh);
            }
            s.units.poison[i] = false;
        }
    }
    exhaust(s, source);
}

fn sign(v: i32) -> i32 {
    v.signum()
}

/// military.push: reference alternating push directions with its centre tie breaks.
pub fn push(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let (n, size) = (s.n, s.size);
    let sz = size as i32;
    let u = &s.units;
    let owner = u.owner[source];
    if owner <= 0 {
        return;
    }
    let (x, y) = ((source % size) as i32, (source / size) as i32);
    let center = sz / 2;
    let cx = sign(center - x);
    let mut cy = sign(center - y);
    if cx == 0 && cy == 0 {
        cy = 1;
    }
    let (px, py) = (u.prev_x[source], u.prev_y[source]);
    let prev_valid = px >= 0 && py >= 0 && (px != x || py != y);
    let flip = if owner != s.current_player { -1 } else { 1 };
    let (mx, my) = (sign(x - px) * flip, sign(y - py) * flip);
    let la = u.last_attack[source] - 1;
    let kind = u.kind[source] as usize;
    let ranged = r.unit_range[kind] > 1 && u.attacked[source] && !(r.escape[kind] && u.moved[source]);
    let (ax, ay) = (sign(la.rem_euclid(sz) - x) * flip, sign(la.div_euclid(sz) - y) * flip);
    let use_move = prev_valid && (la < 0 || u.moved[source]);
    let dx = if la >= 0 && ranged { ax } else if use_move { mx } else { cx };
    let dy = if la >= 0 && ranged { ay } else if use_move { my } else { cy };
    let dirs: [[i32; 2]; 8] = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
    let base = dirs.iter().position(|d| d[0] == dx && d[1] == dy).unwrap_or(0) as i32;
    let offsets = [0i32, -1, 1, -2, 2, -3, 3, 4];
    let mut choices: [[i32; 2]; 8] = [[0; 2]; 8];
    for k in 0..8 {
        let d = dirs[(base + offsets[k]).rem_euclid(8) as usize];
        choices[k] = [d[0] + x, d[1] + y];
    }
    for (a, b) in [(1usize, 2usize), (3, 4), (5, 6)] {
        let dista = (choices[a][0] - center).pow(2) + (choices[a][1] - center).pow(2);
        let distb = (choices[b][0] - center).pow(2) + (choices[b][1] - center).pow(2);
        let va = (choices[a][0] - choices[a][1]).abs();
        let vb = (choices[b][0] - choices[b][1]).abs();
        if distb < dista || (distb == dista && vb < va) {
            choices.swap(a, b);
        }
    }
    crate::cov::hit(29);
    let po = s.pclip(owner);
    let mut target = None;
    let mut first_index = 0usize;
    for (k, ch) in choices.iter().enumerate() {
        let idx = (ch[1] * sz + ch[0]).clamp(0, n as i32 - 1) as usize;
        if k == 0 {
            first_index = idx;
        }
        let in_board = ch[0] >= 0 && ch[0] < sz && ch[1] >= 0 && ch[1] < sz;
        let terrain = s.tiles.terrain[idx];
        let st = s.tiles.structure[idx];
        let mut valid = in_board && u.owner[idx] == 0;
        valid = valid && (terrain == 3 || terrain == 5 || (terrain == 4 && s.players.tech[po][20])
            || (terrain <= 2 && (r.skills[kind][FLOAT] || st == 48 || (st == 8 && s.tiles.owner[idx] == owner))));
        if valid {
            target = Some(idx);
            break;
        }
    }
    let _ = first_index;
    match target {
        Some(t) => relocate(s, source, t, r, g, true, None),
        None => remove(s, source, r, None, true),
    }
}

pub fn superunit(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    push(s, source, r, g);
    summon(s, source, 12, r, g, false);
}

pub fn area_damage(s: &mut State, source: usize, center: usize, r: &Rules, g: &Geom, stomp: bool) {
    let n = s.n;
    let src_owner = s.units.owner[source];
    let src_kind = s.units.kind[source] as usize;
    let hp_s = max_health(s, r, source);
    let ah = s.units.health[source];
    let aa = r.unit_attack[src_kind];
    let mut damage = vec![0f32; n];
    let mut dead = vec![false; n];
    let mut any_hit = false;
    for i in 0..n {
        let u = &s.units;
        let targets = g.d(center, i) == 1 && u.owner[i] != 0 && u.owner[i] != src_owner && !protected(s, src_owner, u.owner[i]);
        if targets {
            let dd = r.unit_defense[u.kind[i] as usize] * if u.poison[i] { 0.7 } else { 1.0 };
            let mut dmg = combat(aa, ah, hp_s, dd, u.health[i], max_health(s, r, i), defense_bonus(s, r, i)).2;
            let _ = stomp;
            dmg = dmg.floor();
            damage[i] = dmg;
            dead[i] = u.health[i] - dmg <= 0.0;
            any_hit |= dmg > 0.0;
        }
    }
    for i in 0..n {
        s.units.health[i] -= damage[i];
    }
    for i in 0..n {
        if dead[i] {
            crate::cov::hit(37);
            remove(s, i, r, Some(source), true);
        }
    }
    if any_hit {
        hostile(s);
    }
}

pub fn upgrade(s: &mut State, source: usize, kind: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let before = max_health(s, r, source);
    let old = s.units.kind[source] as usize;
    s.units.kind[source] = kind as i32;
    let hp = (max_health(s, r, source) - (before - s.units.health[source])).max(1.0);
    s.units.health[source] = hp;
    let p = s.p();
    s.players.stars[p] -= r.unit_cost[kind];
    s.players.score[p] = s.players.score[p].wrapping_add(5 * (r.unit_cost[kind] - r.unit_cost[old]));
    let radius = if r.skills[kind][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(source, t) <= radius).collect();
    discover(s, &reveal, s.current_player);
}
