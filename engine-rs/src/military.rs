//! gpu/military.py: unit actions (native profile).

use crate::diplomacy::{hostile, protected};
use crate::economy::discover;
use crate::geom::Geom;
use crate::kernels::{combat, max_health};
use crate::rules::{Rules, CARRY, FLOAT, HIDE, INDEPENDENT, SCOUT, SPLASH, STOMP, SURPRISE};
use crate::state::State;

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
    let loss = -5 * value;
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
    relocate_as(s, source, target, r, g, involuntary, reveal, true);
}

/// relocate; `stomp` false for a forced move (a push), which skips the arrival abilities.
#[allow(clippy::too_many_arguments)]
fn relocate_as(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom, involuntary: bool, reveal: Option<&[bool]>,
               stomp: bool) {
    let (n, size) = (s.n, s.size);
    let owner = s.units.owner[source];
    let kind = s.units.kind[source] as usize;
    let passenger_src = s.units.passenger[source];
    let moved_src = s.units.moved[source];
    let attacked_src = s.units.attacked[source];
    let invisible_src = s.units.invisible[source];
    let terrain_src = s.tiles.terrain[source];
    s.units.move_unit(source, target);
    s.units.moved[target] = true;
    s.units.attacked[target] = true;
    s.units.prev_x[target] = (source % size) as i32;
    s.units.prev_y[target] = (source / size) as i32;
    // a move faces the unit along the path's last step (core::move_adjacent refines multi-tile moves)
    s.units.direction[target] = direction(size, source, target);
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
    if stomp && r.skills[kind][STOMP] {
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
    // embarking and disembarking set moved and attacked on the new unit: no Dash after either
    let dash = !involuntary && !moved_src && !attacked_src && r.dash[new_kind] && !embark && !disembark;
    s.units.kind[target] = new_kind as i32;
    s.units.passenger[target] = passenger;
    s.units.attacked[target] = if dash { false } else { s.units.attacked[target] || disembark };
    if embark && r.skills[new_kind][STOMP] && !r.skills[kind][STOMP] {
        // embarking: a new boat with Stomp (Giant -> Juggernaut) stomps where it embarked
        area_damage(s, target, target, r, g, true);
    }
    let hidden = r.skills[new_kind][HIDE] && !involuntary;
    s.units.invisible[target] |= hidden;
    s.units.attacked[target] |= hidden && !invisible_src;
    let o = s.pwrap(owner);
    s.players.score[o] = s.players.score[o].wrapping_add(5 * (score_after - score_before));
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
                    && !(u.owner[t] != 0 && u.owner[t] != s.cities.owner[t]) && !s.cities.riot[t]
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
    let dd = r.unit_defense[defender];
    let (damage, retaliation, _) = combat(r.unit_attack[kind], s.units.health[source], hp_s, dd, s.units.health[target], hp_t, bonus);
    let dead = s.units.health[target] - damage <= 0.0;
    // retaliation: the attacker must also be one of the defender's attack
    // options, i.e. stand on a tile the defender's owner has explored
    let seen = s.tiles.explored[s.pclip(s.units.owner[target])][source];
    let retaliate = !r.skills[kind][SURPRISE] && !r.stiff[defender] && g.d(source, target) <= r.unit_range[defender] && seen;
    // the pacifist task resets on damage dealt to a visible unit
    if damage > 0.0 && !s.units.invisible[target] {
        s.players.attacked[s.p()] = true;
    }
    // an attack: both units face each other
    s.units.direction[source] = direction(size, source, target);
    s.units.direction[target] = direction(size, target, source);
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
            // (attack options include hidden units when range <= sight: hidden and unexplored enemies count)
            let mut more = false;
            attack_targets(s, at, r, g, |_| more = true);
            let sight = 1 + r.skills[kind][SCOUT] as i32;
            if !more && r.unit_range[kind] <= sight {
                let owner = s.units.owner[at];
                more = (0..s.n).any(|t| t != at && g.d(at, t) <= r.unit_range[kind] && State::foreign(s.units.owner[t], owner)
                    && !protected(s, owner, s.units.owner[t]));
            }
            s.units.attacked[at] = !more;
        }
    }
}

pub fn convert(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let cp = s.current_player;
    let old = s.pwrap(s.units.owner[target]);
    let tk = s.units.kind[target] as usize;
    // conversion: rehomed to the converter's current capital (none: no home)
    let home = if r.skills[tk][INDEPENDENT] { 0 } else { crate::structures::current_capital(s, cp).map_or(0, |c| c as i32 + 1) };
    let passenger = s.units.passenger[target];
    let cost = r.unit_cost[tk] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    // the full unit score moves from the old owner to the converter, every time
    s.players.score[old] = s.players.score[old].wrapping_sub(5 * cost);
    let cpi = s.p();
    s.players.score[cpi] = s.players.score[cpi].wrapping_add(5 * cost);
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
        let p = s.p();
        self.counts.fill(0);
        for i in 0..n {
            // a city's unit count: every unit homed at the city counts, whoever owns it now
            if u.home[i] >= 1 && (u.home[i] as usize) <= n {
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
    s.units.direction[source] = spawn_direction(s.size, source);
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

/// Disband the unit on `source`, refunding half its full cost (boat plus passenger) to seat
/// index `p` (its owner).
pub fn disband_of(s: &mut State, source: usize, r: &Rules, p: usize) {
    let kind = s.units.kind[source] as usize;
    let passenger = s.units.passenger[source];
    let full = r.unit_cost[kind] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    remove(s, source, r, None, false);
    s.players.stars[p] += full.div_euclid(2);
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

/// Grid directions: SW, W, NW, N, NE, E, SE, S as (dx, dy) in game coordinates (y is the row).
pub const GRID: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)];
/// The game's tile-neighbour order (x outer).
const STEPS: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];

/// Direction of a distance vector: the direction with the largest weighted dot product with the
/// vector (axial 1393, diagonal 985); SW when none is positive.
pub fn vector_direction(vx: i32, vy: i32) -> i32 {
    let (mut best, mut score) = (0, 0);
    for (d, &(x, y)) in GRID.iter().enumerate() {
        let v = (y * vy + x * vx) * if x * x + y * y == 1 { 1393 } else { 985 };
        if score < v {
            best = d as i32;
            score = v;
        }
    }
    best
}

/// Direction from `from` to `to`: a neighbour's direction (8, none, for the same tile), farther tiles by
/// vector_direction.
pub fn direction(size: usize, from: usize, to: usize) -> i32 {
    let (dx, dy) = ((to % size) as i32 - (from % size) as i32, (to / size) as i32 - (from / size) as i32);
    if dx.abs().max(dy.abs()) > 1 {
        return vector_direction(dx, dy);
    }
    GRID.iter().position(|&d| d == (dx, dy)).map_or(8, |d| d as i32)
}

/// A new unit faces the map centre (width / 2, height / 2).
pub fn spawn_direction(size: usize, tile: usize) -> i32 {
    let c = (size / 2) as i32;
    vector_direction(c - (tile % size) as i32, c - (tile / size) as i32)
}

/// The game's one-step route to a neighbour `dest` of the unit on `source`: the unit may finish on the tile
/// and the tile is passable under the unit's route settings (enemy territory allowed, occupied tiles not).
fn push_reachable(s: &State, source: usize, dest: usize, r: &Rules) -> bool {
    let (t, u) = (&s.tiles, &s.units);
    let owner = u.owner[source];
    let p = s.pclip(owner);
    let kind = u.kind[source] as usize;
    if u.owner[dest] != 0 || !t.explored[p][dest] {
        return false;
    }
    let (terrain, structure) = (t.terrain[dest], t.structure[dest]);
    if structure == 1 && State::foreign(t.owner[dest], owner) {
        // not a peace (or cool-down) partner's city, and no Infiltrate unit into an opponent's city
        let q = s.pclip(t.owner[dest]);
        if s.players.peace[p][q] || s.players.truce_breaker[p][q] != 0 || r.skills[kind][crate::rules::INFILTRATE] {
            return false;
        }
    }
    if structure == 48 {
        return !crate::kernels::bridge_blocks(s, owner, source, dest);
    }
    if structure == 1 {
        return true;
    }
    let tech = &s.players.tech[p];
    let terrain_ok = (terrain != 1 || tech[10]) && (terrain != 2 || tech[13] || tech[25]) && (terrain != 4 || tech[20]);
    let land_access = (terrain != 1 && terrain != 2) || (structure == 8 && t.owner[dest] == owner);
    terrain_ok && (land_access || r.skills[kind][FLOAT])
}

/// Pushing a unit off a tile, killing it when that fails (training on an occupied tile): moves the unit on
/// `source` off it for a unit the current player trains there. The occupant's facing (reversed for another player's
/// unit) is tried first, then the directions -1, +1, -2, +2, -3, +3, -4 from it; a boat on water keeps the first land
/// tile as its last resort. The push is a forced move: no stomp, and moved/attacked stay as they were
/// unless the unit embarks or disembarks (those actions exhaust it). Deterministic, no hash draw.
pub fn push(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let size = s.size as i32;
    let owner = s.units.owner[source];
    if owner <= 0 {
        return;
    }
    let mut facing = s.units.direction[source];
    if owner != s.current_player {
        facing = if facing < 4 { facing + 4 } else { facing - 4 };
    }
    let kind = s.units.kind[source];
    let vehicle = r.skills[kind as usize][CARRY];
    let water = |t: usize| s.tiles.terrain[t] == 1 || s.tiles.terrain[t] == 2;
    let (x, y) = (source as i32 % size, source as i32 / size);
    crate::cov::hit(29);
    let (mut target, mut fallback) = (None, None);
    for k in 1..=8i32 {
        let offset = if (k - 1) % 2 == 0 { k / 2 } else { -(k >> 1) };
        let (dx, dy) = GRID[(offset + facing + 8).rem_euclid(8) as usize];
        let (tx, ty) = (x + dx, y + dy);
        if tx < 0 || ty < 0 || tx >= size || ty >= size {
            continue;
        }
        let t = (ty * size + tx) as usize;
        if s.units.owner[t] != 0 {
            continue;
        }
        if water(source) && !water(t) && vehicle && fallback.is_none() {
            fallback = Some(t);
            continue;
        }
        if push_reachable(s, source, t, r) {
            target = Some(t);
            break;
        }
    }
    match target.or(fallback.filter(|&t| push_reachable(s, source, t, r))) {
        Some(t) => {
            let (moved, attacked) = (s.units.moved[source], s.units.attacked[source]);
            relocate_as(s, source, t, r, g, true, None, false);
            if s.units.kind[t] == kind {
                s.units.moved[t] = moved;
                s.units.attacked[t] = attacked;
            }
        }
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
    let mut pacifist_reset = false;
    for i in 0..n {
        let u = &s.units;
        let targets = g.d(center, i) == 1 && u.owner[i] != 0 && u.owner[i] != src_owner && !protected(s, src_owner, u.owner[i]);
        if targets {
            let dd = r.unit_defense[u.kind[i] as usize];
            let mut dmg = combat(aa, ah, hp_s, dd, u.health[i], max_health(s, r, i), defense_bonus(s, r, i)).2;
            let _ = stomp;
            dmg = dmg.floor();
            damage[i] = dmg;
            dead[i] = u.health[i] - dmg <= 0.0;
            any_hit |= dmg > 0.0;
            pacifist_reset |= dmg > 0.0 && !u.invisible[i];
        }
    }
    // area damage: one attack from the centre per target, so each target faces the centre; a stomper
    // ends facing the first target in tile-neighbour (x-outer) order, whose attack runs last
    let size = s.size;
    let mut first = None;
    for (dx, dy) in STEPS {
        let (x, y) = ((center % size) as i32 + dx, (center / size) as i32 + dy);
        if x >= 0 && y >= 0 && (x as usize) < size && (y as usize) < size {
            let i = y as usize * size + x as usize;
            let u = &s.units;
            if u.owner[i] != 0 && u.owner[i] != src_owner && !protected(s, src_owner, u.owner[i]) {
                first = first.or(Some(i));
                s.units.direction[i] = direction(size, i, center);
            }
        }
    }
    if let (true, Some(i)) = (stomp, first) {
        s.units.direction[center] = direction(size, center, i);
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
    if pacifist_reset {
        s.players.attacked[s.p()] = true;
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
