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
    s.units.move_lock[source] = false;
    // make exhausted removes double ready
    s.units.double_ready[source] = false;
}

/// make exhauseted (the city capture, ruins, clear tile effect by a unit): moved and
/// attacked, double ready and Boosted removed.
pub fn make_exhausted(s: &mut State, source: usize) {
    exhaust(s, source);
    s.units.boosted[source] = false;
    // then disable followers: every follower down the chain is moved and attacked (Cymanti)
    let mut f = s.units.follower[source];
    let mut guard = 0;
    while f > 0 && guard < s.n {
        let i = (f - 1) as usize;
        s.units.moved[i] = true;
        s.units.attacked[i] = true;
        s.units.move_lock[i] = false;
        f = s.units.follower[i];
        guard += 1;
    }
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
    // attack action (game version >= 95): no xp for a killer that is a vehicle (Carry: every
    // boat) or Static; the passenger of a boat keeps the xp it had when it embarked
    let xp = killer.is_some() && !(r.skills[u.kind[k] as usize][CARRY] || r.stat[u.kind[k] as usize]);
    // kill unit (Cymanti): a follower becomes a Centipede head, a leader
    // loses the unit (disband unit action and explode unit action unlink first and handle their chains themselves)
    crate::cymanti::kill_links(s, target, r);
    if s.units.poison[target] {
        // kill unit: a Poisoned unit leaves Spores (spawn spore resource, Cymanti)
        crate::cymanti::spawn_spores(s, target);
    }
    s.units.kills[k] += xp as i32;
    s.units.clear(target);
}

pub fn relocate(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom, involuntary: bool, reveal: Option<&[bool]>) {
    relocate_as(s, source, target, r, g, involuntary, reveal, true);
}

/// relocate; `stomp` false for a move action with move reason 2 (a push), which skips the arrival abilities.
#[allow(clippy::too_many_arguments)]
fn relocate_as(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom, involuntary: bool, reveal: Option<&[bool]>,
               stomp: bool) {
    relocate_one(s, source, target, r, g, involuntary, reveal, stomp);
    // move action (version > 119; Cymanti): the follower, when 2 or more tiles away
    // from the new tile, gets move action([leader's origin, follower], reason Command); its own move pulls its follower
    if s.units.follower[target] > 0 {
        let f = (s.units.follower[target] - 1) as usize;
        if g.d(f, target) >= 2 && s.units.owner[source] == 0 {
            relocate_as(s, f, source, r, g, false, None, true);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn relocate_one(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom, involuntary: bool, reveal: Option<&[bool]>,
                stomp: bool) {
    let (n, size) = (s.n, s.size);
    let owner = s.units.owner[source];
    let kind = s.units.kind[source] as usize;
    let passenger_src = s.units.passenger[source];
    let moved_src = s.units.moved[source];
    let attacked_src = s.units.attacked[source];
    let invisible_src = s.units.invisible[source];
    let terrain_src = s.tiles.terrain[source];
    let bubble_src = s.units.bubble[source];
    s.units.move_unit(source, target);
    // Cymanti: move action's check step on poison, any reason
    crate::cymanti::step_on_poison(s, target, r);
    // Aquarion: the Tentacle trap of every move action, any reason (its attack action is pushed now, with the
    // battle as it stands, and runs after the reason-gated arrival actions)
    let trap = crate::aquarion::tentacle_trap(s, target, owner, r);
    s.units.moved[target] = true;
    s.units.attacked[target] = true;
    s.units.prev_x[target] = (source % size) as i32;
    s.units.prev_y[target] = (source / size) as i32;
    // move action: set unit direction from the path's last step (core:move_adjacent refines multi-tile moves)
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
    if stomp && r.algae_trail[kind] {
        // Cymanti: the Island's Algae trail on the path's ends (core:move_adjacent adds the tiles between); its
        // Builds are pushed after the Stomp's attacks, so they run first
        crate::cymanti::algae_trail(s, &[target, source]);
    }
    if stomp && (r.skills[kind][STOMP] || r.tentacles[kind]) {
        // Stomp, and a Tentacles unit's area damage on arrival (Aquarion)
        crate::cov::hit(28);
        area_damage(s, target, target, r, g, true);
    }
    let tt = s.tiles.terrain[target];
    // will unit embark (v >= 120): a land unit embarks on a friendly Port on water or on a wetland (Flooded)
    // tile (a Port on plain Ice is land, Polaris), when the player can embark (Fishing: the only Port tech); a
    // vehicle disembarks onto a City or onto land that is not wetland. A Port on water is reached
    // only through movement_flags' MV_OWN_PORT (friendly, Fishing); on Flooded land any unit may step onto it.
    let flooded_t = s.tiles.flooded[target];
    // a flyer never embarks (will unit embark: a land unit is neither aquatic nor Fly)
    // Cymanti: an Algae water tile is entered without the Port (a friendly Port and can player embark are checked
    // here for it, as for a Flooded one)
    let embark = s.tiles.structure[target] == 8 && !r.skills[kind][FLOAT] && !r.fly[kind] && ((tt <= 2 && !s.tiles.algae[target])
        || ((flooded_t || s.tiles.algae[target]) && {
        let (o, to) = (s.pclip(owner), s.tiles.owner[target]);
        crate::tribes::can_embark(s.players.tribe[o], &s.players.tech[o]) && (to == owner || (to > 0 && s.players.peace[o][s.pclip(to)]))
    }));
    let landing = tt > 2 && (!flooded_t || s.tiles.structure[target] == 1);
    let special_land = (kind == 39 || kind == 40 || kind == 45) && landing;
    let disembark = r.skills[kind][CARRY] && (passenger_src > 0 || special_land) && landing && !embark;
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
    // embark action / disembark action set moved and attacked on the new unit: no Dash after either
    // move action (v > 101): a Skate unit that moves off the ice for any reason but a push is done
    // for the turn (moved and attacked)
    let skate_off = stomp && r.skate[kind] && tt != crate::kernels::ICE;
    let dash = !involuntary && !moved_src && !attacked_src && r.dash[new_kind] && !embark && !disembark && !skate_off;
    s.units.kind[target] = new_kind as i32;
    s.units.passenger[target] = passenger;
    s.units.attacked[target] = if dash { false } else { s.units.attacked[target] || disembark };
    if embark && r.skills[new_kind][STOMP] && !r.skills[kind][STOMP] {
        // embark action: a new boat with Stomp (Giant -> Juggernaut) stomps where it embarked
        area_damage(s, target, target, r, g, true);
    }
    let hidden = r.skills[new_kind][HIDE] && !involuntary;
    s.units.invisible[target] |= hidden;
    s.units.attacked[target] |= hidden && !invisible_src;
    // the move alone set `attacked` (no Dash; the unit had not attacked; not an embark/disembark/hide, which set the
    // game's own attacked): an upgrade to a Dash type may lift it (Units::move_lock)
    s.units.move_lock[target] = s.units.attacked[target] && !attacked_src && !embark && !disembark && !(hidden && !invisible_src)
        && !skate_off;
    let o = s.pwrap(owner);
    s.players.score[o] = s.players.score[o].wrapping_add(5 * (score_after - score_before));
    if disembark {
        grow_on_landing(s, target, r, g);
    }
    // Aquarion: Bubble loss for any move (a push too), AutoFlood (Crab) for any but a push
    let flooded_at_move = crate::aquarion::after_move(s, target, bubble_src, owner, r, g, stomp);
    if stomp {
        crate::polaris::auto_freeze(s, target, new_kind, r, g);
    }
    if let Some((jelly, damage)) = trap {
        crate::aquarion::trap_attack(s, jelly, target, damage, r);
    }
    // the move's own check bubble step, pushed first, runs last
    crate::aquarion::bubble_on_arrival(s, target, bubble_src, flooded_at_move, owner, r);
}

/// The land unit a boat lands as: its passenger, or the land form our engine keeps implicit for a Dinghy (Cloak),
/// Pirate (Dagger) and Juggernaut (Giant), as relocate's landing does.
#[inline]
fn landed_kind(kind: usize, passenger: i32) -> usize {
    match kind {
        39 => 38,
        40 => 37,
        45 => 12,
        _ => passenger as usize,
    }
}

/// can disembark (disembark command, v > 104): a boat with a passenger standing on a wetland
/// (Flooded) tile or on a Bridge (improvement ability bridge). No
/// moved, attacked or owner check of its own: command utils lists unit commands for the mover's units only.
#[inline]
pub fn can_disembark(s: &State, r: &Rules, i: usize) -> bool {
    let kind = s.units.kind[i].clamp(0, 127) as usize;
    let carries = s.units.passenger[i] > 0 || matches!(kind, 39 | 40 | 45);
    s.units.owner[i] != 0 && r.skills[kind][CARRY] && carries && (s.tiles.structure[i] == 48 || s.wetland(i))
}

/// disembark action (v >= 109) for a boat staying on its tile: the passenger takes the boat's place
/// with the boat's health (no boat of these rules has Protect), its facing (direction and flip copied) and
/// moved = attacked = true; decrease score action takes the boat's part of the score (the amounts of a landing move).
/// No exploration and no arrival ability (no stomp, no hiding); a Grow passenger may land grown (grow_on_landing).
pub fn disembark(s: &mut State, i: usize, r: &Rules, g: &Geom) {
    let kind = s.units.kind[i] as usize;
    let passenger = s.units.passenger[i];
    let new_kind = landed_kind(kind, passenger);
    let before = r.unit_cost[kind] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    let after = r.unit_cost[new_kind];
    s.units.kind[i] = new_kind as i32;
    s.units.passenger[i] = 0;
    s.units.moved[i] = true;
    s.units.attacked[i] = true;
    s.units.move_lock[i] = false;
    let o = s.pwrap(s.units.owner[i]);
    s.players.score[o] = s.players.score[o].wrapping_add(5 * (after - before));
    grow_on_landing(s, i, r, g);
}

/// Elyrion, disembark action's Grow branch: a Grow passenger older than 2 turns
/// (a literal 2, not growth rate) lands grown; get unlocked upgrades for unit is never empty for it (Basic lists the
/// dragons for every tribe). `created` rides through the boat: an embark keeps it.
fn grow_on_landing(s: &mut State, i: usize, r: &Rules, g: &Geom) {
    let kind = s.units.kind[i] as usize;
    if r.grow[kind] && r.grows_into[kind] >= 0 && s.turn - s.units.created[i] > 2 {
        grow(s, i, r, g);
    }
}

/// Elyrion upgrade action (cost 0) of a Grow unit into grows_into: the new type
/// with the old damage (health = new max - (old max - health), at least 1), created this turn, no xp or promotion,
/// home / moved / attacked / effects / facing kept; score + 5 x (cost difference) (0 for the dragons); explore from tile
/// with the new unit's sight.
pub fn grow(s: &mut State, i: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let into = r.grows_into[s.units.kind[i] as usize] as usize;
    let old = s.units.kind[i] as usize;
    let damage = max_health(s, r, i) - s.units.health[i];
    s.units.kind[i] = into as i32;
    s.units.veteran[i] = false;
    s.units.kills[i] = 0;
    s.units.created[i] = s.turn;
    let new_max = r.unit_health[into];
    // health in tenths: at least 1 (0.1 here)
    s.units.health[i] = (new_max - damage).max(0.1);
    let owner = s.units.owner[i];
    let o = s.pwrap(owner);
    s.players.score[o] = s.players.score[o].wrapping_add(5 * (r.unit_cost[into] - r.unit_cost[old]));
    let radius = if s.tiles.terrain[i] == 4 || r.skills[into][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(i, t) <= radius).collect();
    discover(s, &reveal, owner);
    // upgrade action's train unit: check step on poison (its poison unit action runs after the effects are copied)
    crate::cymanti::step_on_poison(s, i, r);
}

pub fn attack_mask(s: &State, source: usize, r: &Rules, g: &Geom, out: &mut [bool]) {
    out[..s.n].fill(false);
    attack_targets(s, source, r, g, |t| out[t] = true);
}

/// attack command against a Drench target without an attackable defender (no unit, or one with an active peace
/// treaty: own units included): no diplomacy change, zero battle results (no damage,
/// retaliation or Pacifist reset); the attacker is marked moved (no Escape: Aquapult) and attacked; the attack action
/// with damage 0 turns both units, takes the target's Boosted and Bubble, and its perform attack floods the tile.
fn drench_null(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom) {
    let size = s.size;
    let kind = s.units.kind[source] as usize;
    s.units.direction[source] = direction(size, source, target);
    s.units.prev_x[source] = 2 * (source % size) as i32 - (target % size) as i32;
    s.units.prev_y[source] = 2 * (source / size) as i32 - (target / size) as i32;
    s.units.last_attack[source] = target as i32 + 1;
    let had_bubble = s.units.owner[target] != 0 && s.units.bubble[target];
    if s.units.owner[target] != 0 {
        s.units.direction[target] = direction(size, target, source);
        s.units.boosted[target] = false;
    }
    if r.drench[kind] && crate::aquarion::floodable(s, target) {
        crate::aquarion::flood_tile(s, target, s.units.owner[source], r, g);
    }
    // perform attack: the defender loses Swift (Cymanti)
    s.units.swift[target] = false;
    if had_bubble {
        s.units.bubble[target] = false;
    }
    s.units.attacks_performed[source] += 1;
    s.units.attacked[source] = true;
    s.units.move_lock[source] = false;
    s.units.moved[source] = !r.escape[kind];
}

/// The true entries of attack_mask's row for `source`, visiting only the tiles within the unit's range.
pub fn attack_targets(s: &State, source: usize, r: &Rules, g: &Geom, mut emit: impl FnMut(usize)) {
    let u = &s.units;
    let cp = s.current_player;
    let kind = u.kind[source] as usize;
    let own = u.owner[source] == cp && !u.attacked[source] && u.health[source] > 0.0;
    let mut can = own && (!u.moved[source] || r.dash[kind]) && r.unit_attack[kind] >= 0.0;
    let infil = r.skills[kind][crate::rules::INFILTRATE];
    // get attack options: none for a unit with a leader (a Segment; Cymanti)
    can = can && u.leader[source] == 0;
    // get attack options: 0 attack needs Convert (the Mind Bender, the Siren) or Infiltrate
    can = can && (r.unit_attack[kind] > 0.0 || r.convert[kind] || infil);
    let range = r.unit_range[kind];
    if !can || range < 1 {
        return;
    }
    let drench = r.drench[kind];
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
            } else if drench && crate::aquarion::floodable(s, t) {
                // get attack options at position: every floodable tile is a Drench target,
                // whatever stands there
                true
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
    if s.units.owner[target] == 0 || protected(s, cp, s.units.owner[target]) {
        // only a Drench target: an empty or own / peace-protected floodable tile
        drench_null(s, source, target, r, g);
        return;
    }
    let had_bubble = s.units.bubble[target];
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
    // can defender retaliate: the attacker must also be one of the defender's attack
    // options, i.e. stand on a tile the defender's owner has explored
    let seen = s.tiles.explored[s.pclip(s.units.owner[target])][source];
    // can defender retaliate (v >= 120): never against a Freeze attacker, never by a Frozen defender
    let freeze = r.freeze[kind];
    let retaliate = !r.skills[kind][SURPRISE] && !r.stiff[defender] && g.d(source, target) <= r.unit_range[defender] && seen
        && !freeze && !s.units.frozen[target] && !r.convert[kind] && s.units.leader[target] == 0;
    // should reset pacifist task: damage dealt to a visible unit
    if (damage > 0.0 && !s.units.invisible[target]) || freeze {
        // a Freeze attack resets the Pacifist task in freeze unit action, visible or not
        s.players.attacked[s.p()] = true;
    }
    // attack action: both units face each other
    s.units.direction[source] = direction(size, source, target);
    s.units.direction[target] = direction(size, target, source);
    s.units.health[target] += -damage;
    s.units.boosted[target] = false;
    s.units.prev_x[source] = 2 * (source % size) as i32 - (target % size) as i32;
    s.units.prev_y[source] = 2 * (source / size) as i32 - (target / size) as i32;
    s.units.last_attack[source] = target as i32 + 1;
    if freeze {
        // attack command (v > 120) with Freeze: freeze unit action deals the damage and freezes the target (moved,
        // attacked, Frozen) before a kill removes it; no splash, no retaliation, no advance
        s.units.moved[target] = true;
        s.units.attacked[target] = true;
        s.units.move_lock[target] = false;
        s.units.frozen[target] = true;
    }
    // perform attack: a Poison attacker poisons a defender that survives (Cymanti)
    if !dead && r.poisons[kind] {
        s.units.poison[target] = true;
    }
    if r.skills[kind][SPLASH] && !freeze && r.poisons[kind] {
        // attack command: a Poison splasher (Exida) gives each non-peace neighbour of the target a poison unit action
        // only: no damage, no turn, no relations
        crate::cymanti::poison_splash(s, source, target, g);
    } else if r.skills[kind][SPLASH] && !freeze {
        crate::cov::hit(27);
        area_damage(s, source, target, r, g, false);
        // attack command pushes the retaliation attack action (Target -> Origin, when its damage is >= 1)
        // before the splash ones and the main attack; the action stack is LIFO,
        // so it runs last and the attacker ends facing the defender again
        if !dead && retaliate && retaliation > 0.0 {
            s.units.direction[source] = direction(size, source, target);
        }
    }
    // perform attack: the target loses its Bubble (pushed before a Drench attacker's
    // flood tile, so it runs after it: the flood tile's bubble check still sees it); a Drench attacker floods a
    // floodable target tile, before the kill unit and the retaliation run
    let drench = r.drench[kind] && crate::aquarion::floodable(s, target);
    if drench {
        crate::aquarion::flood_tile(s, target, s.units.owner[source], r, g);
    }
    // perform attack: the defender loses Swift (Cymanti)
    s.units.swift[target] = false;
    if had_bubble {
        s.units.bubble[target] = false;
    }
    if dead {
        remove(s, target, r, Some(source), true);
    } else {
        s.units.health[source] += -(if retaliate { retaliation } else { 0.0 });
        // the retaliation attack action (damage >= 1) is a perform attack on the attacker: it loses its Bubble
        if retaliate && retaliation > 0.0 {
            s.units.bubble[source] = false;
            s.units.swift[source] = false;
        }
    }
    let attacker_dead = s.units.health[source] <= 0.0;
    // the retaliation's perform attack: a Poison defender (Kiton, Exida, Phychi, Island) poisons the attacker it hits
    if !dead && !attacker_dead && retaliate && retaliation > 0.0 && r.poisons[defender] {
        s.units.poison[source] = true;
    }
    if attacker_dead {
        crate::cov::hit(34);
        remove(s, source, r, Some(target), true);
    }
    let st = s.tiles.structure[target];
    // get battle results: the advance follows get path with the unit's allowed terrain (water: Fishing,
    // ocean: Sailing or Free Diving, mountain: Climbing; a Bridge always), a land unit onto water only by an own Port
    let mine = &s.players.tech[s.p()];
    // a flyer enters any tile (is tile accessible: Fly before the terrain tests)
    let can_enter = if r.fly[kind] {
        true
    } else if r.skills[kind][FLOAT] {
        // a Swim unit: Water always, ocean when unlocked, land by its class
        st == 48 || crate::kernels::Swim::of(s, r, kind, cp).terrain_ok(s, mine, target)
    } else {
        let water_ok = crate::kernels::terrain_ok(mine, terrain);
        terrain == 3 || terrain == 5 || terrain == crate::kernels::ICE || (terrain == 4 && mine[20])
            || (terrain <= 2 && (st == 48 || s.tiles.algae[target] || (water_ok && st == 8 && s.tiles.owner[target] == cp
                && crate::tribes::can_embark(s.players.tribe[s.p()], mine))))
    };
    let advance = dead && !attacker_dead && r.unit_range[kind] < 2 && can_enter && !freeze;
    if advance {
        crate::cov::hit(35);
        relocate(s, source, target, r, g, true, None);
    }
    let at = if advance { target } else { source };
    // the advance embarked or disembarked (the unit changed type): the embark action / disembark action the advance's
    // move action pushes runs after attack command's own moved/attacked marks and sets both on the new unit
    // the advance's move action takes a Skate unit off the ice: done for the turn (move action v > 101, after the marks)
    let boarded = advance && (s.units.kind[at] != kind as i32 || (r.skate[kind] && s.tiles.terrain[at] != crate::kernels::ICE));
    if !attacker_dead {
        let persist = r.persist[kind];
        let escape = r.escape[kind];
        s.units.attacks_performed[at] += 1;
        // attack command: a double ready attacker loses it and stays able to attack (Cymanti,
        // the Phychi)
        let double = s.units.double_ready[at];
        s.units.double_ready[at] = false;
        s.units.attacked[at] = !double && !(dead && persist);
        s.units.move_lock[at] = false;
        // attack command: moved = !Escape; a Skate + Escape unit (BattleSled, IceFortress) is moved only when it ends
        // off the ice
        s.units.moved[at] = if escape && r.skate[kind] { s.tiles.terrain[at] != crate::kernels::ICE } else { !escape };
        if dead && persist {
            // attack command: Persist leaves the unit able to attack only when it has
            // another target from where it now stands
            // (get attack options at position with include hidden when range <= sight: hidden and unexplored enemies count)
            let mut more = false;
            attack_targets(s, at, r, g, |_| more = true);
            let sight = 1 + r.skills[kind][SCOUT] as i32;
            if !more && r.unit_range[kind] <= sight {
                let owner = s.units.owner[at];
                more = (0..s.n).any(|t| t != at && g.d(at, t) <= r.unit_range[kind] && State::foreign(s.units.owner[t], owner)
                    && !protected(s, owner, s.units.owner[t]));
            }
            s.units.attacked[at] = !more && !double;
        }
        if boarded {
            s.units.moved[at] = true;
            s.units.attacked[at] = true;
        }
    }
    // attack command: a Convert attacker with an attack (the Shaman, Cymanti) pushes convert action before the main
    // attack action when the damage leaves the defender alive: the damage lands, then the survivor changes sides
    // a kill converts nothing
    if r.convert[kind] && !dead && !attacker_dead && s.units.owner[target] != 0 && s.units.owner[target] != cp {
        convert(s, source, target, r, g);
    }
    // attack command: an Eat attacker that kills a defender pushes eat action for
    // the last unit of its follower chain; it runs after the kill and the advance (Cymanti)
    if r.eat[kind] && dead && !attacker_dead {
        let at = if advance { target } else { source };
        if s.units.owner[at] == cp {
            crate::cymanti::eat(s, crate::cymanti::tail(s, at), r, g);
        }
    }
}

pub fn convert(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom) {
    convert_action(s, source, target, r, g, true);
}

/// convert action; `attack`: the attack command's main attack action ran first (it
/// turns both units and takes the target's Bubble and Swift). (Cymanti) The target's follower and leader each get
/// their own convert action from the target's tile when another player's: a whole chain changes sides.
fn convert_action(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom, attack: bool) {
    let n = s.n;
    let cp = s.current_player;
    let old = s.pwrap(s.units.owner[target]);
    let tk = s.units.kind[target] as usize;
    // convert action: rehomed to the converter's current capital (none: no home)
    let home = if r.skills[tk][INDEPENDENT] { 0 } else { crate::structures::current_capital(s, cp).map_or(0, |c| c as i32 + 1) };
    let passenger = s.units.passenger[target];
    let cost = r.unit_cost[tk] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    // the full unit score moves from the old owner to the converter, every time
    s.players.score[old] = s.players.score[old].wrapping_sub(5 * cost);
    let cpi = s.p();
    s.players.score[cpi] = s.players.score[cpi].wrapping_add(5 * cost);
    // convert action bumps the Convert task while it is started (Rituals; Cymanti)
    if s.players.tech[cpi][49] {
        s.players.converts[cpi] += 1;
    }
    s.units.owner[target] = cp;
    s.units.converted[target] = true;
    s.units.home[target] = home;
    s.units.moved[target] = true;
    s.units.attacked[target] = true;
    s.units.move_lock[target] = false;
    // attack command also pushes the main attack action (Origin -> Target, damage = the converter's
    // attack damage, 0) after the convert action; it runs first and turns the two units
    // toward each other. No retaliation: can defender retaliate is false
    // against Convert.
    if attack {
        s.units.direction[source] = direction(s.size, source, target);
        s.units.direction[target] = direction(s.size, target, source);
        // that attack action's perform attack takes the target's Bubble and Swift
        s.units.bubble[target] = false;
        s.units.swift[target] = false;
        exhaust(s, source);
    } else {
        // the converter (the chain unit converted before) is marked moved (no Escape) and attacked
        s.units.moved[source] = !r.escape[s.units.kind[source].clamp(0, 127) as usize];
        s.units.attacked[source] = true;
        s.units.move_lock[source] = false;
    }
    let radius = if s.tiles.terrain[target] == 4 || r.skills[tk][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(target, t) <= radius).collect();
    discover(s, &reveal, cp);
    for link in [s.units.follower[target], s.units.leader[target]] {
        if link > 0 {
            let o = (link - 1) as usize;
            if s.units.owner[o] != 0 && s.units.owner[o] != cp {
                convert_action(s, target, o, r, g, false);
            }
        }
    }
}

/// out[t*128 + kind]
/// The Raft upgrades (Rammership, Scoutship, Bomber): only summonable onto an own Raft.
pub const UPGRADES: u128 = 1 << 41 | 1 << 42 | 1 << 44;

/// Per-call inputs of summon_mask: units per home city and the unlocked, affordable kinds.
pub struct SummonCtx {
    counts: Vec<i32>,
    ok: u128,
    /// The kinds with movement terrain (water, ocean, wetland: Jelly, Shark, Aquapult; Rules.water).
    water: u128,
}

impl SummonCtx {
    pub fn new(n: usize) -> SummonCtx {
        SummonCtx { counts: vec![0; n], ok: 0, water: 0 }
    }

    pub fn prepare(&mut self, s: &State, r: &Rules) {
        let n = s.n;
        let u = &s.units;
        let p = s.p();
        self.counts.fill(0);
        for i in 0..n {
            // get city unit count: every unit homed at the city counts, whoever owns it now
            if u.home[i] >= 1 && (u.home[i] as usize) <= n {
                self.counts[(u.home[i] - 1) as usize] += 1;
            }
        }
        let mut unlocked = r.unit_static_of(s.players.tribe[p]);
        let mut known = crate::rules::bits128(&s.players.tech[p]);
        let by_tech = r.units_by_tech_of(s.players.tribe[p]);
        while known != 0 {
            unlocked |= by_tech[known.trailing_zeros() as usize];
            known &= known - 1;
        }
        self.ok = unlocked & r.unit_cost_le.le(s.players.stars[p]);
        self.water = crate::rules::bits128(&r.water);
    }
}

/// has unit terrain for a movement terrain unit
/// (water, ocean, wetland): a neighbour of the city is water or wetland (Flooded).
#[inline]
fn wet_neighbour(s: &State, t: usize) -> bool {
    let size = s.size as i32;
    let (x, y) = ((t % s.size) as i32, (t / s.size) as i32);
    (-1..=1).any(|dy| (-1..=1).any(|dx| {
        let (nx, ny) = (x + dx, y + dy);
        if (dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= size || ny >= size {
            return false;
        }
        let k = (ny * size + nx) as usize;
        s.tiles.terrain[k] == 1 || s.tiles.terrain[k] == 2 || s.tiles.flooded[k]
    }))
}

/// One row of military.summon_mask as a bitset of unit kinds.
#[inline]
pub fn summon_row(s: &State, ctx: &SummonCtx, t: usize) -> u128 {
    let (u, c) = (&s.units, &s.cities);
    let cp = s.current_player;
    let available = c.owner[t] == cp && ctx.counts[t] <= c.level[t] && u.owner[t] == 0;
    let friendly = s.tiles.owner[t] == cp || (s.tiles.owner[t] > 0 && s.players.peace[s.p()][s.pclip(s.tiles.owner[t])]);
    let upgrade_allowed = u.owner[t] == cp && u.kind[t] == 43 && friendly;
    // get trainable units: a unit with movement terrain only where a neighbour fits it (Aquarion)
    let ok = if available && ctx.ok & ctx.water != 0 && !wet_neighbour(s, t) { ctx.ok & !ctx.water } else { ctx.ok };
    (if available { ok & !UPGRADES } else { 0 }) | (if upgrade_allowed { ctx.ok & UPGRADES } else { 0 })
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
    // train action: explore from tile with get sight range (Scout units see 2 tiles)
    let radius = if s.tiles.terrain[source] == 4 || r.skills[kind][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(source, t) <= radius).collect();
    discover(s, &reveal, cp);
    // Cymanti: train unit's check step on poison
    crate::cymanti::step_on_poison(s, source, r);
    // train action: a Tentacles unit's area damage (its attack actions run after the
    // exploration)
    if r.tentacles[kind] {
        area_damage(s, source, source, r, g, true);
    }
}

/// The unit on `i` may be promoted: 3 kills, not yet a veteran, not Static
/// (can be promoted): a boat is judged by its passenger, so a non-Static passenger is promoted at sea.
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

/// Disband the unit on `source`, refunding half its full cost (get full cost: boat plus passenger) to seat
/// index `p` (its owner).
pub fn disband_of(s: &mut State, source: usize, r: &Rules, p: usize) {
    disband_chain(s, source, r, p, true);
}

/// disband unit action: the follower gets its own disband unit action (paid,
/// Cymanti), the leader loses the unit; `paid`: half the full cost to seat index `p`.
pub fn disband_chain(s: &mut State, source: usize, r: &Rules, p: usize, paid: bool) {
    let follower = crate::cymanti::unlink(s, source);
    disband_one(s, source, r, p, paid);
    if let Some(f) = follower {
        if s.units.owner[f] != 0 {
            disband_chain(s, f, r, p, true);
        }
    }
}

fn disband_one(s: &mut State, source: usize, r: &Rules, p: usize, paid: bool) {
    let kind = s.units.kind[source] as usize;
    let passenger = s.units.passenger[source];
    let full = r.unit_cost[kind] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    // disband action never spawns spores
    s.units.poison[source] = false;
    remove(s, source, r, None, false);
    if paid {
        s.players.stars[p] += full.div_euclid(2);
    }
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

/// to coordinates: SW, W, NW, N, NE, E, SE, S as (dx, dy) in game coordinates (y is the row).
pub const GRID: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)];
/// get tile neighbors order (x outer).
const STEPS: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];

/// get direction from distance vector: the direction with the largest weighted dot product with the
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

/// get direction(from, to): a neighbour's direction (8, none, for the same tile), farther tiles by
/// vector_direction.
pub fn direction(size: usize, from: usize, to: usize) -> i32 {
    direction_xy(((from % size) as i32, (from / size) as i32), ((to % size) as i32, (to / size) as i32))
}

/// get direction on coordinates that may be off the board: the game's
/// null coordinates (-1, -1) (world coordinates..cctor) are the origin of the attack in
/// infiltration reward action.
pub fn direction_xy(from: (i32, i32), to: (i32, i32)) -> i32 {
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    if dx.abs().max(dy.abs()) > 1 {
        return vector_direction(dx, dy);
    }
    GRID.iter().position(|&d| d == (dx, dy)).map_or(8, |d| d as i32)
}

/// unit state create: a new unit faces the map centre (width / 2, height / 2).
pub fn spawn_direction(size: usize, tile: usize) -> i32 {
    let c = (size / 2) as i32;
    vector_direction(c - (tile % size) as i32, c - (tile / size) as i32)
}

/// get path(source, dest, 1, unit) for a neighbour `dest` of the unit on `source`: is allowed to finish on tile
/// and is tile accessible under create for unit (enemy territory allowed, occupied tiles not).
/// try push unit default's get path(source, dest, max cost 1) for the unit on `source`: the direct step, or,
/// onto a Bridge whose direction rule refuses the direct step, two road steps (half cost each) through a free road
/// tile next to both that the unit may cross (no enemy zone of control), entering the bridge from an allowed side.
fn push_reachable(s: &State, source: usize, dest: usize, r: &Rules, g: &Geom) -> bool {
    if push_step(s, source, source, dest, r) {
        return true;
    }
    let (t, u) = (&s.tiles, &s.units);
    let owner = u.owner[source];
    if t.structure[dest] != 48 || u.owner[dest] != 0 || !crate::kernels::bridge_blocks(s, owner, source, dest) {
        return false;
    }
    let p = s.pclip(owner);
    let road = |i: usize| {
        let friendly = t.owner[i] == 0 || t.owner[i] == owner || s.players.peace[p][s.pclip(t.owner[i])];
        (t.road[i] || t.structure[i] == 1 || s.cities.owner[i] != 0 || t.structure[i] == 48) && friendly
    };
    // get movement cost's zone of control: not from a unit with a leader (Cymanti)
    let zoc = |i: usize| g.adj[i].iter().any(|&j| {
        let j = j as usize;
        u.owner[j] != 0 && u.owner[j] != owner && !u.invisible[j] && !s.players.peace[p][s.pclip(u.owner[j])] && u.leader[j] == 0
    });
    road(source) && g.adj[source].iter().any(|&m| {
        let m = m as usize;
        m != dest && g.d(m, dest) == 1 && u.owner[m] == 0 && road(m) && !zoc(m) && push_step(s, source, source, m, r)
            && push_step(s, source, m, dest, r)
    })
}

/// is tile accessible of `dest` coming from `from` for the unit on `unit`.
fn push_step(s: &State, unit: usize, from: usize, dest: usize, r: &Rules) -> bool {
    let source = unit;
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
        return r.skills[kind][FLOAT] || r.fly[kind] || !crate::kernels::bridge_blocks(s, owner, from, dest);
    }
    if structure == 1 {
        return true;
    }
    // Cymanti: an Algae water tile is wetland, in every unit's allowed terrain (is tile accessible's water branch)
    if t.algae[dest] && (terrain == 1 || terrain == 2) {
        return true;
    }
    let tech = &s.players.tech[p];
    let terrain_ok = crate::kernels::Swim::of(s, r, kind, owner).terrain_ok(s, tech, dest);
    let land_access = (terrain != 1 && terrain != 2)
        || (structure == 8 && t.owner[dest] == owner && crate::tribes::can_embark(s.players.tribe[p], tech));
    terrain_ok && (land_access || r.skills[kind][FLOAT] || r.fly[kind])
}

/// try push unit default, with train unit on occupied space's kill unit action when it fails: moves the unit on
/// `source` off it for a unit the current player trains there. The occupant's facing (reversed for another player's
/// unit) is tried first, then the directions -1, +1, -2, +2, -3, +3, -4 from it; a boat on water keeps the first land
/// tile as its last resort. The push is a move action with move reason 2: no stomp, and moved/attacked stay as they were
/// unless the unit embarks or disembarks (those actions exhaust it). Deterministic, no hash draw.
pub fn push(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let size = s.size as i32;
    let owner = s.units.owner[source];
    if owner <= 0 {
        return;
    }
    // train unit on occupied space (game version > 105): an occupant hidden from the
    // trainer (is invisible: another player's, no peace with them) gets a reveal action without an origin
    // (no turn, no meeting). The action stack runs it before the push's move action, so the unit stays revealed.
    let cp = s.current_player;
    if s.units.invisible[source] && owner != cp && !s.players.peace[s.p()][s.pclip(owner)] {
        s.units.invisible[source] = false;
    }
    let mut facing = s.units.direction[source];
    if owner != s.current_player {
        facing = if facing < 4 { facing + 4 } else { facing - 4 };
    }
    // try push unit default: a chain unit starts from its leader / follower (their average when it has both; Cymanti)
    if let Some(dir) = crate::cymanti::link_direction(s, source) {
        facing = dir;
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
        if push_reachable(s, source, t, r, g) {
            target = Some(t);
            break;
        }
    }
    match target.or(fallback.filter(|&t| push_reachable(s, source, t, r, g))) {
        Some(t) => {
            let (moved, attacked, lock) = (s.units.moved[source], s.units.attacked[source], s.units.move_lock[source]);
            relocate_as(s, source, t, r, g, true, None, false);
            if s.units.kind[t] == kind {
                s.units.moved[t] = moved;
                s.units.attacked[t] = attacked;
                s.units.move_lock[t] = lock;
            }
        }
        None => remove(s, source, r, None, true),
    }
}

pub fn superunit(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    push(s, source, r, g);
    // train unit on occupied space(GIANT) trains the tribe's override (Polaris: Gaami)
    let kind = crate::tribes::unit_override(s.players.tribe[s.p()], 12) as usize;
    summon(s, source, kind, r, g, false);
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
    // the seats hit (modify diplomacy relations from attack per damaged unit's owner)
    let mut hit_seats = [false; crate::state::MAXP + 1];
    let mut pacifist_reset = false;
    for i in 0..n {
        let u = &s.units;
        let targets = g.d(center, i) == 1 && u.owner[i] != 0 && u.owner[i] != src_owner && !protected(s, src_owner, u.owner[i]);
        if targets {
            let dd = r.unit_defense[u.kind[i] as usize];
            let battle = combat(aa, ah, hp_s, dd, u.health[i], max_health(s, r, i), defense_bonus(s, r, i));
            let _ = stomp;
            // area damage: the full battle damage for a Tentacles unit (Passive), else the floored half (Splash)
            let dmg = if r.tentacles[src_kind] { battle.0 } else { battle.2.floor() };
            damage[i] = dmg;
            dead[i] = u.health[i] - dmg <= 0.0;
            if dmg > 0.0 {
                hit_seats[u.owner[i].clamp(0, crate::state::MAXP as i32) as usize] = true;
            }
            pacifist_reset |= dmg > 0.0 && !u.invisible[i];
        }
    }
    // One attack action per target, pushed in get tile neighbors (x-outer) order, from the stomper's tile for a stomp
    // (area damage default, origin = the centre) and from the attacker's tile for
    // a splash (attack command, attack action(Origin, neighbour)). Each
    // attack action turns its target toward the origin and the origin's unit toward the target (
    // 753); the stack is LIFO, so the origin ends facing the first target in neighbour order.
    let size = s.size;
    let origin = if stomp { center } else { source };
    let mut first = None;
    for (dx, dy) in STEPS {
        let (x, y) = ((center % size) as i32 + dx, (center / size) as i32 + dy);
        if x >= 0 && y >= 0 && (x as usize) < size && (y as usize) < size {
            let i = y as usize * size + x as usize;
            let u = &s.units;
            if u.owner[i] != 0 && u.owner[i] != src_owner && !protected(s, src_owner, u.owner[i]) {
                first = first.or(Some(i));
                s.units.direction[i] = direction(size, i, origin);
            }
        }
    }
    if let Some(i) = first {
        s.units.direction[origin] = direction(size, origin, i);
    }
    for i in 0..n {
        s.units.health[i] -= damage[i];
    }
    // every attack action's perform attack takes its target's Bubble, whatever the damage
    for i in 0..n {
        let u = &s.units;
        if g.d(center, i) == 1 && u.owner[i] != 0 && u.owner[i] != src_owner && !protected(s, src_owner, u.owner[i]) {
            s.units.bubble[i] = false;
            s.units.swift[i] = false;
        }
    }
    for i in 0..n {
        if dead[i] {
            crate::cov::hit(37);
            remove(s, i, r, Some(source), true);
        } else if r.poisons[src_kind] && g.d(center, i) == 1 && s.units.owner[i] != 0 && s.units.owner[i] != src_owner
            && !protected(s, src_owner, s.units.owner[i]) {
            // Cymanti: every attack action's perform attack from a Poison origin (the Island's Stomp) poisons its
            // survivor, whatever the damage
            s.units.poison[i] = true;
        }
    }
    for (victim, &hit) in hit_seats.iter().enumerate() {
        if hit {
            hostile(s, victim as i32);
        }
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
    // upgrade action copies moved/attacked; can attack then reads the new type: a Raft that only moved
    // (move_lock) and becomes a Dash ship (Scoutship, Rammership) may still attack
    if s.units.move_lock[source] && r.dash[kind] {
        s.units.attacked[source] = false;
    }
    s.units.move_lock[source] = false;
    let hp = (max_health(s, r, source) - (before - s.units.health[source])).max(1.0);
    s.units.health[source] = hp;
    let p = s.p();
    s.players.stars[p] -= r.unit_cost[kind];
    s.players.score[p] = s.players.score[p].wrapping_add(5 * (r.unit_cost[kind] - r.unit_cost[old]));
    let radius = if r.skills[kind][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(source, t) <= radius).collect();
    discover(s, &reveal, s.current_player);
    // upgrade action's train unit: check step on poison
    crate::cymanti::step_on_poison(s, source, r);
}
