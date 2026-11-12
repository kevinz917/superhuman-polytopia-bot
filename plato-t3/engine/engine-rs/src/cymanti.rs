//! Cymanti (S4): Algae, and (later steps) Poison, Spores, Explode, Swarm and the Centipede chains.
//!
//! Representation: `tiles.algae` is tile effect algae (Foam is a skin-19 effect with no rule, never modelled).

use crate::geom::Geom;
use crate::rules::{Rules, FLOAT};
use crate::state::State;

/// The Algae improvement (38, port -> algae; Discrete: no improvement state).
pub const ALGAE: usize = 38;
/// The Mycelium (39, road -> mycelium): a Network router with the Heal ability.
pub const MYCELIUM: usize = 39;

/// is resource visible to player for the player's tribe and techs: the resource's
/// improvement unlocked or unlockable; as the observation's hides (kernels.rs observe_impl).
pub fn resource_visible(tribe: i32, tech: &[bool; 128], resource: i32) -> bool {
    let cymanti = tribe == crate::tribes::CYMANTI;
    match resource {
        0 => false,
        2 => !cymanti && (tech[6] || tech[8] || tech[9]),
        5 => tech[20] || tech[23] || tech[24],
        7 => cymanti,
        8 => tech[crate::tribes::starfishing_parent(tribe)] || tech[crate::tribes::starfishing_tech(tribe)],
        9 => tech[39],
        _ => true,
    }
}

/// build action of the Algae: the cost is paid, no improvement is
/// placed (Discrete) and none scores; the tile gets the Algae effect; the sub-actions run in order: update routes,
/// update city connections (the effect makes a route tile), then create resource(Fruit) unless the player already sees a
/// resource there (get resource visible: explored, no improvement, is resource visible to player). The fruit
/// overwrites a hidden resource.
pub fn build_algae(s: &mut State, target: usize, r: &Rules, g: &Geom) {
    let pi = s.p();
    let player = s.current_player;
    s.players.stars[pi] -= r.structures.cost[ALGAE].max(0);
    let t = &s.tiles;
    let seen = t.explored[pi][target] && t.structure[target] == 0
        && resource_visible(s.players.tribe[pi], &s.players.tech[pi], t.resource[target]);
    s.tiles.algae[target] = true;
    crate::structures::update_routes(s, r);
    crate::structures::connections(s, player, g, true);
    if !seen {
        s.tiles.resource[target] = 6;
    }
}

/// get unit actions' clear tile effect Algae: the mover's unit on an
/// Algae tile, able to attack (not attacked, not Frozen, not moved unless Dash: engine-rs's `attacked`) or to move
/// (not moved, not Frozen), with Amphibious, Swim or Fly. Any tribe.
pub fn can_clear_algae(s: &State, i: usize, r: &Rules) -> bool {
    let u = &s.units;
    let kind = u.kind[i].clamp(0, 127) as usize;
    u.owner[i] == s.current_player && u.owner[i] != 0 && s.tiles.algae[i] && !u.frozen[i] && (!u.attacked[i] || !u.moved[i])
        && (r.amphibious[kind] || r.skills[kind][FLOAT] || r.fly[kind])
}

/// clear tile effect command -> clear tile effect action(Algae, by unit): the
/// effect (and Foam) goes, the unit is exhausted (make exhausted: moved, attacked, Boosted lost), then the
/// pushed update routes runs before update city connections(player). The Fruit the build made stays.
pub fn clear_algae_command(s: &mut State, i: usize, r: &Rules, g: &Geom) {
    let player = s.current_player;
    s.tiles.algae[i] = false;
    s.units.moved[i] = true;
    s.units.attacked[i] = true;
    s.units.move_lock[i] = false;
    s.units.boosted[i] = false;
    s.units.double_ready[i] = false;
    crate::structures::update_routes(s, r);
    crate::structures::connections(s, player, g, true);
}

#[inline]
fn water(s: &State, t: usize) -> bool {
    s.tiles.terrain[t] == 1 || s.tiles.terrain[t] == 2
}

/// spawn spore resource (version > 118): a Poisoned unit dies (or any unit
/// explodes) on `t`: create resource(Spores) overwrites whatever resource the tile had (a city's, an improved
/// tile's); on water build action(current player, Algae, no cost) runs first (its Fruit is overwritten by the Spores
/// that run last), so a water tile ends with Algae and Spores. The Build's update routes and update city connections
/// (current player) are deferred to the end of the command (`algae_dirty`, core::step): nothing between reads them.
pub fn spawn_spores(s: &mut State, t: usize) {
    if water(s, t) {
        s.tiles.algae[t] = true;
        s.algae_dirty = true;
    }
    s.tiles.resource[t] = 7;
}

/// move action's Algae ability (the Island, reason != Push): every path tile that is
/// water without Algae gets build action(owner, algae spawn, no cost): the effect only (no Fruit), then update routes and
/// update city connections (deferred as for the spores; the mover is the current player).
pub fn algae_trail(s: &mut State, tiles: &[usize]) {
    for &t in tiles {
        if water(s, t) && !s.tiles.algae[t] {
            s.tiles.algae[t] = true;
            s.algae_dirty = true;
        }
    }
}

/// check step on poison: a unit placed on an improvement with the Poison
/// ability (Fungi 37, Clathrus 41) by a move action or a train unit (training, upgrade action: Grow, boat upgrades) is
/// Poisoned unless its owner's tribe has poison resist (Cymanti) or it flies.
pub fn step_on_poison(s: &mut State, t: usize, r: &Rules) {
    let st = s.tiles.structure[t];
    let owner = s.units.owner[t];
    if (st == 37 || st == 41) && owner > 0 {
        let kind = s.units.kind[t].clamp(0, 127) as usize;
        if s.players.tribe[s.pclip(owner)] != crate::tribes::CYMANTI && !r.fly[kind] {
            s.units.poison[t] = true;
        }
    }
}

/// A Poison splasher's splash: every neighbour of the target (get tile neighbors)
/// with a unit not under an active peace treaty with the attacker's player (own units count as peace) is Poisoned:
/// no damage, no turn, no relations.
pub fn poison_splash(s: &mut State, source: usize, target: usize, g: &Geom) {
    let owner = s.units.owner[source];
    for &j in &g.adj[target] {
        let j = j as usize;
        let o = s.units.owner[j];
        if o != 0 && o != owner && !crate::diplomacy::protected(s, owner, o) {
            s.units.poison[j] = true;
        }
    }
}

/// can explode (version > 114): the mover's unit with Explode,
/// not Frozen, not created this turn (a Segment with a leader may: step 6). No moved/attacked test.
pub fn can_explode(s: &State, i: usize, r: &Rules) -> bool {
    let u = &s.units;
    let kind = u.kind[i].clamp(0, 127) as usize;
    // not on its creation turn unless it has a leader (a Segment eaten this turn)
    u.owner[i] == s.current_player && u.owner[i] != 0 && r.explode[kind] && !u.frozen[i] && (u.created[i] != s.turn || u.leader[i] != 0)
}

/// explode command -> explode unit action (version > 120):
/// Spores (and Algae on water) on the tile; every unit in radius 1 not at peace with the player (own units count as
/// peace) takes the floored half of the exploder's battle damage as an attack action from its own tile (the player
/// gets the kill, the victim the casualty; Boosted, Swift and Bubble go; a Drench victim floods its own tile),
/// relations and the Pacifist task as for an attack, and a survivor is Poisoned; the exploder leaves without a
/// kill unit (5 x its full cost off its owner's score, no casualty, no kill).
pub fn explode_command(s: &mut State, src: usize, r: &Rules, g: &Geom) {
    use crate::kernels::{combat, max_health};
    let n = s.n;
    let cp = s.current_player;
    let p = s.p();
    let kind = s.units.kind[src].clamp(0, 127) as usize;
    // a follower gets its own explode unit action, run after this one (the chain explodes tailwards); a leader loses it
    let follower = unlink(s, src);
    spawn_spores(s, src);
    let (aa, ah, am) = (r.unit_attack[kind], s.units.health[src], max_health(s, r, src));
    let mut hits: Vec<(usize, f32)> = Vec::new();
    for t in 0..n {
        let o = s.units.owner[t];
        if g.d(src, t) > 1 || o == 0 || crate::diplomacy::protected(s, cp, o) {
            continue;
        }
        let dd = r.unit_defense[s.units.kind[t].clamp(0, 127) as usize];
        let battle = combat(aa, ah, am, dd, s.units.health[t], max_health(s, r, t), crate::military::defense_bonus(s, r, t));
        hits.push((t, battle.2.floor()));
        crate::diplomacy::hostile(s, o);
        // the Pacifist task resets for every victim, whatever the damage
        s.players.attacked[p] = true;
    }
    crate::military::remove(s, src, r, None, false);
    for (t, dmg) in hits {
        let vk = s.units.kind[t].clamp(0, 127) as usize;
        s.units.boosted[t] = false;
        s.units.swift[t] = false;
        s.units.bubble[t] = false;
        s.units.health[t] -= dmg.min(s.units.health[t]);
        let dead = s.units.health[t] <= 0.0;
        // perform attack from the victim's own tile: its Drench floods the tile, killed or not
        if r.drench[vk] && crate::aquarion::floodable(s, t) {
            crate::aquarion::flood_tile(s, t, cp, r, g);
        }
        if dead {
            s.players.kills[p] += 1;
            crate::military::remove(s, t, r, None, true);
        } else {
            s.units.poison[t] = true;
        }
    }
    if let Some(f) = follower {
        if s.units.owner[f] != 0 {
            explode_command(s, f, r, g);
        }
    }
}

/// get boost options (version >= 116): the neighbours with a unit of the player
/// that is not Boosted and not Swift (and has no leader: step 6).
fn boost_options(s: &State, i: usize, g: &Geom) -> Vec<usize> {
    let owner = s.units.owner[i];
    g.adj[i].iter().map(|&j| j as usize)
        .filter(|&j| s.units.owner[j] == owner && !s.units.boosted[j] && !s.units.swift[j] && s.units.leader[j] == 0)
        .collect()
}

/// get unit actions' Swarm and swarm command validity:
/// the mover's unit with Swarm that can attack (not attacked, not Frozen; engine-rs's `attacked` holds "moved
/// without Dash") and has a boost option.
pub fn can_swarm(s: &State, i: usize, r: &Rules, g: &Geom) -> bool {
    let u = &s.units;
    let kind = u.kind[i].clamp(0, 127) as usize;
    u.owner[i] == s.current_player && u.owner[i] != 0 && r.swarm[kind] && !u.attacked[i] && !u.frozen[i]
        && !boost_options(s, i, g).is_empty()
}

/// swarm command (no command base:Execute): the unit is marked moved and attacked (not make exhausted), then
/// boost others action: every boost option gets Swift.
pub fn swarm_command(s: &mut State, i: usize, g: &Geom) {
    let options = boost_options(s, i, g);
    s.units.moved[i] = true;
    s.units.attacked[i] = true;
    s.units.move_lock[i] = false;
    for j in options {
        s.units.swift[j] = true;
    }
}

// ---- step 6: Centipede chains ----

/// The Centipede (33) and its Segment (34).
pub const CENTIPEDE: usize = 33;
pub const SEGMENT: usize = 34;

#[inline]
fn linked(v: i32) -> Option<usize> {
    if v > 0 { Some((v - 1) as usize) } else { None }
}

/// Take the unit on `i` out of its chain: its leader loses it, its follower loses its leader. Returns the follower.
pub fn unlink(s: &mut State, i: usize) -> Option<usize> {
    if let Some(l) = linked(s.units.leader[i]) {
        s.units.follower[l] = 0;
    }
    let f = linked(s.units.follower[i]);
    if let Some(f) = f {
        s.units.leader[f] = 0;
    }
    s.units.leader[i] = 0;
    s.units.follower[i] = 0;
    f
}

/// kill unit's chain part: the follower takes the dying unit's moved and
/// attacked and gets upgrade action(owner, Centipede, cost 0): a new head with the Segment's damage (the clone keeps its
/// follower); the leader loses the unit. A middle Segment's death splits the chain.
pub fn kill_links(s: &mut State, i: usize, r: &Rules) {
    let (moved, attacked, lock) = (s.units.moved[i], s.units.attacked[i], s.units.move_lock[i]);
    if let Some(f) = unlink(s, i) {
        if s.units.owner[f] != 0 {
            s.units.moved[f] = moved;
            s.units.attacked[f] = attacked;
            s.units.move_lock[f] = lock;
            upgrade_head(s, f, r);
        }
    }
}

/// upgrade action(Centipede, cost 0) on a split-off Segment: train unit's fresh unit
/// (created this turn, no xp or promotion, check step on poison) with the old moved, attacked, home, facing, follower and
/// effects; health = new max - old damage (at least 0.1); score + 5 x the cost difference; explore from tile.
fn upgrade_head(s: &mut State, i: usize, r: &Rules) {
    let old = s.units.kind[i].clamp(0, 127) as usize;
    let damage = crate::kernels::max_health(s, r, i) - s.units.health[i];
    s.units.kind[i] = CENTIPEDE as i32;
    s.units.veteran[i] = false;
    s.units.kills[i] = 0;
    s.units.created[i] = s.turn;
    s.units.health[i] = (r.unit_health[CENTIPEDE] - damage).max(0.1);
    // a Dash type lifts the move-only lock (can attack reads the current type)
    if s.units.move_lock[i] && r.dash[CENTIPEDE] {
        s.units.attacked[i] = false;
        s.units.move_lock[i] = false;
    }
    let owner = s.units.owner[i];
    let o = s.pwrap(owner);
    s.players.score[o] = s.players.score[o].wrapping_add(5 * (r.unit_cost[CENTIPEDE] - r.unit_cost[old]));
    let (n, size) = (s.n, s.size);
    let radius = if s.tiles.terrain[i] == 4 || r.skills[CENTIPEDE][crate::rules::SCOUT] { 2 } else { 1 };
    let (x, y) = ((i % size) as i32, (i / size) as i32);
    let reveal: Vec<bool> = (0..n).map(|t| ((t % size) as i32 - x).abs().max(((t / size) as i32 - y).abs()) <= radius).collect();
    crate::economy::discover(s, &reveal, owner);
    step_on_poison(s, i, r);
}

/// The last unit of the follower chain from `i`.
pub fn tail(s: &State, i: usize) -> usize {
    let mut at = i;
    let mut guard = 0;
    while let Some(f) = linked(s.units.follower[at]) {
        if guard > s.n {
            break;
        }
        at = f;
        guard += 1;
    }
    at
}

/// eat action (version > 119): the first neighbour of the eater (the chain's tail) that
/// is empty and accessible from its tile to a Segment led by it, trying the direction opposite its facing, then
/// -1, +1, -2, +2, -3, +3, -4, gets train unit(Segment) (moved and attacked, created this turn, check step on poison),
/// led by the eater, explored from at the Segment's sight, + the Segment's score.
pub fn eat(s: &mut State, eater: usize, r: &Rules, g: &Geom) {
    use crate::kernels::{movement_flags, MV_LAND, MV_OWN_PORT, MV_PASS};
    let size = s.size as i32;
    let owner = s.units.owner[eater];
    let dir = s.units.direction[eater].clamp(0, 7);
    let opp = if dir < 4 { dir + 4 } else { dir - 4 };
    let mut flags = vec![0u8; s.n];
    movement_flags(s, owner, g, &mut flags);
    let (x, y) = (eater as i32 % size, eater as i32 / size);
    let mut found = None;
    for k in 1..=8i32 {
        let off = if (k - 1) & 1 == 0 { k / 2 } else { -(k >> 1) };
        let (dx, dy) = crate::military::GRID[(off + opp + 8).rem_euclid(8) as usize];
        let (tx, ty) = (x + dx, y + dy);
        if tx < 0 || ty < 0 || tx >= size || ty >= size {
            continue;
        }
        let t = (ty * size + tx) as usize;
        if s.units.owner[t] != 0 {
            continue;
        }
        // is tile accessible for a land Segment: explored, its terrain, land access (a unit with a leader never uses a
        // Port to go into water), no Bridge span across the step
        let f = flags[t];
        let port_only = f & MV_OWN_PORT != 0 && (s.tiles.terrain[t] <= 2 || s.tiles.flooded[t]) && !s.tiles.algae[t]
            && s.tiles.structure[t] != 48;
        if f & (MV_PASS | MV_LAND) == MV_PASS | MV_LAND && !port_only && !crate::kernels::bridge_blocks(s, owner, eater, t) {
            found = Some(t);
            break;
        }
    }
    let Some(t) = found else { return };
    let cp = s.current_player;
    s.current_player = owner;
    crate::military::summon(s, t, SEGMENT, r, g, false);
    s.current_player = cp;
    // make_trained_unit: the home is the tile's city, if it is one
    s.units.home[t] = if s.tiles.structure[t] == 1 { t as i32 + 1 } else { 0 };
    s.units.leader[t] = eater as i32 + 1;
    s.units.follower[eater] = t as i32 + 1;
}

/// try push unit default's chain facing: toward the leader / follower, or the average of
/// the two (grid directions average: opposite directions give first + 2).
pub fn link_direction(s: &State, i: usize) -> Option<i32> {
    use crate::military::{direction, vector_direction, GRID};
    let size = s.size;
    let toward = |o: usize| {
        let d = direction(size, i, o);
        if d > 7 { 0 } else { d }
    };
    match (linked(s.units.leader[i]), linked(s.units.follower[i])) {
        (Some(l), Some(f)) => {
            let (a, b) = (toward(l), toward(f));
            let (ga, gb) = (GRID[a as usize], GRID[b as usize]);
            let (sx, sy) = (ga.0 + gb.0, ga.1 + gb.1);
            Some(if sx == 0 && sy == 0 { (2 + a) % 8 } else { vector_direction(sx, sy) })
        }
        (None, Some(f)) => Some(toward(f)),
        (Some(l), None) => Some(toward(l)),
        (None, None) => None,
    }
}
