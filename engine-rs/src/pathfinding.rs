//! The path a commanded move takes (the game's route, game version 123), which decides the tiles it explores;
//! gpu/pathfinding.py's reference-ordered Dijkstra remains as a fallback.

use crate::geom::Geom;
use crate::kernels::{bridge_blocks, forest_step, hide_passable, peace_unit, movement_flags, INF, MV_FOREST, MV_LAND, MV_MOUNTAIN, MV_OWN_PORT, MV_PASS, MV_ROAD, MV_WATER, MV_ZOC};
use crate::rules::{Rules, CARRY, CREEP, FLOAT, HIDE, INFILTRATE, SCOUT};
use crate::state::State;

struct Heap {
    ids: Vec<i32>,
    costs: Vec<i32>,
    count: usize,
}

impl Heap {
    fn push(&mut self, index: i32, cost: i32) {
        let mut at = self.count;
        self.ids[at] = index;
        self.costs[at] = cost;
        while at > 0 && self.costs[at] < self.costs[(at - 1) / 2] {
            let parent = (at - 1) / 2;
            self.ids.swap(at, parent);
            self.costs.swap(at, parent);
            at = parent;
        }
        self.count += 1;
    }

    fn pop(&mut self) -> (i32, i32) {
        let (index, cost) = (self.ids[0], self.costs[0]);
        self.count -= 1;
        let count = self.count;
        let (last_id, last_cost) = (self.ids[count], self.costs[count]);
        let mut at = 0usize;
        while 2 * at + 1 < count {
            let left = 2 * at + 1;
            let right = left + 1;
            let child = if right < count && self.costs[right] <= self.costs[left] { right } else { left };
            self.ids[at] = self.ids[child];
            self.costs[at] = self.costs[child];
            at = child;
        }
        self.ids[at] = last_id;
        self.costs[at] = last_cost;
        while at > 0 && self.costs[at] < self.costs[(at - 1) / 2] {
            let p = (at - 1) / 2;
            self.ids.swap(at, p);
            self.costs.swap(at, p);
            at = p;
        }
        (index, cost)
    }
}

/// Reusable buffers for path search and reveal (one per GameBatch, inside core::Scratch).
pub struct PathScratch {
    heap: Heap,
    flags: Vec<u8>,
    dist: Vec<i32>,
    parent: Vec<i32>,
    open: OpenList,
    /// path tiles from the target back to (excluding) the source: path[k] has rank k
    pub path: Vec<usize>,
    pub reveal: Vec<bool>,
}

impl PathScratch {
    pub fn new(n: usize) -> PathScratch {
        PathScratch { heap: Heap { ids: vec![0; 8 * n + 1], costs: vec![INF; 8 * n + 1], count: 0 }, flags: vec![0; n],
                      dist: vec![INF; n], parent: vec![-1; n], open: OpenList::default(), path: Vec::with_capacity(n), reveal: vec![false; n] }
    }
}

/// pathfinding.path_details: the reference-ordered shortest path from `source` to `target` (Dijkstra with the
/// reference heap and its tie behaviour, neighbours in kernels' fixed order, movement data's passable/terminal/
/// road/budget rules). Fills `ps.path` (target first, source excluded) and returns whether the target was found;
/// the reference's mask is {source, target} plus the path and its rank is the index in `ps.path` (source: len).
/// The game's route for a commanded move of the unit on `source`: A* over a sorted open list (ties
/// first in, first out), priority the Euclidean distance to the target + cost as float32 (the start: Manhattan
/// distance), movement cost in the game's units (10 a step, 5 road to road, 1000 for a step that ends the
/// move), no closed set (a tile is re-queued whenever its cost improves, or when unvisited), expansion while cost <
/// movement x 10, at most 1000 pops; neighbours in the game's area order (x offset outer, y inner). Fills
/// `ps.path` (target first, source excluded) and returns whether the target was found; falls back to `search_ref`.
pub fn search(s: &State, source: usize, target: usize, r: &Rules, g: &Geom, ps: &mut PathScratch) -> bool {
    if source == target {
        ps.path.clear();
        return true;
    }
    let (n, size) = (s.n, s.size as i32);
    let u = &s.units;
    let cp = s.current_player;
    let owner = u.owner[source];
    let kind = u.kind[source] as usize;
    let sk = &r.skills[kind];
    let (aquatic, creep) = (sk[FLOAT], sk[CREEP]);
    let no_zoc = creep || sk[INFILTRATE] || u.invisible[source];
    let max_cost = (r.unit_movement[kind] + u.boosted[source] as i32) * 10;
    movement_flags(s, owner, g, &mut ps.flags);
    let flags = &ps.flags;
    let pass_need = if aquatic { MV_PASS } else { MV_PASS | MV_LAND };
    let hide = sk[HIDE];
    let accessible = |i: usize| i == source || (flags[i] & pass_need == pass_need && (u.owner[i] == 0 || u.owner[i] == cp || s.hidden_from(i, cp)))
        || ((hide || peace_unit(s, owner, i)) && (aquatic || flags[i] & MV_LAND != 0) && hide_passable(s, owner, i));
    // movement cost of the unit entering `to` from `from`
    let cost = |from: usize, to: usize| -> i32 {
        let f = flags[to];
        if !aquatic && f & MV_OWN_PORT != 0 {
            return 1000; // embarks
        }
        if !no_zoc && f & MV_ZOC != 0 {
            return 1000;
        }
        if f & MV_WATER != 0 {
            if creep {
                return 10;
            }
        } else {
            if aquatic {
                return 1000; // Swim on land
            }
            if creep && f & MV_MOUNTAIN == 0 {
                return 10;
            }
        }
        if flags[from] & MV_ROAD != 0 && f & MV_ROAD != 0 && !creep && !aquatic {
            return 5;
        }
        if f & (MV_MOUNTAIN | MV_FOREST) != 0 {
            return 1000;
        }
        10
    };
    let (tx, ty) = ((target % s.size) as i32, (target / s.size) as i32);
    let dist = |i: usize| {
        let (dx, dy) = (tx - (i % s.size) as i32, ty - (i / s.size) as i32);
        ((dy * dy + dx * dx) as f32).sqrt()
    };
    let (cost_so_far, came_from) = (&mut ps.dist, &mut ps.parent);
    cost_so_far[..n].fill(0);
    came_from[..n].fill(-1);
    let open = &mut ps.open;
    open.clear();
    let (sx, sy) = ((source % s.size) as i32, (source / s.size) as i32);
    open.push(((tx - sx).abs() + (ty - sy).abs()) as f32, source);
    let mut found = false;
    for _ in 0..1000 {
        let Some(at) = open.pop() else { break };
        if at == target {
            found = true;
            break;
        }
        let here = cost_so_far[at];
        if here < max_cost {
            let (ax, ay) = ((at % s.size) as i32, (at / s.size) as i32);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let (x, y) = (ax + dx, ay + dy);
                    if (dx == 0 && dy == 0) || x < 0 || y < 0 || x >= size || y >= size {
                        continue;
                    }
                    let dest = (y * size + x) as usize;
                    if !accessible(dest) || bridge_blocks(s, owner, at, dest) {
                        continue;
                    }
                    let next = cost(at, dest) + here;
                    let old = cost_so_far[dest];
                    if old < 1 || next < old {
                        cost_so_far[dest] = next;
                        came_from[dest] = at as i32;
                        open.push(dist(dest) + next as f32, dest);
                    }
                }
            }
        }
        if open.is_empty() {
            break;
        }
    }
    ps.path.clear();
    if found {
        let mut at = target;
        for _ in 0..=n {
            if at == source {
                return true;
            }
            ps.path.push(at);
            match came_from[at] {
                -1 => break,
                c => at = c as usize,
            }
        }
    }
    search_ref(s, source, target, r, g, ps)
}

/// A sorted open list: pop takes the lowest priority, equal priorities in insertion order.
#[derive(Default)]
pub struct OpenList {
    items: std::collections::VecDeque<(f32, usize)>,
}

impl OpenList {
    fn clear(&mut self) {
        self.items.clear();
    }
    fn push(&mut self, prio: f32, i: usize) {
        let at = self.items.partition_point(|&(p, _)| p <= prio);
        self.items.insert(at, (prio, i));
    }
    fn pop(&mut self) -> Option<usize> {
        self.items.pop_front().map(|(_, i)| i)
    }
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// The reference-ordered Dijkstra (the JAX engine's path), used only if the game's route search fails.
pub fn search_ref(s: &State, source: usize, target: usize, r: &Rules, g: &Geom, ps: &mut PathScratch) -> bool {
    let n = s.n;
    let u = &s.units;
    let cp = s.current_player;
    let owner = u.owner[source];
    let kind = u.kind[source] as usize;
    let sk = &r.skills[kind];
    let (aquatic, creep, carry) = (sk[FLOAT], sk[CREEP], sk[CARRY]);
    let no_zoc = creep || sk[INFILTRATE] || u.invisible[source];
    let budget = (r.unit_movement[kind] + u.boosted[source] as i32) * 2;
    movement_flags(s, owner, g, &mut ps.flags);
    let flags = &ps.flags;
    let term_bits = (if no_zoc { 0 } else { MV_ZOC }) | MV_MOUNTAIN | (if aquatic { 0 } else { MV_OWN_PORT });
    let terminal = |f: u8| f & term_bits != 0 || (f & (MV_FOREST | MV_ROAD) == MV_FOREST && !creep) || (carry && f & MV_WATER == 0);
    let pass_need = if aquatic { MV_PASS } else { MV_PASS | MV_LAND };
    let hide = sk[HIDE];
    let passable = |i: usize| (flags[i] & pass_need == pass_need && (u.owner[i] == 0 || u.owner[i] == cp || s.hidden_from(i, cp)))
        || ((hide || peace_unit(s, owner, i)) && (aquatic || flags[i] & MV_LAND != 0) && hide_passable(s, owner, i));
    let (heap, dist, parent) = (&mut ps.heap, &mut ps.dist, &mut ps.parent);
    dist[..n].fill(INF);
    parent[..n].fill(-1);
    heap.ids[0] = source as i32;
    heap.costs[0] = 0;
    heap.count = 1;
    dist[source] = 0;
    let mut found = false;
    while heap.count > 0 && !found {
        let (at, cost) = heap.pop();
        let at = at as usize;
        found = at == target;
        let expand = !found && cost <= dist[at] && (at == source || !terminal(flags[at])) && cost < budget;
        if expand {
            let road_at = flags[at] & MV_ROAD != 0 && !creep && !aquatic;
            for k in 0..8 {
                if !g.nb_valid[at * 8 + k] {
                    continue;
                }
                let dest = g.nb[at * 8 + k] as usize;
                let road_edge = road_at && flags[dest] & MV_ROAD != 0;
                let newcost = forest_step(creep, road_edge, flags[dest], cost + if road_edge { 1 } else { 2 }, budget);
                if passable(dest) && dest != source && newcost < dist[dest] && !bridge_blocks(s, owner, at, dest) {
                    heap.push(dest as i32, newcost);
                    dist[dest] = newcost;
                    parent[dest] = at as i32;
                }
            }
        }
    }
    ps.path.clear();
    if found {
        let mut at = target as i32;
        while at >= 0 && at as usize != source {
            ps.path.push(at as usize);
            at = parent[at as usize];
        }
    }
    found
}

/// pathfinding.reveal_path into `ps.reveal` for the traveled tiles {source} + ps.path[from..] + extra: every tile
/// within 1 (2 from a mountain, or for scouts) of one of them, visiting only the boxes around them.
pub fn reveal_tiles(s: &State, source: usize, from: usize, extra: Option<usize>, r: &Rules, ps: &mut PathScratch) {
    let size = s.size as i32;
    let scout = r.skills[s.units.kind[source] as usize][SCOUT];
    let out = &mut ps.reveal[..s.n];
    out.fill(false);
    for m in std::iter::once(source).chain(ps.path[from..].iter().copied()).chain(extra) {
        let radius = if s.tiles.terrain[m] == 4 || scout { 2 } else { 1 };
        let (mx, my) = ((m % s.size) as i32, (m / s.size) as i32);
        for y in (my - radius).max(0)..=(my + radius).min(size - 1) {
            for x in (mx - radius).max(0)..=(mx + radius).min(size - 1) {
                out[(y * size + x) as usize] = true;
            }
        }
    }
}
