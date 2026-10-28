//! gpu/pathfinding.py: reference-ordered shortest path (Rust BinaryHeap tie behaviour) for fog reveal.

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
    /// path tiles from the target back to (excluding) the source: path[k] has rank k
    pub path: Vec<usize>,
    pub reveal: Vec<bool>,
}

impl PathScratch {
    pub fn new(n: usize) -> PathScratch {
        PathScratch { heap: Heap { ids: vec![0; 8 * n + 1], costs: vec![INF; 8 * n + 1], count: 0 }, flags: vec![0; n],
                      dist: vec![INF; n], parent: vec![-1; n], path: Vec::with_capacity(n), reveal: vec![false; n] }
    }
}

/// pathfinding.path_details: the reference-ordered shortest path from `source` to `target` (Dijkstra with the
/// reference heap and its tie behaviour, neighbours in kernels' fixed order, movement data's passable/terminal/
/// road/budget rules). Fills `ps.path` (target first, source excluded) and returns whether the target was found;
/// the reference's mask is {source, target} plus the path and its rank is the index in `ps.path` (source: len).
pub fn search(s: &State, source: usize, target: usize, r: &Rules, g: &Geom, ps: &mut PathScratch) -> bool {
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
    let passable = |i: usize| (flags[i] & pass_need == pass_need && (u.owner[i] == 0 || u.owner[i] == cp || u.invisible[i]))
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
            let road_at = flags[at] & MV_ROAD != 0 && !creep;
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
