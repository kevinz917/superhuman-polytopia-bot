use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{self, BufRead, Write};

#[derive(Clone, Copy, Default)]
struct Tile { known: bool, terrain: i32, owner: i32, structure: i32, resource: i32 }
#[derive(Clone, Copy)]
struct Unit { tile: usize, owner: i32, kind: i32, hp: f32, vet: bool, moved: bool, attacked: bool, passenger: i32, home: i32 }
#[derive(Clone, Copy)]
struct City { tile: usize, owner: i32, level: i32 }
struct Act { id: String, kind: i32, src: i32, tgt: i32, par: i32 }

const INF: i32 = 1_000_000;

fn stats(k: i32) -> (f32, f32, f32, i32, i32, i32) {
    match k {
        2 => (10., 2., 2., 1, 1, 2), 3 => (10., 2., 1., 2, 1, 3), 4 => (10., 3.5, 1., 3, 1, 8), 5 => (15., 1., 3., 1, 1, 3),
        8 => (10., 4., 0., 1, 3, 8), 9 => (10., 2., 1., 1, 2, 3), 10 => (10., 0., 1., 1, 1, 5), 11 => (15., 3., 3., 1, 1, 5),
        12 => (40., 5., 4., 1, 1, 10), 37 => (10., 2., 2., 1, 1, 2), 38 => (5., 0., 0.5, 2, 1, 8), 39 => (5., 0., 0.5, 2, 1, 2),
        40 => (10., 2., 2., 2, 1, 2), 41 => (10., 3., 2., 2, 3, 15), 42 => (10., 2., 1., 3, 2, 5), 43 => (10., 0., 1., 2, 2, 0),
        44 => (10., 3., 3., 3, 1, 5), 45 => (40., 4., 4., 2, 1, 10), _ => (10., 2., 2., 1, 1, 2),
    }
}
fn stiff(k: i32) -> bool { matches!(k, 8 | 10 | 43 | 38 | 39 | 41 | 45) }
fn maxhp(u: &Unit) -> f32 {
    let k = if u.passenger > 0 { u.passenger } else { u.kind };
    stats(k).0 + if u.vet { 5.0 } else { 0.0 }
}
fn tech_tier(t: i32) -> i32 {
    match t { 1 | 6 | 10 | 15 | 20 => 1, 2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2, _ => 3 }
}

struct Bot { turn: i32, claimed: HashSet<usize>, stuck_turn: i32, goals: std::collections::HashMap<usize, usize> }

struct G {
    sz: usize, n: usize, me: i32, turn: i32, stars: i32,
    tiles: Vec<Tile>, units: Vec<Unit>, unit_at: Vec<i32>, cities: Vec<City>, tech: [bool; 128],
}

impl G {
    fn d(&self, a: usize, b: usize) -> i32 {
        let (ax, ay, bx, by) = ((a % self.sz) as i32, (a / self.sz) as i32, (b % self.sz) as i32, (b / self.sz) as i32);
        (ax - bx).abs().max((ay - by).abs())
    }
    fn nbrs(&self, a: usize) -> Vec<usize> {
        let (x, y) = ((a % self.sz) as i32, (a / self.sz) as i32);
        let mut v = Vec::with_capacity(8);
        for dy in -1..=1 { for dx in -1..=1 {
            if dx == 0 && dy == 0 { continue; }
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && nx < self.sz as i32 && ny < self.sz as i32 { v.push((ny as usize) * self.sz + nx as usize); }
        } }
        v
    }
    fn is_water(&self, t: usize) -> bool { self.tiles[t].known && (self.tiles[t].terrain == 1 || self.tiles[t].terrain == 2) }
    fn own_port(&self, t: usize) -> bool { self.tiles[t].known && self.tiles[t].structure == 8 && self.tiles[t].owner == self.me }
    fn pass(&self, t: usize) -> bool {
        let tl = &self.tiles[t];
        if !tl.known { return true; }
        match tl.terrain { 4 => self.tech[20], 1 => self.tech[10], 2 => self.tech[13], _ => true }
    }
    fn edge(&self, a: usize, b: usize) -> bool { if self.is_water(b) { self.is_water(a) || self.own_port(b) } else { true } }
    fn bfs(&self, start: usize, reverse: bool) -> Vec<i32> {
        let mut dist = vec![INF; self.n];
        let mut q = std::collections::VecDeque::new();
        dist[start] = 0;
        q.push_back(start);
        while let Some(c) = q.pop_front() {
            let dc = dist[c];
            for nb in self.nbrs(c) {
                if dist[nb] != INF || !self.pass(nb) { continue; }
                let ok = if reverse { self.edge(nb, c) } else { self.edge(c, nb) };
                if !ok { continue; }
                // enemy units block passing through for planning only mildly: allow
                dist[nb] = dc + 1;
                q.push_back(nb);
            }
        }
        dist
    }
    fn enemy_near(&self, t: usize, r: i32) -> i32 {
        self.units.iter().filter(|u| u.owner != self.me && self.d(u.tile, t) <= r).count() as i32
    }
}

fn gi(v: &Value, k: &str) -> i32 { v[k].as_i64().unwrap_or(0) as i32 }
fn gb(v: &Value, k: &str) -> bool { v[k].as_bool().unwrap_or(false) }

impl Bot {
    fn choose(&mut self, obs: &Value) -> String {
        let legal_v = match obs["legal_actions"].as_array() { Some(a) => a, None => return String::new() };
        let acts: Vec<Act> = legal_v.iter().map(|a| {
            let c = &a["command"];
            Act { id: a["action_id"].as_str().unwrap_or("").to_owned(), kind: gi(c, "kind"), src: gi(c, "source"), tgt: gi(c, "target"), par: gi(c, "parameter") }
        }).collect();
        if acts.is_empty() { return String::new(); }
        let fallback = acts.iter().find(|a| a.kind == 10).map(|a| a.id.clone()).unwrap_or_else(|| acts[0].id.clone());
        let vs = &obs["visible_state"];
        let sz = vs["board_size"].as_u64().unwrap_or(14) as usize;
        let n = sz * sz;
        let me = gi(obs, "player");
        let mut g = G { sz, n, me, turn: gi(vs, "turn"), stars: gi(vs, "stars"), tiles: vec![Tile::default(); n], units: vec![], unit_at: vec![-1; n], cities: vec![], tech: [false; 128] };
        if let Some(ts) = vs["technology"].as_array() { for t in ts { let k = t.as_i64().unwrap_or(0); if (0..128).contains(&k) { g.tech[k as usize] = true; } } }
        if let Some(ts) = vs["tiles"].as_array() { for t in ts {
            let i = t["tile"].as_u64().unwrap_or(0) as usize;
            if i >= n { continue; }
            g.tiles[i] = Tile { known: gb(t, "known"), terrain: gi(t, "terrain"), owner: gi(t, "owner"), structure: gi(t, "structure"), resource: gi(t, "resource") };
        } }
        if let Some(us) = vs["units"].as_array() { for u in us {
            let i = u["tile"].as_u64().unwrap_or(0) as usize;
            if i >= n { continue; }
            g.unit_at[i] = g.units.len() as i32;
            g.units.push(Unit { tile: i, owner: gi(u, "owner"), kind: gi(u, "kind"), hp: u["health"].as_f64().unwrap_or(10.0) as f32, vet: gb(u, "veteran"), moved: gb(u, "moved"), attacked: gb(u, "attacked"), passenger: gi(u, "passenger"), home: gi(u, "home") });
        } }
        if let Some(cs) = vs["cities"].as_array() { for c in cs {
            let i = c["tile"].as_u64().unwrap_or(0) as usize;
            if i >= n { continue; }
            g.cities.push(City { tile: i, owner: gi(c, "owner"), level: gi(c, "level") });
        } }
        if g.turn != self.turn { self.turn = g.turn; self.claimed.clear(); }
        let g = &g;

        // 1. rewards
        let rewards: Vec<&Act> = acts.iter().filter(|a| a.kind == 9).collect();
        if !rewards.is_empty() {
            let pref = |p: i32| match p { 3 => 10, 7 => 3, 5 => 8, 6 => 10, 8 => 5, 4 => 2, 1 => 9, 2 => 1, _ => 0 };
            return rewards.iter().max_by_key(|a| pref(a.par)).unwrap().id.clone();
        }
        // 2. promote
        if let Some(a) = acts.iter().find(|a| a.kind == 3 && a.par == 16) { return a.id.clone(); }
        // 3. capture
        if let Some(a) = acts.iter().filter(|a| a.kind == 8).max_by_key(|a| {
            let t = &g.tiles[(a.src.max(0) as usize).min(n - 1)];
            if t.structure == 1 && t.owner != 0 { 3 } else if t.structure == 1 { 2 } else { 1 }
        }) { return a.id.clone(); }

        let my_cities: Vec<City> = g.cities.iter().filter(|c| c.owner == me).cloned().collect();
        let enemy_cities: Vec<City> = g.cities.iter().filter(|c| c.owner != me && c.owner != 0).cloned().collect();
        let capturable = |t: usize| -> bool {
            let tl = &g.tiles[t];
            tl.known && ((tl.structure == 1 && tl.owner != me) || tl.structure == 2)
        };

        let spare = |ct: usize| -> bool {
            match my_cities.iter().find(|c| c.tile == ct) {
                Some(c) => (g.units.iter().filter(|u| u.owner == me && u.home == ct as i32 + 1).count() as i32) <= c.level && g.stars >= 2,
                None => false,
            }
        };
        // 4. attacks
        let mut best: Option<(f32, &Act)> = None;
        for a in acts.iter().filter(|a| a.kind == 2) {
            let (s, t) = ((a.src.max(0) as usize).min(n - 1), (a.tgt.max(0) as usize).min(n - 1));
            if g.unit_at[s] < 0 || g.unit_at[t] < 0 { continue; }
            let su = &g.units[g.unit_at[s] as usize];
            let tu = &g.units[g.unit_at[t] as usize];
            let (_, atk, _, _, rng_s, cost_s) = stats(su.kind);
            let (_, _, def_t, _, rng_t, cost_t) = stats(tu.kind);
            if su.kind == 10 { best = Some((50.0, a)); continue; }
            if atk <= 0.0 { continue; }
            let tt = &g.tiles[t];
            let bonus = if tt.structure == 1 && tt.owner == tu.owner { 1.5 } else { 1.0 };
            let af = atk * su.hp / maxhp(su);
            let df = def_t * tu.hp / maxhp(tu) * bonus;
            let tot = af + df;
            if tot <= 0.0 { continue; }
            let dmg = (af / tot * atk * 4.5 + 0.5).floor();
            let kill = tu.hp - dmg <= 0.0;
            let ret = if kill || stiff(tu.kind) || g.d(s, t) > rng_t { 0.0 } else { (df / tot * def_t * 4.5 + 0.5).floor() };
            let siege = tt.structure == 1 && tt.owner == me;
            if ret >= su.hp && !siege { continue; }
            let st = &g.tiles[s];
            if kill && rng_s < 2 {
                // would advance off an important tile
                if capturable(s) { continue; }
                if st.structure == 1 && st.owner == me && g.enemy_near(s, 2) > 1 && !spare(s) { continue; }
            }
            let vt = 1.0 + 0.1 * cost_t as f32;
            let vsrc = 1.0 + 0.1 * cost_s as f32;
            let mut sc = dmg.min(tu.hp) * vt - ret * vsrc * 0.9 + if kill { 15.0 } else { 0.0 };
            if tt.owner == me { sc += 4.0; }
            if siege { sc += 60.0; }
            if tt.structure == 1 && tt.owner != me && tt.owner != 0 { sc += 3.0; }
            if sc > 0.0 && best.map_or(true, |b| sc > b.0) { best = Some((sc, a)); }
        }
        if let Some((_, a)) = best { return a.id.clone(); }

        // 5. economy
        let my_units: Vec<Unit> = g.units.iter().filter(|u| u.owner == me).cloned().collect();
        let army = my_units.len() as i32;
        let ncity = my_cities.len() as i32;
        let has_port = (0..n).any(|t| g.own_port(t));
        let stuck = self.stuck_turn >= g.turn - 1 && self.stuck_turn >= 0;
        let mut want_tech: Option<i32> = None;
        let mut want_port = false;
        let water_score = |t: usize| -> i32 {
            let mut seen = vec![false; n];
            let mut stack = vec![t];
            seen[t] = true;
            let mut sc = 0;
            while let Some(c) = stack.pop() {
                sc += if g.tiles[c].known { 1 } else { 3 };
                for nb in g.nbrs(c) {
                    if !seen[nb] && (!g.tiles[nb].known || g.is_water(nb)) { seen[nb] = true; stack.push(nb); }
                }
            }
            sc
        };
        let mut port_act: Option<&Act> = None;
        if stuck {
            if !g.tech[20] && (0..n).any(|t| g.tiles[t].known && g.tiles[t].terrain == 4) { want_tech = Some(20); } else if !g.tech[10] { want_tech = Some(10); } else {
                let exist = (0..n).filter(|&t| g.own_port(t)).map(|t| water_score(t)).max().unwrap_or(0);
                let cand = acts.iter().filter(|a| a.kind == 6 && a.par == 8).map(|a| (water_score((a.tgt.max(0) as usize).min(n - 1)), a)).max_by_key(|x| x.0);
                let cand_best = cand.as_ref().map(|x| x.0).unwrap_or(0);
                let any_cand_tile = (0..n).any(|t| g.tiles[t].owner == me && g.tiles[t].terrain == 1 && g.tiles[t].structure == 0 && water_score(t) > exist);
                if cand_best > exist { want_port = true; port_act = cand.map(|x| x.1); }
                else if any_cand_tile && g.stars < 7 { want_port = true; }
                else if !g.tech[13] { want_tech = Some(13); }
            }
        }
        let _ = has_port;
        // emergency defense
        let summons: Vec<&Act> = acts.iter().filter(|a| a.kind == 4 && a.src >= 0 && (a.src as usize) < n && g.tiles[a.src as usize].structure == 1 && matches!(a.par, 2 | 3 | 5 | 9 | 11 | 4)).collect();
        let pick_unit = |city: i32, defend: bool| -> Option<&Act> {
            let opts: Vec<&&Act> = summons.iter().filter(|a| a.src == city).collect();
            let has = |k: i32| opts.iter().find(|a| a.par == k).map(|a| **a);
            let riders = my_units.iter().filter(|u| u.kind == 3).count() as i32;
            let archers = my_units.iter().filter(|u| u.kind == 9).count() as i32;
            if defend { if let Some(a) = has(5) { return Some(a); } }
            if let Some(a) = has(11) { return Some(a); }
            if !defend {
                if archers * 4 < army { if let Some(a) = has(9) { return Some(a); } }
                if riders * 3 < army + 1 { if let Some(a) = has(3) { return Some(a); } }
            }
            has(2)
        };
        for c in &my_cities {
            if g.unit_at[c.tile] < 0 && g.enemy_near(c.tile, 4) > 0 {
                if let Some(a) = pick_unit(c.tile as i32, true) { return a.id.clone(); }
            }
        }
        if let Some(t) = want_tech {
            if let Some(a) = acts.iter().find(|a| a.kind == 7 && a.par == t) { return a.id.clone(); }
        }
        if let Some(a) = port_act { return a.id.clone(); }
        let saving = want_tech.is_some() || want_port;
        // harvest
        if g.stars >= 9 { if let Some(a) = acts.iter().find(|a| a.kind == 4 && a.par == 44) { return a.id.clone(); } }
        if let Some(a) = acts.iter().find(|a| a.kind == 5) { return a.id.clone(); }
        // units whenever legal
        let front_city = |a: &&&Act| -> i32 {
            let t = a.src as usize;
            let de = enemy_cities.iter().map(|c| g.d(c.tile, t)).min().unwrap_or(0);
            let thr = g.enemy_near(t, 3);
            thr * 100 - de
        };
        if !(saving && army >= ncity + 1) {
            if let Some(a) = summons.iter().max_by_key(front_city) {
                if let Some(b) = pick_unit(a.src, false) { return b.id.clone(); }
            }
        }
        let threat = my_cities.iter().any(|c| g.enemy_near(c.tile, 4) > 0);
        let any_spare = my_cities.iter().any(|c| spare(c.tile));
        if !saving && (!threat || !any_spare || g.stars >= 10) {
            // builds
            let mut bb: Option<(f32, &Act)> = None;
            for a in acts.iter().filter(|a| a.kind == 6) {
                let t = (a.tgt.max(0) as usize).min(n - 1);
                let adj = |k: i32| g.nbrs(t).iter().filter(|&&x| g.tiles[x].structure == k && g.tiles[x].owner == me).count() as f32;
                let sc = match a.par {
                    5 | 21 => 0.4,
                    12 => 0.33,
                    13 => { let h = adj(12); if h >= 2.0 { h / 5.0 } else { 0.0 } }
                    22 => { let m = adj(21); if m >= 1.0 { 2.0 * m / 5.0 } else { 0.0 } }
                    6 => { let f = adj(5); if f >= 2.0 { f / 5.0 } else { 0.0 } }
                    23..=29 => 1.0,
                    8 => if g.stars >= 25 { 0.14 } else { 0.0 },
                    17..=20 => if g.stars >= 45 { 0.05 } else { 0.0 },
                    _ => 0.0,
                };
                if sc > 0.0 && bb.map_or(true, |b| sc > b.0) { bb = Some((sc, a)); }
            }
            if let Some((_, a)) = bb { return a.id.clone(); }
            // research
            {
                let terr_has = |res: i32| (0..n).any(|t| g.tiles[t].owner == me && g.tiles[t].resource == res && g.tiles[t].structure == 0);
                let forest_cnt = (0..n).filter(|&t| g.tiles[t].owner == me && g.tiles[t].terrain == 5 && g.tiles[t].structure == 0).count();
                let mut order: Vec<i32> = Vec::new();
                if terr_has(6) { order.push(6); }
                if forest_cnt >= 2 { order.push(16); }
                if terr_has(3) { order.push(10); }
                if terr_has(2) { order.push(8); }
                if terr_has(5) { order.push(23); }
                order.extend_from_slice(&[6, 16, 20, 23, 24, 7, 8, 17, 18, 1, 9, 10]);
                for t in order {
                    if g.tech[t as usize] { continue; }
                    if let Some(a) = acts.iter().find(|a| a.kind == 7 && a.par == t) { return a.id.clone(); }
                    let prereq_known = match t { 23 => g.tech[20], 24 => g.tech[23], 8 | 7 => g.tech[6], 16 | 18 => g.tech[15], 17 => g.tech[16], 9 => g.tech[8], _ => true };
                    if prereq_known { break; }
                }
                if g.stars >= 40 {
                    if let Some(a) = acts.iter().filter(|a| a.kind == 7).min_by_key(|a| match a.par { 13 => 0, 44 => 1, 21 => 2, 2 => 3, 3 => 4, 14 => 5, _ => 9 }) { return a.id.clone(); }
                }
            }
        }

        // 6. moves
        // objectives: (tile, base, per-dist, exclusive, kind) kind:0 village 1 enemy city 2 ruin 3 enemy unit 4 explore 5 defend
        let mut objs: Vec<(usize, f32, f32, bool, i32)> = Vec::new();
        for t in 0..n {
            let tl = &g.tiles[t];
            if !tl.known {
                // frontier: unknown adjacent to known
                if g.nbrs(t).iter().any(|&x| g.tiles[x].known) { let cnt = g.nbrs(t).iter().filter(|&&x| !g.tiles[x].known).count() as f32; objs.push((t, 48.0 + cnt, 4.0, true, 4)); }
                continue;
            }
            let occ_own = g.unit_at[t] >= 0 && g.units[g.unit_at[t] as usize].owner == me;
            if tl.structure == 1 && tl.owner == 0 { if !occ_own { objs.push((t, 100.0, 6.0, true, 0)); } }
            else if tl.structure == 1 && tl.owner != me { objs.push((t, if army < 6 { 20.0 } else if g.unit_at[t] < 0 { 125.0 } else { 95.0 }, 4.0, false, 1)); }
            else if tl.structure == 2 { if !occ_own { objs.push((t, 60.0, 7.0, true, 2)); } }
        }
        for u in g.units.iter().filter(|u| u.owner != me) {
            let inmine = g.tiles[u.tile].owner == me;
            objs.push((u.tile, if inmine { 85.0 } else { 40.0 }, 6.0, false, 3));
        }
        for c in &my_cities {
            if g.unit_at[c.tile] < 0 && g.enemy_near(c.tile, 3) > 0 { objs.push((c.tile, 150.0, 10.0, true, 5)); }
        }
        let mut best_mv: Option<(f32, &Act, usize, bool)> = None;
        let mut any_stuck = false;
        for u in my_units.iter().filter(|u| !u.moved) {
            let mvs: Vec<&Act> = acts.iter().filter(|a| a.kind == 1 && a.src == u.tile as i32).collect();
            if mvs.is_empty() { continue; }
            let ut = &g.tiles[u.tile];
            if capturable(u.tile) { continue; }
            if ut.structure == 1 && ut.owner == me && g.enemy_near(u.tile, 4) > 0 && !spare(u.tile) { continue; }
            // low hp: recover instead
            if u.hp < maxhp(u) * 0.5 && g.enemy_near(u.tile, 1) == 0 {
                if let Some(a) = acts.iter().find(|a| a.kind == 3 && a.par == 7 && a.src == u.tile as i32) {
                    if best_mv.map_or(true, |b| 200.0 > b.0) { best_mv = Some((200.0, a, usize::MAX, false)); }
                    continue;
                }
            }
            let can_fight = stats(u.kind).1 > 0.0;
            let df = g.bfs(u.tile, false);
            let mut bo: Option<(f32, usize, bool)> = None;
            for &(t, base, per, excl, kind) in &objs {
                if df[t] >= INF || t == u.tile { continue; }
                if excl && self.claimed.contains(&t) { continue; }
                if kind == 3 && (!can_fight || df[t] > 4) { continue; }
                let mut sc = base - per * df[t] as f32;
                if self.goals.get(&u.tile) == Some(&t) { sc += 10.0; }
                if bo.map_or(true, |b| sc > b.0) { bo = Some((sc, t, excl)); }
            }
            let (sc, obj, excl) = match bo { Some(b) => b, None => { any_stuck = true; continue; } };
            let dr = g.bfs(obj, true);
            let cur = dr[u.tile];
            let mut bd: Option<(i32, &Act)> = None;
            for a in mvs {
                let d = (a.tgt.max(0) as usize).min(n - 1);
                if dr[d] >= INF { continue; }
                let mut key = dr[d] * 10;
                let dtl = &g.tiles[d];
                if dtl.terrain == 5 || dtl.terrain == 4 { key -= 1; }
                if d != obj && g.enemy_near(d, 1) > 0 && u.hp < maxhp(u) * 0.6 { key += 6; }
                if bd.map_or(true, |b| key < b.0) { bd = Some((key, a)); }
            }
            if let Some((key, a)) = bd {
                if (key + 1) / 10 < cur {
                    if best_mv.map_or(true, |b| sc > b.0) { best_mv = Some((sc, a, obj, excl)); }
                } else if matches!(u.kind, 43 | 44 | 42 | 41) { any_stuck = true; }
            } else if matches!(u.kind, 43 | 44 | 42 | 41) { any_stuck = true; }
        }
        if any_stuck { self.stuck_turn = g.turn; }
        if let Some((_, a, obj, excl)) = best_mv {
            if obj != usize::MAX && a.kind == 1 {
                self.goals.remove(&(a.src.max(0) as usize));
                self.goals.insert(a.tgt.max(0) as usize, obj);
            }
            if excl && obj != usize::MAX {
                self.claimed.insert(obj);
                if !g.tiles[obj].known { for x in g.nbrs(obj) { if !g.tiles[x].known { self.claimed.insert(x); } } }
            }
            return a.id.clone();
        }
        // 7. idle recover
        if let Some(a) = acts.iter().find(|a| a.kind == 3 && a.par == 7) { return a.id.clone(); }
        fallback
    }
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut bot = Bot { turn: -1, claimed: HashSet::new(), stuck_turn: -10, goals: std::collections::HashMap::new() };
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let id = bot.choose(&obs);
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
