use serde_json::{json, Value};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{self, BufRead, Write};

const INF: u32 = 100000;

#[derive(Clone, Default)]
struct T { known: bool, terrain: i32, owner: i32, structure: i32, resource: i32, capital: i32 }
#[derive(Clone, Default)]
struct U { tile: usize, owner: i32, kind: i32, hp: f32, moved: bool, attacked: bool, vet: bool, passenger: i32, home: i32 }
#[derive(Clone, Default)]
struct C { tile: usize, owner: i32, level: i32, progress: i32, wall: bool }
#[derive(Clone)]
struct A { id: String, kind: i32, src: i32, tgt: i32, par: i32 }

// hp, atk, def, mov, range, cost
fn stats(k: i32) -> (f32, f32, f32, i32, i32, i32) {
    match k {
        2 => (10., 2., 2., 1, 1, 2),
        3 => (10., 2., 1., 2, 1, 3),
        4 => (10., 3.5, 1., 3, 1, 8),
        5 => (15., 1., 3., 1, 1, 3),
        8 => (10., 4., 0., 1, 3, 8),
        9 => (10., 2., 1., 1, 2, 3),
        10 => (10., 0., 1., 1, 1, 5),
        11 => (15., 3., 3., 1, 1, 5),
        12 => (40., 5., 4., 1, 1, 10),
        37 => (10., 2., 2., 1, 1, 2),
        38 => (5., 0., 0.5, 2, 1, 8),
        39 => (5., 0., 0.5, 2, 1, 2),
        40 => (10., 2., 2., 2, 1, 2),
        41 => (10., 3., 2., 2, 3, 15),
        42 => (10., 2., 1., 3, 2, 5),
        43 => (10., 0., 1., 2, 2, 0),
        44 => (10., 3., 3., 3, 1, 5),
        45 => (40., 4., 4., 2, 1, 10),
        _ => (10., 2., 2., 1, 1, 2),
    }
}
fn fortify(k: i32) -> bool { matches!(k, 2 | 3 | 4 | 5 | 9) }
fn stiff(k: i32) -> bool { matches!(k, 8 | 10 | 38 | 39 | 43 | 41 | 45) }
fn tier(t: i32) -> i32 {
    match t { 1 | 6 | 10 | 15 | 20 => 1, 2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2, _ => 3 }
}

struct G {
    size: usize, n: usize, me: i32, turn: i32, stars: i32,
    tiles: Vec<T>, units: Vec<U>, uat: Vec<i32>, cities: Vec<C>, cat: Vec<i32>,
    tech: [bool; 128], legal: Vec<A>,
}

impl G {
    fn xy(&self, i: usize) -> (i32, i32) { ((i % self.size) as i32, (i / self.size) as i32) }
    fn d(&self, a: usize, b: usize) -> i32 { let (ax, ay) = self.xy(a); let (bx, by) = self.xy(b); (ax - bx).abs().max((ay - by).abs()) }
    fn nb(&self, i: usize) -> Vec<usize> {
        let (x, y) = self.xy(i); let s = self.size as i32; let mut v = Vec::with_capacity(8);
        for dy in -1..=1 { for dx in -1..=1 { if dx == 0 && dy == 0 { continue; } let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && nx < s && ny < s { v.push((ny * s + nx) as usize); } } }
        v
    }
    fn water(&self, i: usize) -> bool { let t = self.tiles[i].terrain; t == 1 || t == 2 }
    fn pass(&self, i: usize) -> bool {
        let t = &self.tiles[i];
        if !t.known { return false; }
        match t.terrain { 3 | 5 => true, 4 => self.tech[20], 1 => self.tech[10], 2 => self.tech[13], _ => false }
    }
    fn own_port(&self, i: usize) -> bool { self.tiles[i].structure == 8 && self.tiles[i].owner == self.me }
    fn edge(&self, a: usize, b: usize) -> Option<u32> {
        // cost of moving a->b, None if impossible
        if !self.pass(b) { return None; }
        if self.water(b) {
            if self.water(a) { return Some(1); }
            if self.own_port(b) { return Some(4); }
            return None;
        }
        Some(1)
    }
    fn field(&self, src: usize, reverse: bool) -> Vec<u32> {
        let mut dist = vec![INF; self.n];
        let mut h = BinaryHeap::new();
        dist[src] = 0; h.push(Reverse((0u32, src)));
        while let Some(Reverse((d, x))) = h.pop() {
            if d > dist[x] { continue; }
            for y in self.nb(x) {
                if !self.tiles[y].known { continue; }
                let c = if reverse {
                    // edge y -> x ; goal itself always enterable
                    if x == src && !self.pass(x) { Some(1) } else { self.edge(y, x) }
                } else { self.edge(x, y) };
                if let Some(c) = c { let nd = d + c; if nd < dist[y] { dist[y] = nd; h.push(Reverse((nd, y))); } }
            }
        }
        dist
    }
    fn maxhp(&self, u: &U) -> f32 { let k = if u.passenger > 0 { u.passenger } else { u.kind }; stats(k).0 + if u.vet { 5. } else { 0. } }
    fn bonus(&self, u: &U) -> f32 {
        let t = &self.tiles[u.tile]; let mine = u.owner == self.me;
        match t.terrain {
            1 | 2 => if mine && self.tech[12] { 1.5 } else { 1.0 },
            5 => if mine && self.tech[18] { 1.5 } else { 1.0 },
            4 => if !mine || self.tech[20] { 1.5 } else { 1.0 },
            _ => {
                let ci = self.cat[u.tile];
                if ci >= 0 && self.cities[ci as usize].owner == u.owner && fortify(u.kind) {
                    if self.cities[ci as usize].wall { 4.0 } else { 1.5 }
                } else { 1.0 }
            }
        }
    }
    fn combat(&self, a: &U, d: &U) -> (f32, f32) {
        let sa = stats(a.kind); let sd = stats(d.kind);
        let af = sa.1 * (a.hp / self.maxhp(a)).min(1.0);
        let df = sd.2 * (d.hp / self.maxhp(d)).min(1.0) * self.bonus(d);
        let tot = af + df; if tot <= 0.0 { return (0., 0.); }
        let dmg = (af / tot * sa.1 * 4.5 + 0.5).floor();
        let ret = if d.hp > dmg { (df / tot * sd.2 * 4.5 + 0.5).floor() } else { 0. };
        (dmg, ret)
    }
    fn find(&self, kind: i32, src: i32, tgt: i32, par: i32) -> Option<String> {
        self.legal.iter().find(|a| a.kind == kind && (src == -9 || a.src == src) && (tgt == -9 || a.tgt == tgt) && (par == -9 || a.par == par)).map(|a| a.id.clone())
    }
}

fn parse(obs: &Value) -> G {
    let vs = &obs["visible_state"];
    let size = vs["board_size"].as_u64().unwrap_or(14) as usize; let n = size * size;
    let me = obs["player"].as_i64().unwrap_or(1) as i32;
    let gi = |v: &Value, k: &str| v[k].as_i64().unwrap_or(0) as i32;
    let gb = |v: &Value, k: &str| v[k].as_bool().unwrap_or(false);
    let mut tiles = vec![T::default(); n];
    if let Some(a) = vs["tiles"].as_array() { for t in a {
        let i = t["tile"].as_u64().unwrap_or(0) as usize; if i >= n { continue; }
        tiles[i] = T { known: gb(t, "known"), terrain: gi(t, "terrain"), owner: gi(t, "owner"), structure: gi(t, "structure"), resource: gi(t, "resource"), capital: gi(t, "capital") };
    } }
    let mut units = vec![]; let mut uat = vec![-1i32; n];
    if let Some(a) = vs["units"].as_array() { for u in a {
        let i = u["tile"].as_u64().unwrap_or(0) as usize; if i >= n { continue; }
        uat[i] = units.len() as i32;
        units.push(U { tile: i, owner: gi(u, "owner"), kind: gi(u, "kind"), hp: u["health"].as_f64().unwrap_or(10.) as f32, moved: gb(u, "moved"), attacked: gb(u, "attacked"), vet: gb(u, "veteran"), passenger: gi(u, "passenger"), home: gi(u, "home") });
    } }
    let mut cities = vec![]; let mut cat = vec![-1i32; n];
    if let Some(a) = vs["cities"].as_array() { for c in a {
        let i = c["tile"].as_u64().unwrap_or(0) as usize; if i >= n { continue; }
        cat[i] = cities.len() as i32;
        let wall = c["rewards"].as_array().map(|r| r.get(1).and_then(|x| x.as_i64()).unwrap_or(0) > 0).unwrap_or(false);
        cities.push(C { tile: i, owner: gi(c, "owner"), level: gi(c, "level"), progress: gi(c, "progress"), wall });
    } }
    let mut tech = [false; 128];
    if let Some(a) = vs["technology"].as_array() { for t in a { let k = t.as_u64().unwrap_or(0) as usize; if k < 128 { tech[k] = true; } } }
    let mut legal = vec![];
    if let Some(a) = obs["legal_actions"].as_array() { for l in a {
        let c = &l["command"];
        legal.push(A { id: l["action_id"].as_str().unwrap_or("").to_owned(), kind: gi(c, "kind"), src: gi(c, "source"), tgt: gi(c, "target"), par: gi(c, "parameter") });
    } }
    G { size, n, me, turn: gi(vs, "turn"), stars: gi(vs, "stars"), tiles, units, uat, cities, cat, tech, legal }
}

struct Mem { panic: bool, max_cities: usize, naval: bool, last_known: usize, last_prog: i32, last_cities: usize }

fn choose(g: &G, mem: &mut Mem) -> Option<String> {
    let me = g.me;
    // 1. rewards
    let rewards: Vec<&A> = g.legal.iter().filter(|a| a.kind == 9).collect();
    if !rewards.is_empty() {
        let has_en = g.units.iter().any(|u| u.owner != me) || g.cities.iter().any(|c| c.owner != me && c.owner != 0);
        let pref = |p: i32| match p { 6 => 10, 3 => 9, 7 => 8, 5 => 7, 8 => 6, 2 => 5, 4 => 4, 1 => if has_en { 9 } else { 3 }, _ => 0 };
        let iscap = |a: &A| a.tgt >= 0 && (a.tgt as usize) < g.n && g.tiles[a.tgt as usize].capital == me;
        return rewards.iter().max_by_key(|a| if a.par == 1 && iscap(a) { 9 } else { pref(a.par) }).map(|a| a.id.clone());
    }
    let my_cities: Vec<&C> = g.cities.iter().filter(|c| c.owner == me).collect();
    let en_cities: Vec<&C> = g.cities.iter().filter(|c| c.owner != me && c.owner != 0).collect();
    let my_units: Vec<&U> = g.units.iter().filter(|u| u.owner == me).collect();
    let en_units: Vec<&U> = g.units.iter().filter(|u| u.owner != me).collect();
    let ncity = my_cities.len() as i32;
    let nunits = my_units.len() as i32;

    let my_mil: i32 = my_units.iter().map(|u| stats(u.kind).5.max(2)).sum();
    let en_mil: i32 = en_units.iter().map(|u| stats(u.kind).5.max(2)).sum();
    if my_cities.len() > mem.max_cities { mem.max_cities = my_cities.len(); }
    if my_cities.len() < mem.max_cities || (en_mil > my_mil * 2 && en_mil >= 20) { mem.panic = true; }
    let panic = mem.panic;
    // fortress city: highest level city
    let fort: i32 = if panic { my_cities.iter().max_by_key(|c| ((g.tiles[c.tile].capital == me) as i32 + c.wall as i32 * 2, c.level, std::cmp::Reverse(c.tile))).map(|c| c.tile as i32).unwrap_or(-1) } else { -1 };
    let fort_ok = fort >= 0 && g.uat[fort as usize] >= 0 && { let u = &g.units[g.uat[fort as usize] as usize]; u.owner == me && fortify(u.kind) };
    // promote
    if let Some(id) = g.find(3, -9, -9, 16) { return Some(id); }

    // threatened cities
    let mut threat: HashMap<usize, i32> = HashMap::new();
    for c in &my_cities {
        let cnt = en_units.iter().filter(|e| g.d(e.tile, c.tile) <= 2).count() as i32;
        if cnt > 0 { threat.insert(c.tile, cnt); }
    }

    // ---- goals / fields
    let villages: Vec<usize> = g.cities.iter().filter(|c| c.owner == 0).map(|c| c.tile).collect();
    let ruins: Vec<usize> = (0..g.n).filter(|&i| g.tiles[i].known && g.tiles[i].structure == 2).collect();
    let frontier: Vec<usize> = (0..g.n).filter(|&i| g.tiles[i].known && g.nb(i).iter().any(|&j| !g.tiles[j].known)).collect();
    let mut fwd: HashMap<usize, Vec<u32>> = HashMap::new();
    for u in &my_units { fwd.insert(u.tile, g.field(u.tile, false)); }
    let mut stuck = !my_units.is_empty();
    'o: for u in &my_units {
        let f = &fwd[&u.tile];
        for &t in villages.iter().chain(frontier.iter()).chain(en_cities.iter().map(|c| &c.tile)) {
            if f[t] < INF && t != u.tile { stuck = false; break 'o; }
        }
    }
    let known_cnt = (0..g.n).filter(|&i| g.tiles[i].known).count();
    if known_cnt != mem.last_known || my_cities.len() != mem.last_cities { mem.last_known = known_cnt; mem.last_cities = my_cities.len(); mem.last_prog = g.turn; }
    let en_reach = my_units.iter().any(|u| en_cities.iter().any(|c| fwd[&u.tile][c.tile] < INF));
    if !stuck && g.turn - mem.last_prog >= 5 && !en_reach && !my_units.is_empty() { stuck = true; }
    let has_port = (0..g.n).any(|i| g.own_port(i));
    let water_known = (0..g.n).any(|i| g.tiles[i].known && g.water(i));

    // core tech
    let mut core: Vec<i32> = vec![];
    if stuck && water_known { mem.naval = true; }
    // port site evaluation
    let mut minf = vec![INF; g.n];
    for u in &my_units { let f = &fwd[&u.tile]; for i in 0..g.n { if f[i] < minf[i] { minf[i] = f[i]; } } }
    let wpass = |i: usize| g.tiles[i].known && ((g.tiles[i].terrain == 1 && g.tech[10]) || (g.tiles[i].terrain == 2 && g.tech[13]));
    let port_score = |p: usize| -> i32 {
        let mut seen = vec![false; g.n]; let mut st = vec![p]; seen[p] = true; let mut sc = 0;
        let mut landseen = vec![false; g.n];
        while let Some(x) = st.pop() {
            for y in g.nb(x) {
                if !g.tiles[y].known { if !landseen[x] { landseen[x] = true; sc += 1; } continue; }
                if g.water(y) { if !seen[y] && wpass(y) { seen[y] = true; st.push(y); } }
                else if !landseen[y] { landseen[y] = true; if minf[y] >= INF && g.pass(y) { sc += 2; if g.cat[y] >= 0 { sc += 10; } } }
            }
        }
        sc
    };
    let mut port_build: Option<String> = None;
    if stuck && water_known && g.tech[10] {
        let mut b: Option<(i32, String)> = None;
        for a in g.legal.iter().filter(|a| a.kind == 6 && a.par == 8) {
            let t = a.tgt as usize; if t >= g.n { continue; }
            let sc = port_score(t);
            if sc > 0 && b.as_ref().map(|x| sc > x.0).unwrap_or(true) { b = Some((sc, a.id.clone())); }
        }
        port_build = b.map(|x| x.1);
    }
    if stuck && water_known { if !g.tech[10] { core.push(10); } else if !g.tech[13] && port_build.is_none() && (g.stars >= 7 || has_port) { core.push(13); } }
    if mem.naval && g.tech[10] && g.tech[13] && !g.tech[44] { core.push(44); }
    let fruit = (0..g.n).filter(|&i| g.tiles[i].owner == me && g.tiles[i].resource == 6 && g.tiles[i].structure == 0).count();
    let fish = (0..g.n).filter(|&i| g.tiles[i].owner == me && g.tiles[i].resource == 3 && g.tiles[i].structure == 0).count();
    if fruit >= 1 && !g.tech[6] { core.push(6); }
    if fish >= 2 && !g.tech[10] { core.push(10); }
    let forests = (0..g.n).filter(|&i| g.tiles[i].owner == me && g.tiles[i].terrain == 5 && g.tiles[i].structure == 0 && g.tiles[i].resource != 1).count();
    if forests >= 3 && !g.tech[16] { core.push(16); }
    if panic { for t in [6, 7] { if !g.tech[t as usize] { core.push(t); } } }
    for t in [20, 23, 24] { if !g.tech[t as usize] { core.push(t); } }
    let next_core = core.first().copied();
    let contact = !en_cities.is_empty() || !en_units.is_empty();
    let min_army = if contact { ncity * 2 + 1 } else { ncity + 2 + (villages.len() as i32).min(2) };
    let tech_open = contact || g.turn >= 4 || ncity >= 2 || stuck;
    let research_now = |t: i32| -> Option<String> { g.find(7, -9, -9, t) };
    if let Some(t) = next_core {
        if nunits >= min_army.min(3) && g.turn >= 1 && threat.is_empty() && (tech_open || t == 6) {
            if let Some(id) = research_now(t) { return Some(id); }
        }
    }

    // 2. capture
    if let Some(a) = g.legal.iter().find(|a| a.kind == 8) { return Some(a.id.clone()); }

    // 4. attacks
    let mut best: Option<(f32, String)> = None;
    let mut pot: HashMap<usize, f32> = HashMap::new();
    for a in g.legal.iter().filter(|a| a.kind == 2) {
        let (s, t) = (a.src as usize, a.tgt as usize);
        if s >= g.n || t >= g.n || g.uat[s] < 0 || g.uat[t] < 0 { continue; }
        let (dmg, _) = g.combat(&g.units[g.uat[s] as usize], &g.units[g.uat[t] as usize]);
        *pot.entry(t).or_insert(0.0) += dmg;
    }
    for a in g.legal.iter().filter(|a| a.kind == 2) {
        let (s, t) = (a.src as usize, a.tgt as usize);
        if s >= g.n || t >= g.n || g.uat[s] < 0 || g.uat[t] < 0 { continue; }
        let au = &g.units[g.uat[s] as usize]; let du = &g.units[g.uat[t] as usize];
        if au.kind == 10 { // convert
            let sc = 20.0 + stats(du.kind).5 as f32;
            if best.as_ref().map(|b| sc > b.0).unwrap_or(true) { best = Some((sc, a.id.clone())); }
            continue;
        }
        let (dmg, ret) = g.combat(au, du);
        let kill = dmg >= du.hp;
        let sa = stats(au.kind); let sd = stats(du.kind);
        let dist = g.d(s, t);
        let ret_ok = !kill && dist <= sd.4 && !stiff(du.kind);
        let ret = if ret_ok { ret } else { 0. };
        let dies = ret >= au.hp;
        let near_city = my_cities.iter().any(|c| g.d(c.tile, t) <= 2);
        let on_city = g.cat[t] >= 0;
        let mut sc = dmg.min(du.hp) * (1.0 + sd.5 as f32 / 10.0) * if near_city { 1.5 } else { 1.0 };
        if kill { sc += 6.0 + sd.5 as f32; }
        if on_city { sc += 3.0; }
        sc -= ret * (0.8 + sa.5 as f32 / 10.0);
        if dies { sc -= 8.0 + 2.0 * sa.5 as f32; }
        else if !kill && pot.get(&t).copied().unwrap_or(0.0) >= du.hp && dmg >= 1.0 { sc = sc.max(1.0) + 6.0 + dmg; }
        if fort >= 0 && s == fort as usize && kill && sa.4 < 2 { sc -= 100.0; }
        // garrison: do not leave a threatened city by advancing
        if kill && sa.4 < 2 && threat.get(&s).copied().unwrap_or(0) > 1 { sc -= 8.0; }
        if sc > 0.5 && best.as_ref().map(|b| sc > b.0).unwrap_or(true) { best = Some((sc, a.id.clone())); }
    }
    if let Some((_, id)) = best { return Some(id); }

    // 5. moves
    let mut moves: HashMap<usize, Vec<(usize, String)>> = HashMap::new();
    for a in g.legal.iter().filter(|a| a.kind == 1) { moves.entry(a.src as usize).or_default().push((a.tgt as usize, a.id.clone())); }
    let mut movers: Vec<&U> = my_units.iter().copied().filter(|u| moves.contains_key(&u.tile)).collect();
    movers.sort_by_key(|u| u.tile);
    let mut goal: HashMap<usize, usize> = HashMap::new(); // unit tile -> goal tile
    let mut stay: HashSet<usize> = HashSet::new();
    // garrisons
    for (&c, _) in threat.iter() {
        if g.uat[c] >= 0 && g.units[g.uat[c] as usize].owner == me {
            let cnt = my_units.iter().filter(|u| u.home == c as i32 + 1).count() as i32;
            let lvl = if g.cat[c] >= 0 { g.cities[g.cat[c] as usize].level } else { 0 };
            let adj = en_units.iter().any(|e| g.d(e.tile, c) <= 1);
            let can_train = cnt <= lvl && g.stars >= 5 && !adj;
            if !can_train { stay.insert(c); }
            continue;
        }
        let mut bu: Option<(u32, usize)> = None;
        for u in &movers { if stay.contains(&u.tile) || goal.contains_key(&u.tile) { continue; }
            let d = fwd[&u.tile][c]; if d <= 3 && bu.map(|b| d < b.0).unwrap_or(true) { bu = Some((d, u.tile)); } }
        if let Some((_, ut)) = bu { goal.insert(ut, c); }
    }
    if fort_ok { stay.insert(fort as usize); }
    // units standing on capturable things stay
    for u in &my_units {
        let ci = g.cat[u.tile];
        if (ci >= 0 && g.cities[ci as usize].owner != me) || g.tiles[u.tile].structure == 2 { stay.insert(u.tile); }
    }
    // low hp units rest
    for u in &movers {
        if u.hp < g.maxhp(u) * 0.45 && !en_units.iter().any(|e| g.d(e.tile, u.tile) <= 1) { stay.insert(u.tile); }
    }
    // village / ruin targets
    let mut targets: Vec<usize> = vec![];
    for &v in villages.iter().chain(ruins.iter()) {
        if g.uat[v] >= 0 && g.units[g.uat[v] as usize].owner == me { continue; }
        targets.push(v);
    }
    // undefended enemy cities are prime targets too
    for c in &en_cities { if g.uat[c.tile] < 0 { targets.push(c.tile); } }
    let mut pairs: Vec<(u32, usize, usize)> = vec![];
    for &t in &targets { for u in &movers {
        if stay.contains(&u.tile) || goal.contains_key(&u.tile) { continue; }
        if matches!(u.kind, 12) && g.cat[t] < 0 { continue; }
        let d = fwd[&u.tile][t]; if d < INF { pairs.push((d, t, u.tile)); }
    } }
    pairs.sort();
    let mut taken: HashSet<usize> = HashSet::new();
    for (_, t, ut) in pairs { if taken.contains(&t) || goal.contains_key(&ut) { continue; } taken.insert(t); goal.insert(ut, t); }
    // rest: enemy units in my land, enemy cities, explore
    let mut picked_frontier: Vec<usize> = vec![];
    let mut k = 0usize;
    let mut rest: Vec<&U> = movers.iter().copied().filter(|u| !stay.contains(&u.tile) && !goal.contains_key(&u.tile)).collect();
    rest.sort_by_key(|u| { let f = &fwd[&u.tile]; (frontier.iter().map(|&t| f[t]).min().unwrap_or(INF), u.tile) });
    for u in &rest {
        if stay.contains(&u.tile) || goal.contains_key(&u.tile) { continue; }
        let f = &fwd[&u.tile];
        k += 1;
        // intruders
        let mut bi: Option<(u32, usize)> = None;
        for e in &en_units { if (g.tiles[e.tile].owner == me || my_cities.iter().any(|c| g.d(c.tile, e.tile) <= 3)) && f[e.tile] <= 4 {
            if bi.map(|b| f[e.tile] < b.0).unwrap_or(true) { bi = Some((f[e.tile], e.tile)); } } }
        if let Some((_, t)) = bi { goal.insert(u.tile, t); continue; }
        let mut bc: Option<(u32, usize)> = None;
        for c in &en_cities { let d = f[c.tile]; if d < INF && bc.map(|b| d < b.0).unwrap_or(true) { bc = Some((d, c.tile)); } }
        let mut bf: Option<(u32, usize)> = None;
        for &t in &frontier { let d = f[t]; if d >= INF || t == u.tile && false { continue; }
            if picked_frontier.iter().any(|&p| g.d(p, t) <= 3) { continue; }
            if bf.map(|b| d < b.0).unwrap_or(true) { bf = Some((d, t)); } }
        let explore = bc.is_none() || (k % 4 == 0 && bf.is_some());
        if !explore { if let Some((_, t)) = bc { goal.insert(u.tile, t); continue; } }
        if bf.is_none() { for &t in &frontier { let d = f[t]; if d >= INF { continue; } if bf.map(|b| d < b.0).unwrap_or(true) { bf = Some((d, t)); } } }
        if let Some((_, t)) = bf { picked_frontier.push(t); goal.insert(u.tile, t); continue; }
        // nearest enemy unit
        let mut be: Option<(u32, usize)> = None;
        for e in &en_units { let d = f[e.tile]; if d < INF && be.map(|b| d < b.0).unwrap_or(true) { be = Some((d, e.tile)); } }
        if let Some((_, t)) = be { goal.insert(u.tile, t); }
    }
    let mut rev: HashMap<usize, Vec<u32>> = HashMap::new();
    let mut bestmove: Option<(i32, String)> = None;
    for u in &movers {
        if stay.contains(&u.tile) { continue; }
        let gt = match goal.get(&u.tile) { Some(&t) => t, None => continue };
        if gt == u.tile { continue; }
        let r = rev.entry(gt).or_insert_with(|| g.field(gt, true));
        let cur = r[u.tile];
        let mut bm: Option<(i32, &String)> = None;
        let is_front = g.tiles[gt].known && g.cat[gt] < 0 && g.tiles[gt].structure != 2;
        for (t, id) in &moves[&u.tile] {
            let d = r[*t]; if d >= INF { continue; }
            let mut sc = d as i32 * 10;
            if is_front { sc -= 4 * g.nb(*t).iter().filter(|&&j| !g.tiles[j].known).count() as i32; }
            let adj_e = en_units.iter().filter(|e| g.d(e.tile, *t) <= 1).count() as i32;
            if u.hp < g.maxhp(u) * 0.6 { sc += adj_e * 6; }
            if g.tiles[*t].terrain == 4 || (g.tiles[*t].terrain == 5 && g.tech[18]) { sc -= 1; }
            // don't step on a foreign city tile by accident is fine; avoid ports unless on path
            if bm.map(|b| sc < b.0).unwrap_or(true) { bm = Some((sc, id)); }
        }
        if let Some((sc, id)) = bm {
            let curs = if cur >= INF { i32::MAX } else { cur as i32 * 10 };
            if sc < curs {
                // prioritise units closest to goal first (so they clear the way)
                let pr = sc;
                if bestmove.as_ref().map(|b| pr < b.0).unwrap_or(true) { bestmove = Some((pr, id.clone())); }
            }
        }
    }
    if let Some((_, id)) = bestmove { return Some(id); }

    // 6. economy
    let stars = g.stars;
    // monuments
    if let Some(a) = g.legal.iter().find(|a| a.kind == 6 && (23..=29).contains(&a.par)) { return Some(a.id.clone()); }
    let summon_at = |city: usize, prefs: &[i32]| -> Option<String> {
        for &p in prefs { if let Some(id) = g.find(4, city as i32, -9, p) { return Some(id); } }
        None
    };
    if fort >= 0 && g.uat[fort as usize] < 0 { if let Some(id) = summon_at(fort as usize, &[5, 2]) { return Some(id); } }
    // threatened empty cities
    for (&c, _) in threat.iter() {
        if g.uat[c] < 0 { if let Some(id) = summon_at(c, &[11, 5, 2, 9, 3]) { return Some(id); } }
    }
    // port when stuck
    let mut reserve = 0;
    if stuck && water_known && g.tech[10] {
        if let Some(id) = port_build.clone() { return Some(id); }
        if g.stars < 7 { reserve = 7; }
    }
    if stars >= reserve + 5 { if let Some(id) = g.find(4, -9, -9, 44) { return Some(id); } }
    let train_any = |need_front: bool| -> Option<String> {
        let mut bc: Option<(i32, String)> = None;
        for c in &my_cities {
            let riders = my_units.iter().filter(|u| u.kind == 3).count();
            let expanding = !villages.is_empty() || (en_cities.is_empty() && !frontier.is_empty());
            let prefs: Vec<i32> = if expanding && riders < 2 && !contact { vec![3, 11, 2] }
                else if g.turn % 5 == 4 { vec![8, 4, 11, 9, 2, 3] }
                else if g.turn % 3 == 1 && !g.tech[24] { vec![11, 9, 2, 3] }
                else if g.tech[24] && threat.is_empty() && nunits >= ncity { vec![11] }
                else { vec![11, 2, 3] };
            if let Some(id) = summon_at(c.tile, &prefs) {
                let mut dmin = 50;
                for e in &en_cities { dmin = dmin.min(g.d(e.tile, c.tile)); }
                for e in &en_units { dmin = dmin.min(g.d(e.tile, c.tile)); }
                let sc = if need_front { dmin } else { -c.level };
                if bc.as_ref().map(|b| sc < b.0).unwrap_or(true) { bc = Some((sc, id)); }
            }
        }
        bc.map(|b| b.1)
    };
    // min army
    if nunits < min_army && stars >= reserve + 2 { if let Some(id) = train_any(contact) { return Some(id); } }
    // harvest
    if stars >= reserve + 2 {
        let mut bh: Option<(i32, String)> = None;
        for a in g.legal.iter().filter(|a| a.kind == 5) {
            let t = a.tgt as usize;
            let mut sc = 100;
            for c in &my_cities { let d = g.d(c.tile, t); if d <= 2 { sc = sc.min((c.level + 1 - c.progress) * 3 + d); } }
            if bh.as_ref().map(|b| sc < b.0).unwrap_or(true) { bh = Some((sc, a.id.clone())); }
        }
        if let Some((_, id)) = bh { return Some(id); }
    }
    // core research
    if let Some(t) = next_core {
        let cost = 4 + ncity * tier(t);
        if tech_open || t == 6 {
            if let Some(id) = research_now(t) { if threat.is_empty() || stars >= cost + 4 { return Some(id); } }
            if nunits >= min_army { reserve = reserve.max(cost); }
        }
    }
    // buildings
    let mut bb: Option<(i32, String)> = None;
    for a in g.legal.iter().filter(|a| a.kind == 6) {
        let t = a.tgt as usize; if t >= g.n { continue; }
        let adj = |s: i32| g.nb(t).iter().filter(|&&j| g.tiles[j].structure == s && g.tiles[j].owner == me).count() as i32;
        let (val, cost) = match a.par {
            5 => (20, 5), 21 => (20, 5),
            12 => (if g.tiles[t].resource == 1 { 0 } else { 14 }, 3),
            22 => (if adj(21) >= 1 { 18 * adj(21) } else { 0 }, 5),
            13 => (if adj(12) >= 2 { 9 * adj(12) } else { 0 }, 5),
            6 => (if adj(5) >= 2 { 9 * adj(5) } else { 0 }, 5),
            8 => (if stars >= 20 && !has_port && g.tech[13] { 5 } else { 0 }, 7),
            17 | 18 | 19 | 20 => (if stars >= 40 { 3 } else { 0 }, 20),
            _ => (0, 0),
        };
        if val > 0 && stars - cost >= reserve && bb.as_ref().map(|b| val > b.0).unwrap_or(true) { bb = Some((val, a.id.clone())); }
    }
    if let Some((_, id)) = bb { return Some(id); }
    // army up to target
    let army_target = if contact { 100 } else { 2 * ncity + 2 };
    if nunits < army_target && stars >= reserve + 2 {
        // keep enough for reserve
        let mut ok = true;
        if reserve > 0 && stars - 5 < reserve { ok = stars - 2 >= reserve && g.find(4, -9, -9, 11).is_none(); }
        if ok { if let Some(id) = train_any(contact) { return Some(id); } }
    }
    // optional research
    if next_core.is_none() || stars >= reserve + 12 {
        for t in [6, 7, 16, 18, 8, 1, 17, 10, 2, 3, 9, 13, 44, 21, 22, 14, 12, 19] {
            if g.tech[t as usize] { continue; }
            let cost = 4 + ncity * tier(t);
            if stars - cost >= reserve.min(8) + 3 { if let Some(id) = research_now(t) { return Some(id); } }
        }
    }
    if stars >= reserve + 10 { if let Some(id) = train_any(contact) { return Some(id); } }
    None
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Mem { panic: false, max_cities: 0, naval: false, last_known: 0, last_prog: 0, last_cities: 0 };
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => Value::Null };
        let g = parse(&obs);
        let mut id = choose(&g, &mut mem);
        if let Some(ref i) = id { if !g.legal.iter().any(|a| &a.id == i) { id = None; } }
        let id = id.or_else(|| g.find(10, -9, -9, -9))
            .or_else(|| g.legal.iter().find(|a| a.kind != 11).map(|a| a.id.clone()))
            .or_else(|| g.legal.first().map(|a| a.id.clone()))
            .unwrap_or_default();
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
