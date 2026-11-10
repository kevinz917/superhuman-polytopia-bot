use serde_json::{json, Value};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{self, BufRead, Write};

#[derive(Clone)]
struct U { tile: usize, owner: i64, kind: i64, hp: f64, moved: bool, attacked: bool, vet: bool, passenger: i64, home: i64 }
struct A { id: String, k: i64, s: i64, t: i64, p: i64 }
struct Mem { skip: HashSet<usize>, claims: HashMap<usize, i32>, turn: i64, acts: u32 }

const INF: i32 = 1_000_000;

fn stats(kind: i64) -> (f64, f64, f64, i64) {
    match kind {
        2 => (2.0, 2.0, 10.0, 1), 3 => (2.0, 1.0, 10.0, 1), 4 => (3.5, 1.0, 10.0, 1), 5 => (1.0, 3.0, 15.0, 1),
        8 => (4.0, 0.0, 10.0, 3), 9 => (2.0, 1.0, 10.0, 2), 10 => (0.0, 1.0, 10.0, 1), 11 => (3.0, 3.0, 15.0, 1),
        12 => (5.0, 4.0, 40.0, 1), 37 => (2.0, 2.0, 10.0, 1), 38 => (0.0, 0.5, 5.0, 1), 39 => (0.0, 0.5, 5.0, 1),
        40 => (2.0, 2.0, 10.0, 1), 41 => (3.0, 2.0, 10.0, 3), 42 => (2.0, 1.0, 10.0, 2), 43 => (0.0, 1.0, 10.0, 2),
        44 => (3.0, 3.0, 10.0, 1), 45 => (4.0, 4.0, 40.0, 1), _ => (2.0, 2.0, 10.0, 1),
    }
}
fn stiff(kind: i64) -> bool { matches!(kind, 8 | 10 | 38 | 39 | 41 | 43 | 45) }
fn maxhp(u: &U) -> f64 {
    let k = if u.passenger > 0 { u.passenger } else { u.kind };
    stats(k).2 + if u.vet { 5.0 } else { 0.0 }
}

struct G {
    size: usize, n: usize, me: i64, turn: i64, stars: i64, tech: [bool; 128],
    ter: Vec<i64>, own: Vec<i64>, st: Vec<i64>, res: Vec<i64>, road: Vec<bool>,
    units: Vec<U>, uat: Vec<Option<usize>>, my_cities: Vec<usize>, clevel: HashMap<usize, i64>,
}

impl G {
    fn d(&self, a: usize, b: usize) -> i64 {
        let s = self.size as i64;
        let (ax, ay, bx, by) = (a as i64 % s, a as i64 / s, b as i64 % s, b as i64 / s);
        (ax - bx).abs().max((ay - by).abs())
    }
    fn nb(&self, i: usize) -> Vec<usize> {
        let s = self.size as i64;
        let (x, y) = (i as i64 % s, i as i64 / s);
        let mut v = Vec::with_capacity(8);
        for dy in -1..=1 { for dx in -1..=1 {
            if dx == 0 && dy == 0 { continue; }
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && nx < s && ny >= 0 && ny < s { v.push((ny * s + nx) as usize); }
        } }
        v
    }
    fn water(&self, i: usize) -> bool { self.ter[i] == 1 || self.ter[i] == 2 }
    // cost of stepping a -> b, None if impossible
    fn step(&self, a: usize, b: usize, water_ok: bool) -> Option<i32> {
        let tb = self.ter[b];
        if tb < 0 { return Some(3); }
        if tb == 1 || tb == 2 {
            if !water_ok { return None; }
            let port = self.st[b] == 8 && self.own[b] == self.me;
            let from_water = self.water(a) || self.ter[a] < 0;
            if port { return Some(if from_water { 2 } else { 6 }); }
            if !from_water { return None; }
            if tb == 1 && !self.tech[10] { return None; }
            if tb == 2 && !self.tech[13] { return None; }
            return Some(2);
        }
        if tb == 4 { return if self.tech[20] { Some(4) } else { None }; }
        if let Some(ui) = self.uat[b] { if self.units[ui].owner != self.me { return Some(6); } }
        if self.water(a) { return Some(3); }
        if tb == 5 { return Some(if self.road[b] { 2 } else { 4 }); }
        Some(if self.road[b] || self.st[b] == 1 { 1 } else { 2 })
    }
    fn dijkstra(&self, src: &[usize], reverse: bool, water_ok: bool) -> Vec<i32> {
        let mut dist = vec![INF; self.n];
        let mut h = BinaryHeap::new();
        for &s in src { dist[s] = 0; h.push((Reverse(0i32), s)); }
        while let Some((Reverse(c), at)) = h.pop() {
            if c > dist[at] { continue; }
            for b in self.nb(at) {
                let w = if reverse { self.step(b, at, water_ok) } else { self.step(at, b, water_ok) };
                if let Some(w) = w {
                    let nd = c + w;
                    if nd < dist[b] { dist[b] = nd; h.push((Reverse(nd), b)); }
                }
            }
        }
        dist
    }
    fn enemy_near(&self, t: usize, r: i64) -> bool {
        self.units.iter().any(|u| u.owner != self.me && self.d(u.tile, t) <= r)
    }
}

fn parse(obs: &Value) -> (G, Vec<A>) {
    let v = &obs["visible_state"];
    let size = v["board_size"].as_u64().unwrap_or(14) as usize;
    let n = size * size;
    let me = obs["player"].as_i64().unwrap_or(1);
    let mut g = G { size, n, me, turn: v["turn"].as_i64().unwrap_or(0), stars: v["stars"].as_i64().unwrap_or(0), tech: [false; 128],
        ter: vec![-1; n], own: vec![0; n], st: vec![0; n], res: vec![0; n], road: vec![false; n], units: vec![], uat: vec![None; n], my_cities: vec![], clevel: HashMap::new() };
    if let Some(t) = v["technology"].as_array() { for x in t { if let Some(i) = x.as_u64() { if i < 128 { g.tech[i as usize] = true; } } } }
    if let Some(ts) = v["tiles"].as_array() { for t in ts {
        let i = t["tile"].as_u64().unwrap_or(0) as usize;
        if i >= n || !t["known"].as_bool().unwrap_or(false) { continue; }
        g.ter[i] = t["terrain"].as_i64().unwrap_or(3);
        g.own[i] = t["owner"].as_i64().unwrap_or(0);
        g.st[i] = t["structure"].as_i64().unwrap_or(0);
        g.res[i] = t["resource"].as_i64().unwrap_or(0);
        g.road[i] = t["road"].as_i64().unwrap_or(0) != 0;
    } }
    if let Some(us) = v["units"].as_array() { for u in us {
        let tile = u["tile"].as_u64().unwrap_or(0) as usize;
        if tile >= n { continue; }
        let uu = U { tile, owner: u["owner"].as_i64().unwrap_or(0), kind: u["kind"].as_i64().unwrap_or(2), hp: u["health"].as_f64().unwrap_or(10.0),
            moved: u["moved"].as_bool().unwrap_or(false), attacked: u["attacked"].as_bool().unwrap_or(false), vet: u["veteran"].as_bool().unwrap_or(false),
            passenger: u["passenger"].as_i64().unwrap_or(0), home: u["home"].as_i64().unwrap_or(0) };
        g.uat[tile] = Some(g.units.len());
        g.units.push(uu);
    } }
    if let Some(cs) = v["cities"].as_array() { for c in cs {
        let tile = c["tile"].as_u64().unwrap_or(0) as usize;
        if tile < n && c["owner"].as_i64().unwrap_or(0) == me { g.my_cities.push(tile); g.clevel.insert(tile, c["level"].as_i64().unwrap_or(1)); }
    } }
    let mut legal = vec![];
    if let Some(ls) = obs["legal_actions"].as_array() { for a in ls {
        let c = &a["command"];
        let id = match &a["action_id"] { Value::String(s) => s.clone(), o => o.to_string() };
        legal.push(A { id, k: c["kind"].as_i64().unwrap_or(0), s: c["source"].as_i64().unwrap_or(-1), t: c["target"].as_i64().unwrap_or(-1), p: c["parameter"].as_i64().unwrap_or(0) });
    } }
    (g, legal)
}

fn combat(aatk: f64, ahp: f64, amax: f64, ddef: f64, dhp: f64, dmax: f64, bonus: f64) -> (f64, f64) {
    let af = aatk * (ahp / amax.max(1.0));
    let df = ddef * (dhp / dmax.max(1.0)) * bonus;
    let tot = af + df;
    if tot <= 0.0 { return (0.0, 0.0); }
    let dmg = (af / tot * aatk * 4.5 + 0.5).floor();
    let ret = if dhp > dmg { (df / tot * ddef * 4.5 + 0.5).floor() } else { 0.0 };
    (dmg, ret)
}

fn choose(obs: &Value, mem: &mut Mem) -> String {
    let (g, legal) = parse(obs);
    if legal.is_empty() { return "0".to_string(); }
    let me = g.me;
    // rewards
    let rw: Vec<&A> = legal.iter().filter(|a| a.k == 9).collect();
    if !rw.is_empty() {
        let rank = |p: i64| match p { 6 => 0, 3 => 1, 1 => 2, 8 => 3, 5 => 4, 2 => 5, 1 => 6, _ => 7 };
        let no_port = !(0..g.n).any(|i| g.st[i] == 8 && g.own[i] == me);
        let rank2 = |a: &A| -> i64 {
            if a.p == 5 && no_port && a.t >= 0 && (a.t as usize) < g.n {
                let c = a.t as usize;
                let near_water = (0..g.n).any(|i| g.ter[i] == 1 && g.d(i, c) == 2);
                let has_water = (0..g.n).any(|i| g.ter[i] == 1 && g.own[i] == me);
                if near_water && !has_water { return -1; }
            }
            rank(a.p)
        };
        return match rw.iter().min_by_key(|a| rank2(a)) { Some(a) => a.id.clone(), None => legal[0].id.clone() };
    }
    if g.turn != mem.turn { mem.turn = g.turn; mem.skip.clear(); mem.claims.clear(); mem.acts = 0; }
    mem.acts += 1;
    let end_id = legal.iter().find(|a| a.k == 10).map(|a| a.id.clone()).unwrap_or_else(|| legal[0].id.clone());
    if mem.acts > 500 { return end_id; }
    // captures / promote
    if let Some(a) = legal.iter().find(|a| a.k == 8) { return a.id.clone(); }
    if let Some(a) = legal.iter().find(|a| a.k == 3 && a.p == 16) { return a.id.clone(); }
    let best_unit = |city: usize, emergency: bool| -> Option<&A> {
        let opts: Vec<&A> = legal.iter().filter(|a| a.k == 4 && a.s == city as i64).collect();
        let pref: &[i64] = if emergency { &[5, 11, 2, 3] } else { &[11, 2, 3] };
        for &p in pref {
            if !emergency && p == 2 && g.tech[24] && g.turn > 12 && g.stars >= 3 && !opts.iter().any(|a| a.p == 11) && !g.enemy_near(city, 4) { continue; }
            if let Some(a) = opts.iter().find(|a| a.p == p) { return Some(*a); }
        }
        None
    };
    // emergency garrison
    for &c in &g.my_cities {
        if g.uat[c].is_none() && g.enemy_near(c, 3) {
            let def_adj = g.units.iter().any(|x| x.owner == me && x.kind == 5 && g.d(x.tile, c) <= 1);
            if !g.enemy_near(c, 2) || def_adj {
                if let Some(a) = best_unit(c, false) { return a.id.clone(); }
            }
            if let Some(a) = best_unit(c, true) { return a.id.clone(); }
        }
    }
    // attacks
    let mut best: Option<(f64, &A)> = None;
    for a in legal.iter().filter(|a| a.k == 2) {
        let (s, t) = (a.s as usize, a.t as usize);
        if s >= g.n || t >= g.n { continue; }
        let (au, du) = match (g.uat[s], g.uat[t]) { (Some(x), Some(y)) => (&g.units[x], &g.units[y]), _ => continue };
        let (aatk, _, _, _) = stats(au.kind);
        let (_, ddef, _, drange) = stats(du.kind);
        let on_city = g.st[t] == 1 && g.own[t] == du.owner;
        let bonus = if on_city && matches!(du.kind, 2 | 3 | 4 | 5 | 9) { 1.5 } else { 1.0 };
        let (dmg, ret) = combat(aatk, au.hp, maxhp(au), ddef, du.hp, maxhp(du), bonus);
        let kill = du.hp - dmg <= 0.0;
        let r = if !kill && g.d(s, t) <= drange && !stiff(du.kind) { ret } else { 0.0 };
        let mut sc = dmg - r * 0.8 + if kill { 8.0 } else { 0.0 } + if on_city { 3.0 } else { 0.0 };
        if g.own[t] == me { sc += 2.0; }
        if r >= au.hp && !kill { sc -= 10.0; }
        if dmg <= 0.0 { continue; }
        if kill && g.my_cities.contains(&s) && stats(au.kind).3 < 2
            && g.units.iter().any(|x| x.owner != me && x.tile != t && g.d(x.tile, s) <= 3) {
            let cnt = g.units.iter().filter(|x| x.owner == me && x.home == s as i64 + 1).count() as i64;
            if !(g.stars >= 3 && cnt <= *g.clevel.get(&s).unwrap_or(&1)) { continue; }
        }
        if sc > 0.0 && best.as_ref().map_or(true, |b| sc > b.0) { best = Some((sc, a)); }
    }
    if let Some((_, a)) = best { return a.id.clone(); }
    // recover
    for a in legal.iter().filter(|a| a.k == 3 && a.p == 7) {
        let s = a.s as usize;
        if s >= g.n { continue; }
        if let Some(ui) = g.uat[s] {
            let u = &g.units[ui];
            if u.hp < 0.5 * maxhp(u) && !u.moved && !u.attacked && !g.enemy_near(s, 1) && !mem.skip.contains(&s) { return a.id.clone(); }
        }
    }
    // objectives
    let enemy_cities: Vec<usize> = (0..g.n).filter(|&i| g.st[i] == 1 && g.own[i] != 0 && g.own[i] != me).collect();
    let mut objs: Vec<(usize, i32, i32)> = vec![];
    for i in 0..g.n {
        if g.st[i] == 1 && g.own[i] == 0 { objs.push((i, -14, 1)); }
        else if g.st[i] == 1 && g.own[i] != me { objs.push((i, if g.uat[i].map_or(true, |u| g.units[u].owner == me) { -6 } else { 2 }, 99)); }
        else if g.st[i] == 2 { objs.push((i, 0, 1)); }
        else if g.ter[i] < 0 && g.nb(i).iter().any(|&j| g.ter[j] >= 0) { objs.push((i, 14, 3)); }
        else if g.own[i] != 0 && g.own[i] != me && enemy_cities.is_empty() { objs.push((i, 6, 6)); }
    }
    for u in g.units.iter().filter(|u| u.owner != me) {
        let near = g.my_cities.iter().any(|&c| g.d(c, u.tile) <= 3);
        if g.st[u.tile] == 1 { continue; }
        objs.push((u.tile, if near { -4 } else { 8 }, 2));
    }
    // moves
    let mut movers: Vec<usize> = legal.iter().filter(|a| a.k == 1).map(|a| a.s as usize).collect();
    movers.sort(); movers.dedup();
    let empty_thr = g.my_cities.iter().filter(|&&c| g.uat[c].is_none() && g.enemy_near(c, 4)).count() as i64;
    for &c in &g.my_cities {
        if g.uat[c].is_none() { objs.push((c, if g.enemy_near(c, 4) { -12 } else { -20 }, 1)); }
    }
    let mut cand: Vec<(i32, usize, Vec<i32>)> = vec![];
    for &s in &movers {
        if s >= g.n || mem.skip.contains(&s) { continue; }
        let u = match g.uat[s] { Some(x) => &g.units[x], None => continue };
        if g.my_cities.contains(&s) {
            let cnt = g.units.iter().filter(|x| x.owner == me && x.home == s as i64 + 1).count() as i64;
            let can_train = cnt <= *g.clevel.get(&s).unwrap_or(&1);
            if u.kind == 5 {
                if g.enemy_near(s, 2) || !can_train || g.stars < 5 + 3 * empty_thr { mem.skip.insert(s); continue; }
                if let Some(a) = legal.iter().find(|a| a.k == 1 && a.s == s as i64 && (a.t as usize) < g.n && g.own[a.t as usize] == me && g.st[a.t as usize] != 1 && !g.water(a.t as usize)) { return a.id.clone(); }
                mem.skip.insert(s); continue;
            }
            if g.enemy_near(s, 3) && !(can_train && g.stars >= 3 + 3 * empty_thr) { mem.skip.insert(s); continue; }
        }
        let dist = g.dijkstra(&[s], false, true);
        let mut bc = INF;
        for &(t, off, cap) in &objs {
            if dist[t] >= INF || t == s { continue; }
            if *mem.claims.get(&t).unwrap_or(&0) >= cap { continue; }
            if (u.kind == 5) != (off == -20 || off == -12) && off != -12 { continue; }
            bc = bc.min(dist[t] + off);
        }
        cand.push((bc, s, dist));
    }
    cand.sort_by_key(|c| (c.0, c.1));
    for (_, s, dist) in cand.iter() {
        let s = *s;
        let u = match g.uat[s] { Some(x) => &g.units[x], None => continue };
        let mut bo: Option<(i32, usize)> = None;
        for &(t, off, cap) in &objs {
            if dist[t] >= INF || t == s { continue; }
            if *mem.claims.get(&t).unwrap_or(&0) >= cap { continue; }
            if (u.kind == 5) != (off == -20 || off == -12) && off != -12 { continue; }
            let mut c = dist[t] + off;
            if u.kind == 12 && off >= 8 { c += 20; }
            if bo.map_or(true, |b| c < b.0) { bo = Some((c, t)); }
        }
        let (_, obj) = match bo { Some(b) => b, None => { mem.skip.insert(s); continue; } };
        let rd = g.dijkstra(&[obj], true, true);
        let mut bm: Option<(i32, &A)> = None;
        for a in legal.iter().filter(|a| a.k == 1 && a.s == s as i64) {
            let t = a.t as usize;
            if t >= g.n { continue; }
            let c = rd[t];
            if bm.as_ref().map_or(true, |b| c < b.0) { bm = Some((c, a)); }
        }
        *mem.claims.entry(obj).or_insert(0) += 1;
        if let Some((c, a)) = bm {
            if c < rd[s] { return a.id.clone(); }
        }
        mem.skip.insert(s);
    }
    // economy
    let army = g.units.iter().filter(|u| u.owner == me).count() as i64;
    let ncity = g.my_cities.len() as i64;
    let min_army = 1 + ncity + g.turn / 6;
    let army_target = (3 + ncity * 3 + g.turn / 4).min(40);
    let threatened = g.my_cities.iter().any(|&c| g.enemy_near(c, 4));
    let train_city = |emergency: bool| -> Option<&A> {
        let mut cs = g.my_cities.clone();
        let key = |c: usize| -> i64 {
            if enemy_cities.is_empty() { let m = g.n / 2 + g.size / 2; g.d(c, m) } else { enemy_cities.iter().map(|&e| g.d(c, e)).min().unwrap() }
        };
        cs.sort_by_key(|&c| key(c));
        for c in cs { if let Some(a) = best_unit(c, emergency) { return Some(a); } }
        None
    };
    if army < min_army || threatened {
        if let Some(a) = train_city(false) { return a.id.clone(); }
    }
    if let Some(a) = legal.iter().find(|a| a.k == 5) { return a.id.clone(); }
    // water need
    let mut water_needed = false;
    if !g.my_cities.is_empty() {
        let ld = g.dijkstra(&g.my_cities, false, false);
        water_needed = !objs.iter().any(|&(t, off, _)| off != 8 && off != -4 && off != -12 && off != -20 && ld[t] < INF);
    }
    let research = |p: i64| legal.iter().find(|a| a.k == 7 && a.p == p);
    let has_port = (0..g.n).any(|i| g.st[i] == 8 && g.own[i] == me);
    if water_needed {
        if !g.tech[10] { if let Some(a) = research(10) { return a.id.clone(); } }
        else if !has_port {
            let ports: Vec<&A> = legal.iter().filter(|a| a.k == 6 && a.p == 8).collect();
            if let Some(a) = ports.iter().filter(|a| a.t >= 0 && (a.t as usize) < g.n).max_by_key(|a| g.nb(a.t as usize).iter().filter(|&&j| g.water(j) || g.ter[j] < 0).count()) { return a.id.clone(); }
        } else if !g.tech[13] { if let Some(a) = research(13) { return a.id.clone(); } }
        if g.tech[10] && has_port && !g.tech[13] { if let Some(a) = research(13) { return a.id.clone(); } }
        if (!g.tech[10] || !has_port || !g.tech[13]) && g.stars < 10 && army >= 2 && !threatened {
            mem.skip.clear(); mem.claims.clear(); return end_id;
        }
    }
    // free monuments and pop buildings
    if let Some(a) = legal.iter().find(|a| a.k == 6 && (23..=29).contains(&a.p)) { return a.id.clone(); }
    let reserve = if threatened { 4 } else { 0 };
    if (army >= min_army && g.stars >= 5 + reserve) || g.stars >= 12 {
        for p in [21i64, 5, 12] {
            if let Some(a) = legal.iter().find(|a| a.k == 6 && a.p == p && (p != 12 || (a.t >= 0 && (a.t as usize) < g.n && g.res[a.t as usize] == 0))) { return a.id.clone(); }
        }
    }
    if g.stars >= 14 {
        if let Some(a) = legal.iter().find(|a| a.k == 4 && a.p == 44) { return a.id.clone(); }
    }
    if g.stars >= 16 {
        let nports = (0..g.n).filter(|&i| g.st[i] == 8 && g.own[i] == me).count();
        for p in [22i64, 13, 6, 50, 8] {
            if p == 8 && nports >= 2 { continue; }
            if let Some(a) = legal.iter().find(|a| a.k == 6 && a.p == p) { return a.id.clone(); }
        }
    }
    if g.stars >= 60 {
        if let Some(a) = legal.iter().find(|a| a.k == 6 && matches!(a.p, 17 | 18 | 19 | 20)) { return a.id.clone(); }
    }
    // research
    let ntech = g.tech.iter().filter(|&&t| t).count() as i64 - 2;
    let fruit = (0..g.n).any(|i| g.res[i] == 6 && g.own[i] == me);
    let _ = fruit;
    let order: Vec<i64> = vec![6, 7, 20, 16, 23, 24, 8, 1, 18, 17, 4, 9, 10, 2, 3];
    let next = order.iter().copied().find(|&t| !g.tech[t as usize]);
    let saving = next.is_some() && ntech < 1 + g.turn / 6 && g.turn >= 3;
    if next.is_some() && (saving || g.stars >= 20) && !(threatened && g.stars < 12) {
        if let Some(a) = research(next.unwrap()) { return a.id.clone(); }
    }
    if g.stars >= 45 { if let Some(a) = legal.iter().find(|a| a.k == 7) { return a.id.clone(); } }
    // more units
    if army < army_target && (!saving || army < min_army || g.stars >= 10) {
        if let Some(a) = train_city(false) { return a.id.clone(); }
    }
    let _ = reserve;
    mem.skip.clear(); mem.claims.clear();
    end_id
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Mem { skip: HashSet::new(), claims: HashMap::new(), turn: -1, acts: 0 };
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let id = choose(&obs, &mut mem);
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
