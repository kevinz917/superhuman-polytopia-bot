use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

const INF: i32 = 9999;

#[derive(Clone)]
struct Act { id: String, kind: i32, src: i32, dst: i32, par: i32 }

struct G {
    size: usize, n: usize, me: i32, turn: i32, stars: i32,
    terrain: Vec<i32>, owner: Vec<i32>, structure: Vec<i32>, resource: Vec<i32>,
    uo: Vec<i32>, uk: Vec<i32>, uhp: Vec<f32>, uvet: Vec<bool>, upass: Vec<i32>,
    tech: [bool; 128],
    my_cities: Vec<usize>, cap_total: i32, capital: Vec<i32>, level: Vec<i32>, homecnt: Vec<i32>,
}

// (maxhp, atk, def, mov, range, cost)
fn stats(k: i32) -> (f32, f32, f32, i32, i32, i32) {
    match k {
        2 => (10., 2., 2., 1, 1, 2), 3 => (10., 2., 1., 2, 1, 3), 4 => (10., 3.5, 1., 3, 1, 8),
        5 => (15., 1., 3., 1, 1, 3), 8 => (10., 4., 0., 1, 3, 8), 9 => (10., 2., 1., 1, 2, 3),
        10 => (10., 0., 1., 1, 1, 5), 11 => (15., 3., 3., 1, 1, 5), 12 => (40., 5., 4., 1, 1, 10),
        37 => (10., 2., 2., 1, 1, 2), 38 => (5., 0., 0.5, 2, 1, 8), 39 => (5., 0., 0.5, 2, 1, 2),
        40 => (10., 2., 2., 2, 1, 2), 41 => (10., 3., 2., 2, 3, 15), 42 => (10., 2., 1., 3, 2, 5),
        43 => (10., 0., 1., 2, 2, 2), 44 => (10., 3., 3., 3, 1, 5), 45 => (40., 4., 4., 2, 1, 10),
        _ => (10., 2., 2., 1, 1, 2),
    }
}
fn stiff(k: i32) -> bool { matches!(k, 8 | 10 | 38 | 39 | 41 | 43 | 45) }
fn tier(t: i32) -> i32 {
    match t { 1 | 6 | 10 | 15 | 20 => 1, 2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2, _ => 3 }
}

impl G {
    fn xy(&self, t: usize) -> (i32, i32) { ((t % self.size) as i32, (t / self.size) as i32) }
    fn d(&self, a: usize, b: usize) -> i32 {
        let (ax, ay) = self.xy(a); let (bx, by) = self.xy(b);
        (ax - bx).abs().max((ay - by).abs())
    }
    fn nb(&self, t: usize) -> Vec<usize> {
        let (x, y) = self.xy(t); let s = self.size as i32; let mut v = Vec::with_capacity(8);
        for dy in -1..=1 { for dx in -1..=1 {
            if dx == 0 && dy == 0 { continue; }
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && nx < s && ny < s { v.push((ny * s + nx) as usize); }
        } }
        v
    }
    fn water(&self, t: usize) -> bool { self.terrain[t] == 1 || self.terrain[t] == 2 }
    fn pass(&self, t: usize, land_only: bool) -> bool {
        match self.terrain[t] {
            -1 => true, 3 | 5 => true,
            4 => land_only || self.tech[20],
            1 => !land_only && self.tech[10],
            2 => !land_only && self.tech[13],
            _ => true,
        }
    }
    fn allowed(&self, a: usize, b: usize) -> bool {
        if self.water(b) && !self.water(a) { self.structure[b] == 8 && self.owner[b] == self.me } else { true }
    }
    fn rev_bfs(&self, t: usize) -> Vec<i32> {
        let mut dist = vec![INF; self.n];
        let mut q = std::collections::VecDeque::new();
        dist[t] = 0; q.push_back(t);
        while let Some(b) = q.pop_front() {
            for a in self.nb(b) {
                if dist[a] == INF && self.pass(a, false) && self.allowed(a, b) {
                    dist[a] = dist[b] + 1; q.push_back(a);
                }
            }
        }
        dist
    }
    fn maxhp(&self, t: usize) -> f32 {
        let k = if self.upass[t] > 0 { self.upass[t] } else { self.uk[t] };
        stats(k).0 + if self.uvet[t] { 5. } else { 0. }
    }
    fn def_bonus(&self, t: usize) -> f32 {
        let mine = self.uo[t] == self.me;
        match self.terrain[t] {
            5 => if mine { if self.tech[18] { 1.5 } else { 1.0 } } else { 1.2 },
            4 => if mine { if self.tech[20] { 1.5 } else { 1.0 } } else { 1.5 },
            1 | 2 => 1.0,
            _ => if self.structure[t] == 1 && self.owner[t] == self.uo[t] { 1.5 } else { 1.0 },
        }
    }
    // (damage, retaliation)
    fn combat(&self, s: usize, t: usize) -> (f32, f32) {
        let (_, aa, _, _, _, _) = stats(self.uk[s]);
        let (_, _, dd, _, dr, _) = stats(self.uk[t]);
        let af = aa * (self.uhp[s] / self.maxhp(s));
        let df = dd * (self.uhp[t] / self.maxhp(t)) * self.def_bonus(t);
        let tot = af + df;
        if tot <= 0.0 { return (0., 0.); }
        let dmg = (af / tot * aa * 4.5 + 0.5).floor();
        let mut ret = if self.uhp[t] > dmg { (df / tot * dd * 4.5 + 0.5).floor() } else { 0. };
        if stiff(self.uk[t]) || self.d(s, t) > dr { ret = 0.; }
        (dmg, ret)
    }
    fn enemies_near(&self, t: usize, r: i32) -> i32 {
        let mut c = 0;
        for i in 0..self.n { if self.uo[i] > 0 && self.uo[i] != self.me && self.d(i, t) <= r { c += 1; } }
        c
    }
}

fn parse(obs: &Value) -> Option<(G, Vec<Act>)> {
    let v = &obs["visible_state"];
    let size = v["board_size"].as_u64()? as usize;
    let n = size * size;
    let me = obs["player"].as_i64()? as i32;
    let mut g = G { size, n, me, turn: v["turn"].as_i64().unwrap_or(0) as i32, stars: v["stars"].as_i64().unwrap_or(0) as i32,
        terrain: vec![-1; n], owner: vec![0; n], structure: vec![0; n], resource: vec![0; n],
        uo: vec![0; n], uk: vec![0; n], uhp: vec![0.; n], uvet: vec![false; n], upass: vec![0; n],
        tech: [false; 128], my_cities: vec![], cap_total: 0, capital: vec![0; n], level: vec![0; n], homecnt: vec![0; n] };
    for t in v["technology"].as_array()? { let k = t.as_i64().unwrap_or(0) as usize; if k < 128 { g.tech[k] = true; } }
    for t in v["tiles"].as_array()? {
        let i = t["tile"].as_u64().unwrap_or(0) as usize;
        if i >= n || !t["known"].as_bool().unwrap_or(false) { continue; }
        g.terrain[i] = t["terrain"].as_i64().unwrap_or(3) as i32;
        g.owner[i] = t["owner"].as_i64().unwrap_or(0) as i32;
        g.structure[i] = t["structure"].as_i64().unwrap_or(0) as i32;
        g.resource[i] = t["resource"].as_i64().unwrap_or(0) as i32;
        g.capital[i] = t["capital"].as_i64().unwrap_or(0) as i32;
    }
    for u in v["units"].as_array()? {
        let i = u["tile"].as_u64().unwrap_or(0) as usize;
        if i >= n { continue; }
        g.uo[i] = u["owner"].as_i64().unwrap_or(0) as i32;
        g.uk[i] = u["kind"].as_i64().unwrap_or(2) as i32;
        g.uhp[i] = u["health"].as_f64().unwrap_or(10.) as f32;
        g.uvet[i] = u["veteran"].as_bool().unwrap_or(false);
        g.upass[i] = u["passenger"].as_i64().unwrap_or(0) as i32;
        let h = u["home"].as_i64().unwrap_or(0);
        if g.uo[i] == me && h >= 1 && (h as usize) <= n { g.homecnt[(h - 1) as usize] += 1; }
    }
    for c in v["cities"].as_array()? {
        let i = c["tile"].as_u64().unwrap_or(0) as usize;
        if i < n && c["owner"].as_i64().unwrap_or(0) as i32 == me {
            g.my_cities.push(i);
            g.level[i] = c["level"].as_i64().unwrap_or(1) as i32;
            g.cap_total += c["level"].as_i64().unwrap_or(1) as i32 + 1;
        }
    }
    let mut acts = vec![];
    for a in obs["legal_actions"].as_array()? {
        let c = &a["command"];
        acts.push(Act { id: a["action_id"].as_str()?.to_owned(), kind: c["kind"].as_i64().unwrap_or(0) as i32,
            src: c["source"].as_i64().unwrap_or(-1) as i32, dst: c["target"].as_i64().unwrap_or(-1) as i32,
            par: c["parameter"].as_i64().unwrap_or(0) as i32 });
    }
    Some((g, acts))
}

struct Mem { turn: i32, count: i32 }

fn choose(obs: &Value, mem: &mut Mem) -> Option<String> {
    let (g, acts) = parse(obs)?;
    if acts.is_empty() { return None; }
    let end = acts.iter().find(|a| a.kind == 10).map(|a| a.id.clone());
    if mem.turn != g.turn { mem.turn = g.turn; mem.count = 0; }
    mem.count += 1;
    if mem.count > 400 { if let Some(e) = &end { return Some(e.clone()); } }
    let me = g.me;
    // 1 rewards
    let mut best: Option<(f32, &Act)> = None;
    for a in acts.iter().filter(|a| a.kind == 9) {
        let c = a.dst.max(0) as usize;
        let wallw = if c < g.n && (g.capital[c] == me || g.enemies_near(c, 6) > 0) { 12. } else { 6. };
        let mut w1 = false; let mut w2 = false;
        if c < g.n { for i in 0..g.n { if g.terrain[i] == 1 { let dd = g.d(c, i); if dd <= 1 && g.owner[i] == me { w1 = true; } if dd == 2 && g.owner[i] == 0 { w2 = true; } } } }
        let bw = if w2 && !w1 { 12. } else { 8. };
        let s = match a.par { 3 => 10., 7 => 10., 5 => bw, 6 => 10., 8 => 10., 2 => 5., 1 => wallw, 4 => 4., _ => 1. };
        if best.map_or(true, |b| s > b.0) { best = Some((s, a)); }
    }
    if let Some(b) = best { return Some(b.1.id.clone()); }
    // 2 promote, 3 capture
    if let Some(a) = acts.iter().find(|a| a.kind == 3 && a.par == 16) { return Some(a.id.clone()); }
    if let Some(a) = acts.iter().find(|a| a.kind == 8) { return Some(a.id.clone()); }

    let my_units: Vec<usize> = (0..g.n).filter(|&i| g.uo[i] == me).collect();
    let enemy_units: Vec<usize> = (0..g.n).filter(|&i| g.uo[i] > 0 && g.uo[i] != me).collect();
    let capturable = |t: usize| (g.structure[t] == 1 && g.owner[t] != me) || g.structure[t] == 2;
    let city_threat = |c: usize, r: i32| g.enemies_near(c, r);

    let empty_thr = g.my_cities.iter().filter(|&&c| g.uo[c] == 0 && city_threat(c, 3) > 0).count() as i32;
    let unit_cost = if g.tech[7] { 3 } else { 2 };
    let can_rot = |s: usize| -> bool {
        g.structure[s] == 1 && g.owner[s] == me && city_threat(s, 3) > 0 && g.uk[s] != 12
            && g.homecnt[s] <= g.level[s] && g.stars >= unit_cost * (empty_thr + 1) + if g.uk[s] == 5 { 2 } else { 0 }
            && g.uhp[s] >= 0.6 * g.maxhp(s)
    };
    // 4 attacks
    let mut best: Option<(f32, &Act)> = None;
    for a in acts.iter().filter(|a| a.kind == 2) {
        let (s, t) = (a.src as usize, a.dst as usize);
        if s >= g.n || t >= g.n || g.uo[t] == 0 { continue; }
        if can_rot(s) && acts.iter().any(|b| b.kind == 1 && b.src == a.src) && stats(g.uk[s]).4 < 2 && g.uk[s] != 5 { continue; }
        let (dmg, ret) = g.combat(s, t);
        let kill = dmg >= g.uhp[t];
        let melee = stats(g.uk[s]).4 < 2;
        let cost_d = stats(g.uk[t]).5 as f32 + if g.upass[t] > 0 { 2. } else { 0. };
        let cost_a = stats(g.uk[s]).5 as f32;
        let mut sc = dmg.min(g.uhp[t]) * (0.6 + cost_d / 5.0) + if kill { 6. + cost_d } else { 0. } - ret * (0.5 + cost_a / 5.0);
        if ret >= g.uhp[s] { sc -= 30.; }
        if g.structure[t] == 1 { sc += 3.; }
        if g.owner[t] == me { sc += 3.; }
        if kill && melee {
            if capturable(s) { continue; }
            if g.structure[s] == 1 && g.owner[s] == me && city_threat(s, 2) >= 2 { continue; }
            if g.structure[t] == 1 && g.owner[t] != me { sc += 15.; }
        }
        if dmg <= 0. { continue; }
        if best.map_or(true, |b| sc > b.0) { best = Some((sc, a)); }
    }
    if let Some(b) = best { if b.0 > 0. { return Some(b.1.id.clone()); } }

    // targets
    // land reachability for navy need
    let mut land = vec![false; g.n];
    {
        let mut q: Vec<usize> = vec![];
        for &c in &g.my_cities { land[c] = true; q.push(c); }
        for &u in &my_units { if !g.water(u) && !land[u] { land[u] = true; q.push(u); } }
        while let Some(b) = q.pop() {
            if g.terrain[b] == -1 { continue; }
            for a in g.nb(b) { if !land[a] && g.pass(a, true) { land[a] = true; q.push(a); } }
        }
    }
    let land_goal = (0..g.n).any(|i| land[i] && (g.terrain[i] == -1 || (g.structure[i] == 1 && g.owner[i] != me)));
    let need_navy = !land_goal;
    let have_port = (0..g.n).any(|i| g.structure[i] == 8 && g.owner[i] == me);

    let strong = my_units.len() as i32 >= (2 * g.my_cities.len() as i32 + 3).min(g.cap_total - 1) || g.turn > 40;
    // (tile, value, exclusive)
    let mut targets: Vec<(usize, f32, bool)> = vec![];
    for i in 0..g.n {
        if g.terrain[i] == -1 { continue; }
        if g.structure[i] == 2 && !g.pass(i, false) { continue; }
        if g.structure[i] == 1 && g.owner[i] == 0 { targets.push((i, 70., true)); }
        else if g.structure[i] == 1 && g.owner[i] != me {
            targets.push((i, if g.uo[i] == 0 || g.uo[i] == me { 85. } else if strong { 62. } else { 30. }, false));
        } else if g.structure[i] == 2 { targets.push((i, 38., true)); }
        else if g.structure[i] == 1 && g.owner[i] == me && g.uo[i] != me && city_threat(i, 3) > 0 { targets.push((i, 90., true)); }
        if g.uo[i] > 0 && g.uo[i] != me && (g.owner[i] == me || g.my_cities.iter().any(|&c| g.d(c, i) <= 3)) { targets.push((i, 55., false)); }
    }
    let mut fr = 0;
    for i in 0..g.n {
        if g.terrain[i] == -1 || !g.pass(i, false) || g.structure[i] == 1 { continue; }
        if (i * 7 + g.turn as usize) % 2 == 0 && fr > 12 { continue; }
        if g.nb(i).iter().any(|&j| g.terrain[j] == -1) { let dc = g.my_cities.iter().map(|&c| g.d(c, i)).min().unwrap_or(0) as f32; targets.push((i, 26. - 1.5 * dc, true)); fr += 1; if fr > 40 { break; } }
    }
    let dists: Vec<Vec<i32>> = targets.iter().map(|t| g.rev_bfs(t.0)).collect();
    let mut pairs: Vec<(f32, usize, usize)> = vec![];
    for (ui, &u) in my_units.iter().enumerate() {
        for (ti, t) in targets.iter().enumerate() {
            let d = dists[ti][u];
            if d >= INF { continue; }
            let mut val = t.1 - 5.0 * d as f32;
            let k = g.uk[u];
            if (k == 12 || k == 11 || k == 8) && !t.2 && g.structure[t.0] == 1 { val += 15.; }
            if (k == 8 || k == 12) && t.2 && d > 1 { val -= 30.; }
            pairs.push((val, ui, ti));
        }
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut assign: Vec<Option<usize>> = vec![None; my_units.len()];
    let mut taken = vec![false; targets.len()];
    // pre-claim: units standing on exclusive targets
    for (ui, &u) in my_units.iter().enumerate() {
        for (ti, t) in targets.iter().enumerate() {
            if t.0 == u && t.2 && capturable(u) && !taken[ti] { taken[ti] = true; assign[ui] = Some(ti); }
        }
    }
    for &(_, ui, ti) in &pairs {
        if assign[ui].is_some() || (targets[ti].2 && taken[ti]) { continue; }
        assign[ui] = Some(ti);
        if targets[ti].2 { taken[ti] = true; }
    }

    // 5 moves
    let mut best: Option<(f32, &Act)> = None;
    for a in acts.iter().filter(|a| a.kind == 1) {
        let (s, t) = (a.src as usize, a.dst as usize);
        if s >= g.n || t >= g.n { continue; }
        let ui = match my_units.iter().position(|&u| u == s) { Some(x) => x, None => continue };
        if capturable(s) && g.structure[s] != 2 { continue; }
        let on_city = g.structure[s] == 1 && g.owner[s] == me;
        let rot = can_rot(s);
        if on_city && city_threat(s, 3) > 0 && !rot { continue; }
        let hp = g.uhp[s]; let mh = g.maxhp(s);
        let near_enemy = g.enemies_near(s, 2) > 0;
        if hp < 0.5 * mh && !near_enemy && g.owner[s] == me && !on_city { continue; }
        let mut sc = 0.0f32;
        let mut found = false;
        for &(_, pu, ti) in &pairs {
            if pu != ui { continue; }
            if targets[ti].2 && taken[ti] && assign[ui] != Some(ti) { continue; }
            let dt = &dists[ti];
            if dt[s] == 0 { found = true; sc = -50.; break; }
            let tt = targets[ti].0;
            if dt[s] == 1 && g.uo[tt] > 0 && g.uo[tt] != me { found = true; sc = -1.; break; }
            let ok = acts.iter().any(|b| b.kind == 1 && b.src == a.src && (b.dst as usize) < g.n && dt[b.dst as usize] < dt[s]);
            if !ok { continue; }
            sc = 10.0 * (dt[s] - dt[t].min(dt[s] + 2)) as f32;
            if t == tt { sc += 4.; }
            found = true;
            break;
        }
        if !found { if on_city && g.structure[t] != 1 { sc = 3.; } }
        if rot && sc < 2. && g.structure[t] != 1 && !g.water(t) { sc = 2.; }
        // terrain / safety tweaks
        if g.terrain[t] == 4 { sc += 0.5; }
        if g.terrain[t] == 5 && g.tech[18] { sc += 0.3; }
        let (_, aa, dd, _, rng, _) = stats(g.uk[s]);
        let mut danger = 0.0f32;
        for &e in &enemy_units {
            let (_, ea, _, em, er, _) = stats(g.uk[e]);
            if ea <= 0. { continue; }
            let de = g.d(e, t);
            if de <= em + er - if rng >= 2 && de > 1 { 0 } else { 0 } { danger += ea / (dd + 0.5); }
        }
        let weak = hp < 0.5 * mh || aa <= 0. || g.uk[s] == 8 || g.uk[s] == 43 || g.water(t);
        if weak { sc -= 3.0 * danger; } else if danger > 2.5 { sc -= 1.5 * (danger - 2.5); }
        if g.structure[t] == 8 && g.owner[t] == me && sc < 10. { sc -= 6.; }
        if best.map_or(true, |b| sc > b.0) { best = Some((sc, a)); }
    }
    if let Some(b) = best { if b.0 > 0.5 { return Some(b.1.id.clone()); } }

    // 6 economy
    let ncity = g.my_cities.len() as i32;
    let army = my_units.len() as i32;
    let war = g.my_cities.iter().any(|&c| city_threat(c, 4) > 0);
    let army_min = ncity * 2 + 1 + if war { 2 } else { 0 } + g.turn / 10;
    let army_target = ncity * 2 + 2 + g.turn / 5;
    let mut fruit = 0; let mut fish = 0;
    for i in 0..g.n { if g.owner[i] == me { if g.resource[i] == 6 { fruit += 1; } if g.resource[i] == 3 { fish += 1; } } }
    let can_summon = acts.iter().any(|a| a.kind == 4 && a.par != 44 && a.par != 42 && a.par != 41);
    let mut plan: Vec<i32> = vec![];
    if need_navy { plan.push(10); plan.push(13); }
    let _ = fruit;
    plan.extend_from_slice(&[1, 16, 6, 7, 20, 23, 24]);
    if need_navy { plan.push(44); }
    if fish >= 2 { plan.push(10); }
    plan.extend_from_slice(&[18, 17, 8, 10, 13, 9, 4, 2, 3, 21, 5]);
    let next_tech = plan.iter().copied().find(|&t| !g.tech[t as usize]);
    let next_cost = next_tech.map(|t| 4 + ncity * tier(t)).unwrap_or(0);
    let cnt = |k: i32| my_units.iter().filter(|&&u| g.uk[u] == k).count() as i32;
    let mut best: Option<(f32, &Act)> = None;
    for a in acts.iter() {
        let sc: f32 = match a.kind {
            5 => 500.,
            6 => match a.par {
                23..=29 => 800.,
                5 | 21 => 400.,
                12 => 380.,
                8 => if need_navy && !have_port { 750. } else if g.stars >= 12 && (need_navy || fish > 0) { 200. } else { -1. },
                6 | 13 | 22 => {
                    let t = a.dst as usize;
                    let want = match a.par { 6 => 5, 13 => 12, _ => 21 };
                    let c = g.nb(t).iter().filter(|&&j| g.structure[j] == want && g.owner[j] == me).count();
                    if c >= 2 || (a.par == 22 && c >= 1) { 390. } else { -1. }
                }
                17 | 18 | 19 | 20 => if g.stars >= 45 && !war { 60. } else { -1. },
                _ => -1.,
            },
            7 => {
                if Some(a.par) == next_tech {
                    if need_navy && (a.par == 10 || a.par == 13) { 700. }
                    else if war && (a.par == 6 || a.par == 7) { 640. }
                    else if war && g.stars < next_cost + 4 { if can_summon { -1. } else { 300. } }
                    else if army >= army_min || g.stars >= next_cost + 6 { 450. } else { 100. }
                } else if next_tech.is_none() && g.stars >= 40 { 40. } else { -1. }
            }
            4 => {
                let c = a.src as usize;
                let k = a.par;
                let cost = stats(k).5;
                let threat = c < g.n && city_threat(c, 3) > 0;
                if k == 44 || k == 42 || k == 41 {
                    if k == 44 && g.stars >= 8 { 360. } else { -1. }
                } else {
                let pref: f32 = if threat {
                    match k { 5 => if cnt(5) < 2 * ncity { 9. } else { 5. }, 11 => 9.5, 2 => 6., 9 => 5., 3 => 4., _ => -100. }
                } else {
                    match k {
                        11 => 9.,
                        8 => if cnt(8) * 4 < army && army >= 4 { 9.5 } else { -100. },
                        9 => if cnt(9) * 3 < army && army >= 4 { 7. } else { 4. },
                        4 => if cnt(4) * 4 < army { 7.5 } else { -100. },
                        2 => if g.tech[24] { 3. } else { 6. },
                        3 => if cnt(3) < 3 + ncity && !g.tech[24] { 6.5 } else { 3.5 },
                        5 => 2.,
                        _ => -100.,
                    }
                };
                if pref < 0. { -1. }
                else if threat { 900. + pref }
                else if army < army_min { 600. + pref }
                else if next_tech.is_some() && g.stars - cost < next_cost && army >= army_target && !war { -1. }
                else if g.tech[24] && k != 11 && k != 8 && g.stars < 5 && army >= army_min { -1. }
                else if war { 650. + pref }
                else { 350. + pref }
                }
            }
            _ => -1.,
        };
        if sc > 0. && best.map_or(true, |b| sc > b.0) { best = Some((sc, a)); }
    }
    if let Some(b) = best { return Some(b.1.id.clone()); }
    end.or_else(|| Some(acts[0].id.clone()))
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Mem { turn: -1, count: 0 };
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => Value::Null };
        let id = choose(&obs, &mut mem).unwrap_or_else(|| {
            obs["legal_actions"].as_array().and_then(|l| {
                l.iter().find(|a| a["command"]["kind"] == 10).or(l.first()).and_then(|a| a["action_id"].as_str().map(|s| s.to_owned()))
            }).unwrap_or_default()
        });
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
