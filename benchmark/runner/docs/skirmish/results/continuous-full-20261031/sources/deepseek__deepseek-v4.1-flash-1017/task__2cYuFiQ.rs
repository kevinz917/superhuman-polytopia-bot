
// Bardur heuristic bot: economy + expansion first, opportunistic combat.
use serde_json::Value;
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};

#[derive(Clone)]
struct Act {
    id: String,
    kind: i32,
    source: i32,
    target: i32,
    param: i32,
}

fn fi(v: &Value, k: &str) -> i32 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(0) as i32
}
fn fb(v: &Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}
fn ff(v: &Value, k: &str) -> f32 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32
}

struct St {
    n: usize,
    size: i32,
    turn: i32,
    stars: i32,
    me: i32,
    tech: Vec<bool>,
    terrain: Vec<i32>,
    owner: Vec<i32>,
    strc: Vec<i32>,
    known: Vec<bool>,
    uowner: Vec<i32>,
    ukind: Vec<i32>,
    uhp: Vec<f32>,
    umoved: Vec<bool>,
    uhome: Vec<i32>,
    acts: Vec<Act>,
}

impl St {
    fn parse(v: &Value) -> Option<St> {
        let vis = v.get("visible_state")?;
        let size = fi(vis, "board_size");
        if size <= 1 || size > 40 {
            return None;
        }
        let n = (size * size) as usize;
        let mut st = St {
            n,
            size,
            turn: fi(vis, "turn"),
            stars: fi(vis, "stars"),
            me: fi(v, "player"),
            tech: vec![false; 128],
            terrain: vec![0; n],
            owner: vec![0; n],
            strc: vec![0; n],
            known: vec![false; n],
            uowner: vec![0; n],
            ukind: vec![0; n],
            uhp: vec![0.0; n],
            umoved: vec![false; n],
            uhome: vec![0; n],
            acts: Vec::new(),
        };
        if let Some(ts) = vis.get("technology").and_then(|x| x.as_array()) {
            for t in ts {
                if let Some(k) = t.as_i64() {
                    if (0..128).contains(&k) {
                        st.tech[k as usize] = true;
                    }
                }
            }
        }
        if let Some(tiles) = vis.get("tiles").and_then(|x| x.as_array()) {
            for t in tiles {
                let i = fi(t, "tile");
                if i < 0 || i as usize >= n {
                    continue;
                }
                let i = i as usize;
                if fb(t, "known") {
                    st.known[i] = true;
                    st.terrain[i] = fi(t, "terrain");
                    st.owner[i] = fi(t, "owner");
                    st.strc[i] = fi(t, "structure");
                }
            }
        }
        if let Some(us) = vis.get("units").and_then(|x| x.as_array()) {
            for u in us {
                let i = fi(u, "tile");
                if i < 0 || i as usize >= n {
                    continue;
                }
                let i = i as usize;
                st.uowner[i] = fi(u, "owner");
                st.ukind[i] = fi(u, "kind");
                st.uhp[i] = ff(u, "health").max(1.0);
                st.umoved[i] = fb(u, "moved");
                st.uhome[i] = fi(u, "home");
            }
        }
        if let Some(ls) = v.get("legal_actions").and_then(|x| x.as_array()) {
            for a in ls {
                let cmd = match a.get("command") {
                    Some(c) => c,
                    None => continue,
                };
                let id = match a.get("action_id").and_then(|x| x.as_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                st.acts.push(Act {
                    id,
                    kind: fi(cmd, "kind"),
                    source: fi(cmd, "source"),
                    target: fi(cmd, "target"),
                    param: fi(cmd, "parameter"),
                });
            }
        }
        Some(st)
    }
}

fn stats(kind: i32) -> (f32, f32, f32, i32) {
    match kind {
        2 => (10.0, 2.0, 2.0, 1),
        3 => (10.0, 2.0, 1.0, 1),
        4 => (10.0, 3.5, 1.0, 1),
        5 => (15.0, 1.0, 3.0, 1),
        8 => (10.0, 4.0, 0.0, 3),
        9 => (10.0, 2.0, 1.0, 2),
        10 => (10.0, 0.0, 1.0, 1),
        11 => (15.0, 3.0, 3.0, 1),
        12 => (40.0, 5.0, 4.0, 1),
        37 => (10.0, 2.0, 2.0, 1),
        38 => (5.0, 0.0, 0.5, 1),
        39 => (5.0, 0.0, 0.5, 1),
        40 => (10.0, 2.0, 2.0, 1),
        41 => (10.0, 3.0, 2.0, 3),
        42 => (10.0, 2.0, 1.0, 2),
        43 => (10.0, 0.0, 1.0, 2),
        44 => (10.0, 3.0, 3.0, 1),
        45 => (40.0, 4.0, 4.0, 1),
        _ => (10.0, 2.0, 2.0, 1),
    }
}

fn unit_value(kind: i32) -> i32 {
    match kind {
        2 => 10,
        3 => 12,
        4 => 24,
        5 => 12,
        8 => 22,
        9 => 12,
        11 => 22,
        12 => 45,
        43 => 4,
        44 => 14,
        42 => 8,
        41 => 22,
        _ => 10,
    }
}

fn fortify(kind: i32) -> bool {
    matches!(kind, 2 | 3 | 4 | 5 | 9 | 15)
}

fn combat(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    let div = if total > 0.0 { total } else { 1.0 };
    let atk = ((if total > 0.0 { (af / div) * aa * 4.5 } else { 0.0 }) + 0.000001 + 0.5).floor();
    let def = ((if dh > atk && total > 0.0 { (df / div) * dd * 4.5 } else { 0.0 }) + 0.000001 + 0.5)
        .floor();
    (atk, def)
}

fn cheb(size: i32, a: usize, b: usize) -> i32 {
    let ax = (a as i32) % size;
    let ay = (a as i32) / size;
    let bx = (b as i32) % size;
    let by = (b as i32) / size;
    (ax - bx).abs().max((ay - by).abs())
}

fn def_bonus(st: &St, tile: usize, kind: i32) -> f32 {
    let terr = st.terrain[tile];
    if st.strc[tile] == 1 && st.owner[tile] == st.uowner[tile] && fortify(kind) {
        return 1.5;
    }
    match terr {
        1 | 2 => {
            if st.tech[12] {
                1.5
            } else {
                1.0
            }
        }
        5 => {
            if st.tech[18] {
                1.5
            } else {
                1.0
            }
        }
        4 => {
            if st.tech[20] {
                1.5
            } else {
                1.0
            }
        }
        _ => 1.0,
    }
}

fn unk_adj(st: &St, t: usize) -> i32 {
    let size = st.size;
    let x = (t as i32) % size;
    let y = (t as i32) / size;
    let mut c = 0;
    for dy in -1..=1i32 {
        for dx in -1..=1i32 {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= size || ny >= size {
                continue;
            }
            if !st.known[(ny * size + nx) as usize] {
                c += 1;
            }
        }
    }
    c
}

fn adj_enemies(st: &St, t: usize) -> i32 {
    let size = st.size;
    let x = (t as i32) % size;
    let y = (t as i32) / size;
    let mut c = 0;
    for dy in -1..=1i32 {
        for dx in -1..=1i32 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= size || ny >= size {
                continue;
            }
            let u = (ny * size + nx) as usize;
            if st.uowner[u] != 0 && st.uowner[u] != st.me {
                c += 1;
            }
        }
    }
    c
}

fn passable(st: &St, t: usize) -> bool {
    if !st.known[t] {
        return true;
    }
    let terr = st.terrain[t];
    if terr == 1 || terr == 2 {
        return false;
    }
    if terr == 4 {
        return st.tech[20];
    }
    true
}

fn field(st: &St, sources: &[usize]) -> Vec<i32> {
    let mut d = vec![-1i32; st.n];
    let mut q: VecDeque<usize> = VecDeque::with_capacity(st.n);
    for &src in sources {
        if src < st.n && d[src] < 0 {
            d[src] = 0;
            q.push_back(src);
        }
    }
    let size = st.size;
    while let Some(t) = q.pop_front() {
        let dt = d[t];
        let x = (t as i32) % size;
        let y = (t as i32) / size;
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= size || ny >= size {
                    continue;
                }
                let u = (ny * size + nx) as usize;
                if d[u] >= 0 || !passable(st, u) {
                    continue;
                }
                d[u] = dt + 1;
                q.push_back(u);
            }
        }
    }
    d
}

struct Ctx {
    my_units: Vec<usize>,
    enemies: Vec<usize>,
    my_cities: Vec<usize>,
    enemy_cities: Vec<usize>,
    villages: Vec<usize>,
    ruins: Vec<usize>,
    want_units: i32,
    fields: Vec<Vec<i32>>,
    pick: Vec<Option<(usize, i32)>>,
    guard: Vec<bool>,
    stars: i32,
    ncities: i32,
    threat: Vec<bool>,
    garrison: Vec<bool>,
    defensive: bool,
    size: i32,
}

fn tech_tier(k: i32) -> i32 {
    match k {
        1 | 6 | 10 | 15 | 20 => 1,
        2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 | 121 => 2,
        3 | 5 | 9 | 12 | 14 | 17 | 19 | 22 | 24 | 38 => 3,
        _ => 2,
    }
}

const PLAN: [i32; 16] = [1, 6, 8, 16, 18, 20, 23, 4, 24, 7, 21, 13, 10, 44, 17, 35];

struct Game {
    /// (unit tile, target tile, is_explore)
    goals: Vec<(usize, usize, bool)>,
}

impl Game {
    fn new() -> Game {
        Game { goals: Vec::new() }
    }

    fn decide(&mut self, st: &St) -> String {
        // 1. mandatory city rewards
        let mut best_rew: Option<&Act> = None;
        let mut best_r = -1000;
        for a in st.acts.iter() {
            if a.kind == 9 {
                let s = match a.param {
                    3 => 100,
                    7 => 95,
                    8 => 90,
                    6 => 85,
                    5 => 50,
                    4 => 45,
                    1 => 40,
                    2 => 30,
                    _ => 10,
                };
                if s > best_r {
                    best_r = s;
                    best_rew = Some(a);
                }
            }
        }
        if let Some(a) = best_rew {
            return a.id.clone();
        }

        // 2. context
        let mut c = Ctx {
            my_units: Vec::new(),
            enemies: Vec::new(),
            my_cities: Vec::new(),
            enemy_cities: Vec::new(),
            villages: Vec::new(),
            ruins: Vec::new(),
            want_units: 0,
            fields: Vec::new(),
            pick: vec![None; st.n],
            guard: vec![false; st.n],
            stars: 0,
            ncities: 0,
            threat: vec![false; st.n],
            garrison: vec![false; st.n],
            defensive: false,
            size: st.size,
        };
        for i in 0..st.n {
            if st.uowner[i] == st.me {
                c.my_units.push(i);
            } else if st.uowner[i] != 0 {
                c.enemies.push(i);
            }
            if st.strc[i] == 1 {
                if st.owner[i] == st.me {
                    c.my_cities.push(i);
                } else if st.owner[i] != 0 {
                    if st.known[i] {
                        c.enemy_cities.push(i);
                    }
                } else if st.known[i] {
                    c.villages.push(i);
                }
            } else if st.strc[i] == 2 && st.known[i] {
                c.ruins.push(i);
            }
        }
        let ncities = c.my_cities.len() as i32;
        c.ncities = ncities;
        c.stars = st.stars;
        c.want_units = 3 + 2 * ncities;
        c.my_units.sort();

        // 3. objectives (persistent across turns, spread across units)
        let unk: Vec<usize> = (0..st.n).filter(|&t| !st.known[t]).collect();
        let unkf = field(st, &unk);
        let mut cands: Vec<usize> = Vec::new();
        cands.extend(c.villages.iter().cloned());
        cands.extend(c.enemy_cities.iter().cloned());
        cands.extend(c.ruins.iter().cloned());

        let valid_target = |t: usize, explore: bool| -> bool {
            if t >= st.n {
                return false;
            }
            if explore {
                !st.known[t]
            } else {
                (st.strc[t] == 1 && st.owner[t] != st.me) || st.owner[t] == st.me
            }
        };

        // carry over goals from previous turns
        let mut keep: Vec<(usize, usize, bool)> = Vec::new();
        let mut used: Vec<usize> = Vec::new();
        for &u in c.my_units.iter() {
            let mut found: Option<(usize, usize, bool)> = None;
            for g in self.goals.iter() {
                if cheb(st.size, g.0, u) <= 1 && valid_target(g.1, g.2) {
                    found = Some((u, g.1, g.2));
                    break;
                }
            }
            if let Some(g) = found {
                used.push(g.1);
                keep.push(g);
            } else {
                keep.push((u, usize::MAX, false));
            }
        }
        self.goals = keep.clone();

        // 3a. garrisons for threatened cities
        for ci in 0..c.my_cities.len() {
            let ct = c.my_cities[ci];
            let de = c
                .enemies
                .iter()
                .map(|&e| cheb(st.size, e, ct))
                .min()
                .unwrap_or(99);
            if de <= 2 {
                c.garrison[ct] = true;
            }
            if de <= 6 {
                c.threat[ct] = true;
            }
        }
        c.defensive = c.threat.iter().any(|&x| x);
        for &u in c.my_units.iter() {
            if c.garrison[u] {
                c.guard[u] = true;
            }
        }
        for ci in 0..c.my_cities.len() {
            let ct = c.my_cities[ci];
            if !c.garrison[ct] {
                continue;
            }
            if c.my_units.iter().any(|&u| u == ct) {
                continue;
            }
            let mut pu = usize::MAX;
            let mut bd = i32::MAX;
            for &u in c.my_units.iter() {
                if c.guard[u] {
                    continue;
                }
                let d = cheb(st.size, u, ct);
                if d > 5 {
                    continue;
                }
                let v = d * 10 - st.uhp[u] as i32;
                if v < bd {
                    bd = v;
                    pu = u;
                }
            }
            if pu != usize::MAX {
                c.guard[pu] = true;
                for g in self.goals.iter_mut() {
                    if g.0 == pu {
                        g.1 = ct;
                        g.2 = false;
                    }
                }
            }
        }

        // 3b. new targets: villages/cities first, then frontier
        for iu in 0..c.my_units.len() {
            let u = c.my_units[iu];
            if c.guard[u] || self.goals[iu].1 != usize::MAX {
                continue;
            }
            let mut order: Vec<(i32, usize)> =
                cands.iter().map(|&t| (cheb(st.size, u, t), t)).collect();
            order.sort();
            let mut picked = usize::MAX;
            for &(_, t) in order.iter().take(10) {
                if used.contains(&t) {
                    continue;
                }
                picked = t;
                break;
            }
            if picked != usize::MAX {
                self.goals[iu] = (u, picked, false);
                used.push(picked);
                continue;
            }
            if c.defensive {
                let mut bct = usize::MAX;
                let mut bd = i32::MAX;
                for &ct in c.my_cities.iter() {
                    if !c.threat[ct] {
                        continue;
                    }
                    let d = cheb(st.size, u, ct);
                    if d < bd {
                        bd = d;
                        bct = ct;
                    }
                }
                if bct != usize::MAX {
                    self.goals[iu] = (u, bct, false);
                    used.push(bct);
                    continue;
                }
            }
            // exploration target
            let mut order: Vec<(i32, usize)> = Vec::new();
            for t in 0..st.n {
                if !st.known[t] {
                    let d = cheb(st.size, u, t);
                    if d < 2 {
                        continue;
                    }
                    if c.defensive
                        && !c
                            .my_cities
                            .iter()
                            .any(|&ct| cheb(st.size, ct, t) <= 5)
                    {
                        continue;
                    }
                    if c.enemies.iter().any(|&e| cheb(st.size, e, t) <= 3) {
                        continue;
                    }
                    order.push((d, t));
                }
            }
            order.sort();
            for &(_, t) in order.iter().take(20) {
                if used.iter().any(|&x| cheb(st.size, x, t) <= 3) {
                    continue;
                }
                picked = t;
                break;
            }
            if picked != usize::MAX {
                self.goals[iu] = (u, picked, true);
                used.push(picked);
            }
        }

        // 3c. distance fields per unit
        for iu in 0..c.my_units.len() {
            let u = c.my_units[iu];
            let t = self.goals[iu].1;
            if t == usize::MAX {
                continue;
            }
            let f = field(st, &[t]);
            let d = f[u];
            if d >= 1 {
                c.fields.push(f);
                let k = c.fields.len() - 1;
                c.pick[u] = Some((k, d));
            }
        }

        // 4. pick best action
        let mut best: Option<&Act> = None;
        let mut best_s = i32::MIN;
        let mut end_act: Option<&Act> = None;
        for a in st.acts.iter() {
            if a.kind == 11 || a.kind == 9 {
                continue;
            }
            if a.kind == 10 {
                end_act = Some(a);
                continue;
            }
            let s = self.score(st, a, &c);
            if s > best_s {
                best_s = s;
                best = Some(a);
            }
        }
        if best_s >= 6 {
            if let Some(a) = best {
                return a.id.clone();
            }
        }
        if let Some(a) = end_act {
            return a.id.clone();
        }
        st.acts
            .last()
            .map(|a| a.id.clone())
            .unwrap_or_else(|| "0".to_string())
    }

    fn score(&self, st: &St, a: &Act, c: &Ctx) -> i32 {
        match a.kind {
            8 => {
                let t = a.source as usize;
                let mut s = 1000;
                if st.strc[t] == 1 && st.owner[t] != 0 {
                    s += 300;
                }
                s
            }
            2 => {
                let src = a.source as usize;
                let dst = a.target as usize;
                if src >= st.n || dst >= st.n {
                    return -100;
                }
                if st.uowner[dst] == 0 || st.uowner[dst] == st.me {
                    return -100;
                }
                let ak = st.ukind[src];
                if ak == 10 {
                    return -100;
                }
                let dk = st.ukind[dst];
                let (ahpm, aatk, _, _) = stats(ak);
                let (dhpm, _, ddef, drange) = stats(dk);
                let am = ahpm + if false { 5.0 } else { 0.0 };
                let dm = dhpm + if false { 5.0 } else { 0.0 };
                let bonus = def_bonus(st, dst, dk);
                let (dmg, ret) = combat(aatk, st.uhp[src], am, ddef, st.uhp[dst], dm, bonus);
                let killed = dmg >= st.uhp[dst];
                let mut s;
                if killed {
                    s = 380 + 15 * unit_value(dk);
                } else {
                    s = (7.0 * dmg - 7.0 * ret) as i32 + 2;
                }
                let can_ret = ret > 0.0 && cheb(st.size, src, dst) <= drange;
                if !can_ret {
                    s += 20;
                }
                if ret >= st.uhp[src] && !killed {
                    s -= 250;
                }
                if c.my_cities
                    .iter()
                    .any(|&ct| cheb(st.size, dst, ct) <= 2)
                {
                    s += 25;
                }
                if c.guard[src] {
                    s -= 300;
                }
                s
            }
            1 => {
                let src = a.source as usize;
                let dst = a.target as usize;
                if src >= st.n || dst >= st.n || st.uowner[src] != st.me {
                    return -100;
                }
                let mut s = 0;
                if let Some((k, d0)) = c.pick[src] {
                    let d1 = c.fields[k][dst];
                    if d0 > 0 && d1 >= 0 && d1 < d0 {
                        s = 12 * (d0 - d1);
                    }
                }
                if !c.guard[src] {
                    let reveal = unk_adj(st, dst);
                    if reveal > 0 && adj_enemies(st, dst) == 0 {
                        s += (3 * reveal).min(12);
                    }
                    if st.strc[dst] == 1 && st.owner[dst] == 0 {
                        s += 6;
                    }
                }
                s
            }
            4 => {
                let mut s = 0;
                let n_units = c.my_units.len() as i32;
                if n_units < c.want_units {
                    s = 58;
                }
                match a.param {
                    11 => s += 16,
                    9 => s += 8,
                    3 => {
                        if c.defensive {
                            s += 2
                        } else {
                            s += 10
                        }
                    }
                    5 => s += 6,
                    2 => s += 4,
                    43 | 42 | 44 | 41 | 45 => s -= 15,
                    _ => s -= 8,
                }
                let t = a.source as usize;
                if t < st.n && c.threat[t] {
                    s += 45;
                }
                if c.defensive {
                    s += 8;
                }
                if s < 0 {
                    s = 0;
                }
                s
            }
            7 => {
                let tier = tech_tier(a.param);
                let cost = 4 + c.ncities * tier;
                let slack = (c.stars - cost).max(0).min(24);
                let pos = PLAN.iter().position(|&p| p == a.param);
                let base = match pos {
                    Some(p) if p < 4 => 46,
                    Some(p) if p < 8 => 40,
                    Some(_) => 32,
                    None => 20,
                };
                base + slack
            }
            5 => {
                if c.defensive {
                    52
                } else {
                    62
                }
            }
            6 => match a.param {
                5 | 21 | 12 => 54,
                6 | 13 | 22 => 44,
                8 => 40,
                23 | 24 | 25 | 26 | 27 | 28 | 29 | 123 => 74,
                17 | 18 | 19 | 20 => 22 + (c.stars - 20).max(0).min(12),
                3 => 10,
                48 => 6,
                _ => 4,
            },
            3 => match a.param {
                16 => 92,
                7 => {
                    let t = a.source as usize;
                    if t < st.n && st.uhp[t] * 2.0 < stats(st.ukind[t]).0 {
                        30
                    } else {
                        0
                    }
                }
                _ => -5,
            },
            _ => 0,
        }
    }
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut game = Game::new();
    let mut buf = String::new();
    let mut lock = stdin.lock();
    loop {
        buf.clear();
        match lock.read_line(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if buf.trim().is_empty() {
            continue;
        }
        let id = match serde_json::from_str::<Value>(&buf) {
            Ok(v) => match St::parse(&v) {
                Some(st) => game.decide(&st),
                None => "0".to_string(),
            },
            Err(_) => "0".to_string(),
        };
        let _ = writeln!(stdout, "{{\"action_id\":\"{}\"}}", id);
        let _ = stdout.flush();
    }
}
