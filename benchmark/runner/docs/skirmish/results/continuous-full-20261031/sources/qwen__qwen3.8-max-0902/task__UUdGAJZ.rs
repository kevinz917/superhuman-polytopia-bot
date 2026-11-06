// Bardur Polytopia 1v1 bot: heuristic policy over the legal action catalog.
use serde_json::{json, Value};
use std::cell::RefCell;
use std::io::{self, BufRead, Write};

thread_local! {
    static MOVES: RefCell<Vec<(i32, usize, usize)>> = RefCell::new(Vec::new());
}

// ---------- unit tables ----------
// cost, hp, atk, def, mov, range
fn ustat(k: i32) -> (f32, f32, f32, f32, i32, i32) {
    match k {
        2 => (2., 10., 2., 2., 1, 1),
        3 => (3., 10., 2., 1., 2, 1),
        4 => (8., 10., 3.5, 1., 3, 1),
        5 => (3., 15., 1., 3., 1, 1),
        8 => (8., 10., 4., 0., 1, 3),
        9 => (3., 10., 2., 1., 1, 2),
        10 => (5., 10., 0., 1., 1, 1),
        11 => (5., 15., 3., 3., 1, 1),
        12 => (10., 40., 5., 4., 1, 1),
        15 => (2., 15., 3., 1., 1, 1),
        20 => (3., 10., 2., 1., 2, 1),
        21 => (8., 10., 2.5, 1., 2, 2),
        26 => (10., 40., 4., 5., 2, 1),
        37 => (2., 10., 2., 2., 1, 1),
        40 => (2., 10., 2., 2., 2, 1),
        41 => (15., 10., 3., 2., 2, 3),
        42 => (5., 10., 2., 1., 3, 2),
        43 => (0., 10., 0., 1., 2, 2),
        44 => (5., 10., 3., 3., 3, 1),
        45 => (10., 40., 4., 4., 2, 1),
        _ => (5., 10., 1., 1., 1, 1),
    }
}
fn fortify_ok(k: i32) -> bool { matches!(k, 2 | 3 | 5 | 9 | 15 | 20 | 25) }
fn stiff_ok(k: i32) -> bool { matches!(k, 8 | 10 | 22 | 41) }

// ---------- state ----------
struct St {
    size: i32,
    n: usize,
    me: i32,
    opp: i32,
    stars: i32,
    turn: i32,
    tech: [bool; 128],
    known: Vec<bool>,
    terr: Vec<i32>,
    owner: Vec<i32>,
    res: Vec<i32>,
    stru: Vec<i32>,
    road: Vec<i32>,
    cap: Vec<i32>,
    uk: Vec<i32>,
    uo: Vec<i32>,
    uh: Vec<f32>,
    umh: Vec<f32>,
    umov: Vec<bool>,
    uatt: Vec<bool>,
    uvet: Vec<bool>,
    cown: Vec<i32>,
    clvl: Vec<i32>,
    cprog: Vec<i32>,
    cprod: Vec<i32>,
    creward: Vec<i32>,
}

fn gi(v: &Value, k: &str) -> i32 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(0) as i32
}
fn gf(v: &Value, k: &str) -> f32 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32
}
fn gb(v: &Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}

fn parse(obs: &Value) -> Option<St> {
    let v = obs.get("visible_state")?;
    let size = gi(v, "board_size");
    if size < 2 || size > 64 {
        return None;
    }
    let n = (size * size) as usize;
    let me = gi(obs, "player");
    let mut s = St {
        size,
        n,
        me,
        opp: 3 - me,
        stars: gi(v, "stars"),
        turn: gi(v, "turn"),
        tech: [false; 128],
        known: vec![false; n],
        terr: vec![3; n],
        owner: vec![0; n],
        res: vec![0; n],
        stru: vec![0; n],
        road: vec![0; n],
        cap: vec![0; n],
        uk: vec![0; n],
        uo: vec![0; n],
        uh: vec![0.; n],
        umh: vec![1.; n],
        umov: vec![false; n],
        uatt: vec![false; n],
        uvet: vec![false; n],
        cown: vec![0; n],
        clvl: vec![0; n],
        cprog: vec![0; n],
        cprod: vec![0; n],
        creward: vec![0; n],
    };
    if let Some(ts) = v.get("technology").and_then(|x| x.as_array()) {
        for t in ts {
            let id = t.as_i64().unwrap_or(-1);
            if id >= 0 && id < 128 {
                s.tech[id as usize] = true;
            }
        }
    }
    if let Some(ts) = v.get("tiles").and_then(|x| x.as_array()) {
        for t in ts {
            let i = gi(t, "tile");
            if i < 0 || i as usize >= n {
                continue;
            }
            let i = i as usize;
            s.known[i] = gb(t, "known");
            if s.known[i] {
                s.terr[i] = gi(t, "terrain");
                s.owner[i] = gi(t, "owner");
                s.res[i] = gi(t, "resource");
                s.stru[i] = gi(t, "structure");
                s.road[i] = gi(t, "road");
                s.cap[i] = gi(t, "capital");
            }
        }
    }
    if let Some(us) = v.get("units").and_then(|x| x.as_array()) {
        for u in us {
            let i = gi(u, "tile");
            if i < 0 || i as usize >= n {
                continue;
            }
            let i = i as usize;
            let k = gi(u, "kind");
            s.uk[i] = k;
            s.uo[i] = gi(u, "owner");
            let hp = gf(u, "health");
            s.uh[i] = if hp > 0.0 { hp } else { 1.0 };
            let (_, bhp, _, _, _, _) = ustat(k);
            s.umh[i] = bhp + if gb(u, "veteran") { 5.0 } else { 0.0 };
            s.umov[i] = gb(u, "moved");
            s.uatt[i] = gb(u, "attacked");
            s.uvet[i] = gb(u, "veteran");
        }
    }
    if let Some(cs) = v.get("cities").and_then(|x| x.as_array()) {
        for c in cs {
            let i = gi(c, "tile");
            if i < 0 || i as usize >= n {
                continue;
            }
            let i = i as usize;
            s.cown[i] = gi(c, "owner");
            s.clvl[i] = gi(c, "level");
            s.cprog[i] = gi(c, "progress");
            s.cprod[i] = gi(c, "production");
            if let Some(r) = c.get("rewards").and_then(|x| x.as_array()) {
                let mut sum = 0;
                for x in r {
                    sum += x.as_i64().unwrap_or(0) as i32;
                }
                s.creward[i] = sum;
            }
        }
    }
    Some(s)
}

impl St {
    fn d(&self, a: usize, b: usize) -> i32 {
        let sz = self.size as usize;
        let dx = (a % sz) as i32 - (b % sz) as i32;
        let dy = (a / sz) as i32 - (b / sz) as i32;
        if dx.abs() > dy.abs() { dx.abs() } else { dy.abs() }
    }
    fn my_units(&self) -> Vec<usize> {
        (0..self.n).filter(|&i| self.uo[i] == self.me).collect()
    }
    fn foe_units(&self) -> Vec<usize> {
        (0..self.n).filter(|&i| self.uo[i] == self.opp).collect()
    }
    fn my_cities(&self) -> Vec<usize> {
        (0..self.n).filter(|&i| self.cown[i] == self.me).collect()
    }
    fn foe_cities(&self) -> Vec<usize> {
        (0..self.n).filter(|&i| self.cown[i] == self.opp).collect()
    }
    // defense bonus for a defender of `owner` standing on tile i
    fn dbonus(&self, i: usize, owner: i32, dkind: i32) -> f32 {
        let t = self.terr[i];
        let mine = owner == self.me;
        if t == 1 || t == 2 {
            if mine && self.tech[12] { 1.5 } else { 1.0 }
        } else if t == 5 {
            if mine && self.tech[18] { 1.5 } else { 1.0 }
        } else if t == 4 {
            if mine && self.tech[20] { 1.5 } else { 1.0 }
        } else {
            if self.cown[i] == owner && fortify_ok(dkind) {
                if mine && self.creward[i] >= 1 { 4.0 } else { 1.5 }
            } else {
                1.0
            }
        }
    }
}

fn combat(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let tot = af + df;
    let div = if tot > 0.0 { tot } else { 1.0 };
    let atk = ((if tot > 0.0 { af / div * aa * 4.5 } else { 0.0 }) + 0.000001 + 0.5).floor();
    let ret = ((if dh > atk && tot > 0.0 { df / div * dd * 4.5 } else { 0.0 }) + 0.000001 + 0.5).floor();
    (atk, ret)
}

struct Goal {
    t: usize,
    val: f64,
}

fn choose(obs: &Value) -> String {
    let legal = match obs.get("legal_actions").and_then(|x| x.as_array()) {
        Some(a) => a,
        None => return String::new(),
    };
    if legal.is_empty() {
        return String::new();
    }
    let fallback = legal[0].get("action_id").and_then(|x| x.as_str()).unwrap_or("").to_string();
    let st = match parse(obs) {
        Some(s) => s,
        None => return fallback,
    };
    let n = st.n;
    let myu = st.my_units();
    let foeu = st.foe_units();
    let myc = st.my_cities();
    let foec = st.foe_cities();

    // army strength
    let mut mypow = 0f64;
    for &i in &myu {
        let (_, hp, atk, def, _, _) = ustat(st.uk[i]);
        mypow += (atk as f64 * 2.0 + def as f64 * 1.5 + 2.0) * (st.uh[i] / st.umh[i]) as f64;
    }
    let mut foepow = 0f64;
    for &i in &foeu {
        let (_, hp, atk, def, _, _) = ustat(st.uk[i]);
        foepow += (atk as f64 * 2.0 + def as f64 * 1.5 + 2.0) * (st.uh[i] / st.umh[i]) as f64;
    }
    let aggressive = mypow > foepow * 1.1 + 4.0 || (foeu.is_empty() && !foec.is_empty());

    // threats: nearest foe unit distance to each of our cities
    let mut city_threat = vec![99i32; n];
    for &c in &myc {
        let mut best = 99;
        for &f in &foeu {
            let d = st.d(c, f);
            if d < best {
                best = d;
            }
        }
        city_threat[c] = best;
    }
    let any_threat = myc.iter().any(|&c| city_threat[c] <= 2);

    // danger map: max attacker power that can reach a tile this turn
    let mut danger = vec![0f32; n];
    for &f in &foeu {
        let (_, _, atk, _, mov, rng) = ustat(st.uk[f]);
        let reach = mov + rng;
        let pw = atk * (st.uh[f] / st.umh[f]);
        for t in 0..n {
            if st.d(f, t) <= reach && pw > danger[t] {
                danger[t] = pw;
            }
        }
    }

    // goals
    let mut goals: Vec<Goal> = Vec::new();
    for i in 0..n {
        if !st.known[i] {
            continue;
        }
        if st.stru[i] == 1 && st.owner[i] == 0 {
            goals.push(Goal { t: i, val: 2000.0 });
        } else if st.stru[i] == 2 {
            goals.push(Goal { t: i, val: 320.0 });
        } else if st.cown[i] == st.opp {
            goals.push(Goal { t: i, val: if aggressive { 900.0 } else { 180.0 } });
        } else if st.res[i] == 8 && st.tech[14] {
            goals.push(Goal { t: i, val: 200.0 });
        }
    }
    for &f in &foeu {
        let (_, dhp, _, ddef, _, _) = ustat(st.uk[f]);
        let mut best = 0f64;
        for &u in &myu {
            let (_, _, atk, _, _, rng) = ustat(st.uk[u]);
            let db = st.dbonus(f, st.opp, st.uk[f]);
            let (dmg, ret) = combat(atk, st.uh[u], st.umh[u], ddef, st.uh[f], st.umh[f], db);
            let kill = st.uh[f] - dmg <= 0.0;
            let v = if kill { 520.0 } else { (dmg / dhp.max(1.0)) as f64 * 160.0 }
                - (ret / st.umh[u].max(1.0)) as f64 * 120.0;
            let _ = rng;
            if v > best {
                best = v;
            }
        }
        if best > 0.0 {
            goals.push(Goal { t: f, val: best });
        }
    }
    // defend threatened cities
    if any_threat {
        for &c in &myc {
            if city_threat[c] <= 3 {
                goals.push(Goal { t: c, val: 850.0 });
            }
        }
    }

    let mut unk1 = vec![0i32; n];
    {
        let sz = st.size as usize;
        for i in 0..n {
            let (x, y) = (i % sz, i / sz);
            let mut c = 0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= st.size || ny >= st.size {
                        continue;
                    }
                    let j = ny as usize * sz + nx as usize;
                    if !st.known[j] {
                        c += 1;
                    }
                }
            }
            unk1[i] = c;
        }
    }
    let mut rev: Vec<(usize, usize)> = Vec::new();
    MOVES.with(|m| {
        let mut v = m.borrow_mut();
        v.retain(|&(t, _, _)| t >= st.turn - 3);
        for &(_, s0, d0) in v.iter() {
            rev.push((d0, s0));
        }
    });

    // rank each goal per unit (closest unit gets full value)
    let mut rank_adj: Vec<Vec<f64>> = Vec::new(); // per unit: per goal value
    for g in &goals {
        let mut order: Vec<(i32, usize)> = myu.iter().map(|&u| (st.d(u, g.t), u)).collect();
        order.sort();
        let _ = order;
    }
    let nu = myu.len();
    let mut gval = vec![0f64; nu * goals.len()];
    for (gi, g) in goals.iter().enumerate() {
        let mut order: Vec<(i32, usize)> = myu.iter().enumerate().map(|(idx, &u)| (st.d(u, g.t), idx)).collect();
        order.sort();
        for (rank, (_, idx)) in order.iter().enumerate() {
            gval[idx * goals.len() + gi] = g.val - 40.0 * rank as f64;
        }
    }
    let _ = rank_adj;

    // research priority
    fn tech_prio(id: i32, st: &St, aggressive: bool, ncity: usize) -> f64 {
        let base = match id {
            6 => 100.0,  // Organization
            1 => 92.0,   // Riding
            8 => 96.0,   // Farming
            20 => 90.0,  // Climbing
            23 => 88.0,  // Mining
            24 => 86.0,  // Smithery
            18 => 82.0,  // Archery
            17 => 80.0,  // Mathematics (catapult)
            3 => 78.0,   // Chivalry (knight)
            7 => 74.0,   // Strategy (defender)
            5 => 70.0,   // Trade (market)
            9 => 68.0,   // Construction (windmill)
            16 => 64.0,  // Forestry
            4 => 60.0,   // Roads
            10 => 56.0,  // Fishing
            44 => 50.0,  // Ramming
            13 => 46.0,  // Sailing
            2 => 44.0,   // FreeSpirit
            22 => 40.0,  // Philosophy
            19 => 34.0,  // Spiritualism
            21 => 32.0,  // Meditation
            12 => 30.0,
            14 => 30.0,
            _ => 20.0,
        };
        let mut b = base;
        if aggressive && matches!(id, 24 | 17 | 3 | 7) {
            b += 12.0;
        }
        if ncity <= 2 {
            // early: cheap high tier techs are a bargain
            if matches!(id, 24 | 17 | 3 | 5 | 9 | 22) {
                b += 8.0;
            }
        }
        b
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut best_id = fallback.clone();
    let ncity = myc.len();

    for a in legal {
        let id = match a.get("action_id").and_then(|x| x.as_str()) {
            Some(s) => s,
            None => continue,
        };
        let c = match a.get("command") {
            Some(c) => c,
            None => continue,
        };
        let kind = gi(c, "kind");
        let src = gi(c, "source");
        let dst = gi(c, "target");
        let par = gi(c, "parameter");
        let si = src as usize;
        let di = dst as usize;
        let ok_t = |x: usize| x < n;
        let mut score: f64 = -1e5;
        match kind {
            11 => score = -1e9,
            10 => score = 0.0,
            9 => {
                let slot = if ok_t(di) { st.creward[di] + 1 } else { 1 };
                let pref = match par {
                    3 => 100.0, // workshop: +1 production
                    4 => 60.0,  // explorer reveal
                    7 => 90.0,  // +5 stars
                    1 => {
                        if any_threat || st.turn > 40 { 95.0 } else { 40.0 }
                    }
                    8 => 85.0, // +3 population
                    5 => 80.0, // border growth
                    6 => {
                        if !foeu.is_empty() || !foec.is_empty() { 110.0 } else { 50.0 }
                    }
                    2 => 70.0, // park: +1 production
                    _ => 10.0,
                };
                let _ = slot;
                score = 1e6 + pref;
            }
            8 => {
                let t = if ok_t(si) { si } else { di };
                let base = if ok_t(t) && st.cown[t] == st.opp {
                    5e5 + 2000.0
                } else if ok_t(t) && st.stru[t] == 1 {
                    5e5 + 900.0
                } else if ok_t(t) && st.stru[t] == 2 {
                    5e5 + 500.0
                } else {
                    5e5 + 200.0
                };
                score = base;
            }
            2 => {
                if ok_t(si) && ok_t(di) && st.uo[si] == st.me && st.uo[di] != 0 {
                    let ak = st.uk[si];
                    let dk = st.uk[di];
                    let (_, _, aatk, _, _, arng) = ustat(ak);
                    let (dcost, _, _, ddef, _, drng) = ustat(dk);
                    let (_, acost, _, _, _, _) = (0, ustat(ak).1, 0, 0, 0, 0);
                    let db = st.dbonus(di, st.uo[di], dk);
                    let (dmg, ret) = combat(aatk, st.uh[si], st.umh[si], ddef, st.uh[di], st.umh[di], db);
                    let dist = st.d(si, di);
                    let can_ret = !stiff_ok(dk) && dist <= drng;
                    let ret = if can_ret { ret } else { 0.0 };
                    let edead = st.uh[di] - dmg <= 0.0;
                    let adead = st.uh[si] - ret <= 0.0;
                    let gain = (dmg / st.umh[di].max(1.0)) as f64 * (25.0 + 10.0 * dcost as f64)
                        + if edead { 70.0 + 14.0 * dcost as f64 } else { 0.0 };
                    let loss = (ret / st.umh[si].max(1.0)) as f64 * (25.0 + 10.0 * acost as f64)
                        + if adead { 70.0 + 14.0 * acost as f64 } else { 0.0 };
                    let mut ev = gain - loss * 1.15;
                    if edead && !adead {
                        if st.cown[di] == st.opp {
                            ev += 1500.0;
                        } else if st.stru[di] == 1 {
                            ev += 400.0;
                        }
                        if city_threat.get(di).copied().unwrap_or(99) <= 3 {
                            ev += 200.0;
                        }
                    }
                    if st.cown[di] == st.opp {
                        ev -= 40.0; // city garrison may have a wall
                    }
                    let mut foes_near = 0;
                    let mut mates_near = 0;
                    for &f in &foeu {
                        if f != di && st.d(f, si) <= 2 {
                            foes_near += 1;
                        }
                    }
                    for &m in &myu {
                        if m != si && st.d(m, si) <= 2 {
                            mates_near += 1;
                        }
                    }
                    if foes_near > mates_near {
                        ev -= 55.0 * (foes_near - mates_near) as f64;
                    }
                    if ev > 5.0 {
                        score = 1e5 + ev;
                    } else {
                        score = -1e5 + ev;
                    }
                    let _ = arng;
                }
            }
            5 => {
                if ok_t(di) {
                    let r = st.res[di];
                    let wasteful = (r == 2 && st.tech[8]) || (r == 5 && st.tech[23]) || (r == 7);
                    let cityi = st.cown[di];
                    let _ = cityi;
                    score = if wasteful {
                        9000.0
                    } else {
                        20000.0 + 30.0 * st.stars.min(10) as f64
                    };
                }
            }
            6 => {
                if ok_t(di) {
                    let val = match par {
                        5 => 320.0,  // farm
                        21 => 320.0, // mine
                        22 => 280.0, // forge
                        50 => 300.0, // market
                        12 => 220.0, // lumber hut
                        6 => 170.0,  // windmill
                        13 => 170.0, // sawmill
                        37 => 150.0, // fungi
                        8 => 130.0,  // port
                        33 => 120.0, // outpost
                        48 => 100.0, // bridge
                        32 => 90.0,  // sanctuary
                        3 => 55.0,   // road
                        _ => 60.0,
                    };
                    score = 30000.0 + val;
                }
            }
            7 => {
                let cost = 4 + (ncity as i32) * tier_of(par);
                let p = tech_prio(par, &st, aggressive, ncity);
                score = 15000.0 + p * 10.0 - 3.0 * cost as f64;
            }
            4 => {
                let (cost, hp, atk, def, mov, rng) = ustat(par);
                let power = atk * 2.0 + def * 1.5 + (rng - 1) as f32 * 3.0 + mov as f32 * 4.0;
                let mut val = 26.0 * power as f64 - 10.0 * cost as f64;
                let defensive = def >= 3.0 || par == 5;
                // need defenders when threatened
                if any_threat && defensive {
                    val += 160.0;
                }
                if aggressive && atk >= 3.0 {
                    val += 90.0;
                }
                if myu.len() + 1 < foeu.len() + 2 {
                    val += 70.0;
                }
                if par == 2 && myu.len() >= 4 {
                    val -= 160.0;
                }
                if mov >= 2 && st.turn < 40 {
                    val += 70.0;
                }
                if par == 5 && any_threat {
                    val += 120.0;
                }
                val -= 18.0 * myu.len() as f64;
                if ok_t(si) && st.cown[si] == st.me && city_threat[si] <= 3 && defensive {
                    val += 120.0;
                }
                // keep some stars for research/builds
                if (st.stars as f32) - cost < 2.0 {
                    val -= 40.0;
                }
                let _ = hp;
                score = 26000.0 + val;
            }
            3 => {
                score = match par {
                    16 => 8000.0, // promote
                    9 => {
                        // heal others
                        if ok_t(si) {
                            let mut hurt = false;
                            let sz = st.size as usize;
                            let (x, y) = (si % sz, si / sz);
                            for dy in -1i32..=1 {
                                for dx in -1i32..=1 {
                                    let nx = x as i32 + dx;
                                    let ny = y as i32 + dy;
                                    if nx < 0 || ny < 0 || nx >= st.size || ny >= st.size {
                                        continue;
                                    }
                                    let j = ny as usize * sz + nx as usize;
                                    if st.uo[j] == st.me && st.uh[j] < st.umh[j] {
                                        hurt = true;
                                    }
                                }
                            }
                            if hurt { 7500.0 } else { -1e5 }
                        } else {
                            -1e5
                        }
                    }
                    7 => {
                        // recover
                        if ok_t(si) && st.uh[si] < st.umh[si] {
                            let frac = st.uh[si] / st.umh[si];
                            if frac < 0.65 {
                                4000.0 + (1.0 - frac as f64) * 1000.0
                            } else {
                                1000.0
                            }
                        } else {
                            -1e5
                        }
                    }
                    19 => {
                        if dst == 1 {
                            if mypow < foepow * 0.8 { 2500.0 } else { -1e5 }
                        } else {
                            if mypow >= foepow * 0.8 { 2500.0 } else { -1e5 }
                        }
                    }
                    _ => -1e5,
                };
            }
            1 => {
                if ok_t(si) && ok_t(di) && st.uo[si] == st.me && st.stru[di] == 1 && st.owner[di] != st.me && st.uo[di] == 0 {
                    // walk onto a village / enemy city tile so we can capture next turn
                    score = if st.cown[di] == st.opp { 4.5e5 } else { 4e5 } - 20.0 * st.d(si, di) as f64;
                } else if ok_t(si) && ok_t(di) && st.uo[si] == st.me && st.stru[di] == 2 && st.uo[di] == 0 {
                    score = 3.5e5;
                } else if ok_t(si) && ok_t(di) && st.uo[si] == st.me {
                    let uidx = myu.iter().position(|&u| u == si);
                    let mut fs = -1e4;
                    if let Some(ui) = uidx {
                        for (gidx, g) in goals.iter().enumerate() {
                            let gv = gval[ui * goals.len() + gidx];
                            let v = gv - 9.0 * st.d(di, g.t) as f64;
                            if v > fs {
                                fs = v;
                            }
                        }
                    } else {
                        fs = -50.0 - 9.0 * st.d(di, si) as f64;
                    }
                    // terrain / safety
                    let (_, _, _, udef, _, _) = ustat(st.uk[si]);
                    let hpf = (st.uh[si] / st.umh[si]) as f64;
                    let dg = danger[di];
                    let dbon = st.dbonus(di, st.me, st.uk[si]);
                    let est = dg * 4.5 * dg / (dg + udef * dbon).max(0.5);
                    let mut pen = est as f64 * 6.0 * hpf;
                    if est as f32 >= st.uh[si] {
                        pen += 500.0;
                    }
                    if st.terr[di] == 5 {
                        fs += 6.0;
                    }
                    if st.terr[di] == 4 {
                        fs += 8.0;
                    }
                    if st.road[di] == 1 {
                        fs += 3.0;
                    }
                    if st.cown[di] == st.me {
                        fs += 12.0;
                    }
                    fs += 20.0 * unk1[di] as f64;
                    if goals.is_empty() || unk1[di] == 0 {
                        let home = myc.iter().map(|&c| st.d(di, c)).min().unwrap_or(0);
                        fs += 3.0 * home as f64;
                    }
                    if rev.contains(&(di, si)) {
                        pen += 500.0;
                    }
                    if rev.contains(&(di, di)) {
                        pen += 0.0;
                    }
                    // stay near home when weak
                    if !aggressive && foepow > mypow {
                        let home = myc.iter().map(|&c| st.d(di, c)).min().unwrap_or(9);
                        if home > 4 {
                            pen += 60.0 * (home - 4) as f64;
                        }
                    }
                    score = 5000.0 + fs - pen;
                }
            }
            _ => score = -1e5,
        }
        if score > best_score {
            best_score = score;
            best_id = id.to_string();
        }
    }
    if best_score > 4000.0 {
        for a in legal {
            if a.get("action_id").and_then(|x| x.as_str()).unwrap_or("") == best_id {
                if let Some(c) = a.get("command") {
                    if gi(c, "kind") == 1 {
                        let s0 = gi(c, "source") as usize;
                        let d0 = gi(c, "target") as usize;
                        MOVES.with(|m| m.borrow_mut().push((st.turn, s0, d0)));
                    }
                }
                break;
            }
        }
    }
    if best_id.is_empty() {
        return fallback;
    }
    best_id
}

fn tier_of(id: i32) -> i32 {
    match id {
        0 | 15 => 0,
        1 | 6 | 10 | 20 | 25 | 26 | 28 | 30 | 31 | 32 | 35 | 36 | 37 | 41 | 42 | 44 | 48 | 49 | 120 | 121 => 1,
        2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 => 2,
        3 | 5 | 9 | 12 | 14 | 17 | 19 | 22 | 24 | 38 => 3,
        _ => 2,
    }
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let obs: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = choose(&obs);
        let out = json!({ "action_id": id });
        if writeln!(stdout, "{}", out).is_err() {
            break;
        }
        if stdout.flush().is_err() {
            break;
        }
    }
}
