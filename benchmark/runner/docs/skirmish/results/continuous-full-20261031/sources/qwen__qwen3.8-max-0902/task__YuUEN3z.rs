// Bardur Polytopia 1v1 bot: scored-greedy over the supplied legal action catalog.
use serde_json::Value;
use std::io::{self, BufRead, Write};

// ---------- unit stat tables ----------
fn uhp(k: i64) -> f64 {
    match k {
        5 | 11 => 15.0,
        12 => 40.0,
        38 | 39 => 5.0,
        45 => 40.0,
        _ => 10.0,
    }
}
fn uatk(k: i64) -> f64 {
    match k {
        2 | 3 | 9 | 20 => 2.0,
        4 => 3.5,
        5 => 1.0,
        8 => 4.0,
        10 => 0.0,
        11 | 18 | 23 | 28 | 32 | 33 | 35 | 40 | 41 | 59 => 3.0,
        12 => 5.0,
        15 => 3.0,
        19 => 4.0,
        21 => 2.5,
        24 | 26 | 27 | 29 | 44 | 45 | 57 => 4.0,
        38 | 39 | 43 => 0.0,
        42 => 2.0,
        _ => 2.0,
    }
}
fn udef(k: i64) -> f64 {
    match k {
        2 | 37 | 40 => 2.0,
        3 | 9 | 20 | 42 => 1.0,
        4 | 21 | 29 | 32 => 1.0,
        5 | 11 | 23 | 31 | 44 | 59 => 3.0,
        8 | 41 => 0.0,
        10 | 22 | 25 | 30 | 35 | 36 => 1.0,
        12 => 4.0,
        15 => 1.0,
        24 | 27 => 3.0,
        26 | 45 => 5.0,
        38 | 39 => 0.5,
        43 => 1.0,
        _ => 1.0,
    }
}
fn umv(k: i64) -> i64 {
    match k {
        4 | 19 | 29 | 42 => 3,
        3 | 18 | 20 | 21 | 23 | 26 | 28 | 30 | 35 | 38 | 39 | 40 | 41 | 43 | 44 | 45 | 56 => 2,
        _ => 1,
    }
}
fn urng(k: i64) -> i64 {
    match k {
        8 | 32 | 41 => 3,
        9 | 19 | 21 | 24 | 25 | 30 | 42 => 2,
        _ => 1,
    }
}
fn ucost(k: i64) -> f64 {
    match k {
        2 | 15 | 37 | 39 | 40 => 2.0,
        3 | 5 | 9 | 20 | 25 | 28 | 30 | 31 => 3.0,
        4 | 8 | 21 | 32 => 8.0,
        10 | 11 | 23 | 35 | 36 | 42 | 44 | 56 | 59 => 5.0,
        12 | 17 | 18 | 19 | 26 | 29 | 41 | 45 => 10.0,
        43 => 0.0,
        _ => 3.0,
    }
}
fn udash(k: i64) -> bool {
    matches!(k, 2 | 3 | 4 | 9 | 11 | 15 | 18 | 20 | 21 | 23 | 28 | 29 | 33 | 34 | 35 | 37 | 39 | 40 | 42 | 44 | 56 | 59 | 62)
}
fn ustiff(k: i64) -> bool {
    matches!(k, 8 | 10 | 17 | 22 | 25 | 32 | 34 | 38 | 39 | 41 | 43 | 45 | 60)
}
fn ufort(k: i64) -> bool {
    matches!(k, 2 | 3 | 4 | 5 | 9 | 17 | 20 | 25)
}
fn uscout(k: i64) -> bool {
    matches!(k, 18 | 19 | 24 | 38 | 39 | 42)
}
fn ufloat(k: i64) -> bool {
    matches!(k, 39 | 40 | 41 | 42 | 43 | 44 | 45)
}

fn cheb(a: usize, b: usize, size: usize) -> i64 {
    let (ax, ay) = ((a % size) as i64, (a / size) as i64);
    let (bx, by) = ((b % size) as i64, (b / size) as i64);
    (ax - bx).abs().max((ay - by).abs())
}

const NEG: f64 = -1e9;

struct St {
    size: usize,
    n: usize,
    me: i64,
    stars: i64,
    turn: i64,
    known: Vec<bool>,
    terr: Vec<i64>,
    own: Vec<i64>,
    res: Vec<i64>,
    stru: Vec<i64>,
    road: Vec<bool>,
    cap: Vec<i64>,
    uo: Vec<i64>,
    uk: Vec<i64>,
    uh: Vec<f64>,
    uvet: Vec<bool>,
    clev: Vec<i64>,
    cown: Vec<i64>,
    cwall: Vec<bool>,
    tech: [bool; 128],
    mycities: Vec<usize>,
    mycap: Option<usize>,
    enemycities: Vec<usize>,
    villages: Vec<usize>,
    myunits: Vec<usize>,
    enemyunits: Vec<usize>,
    // derived
    pot: Vec<f64>,
    danger: Vec<f64>,
    aggr: f64,
    naval: bool,
    crowd: Vec<f64>,
}

fn gi(v: &Value, k: &str) -> i64 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(-1)
}
fn gf(v: &Value, k: &str) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0)
}
fn gb(v: &Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}

impl St {
    fn parse(obs: &Value) -> Option<St> {
        let v = obs.get("visible_state")?;
        let size = v.get("board_size")?.as_i64()? as usize;
        if size < 2 || size > 64 {
            return None;
        }
        let n = size * size;
        let me = obs.get("player")?.as_i64()?;
        let mut s = St {
            size,
            n,
            me,
            stars: gi(v, "stars"),
            turn: gi(v, "turn"),
            known: vec![false; n],
            terr: vec![0; n],
            own: vec![0; n],
            res: vec![0; n],
            stru: vec![0; n],
            road: vec![false; n],
            cap: vec![0; n],
            uo: vec![0; n],
            uk: vec![0; n],
            uh: vec![0.0; n],
            uvet: vec![false; n],
            clev: vec![0; n],
            cown: vec![0; n],
            cwall: vec![false; n],
            tech: [false; 128],
            mycities: Vec::new(),
            mycap: None,
            enemycities: Vec::new(),
            villages: Vec::new(),
            myunits: Vec::new(),
            enemyunits: Vec::new(),
            pot: vec![NEG; n],
            danger: vec![0.0; n],
            aggr: 0.5,
            naval: false,
            crowd: vec![0.0; n],
        };
        if let Some(ts) = v.get("tiles").and_then(|x| x.as_array()) {
            for t in ts {
                let i = gi(t, "tile");
                if i < 0 || (i as usize) >= n {
                    continue;
                }
                let i = i as usize;
                s.known[i] = gb(t, "known");
                if !s.known[i] {
                    continue;
                }
                s.terr[i] = gi(t, "terrain").max(0);
                s.own[i] = gi(t, "owner").max(0);
                s.res[i] = gi(t, "resource").max(0);
                s.stru[i] = gi(t, "structure").max(0);
                s.road[i] = gi(t, "road") != 0;
                s.cap[i] = gi(t, "capital").max(0);
            }
        }
        if let Some(us) = v.get("units").and_then(|x| x.as_array()) {
            for u in us {
                let i = gi(u, "tile");
                if i < 0 || (i as usize) >= n {
                    continue;
                }
                let i = i as usize;
                s.uo[i] = gi(u, "owner");
                s.uk[i] = gi(u, "kind");
                s.uh[i] = gf(u, "health");
                s.uvet[i] = gb(u, "veteran");
                if s.uo[i] == me {
                    s.myunits.push(i);
                } else if s.uo[i] > 0 {
                    s.enemyunits.push(i);
                }
            }
        }
        if let Some(cs) = v.get("cities").and_then(|x| x.as_array()) {
            for c in cs {
                let i = gi(c, "tile");
                if i < 0 || (i as usize) >= n {
                    continue;
                }
                let i = i as usize;
                let o = gi(c, "owner");
                s.cown[i] = o;
                if o == me {
                    s.clev[i] = gi(c, "level").max(1);
                    let rw = c.get("rewards").and_then(|x| x.as_array());
                    if let Some(rw) = rw {
                        if rw.len() == 9 && rw[0].as_i64().unwrap_or(0) > 0 {
                            s.cwall[i] = true;
                        }
                    }
                    s.mycities.push(i);
                    if s.cap[i] == me {
                        s.mycap = Some(i);
                    }
                } else if o > 0 {
                    s.enemycities.push(i);
                } else {
                    s.villages.push(i);
                }
            }
        }
        if let Some(ts) = v.get("technology").and_then(|x| x.as_array()) {
            for t in ts {
                if let Some(k) = t.as_i64() {
                    if k >= 0 && k < 128 {
                        s.tech[k as usize] = true;
                    }
                }
            }
        }
        // power ratio
        let mut mp = 0f64;
        for &i in &s.myunits {
            mp += uatk(s.uk[i]) * s.uh[i] + udef(s.uk[i]) * s.uh[i] * 0.5;
        }
        let mut ep = 0f64;
        for &i in &s.enemyunits {
            ep += uatk(s.uk[i]) * s.uh[i] + udef(s.uk[i]) * s.uh[i] * 0.5;
        }
        s.aggr = if ep <= 0.0 { 1.0 } else { (mp / (mp + ep)).clamp(0.05, 0.95) };
        // naval map? count known water
        let mut w = 0i64;
        let mut kn = 0i64;
        for i in 0..n {
            if s.known[i] {
                kn += 1;
                if s.terr[i] <= 2 {
                    w += 1;
                }
            }
        }
        s.naval = kn > 0 && (w as f64) / (kn as f64) > 0.12;
        s.build_fields();
        Some(s)
    }

    fn build_fields(&mut self) {
        let n = self.n;
        // danger from visible enemy units
        for &e in &self.enemyunits.clone() {
            let k = self.uk[e];
            let r = urng(k) + if udash(k) { umv(k) } else { 0 };
            let pow = 26.0 * uatk(k) + 45.0;
            let (ex, ey) = ((e % self.size) as i64, (e / self.size) as i64);
            for dy in -r..=r {
                for dx in -r..=r {
                    let (x, y) = (ex + dx, ey + dy);
                    if x < 0 || y < 0 || x >= self.size as i64 || y >= self.size as i64 {
                        continue;
                    }
                    self.danger[(y as usize) * self.size + x as usize] += pow;
                }
            }
        }
        // crowding: spread units out
        for &u in &self.myunits.clone() {
            let (x, y) = ((u % self.size) as i64, (u / self.size) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= self.size as i64 || ny >= self.size as i64 {
                        continue;
                    }
                    self.crowd[(ny as usize) * self.size + nx as usize] += 16.0;
                }
            }
        }
        // objectives (tile, weight, decay per tile)
        let mut obj: Vec<(usize, f64, f64)> = Vec::new();
        for i in 0..n {
            if !self.known[i] {
                continue;
            }
            if self.stru[i] == 1 && self.own[i] == 0 {
                obj.push((i, 980.0, 30.0));
            } else if self.stru[i] == 2 {
                obj.push((i, 730.0, 30.0));
            } else if self.stru[i] == 1 && self.own[i] > 0 && self.own[i] != self.me {
                obj.push((i, 120.0 + 880.0 * self.aggr, 24.0));
            } else if self.res[i] == 8 && self.tech[14] {
                obj.push((i, 320.0, 30.0));
            }
        }
        let ec = self.enemycities.len();
        for &c in &self.mycities.clone() {
            let mut mind = 99i64;
            for &e in &self.enemyunits {
                mind = mind.min(cheb(c, e, self.size));
            }
            let iscap = self.cap[c] == self.me;
            obj.push((c, if iscap { 360.0 } else { 320.0 }, 45.0));
            if mind <= 9 && ec > 0 {
                obj.push((c, 1250.0 + (9 - mind) as f64 * 25.0, 105.0));
            } else if mind <= 9 {
                obj.push((c, 700.0, 105.0));
            }
        }
        // intercept enemies that are close to home or on our land
        for &e in &self.enemyunits.clone() {
            let mut mind = 99i64;
            for &c in &self.mycities {
                mind = mind.min(cheb(c, e, self.size));
            }
            let w = if self.own[e] == self.me {
                900.0
            } else if mind <= 6 {
                760.0
            } else {
                300.0 + 300.0 * self.aggr
            };
            obj.push((e, w, 80.0));
        }
        let mut pot = vec![NEG; n];
        for i in 0..n {
            let mut best = NEG;
            for &(o, w, dc) in &obj {
                let d = cheb(i, o, self.size) as f64;
                let val = w - dc * d;
                if val > best {
                    best = val;
                }
            }
            pot[i] = if best == NEG { 0.0 } else { best };
        }
        self.pot = pot;
    }

    fn revealed(&self, t: usize, k: i64) -> f64 {
        let rad = if self.terr[t] == 4 || uscout(k) { 2 } else { 1 };
        let (x, y) = ((t % self.size) as i64, (t / self.size) as i64);
        let mut c = 0f64;
        for dy in -rad..=rad {
            for dx in -rad..=rad {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= self.size as i64 || ny >= self.size as i64 {
                    continue;
                }
                if !self.known[(ny as usize) * self.size + nx as usize] {
                    c += 1.0;
                }
            }
        }
        c
    }

    fn defbonus(&self, t: usize, enemy: bool) -> f64 {
        let techok = if enemy { true } else { false };
        let ter = self.terr[t];
        let mut b = 1.0f64;
        if ter == 5 {
            b = if enemy || self.tech[19] { 1.5 } else { 1.0 };
        } else if ter == 4 {
            b = if enemy || self.tech[21] { 1.5 } else { 1.0 };
        } else if ter <= 2 {
            b = if enemy || self.tech[12] { 1.5 } else { 1.0 };
        } else if self.cown[t] != 0 && self.cown[t] == self.uo[t] && ufort(self.uk[t]) {
            b = if self.cwall[t] && !enemy { 4.0 } else if enemy && techok { 1.5 } else { 1.5 };
        }
        b
    }

    fn maxhp(&self, t: usize) -> f64 {
        uhp(self.uk[t]) + if self.uvet[t] { 5.0 } else { 0.0 }
    }

    // damage from attacker tile a to defender tile d (defender bonus precomputed)
    fn combat(&self, a: usize, d: usize, bonus: f64) -> (f64, f64) {
        let ka = self.uk[a];
        let kd = self.uk[d];
        let aa = uatk(ka);
        let dd = udef(kd);
        let af = aa * (self.uh[a] / self.maxhp(a).max(1.0));
        let df = dd * (self.uh[d] / self.maxhp(d).max(1.0)) * bonus;
        let tot = af + df;
        if tot <= 0.0 {
            return (0.0, 0.0);
        }
        let dmg = ((af / tot) * aa * 4.5 + 1e-6 + 0.5).floor();
        let mut ret = 0.0;
        if self.uh[d] > dmg {
            let can = !ustiff(kd) && cheb(a, d, self.size) <= urng(kd) && self.known[a];
            if can {
                ret = ((df / tot) * dd * 4.5 + 1e-6 + 0.5).floor();
            }
        }
        (dmg, ret)
    }

    fn connected(&self, extra: i64) -> i64 {
        let n = self.n;
        let cap = match self.mycap {
            Some(c) => c,
            None => return 0,
        };
        let mut nodes = vec![false; n];
        for i in 0..n {
            let friendly = self.own[i] == 0 || self.own[i] == self.me;
            let isnode = self.road[i] || (extra >= 0 && i as i64 == extra) || self.stru[i] == 48 || self.stru[i] == 8
                || (self.cown[i] > 0 && friendly);
            nodes[i] = isnode && friendly;
        }
        let mut seen = vec![false; n];
        let mut stack = vec![cap];
        seen[cap] = true;
        while let Some(x) = stack.pop() {
            let (px, py) = ((x % self.size) as i64, (x / self.size) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (px + dx, py + dy);
                    if nx < 0 || ny < 0 || nx >= self.size as i64 || ny >= self.size as i64 {
                        continue;
                    }
                    let j = (ny as usize) * self.size + nx as usize;
                    if !seen[j] && nodes[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            if self.stru[x] == 8 {
                for j in 0..n {
                    if !seen[j] && self.stru[j] == 8 && cheb(x, j, self.size) <= 4 {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        let mut c = 0;
        for &city in &self.mycities {
            if city != cap && seen[city] {
                c += 1;
            }
        }
        c
    }

    fn reward_pref(&self, city: usize, param: i64) -> f64 {
        let count: i64 = if city < self.n { self.clev[city] } else { 1 };
        let _ = count;
        match param {
            3 => 100.0, // workshop: +1 production
            4 => 60.0,  // explorer
            1 => {
                let threat = self.enemyunits.iter().any(|&e| cheb(e, city, self.size) <= 13) || self.turn > 4;
                if threat { 112.0 } else { 78.0 }
            }
            7 => 70.0, // +5 stars
            8 => 100.0, // +3 population
            5 => 70.0,  // border growth
            6 => {
                if self.aggr > 0.4 || !self.enemycities.is_empty() { 100.0 } else { 80.0 }
            }
            2 => 85.0, // park
            _ => 10.0,
        }
    }

    fn tech_pri(&self, k: i64) -> f64 {
        let naval = self.naval as i64 as f64;
        match k {
            6 => 100.0,
            16 => 97.0,
            20 => 93.0,
            23 => 92.0,
            10 => 80.0 + 12.0 * naval,
            24 => 88.0,
            17 => 86.0,
            18 => 83.0,
            1 => 80.0,
            3 => 76.0,
            7 => 72.0,
            4 => 70.0,
            5 => 68.0,
            13 => 40.0 + 30.0 * naval,
            44 => 35.0 + 28.0 * naval,
            9 => 50.0,
            2 => 45.0,
            22 => 45.0,
            21 => 40.0,
            8 => 35.0,
            14 => 30.0 + 10.0 * naval,
            19 => 30.0,
            12 => 25.0 + 10.0 * naval,
            48 => 20.0,
            25 => 20.0,
            38 => 5.0,
            _ => 12.0,
        }
    }
}

static mut LAST_TURN: i64 = -1;
static mut SEEN: Vec<i64> = Vec::new();
static mut STUCK: Vec<i64> = Vec::new();

fn note_turn(s: &St) {
    unsafe {
        if SEEN.len() != s.n {
            SEEN = vec![-10i64; s.n];
            STUCK = vec![0i64; s.n];
            LAST_TURN = -1;
        }
        if s.turn != LAST_TURN {
            let fresh = s.turn > LAST_TURN;
            LAST_TURN = s.turn;
            if fresh {
                for &u in &s.myunits {
                    if SEEN[u] == s.turn - 1 {
                        STUCK[u] += 1;
                    } else {
                        STUCK[u] = 0;
                    }
                    SEEN[u] = s.turn;
                }
            }
        }
    }
}

fn stuck_bonus(s: &St, a: usize, d: usize) -> f64 {
    unsafe {
        if a >= s.n || d >= s.n || STUCK.len() != s.n {
            return 0.0;
        }
        let k = STUCK[a].max(0);
        if k < 1 {
            return 0.0;
        }
        let jit = ((s.turn * 31 + d as i64 * 7 + a as i64 * 13) % 23) as f64;
        55.0 * (k.min(3) as f64) + jit * 4.0
    }
}

fn score(s: &St, c: &Value) -> f64 {
    let kind = gi(c, "kind");
    let src = gi(c, "source");
    let dst = gi(c, "target");
    let par = gi(c, "parameter");
    let n = s.n as i64;
    let sok = src >= 0 && src < n;
    let dok = dst >= 0 && dst < n;
    match kind {
        11 => -1e9,
        10 => 0.0,
        9 => {
            if !dok {
                return 1e7;
            }
            1e7 + s.reward_pref(dst as usize, par)
        }
        8 => {
            if !sok {
                return -1e5;
            }
            let t = src as usize;
            if s.stru[t] == 2 {
                return 8e6;
            }
            if s.cown[t] > 0 && s.cown[t] != s.me {
                // enemy city: winning move if it is their last
                let last = s.enemycities.len() <= 1;
                return if last { 9.5e6 } else { 9e6 };
            }
            if s.stru[t] == 1 && s.own[t] == 0 {
                return 8.5e6;
            }
            if s.res[t] == 8 {
                return 1e5;
            }
            1e5
        }
        3 => {
            match par {
                16 => 5e6,
                7 => {
                    let u = (if sok { src } else { dst }) as usize;
                    if u >= s.n || s.uo[u] != s.me {
                        -1e5
                    } else {
                        let miss = s.maxhp(u) - s.uh[u];
                        if miss <= 0.5 {
                            -1e5
                        } else {
                            300.0 + 28.0 * miss - s.danger[u] * 0.5
                        }
                    }
                }
                9 => 500.0,
                1 => {
                    if s.tech[8] {
                        520.0
                    } else {
                        150.0
                    }
                }
                _ => -1e5,
            }
        }
        2 => {
            if !sok || !dok {
                return -1e5;
            }
            let (a, d) = (src as usize, dst as usize);
            if s.uo[a] != s.me || s.uo[d] == s.me || s.uo[d] == 0 {
                return -1e5;
            }
            let bonus = s.defbonus(d, true);
            let (dmg, ret) = s.combat(a, d, bonus);
            let ehp = s.uh[d];
            let mhp = s.uh[a];
            let kill = dmg >= ehp;
            let mut sc = 1e6;
            sc += 16.0 * dmg.min(ehp);
            if kill {
                sc += 300.0 + 10.0 * ucost(s.uk[d]);
            }
            sc -= 12.0 * ret;
            if ret >= mhp {
                sc -= 500.0;
            }
            if s.cown[d] > 0 && s.cown[d] != s.me {
                sc += 250.0;
            }
            if s.own[d] == s.me || s.cown[d] == s.me {
                sc += 260.0; // evict invaders from our land
            }
            // defend our cities
            for &c in &s.mycities {
                if cheb(c, d, s.size) <= 4 {
                    sc += 200.0;
                    break;
                }
            }
            sc
        }
        5 => {
            if !dok {
                return -1e5;
            }
            900.0
        }
        6 => {
            if !dok {
                return -1e5;
            }
            let t = dst as usize;
            match par {
                23..=29 => 4e6, // monuments are free and give +3 population
                21 => 870.0,    // mine +2 pop
                5 => 860.0,     // farm +2 pop
                22 => 850.0,    // forge
                12 => 820.0,    // lumber hut
                13 => 800.0,    // sawmill
                32 => 790.0,    // sanctuary +1 star/turn
                50 => {
                    let mut lev = 0i64;
                    let (x, y) = ((t % s.size) as i64, (t / s.size) as i64);
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let (nx, ny) = (x + dx, y + dy);
                            if nx < 0 || ny < 0 || nx >= s.size as i64 || ny >= s.size as i64 {
                                continue;
                            }
                            let j = (ny as usize) * s.size + nx as usize;
                            if s.own[j] == s.me && (s.stru[j] == 13 || s.stru[j] == 22 || s.stru[j] == 6) {
                                lev += 1;
                            }
                        }
                    }
                    600.0 + 90.0 * lev as f64
                }
                8 => {
                    if s.naval { 780.0 } else { 620.0 }
                }
                3 | 48 => {
                    let before = s.connected(-1);
                    let after = s.connected(dst);
                    if after > before {
                        1500.0 + 200.0 * (after - before) as f64
                    } else {
                        // adjacent to own network?
                        let (x, y) = ((t % s.size) as i64, (t / s.size) as i64);
                        let mut near = false;
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let (nx, ny) = (x + dx, y + dy);
                                if nx < 0 || ny < 0 || nx >= s.size as i64 || ny >= s.size as i64 {
                                    continue;
                                }
                                let j = (ny as usize) * s.size + nx as usize;
                                if s.road[j] || s.cown[j] == s.me || s.stru[j] == 8 {
                                    near = true;
                                }
                            }
                        }
                        if near { 330.0 } else { 40.0 }
                    }
                }
                17 | 18 | 19 | 20 => 60.0,
                _ => 100.0,
            }
        }
        7 => {
            let pri = s.tech_pri(par);
            let cities = s.mycities.len().max(1) as f64;
            let tier = if par >= 0 { 1.0 } else { 1.0 };
            let _ = tier;
            let cost = 4.0 + cities * 2.0;
            300.0 + 4.0 * pri - 2.0 * cost
        }
        4 => {
            if !sok {
                return -1e5;
            }
            let city = src as usize;
            let base = match par {
                12 => 1050.0,
                11 => 760.0,
                4 => 740.0,
                5 => 730.0,
                8 => 700.0,
                9 => 700.0,
                3 => 660.0,
                44 => 700.0,
                42 => 620.0,
                41 => 620.0,
                2 => 600.0,
                40 => 500.0,
                43 => 120.0,
                _ => 300.0,
            };
            let mut sc = base;
            if !s.naval && ufloat(par) {
                sc -= 500.0;
            }
            if s.naval && !ufloat(par) && par != 12 && s.mycities.iter().all(|&c| c != city) {
                sc -= 0.0;
            }
            // defensive need
            let mut mind = 99i64;
            for &e in &s.enemyunits {
                mind = mind.min(cheb(e, city, s.size));
            }
            let guard = s.myunits.iter().any(|&u| cheb(u, city, s.size) <= 1);
            if mind <= 8 && !guard {
                sc += 900.0;
                if ufort(par) || udef(par) >= 3.0 {
                    sc += 200.0;
                }
            } else if !guard {
                sc += 240.0;
                if ufort(par) {
                    sc += 90.0;
                }
            }
            let army = s.myunits.len() as f64;
            let enemy = s.enemyunits.len() as f64;
            let want = 3.0 + 3.0 * s.mycities.len() as f64;
            if army < want {
                sc += 470.0 + (want - army) * 110.0;
            }
            if enemy > army {
                sc += 150.0 * (enemy - army).min(5.0);
            }
            if mind <= 7 && (ufort(par) || udef(par) >= 3.0) {
                sc += 170.0;
            }
            if par == 5 && !guard {
                sc += 150.0;
            }
            if s.enemycities.is_empty() && s.villages.is_empty() && par == 12 {
                sc -= 300.0;
            }
            // keep some stars for economy
            if s.stars < 8 && (par == 12 || par == 4 || par == 8) {
                sc -= 150.0;
            }
            sc
        }
        1 => {
            if !sok || !dok {
                return -1e5;
            }
            let (a, d) = (src as usize, dst as usize);
            if s.uo[a] != s.me {
                return -1e5;
            }
            let k = s.uk[a];
            let revd = s.revealed(d, k);
            let revs = s.revealed(a, k);
            let mut sc = 26.0 + (s.pot[d] - s.pot[a]) + 9.0 * (revd - revs).min(5.0);
            if s.stru[d] == 1 && s.own[d] == 0 {
                sc += 950.0; // step onto a neutral village: capture next action
            } else if s.stru[d] == 2 {
                sc += 700.0; // ruin
            } else if s.cown[d] > 0 && s.cown[d] != s.me {
                sc += 420.0 * s.aggr; // siege position
            } else if s.res[d] == 8 && s.tech[14] {
                sc += 260.0;
            }
            let ter = s.terr[d];
            if ter == 5 {
                sc += 10.0;
            }
            if ter == 4 {
                sc += 16.0;
            }
            if s.cown[d] == s.me && ufort(k) {
                sc += 26.0;
            }
            sc -= s.crowd[d];
            let outn = s.enemyunits.len() as f64;
            let mine = s.myunits.len().max(1) as f64;
            let weak = (if uatk(k) * s.uh[a] < 22.0 { 1.0 } else { 0.4 }) * if outn > mine * 1.4 { 1.8 } else { 1.0 };
            sc -= s.danger[d] * weak;
            sc += (s.danger[a] - s.danger[d]) * 0.55 * weak;
            // heal-damaged units fall back toward home
            if s.uh[a] < s.maxhp(a) * 0.6 {
                sc += (s.danger[a] - s.danger[d]) * 0.5;
                for &c in &s.mycities {
                    let dc = cheb(d, c, s.size);
                    let ac = cheb(a, c, s.size);
                    sc += 26.0 * (ac - dc) as f64;
                }
            }
            if !s.naval && ufloat(k) {
                sc -= 400.0;
            }
            if s.naval && !ufloat(k) && ter <= 2 {
                sc -= 250.0;
            }
            // embark to cross water when the enemy is across the sea
            if s.naval && !ufloat(k) && s.stru[d] == 8 && s.own[d] == s.me && !s.enemycities.is_empty() && uatk(k) > 0.0 {
                sc += 150.0;
            }
            sc += stuck_bonus(s, a, d);
            sc
        }
        _ => -1e5,
    }
}

fn decide(obs: &Value) -> Option<String> {
    let legal = obs.get("legal_actions")?.as_array()?;
    if legal.is_empty() {
        return None;
    }
    let fallback = || -> String {
        for a in legal {
            let c = &a["command"];
            if gi(c, "kind") == 10 {
                return a.get("action_id").and_then(|x| x.as_str()).unwrap_or("0").to_string();
            }
        }
        legal[0].get("action_id").and_then(|x| x.as_str()).unwrap_or("0").to_string()
    };
    let st = match St::parse(obs) {
        Some(s) => s,
        None => return Some(fallback()),
    };
    note_turn(&st);
    let mut best = f64::NEG_INFINITY;
    let mut bestid: Option<String> = None;
    for a in legal {
        let c = match a.get("command") {
            Some(c) => c,
            None => continue,
        };
        let id = match a.get("action_id").and_then(|x| x.as_str()) {
            Some(i) => i,
            None => continue,
        };
        let sc = score(&st, c);
        if sc > best {
            best = sc;
            bestid = Some(id.to_string());
        }
    }
    Some(bestid.unwrap_or_else(fallback))
}

fn raw_fallback(line: &str) -> String {
    if let Some(p) = line.find("\"action_id\"") {
        let rest = &line[p..];
        if let Some(q1) = rest.find('"') {
            let r2 = &rest[q1 + 1..];
            if let Some(q2) = r2.find('"') {
                return r2[..q2].to_string();
            }
        }
    }
    "0".to_string()
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
        let id = match serde_json::from_str::<Value>(&line) {
            Ok(obs) => match decide(&obs) {
                Some(i) => i,
                None => raw_fallback(&line),
            },
            Err(_) => raw_fallback(&line),
        };
        let _ = writeln!(stdout, "{{\"action_id\":\"{}\"}}", id);
        let _ = stdout.flush();
    }
}
