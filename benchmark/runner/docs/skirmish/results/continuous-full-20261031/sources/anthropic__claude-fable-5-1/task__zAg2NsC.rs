use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};

#[derive(Clone, Default)]
struct Unit { tile: usize, owner: i64, kind: i64, hp: f64, vet: bool, moved: bool, attacked: bool, home: i64 }

struct G {
    size: usize, n: usize, me: i64, enemy: i64, turn: i64, stars: i64,
    known: Vec<bool>, terrain: Vec<i64>, owner: Vec<i64>, structure: Vec<i64>, resource: Vec<i64>, capital: Vec<i64>,
    unit_at: Vec<Option<Unit>>, tech: Vec<bool>, cities: Vec<(usize, i64, i64)>, // tile, level, unit count n/a
}

fn stats(kind: i64) -> (f64, f64, f64, i64, bool) { // atk, def, hp, range, dash
    match kind {
        2 => (2.0, 2.0, 10.0, 1, true), 3 => (2.0, 1.0, 10.0, 1, true), 4 => (3.5, 1.0, 10.0, 1, true),
        5 => (1.0, 3.0, 15.0, 1, false), 8 => (4.0, 0.0, 10.0, 3, false), 9 => (2.0, 1.0, 10.0, 2, true),
        10 => (0.0, 1.0, 10.0, 1, false), 11 => (3.0, 3.0, 15.0, 1, true), 12 => (5.0, 4.0, 40.0, 1, false),
        37 => (2.0, 2.0, 10.0, 1, true), 38 => (0.0, 0.5, 5.0, 1, true), 39 => (0.0, 0.5, 5.0, 1, true),
        40 => (2.0, 2.0, 10.0, 1, true), 41 => (3.0, 2.0, 10.0, 3, false), 42 => (2.0, 1.0, 10.0, 2, true),
        43 => (0.0, 1.0, 10.0, 2, false), 44 => (3.0, 3.0, 10.0, 1, true), 45 => (4.0, 4.0, 40.0, 1, false),
        _ => (2.0, 2.0, 10.0, 1, true),
    }
}
fn tier(k: i64) -> i64 { match k { 1|6|10|15|20 => 1, 2|4|7|8|13|16|18|21|23|44 => 2, _ => 3 } }
fn is_boat(kind: i64) -> bool { matches!(kind, 39 | 40 | 41 | 42 | 43 | 44 | 45) }
fn cheb(size: usize, a: usize, b: usize) -> i64 {
    let (ax, ay) = ((a % size) as i64, (a / size) as i64);
    let (bx, by) = ((b % size) as i64, (b / size) as i64);
    (ax - bx).abs().max((ay - by).abs())
}

impl G {
    fn parse(obs: &Value) -> G {
        let v = &obs["visible_state"];
        let size = v["board_size"].as_u64().unwrap() as usize;
        let n = size * size;
        let me = obs["player"].as_i64().unwrap();
        let mut g = G { size, n, me, enemy: 3 - me, turn: v["turn"].as_i64().unwrap_or(0), stars: v["stars"].as_i64().unwrap_or(0),
            known: vec![false; n], terrain: vec![0; n], owner: vec![0; n], structure: vec![0; n], resource: vec![0; n], capital: vec![0; n],
            unit_at: vec![None; n], tech: vec![false; 128], cities: vec![] };
        for t in v["tiles"].as_array().unwrap() {
            let i = t["tile"].as_u64().unwrap() as usize;
            if t["known"].as_bool().unwrap_or(false) {
                g.known[i] = true;
                g.terrain[i] = t["terrain"].as_i64().unwrap_or(0);
                g.owner[i] = t["owner"].as_i64().unwrap_or(0);
                g.structure[i] = t["structure"].as_i64().unwrap_or(0);
                g.resource[i] = t["resource"].as_i64().unwrap_or(0);
                g.capital[i] = t["capital"].as_i64().unwrap_or(0);
            }
        }
        for u in v["units"].as_array().unwrap() {
            let i = u["tile"].as_u64().unwrap() as usize;
            g.unit_at[i] = Some(Unit { tile: i, owner: u["owner"].as_i64().unwrap_or(0), kind: u["kind"].as_i64().unwrap_or(2),
                hp: u["health"].as_f64().unwrap_or(10.0), vet: u["veteran"].as_bool().unwrap_or(false),
                moved: u["moved"].as_bool().unwrap_or(false), attacked: u["attacked"].as_bool().unwrap_or(false), home: u["home"].as_i64().unwrap_or(0) });
        }
        for t in v["technology"].as_array().unwrap() { if let Some(k) = t.as_u64() { if k < 128 { g.tech[k as usize] = true; } } }
        for c in v["cities"].as_array().unwrap() {
            if c["owner"].as_i64() == Some(me) {
                g.cities.push((c["tile"].as_u64().unwrap() as usize, c["level"].as_i64().unwrap_or(1), 0));
            }
        }
        g
    }
    fn neighbors(&self, i: usize) -> Vec<usize> {
        let (x, y) = ((i % self.size) as i64, (i / self.size) as i64);
        let mut v = Vec::with_capacity(8);
        for dy in -1..=1 { for dx in -1..=1 { if dx == 0 && dy == 0 { continue; }
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && (nx as usize) < self.size && (ny as usize) < self.size { v.push(ny as usize * self.size + nx as usize); }
        } }
        v
    }
    fn passable(&self, i: usize, boat: bool) -> bool {
        if let Some(u) = &self.unit_at[i] { if u.owner != self.me { return false; } }
        if !self.known[i] { return true; }
        if boat { return true; }
        match self.terrain[i] {
            3 | 5 => true,
            4 => self.tech[20],
            1 | 2 => self.structure[i] == 8 && self.owner[i] == self.me || self.structure[i] == 48,
            _ => false,
        }
    }
    fn bfs(&self, from: usize, boat: bool) -> Vec<i64> {
        let mut d = vec![i64::MAX; self.n];
        let mut q = VecDeque::new();
        d[from] = 0; q.push_back(from);
        while let Some(c) = q.pop_front() {
            for nb in self.neighbors(c) {
                if d[nb] == i64::MAX && self.passable(nb, boat) {
                    d[nb] = d[c] + 1; q.push_back(nb);
                }
            }
        }
        d
    }
    fn is_enemy_city(&self, i: usize) -> bool { self.known[i] && self.structure[i] == 1 && self.owner[i] == self.enemy }
    fn enemy_units(&self) -> Vec<&Unit> { self.unit_at.iter().flatten().filter(|u| u.owner != self.me).collect() }
    fn my_units(&self) -> Vec<&Unit> { self.unit_at.iter().flatten().filter(|u| u.owner == self.me).collect() }
    fn enemy_near(&self, tile: usize, r: i64) -> usize {
        self.enemy_units().iter().filter(|u| cheb(self.size, u.tile, tile) <= r).count()
    }
    fn def_bonus(&self, u: &Unit) -> f64 {
        let t = u.tile;
        let own_tech = u.owner == self.me;
        let bonus = match self.terrain[t] {
            5 => if own_tech { if self.tech[18] { 1.5 } else { 1.0 } } else { 1.3 },
            4 => if own_tech { if self.tech[20] { 1.5 } else { 1.0 } } else { 1.4 },
            1 | 2 => 1.0,
            _ => if self.structure[t] == 1 && self.owner[t] == u.owner && !matches!(u.kind, 11 | 12 | 8) { 1.5 } else { 1.0 },
        };
        bonus
    }
    // returns (damage to defender, retaliation to attacker, defender dies, attacker dies)
    fn combat(&self, a: &Unit, d: &Unit) -> (f64, f64, bool, bool) {
        let (aa, _, ahm, _, _) = stats(a.kind);
        let (_, dd, dhm, drange, _) = stats(d.kind);
        let ahm = ahm + if a.vet { 5.0 } else { 0.0 };
        let dhm = dhm + if d.vet { 5.0 } else { 0.0 };
        let af = aa * a.hp / ahm;
        let df = dd * d.hp / dhm * self.def_bonus(d);
        let tot = af + df;
        if tot <= 0.0 { return (0.0, 0.0, false, false); }
        let dmg = (af / tot * aa * 4.5 + 0.5).floor();
        let dies = d.hp - dmg <= 0.0;
        let dist = cheb(self.size, a.tile, d.tile);
        let stiff = matches!(d.kind, 8 | 10 | 38 | 39 | 41 | 43 | 45);
        let ret = if dies || stiff || dist > drange { 0.0 } else { (df / tot * dd * 4.5 + 0.5).floor() };
        (dmg, ret, dies, a.hp - ret <= 0.0)
    }
}

struct Bot { last_turn: i64, end_count: i64 }

fn pick(legal: &[Value], f: impl Fn(&Value) -> bool) -> Option<String> {
    legal.iter().find(|a| f(&a["command"])).map(|a| a["action_id"].as_str().unwrap().to_owned())
}

impl Bot {
    fn choose(&mut self, obs: &Value) -> String {
        let legal = obs["legal_actions"].as_array().expect("legal_actions");
        if legal.is_empty() { return "0".into(); }
        let g = G::parse(obs);
        let end_id = pick(legal, |c| c["kind"] == 10);
        if g.turn != self.last_turn { self.last_turn = g.turn; self.end_count = 0; }

        // 1. City reward
        if let Some(a) = legal.iter().filter(|a| a["command"]["kind"] == 9).max_by_key(|a| {
            match a["command"]["parameter"].as_i64().unwrap_or(0) { 3 => 10, 4 => 5, 7 => 10, 1 => 4, 8 => 10, 5 => 6, 6 => 10, 2 => 3, _ => 0 }
        }) { return a["action_id"].as_str().unwrap().to_owned(); }
        // 2. Capture
        if let Some(id) = pick(legal, |c| c["kind"] == 8) { return id; }
        // 3. Promote
        if let Some(id) = pick(legal, |c| c["kind"] == 3 && c["parameter"] == 16) { return id; }
        // 4. Attacks
        let mut best_att: Option<(f64, String)> = None;
        for a in legal.iter().filter(|a| a["command"]["kind"] == 2) {
            let s = a["command"]["source"].as_u64().unwrap() as usize;
            let t = a["command"]["target"].as_u64().unwrap() as usize;
            let (au, du) = match (&g.unit_at[s], &g.unit_at[t]) { (Some(a), Some(d)) => (a, d), _ => continue };
            if au.kind == 10 { continue; }
            let (dmg, ret, dies, adies) = g.combat(au, du);
            let mut score = dmg * 1.0 - ret * 0.8;
            if dies { score += 12.0 + stats(du.kind).2 * 0.3; }
            if adies { score -= 25.0; }
            // enemy near our city: more aggressive
            if g.cities.iter().any(|c| cheb(g.size, c.0, t) <= 2) { score += 4.0; }
            if g.cities.iter().any(|c| c.0 == t) { score += 15.0; }
            if g.is_enemy_city(t) { score += 3.0; }
            if score > 0.5 && best_att.as_ref().map_or(true, |b| score > b.0) {
                best_att = Some((score, a["action_id"].as_str().unwrap().to_owned()));
            }
        }
        if let Some((_, id)) = best_att { return id; }
        // heal others (mind bender)
        if let Some(id) = pick(legal, |c| c["kind"] == 3 && c["parameter"] == 9) { return id; }

        let my_units = g.my_units();
        let enemy_units = g.enemy_units();
        let n_cities = g.cities.len().max(1) as i64;
        // threatened cities: enemy within 3 and no own unit on city
        let threatened: Vec<usize> = g.cities.iter().map(|c| c.0)
            .filter(|&t| g.enemy_near(t, 3) > 0 && g.unit_at[t].as_ref().map_or(true, |u| u.owner != g.me)).collect();

        let target_units = n_cities * 3 + 3 + (g.turn / 5).min(10) + if enemy_units.is_empty() { 0 } else { enemy_units.len() as i64 / 2 };
        let want_units = (my_units.len() as i64) < target_units || (!enemy_units.is_empty() && (my_units.len() as i64) < enemy_units.len() as i64 + 2);
        // 5. Urgent defense summon
        let n_defenders = my_units.iter().filter(|u| u.kind == 5).count() as i64;
        let summon_choice = |city: usize, defensive: bool| -> Option<String> {
            let defensive = defensive && n_defenders < n_cities + 1;
            let order: Vec<i64> = if defensive { vec![5, 11, 2, 9, 3] } else if g.turn < 20 { vec![3, 2] } else if g.turn % 3 == 0 { vec![9, 11, 3, 2] } else { vec![4, 11, 3, 9, 2] };
            for k in order {
                if let Some(id) = pick(legal, |c| c["kind"] == 4 && c["source"].as_u64() == Some(city as u64) && c["parameter"] == k) { return Some(id); }
            }
            pick(legal, |c| c["kind"] == 4 && c["source"].as_u64() == Some(city as u64) && c["parameter"] == 2)
        };
        for &t in &threatened { if let Some(id) = summon_choice(t, true) { return id; } }

        // 6. Harvest (kind 5)
        if let Some(id) = pick(legal, |c| c["kind"] == 5) { return id; }
        // 7. Builds: pop per star
        let water_frac = {
            let kn = g.known.iter().filter(|&&k| k).count().max(1) as f64;
            let w = (0..g.n).filter(|&i| g.known[i] && (g.terrain[i] == 1 || g.terrain[i] == 2)).count() as f64;
            w / kn
        };
        let land_reach = { // can we reach enemy city or neutral village or frontier by land from capital?
            let cap = g.cities.iter().map(|c| c.0).next().unwrap_or(0);
            let d = g.bfs(cap, false);
            let enemy_city_known = (0..g.n).any(|i| g.is_enemy_city(i));
            if enemy_city_known {
                (0..g.n).any(|i| d[i] < i64::MAX && g.is_enemy_city(i))
            } else {
                (0..g.n).any(|i| d[i] < i64::MAX && ((g.known[i] && g.structure[i] == 1 && g.owner[i] != g.me) || (!g.known[i] && g.neighbors(i).iter().any(|&j| g.known[j] && d[j] < i64::MAX && g.terrain[j] >= 3))))
            }
        };
        let need_navy = !land_reach && g.tech[10];
        let has_port = (0..g.n).any(|i| g.known[i] && g.structure[i] == 8 && g.owner[i] == g.me);
        {
            let mut best: Option<(f64, String)> = None;
            for a in legal.iter().filter(|a| a["command"]["kind"] == 6) {
                let k = a["command"]["parameter"].as_i64().unwrap_or(0);
                let t = a["command"]["target"].as_u64().unwrap_or(0) as usize;
                let adj_count = |kind: i64| g.neighbors(t).iter().filter(|&&j| g.known[j] && g.structure[j] == kind && g.owner[j] == g.me).count() as f64;
                let (pop, cost) = match k {
                    12 => (1.0, 3.0), 5 => (2.0, 5.0), 21 => (2.0, 5.0),
                    13 => (adj_count(12), 5.0), 6 => (adj_count(5), 5.0), 22 => (2.0 * adj_count(21), 5.0),
                    8 => (if need_navy && !has_port { 6.0 } else if water_frac > 0.25 { 1.2 } else { 0.6 }, 7.0),
                    17 | 18 | 19 | 20 => (0.6, 15.0),
                    _ => (0.0, 1.0),
                };
                if pop <= 0.0 { continue; }
                let v = pop / cost;
                let thr = if g.stars >= 25 { 0.03 } else if g.stars >= 12 { 0.15 } else { 0.3 };
                if v < thr && !(k == 8 && need_navy) { continue; }
                if best.as_ref().map_or(true, |b| v > b.0) { best = Some((v, a["action_id"].as_str().unwrap().to_owned())); }
            }
            if let Some((_, id)) = best { return id; }
        }
        // 9. Summon
        {
            let want = want_units;
            if want {
                // choose city: prefer threatened, then nearest to enemy
                let mut cities: Vec<usize> = g.cities.iter().map(|c| c.0).filter(|&t| g.unit_at[t].is_none()).collect();
                cities.sort_by_key(|&t| enemy_units.iter().map(|u| cheb(g.size, u.tile, t)).min().unwrap_or(0));
                for t in cities {
                    let defensive = g.enemy_near(t, 3) > 0;
                    if let Some(id) = summon_choice(t, defensive) { return id; }
                }
            }
        }
        // 8. Research
        {
            let mut prio: Vec<i64> = vec![1, 6, 7, 16, 18, 8, 20, 23, 17, 24, 9, 3, 4, 21, 22, 5, 2, 38];
            if water_frac > 0.3 || !land_reach { prio.insert(1, 10); prio.insert(3, 13); } else if water_frac > 0.15 { prio.insert(4, 10); }
            let deficit = (my_units.len() as i64) < n_cities;
            for k in prio {
                if g.tech[k as usize] { continue; }
                if let Some(a) = legal.iter().find(|a| a["command"]["kind"] == 7 && a["command"]["parameter"] == k) {
                    let cost = 4 + n_cities * tier(k);
                    let no_empty_city = g.cities.iter().all(|c| g.unit_at[c.0].is_some());
                    if (!deficit && g.stars >= cost + 2) || g.stars >= cost + 6 || (no_empty_city && g.stars >= cost) { return a["action_id"].as_str().unwrap().to_owned(); }
                    break; // affordable later: save for it
                }
            }
        }
        // 10. Moves
        {
            let mut goals: Vec<(usize, f64)> = Vec::new(); // tile, value
            let my_strength: f64 = my_units.iter().map(|u| u.hp * stats(u.kind).0).sum();
            let en_strength: f64 = enemy_units.iter().map(|u| u.hp * stats(u.kind).0).sum::<f64>();
            let aggressive = my_strength > en_strength * 0.9 && (g.turn > 12 || enemy_units.len() < 2);
            for i in 0..g.n {
                if !g.known[i] {
                    // frontier: unknown tile adjacent to known; value by how much it reveals and ring distance from cities
                    let nbs = g.neighbors(i);
                    if nbs.iter().any(|&j| g.known[j]) {
                        let (x, y) = ((i % g.size) as i64, (i / g.size) as i64);
                        let mut unk = 0.0;
                        for dy in -2..=2i64 { for dx in -2..=2i64 { let (nx, ny) = (x + dx, y + dy);
                            if nx >= 0 && ny >= 0 && (nx as usize) < g.size && (ny as usize) < g.size && !g.known[ny as usize * g.size + nx as usize] { unk += 1.0; } } }
                        let dc = g.cities.iter().map(|c| cheb(g.size, c.0, i)).min().unwrap_or(10);
                        let mut v = 4.0 + 1.0 * unk;
                        if g.turn < 20 && dc >= 3 && dc <= 5 { v += 8.0; }
                        goals.push((i, v));
                    }
                    continue;
                }
                if g.structure[i] == 1 && g.owner[i] == 0 {
                    let occupied = g.unit_at[i].as_ref().map_or(false, |u| u.owner == g.me);
                    if !occupied { goals.push((i, 100.0)); }
                } else if g.structure[i] == 2 {
                    if g.unit_at[i].as_ref().map_or(true, |u| u.owner != g.me) { goals.push((i, 60.0)); }
                } else if g.is_enemy_city(i) {
                    let defended = g.unit_at[i].as_ref().map_or(false, |u| u.owner != g.me);
                    let v = if aggressive { if g.capital[i] == g.enemy { 95.0 } else { 90.0 } } else { 25.0 };
                    goals.push((i, if defended { v * 0.6 } else { v }));
                }
                if need_navy && g.structure[i] == 8 && g.owner[i] == g.me { goals.push((i, 70.0)); }
            }
            // claimed villages by moved units standing adjacent/on
            let mut best: Option<(f64, String)> = None;
            let mut units_moves: std::collections::HashMap<usize, Vec<(usize, String)>> = std::collections::HashMap::new();
            for a in legal.iter().filter(|a| a["command"]["kind"] == 1) {
                let s = a["command"]["source"].as_u64().unwrap() as usize;
                let t = a["command"]["target"].as_u64().unwrap() as usize;
                units_moves.entry(s).or_default().push((t, a["action_id"].as_str().unwrap().to_owned()));
            }
            for (&s, moves) in &units_moves {
                let u = match &g.unit_at[s] { Some(u) => u.clone(), None => continue };
                let boat = is_boat(u.kind);
                let (uatk, _, uhm, urange, dash) = stats(u.kind);
                let uhm = uhm + if u.vet { 5.0 } else { 0.0 };
                let on_city = g.structure[s] == 1 && g.owner[s] == g.me;
                let eval = |dest: usize| -> f64 {
                    let d = g.bfs(dest, boat);
                    let mut sc = 0.0f64;
                    for &(gt, gv) in &goals {
                        let mut v = gv;
                        // discount villages other own units are closer to
                        if gv >= 60.0 && gv <= 100.0 {
                            let others_closer = my_units.iter().any(|o| o.tile != s && cheb(g.size, o.tile, gt) < cheb(g.size, dest, gt).min(cheb(g.size, s, gt)) && (o.moved || o.kind == u.kind));
                            if others_closer { v *= 0.3; }
                        }
                        if d[gt] == i64::MAX { continue; }
                        let dist = d[gt] as f64;
                        let val = v / (1.0 + dist);
                        if val > sc { sc = val; }
                    }
                    // defense: stay on/near own cities when enemies are around
                    for c in &g.cities {
                        let en = g.enemy_near(c.0, 6) + if g.turn >= 8 && g.capital[c.0] == g.me { 1 } else { 0 };
                        let on_city_other = my_units.iter().any(|o| o.tile != s && o.tile == c.0);
                        let adj_other = my_units.iter().filter(|o| o.tile != s && o.tile != c.0 && cheb(g.size, o.tile, c.0) <= 1).count();
                        if en > 0 && !on_city_other {
                            if dest == c.0 { sc += 60.0; } else if cheb(g.size, dest, c.0) <= 1 && adj_other == 0 { sc += 25.0; }
                        }
                        // do not block production in a city when we can actually train a unit there
                        let homed = my_units.iter().filter(|o| o.home == c.0 as i64 + 1).count() as i64;
                        let can_train = homed <= c.1 && g.stars >= 2;
                        if dest == c.0 && want_units && can_train && g.enemy_near(c.0, 1) == 0 { sc -= 55.0; }
                    }
                    // dash attack potential from dest
                    if dash && uatk > 0.0 {
                        for e in &enemy_units {
                            if cheb(g.size, dest, e.tile) <= urange {
                                let mut au = u.clone(); au.tile = dest;
                                let (dmg, ret, dies, adies) = g.combat(&au, e);
                                let mut v = dmg - ret * 0.8 + if dies { 12.0 } else { 0.0 } - if adies { 25.0 } else { 0.0 };
                                if v > 0.0 { if urange >= 2 && cheb(g.size, dest, e.tile) == 2 { v += 5.0; } sc += v * 1.5; }
                            }
                        }
                    }
                    // danger: enemies adjacent to dest when weak
                    let threats: f64 = enemy_units.iter().filter(|e| cheb(g.size, dest, e.tile) <= stats(e.kind).3.max(1) + 1).map(|e| stats(e.kind).0).sum();
                    if u.hp / uhm < 0.5 { sc -= threats * 3.0; } else { sc -= threats * 0.8; }
                    if !boat && g.known[dest] && (g.terrain[dest] == 1 || g.terrain[dest] == 2) && !need_navy { sc -= 30.0; }
                    sc
                };
                let stay = eval(s) + if on_city && g.enemy_near(s, 4) > 0 { 10.0 } else { 0.0 };
                for (t, id) in moves {
                    if g.unit_at[*t].is_some() { continue; }
                    let v = eval(*t) - stay;
                    if v > 0.05 && best.as_ref().map_or(true, |b| v > b.0) { best = Some((v, id.clone())); }
                }
            }
            if let Some((_, id)) = best { return id; }
        }
        // 11. End turn
        if let Some(id) = end_id { return id; }
        legal[0]["action_id"].as_str().unwrap().to_owned()
    }
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut bot = Bot { last_turn: -1, end_count: 0 };
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let id = bot.choose(&obs);
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
