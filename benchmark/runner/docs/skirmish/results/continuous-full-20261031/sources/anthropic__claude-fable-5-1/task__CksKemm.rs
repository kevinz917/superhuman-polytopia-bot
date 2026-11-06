use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};

struct Mem {
    turn: i64,
    seen_enemy: bool,
    skipped: HashSet<usize>,
}

struct Tile {
    known: bool,
    terrain: i64,
    owner: i64,
    structure: i64,
    resource: i64,
}

struct Unit {
    tile: usize,
    owner: i64,
    kind: i64,
    hp: f64,
    veteran: bool,
    moved: bool,
    attacked: bool,
    home: i64,
}

fn stats(kind: i64) -> (f64, f64, f64, i64, i64) {
    // attack, defense, hp, range, cost
    match kind {
        2 => (2.0, 2.0, 10.0, 1, 2),
        3 => (2.0, 1.0, 10.0, 1, 3),
        4 => (3.5, 1.0, 10.0, 1, 8),
        5 => (1.0, 3.0, 15.0, 1, 3),
        8 => (4.0, 0.0, 10.0, 3, 8),
        9 => (2.0, 1.0, 10.0, 2, 3),
        10 => (0.0, 1.0, 10.0, 1, 5),
        11 => (3.0, 3.0, 15.0, 1, 5),
        12 => (5.0, 4.0, 40.0, 1, 10),
        37 => (2.0, 2.0, 10.0, 1, 2),
        38 => (0.0, 0.5, 5.0, 1, 8),
        39 => (0.0, 0.5, 5.0, 1, 2),
        40 => (2.0, 2.0, 10.0, 1, 2),
        41 => (3.0, 2.0, 10.0, 3, 15),
        42 => (2.0, 1.0, 10.0, 2, 5),
        43 => (0.0, 1.0, 10.0, 2, 0),
        44 => (3.0, 3.0, 10.0, 1, 5),
        45 => (4.0, 4.0, 40.0, 1, 10),
        _ => (2.0, 2.0, 10.0, 1, 3),
    }
}

fn cheb(a: usize, b: usize, size: usize) -> i64 {
    let (ax, ay) = ((a % size) as i64, (a / size) as i64);
    let (bx, by) = ((b % size) as i64, (b / size) as i64);
    (ax - bx).abs().max((ay - by).abs())
}

fn gi(v: &Value, k: &str) -> i64 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(0)
}
fn gb(v: &Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}

fn choose(obs: &Value, mem: &mut Mem) -> String {
    let legal = obs["legal_actions"].as_array().expect("legal_actions");
    let vs = &obs["visible_state"];
    let me = obs["player"].as_i64().unwrap_or(1);
    let opp = 3 - me;
    let size = vs["board_size"].as_u64().unwrap_or(14) as usize;
    let n = size * size;
    let turn = vs["turn"].as_i64().unwrap_or(0);
    let stars = vs["stars"].as_i64().unwrap_or(0);
    if turn != mem.turn {
        mem.turn = turn;
        mem.skipped.clear();
    }
    let tech: HashSet<i64> = vs["technology"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();

    let mut tiles: Vec<Tile> = (0..n).map(|_| Tile { known: false, terrain: 0, owner: 0, structure: 0, resource: 0 }).collect();
    if let Some(arr) = vs["tiles"].as_array() {
        for t in arr {
            let i = gi(t, "tile") as usize;
            if i < n {
                tiles[i] = Tile { known: gb(t, "known"), terrain: gi(t, "terrain"), owner: gi(t, "owner"), structure: gi(t, "structure"), resource: gi(t, "resource") };
            }
        }
    }
    let mut units: Vec<Unit> = Vec::new();
    let mut unit_at: HashMap<usize, usize> = HashMap::new();
    if let Some(arr) = vs["units"].as_array() {
        for u in arr {
            let t = gi(u, "tile") as usize;
            unit_at.insert(t, units.len());
            units.push(Unit { tile: t, owner: gi(u, "owner"), kind: gi(u, "kind"), hp: u["health"].as_f64().unwrap_or(10.0), veteran: gb(u, "veteran"), moved: gb(u, "moved"), attacked: gb(u, "attacked"), home: gi(u, "home") });
        }
    }
    let mut my_cities: Vec<(usize, i64)> = Vec::new();
    let mut enemy_cities: Vec<usize> = Vec::new();
    if let Some(arr) = vs["cities"].as_array() {
        for c in arr {
            let t = gi(c, "tile") as usize;
            let o = gi(c, "owner");
            if o == me {
                my_cities.push((t, gi(c, "level")));
            } else if o == opp {
                enemy_cities.push(t);
            }
        }
    }
    let my_units: Vec<usize> = (0..units.len()).filter(|&i| units[i].owner == me).collect();
    let enemy_units: Vec<usize> = (0..units.len()).filter(|&i| units[i].owner == opp).collect();
    let n_my_units = my_units.len() as i64;
    let n_cities = my_cities.len().max(1) as i64;

    // Terrain stats
    let known_cnt = tiles.iter().filter(|t| t.known).count().max(1) as f64;
    let water_cnt = tiles.iter().filter(|t| t.known && (t.terrain == 1 || t.terrain == 2)).count() as f64;
    let water_frac = water_cnt / known_cnt;
    let water_near_city = my_cities.iter().any(|&(c, _)| (0..n).any(|t| tiles[t].known && tiles[t].terrain == 1 && cheb(c, t, size) <= 2 && tiles[t].owner == me));

    // Threat detection
    let mut threatened: Vec<usize> = Vec::new();
    for &(c, _) in &my_cities {
        if enemy_units.iter().any(|&e| cheb(units[e].tile, c, size) <= 3) {
            threatened.push(c);
        }
    }
    let danger = !threatened.is_empty();
    let dominant = stars >= 60 || (my_units.len() >= 10 && my_units.len() >= 2 * enemy_units.len() + 4);

    let find = |kind: i64, pred: &dyn Fn(&Value) -> bool| -> Option<String> {
        legal.iter().find(|a| a["command"]["kind"] == kind && pred(&a["command"])).map(|a| a["action_id"].as_str().unwrap().to_owned())
    };

    // 1. City reward
    if !enemy_units.is_empty() || !enemy_cities.is_empty() { mem.seen_enemy = true; }
    let seen_enemy = mem.seen_enemy;
    if legal.iter().any(|a| a["command"]["kind"] == 9) {
        let prefs: [i64; 9] = if seen_enemy { [3, 1, 8, 6, 4, 7, 5, 2, 0] } else { [3, 7, 8, 6, 4, 1, 5, 2, 0] };
        for pref in prefs {
            if let Some(id) = find(9, &|c| gi(c, "parameter") == pref) {
                return id;
            }
        }
    }
    // Opening: Riding immediately
    if turn <= 2 && !tech.contains(&1) && stars >= 5 {
        if let Some(id) = find(7, &|c| gi(c, "parameter") == 1) { return id; }
    }
    // 2. Promote
    if let Some(id) = find(3, &|c| gi(c, "parameter") == 16) {
        return id;
    }
    // 3. Capture
    if let Some(id) = find(8, &|_| true) {
        return id;
    }
    // 4. Attack
    {
        let mut best: Option<(f64, String)> = None;
        for a in legal.iter().filter(|a| a["command"]["kind"] == 2) {
            let c = &a["command"];
            let s = gi(c, "source") as usize;
            let t = gi(c, "target") as usize;
            let (Some(&ui), Some(&ti)) = (unit_at.get(&s), unit_at.get(&t)) else { continue };
            let u = &units[ui];
            let d = &units[ti];
            let (ua, _ud, uh, _ur, _uc) = stats(u.kind);
            let (_da, dd, dh, dr, dc) = stats(d.kind);
            if ua <= 0.0 { continue; }
            let umax = uh + if u.veteran { 5.0 } else { 0.0 };
            let dmax = dh + if d.veteran { 5.0 } else { 0.0 };
            let dt = &tiles[t];
            let mut bonus = 1.0;
            if dt.structure == 1 && dt.owner == opp { bonus = 1.5; }
            else if dt.terrain == 4 { bonus = 1.5; }
            else if dt.terrain == 5 || dt.terrain == 1 || dt.terrain == 2 { bonus = 1.2; }
            let af = ua * u.hp / umax;
            let df = dd * d.hp / dmax * bonus;
            let tot = af + df;
            let dmg = ((af / tot) * ua * 4.5 + 0.5).floor();
            let kill = dmg >= d.hp;
            let dist = cheb(s, t, size);
            let ret = if kill || dist > dr { 0.0 } else { ((df / tot) * dd * 4.5 + 0.5).floor() };
            let mut score = if kill { 10.0 + dc as f64 * 1.5 } else { dmg - ret * 1.3 };
            if !kill && ret >= u.hp { score -= 40.0; }
            if !kill && u.hp - ret < 3.0 { score -= 10.0; }
            // enemy on/near my city: more urgent
            if my_cities.iter().any(|&(c, _)| cheb(c, t, size) <= 1) { score += 4.0; }
            if kill && dt.structure == 1 && dt.owner == opp { score += 3.0; }
            let (_, _, _, ur2, _) = stats(u.kind);
            if kill && ur2 < 2 && my_cities.iter().any(|&(c, _)| c == s) {
                let others = enemy_units.iter().filter(|&&e| units[e].tile != t && cheb(units[e].tile, s, size) <= 3).count();
                if others > 0 { score -= if dominant { 3.0 } else { 100.0 }; }
            }
            if best.as_ref().map_or(true, |b| score > b.0) {
                best = Some((score, a["action_id"].as_str().unwrap().to_owned()));
            }
        }
        if let Some((sc, id)) = best {
            if sc > 0.5 { return id; }
        }
    }

    // Summon helper
    let summon = |stars: i64| -> Option<String> {
        let mut prefs: Vec<i64> = Vec::new();
        let archers = my_units.iter().filter(|&&i| units[i].kind == 9).count();
        let want_archer = tech.contains(&18) && archers * 3 < my_units.len();
        let defenders = my_units.iter().filter(|&&i| units[i].kind == 5).count();
        if tech.contains(&7) && danger && defenders < my_cities.len() { prefs.push(5); }
        if tech.contains(&24) && stars >= 5 { prefs.push(11); }
        if tech.contains(&3) && stars >= 8 { prefs.push(4); }
        if want_archer { prefs.push(9); }
        if danger { prefs.push(2); }
        if tech.contains(&1) { prefs.push(3); }
        prefs.push(9);
        prefs.push(2);
        for k in prefs {
            // prefer summoning in threatened cities, else any
            let mut cands: Vec<&Value> = legal.iter().filter(|a| a["command"]["kind"] == 4 && gi(&a["command"], "parameter") == k).collect();
            if cands.is_empty() { continue; }
            cands.sort_by_key(|a| {
                let s = gi(&a["command"], "source") as usize;
                let thr = if threatened.contains(&s) { 0 } else { 1 };
                let near_enemy = enemy_cities.iter().map(|&e| cheb(s, e, size)).min().unwrap_or(20);
                (thr, near_enemy)
            });
            return Some(cands[0]["action_id"].as_str().unwrap().to_owned());
        }
        None
    };

    let min_army = n_cities + 1;
    let cap = if seen_enemy { 4 * n_cities + 8 } else { 2 * n_cities + 3 };
    let any_summon = legal.iter().any(|a| a["command"]["kind"] == 4);
    let has_room = |c: usize, lvl: i64| -> bool { (my_units.iter().filter(|&&i| units[i].home == c as i64 + 1).count() as i64) <= lvl };
    let occupied_city: Option<usize> = my_cities.iter().filter(|&&(c, lvl)| has_room(c, lvl)).map(|&(c, _)| c).find(|&c| unit_at.get(&c).map_or(false, |&ui| units[ui].owner == me && !units[ui].moved));
    let city_safe = occupied_city.map_or(false, |c| !enemy_units.iter().any(|&e| cheb(units[e].tile, c, size) <= 2));
    let want_vacate = !any_summon && stars >= 2 && occupied_city.is_some() && (city_safe || stars >= 25) && (n_my_units < cap || stars >= 10);
    if (n_my_units < min_army || danger) && n_my_units < cap {
        if let Some(id) = summon(stars) { return id; }
    }

    let economy = || -> Option<String> {
    // 5. Harvest
    if let Some(id) = find(5, &|_| true) { return Some(id); }

    // 6. Build
    {
        let my_ports = (0..n).filter(|&t| tiles[t].structure == 8 && tiles[t].owner == me).count() as i64;
        let mut best: Option<(f64, String)> = None;
        for a in legal.iter().filter(|a| a["command"]["kind"] == 6) {
            let c = &a["command"];
            let p = gi(c, "parameter");
            let t = gi(c, "target") as usize;
            let adj = |k: i64| -> i64 {
                (0..n).filter(|&j| j != t && cheb(j, t, size) == 1 && tiles[j].structure == k && tiles[j].owner == me).count() as i64
            };
            let (benefit, cost): (f64, f64) = match p {
                5 => (2.0, 5.0),
                21 => (2.0, 5.0),
                12 => (1.0, 3.0),
                8 => (if (water_frac > 0.15 || stars >= 30) && my_ports < n_cities { 2.0 } else { 0.6 }, 7.0),
                13 => (adj(12) as f64, 5.0),
                6 => (adj(5) as f64, 5.0),
                22 => (2.0 * adj(21) as f64, 5.0),
                50 => ((adj(6) + adj(13) + adj(22)) as f64 * 0.8, 5.0),
                23..=29 | 123 => (3.0, 1.0),
                _ => (0.0, 1.0),
            };
            let v = benefit / cost;
            if v >= 0.19 && best.as_ref().map_or(true, |b| v > b.0) {
                best = Some((v, a["action_id"].as_str().unwrap().to_owned()));
            }
        }
        if let Some((_, id)) = best { return Some(id); }
    }

    // 7. Research
    if (!danger && n_my_units >= n_cities + 1) || stars >= 12 {
        let mountains = (0..n).filter(|&t| tiles[t].owner == me && tiles[t].terrain == 4).count();
        let fruit = (0..n).filter(|&t| tiles[t].owner == me && tiles[t].resource == 6).count();
        let forests = (0..n).filter(|&t| tiles[t].owner == me && tiles[t].terrain == 5).count();
        let mut order: Vec<i64> = Vec::new();
        if water_frac > 0.35 || (water_near_city && water_frac > 0.2) { order.push(10); }
        order.push(1);
        if fruit >= 2 || n_cities >= 3 { order.push(6); }
        if seen_enemy && tech.contains(&6) { order.push(7); }
        if water_frac > 0.2 && n_cities >= 2 { order.push(10); }
        if n_cities >= 3 { order.push(18); }
        if forests >= 3 && n_cities >= 3 { order.push(16); }
        if n_cities >= 3 { order.push(8); }
        if mountains >= 3 && n_cities >= 3 { order.push(20); }
        if mountains >= 3 && n_cities >= 3 { order.push(23); }
        if water_frac > 0.3 && n_cities >= 2 { order.push(13); }
        if seen_enemy { order.extend([6, 7, 18, 20, 23, 24, 2, 3]); }
        if n_cities >= 3 || stars >= 20 { order.extend([7, 24, 4, 2, 3, 17, 13, 9, 21, 22, 5, 44, 14, 19, 12]); }
        let reserve = if danger { 6 } else if seen_enemy { 3 } else if turn <= 2 { 0 } else { 3 };
        let tier = |k: i64| -> i64 { match k { 1 | 6 | 10 | 15 | 20 => 1, 2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2, _ => 3 } };
        for k in order {
            if tech.contains(&k) { continue; }
            let Some(a) = legal.iter().find(|a| a["command"]["kind"] == 7 && gi(&a["command"], "parameter") == k) else { continue };
            let cost = 4 + n_cities * tier(k);
            if stars - cost < reserve { continue; }
            return Some(a["action_id"].as_str().unwrap().to_owned());
        }
    }

        None
    };
    if !want_vacate { if let Some(id) = economy() { return id; } }
    // 8. Summon with leftovers
    if n_my_units < cap || stars >= 10 {
        if let Some(id) = summon(stars) { return id; }
    }

    // 9. Moves
    {
        // goals
        let mut claimed: HashSet<usize> = HashSet::new();
        for &i in &my_units {
            let t = units[i].tile;
            if (tiles[t].structure == 1 && tiles[t].owner != me) || tiles[t].structure == 2 { claimed.insert(t); }
        }
        let villages: Vec<usize> = (0..n).filter(|&t| tiles[t].known && ((tiles[t].structure == 1 && tiles[t].owner == 0) || tiles[t].structure == 2) && !claimed.contains(&t)).collect();
        let movers: Vec<usize> = my_units.iter().cloned().filter(|&i| !units[i].moved && !mem.skipped.contains(&units[i].tile)).collect();
        let mut goal: HashMap<usize, usize> = HashMap::new();
        // threatened cities: nearest unassigned unit defends if city empty
        for &c in &threatened {
            if unit_at.get(&c).map_or(false, |&ui| units[ui].owner == me) { continue; }
            let mut best: Option<(i64, usize)> = None;
            for &i in &movers {
                if goal.contains_key(&i) { continue; }
                let d = cheb(units[i].tile, c, size);
                if best.map_or(true, |b| d < b.0) { best = Some((d, i)); }
            }
            if let Some((_, i)) = best { goal.insert(i, c); }
        }
        // assign villages greedily
        let mut pairs: Vec<(i64, usize, usize)> = Vec::new();
        for &i in &movers {
            for &v in &villages {
                pairs.push((cheb(units[i].tile, v, size), i, v));
            }
        }
        pairs.sort();
        let mut taken: HashSet<usize> = HashSet::new();
        for (_, i, v) in pairs {
            if goal.contains_key(&i) || taken.contains(&v) { continue; }
            goal.insert(i, v);
            taken.insert(v);
        }
        // defence: units near a threatened city stay close
        for &i in &movers {
            if dominant { break; }
            let t = units[i].tile;
            let (ua, _, _, _, _) = stats(units[i].kind);
            if ua <= 0.0 { continue; }
            if let Some(&g0) = goal.get(&i) { if villages.contains(&g0) && cheb(t, g0, size) <= 3 { continue; } }
            let threatened4: Vec<usize> = my_cities.iter().map(|&(c, _)| c).filter(|&c| enemy_units.iter().any(|&e| cheb(units[e].tile, c, size) <= 4)).collect();
            if let Some(&c) = threatened4.iter().min_by_key(|&&c| cheb(t, c, size)) {
                let d = cheb(t, c, size);
                let empty = !unit_at.get(&c).map_or(false, |&ui| units[ui].owner == me);
                if d <= 5 && d >= 2 { goal.insert(i, c); }
                else if d == 1 && empty { goal.insert(i, c); }
                else if d < 2 { goal.remove(&i); mem.skipped.insert(t); }
            }
        }
        // frontier tiles
        let frontier: Vec<usize> = (0..n).filter(|&t| tiles[t].known && tiles[t].terrain >= 3 && (0..n).any(|j| !tiles[j].known && cheb(t, j, size) == 1)).collect();
        let enemy_unit_tiles: Vec<usize> = enemy_units.iter().map(|&e| units[e].tile).collect();
        for &i in &movers {
            if goal.contains_key(&i) { continue; }
            let t = units[i].tile;
            let (mut ua, _, _, _, _) = stats(units[i].kind);
            if units[i].kind == 43 { ua = 1.0; }
            // enemy city if army is big enough
            if !enemy_cities.is_empty() && (n_my_units >= 4 || turn > 40) && ua > 0.0 {
                let g = *enemy_cities.iter().min_by_key(|&&e| cheb(t, e, size)).unwrap();
                goal.insert(i, g);
                continue;
            }
            if !enemy_unit_tiles.is_empty() && ua > 0.0 && n_my_units >= 3 {
                let g = *enemy_unit_tiles.iter().min_by_key(|&&e| cheb(t, e, size)).unwrap();
                if cheb(t, g, size) <= 4 { goal.insert(i, g); continue; }
            }
            if !frontier.is_empty() {
                let mut bestf: Option<(f64, usize)> = None;
                for &f in &frontier {
                    if taken.contains(&f) { continue; }
                    let unk = (0..n).filter(|&j| !tiles[j].known && cheb(f, j, size) <= 2).count() as f64;
                    let crowd = my_units.iter().filter(|&&j| j != i && cheb(units[j].tile, f, size) <= 2).count() as f64;
                    let sc = unk * 0.4 - cheb(t, f, size) as f64 - crowd * 2.0;
                    if bestf.map_or(true, |b| sc > b.0) { bestf = Some((sc, f)); }
                }
                if let Some((_, g)) = bestf { goal.insert(i, g); taken.insert(g); }
            }
        }
        // choose best move
        let mut best: Option<(f64, String)> = None;
        let mut has_move: HashSet<usize> = HashSet::new();
        for a in legal.iter().filter(|a| a["command"]["kind"] == 1) {
            let c = &a["command"];
            let s = gi(c, "source") as usize;
            let t = gi(c, "target") as usize;
            has_move.insert(s);
            let Some(&ui) = unit_at.get(&s) else { continue };
            let vacate = want_vacate && occupied_city == Some(s);
            let g = match goal.get(&ui) { Some(&g) => g, None => { if vacate { s } else { continue } } };
            let u = &units[ui];
            let (ua, _ud, uh, _, _) = stats(u.kind);
            let umax = uh + if u.veteran { 5.0 } else { 0.0 };
            let mut imp = (cheb(s, g, size) - cheb(t, g, size)) as f64;
            if vacate { imp += if cheb(s, t, size) <= 1 { 2.0 } else { 1.2 }; }
            if imp <= 0.0 { continue; }
            // bonus: landing on village/ruin
            if villages.contains(&t) { imp += 2.0; }
            // danger: enemy adjacency when weak
            let adj_enemies = enemy_unit_tiles.iter().filter(|&&e| cheb(e, t, size) <= 1).count() as f64;
            let on_my_city = my_cities.iter().any(|&(c, _)| c == t);
            if u.hp < umax * 0.5 { imp -= adj_enemies * 1.5; }
            if !on_my_city && adj_enemies >= 2.0 && !villages.contains(&t) { imp -= (adj_enemies - 1.0) * 1.0; }
            // don't rush enemy city alone
            if enemy_cities.contains(&g) && cheb(t, g, size) <= 1 {
                let support = my_units.iter().filter(|&&j| j != ui && cheb(units[j].tile, g, size) <= 2).count();
                if support < 2 && !dominant { imp -= 3.0; }
            }
            // avoid parking on own city (blocks training) unless defending
            if my_cities.iter().any(|&(c, _)| c == t) && !threatened.contains(&t) { imp -= 0.5; }
            // prefer not ending in forest/mountain if alternatives (slows next move)
            if tiles[t].terrain == 5 { imp -= 0.2; }
            if ua <= 0.0 && adj_enemies > 0.0 { imp -= 2.0; }
            if best.as_ref().map_or(true, |b| imp > b.0) {
                best = Some((imp, a["action_id"].as_str().unwrap().to_owned()));
            }
        }
        if let Some((imp, id)) = best {
            if imp > 0.0 { return id; }
        }
        // recover for hurt idle units
        for &i in &my_units {
            let u = &units[i];
            if u.moved || u.attacked { continue; }
            let (_, _, uh, _, _) = stats(u.kind);
            let umax = uh + if u.veteran { 5.0 } else { 0.0 };
            if u.hp < umax * 0.6 {
                let t = u.tile;
                if let Some(id) = find(3, &|c| gi(c, "parameter") == 7 && gi(c, "source") as usize == t) { return id; }
            }
        }
        // mark all unmoved movers as skipped
        for &i in &movers { mem.skipped.insert(units[i].tile); }
        let _ = has_move;
    }

    if want_vacate { if let Some(id) = economy() { return id; } }
    // 10. End turn
    if let Some(id) = find(10, &|_| true) { return id; }
    legal[0]["action_id"].as_str().unwrap().to_owned()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Mem { turn: -1, seen_enemy: false, skipped: HashSet::new() };
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let id = choose(&obs, &mut mem);
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
