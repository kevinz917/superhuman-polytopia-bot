//! The packed public view (engine-rs kernels::observe) as policy features (rl/observation.py::encode_view, the
//! path training used: unpack_view of the int16 view) and as the Studio observation JSON
//! (learning/adapter.py::serialize plus studio/service.py::with_hud and world_view).

use polytopia_engine_rs::geom::Geom;
use polytopia_engine_rs::kernels::{observe, view_width, VIEW_FIELDS};
use polytopia_engine_rs::rules::Rules;
use polytopia_engine_rs::state::State;
use polytopia_engine_rs::structures::income;
use serde_json::{json, Map, Value};

use crate::net::{BOARD, GLOBAL};

/// rl/observation.py TRIBES (version 5): every tribe type but Bardur (4), in id order.
pub const TRIBES: [i32; 15] = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];

/// Start of every VIEW_FIELDS entry in the packed view.
pub struct Layout {
    pub n: usize,
    at: Vec<usize>,
}

impl Layout {
    pub fn new(n: usize) -> Layout {
        let mut at = Vec::with_capacity(VIEW_FIELDS.len());
        let mut pos = 0;
        for name in VIEW_FIELDS {
            at.push(pos);
            pos += match name {
                "embassies_received" | "embassies_sent" | "known_players" | "peace" | "peace_offers" | "tribe" => 2,
                "own_built" | "own_monuments" | "tech" | "enemy_tech" => 128,
                "hist_enemy_unit" => polytopia_engine_rs::state::HIST_TURNS * n,
                "player" | "stars" | "turn" | "converts" => 1,
                "own_city_rewards" => 9 * n,
                _ => n,
            };
        }
        assert_eq!(pos, view_width(n));
        Layout { n, at }
    }

    pub fn field<'a>(&self, view: &'a [i16], name: &str) -> &'a [i16] {
        let k = VIEW_FIELDS.iter().position(|f| *f == name).expect("view field");
        let end = if k + 1 < self.at.len() { self.at[k + 1] } else { view.len() };
        &view[self.at[k]..end]
    }
}

pub fn packed(s: &State, player: i32, g: &Geom) -> Vec<i16> {
    let mut out = vec![0i16; view_width(s.n)];
    observe(s, player, g, &mut out);
    out
}

/// encode_view: board [n, 138] and global [397] features of one seat's packed view.
pub fn features(l: &Layout, v: &[i16], size: usize, structures: &[i32], units: &[i32], board: &mut [f32], global: &mut [f32]) {
    let n = l.n;
    let f = |name: &str| l.field(v, name);
    let p = f("player")[0] as i32;
    let (known, terrain, resource, structure, unit_type) = (f("explored"), f("terrain"), f("resource"), f("structure"), f("unit_type"));
    let (owner, unit_owner, capital) = (f("owner"), f("unit_owner"), f("capital"));
    let flags = ["moved", "attacked", "poison", "frozen", "boosted", "invisible"].map(|k| f(&format!("own_unit_{k}")));
    let city = ["level", "progress", "population", "production"].map(|k| f(&format!("own_city_{k}")));
    let (road, level, health, veteran, detect) = (f("road"), f("structure_level"), f("unit_health"), f("unit_veteran"), f("detect"));
    let (kills, home, passenger) = (f("own_unit_kills"), f("own_unit_home"), f("own_unit_passenger"));
    let (connected, riot, rewards) = (f("own_city_connected"), f("own_city_riot"), f("own_city_rewards"));
    let channels = board.len() / n;
    // XLA folds `x / c` for a constant c into `x * (1 / c)` (training's encode_view under jit): the same here, so
    // the features are bit-identical (x / 10 and x * 0.1 differ by an ulp)
    let inv = |c: f32| 1.0f32 / c;
    board.fill(0.);
    let edge = (size - 1) as f32;
    for i in 0..n {
        let row = &mut board[i * channels..(i + 1) * channels];
        let k = known[i] != 0;
        let kf = k as i32 as f32;
        let mut at = 0;
        let mut categorical = |value: i16, values: &mut dyn Iterator<Item = i32>, at: &mut usize| {
            for c in values {
                row[*at] = (value as i32 == c) as i32 as f32 * kf;
                *at += 1;
            }
        };
        categorical(terrain[i], &mut (0..8), &mut at);
        categorical(resource[i], &mut (0..8), &mut at);
        categorical(structure[i], &mut structures.iter().copied(), &mut at);
        categorical(unit_type[i], &mut units.iter().copied(), &mut at);
        for a in [owner[i], unit_owner[i], capital[i]] {
            let a = a as i32;
            row[at] = (k && a == 0) as i32 as f32;
            row[at + 1] = (k && a == p) as i32 as f32;
            row[at + 2] = (k && a == 3 - p) as i32 as f32;
            at += 3;
        }
        let scalars = [
            kf, (i % size) as f32 * inv(edge), (i / size) as f32 * inv(edge), road[i].max(0) as f32, level[i].max(0) as f32 * inv(10.),
            health[i].max(0) as f32 * inv(40.), veteran[i].max(0) as f32, detect[i] as f32,
            flags[0][i] as f32, flags[1][i] as f32, flags[2][i] as f32, flags[3][i] as f32, flags[4][i] as f32, flags[5][i] as f32,
            kills[i] as f32 * inv(10.), home[i] as f32 * inv(n as f32), passenger[i] as f32 * inv(128.),
            city[0][i] as f32 * inv(20.), city[1][i] as f32 * inv(20.), city[2][i] as f32 * inv(20.), city[3][i] as f32 * inv(20.),
            connected[i] as f32, riot[i] as f32,
        ];
        row[at..at + scalars.len()].copy_from_slice(&scalars);
        at += scalars.len();
        for r in 0..9 {
            row[at + r] = rewards[i * 9 + r] as f32 * inv(10.);
        }
        debug_assert_eq!(at + 9, BOARD);
        if channels >= 151 {
            for d in 0..8 { row[BOARD + d] = (k && f("unit_direction")[i] == d as i16 + 1) as i32 as f32; }
            for h in 0..3 { row[BOARD + 8 + h] = (f("hist_enemy_unit")[h * n + i] > 0) as i32 as f32; }
            row[BOARD + 11] = (f("hist_enemy_built")[i] > 0) as i32 as f32;
            row[BOARD + 12] = (f("hist_enemy_trained")[i] > 0) as i32 as f32;
        }
        if channels >= 153 {
            // observation version 4: categorical("resource", WATER_RESOURCES) (Starfish 8, aqua crop 9) on known tiles
            for (j, r) in [8i16, 9].iter().enumerate() { row[BOARD + 13 + j] = (k && resource[i] == *r) as i32 as f32; }
        }
    }
    global.fill(0.);
    global[0] = f("turn")[0] as f32 * inv(100.);
    global[1] = (f("stars")[0] as f32).ln_1p() * inv(8.);
    global[2] = (p - 1) as f32;
    let mut at = 3;
    for name in ["tech", "own_monuments", "own_built"] {
        for &b in f(name) {
            global[at] = b as f32;
            at += 1;
        }
    }
    let order = [(p - 1) as usize, (2 - p) as usize];
    for name in ["known_players", "peace", "embassies_sent", "embassies_received", "peace_offers"] {
        let r = f(name);
        for o in order {
            global[at] = r[o] as f32;
            at += 1;
        }
    }
    debug_assert_eq!(at, GLOBAL);
    if global.len() >= 525 { for (out, &v) in global[GLOBAL..GLOBAL + 128].iter_mut().zip(f("enemy_tech")) { *out = v as f32; } }
    if global.len() >= 555 {
        // observation version 5: a one-hot of my tribe, then a multi-hot of the other seats' (view field "tribe":
        // another seat's only once met, else 0), over the tribe types but Bardur (observation.TRIBES)
        let tribe = f("tribe");
        let hot = |t: i16| TRIBES.iter().map(move |&x| (t as i32 == x) as i32 as f32);
        for (out, v) in global[525..540].iter_mut().zip(hot(tribe[(p - 1) as usize])) { *out = v; }
        for (j, out) in global[540..555].iter_mut().enumerate() {
            *out = (0..2).filter(|&q| q != (p - 1) as usize).any(|q| tribe[q] as i32 == TRIBES[j]) as i32 as f32;
        }
    }
}

fn indices(bits: &[i16]) -> Vec<usize> {
    bits.iter().enumerate().filter(|(_, &b)| b != 0).map(|(i, _)| i).collect()
}

fn bools(bits: &[i16]) -> Vec<bool> {
    bits.iter().map(|&b| b != 0).collect()
}

/// adapter.serialize of one packed view. Unit health: exact from the state for the viewer's own units (the packed
/// view truncates it); for another seat's, the packed view's whole points (toward zero), i.e. the public projection
/// (a ceil of the true value would still tell 9.0 from 9.5).
pub fn visible_state(l: &Layout, v: &[i16], s: &State, omniscient: bool) -> (i32, Value) {
    let n = l.n;
    let f = |name: &str| l.field(v, name);
    let player = f("player")[0] as i32;
    let (mut tiles, mut units, mut cities) = (Vec::with_capacity(n), Vec::new(), Vec::new());
    for i in 0..n {
        let known = f("explored")[i] != 0;
        let mut t = Map::new();
        t.insert("tile".into(), json!(i));
        t.insert("known".into(), json!(known));
        if known {
            for name in ["terrain", "owner", "structure", "resource", "road", "capital", "structure_level"] {
                t.insert(name.into(), json!(f(name)[i]));
            }
            if f("unit_owner")[i] > 0 {
                let mut u = Map::new();
                u.insert("tile".into(), json!(i));
                u.insert("owner".into(), json!(f("unit_owner")[i]));
                u.insert("kind".into(), json!(f("unit_type")[i]));
                // Another seat's unit: the packed view's whole points (the public projection), never the state's value.
                let own = f("unit_owner")[i] as i32 == player;
                u.insert("health".into(), if own || omniscient { json!(s.units.health[i]) } else { json!(f("unit_health")[i] as f32) });
                u.insert("veteran".into(), json!(f("unit_veteran")[i] != 0));
                if f("unit_owner")[i] as i32 == player {
                    for name in ["moved", "attacked", "poison", "frozen", "boosted", "invisible"] {
                        u.insert(name.into(), json!(f(&format!("own_unit_{name}"))[i] != 0));
                    }
                    for name in ["kills", "home", "passenger"] {
                        u.insert(name.into(), json!(f(&format!("own_unit_{name}"))[i]));
                    }
                    u.insert("detect".into(), json!(f("detect")[i] != 0));
                }
                units.push(Value::Object(u));
            }
            if f("structure")[i] == 1 {
                let mut c = Map::new();
                c.insert("tile".into(), json!(i));
                c.insert("owner".into(), json!(f("owner")[i]));
                if f("owner")[i] as i32 == player {
                    for name in ["level", "progress", "population", "production"] {
                        c.insert(name.into(), json!(f(&format!("own_city_{name}"))[i]));
                    }
                    for name in ["connected", "riot"] {
                        c.insert(name.into(), json!(f(&format!("own_city_{name}"))[i] != 0));
                    }
                    c.insert("rewards".into(), json!(&f("own_city_rewards")[i * 9..i * 9 + 9]));
                }
                cities.push(Value::Object(c));
            }
        }
        tiles.push(Value::Object(t));
    }
    let relations: Map<String, Value> = ["known_players", "peace", "embassies_sent", "embassies_received", "peace_offers"]
        .iter().map(|k| (k.to_string(), json!(bools(f(k))))).collect();
    let visible = json!({
        "board_size": s.size, "turn": f("turn")[0], "stars": f("stars")[0], "technology": indices(f("tech")),
        "tiles": tiles, "units": units, "cities": cities, "relations": relations,
        "achievements": {"monuments": indices(f("own_monuments")), "built": indices(f("own_built"))},
    });
    (player, visible)
}

/// Display only (not in the packed view, so no observation or training change): each corner lighthouse
/// (get corners order: (w-1, h-1), (0, 0), (0, h-1), (w-1, 0)) and the seats that have
/// explored it (`tiles.explored[p][tile]`). A seat's view lists only its own find (`viewer`): another seat's
/// exploration is not in its packed view and nothing shows the real game displays opponent emblems (so they
/// stay hidden; covered by the every-field UI test). The spectator view (`None`) lists
/// every finder. Always from the true state (world_view's all-explored copy would list every seat).
pub fn lighthouses(s: &State, viewer: Option<i32>) -> Value {
    let w = s.size;
    let corners = [w * w - 1, 0, w * (w - 1), w - 1];
    Value::Array(corners.iter().map(|&t| {
        let found_by: Vec<usize> = (0..s.np).filter(|&p| s.tiles.explored[p][t] && viewer.map_or(true, |v| v as usize == p + 1))
            .map(|p| p + 1).collect();
        json!({"tile": t, "found_by": found_by})
    }).collect())
}

/// studio/service.py::with_hud: score, income, the known tiles of each of the viewer's own cities, and the
/// viewer's own leaderboard row. About another
/// seat the UI shows only what the packed public view holds (or the real game is known to show), so a foreign city's tile
/// memberships (tiles.city), a foreign seat's score and its city total are not sent: their rows carry only the seat.
fn hud(s: &State, r: &Rules, g: &Geom, player: i32, visible: &Value, omniscient: bool) -> Value {
    let tiles = visible["tiles"].as_array().unwrap();
    let mut territory = Map::new();
    for c in visible["cities"].as_array().unwrap() {
        let owner = c["owner"].as_i64().unwrap_or(0) as i32;
        if owner == player || (omniscient && owner != 0) {
            let city = c["tile"].as_i64().unwrap() as i32;
            let members: Vec<usize> = tiles.iter().filter(|t| t["known"].as_bool() == Some(true))
                .map(|t| t["tile"].as_u64().unwrap() as usize).filter(|&t| s.tiles.city[t] == city).collect();
            territory.insert(city.to_string(), json!(members));
        }
    }
    let players: Vec<Value> = (1..=2).map(|p| {
        if p != player && !omniscient {
            return json!({"player": p});
        }
        let cities = (0..s.n).filter(|&i| s.tiles.structure[i] == 1 && s.tiles.owner[i] == p).count();
        json!({"player": p, "score": s.players.score[(p - 1) as usize], "cities": cities})
    }).collect();
    json!({"score": s.players.score[(player - 1) as usize], "income": income(s, player, r, g),
           "territory": territory, "players": players})
}

/// One seat's frame view, as studio's `with_hud(compact(env.observe(player)))`.
pub fn seat_view(s: &State, r: &Rules, g: &Geom, l: &Layout, player: i32) -> Value {
    let v = packed(s, player, g);
    let (player, mut visible) = visible_state(l, &v, s, false);
    visible["lighthouses"] = lighthouses(s, Some(player));
    // The seat's HUD (own score, income, own city count and territory) from its public projection: the viewer's own
    // state whole, nothing it cannot see (r12 every-field UI test: a hidden tile's owner reached the city count).
    let hud = hud(&crate::public::public_state(s, player), r, g, player, &visible, false);
    json!({"player": player, "state_version": s.state_version, "visible_state": visible, "hud": hud})
}

/// studio's world_view: rendering-only omniscience (all explored, nothing invisible), from the mover's seat,
/// with every city's level, progress, population and production.
pub fn world_view(s: &State, r: &Rules, g: &Geom, l: &Layout) -> Value {
    let mut w = s.clone();
    for seat in w.tiles.explored.iter_mut() {
        seat.fill(true);
    }
    w.units.invisible.fill(false);
    let player = s.current_player;
    let v = packed(&w, player, g);
    let (player, mut visible) = visible_state(l, &v, &w, true);
    for c in visible["cities"].as_array_mut().unwrap() {
        let i = c["tile"].as_u64().unwrap() as usize;
        c["level"] = json!(s.cities.level[i]);
        c["progress"] = json!(s.cities.progress[i]);
        c["population"] = json!(s.cities.population[i]);
        c["production"] = json!(s.cities.production[i]);
    }
    visible["lighthouses"] = lighthouses(s, None);
    let hud = hud(s, r, g, player, &visible, true);
    json!({"player": player, "state_version": s.state_version, "visible_state": visible, "hud": hud})
}

#[cfg(test)]
mod tests {
    use super::*;
    use polytopia_engine_rs::mapgen;

    const SIZE: usize = 14;

    fn setup() -> (State, Rules, Geom, Layout) {
        let rules = Rules::from_json(include_str!("../../src/polytopia_bot/gpu/rules.json")).unwrap();
        let g = Geom::new(SIZE);
        let s = mapgen::reset(7, &g, None);
        (s, rules, g, Layout::new(SIZE * SIZE))
    }

    fn found(v: &Value) -> Vec<(u64, Vec<u64>)> {
        v["visible_state"]["lighthouses"].as_array().unwrap().iter().map(|x| (
            x["tile"].as_u64().unwrap(),
            x["found_by"].as_array().unwrap().iter().map(|p| p.as_u64().unwrap()).collect(),
        )).collect()
    }

    #[test]
    fn lists_the_four_corners_in_get_corners_order() {
        let (s, r, g, l) = setup();
        let n = SIZE as u64;
        let tiles: Vec<u64> = found(&seat_view(&s, &r, &g, &l, 1)).iter().map(|x| x.0).collect();
        assert_eq!(tiles, vec![n * n - 1, 0, n * (n - 1), n - 1]);
        for &t in &tiles {
            assert_eq!(s.tiles.structure[t as usize], 47, "corner {t} holds the lighthouse");
        }
    }

    #[test]
    fn found_by_is_the_viewers_own_find_and_every_find_for_the_spectator() {
        let (mut s, r, g, l) = setup();
        let w = SIZE;
        for p in 0..s.np {
            for &t in &[w * w - 1, 0, w * (w - 1), w - 1] {
                s.tiles.explored[p][t] = false;
            }
        }
        s.tiles.explored[0][0] = true;
        s.tiles.explored[1][0] = true;
        s.tiles.explored[1][w - 1] = true;
        let want = vec![
            ((w * w - 1) as u64, vec![]), (0, vec![1, 2]), ((w * (w - 1)) as u64, vec![]), ((w - 1) as u64, vec![2]),
        ];
        // A seat sees only its own finds (opponent emblems stay hidden).
        let only = |seat: u64| want.iter().map(|(t, f)| (*t, f.iter().copied().filter(|&p| p == seat).collect::<Vec<u64>>())).collect::<Vec<_>>();
        assert_eq!(found(&seat_view(&s, &r, &g, &l, 1)), only(1));
        assert_eq!(found(&seat_view(&s, &r, &g, &l, 2)), only(2));
        // world_view explores everything for rendering, but the finders are the true ones.
        assert_eq!(found(&world_view(&s, &r, &g, &l)), want);
    }

    /// Seat 1 has explored the whole board, so seat 2's capital and units are in its view.
    fn opened() -> (State, Rules, Geom, Layout, usize, usize) {
        let (mut s, r, g, l) = setup();
        s.tiles.explored[0].fill(true);
        let unit = (0..s.n).find(|&i| s.units.owner[i] == 2).expect("seat 2 starts with a unit");
        let city = (0..s.n).find(|&i| s.tiles.structure[i] == 1 && s.tiles.owner[i] == 2).expect("seat 2 capital");
        (s, r, g, l, unit, city)
    }

    fn unit_health(v: &Value, tile: usize) -> Value {
        v["visible_state"]["units"].as_array().unwrap().iter()
            .find(|u| u["tile"].as_u64() == Some(tile as u64)).expect("unit in view")["health"].clone()
    }

    #[test]
    fn foreign_unit_health_is_whole_displayed_points() {
        // Never a sub-point value for another seat's unit (9.125 -> 9.5 was sent).
        let (mut s, r, g, l, unit, _) = opened();
        let mut seen = Vec::new();
        for h in [9.125f32, 9.5, 9.875, 10.0] {
            s.units.health[unit] = h;
            seen.push(unit_health(&seat_view(&s, &r, &g, &l, 1), unit));
            // The owner still gets its exact value, and so does the omniscient spectator view.
            assert_eq!(unit_health(&seat_view(&s, &r, &g, &l, 2), unit), json!(h));
            assert_eq!(unit_health(&world_view(&s, &r, &g, &l), unit), json!(h));
        }
        assert_eq!(seen, vec![json!(9.0), json!(9.0), json!(9.0), json!(10.0)]);
    }

    #[test]
    fn hud_holds_no_foreign_score_city_total_or_city_membership() {
        // Hidden from another seat: foreign tiles.city, foreign score ( 515 -> 552, cities 1 -> 2).
        let (mut s, r, g, l, _, city) = opened();
        let v = seat_view(&s, &r, &g, &l, 1);
        let hud = &v["hud"];
        assert_eq!(hud["players"][1], json!({"player": 2}));
        assert!(hud["players"][0].get("score").is_some() && hud["players"][0].get("cities").is_some());
        assert!(v["visible_state"]["cities"].as_array().unwrap().iter().any(|c| c["tile"].as_u64() == Some(city as u64)),
            "seat 2's capital is in seat 1's view");
        for key in hud["territory"].as_object().unwrap().keys() {
            let t: usize = key.parse().unwrap();
            assert_eq!(s.tiles.owner[t], 1, "territory of a foreign city {t}");
        }
        // Seat 2's score, its city total (a city on a tile seat 1 has not explored) and the city memberships of
        // tiles seat 1 sees: changing all three leaves seat 1's whole frame, HUD included, unchanged.
        let hidden = (0..s.n).find(|&i| s.tiles.structure[i] == 0 && s.tiles.owner[i] == 0 && s.units.owner[i] == 0).unwrap();
        let mut t = s.clone();
        t.players.score[1] += 37;
        t.tiles.explored[0][hidden] = false;
        s.tiles.explored[0][hidden] = false;
        t.tiles.structure[hidden] = 1;
        t.tiles.owner[hidden] = 2;
        let mut moved = 0;
        for i in 0..t.n {
            if t.tiles.city[i] == city as i32 && i != city {
                t.tiles.city[i] = -1;
                moved += 1;
            }
        }
        assert!(moved > 0, "the capital owns visible tiles");
        assert_eq!(seat_view(&s, &r, &g, &l, 1), seat_view(&t, &r, &g, &l, 1));
        // The world view (watch / finished games) keeps every seat's row and territory.
        let w = world_view(&s, &r, &g, &l);
        assert!(w["hud"]["players"].as_array().unwrap().iter().all(|p| p.get("score").is_some()));
    }

    #[test]
    fn display_only_the_packed_view_is_unchanged() {
        let (mut s, _r, g, _l) = setup();
        let before = packed(&s, 1, &g);
        assert_eq!(packed(&s, 1, &g), before);
        let t = SIZE * SIZE - 1;
        let had = s.tiles.explored[0][t];
        s.tiles.explored[1][t] = !s.tiles.explored[1][t];
        // Seat 2's finds never reach seat 1's policy input.
        assert_eq!(packed(&s, 1, &g), before);
        assert_eq!(s.tiles.explored[0][t], had);
    }
}
