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
                "embassies_received" | "embassies_sent" | "known_players" | "peace" | "peace_offers" => 2,
                "own_built" | "own_monuments" | "tech" => 128,
                "player" | "stars" | "turn" => 1,
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
    board[..n * BOARD].fill(0.);
    let edge = (size - 1) as f32;
    for i in 0..n {
        let row = &mut board[i * BOARD..(i + 1) * BOARD];
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
            kf, (i % size) as f32 / edge, (i / size) as f32 / edge, road[i].max(0) as f32, level[i].max(0) as f32 / 10.,
            health[i].max(0) as f32 / 40., veteran[i].max(0) as f32, detect[i] as f32,
            flags[0][i] as f32, flags[1][i] as f32, flags[2][i] as f32, flags[3][i] as f32, flags[4][i] as f32, flags[5][i] as f32,
            kills[i] as f32 / 10., home[i] as f32 / n as f32, passenger[i] as f32 / 128.,
            city[0][i] as f32 / 20., city[1][i] as f32 / 20., city[2][i] as f32 / 20., city[3][i] as f32 / 20.,
            connected[i] as f32, riot[i] as f32,
        ];
        row[at..at + scalars.len()].copy_from_slice(&scalars);
        at += scalars.len();
        for r in 0..9 {
            row[at + r] = rewards[i * 9 + r] as f32 / 10.;
        }
        debug_assert_eq!(at + 9, BOARD);
    }
    global[..GLOBAL].fill(0.);
    global[0] = f("turn")[0] as f32 / 100.;
    global[1] = (f("stars")[0] as f32).ln_1p() / 8.;
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
}

fn indices(bits: &[i16]) -> Vec<usize> {
    bits.iter().enumerate().filter(|(_, &b)| b != 0).map(|(i, _)| i).collect()
}

fn bools(bits: &[i16]) -> Vec<bool> {
    bits.iter().map(|&b| b != 0).collect()
}

/// adapter.serialize of one packed view. Unit health comes from the state (the packed view truncates it), and
/// only where the view shows the unit.
pub fn visible_state(l: &Layout, v: &[i16], s: &State) -> (i32, Value) {
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
                u.insert("health".into(), json!(s.units.health[i]));
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

/// studio/service.py::with_hud: score, income, the known tiles of each owned city, and the public leaderboard.
fn hud(s: &State, r: &Rules, g: &Geom, player: i32, visible: &Value) -> Value {
    let tiles = visible["tiles"].as_array().unwrap();
    let mut territory = Map::new();
    for c in visible["cities"].as_array().unwrap() {
        if c["owner"].as_i64().unwrap_or(0) != 0 {
            let city = c["tile"].as_i64().unwrap() as i32;
            let members: Vec<usize> = tiles.iter().filter(|t| t["known"].as_bool() == Some(true))
                .map(|t| t["tile"].as_u64().unwrap() as usize).filter(|&t| s.tiles.city[t] == city).collect();
            territory.insert(city.to_string(), json!(members));
        }
    }
    let players: Vec<Value> = (1..=2).map(|p| {
        let cities = (0..s.n).filter(|&i| s.tiles.structure[i] == 1 && s.tiles.owner[i] == p).count();
        json!({"player": p, "score": s.players.score[(p - 1) as usize], "cities": cities})
    }).collect();
    json!({"score": s.players.score[(player - 1) as usize], "income": income(s, player, r, g),
           "territory": territory, "players": players})
}

/// One seat's frame view, as studio's `with_hud(compact(env.observe(player)))`.
pub fn seat_view(s: &State, r: &Rules, g: &Geom, l: &Layout, player: i32) -> Value {
    let v = packed(s, player, g);
    let (player, visible) = visible_state(l, &v, s);
    let hud = hud(s, r, g, player, &visible);
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
    let (player, mut visible) = visible_state(l, &v, &w);
    for c in visible["cities"].as_array_mut().unwrap() {
        let i = c["tile"].as_u64().unwrap() as usize;
        c["level"] = json!(s.cities.level[i]);
        c["progress"] = json!(s.cities.progress[i]);
        c["population"] = json!(s.cities.population[i]);
        c["production"] = json!(s.cities.production[i]);
    }
    let hud = hud(s, r, g, player, &visible);
    json!({"player": player, "state_version": s.state_version, "visible_state": visible, "hud": hud})
}
