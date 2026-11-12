//! The game engine and the trained policy in the browser: engine-rs (the real game's rules, polytopia-2.17.3, Bardur
//! 1v1; no peace, as in training) compiled to WebAssembly plus a CPU port of
//! the policy network, behind a small C ABI (no wasm-bindgen). JavaScript writes inputs through `input`, calls an
//! export, and reads JSON results through `output_ptr` / `output_len`. Errors return a negative code and leave
//! {"error": ...} in the output.

pub mod net;
pub mod v3;
pub mod policy;
pub mod public;
pub mod search;
pub mod view;
pub mod ui;

use std::cell::RefCell;

use polytopia_engine_rs::core::{self, decode, step, terminal_rewards, Scratch, UNIT_ROW};
use polytopia_engine_rs::geom::Geom;
use polytopia_engine_rs::mapgen;
use polytopia_engine_rs::masks::{legal_masks, mask_width, MaskScratch};
use polytopia_engine_rs::rules::Rules;
use polytopia_engine_rs::state::State;
use serde_json::{json, Value};

use crate::net::Net;
use crate::policy::{Catalog, Policy};
use crate::view::Layout;

/// The board of game_new / game_from_actions (the Dryland 14 x 14 games of every release before u88890).
pub const SIZE: usize = 14;
/// Board sides the engine plays (the real game's 11 / 14 / 16 / 18 / 20; u88890 trained on all five).
pub const SIZES: [usize; 5] = [11, 14, 16, 18, 20];
/// The 12 regular tribe types (maps/state.py ENGINE_TRIBES at c7e34799): Ai-Mo 2, Bardur 4, Hoodrick 6, Imperius 7,
/// Kickoo 8, Luxidoor 9, Oumaji 10, Quetzali 11, Vengir 12, Xin-xi 13, Yadakk 14, Zebasi 15.
pub const TRIBES: [i32; 12] = [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// GameBatch's default action cap (engine-rs), the same as the training actors and the Studio engine.
pub const MAX_ACTIONS: i32 = 10000;
const RULES_JSON: &str = include_str!("../../src/polytopia_bot/gpu/rules.json");

/// The recover family (engine-rs core::decode family 9) and its entries per tile in the site's action layout 2.
pub const RECOVER: usize = 9;
pub const SITE_UNIT_ROW: usize = 2;

/// The engine's fixed tables and scratch buffers, shared by every game in this worker.
pub struct Engine {
    pub no_peace: bool,
    pub rules: Rules,
    pub geom: Geom,
    pub layout: Layout,
    pub catalog: Catalog,
    /// Sorted rule-table ids of structures and units (rl/observation.py STRUCTURES, UNITS): the one-hot planes.
    pub structure_ids: Vec<i32>,
    pub unit_ids: Vec<i32>,
    pub scratch: Scratch,
    pub mscratch: MaskScratch,
    pub mask: Vec<u8>,
}

impl Engine {
    fn new(size: usize) -> Engine {
        let n = size * size;
        let table: Value = serde_json::from_str(RULES_JSON).expect("rule table");
        let ids = |group: &str| {
            let mut v: Vec<i32> = table[group].as_array().unwrap().iter().map(|r| r["id"].as_i64().unwrap() as i32).collect();
            v.sort();
            v.dedup();
            v
        };
        Engine {
            no_peace: true,
            rules: Rules::from_json(RULES_JSON).expect("rule table"),
            // rl/observation.py: STRUCTURES leave out 30, 49 and 51 and UNITS the kinds 46-55 (the
            // special tribes' rows), so every checkpoint keeps its one-hot width; version 6 adds them as extra channels
            structure_ids: ids("structures").into_iter().filter(|x| ![30, 49, 51].contains(x)).collect(),
            unit_ids: ids("units").into_iter().filter(|x| !(46..56).contains(x)).collect(),
            geom: Geom::new(size),
            layout: Layout::new(n),
            catalog: Catalog::new(n),
            scratch: Scratch::new(n),
            mscratch: MaskScratch::new(n),
            mask: vec![0; mask_width(n)],
        }
    }

    /// Legal catalog entries of the player to move (engine-rs legal_masks, flat catalog order).
    pub fn legal(&mut self, s: &State, no_resign: bool) -> &[u8] {
        legal_masks(s, &self.rules, &self.geom, polytopia_engine_rs::masks::MaskOpts { no_resign, no_peace: self.no_peace }, &mut self.mscratch, &mut self.mask);
        &self.mask
    }

    /// The site's action id of engine catalog entry `i`. The site keeps action layout 2 (the recover row of two entries
    /// per tile: Recover, Disembark), so saves, links and game files of every earlier release keep their ids. T3's
    /// engine (action layout 5) widens that row to UNIT_ROW entries; entries 2.. are the special
    /// tribes' unit abilities (FreezeArea .. Swarm), never legal for the 12 tribes the site offers: they have no site
    /// id (None). Every other family keeps its order, shifted by the row's extra width.
    pub fn site_id(&self, i: usize) -> Option<usize> {
        let (start, end) = (self.catalog.offset[RECOVER], self.catalog.offset[RECOVER + 1]);
        let extra = (UNIT_ROW - SITE_UNIT_ROW) * self.catalog.n;
        if i < start {
            Some(i)
        } else if i < end {
            let (tile, k) = ((i - start) / UNIT_ROW, (i - start) % UNIT_ROW);
            (k < SITE_UNIT_ROW).then_some(start + tile * SITE_UNIT_ROW + k)
        } else {
            (i < self.catalog.offset[crate::net::FAMILIES]).then_some(i - extra)
        }
    }

    /// The engine catalog entry of site action id `i` (site_id's inverse); None outside the site's catalog.
    pub fn engine_id(&self, i: usize) -> Option<usize> {
        let start = self.catalog.offset[RECOVER];
        let extra = (UNIT_ROW - SITE_UNIT_ROW) * self.catalog.n;
        let site_end = start + SITE_UNIT_ROW * self.catalog.n;
        if i < start {
            Some(i)
        } else if i < site_end {
            let (tile, k) = ((i - start) / SITE_UNIT_ROW, (i - start) % SITE_UNIT_ROW);
            Some(start + tile * UNIT_ROW + k)
        } else {
            (i + extra < self.catalog.offset[crate::net::FAMILIES]).then_some(i + extra)
        }
    }

    /// site_id as the JSON string the site uses; an entry with no site id is an error (special tribes only).
    pub fn site_str(&self, i: usize) -> Result<String, String> {
        self.site_id(i).map(|x| x.to_string()).ok_or_else(|| format!("catalog entry {i} has no site action id"))
    }

    /// The engine entry of a site action id given as text.
    pub fn engine_of(&self, a: &str) -> Result<usize, String> {
        let i: usize = a.parse().map_err(|_| "bad action id".to_string())?;
        self.engine_id(i).ok_or_else(|| "action index outside the catalog".to_string())
    }

    /// Applies catalog entry `index` (family/source/target decoded as rl.actions.unflatten + gpu.actions.decode).
    pub fn apply(&mut self, s: &mut State, index: usize) -> i32 {
        let (f, src, t) = self.catalog.unflatten(index);
        let a = decode(s, f as i32, src as i32, t as i32);
        let v = s.state_version;
        step(s, a, v, &self.rules, &self.geom, MAX_ACTIONS, &mut self.scratch)
    }
}

struct Ctx {
    /// the engine of the board size in use; `spare` keeps the others (select)
    engine: Engine,
    spare: Vec<Engine>,
    games: Vec<Option<State>>,
    policy: Option<Policy>,
    input: Vec<u8>,
    output: Vec<u8>,
}

thread_local! {
    static CTX: RefCell<Option<Ctx>> = const { RefCell::new(None) };
}

fn with<T>(f: impl FnOnce(&mut Ctx) -> T) -> T {
    CTX.with(|c| {
        let mut c = c.borrow_mut();
        let ctx = c.get_or_insert_with(|| Ctx { engine: Engine::new(SIZE), spare: Vec::new(), games: Vec::new(), policy: None, input: Vec::new(),
                                                 output: Vec::new() });
        f(ctx)
    })
}

/// Runs `f` and stores its JSON (or error) as the output; returns `ok` or -1.
fn respond(ok: i32, f: impl FnOnce(&mut Ctx) -> Result<Value, String>) -> i32 {
    with(|ctx| {
        let (code, value) = match f(ctx) {
            Ok(v) => (ok, v),
            Err(e) => (-1, json!({"error": e})),
        };
        ctx.output = serde_json::to_vec(&value).unwrap();
        code
    })
}

/// Makes ctx.engine (and the policy's buffers) the ones for a `size` x `size` board.
fn select(ctx: &mut Ctx, size: usize) -> Result<(), String> {
    if !SIZES.contains(&size) {
        return Err(format!("board size {size} is not one of {SIZES:?}"));
    }
    if ctx.engine.geom.size != size {
        let e = match ctx.spare.iter().position(|e| e.geom.size == size) {
            Some(i) => ctx.spare.swap_remove(i),
            None => Engine::new(size),
        };
        let mut old = std::mem::replace(&mut ctx.engine, e);
        ctx.engine.no_peace = old.no_peace;
        old.mask.shrink_to_fit();
        ctx.spare.push(old);
    }
    if let Some(p) = ctx.policy.as_mut() {
        p.resize(size);
    }
    Ok(())
}

/// Game `id`, with ctx.engine selected for its board.
fn game_sel(ctx: &mut Ctx, id: i32) -> Result<State, String> {
    let s = game(ctx, id)?.clone();
    select(ctx, s.size)?;
    Ok(s)
}

fn game(ctx: &mut Ctx, id: i32) -> Result<&mut State, String> {
    ctx.games.get_mut(id as usize).and_then(|g| g.as_mut()).ok_or_else(|| "unknown game".to_string())
}

fn status(s: &State) -> Value {
    json!({"current_player": s.current_player, "turn": s.turn, "state_version": s.state_version,
           "terminated": s.terminated, "truncated": s.truncated, "rewards": terminal_rewards(s)})
}

/// A buffer of `len` bytes for the next call's input.
#[no_mangle]
pub extern "C" fn input(len: usize) -> *mut u8 {
    with(|ctx| {
        ctx.input.clear();
        ctx.input.resize(len, 0);
        ctx.input.as_mut_ptr()
    })
}

#[no_mangle]
pub extern "C" fn output_ptr() -> *const u8 {
    with(|ctx| ctx.output.as_ptr())
}

#[no_mangle]
pub extern "C" fn output_len() -> usize {
    with(|ctx| ctx.output.len())
}

/// Loads the policy weights from the input buffer (scripts/export_browser_weights.py format).
#[no_mangle]
pub extern "C" fn load_model() -> i32 {
    respond(0, |ctx| {
        let net = Net::parse(&ctx.input)?;
        // Plato v3 (151), u88890 v5 (153) and T3 u96070 v6 (174) trained with peace; the older bots without
        let version = net.obs_version();
        let peace = matches!(net.board_channels(), 151 | 153 | 174);
        ctx.engine.no_peace = !peace;
        for e in ctx.spare.iter_mut() {
            e.no_peace = !peace;
        }
        let size = ctx.engine.geom.size;
        ctx.policy = Some(Policy::new(net, size));
        ctx.input = Vec::new();
        // "joint": this engine's bot_act has mode 2, the global joint top-1 (r17, T3); the page refuses a joint game
        // on an engine without it rather than let mode 2 fall back to the hierarchical argmax
        Ok(json!({"loaded": true, "obs_version": version, "joint": true}))
    })
}

/// A new game from mapgen.reset(seed); returns its id.
#[no_mangle]
pub extern "C" fn game_new(seed: i32) -> i32 {
    with(|ctx| {
        if select(ctx, SIZE).is_err() {
            return -1;
        }
        let s = mapgen::reset(seed, &ctx.engine.geom, None);
        let id = ctx.games.iter().position(|g| g.is_none()).unwrap_or(ctx.games.len());
        if id == ctx.games.len() {
            ctx.games.push(None);
        }
        ctx.games[id] = Some(s);
        id as i32
    })
}

/// Create a game from an exported training-generator layout. Browser saves retain
/// the actual layout so later map-bank updates cannot change an existing game.
#[no_mangle]
pub extern "C" fn game_new_layout(seed: i32) -> i32 {
    respond(0, |ctx| {
        let value: Value = serde_json::from_slice(&ctx.input).map_err(|e|e.to_string())?;
        let read = |name: &str, len: usize| -> Result<Vec<i32>,String> {
            let a = value[name].as_array().ok_or(format!("missing {name}"))?;
            if a.len() != len { return Err(format!("invalid {name} length")); }
            a.iter().map(|x|x.as_i64().and_then(|v|i32::try_from(v).ok()).ok_or(format!("invalid {name}"))).collect()
        };
        // (u88890): optional "size" (default 14) and "tribes" [seat 1, seat 2] (default Bardur, Bardur),
        // the 12 regular tribe types (TRIBES); the layout is a training-table row (maps/state.py stack_layouts)
        let size = match value.get("size") { None => SIZE, Some(v) => v.as_u64().ok_or("invalid size")? as usize };
        select(ctx, size)?;
        let n = size * size;
        let tribes = match value.get("tribes") { None => vec![4, 4], Some(_) => read("tribes", 2)? };
        if tribes.iter().any(|t| !TRIBES.contains(t)) {
            return Err(format!("tribes must be two of {TRIBES:?}"));
        }
        let terrain = read("terrain",n)?;
        let structure = read("structure",n)?;
        let resource = read("resource",n)?;
        let caps = read("capitals",2)?;
        if terrain.iter().any(|x|!(1..=5).contains(x)) || resource.iter().any(|x|!(0..=9).contains(x)) ||
            structure.iter().any(|x|![0,1,2,47].contains(x)) || caps[0] == caps[1] ||
            caps.iter().any(|&c|c<0 || c as usize>=n || terrain[c as usize]<3 || structure[c as usize]!=1) {
            return Err("invalid map layout".into());
        }
        // packed capitals: tile | tribe << 16 (engine-rs tribes.rs; a plain tile is Bardur)
        let caps: Vec<usize> = caps.into_iter().zip(&tribes).map(|(c, &t)| c as usize | (t as usize) << 16).collect();
        let state = mapgen::initial_state(seed as u32,&terrain,&structure,&resource,&caps,&ctx.engine.geom,mapgen::NO_ROUND_LIMIT);
        let id = ctx.games.iter().position(|g|g.is_none()).unwrap_or(ctx.games.len());
        if id == ctx.games.len() {ctx.games.push(None);}
        ctx.games[id] = Some(state);
        Ok(json!({"id":id}))
    })
}

/// Reference-parity probe: the mover's packed public view, exact encoded inputs
/// and policy outputs. Does not read an opponent's private view.
#[no_mangle]
pub extern "C" fn bot_probe(id: i32) -> i32 {
    respond(0, |ctx| {
        let s = game_sel(ctx,id)?;
        let size = s.size;
        let e = &ctx.engine;
        let policy = ctx.policy.as_mut().ok_or("model not loaded")?;
        let packed = view::packed(&s,s.current_player,&e.geom);
        let mut board = vec![0.; size*size*policy.net.board_channels()];
        let mut global = vec![0.;policy.net.global_channels()];
        view::features(&e.layout,&packed,size,&e.structure_ids,&e.unit_ids,&mut board,&mut global);
        let mut buffers = net::Buffers::new(size*size,policy.net.width);
        let t = policy.net.trunk(size,&board,&global,&mut buffers);
        Ok(json!({"packed":packed,"board":board,"global":global,"tiles":t.tiles,"g":t.g,
            "family":policy.net.family_logits(&t),"source":policy.net.source_logits(&t,0),
            "target":policy.net.target_logits(&t,0,0)}))
    })
}

/// The rule table the engine runs (src/polytopia_bot/gpu/rules.json), for the interface's names and costs.
#[no_mangle]
pub extern "C" fn rules_json() -> i32 {
    respond(0, |_| serde_json::from_str(RULES_JSON).map_err(|e| e.to_string()))
}

/// A copy of game `id` under a new id.
#[no_mangle]
pub extern "C" fn game_clone(id: i32) -> i32 {
    with(|ctx| {
        let Some(Some(s)) = ctx.games.get(id as usize).cloned() else { return -1 };
        let at = ctx.games.iter().position(|g| g.is_none()).unwrap_or(ctx.games.len());
        if at == ctx.games.len() {
            ctx.games.push(None);
        }
        ctx.games[at] = Some(s);
        at as i32
    })
}

/// A new game holding `viewer`'s public projection of game `id` (public.rs `public_state`: the viewer's own state
/// whole, everything else only as its public view shows it); -1 if `id` is not a game or `viewer` not a seat. The
/// site builds a live seat's legal actions, attack forecasts and threat view on it (11 Nov, r12): the
/// whole UI payload is then a function of the seat's public view. The true game still referees every action.
#[no_mangle]
pub extern "C" fn game_public(id: i32, viewer: i32) -> i32 {
    with(|ctx| {
        let Some(Some(s)) = ctx.games.get(id as usize) else { return -1 };
        if viewer < 1 || viewer as usize > s.np {
            return -1;
        }
        let p = crate::public::public_state(s, viewer);
        let at = ctx.games.iter().position(|g| g.is_none()).unwrap_or(ctx.games.len());
        if at == ctx.games.len() {
            ctx.games.push(None);
        }
        ctx.games[at] = Some(p);
        at as i32
    })
}

/// Every legal attack of `viewer` (the player to move) in game `id`, forecast over the public variants of its
/// projection (ui.rs `forecasts`: ranges; a number only when every world consistent with the view agrees).
#[no_mangle]
pub extern "C" fn game_forecast(id: i32, viewer: i32) -> i32 {
    respond(0, |ctx| {
        let s = game_sel(ctx, id)?;
        Ok(crate::ui::forecasts(&mut ctx.engine, &s, viewer))
    })
}

/// The threat view for `viewer` (the player to move) in game `id`: ui.rs `threats`, the union over the enemy's
/// provable-minimum and everything worlds. [[tile, moves, attacks]], the first row the danger row (tile -1).
#[no_mangle]
pub extern "C" fn game_threat(id: i32, viewer: i32) -> i32 {
    respond(0, |ctx| {
        let s = game_sel(ctx, id)?;
        Ok(crate::ui::threats(&mut ctx.engine, &s, viewer))
    })
}

/// The engine command {kind, source, target, parameter} that catalog entry `index` stands for in game `id`.
#[no_mangle]
pub extern "C" fn game_decode(id: i32, index: i32) -> i32 {
    respond(0, |ctx| {
        let s = game_sel(ctx, id)?;
        let s = &s;
        let index = if index < 0 { None } else { ctx.engine.engine_id(index as usize) };
        let Some(index) = index.filter(|&i| i < mask_width(s.n)) else {
            return Err("action index outside the catalog".into());
        };
        let (f, src, t) = ctx.engine.catalog.unflatten(index);
        let a = decode(s, f as i32, src as i32, t as i32);
        Ok(json!({"kind": a.kind, "source": a.source, "target": a.target, "parameter": a.parameter}))
    })
}

#[no_mangle]
pub extern "C" fn game_free(id: i32) {
    with(|ctx| {
        if let Some(g) = ctx.games.get_mut(id as usize) {
            *g = None;
        }
    })
}

#[no_mangle]
pub extern "C" fn game_status(id: i32) -> i32 {
    respond(0, |ctx| Ok(status(game(ctx, id)?)))
}

/// One frame view: `player` 1 or 2 is that seat's fogged observation, 0 the omniscient spectator view.
#[no_mangle]
pub extern "C" fn game_view(id: i32, player: i32) -> i32 {
    respond(0, |ctx| {
        let s = game_sel(ctx, id)?;
        let e = &ctx.engine;
        match player {
            1 | 2 => Ok(view::seat_view(&s, &e.rules, &e.geom, &e.layout, player)),
            0 => Ok(view::world_view(&s, &e.rules, &e.geom, &e.layout)),
            _ => Err("invalid seat".into()),
        }
    })
}

/// The mover's legal actions [{action_id, command}] (resignation included: this is the human's catalog).
#[no_mangle]
pub extern "C" fn game_legal(id: i32) -> i32 {
    respond(0, |ctx| {
        let s = game_sel(ctx, id)?;
        if s.terminated || s.truncated {
            return Ok(json!([]));
        }
        let e = &mut ctx.engine;
        let indices: Vec<usize> = e.legal(&s, false).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
        Ok(Value::Array(indices.into_iter().map(|i| {
            let (f, src, t) = e.catalog.unflatten(i);
            let a = decode(&s, f as i32, src as i32, t as i32);
            Ok(json!({"action_id": e.site_str(i)?,
                      "command": {"kind": a.kind, "source": a.source, "target": a.target, "parameter": a.parameter}}))
        }).collect::<Result<Vec<_>, String>>()?))
    })
}

/// Applies a legal catalog entry for the player to move; returns the engine status (0 = OK).
#[no_mangle]
pub extern "C" fn game_step(id: i32, index: i32) -> i32 {
    respond(0, |ctx| {
        let mut s = game_sel(ctx, id)?;
        if s.terminated || s.truncated {
            return Err("the game is over".into());
        }
        let e = &mut ctx.engine;
        let index = if index < 0 { None } else { e.engine_id(index as usize) };
        let Some(index) = index.filter(|&i| e.legal(&s, false).get(i) == Some(&1)) else {
            return Err("that action is not legal in the current state".into());
        };
        let code = e.apply(&mut s, index);
        if code != core::OK {
            return Err(format!("catalog/transition discrepancy: status {code}"));
        }
        let out = status(&s);
        *game(ctx, id)? = s;
        Ok(out)
    })
}

/// The revision of the bot's public projection (public.rs); the caller referees the one proposal. Engines without
/// this export compute the bot's candidates from the full state. 1: hidden tiles, units, economies and maps
/// (87caa3b8). 2: also every foreign-city internal (riot), tech-hidden resources, routes, ruling-city
/// links, sub-point health and foreign relation clocks; every State field classified (public.rs `fields` test).
/// The site lets the bot move only on revision 2 or later.
#[no_mangle]
pub extern "C" fn bot_public_candidates() -> i32 {
    2
}

/// The bot's candidate set for the player to move: the catalog entries the policy may propose, from the mover's
/// public view only (public.rs; resignation excluded as in bot_act). For audits: on two states with the same public
/// view it is the same list.
#[no_mangle]
pub extern "C" fn bot_candidates(id: i32) -> i32 {
    respond(0, |ctx| {
        let s = game_sel(ctx, id)?;
        if s.terminated || s.truncated {
            return Ok(json!([]));
        }
        let e = &mut ctx.engine;
        let p = crate::public::public_state(&s, s.current_player);
        let ids: Vec<usize> = e.legal(&p, true).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
        let ids: Vec<String> = ids.into_iter().map(|i| e.site_str(i)).collect::<Result<_, _>>()?;
        Ok(json!(ids))
    })
}

/// The policy's decision for the player to move: `deterministic` 0 samples with `seed`, 1 is the hierarchical argmax,
/// 2 the global joint top-1 (T3's reference chooser).
/// Output {action_id, value, log_probability}; does not apply it.
#[no_mangle]
pub extern "C" fn bot_act(id: i32, deterministic: i32, seed: u32) -> i32 {
    respond(0, |ctx| {
        let s = game_sel(ctx, id)?;
        let policy = ctx.policy.as_mut().ok_or("model not loaded")?;
        let mut rng = search::Rng::new(seed as u64);
        // r17 (T3): 0 sample, 1 hierarchical argmax, 2 global joint top-1 (the T3 reference chooser)
        if !(0..=2).contains(&deterministic) {
            return Err(format!("unknown bot mode {deterministic}"));
        }
        let d = policy.act_mode(&mut ctx.engine, &s, deterministic, &mut rng)?;
        Ok(json!({"action_id": ctx.engine.site_str(d.index)?, "value": d.value, "log_probability": d.logp}))
    })
}

/// Leak-free turn search for the player to move (rl/search.py, depth 1), all lanes on this thread: plans the rest
/// of its turn and returns {plan: [action_id...], plans, overrides, fallbacks, value}.
#[no_mangle]
pub extern "C" fn bot_plan(id: i32, lanes: i32, seed: u32) -> i32 {
    respond(0, |ctx| {
        let s = game_sel(ctx, id)?;
        let policy = ctx.policy.as_mut().ok_or("model not loaded")?;
        search::plan_turn(&mut ctx.engine, policy, &s, lanes.max(1) as usize, seed as u64)
    })
}

/// A game rebuilt from mapgen.reset(seed) and the JSON list of catalog entries in the input; returns its id.
#[no_mangle]
pub extern "C" fn game_from_actions(seed: i32) -> i32 {
    respond(0, |ctx| {
        let actions: Vec<String> = serde_json::from_slice(&ctx.input).map_err(|e| e.to_string())?;
        select(ctx, SIZE)?;
        let mut s = mapgen::reset(seed, &ctx.engine.geom, None);
        for a in &actions {
            let index = ctx.engine.engine_of(a)?;
            if ctx.engine.apply(&mut s, index) != core::OK {
                return Err(format!("action {a} does not replay"));
            }
        }
        let id = ctx.games.iter().position(|g| g.is_none()).unwrap_or(ctx.games.len());
        if id == ctx.games.len() {
            ctx.games.push(None);
        }
        ctx.games[id] = Some(s);
        Ok(json!({"id": id}))
    })
}

/// Search lanes [from, to) of plan round `round` for the player to move in game `id`, on the round's world
/// (sampled when `sample`): [{lane, actions, score, steps}]. Any worker computes the same world.
#[no_mangle]
pub extern "C" fn search_lanes(id: i32, round: u32, sample: i32, seed: u32, from: i32, to: i32) -> i32 {
    respond(0, |ctx| {
        let s = game_sel(ctx, id)?;
        let policy = ctx.policy.as_mut().ok_or("model not loaded")?;
        let w = search::world(&ctx.engine, &s, sample != 0, seed as u64, round as u64);
        let mut out = Vec::new();
        for lane in from.max(0)..to {
            let mut rng = search::Rng::new(search::mix(seed as u64, round as u64, lane as u64));
            let l = search::run_lane(&mut ctx.engine, policy, &w, lane == 0, &mut rng)?;
            let actions: Vec<String> = l.actions.iter().map(|&i| ctx.engine.site_str(i)).collect::<Result<_, _>>()?;
            out.push(json!({"lane": lane, "actions": actions,
                            "score": l.score, "steps": l.steps}));
        }
        Ok(Value::Array(out))
    })
}

/// Commits round `round`'s chosen turn (JSON list of catalog entries in the input) to game `id`: guarded
/// against the round's world, or only its first entry when the round searched the true state. {applied: [...]}.
#[no_mangle]
pub extern "C" fn search_commit(id: i32, round: u32, sample: i32, seed: u32) -> i32 {
    respond(0, |ctx| {
        let chosen: Vec<String> = serde_json::from_slice(&ctx.input).map_err(|e| e.to_string())?;
        game_sel(ctx, id)?; // the game's board size selects the catalog the ids map through
        let chosen: Vec<usize> = chosen.iter().map(|a| ctx.engine.engine_of(a)).collect::<Result<_, _>>()?;
        if chosen.is_empty() {
            return Err("empty plan".into());
        }
        let mut s = game_sel(ctx, id)?;
        let w = search::world(&ctx.engine, &s, sample != 0, seed as u64, round as u64);
        let applied = search::commit_round(&mut ctx.engine, &mut s, &w, sample != 0, &chosen)?;
        *game(ctx, id)? = s;
        let applied: Vec<String> = applied.iter().map(|&i| ctx.engine.site_str(i)).collect::<Result<_, _>>()?;
        Ok(json!({"applied": applied}))
    })
}

#[cfg(test)]
mod parity {
    //! Browser vs TRAINING parity for u88890 (obs v5) at sizes 11-20. The fixture is made from the
    //! training source (fixture not published): each case replays its
    //! actions from the table layout here, then compares the mover's features and greedy heads.
    //!     GEN_WASM_FIXTURE=<dir> GEN_WASM_WEIGHTS=<u88890 .bin> cargo test --release --target aarch64-apple-darwin --lib
    use super::*;
    use crate::net::{argmax, log_distribution};

    fn f32s(path: &std::path::Path) -> Vec<f32> {
        std::fs::read(path).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    fn maxdiff(a: &[f32], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (*x as f64 - y).abs()).fold(0., f64::max)
    }

    #[test]
    fn u88890_matches_training_at_every_size() {
        let (Ok(dir), Ok(weights)) = (std::env::var("GEN_WASM_FIXTURE"), std::env::var("GEN_WASM_WEIGHTS")) else {
            eprintln!("GEN_WASM_FIXTURE / GEN_WASM_WEIGHTS not set: skipped");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let fixture: Value = serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
        let net = Net::parse(&std::fs::read(weights).unwrap()).unwrap();
        assert_eq!(net.obs_version(), 5);
        let mut policy = Policy::new(net, SIZE);
        let (mut checked, mut agree, mut worst) = (0, 0, 0f64);
        let mut sizes = std::collections::BTreeSet::new();
        for case in fixture["cases"].as_array().unwrap() {
            let size = case["size"].as_u64().unwrap() as usize;
            let mut e = Engine::new(size);
            e.no_peace = false;
            policy.resize(size);
            let lay = &case["layout"];
            let ints = |k: &str| lay[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect::<Vec<_>>();
            let tribes = ints("tribes");
            let caps: Vec<usize> = ints("capitals").iter().zip(&tribes).map(|(&c, &t)| c as usize | (t as usize) << 16).collect();
            let mut s = mapgen::initial_state(case["seed"].as_u64().unwrap() as u32, &ints("terrain"), &ints("structure"),
                                              &ints("resource"), &caps, &e.geom, 200);
            let actions = case["actions"].as_array().unwrap();
            let mut played = 0;
            for check in case["checks"].as_array().unwrap() {
                let after = check["after"].as_u64().unwrap() as usize;
                while played < after {
                    let a: Vec<usize> = actions[played].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
                    let index = e.catalog.flatten(a[0], a[1], a[2]);
                    assert_eq!(e.legal(&s, true)[index], 1, "{} action {played} not legal", case["name"]);
                    assert_eq!(e.apply(&mut s, index), core::OK);
                    played += 1;
                }
                assert_eq!(s.current_player as u64, check["current_player"].as_u64().unwrap());
                let name = format!("{}-{}", case["name"].as_str().unwrap(), check["k"]);
                let board = f32s(&dir.join(format!("{name}.board.f32")));
                let global = f32s(&dir.join(format!("{name}.global.f32")));
                let v = view::packed(&s, s.current_player, &e.geom);
                let (mut b, mut g) = (vec![0.; size * size * 153], vec![0.; 555]);
                view::features(&e.layout, &v, size, &e.structure_ids, &e.unit_ids, &mut b, &mut g);
                let bad: Vec<(usize, usize, f32, f32)> = b.iter().zip(&board).enumerate().filter(|(_, (x, y))| x != y)
                    .map(|(i, (x, y))| (i / 153, i % 153, *x, *y)).take(8).collect();
                assert!(bad.is_empty(), "{name}: board (tile, channel, wasm, training) {bad:?}");
                // entry 1 is ln_1p(stars) / 8: XLA's log1p and Rust's libm differ by an ulp at some star counts
                let bad: Vec<(usize, f32, f32)> = g.iter().zip(&global).enumerate()
                    .filter(|(i, (x, y))| if *i == 1 { (*x - *y).abs() > 1e-6 } else { x != y })
                    .map(|(i, (x, y))| (i, *x, *y)).take(8).collect();
                assert!(bad.is_empty(), "{name}: global (entry, wasm, training) {bad:?}");
                let t = policy.trunk(&e, &s, s.current_player);
                let mask = e.legal(&s, true).to_vec();
                let c = &e.catalog;
                let get = |k: &str| check[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<_>>();
                let fl = policy.net.family_logits(&t);
                let f = check["family"].as_u64().unwrap() as usize;
                let sl = policy.net.source_logits(&t, f);
                let src = check["source"].as_u64().unwrap() as usize;
                let tl = policy.net.target_logits(&t, f, src);
                let d = maxdiff(&fl, &get("family_logits")).max(maxdiff(&sl, &get("source_logits")))
                    .max(maxdiff(&tl, &get("target_logits")));
                worst = worst.max(d);
                let mine = (argmax(&log_distribution(&fl, &c.family_mask(&mask))),
                            argmax(&log_distribution(&sl, &c.source_mask(&mask, f))),
                            argmax(&log_distribution(&tl, &c.target_mask(&mask, f, src))));
                let theirs = (f, src, check["target"].as_u64().unwrap() as usize);
                agree += (mine == theirs) as usize;
                checked += 1;
                sizes.insert(size);
                eprintln!("{name}: size {size} logits max |diff| {d:.2e} greedy {mine:?} vs {theirs:?}");
            }
        }
        eprintln!("checked {checked}, greedy agree {agree}, worst logit diff {worst:.3e}, sizes {sizes:?}");
        assert_eq!(sizes.len(), 5);
        assert_eq!(agree, checked);
        assert!(worst < 1e-3, "{worst}");
    }
}

#[cfg(test)]
mod bank {
    //! Every board of the u88890 site bank (scripts/export_browser_bank_v5.py) opens through game_new_layout, has
    //! legal moves, and (with the weights) the bot moves on it.
    //!     GEN_WASM_BANK=<dir> [GEN_WASM_WEIGHTS=<.bin>] cargo test --release --target aarch64-apple-darwin --lib bank
    use super::*;

    fn call(input: &[u8], f: impl FnOnce() -> i32) -> (i32, Value) {
        with(|ctx| ctx.input = input.to_vec());
        let code = f();
        (code, with(|ctx| serde_json::from_slice(&ctx.output).unwrap()))
    }

    #[test]
    fn every_bank_board_opens_and_plays() {
        let Ok(dir) = std::env::var("GEN_WASM_BANK") else { eprintln!("GEN_WASM_BANK not set: skipped"); return };
        if let Ok(w) = std::env::var("GEN_WASM_WEIGHTS") {
            let (code, out) = call(&std::fs::read(w).unwrap(), || load_model());
            assert_eq!(code, 0, "{out}");
            assert!(matches!(out["obs_version"].as_u64(), Some(5 | 6)), "{out}");
        }
        let mut opened = 0;
        for size in SIZES {
            let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{dir}/bank-{size}.json")).unwrap()).unwrap();
            for (k, row) in rows.iter().enumerate() {
                let seed = row["seed"].as_i64().unwrap() as i32;
                let (code, out) = call(&serde_json::to_vec(row).unwrap(), || game_new_layout(seed));
                assert_eq!(code, 0, "{size} {k}: {out}");
                let id = out["id"].as_i64().unwrap() as i32;
                let (code, legal) = call(&[], || game_legal(id));
                assert!(code == 0 && !legal.as_array().unwrap().is_empty(), "{size} {k}");
                let (code, view) = call(&[], || game_view(id, 1));
                assert_eq!(code, 0);
                assert_eq!(view["visible_state"]["board_size"].as_u64(), Some(size as u64));
                assert_eq!(view["visible_state"]["lighthouses"].as_array().unwrap().len(), 4);
                if k == 0 && std::env::var("GEN_WASM_WEIGHTS").is_ok() {
                    for _ in 0..6 {
                        let (code, d) = call(&[], || bot_act(id, 1, 7));
                        assert_eq!(code, 0, "{d}");
                        let index: i32 = d["action_id"].as_str().unwrap().parse().unwrap();
                        let (code, st) = call(&[], || game_step(id, index));
                        assert_eq!(code, 0, "{st}");
                    }
                }
                game_free(id);
                opened += 1;
            }
        }
        eprintln!("opened {opened} bank boards");
        assert!(opened >= 2500);
    }
}

#[cfg(test)]
mod t3 {
    //! T3 (u96070, obs v6, 2026-11-11) on the site. Without weights: the v6 features of every regular-tribe game are
    //! the v5 features plus zeros. With them: the loader reports version 6, and bot_act mode 2 is the global joint
    //! top-1 over the public candidates, which hierarchical argmax is not.
    //!     GEN_WASM_BANK=<dir> [GEN_WASM_T3=<u96070.bin>] cargo test --release --target aarch64-apple-darwin --lib t3
    use super::*;

    fn bank_games(dir: &str, size: usize, take: usize) -> Vec<(Engine, State)> {
        let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{dir}/bank-{size}.json")).unwrap()).unwrap();
        rows.iter().take(take).map(|row| {
            let mut e = Engine::new(size);
            e.no_peace = false;
            let ints = |k: &str| row[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect::<Vec<_>>();
            let tribes = ints("tribes");
            let caps: Vec<usize> = ints("capitals").iter().zip(&tribes).map(|(&c, &t)| c as usize | (t as usize) << 16).collect();
            let s = mapgen::initial_state(row["seed"].as_u64().unwrap() as u32, &ints("terrain"), &ints("structure"),
                                          &ints("resource"), &caps, &e.geom, 200);
            (e, s)
        }).collect()
    }

    /// The site keeps action layout 2: site_id/engine_id are inverse on every site id, cover the whole engine
    /// catalog but the special tribes' unit abilities, and keep the catalog order. On the bank's regular-tribe games
    /// every legal entry (resignation included) has a site id, and it decodes to the same command.
    #[test]
    fn site_action_ids_keep_layout_2() {
        for size in SIZES {
            let e = Engine::new(size);
            let n = size * size;
            let width = mask_width(n);
            let site_width = width - (UNIT_ROW - SITE_UNIT_ROW) * n;
            let mut last = None;
            for i in 0..width {
                match e.site_id(i) {
                    Some(x) => {
                        assert!(x < site_width && e.engine_id(x) == Some(i), "size {size} entry {i}");
                        assert!(last.map_or(true, |l| x > l), "order at {i}");
                        last = Some(x);
                    }
                    None => {
                        let (f, _, k) = e.catalog.unflatten(i);
                        assert!(f == RECOVER && k >= SITE_UNIT_ROW, "only special abilities lack a site id: {i}");
                    }
                }
            }
            assert_eq!(last, Some(site_width - 1));
            assert_eq!(e.engine_id(site_width), None);
            // the end turn entry (last family) keeps its layout-2 id: width - 7n
            assert_eq!(e.site_id(width - 1), Some(site_width - 1));
        }
        let Ok(dir) = std::env::var("GEN_WASM_BANK") else { eprintln!("GEN_WASM_BANK not set: bank part skipped"); return };
        let (mut r, mut checked) = (0x9e37_79b9_7f4a_7c15u64, 0usize);
        for size in SIZES {
            for (mut e, mut s) in bank_games(&dir, size, 2) {
                for _ in 0..200 {
                    if s.terminated || s.truncated { break; }
                    let legal: Vec<usize> = e.legal(&s, false).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
                    for &i in &legal {
                        let x = e.site_id(i).expect("a regular-tribe legal entry has a site id");
                        assert_eq!(e.engine_id(x), Some(i));
                        checked += 1;
                    }
                    random_step(&mut e, &mut s, &mut r);
                }
            }
        }
        eprintln!("site action ids: {checked} legal entries of regular-tribe games, every one with a site id");
        assert!(checked > 10_000);
    }

    fn random_step(e: &mut Engine, s: &mut State, r: &mut u64) {
        let legal: Vec<usize> = e.legal(s, true).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
        *r ^= *r << 13; *r ^= *r >> 7; *r ^= *r << 17;
        assert_eq!(e.apply(s, legal[(*r % legal.len() as u64) as usize]), core::OK);
    }

    #[test]
    fn v6_features_are_v5_features_plus_zeros() {
        let Ok(dir) = std::env::var("GEN_WASM_BANK") else { eprintln!("GEN_WASM_BANK not set: skipped"); return };
        let (mut r, mut positions) = (0x9e37_79b9_7f4a_7c15u64, 0);
        for size in SIZES {
            for (mut e, mut s) in bank_games(&dir, size, 3) {
                for step in 0..400 {
                    if s.terminated || s.truncated { break; }
                    if step % 5 == 0 {
                        let n = size * size;
                        let v = view::packed(&s, s.current_player, &e.geom);
                        let (mut b5, mut g5, mut b6, mut g6) = (vec![0.; n * 153], vec![0.; 555], vec![1.; n * 174], vec![1.; 556]);
                        view::features(&e.layout, &v, size, &e.structure_ids, &e.unit_ids, &mut b5, &mut g5);
                        view::features(&e.layout, &v, size, &e.structure_ids, &e.unit_ids, &mut b6, &mut g6);
                        for i in 0..n {
                            assert!(b6[i * 174..i * 174 + 153] == b5[i * 153..(i + 1) * 153], "size {size} step {step} tile {i}");
                            assert!(b6[i * 174 + 153..(i + 1) * 174].iter().all(|&x| x == 0.), "special channels: size {size} tile {i}");
                        }
                        assert!(g6[..555] == g5[..] && g6[555] == 0., "global: size {size} step {step}");
                        positions += 1;
                    }
                    random_step(&mut e, &mut s, &mut r);
                }
            }
        }
        eprintln!("v6 == v5 + zeros at {positions} positions");
        assert!(positions > 500);
    }

    #[test]
    fn joint_top1_is_the_first_maximum_over_the_public_candidates() {
        let (Ok(dir), Ok(weights)) = (std::env::var("GEN_WASM_BANK"), std::env::var("GEN_WASM_T3")) else {
            eprintln!("GEN_WASM_BANK / GEN_WASM_T3 not set: skipped");
            return;
        };
        let bytes = std::fs::read(&weights).unwrap();
        let net = Net::parse(&bytes).unwrap();
        assert_eq!((net.board_channels(), net.global_channels(), net.obs_version()), (174, 556, 6));
        with(|ctx| ctx.input = bytes.clone());
        assert_eq!(load_model(), 0);
        let out: Value = with(|ctx| serde_json::from_slice(&ctx.output).unwrap());
        assert_eq!((out["obs_version"].as_u64(), out["joint"].as_bool()), (Some(6), Some(true)), "{out}");
        // bot_act: modes 0-2 only
        let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(format!("{dir}/bank-11.json")).unwrap()).unwrap();
        with(|ctx| ctx.input = serde_json::to_vec(&rows[0]).unwrap());
        assert_eq!(game_new_layout(rows[0]["seed"].as_i64().unwrap() as i32), 0);
        let id = with(|ctx| serde_json::from_slice::<Value>(&ctx.output).unwrap())["id"].as_i64().unwrap() as i32;
        for (mode, ok) in [(0, true), (1, true), (2, true), (3, false), (-1, false)] {
            assert_eq!(bot_act(id, mode, 5) == 0, ok, "mode {mode}");
        }
        game_free(id);
        let mut policy = Policy::new(net, 11);
        let (mut r, mut positions, mut differs, mut candidates, mut fallbacks) = (0x2545_f491_4f6c_dd1du64, 0, 0, 0, 0usize);
        for size in SIZES {
            policy.resize(size);
            for (mut e, mut s) in bank_games(&dir, size, 2) {
                for step in 0..300 {
                    if s.terminated || s.truncated { break; }
                    if step % 6 == 0 {
                        let cp = s.current_player;
                        let t = policy.trunk(&e, &s, cp);
                        // the reference chooser's catalogue: the public legal ids minus Diplomacy params 18-22 (joint_catalogue)
                        let mask = e.legal(&public::public_state(&s, cp), true).to_vec();
                        let (mask, fell_back) = policy::joint_catalogue_fallback(&e, &s, mask);
                        fallbacks += fell_back as usize;
                        let all = policy.joint_logp(&e.catalog, &t, &mask);
                        assert!(all.windows(2).all(|w| w[0].0 < w[1].0), "ascending flat ids");
                        assert_eq!(all.len(), mask.iter().filter(|&&m| m != 0).count());
                        let mut best = 0;
                        for k in 1..all.len() { if all[k].1.exp() > all[best].1.exp() { best = k; } }
                        let j = policy.act_mode(&mut e, &s, 2, &mut search::Rng::new(1)).unwrap();
                        assert_eq!((j.index, j.logp.to_bits()), (all[best].0, all[best].1.to_bits()), "size {size} step {step}");
                        let total: f64 = all.iter().map(|(_, lp)| (*lp as f64).exp()).sum();
                        assert!((total - 1.).abs() < 1e-3, "joint distribution sums to {total}");
                        let g = policy.act_mode(&mut e, &s, 1, &mut search::Rng::new(1)).unwrap();
                        differs += (g.index != j.index) as usize;
                        candidates += all.len();
                        positions += 1;
                    }
                    random_step(&mut e, &mut s, &mut r);
                }
            }
        }
        eprintln!("joint top-1 checked at {positions} positions ({candidates} candidates); hierarchical argmax differs at {differs}; forced peace answers (unfiltered fallback) {fallbacks}");
        assert!(positions > 100);
    }
}

#[cfg(test)]
mod t3_native {
    //! 12 Nov: browser T3 vs NATIVE T3, bit for bit where the arithmetic allows. The fixture
    //! (not published) comes from the native reference path for T3: the pinned T3
    //! native engine, the mover's public legal masks and own view, and the frozen joint chooser (Diplomacy 18-22
    //! dropped, no resign, global joint top-1). Both seats play T3; each case replays its actions from the layout.
    //! Per check: features byte-equal (global[1] = ln_1p(stars)/8 within 1e-6, XLA vs libm), public candidates equal,
    //! the filtered catalogue equal, every candidate's joint logp close, and the chosen top-1 EQUAL.
    //!     GEN_WASM_T3_FIXTURE=<dir> GEN_WASM_T3=<u96070.bin> cargo test --release --target aarch64-apple-darwin --lib t3_native
    use super::*;
    use crate::policy::joint_catalogue;

    fn f32s(path: &std::path::Path) -> Vec<f32> {
        std::fs::read(path).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    #[test]
    fn t3_matches_native_at_every_size() {
        let (Ok(dir), Ok(weights)) = (std::env::var("GEN_WASM_T3_FIXTURE"), std::env::var("GEN_WASM_T3")) else {
            eprintln!("GEN_WASM_T3_FIXTURE / GEN_WASM_T3 not set: skipped");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let fixture: Value = serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
        let net = Net::parse(&std::fs::read(weights).unwrap()).unwrap();
        assert_eq!(net.obs_version(), 6);
        let mut policy = Policy::new(net, SIZE);
        let (mut checks, mut top1_equal, mut worst_lp, mut filtered, mut closest) = (0usize, 0usize, 0f64, 0usize, f64::MAX);
        let mut sizes = std::collections::BTreeSet::new();
        let mut maps = std::collections::BTreeSet::new();
        let mut report = Vec::new();
        for case in fixture["cases"].as_array().unwrap() {
            let size = case["size"].as_u64().unwrap() as usize;
            let name = case["name"].as_str().unwrap();
            let mut e = Engine::new(size);
            e.no_peace = false;
            policy.resize(size);
            let lay = &case["layout"];
            let ints = |k: &str| lay[k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect::<Vec<_>>();
            let tribes = ints("tribes");
            let caps: Vec<usize> = ints("capitals").iter().zip(&tribes).map(|(&c, &t)| c as usize | (t as usize) << 16).collect();
            let mut s = mapgen::initial_state(case["seed"].as_u64().unwrap() as u32, &ints("terrain"), &ints("structure"),
                                              &ints("resource"), &caps, &e.geom, 200);
            let actions = case["actions"].as_array().unwrap();
            let mut played = 0;
            for check in case["checks"].as_array().unwrap() {
                let after = check["after"].as_u64().unwrap() as usize;
                while played < after {
                    let a = &actions[played];
                    let index = if let Some(c) = a.get("choice") {
                        assert!(a["refused"].as_bool() == Some(true));
                        let _ = c;
                        e.catalog.flatten(15, 0, 0) // the native public referee refused it: a forced end of turn
                    } else {
                        let v: Vec<usize> = a.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
                        e.catalog.flatten(v[0], v[1], v[2])
                    };
                    assert_eq!(e.legal(&s, true)[index], 1, "{name}: action {played} not legal here");
                    assert_eq!(e.apply(&mut s, index), core::OK, "{name}: action {played}");
                    played += 1;
                }
                let k = check["k"].as_u64().unwrap();
                let cp = s.current_player;
                assert_eq!(cp as u64, check["current_player"].as_u64().unwrap(), "{name}-{k}: mover");
                // features (the mover's own view)
                let board = f32s(&dir.join(format!("{name}-{k}.board.f32")));
                let global = f32s(&dir.join(format!("{name}-{k}.global.f32")));
                let v = view::packed(&s, cp, &e.geom);
                let (mut b, mut g) = (vec![0.; size * size * 174], vec![0.; 556]);
                view::features(&e.layout, &v, size, &e.structure_ids, &e.unit_ids, &mut b, &mut g);
                let bad: Vec<(usize, usize, f32, f32)> = b.iter().zip(&board).enumerate().filter(|(_, (x, y))| x != y)
                    .map(|(i, (x, y))| (i / 174, i % 174, *x, *y)).take(8).collect();
                assert!(bad.is_empty(), "{name}-{k}: board (tile, channel, wasm, native) {bad:?}");
                let bad: Vec<(usize, f32, f32)> = g.iter().zip(&global).enumerate()
                    .filter(|(i, (x, y))| if *i == 1 { (*x - *y).abs() > 1e-6 } else { x != y })
                    .map(|(i, (x, y))| (i, *x, *y)).take(8).collect();
                assert!(bad.is_empty(), "{name}-{k}: global (entry, wasm, native) {bad:?}");
                // candidates: public legal, then the reference filter
                // compared as (family, source, target): the network's coordinates (flat ids depend on each catalogue's
                // family widths; the native T3 catalogue has a wider family before End turn, the same triples)
                let public = e.legal(&public::public_state(&s, cp), true).to_vec();
                let triples = |m: &[u8]| { let mut v: Vec<Vec<u64>> = m.iter().enumerate().filter(|(_, &x)| x != 0)
                    .map(|(i, _)| { let (f, a, t) = e.catalog.unflatten(i); vec![f as u64, a as u64, t as u64] }).collect(); v.sort(); v };
                let want = |key: &str| { let mut v: Vec<Vec<u64>> = check[key].as_array().unwrap().iter()
                    .map(|c| c.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()).collect(); v.sort(); v };
                assert_eq!(triples(&public), want("public_choices"), "{name}-{k}: public candidates");
                let mask = joint_catalogue(&e, &s, public);
                assert_eq!(triples(&mask), want("candidate_choices"), "{name}-{k}: filtered catalogue");
                filtered += check["filtered_out"].as_array().unwrap().len();
                // every candidate's joint logp, then the top-1 the site plays
                let t = policy.trunk(&e, &s, cp);
                let mine: std::collections::HashMap<Vec<u64>, f32> = policy.joint_logp(&e.catalog, &t, &mask).into_iter()
                    .map(|(i, lp)| { let (f, a, t) = e.catalog.unflatten(i); (vec![f as u64, a as u64, t as u64], lp) }).collect();
                let theirs: Vec<(Vec<u64>, f64)> = check["candidate_choices"].as_array().unwrap().iter()
                    .zip(check["logp"].as_array().unwrap())
                    .map(|(c, lp)| (c.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect(), lp.as_f64().unwrap())).collect();
                assert_eq!(mine.len(), theirs.len());
                let d = theirs.iter().map(|(c, lp)| (mine[c] as f64 - lp).abs()).fold(0., f64::max);
                worst_lp = worst_lp.max(d);
                let pick = policy.act_mode(&mut e, &s, 2, &mut search::Rng::new(0)).unwrap();
                let top1: Vec<u64> = check["top1_choice"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
                let (pf, ps, pt) = e.catalog.unflatten(pick.index);
                let picked = vec![pf as u64, ps as u64, pt as u64];
                top1_equal += (picked == top1) as usize;
                if let Some(gap) = check["top2_gap"].as_f64() { closest = closest.min(gap); }
                checks += 1;
                sizes.insert(size);
                maps.insert(case["map_type"].as_str().unwrap().to_string());
                report.push(format!("{name}-{k}: turn {} candidates {} top1 wasm {picked:?} native {top1:?} logp max|diff| {d:.2e}",
                                    check["turn"], mine.len()));
            }
        }
        for line in &report { eprintln!("{line}"); }
        eprintln!("T3-NATIVE-PARITY checks {checks}, top-1 equal {top1_equal}, worst joint logp diff {worst_lp:.3e}, \
                   smallest native top-2 gap {closest:.3e}, diplomacy candidates filtered {filtered}, sizes {sizes:?}, maps {maps:?}");
        assert_eq!(sizes.len(), 5);
        assert!(maps.contains("lakes"));
        assert_eq!(top1_equal, checks, "top-1 must equal native at every check");
        assert!(worst_lp < 1e-3, "{worst_lp}");
    }
}

