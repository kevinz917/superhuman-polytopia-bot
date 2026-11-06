//! The game engine and the trained policy in the browser: engine-rs compiled to WebAssembly plus a CPU port of
//! the policy network, behind a small C ABI (no wasm-bindgen). JavaScript writes inputs through `input`, calls an
//! export, and reads JSON results through `output_ptr` / `output_len`. Errors return a negative code and leave
//! {"error": ...} in the output.

pub mod net;
pub mod policy;
pub mod search;
pub mod view;

use std::cell::RefCell;

use polytopia_engine_rs::core::{self, decode, step, terminal_rewards, Scratch};
use polytopia_engine_rs::geom::Geom;
use polytopia_engine_rs::mapgen;
use polytopia_engine_rs::masks::{legal_masks, mask_width, MaskScratch};
use polytopia_engine_rs::rules::Rules;
use polytopia_engine_rs::state::State;
use serde_json::{json, Value};

use crate::net::Net;
use crate::policy::{Catalog, Policy};
use crate::view::Layout;

pub const SIZE: usize = 14;
/// GameBatch's default action cap (engine-rs), the same as the training actors and the Studio engine.
pub const MAX_ACTIONS: i32 = 10000;
const RULES_JSON: &str = include_str!("../../src/polytopia_bot/gpu/rules.json");

/// The engine's fixed tables and scratch buffers, shared by every game in this worker.
pub struct Engine {
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
    fn new() -> Engine {
        let n = SIZE * SIZE;
        let table: Value = serde_json::from_str(RULES_JSON).expect("rule table");
        let ids = |group: &str| {
            let mut v: Vec<i32> = table[group].as_array().unwrap().iter().map(|r| r["id"].as_i64().unwrap() as i32).collect();
            v.sort();
            v.dedup();
            v
        };
        Engine {
            rules: Rules::from_json(RULES_JSON).expect("rule table"),
            structure_ids: ids("structures"),
            unit_ids: ids("units"),
            geom: Geom::new(SIZE),
            layout: Layout::new(n),
            catalog: Catalog::new(n),
            scratch: Scratch::new(n),
            mscratch: MaskScratch::new(n),
            mask: vec![0; mask_width(n)],
        }
    }

    /// Legal catalog entries of the player to move (engine-rs legal_masks, flat catalog order).
    pub fn legal(&mut self, s: &State, no_resign: bool) -> &[u8] {
        legal_masks(s, &self.rules, &self.geom, no_resign, &mut self.mscratch, &mut self.mask);
        &self.mask
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
    engine: Engine,
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
        let ctx = c.get_or_insert_with(|| Ctx { engine: Engine::new(), games: Vec::new(), policy: None, input: Vec::new(),
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
        ctx.policy = Some(Policy::new(net, SIZE));
        ctx.input = Vec::new();
        Ok(json!({"loaded": true}))
    })
}

/// A new game from mapgen.reset(seed); returns its id.
#[no_mangle]
pub extern "C" fn game_new(seed: i32) -> i32 {
    with(|ctx| {
        let s = mapgen::reset(seed, &ctx.engine.geom, None);
        let id = ctx.games.iter().position(|g| g.is_none()).unwrap_or(ctx.games.len());
        if id == ctx.games.len() {
            ctx.games.push(None);
        }
        ctx.games[id] = Some(s);
        id as i32
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

/// The engine command {kind, source, target, parameter} that catalog entry `index` stands for in game `id`.
#[no_mangle]
pub extern "C" fn game_decode(id: i32, index: i32) -> i32 {
    respond(0, |ctx| {
        let s = ctx.games.get(id as usize).and_then(|g| g.as_ref()).ok_or("unknown game")?;
        if index < 0 || index as usize >= mask_width(s.n) {
            return Err("action index outside the catalog".into());
        }
        let (f, src, t) = ctx.engine.catalog.unflatten(index as usize);
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
        let s = game(ctx, id)?.clone();
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
        let s = game(ctx, id)?.clone();
        if s.terminated || s.truncated {
            return Ok(json!([]));
        }
        let e = &mut ctx.engine;
        let indices: Vec<usize> = e.legal(&s, false).iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i).collect();
        Ok(Value::Array(indices.into_iter().map(|i| {
            let (f, src, t) = e.catalog.unflatten(i);
            let a = decode(&s, f as i32, src as i32, t as i32);
            json!({"action_id": i.to_string(),
                   "command": {"kind": a.kind, "source": a.source, "target": a.target, "parameter": a.parameter}})
        }).collect()))
    })
}

/// Applies a legal catalog entry for the player to move; returns the engine status (0 = OK).
#[no_mangle]
pub extern "C" fn game_step(id: i32, index: i32) -> i32 {
    respond(0, |ctx| {
        let mut s = game(ctx, id)?.clone();
        if s.terminated || s.truncated {
            return Err("the game is over".into());
        }
        let e = &mut ctx.engine;
        if index < 0 || e.legal(&s, false).get(index as usize) != Some(&1) {
            return Err("that action is not legal in the current state".into());
        }
        let code = e.apply(&mut s, index as usize);
        if code != core::OK {
            return Err(format!("catalog/transition discrepancy: status {code}"));
        }
        let out = status(&s);
        *game(ctx, id)? = s;
        Ok(out)
    })
}

/// The policy's decision for the player to move: argmax when `deterministic`, else sampled with `seed`.
/// Output {action_id, value, log_probability}; does not apply it.
#[no_mangle]
pub extern "C" fn bot_act(id: i32, deterministic: i32, seed: u32) -> i32 {
    respond(0, |ctx| {
        let s = game(ctx, id)?.clone();
        let policy = ctx.policy.as_mut().ok_or("model not loaded")?;
        let mut rng = search::Rng::new(seed as u64);
        let d = policy.act(&mut ctx.engine, &s, deterministic != 0, &mut rng)?;
        Ok(json!({"action_id": d.index.to_string(), "value": d.value, "log_probability": d.logp}))
    })
}

/// Leak-free turn search for the player to move (rl/search.py, depth 1), all lanes on this thread: plans the rest
/// of its turn and returns {plan: [action_id...], plans, overrides, fallbacks, value}.
#[no_mangle]
pub extern "C" fn bot_plan(id: i32, lanes: i32, seed: u32) -> i32 {
    respond(0, |ctx| {
        let s = game(ctx, id)?.clone();
        let policy = ctx.policy.as_mut().ok_or("model not loaded")?;
        search::plan_turn(&mut ctx.engine, policy, &s, lanes.max(1) as usize, seed as u64)
    })
}

/// A game rebuilt from mapgen.reset(seed) and the JSON list of catalog entries in the input; returns its id.
#[no_mangle]
pub extern "C" fn game_from_actions(seed: i32) -> i32 {
    respond(0, |ctx| {
        let actions: Vec<String> = serde_json::from_slice(&ctx.input).map_err(|e| e.to_string())?;
        let mut s = mapgen::reset(seed, &ctx.engine.geom, None);
        for a in &actions {
            let index: usize = a.parse().map_err(|_| "bad action id")?;
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
        let s = game(ctx, id)?.clone();
        let policy = ctx.policy.as_mut().ok_or("model not loaded")?;
        let w = search::world(&ctx.engine, &s, sample != 0, seed as u64, round as u64);
        let mut out = Vec::new();
        for lane in from.max(0)..to {
            let mut rng = search::Rng::new(search::mix(seed as u64, round as u64, lane as u64));
            let l = search::run_lane(&mut ctx.engine, policy, &w, lane == 0, &mut rng)?;
            out.push(json!({"lane": lane, "actions": l.actions.iter().map(|i| i.to_string()).collect::<Vec<_>>(),
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
        let chosen: Vec<usize> = chosen.iter().map(|a| a.parse::<usize>().map_err(|_| "bad action id".to_string()))
            .collect::<Result<_, _>>()?;
        if chosen.is_empty() {
            return Err("empty plan".into());
        }
        let mut s = game(ctx, id)?.clone();
        let w = search::world(&ctx.engine, &s, sample != 0, seed as u64, round as u64);
        let applied = search::commit_round(&mut ctx.engine, &mut s, &w, sample != 0, &chosen)?;
        *game(ctx, id)? = s;
        Ok(json!({"applied": applied.iter().map(|i| i.to_string()).collect::<Vec<_>>()}))
    })
}
