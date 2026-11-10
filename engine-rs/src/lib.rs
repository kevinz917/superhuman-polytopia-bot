//! PyO3 bindings: a batch of independent games mirroring rl/cpu_engine.py::CpuGames.

pub mod capture;
pub mod core;
pub mod cov;
pub mod diplomacy;
pub mod discovery;
pub mod economy;
pub mod geom;
pub mod hash;
pub mod history;
pub mod infiltration;
pub mod kernels;
pub mod mapgen;
pub mod masks;
pub mod military;
pub mod pathfinding;
pub mod prof;
pub mod rules;
pub mod state;
pub mod structures;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::core::{decode, step, step_with, terminal_rewards, Scratch};
use crate::geom::Geom;
use crate::kernels::{observe, view_fields, view_width_np};
use crate::masks::{legal_masks, mask_width, masks_into, packed_width, Dense, LegalCache, MaskScratch, Packed, Tee};
use crate::rules::Rules;
use crate::state::State;

#[pyclass]
pub struct GameBatch {
    size: usize,
    games: usize,
    /// seats per game (2 or 4); per-seat outputs (rewards, final views, capitals) have this many rows
    players: usize,
    rules: Rules,
    geom: Geom,
    states: Vec<State>,
    max_turns: Option<i32>,
    no_resign: bool,
    no_peace: bool,
    max_actions: i32,
    scratch: Scratch,
    mscratch: MaskScratch,
    /// per game: the masks tick() last returned, while they describe the game's state (step validation)
    legal: Vec<LegalCache>,
    /// set_threads: the worker pool tick() steps the games on, and one pair of scratch buffers per worker
    pool: Option<rayon::ThreadPool>,
    pool_scratch: Vec<(Scratch, MaskScratch)>,
}

impl GameBatch {
    fn mask_opts(&self) -> crate::masks::MaskOpts {
        crate::masks::MaskOpts { no_resign: self.no_resign, no_peace: self.no_peace }
    }
}

fn err(msg: &str) -> PyErr {
    PyValueError::new_err(msg.to_string())
}

#[pymethods]
impl GameBatch {
    #[new]
    #[pyo3(signature = (size, games, rules_json, engine_revision=None, max_turns=None, no_resign=true, max_actions=10000, players=2, no_peace=false))]
    fn new(size: usize, games: usize, rules_json: &str, engine_revision: Option<&str>, max_turns: Option<i32>, no_resign: bool,
           max_actions: i32, players: usize, no_peace: bool) -> PyResult<Self> {
        if mapgen::ruins(size).is_none() {
            return Err(err("board size must be 11, 14 or 18"));
        }
        if !(2..=crate::state::MAXP).contains(&players) {
            return Err(err("players must be 2, 3 or 4"));
        }
        let rules = Rules::from_json(rules_json).map_err(|e| err(&e))?;
        if let Some(rev) = engine_revision {
            if rules.engine_revision != rev {
                return Err(err("rule table and pinned engine revision differ"));
            }
        }
        let geom = Geom::new(size);
        let states = (0..games).map(|_| State::empty_players(size, players)).collect();
        Ok(GameBatch { size, games, players, rules, geom, states, max_turns, no_resign, no_peace, max_actions, scratch: Scratch::new(size * size),
                     mscratch: MaskScratch::new(size * size), legal: (0..games).map(|_| LegalCache::new(size * size)).collect(),
                     pool: None, pool_scratch: Vec::new() })
    }

    #[getter]
    fn size(&self) -> usize { self.size }
    #[getter]
    fn games(&self) -> usize { self.games }
    #[getter]
    fn players(&self) -> usize { self.players }
    #[getter]
    fn view_width(&self) -> usize { view_width_np(self.size * self.size, self.players) }
    #[getter]
    fn mask_width(&self) -> usize { mask_width(self.size * self.size) }
    fn view_fields(&self) -> Vec<&'static str> { view_fields(self.players) }
    fn family_widths(&self) -> Vec<usize> { core::family_widths(self.size * self.size).to_vec() }

    /// Fresh games from mapgen.reset(seed) for every lane.
    fn reset(&mut self, seeds: PyReadonlyArray1<i32>) -> PyResult<()> {
        let seeds = seeds.as_slice()?;
        if seeds.len() != self.games {
            return Err(err("seeds must have one entry per game"));
        }
        for (i, &seed) in seeds.iter().enumerate() {
            self.states[i] = mapgen::reset_players(seed, &self.geom, self.max_turns, self.players);
        }
        self.forget_masks();
        Ok(())
    }

    /// Fresh games from explicit layouts (maps.state.from_batch_row): terrain/structure/resource [G, n], capitals [G, players].
    fn reset_layouts(&mut self, seeds: PyReadonlyArray1<i32>, terrain: PyReadonlyArray2<i32>, structure: PyReadonlyArray2<i32>,
                     resource: PyReadonlyArray2<i32>, capitals: PyReadonlyArray2<i32>) -> PyResult<()> {
        let n = self.size * self.size;
        let seeds = seeds.as_slice()?;
        let (t, st, rs, caps) = (terrain.as_array(), structure.as_array(), resource.as_array(), capitals.as_array());
        if seeds.len() != self.games || t.shape() != [self.games, n] || st.shape() != [self.games, n] || rs.shape() != [self.games, n]
            || caps.shape() != [self.games, self.players] {
            return Err(err("layout arrays must be [games, n] and capitals [games, players]"));
        }
        for g in 0..self.games {
            let tv: Vec<i32> = t.row(g).to_vec();
            let sv: Vec<i32> = st.row(g).to_vec();
            let rv: Vec<i32> = rs.row(g).to_vec();
            let cap: Vec<usize> = caps.row(g).iter().map(|&c| c as usize).collect();
            self.states[g] = mapgen::initial_state(seeds[g] as u32, &tv, &sv, &rv, &cap, &self.geom,
                                                   self.max_turns.unwrap_or(mapgen::NO_ROUND_LIMIT));
        }
        self.forget_masks();
        Ok(())
    }

    /// Only the mapgen layout of one seed (terrain, structure, resource, capitals, plus the kept attempt's seed and
    /// GetInequality), for map checks. exact=True: GenerateInternal(seed) alone, the real GenerateWithSeed board.
    #[pyo3(signature = (seed, exact=false))]
    fn layout<'py>(&self, py: Python<'py>, seed: i32, exact: bool) -> PyResult<Bound<'py, PyDict>> {
        let (l, attempt, inequality) = if exact {
            let l = mapgen::generate_internal(seed, self.size, self.players);
            let q = mapgen::inequality(&l, self.size);
            (l, seed, q)
        } else {
            mapgen::generate(seed, self.size, self.players)
        };
        let d = PyDict::new(py);
        d.set_item("attempt_seed", attempt)?;
        d.set_item("inequality", inequality)?;
        d.set_item("terrain", l.terrain.into_pyarray(py))?;
        d.set_item("structure", l.structure.into_pyarray(py))?;
        d.set_item("resource", l.resource.into_pyarray(py))?;
        d.set_item("capitals", l.capitals.iter().map(|&c| c as i32).collect::<Vec<i32>>().into_pyarray(py))?;
        Ok(d)
    }

    /// The engine action (kind, source, target, parameter) of one (family, source, target) choice per game, as
    /// `step` would decode it in the current states: [G, 4].
    fn decode<'py>(&self, py: Python<'py>, family: PyReadonlyArray1<i32>, source: PyReadonlyArray1<i32>,
                   target: PyReadonlyArray1<i32>) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let (f, s, t) = (family.as_slice()?, source.as_slice()?, target.as_slice()?);
        let mut out = Vec::with_capacity(self.games * 4);
        for g in 0..self.games {
            let a = decode(&self.states[g], f[g], s[g], t[g]);
            out.extend_from_slice(&[a.kind, a.source, a.target, a.parameter]);
        }
        Ok(PyArray1::from_vec(py, out).reshape([self.games, 4])?)
    }

    /// Apply one decoded choice per game (no auto-reset, no prepare). Returns status [G].
    fn step<'py>(&mut self, py: Python<'py>, family: PyReadonlyArray1<i32>, source: PyReadonlyArray1<i32>,
                 target: PyReadonlyArray1<i32>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let (f, s, t) = (family.as_slice()?, source.as_slice()?, target.as_slice()?);
        let mut status = vec![0i32; self.games];
        for g in 0..self.games {
            let st = &mut self.states[g];
            let a = decode(st, f[g], s[g], t[g]);
            let v = st.state_version;
            status[g] = step(st, a, v, &self.rules, &self.geom, self.max_actions, &mut self.scratch);
        }
        self.forget_masks();
        Ok(status.into_pyarray(py))
    }

    /// One actor tick: step, auto-reset finished games from `seeds`, then masks and packed views.
    /// packed_masks: "masks" holds [G, ceil(mask_width/8)] bits (numpy.packbits order) instead of [G, mask_width] bytes.
    /// Per-seat outputs: rewards [G, players], final_view [G, players, vw]. both_views: "opp_view" is [G, vw] (the seat
    /// not to move) with two seats, else [G, players - 1, vw]: the other seats in turn order after the mover.
    /// With set_threads(k > 1) the games are stepped on k worker threads with the GIL released (each game is
    /// independent; results are identical to the sequential loop).
    #[pyo3(signature = (family, source, target, seeds, both_views=true, layouts=None, packed_masks=false))]
    fn tick<'py>(&mut self, py: Python<'py>, family: PyReadonlyArray1<i32>, source: PyReadonlyArray1<i32>,
                 target: PyReadonlyArray1<i32>, seeds: PyReadonlyArray1<i32>, both_views: bool,
                 layouts: Option<(PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>)>,
                 packed_masks: bool)
                 -> PyResult<Bound<'py, PyDict>> {
        let n = self.size * self.size;
        let games = self.games;
        let np = self.players;
        let vw = view_width_np(n, np);
        let others = np - 1;
        let mw = if packed_masks { packed_width(n) } else { mask_width(n) };
        let ow = if both_views { others * vw } else { 0 };
        let (mut masks, mut view, mut opp, mut final_view) = prof::time(prof::TICK_ALLOC, || (
            vec![0u8; games * mw], vec![0i16; games * vw], vec![0i16; games * ow], vec![0i16; games * np * vw]));
        let (f, s, t, seeds) = (family.as_slice()?, source.as_slice()?, target.as_slice()?, seeds.as_slice()?);
        let lay = layout_slices(&layouts, games, n, np)?;
        let (flags, rewards) = self.step_games(py, f, s, t, seeds, lay, both_views, packed_masks, &mut masks, &mut view, &mut opp,
                                               &mut final_view)?;
        prof::time(prof::TICK_OUTPUT, || {
        let d = PyDict::new(py);
        d.set_item("masks", PyArray1::from_vec(py, masks).reshape([games, mw])?)?;
        d.set_item("view", PyArray1::from_vec(py, view).reshape([games, vw])?)?;
        if both_views {
            if np > 2 {
                d.set_item("opp_view", PyArray1::from_vec(py, opp).reshape([games, others, vw])?)?;
            } else {
                d.set_item("opp_view", PyArray1::from_vec(py, opp).reshape([games, vw])?)?;
            }
        }
        d.set_item("final_view", PyArray1::from_vec(py, final_view).reshape([games, np, vw])?)?;
        self.small_outputs(py, &d, &flags, rewards)?;
        Ok(d)
        })
    }

    /// tick() writing the views and masks into caller-owned arrays instead of fresh ones (no per-tick allocation:
    /// fresh multi-MB arrays page-fault on first write and serialize the worker threads on the address-space lock).
    /// view [G, vw] i16, opp_view [G * (players - 1), vw] i16 (both_views) or [0, vw], masks [G, mask bytes] u8 (packed
    /// when packed_masks; every row is rewritten), final_view [G * players, vw] i16 (rows written for truncated games
    /// only). Returns the small outputs: done, terminated, truncated, rewards, status, current_player, turn, score,
    /// killed_turn, resigned_turn.
    #[pyo3(signature = (family, source, target, seeds, view, opp_view, masks, final_view, both_views=true, layouts=None,
                        packed_masks=true))]
    fn tick_into<'py>(&mut self, py: Python<'py>, family: PyReadonlyArray1<i32>, source: PyReadonlyArray1<i32>,
                      target: PyReadonlyArray1<i32>, seeds: PyReadonlyArray1<i32>, view: Bound<'py, PyArray2<i16>>,
                      opp_view: Bound<'py, PyArray2<i16>>, masks: Bound<'py, PyArray2<u8>>, final_view: Bound<'py, PyArray2<i16>>,
                      both_views: bool,
                      layouts: Option<(PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>)>,
                      packed_masks: bool)
                      -> PyResult<Bound<'py, PyDict>> {
        let n = self.size * self.size;
        let games = self.games;
        let np = self.players;
        let vw = view_width_np(n, np);
        let others = np - 1;
        let mw = if packed_masks { packed_width(n) } else { mask_width(n) };
        let ow = if both_views { others * vw } else { 0 };
        if view.shape() != [games, vw] || masks.shape() != [games, mw] || final_view.shape() != [games * np, vw]
            || (both_views && opp_view.shape() != [games * others, vw]) {
            return Err(err("tick_into: output arrays have the wrong shapes"));
        }
        let (f, s, t, seeds) = (family.as_slice()?, source.as_slice()?, target.as_slice()?, seeds.as_slice()?);
        let lay = layout_slices(&layouts, games, n, np)?;
        // SAFETY: the caller hands over contiguous arrays it does not touch during the call (checked contiguous by
        // as_slice_mut); the GIL is held except inside step_games, where only these slices are written.
        let (vs, os, ms, fs) = unsafe { (view.as_slice_mut()?, if both_views { opp_view.as_slice_mut()? } else { &mut [] },
                                         masks.as_slice_mut()?, final_view.as_slice_mut()?) };
        let _ = ow;
        let (flags, rewards) = self.step_games(py, f, s, t, seeds, lay, both_views, packed_masks, ms, vs, os, fs)?;
        let d = PyDict::new(py);
        self.small_outputs(py, &d, &flags, rewards)?;
        Ok(d)
    }

    /// Turn the phase profile on (resetting it) or off on the calling thread and every worker thread.
    fn profile_threads(&self, on: bool) {
        prof::enable(on);
        if let Some(pool) = &self.pool {
            pool.broadcast(|_| prof::enable(on));
        }
    }

    /// phase_profile() summed over the calling thread and the worker threads.
    fn phase_profile_threads<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut all: Vec<Vec<(String, u64, f64)>> = vec![prof::snapshot()];
        if let Some(pool) = &self.pool {
            all.extend(pool.broadcast(|_| prof::snapshot()));
        }
        let mut sum: std::collections::BTreeMap<String, (u64, f64)> = std::collections::BTreeMap::new();
        for snap in all {
            for (name, calls, ms) in snap {
                let e = sum.entry(name).or_insert((0, 0.));
                e.0 += calls;
                e.1 += ms;
            }
        }
        let d = PyDict::new(py);
        for (name, v) in sum {
            d.set_item(name, v)?;
        }
        Ok(d)
    }

    /// Step tick()'s games on `threads` worker threads (1: the calling thread, the default).
    fn set_threads(&mut self, threads: usize) -> PyResult<()> {
        self.pool = if threads > 1 {
            Some(rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| err(&e.to_string()))?)
        } else {
            None
        };
        Ok(())
    }

    /// Turn the phase profile on (resetting it) or off; read it with phase_profile().
    fn profile_phases(&self, on: bool) {
        prof::enable(on);
    }

    /// {phase: (calls, total ms)} accumulated since profile_phases(True), for the phases that ran.
    fn phase_profile<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, calls, ms) in prof::snapshot() {
            d.set_item(name, (calls, ms))?;
        }
        Ok(d)
    }

    /// Median milliseconds per batch for each legal-mask family (and the full mask), for profiling. The families
    /// emit into a counting sink (computation only); "full" is masks_into a zeroed dense buffer, "observe" one seat.
    fn profile_masks<'py>(&self, py: Python<'py>, repeats: usize) -> PyResult<Bound<'py, PyDict>> {
        use std::cell::RefCell;
        use std::time::Instant;
        let n = self.size * self.size;
        let d = PyDict::new(py);
        let ms = RefCell::new(MaskScratch::new(n));
        let mut flat = vec![0u8; mask_width(n)];
        let time = |name: &str, f: &mut dyn FnMut(&State)| -> PyResult<()> {
            let mut samples: Vec<f64> = Vec::with_capacity(repeats);
            for _ in 0..repeats.max(1) {
                let t0 = Instant::now();
                for st in &self.states { f(st); }
                samples.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            d.set_item(name, samples[samples.len() / 2])
        };
        let (r, g) = (&self.rules, &self.geom);
        let mut c = masks::Count(0);
        time("reward", &mut |st| { masks::reward_rows(st, &mut c); })?;
        time("move_rows", &mut |st| masks::move_rows(st, r, g, &mut ms.borrow_mut().mv, &mut c))?;
        time("attack_rows", &mut |st| masks::attack_rows(st, r, g, &mut c))?;
        time("summon", &mut |st| masks::summon_rows(st, r, &mut ms.borrow_mut().summon, &mut c))?;
        time("economic_tiles", &mut |st| economy::economic_tiles(st, &mut ms.borrow_mut().build.econ))?;
        time("build", &mut |st| masks::build_rows(st, r, g, &mut ms.borrow_mut().build, &mut c))?;
        time("research", &mut |st| { c.0 += kernels::research_bits(st, r).count_ones() as usize; })?;
        time("harvest_forest_destroy", &mut |st| {
            let b = &mut ms.borrow_mut().build;
            b.prepare(st, r, g);
            masks::economy_rows(st, &b.econ_list, &mut c)
        })?;
        time("units", &mut |st| masks::unit_rows(st, r, g, &mut c))?;
        time("pending_reward", &mut |st| { let _ = st.pending_reward(); })?;
        time("full_count", &mut |st| masks_into(st, r, g, self.mask_opts(), &mut ms.borrow_mut(), &mut c))?;
        time("full", &mut |st| legal_masks(st, r, g, self.mask_opts(), &mut ms.borrow_mut(), &mut flat))?;
        let mut v = vec![0i16; view_width_np(n, self.players)];
        time("observe", &mut |st| observe(st, st.current_player, g, &mut v))?;
        d.set_item("entries", c.0)?;
        Ok(d)
    }

    /// Branch-coverage counters accumulated by step() on this thread (diagnostic).
    fn coverage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, v) in cov::NAMES.iter().zip(cov::snapshot().iter()) {
            d.set_item(name, *v)?;
        }
        Ok(d)
    }

    fn reset_coverage(&self) { cov::reset(); }

    /// [G, mask_width] 0/1 bytes, or with packed=true [G, ceil(mask_width/8)] bits (numpy.packbits order).
    #[pyo3(signature = (packed=false))]
    fn legal_masks<'py>(&mut self, py: Python<'py>, packed: bool) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let n = self.size * self.size;
        let mw = if packed { packed_width(n) } else { mask_width(n) };
        let mut masks = vec![0u8; self.games * mw];
        for g in 0..self.games {
            let row = &mut masks[g * mw..(g + 1) * mw];
            if packed {
                masks_into(&self.states[g], &self.rules, &self.geom, self.mask_opts(), &mut self.mscratch, &mut Packed(row));
            } else {
                masks_into(&self.states[g], &self.rules, &self.geom, self.mask_opts(), &mut self.mscratch, &mut Dense(row));
            }
        }
        Ok(PyArray1::from_vec(py, masks).reshape([self.games, mw])?)
    }

    /// Packed public view of `player` (None: the player to move) for every game.
    #[pyo3(signature = (player=None))]
    fn observe<'py>(&self, py: Python<'py>, player: Option<i32>) -> PyResult<Bound<'py, PyArray2<i16>>> {
        let vw = view_width_np(self.size * self.size, self.players);
        let mut view = vec![0i16; self.games * vw];
        for g in 0..self.games {
            let st = &self.states[g];
            let p = match player { Some(p) => p, None => st.current_player };
            observe(st, p, &self.geom, &mut view[g * vw..(g + 1) * vw]);
        }
        Ok(PyArray1::from_vec(py, view).reshape([self.games, vw])?)
    }

    /// Packed public views of the seats NOT to move, for every game: [G, vw] with two seats, else
    /// [G * (players - 1), vw] (reshape to [G, players - 1, vw]), the other seats in turn order after the mover.
    fn observe_opponent<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i16>>> {
        let vw = view_width_np(self.size * self.size, self.players);
        let others = self.players - 1;
        let mut view = vec![0i16; self.games * others * vw];
        for g in 0..self.games {
            let st = &self.states[g];
            for (k, seat) in other_seats(st).enumerate() {
                let off = (g * others + k) * vw;
                observe(st, seat, &self.geom, &mut view[off..off + vw]);
            }
        }
        Ok(PyArray1::from_vec(py, view).reshape([self.games * others, vw])?)
    }

    fn terminal_rewards<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let np = self.players;
        let mut out = vec![0f32; self.games * np];
        for g in 0..self.games {
            let r = terminal_rewards(&self.states[g]);
            out[g * np..(g + 1) * np].copy_from_slice(&r[..np]);
        }
        Ok(PyArray1::from_vec(py, out).reshape([self.games, np])?)
    }

    /// Per game: current_player, turn, terminated, truncated (cheap; no full export).
    fn scalars<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("current_player", self.states.iter().map(|s| s.current_player).collect::<Vec<i32>>().into_pyarray(py))?;
        d.set_item("turn", self.states.iter().map(|s| s.turn).collect::<Vec<i32>>().into_pyarray(py))?;
        d.set_item("terminated", self.states.iter().map(|s| s.terminated).collect::<Vec<bool>>().into_pyarray(py))?;
        d.set_item("truncated", self.states.iter().map(|s| s.truncated).collect::<Vec<bool>>().into_pyarray(py))?;
        Ok(d)
    }

    /// Per game: the stars `player` (1-based) would earn at the start of their turn (structures::income).
    fn income<'py>(&self, py: Python<'py>, player: i32) -> PyResult<Bound<'py, PyArray1<i32>>> {
        if player < 1 || player as usize > self.players {
            return Err(err("player out of range"));
        }
        Ok(self.states.iter().map(|s| structures::income(s, player, &self.rules, &self.geom)).collect::<Vec<i32>>().into_pyarray(py))
    }

    /// Every state field as a NumPy array with a leading game axis; keys "tiles.terrain", ..., "seed".
    fn export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let n = self.size * self.size;
        let gm = self.games;
        let d = PyDict::new(py);
        macro_rules! field {
            ($name:expr, $ty:ty, [$($dim:expr),*], $get:expr) => {{
                let dims: Vec<usize> = vec![$($dim),*];
                let per: usize = dims.iter().product();
                let mut buf: Vec<$ty> = Vec::with_capacity(gm * per);
                for g in 0..gm { let s = &self.states[g]; let f: &dyn Fn(&State, &mut Vec<$ty>) = &$get; f(s, &mut buf); }
                let mut shape = vec![gm]; shape.extend(dims);
                d.set_item($name, PyArray1::from_vec(py, buf).reshape(shape)?)?;
            }};
        }
        field!("tiles.terrain", i32, [n], |s, b| b.extend_from_slice(&s.tiles.terrain));
        field!("tiles.owner", i32, [n], |s, b| b.extend_from_slice(&s.tiles.owner));
        field!("tiles.capital", i32, [n], |s, b| b.extend_from_slice(&s.tiles.capital));
        field!("tiles.city", i32, [n], |s, b| b.extend_from_slice(&s.tiles.city));
        field!("tiles.road", bool, [n], |s, b| b.extend_from_slice(&s.tiles.road));
        field!("tiles.structure", i32, [n], |s, b| b.extend_from_slice(&s.tiles.structure));
        field!("tiles.resource", i32, [n], |s, b| b.extend_from_slice(&s.tiles.resource));
        let np = self.players;
        field!("tiles.explored", bool, [np, n], |s, b| for k in 0..s.np { b.extend_from_slice(&s.tiles.explored[k]); });
        field!("tiles.structure_level", i32, [n], |s, b| b.extend_from_slice(&s.tiles.structure_level));
        field!("tiles.structure_turn", i32, [n], |s, b| b.extend_from_slice(&s.tiles.structure_turn));
        field!("tiles.route", bool, [n], |s, b| b.extend_from_slice(&s.tiles.route));
        field!("units.owner", i32, [n], |s, b| b.extend_from_slice(&s.units.owner));
        field!("units.kind", i32, [n], |s, b| b.extend_from_slice(&s.units.kind));
        field!("units.health", f32, [n], |s, b| b.extend_from_slice(&s.units.health));
        field!("units.veteran", bool, [n], |s, b| b.extend_from_slice(&s.units.veteran));
        field!("units.moved", bool, [n], |s, b| b.extend_from_slice(&s.units.moved));
        field!("units.attacked", bool, [n], |s, b| b.extend_from_slice(&s.units.attacked));
        field!("units.kills", i32, [n], |s, b| b.extend_from_slice(&s.units.kills));
        field!("units.poison", bool, [n], |s, b| b.extend_from_slice(&s.units.poison));
        field!("units.frozen", bool, [n], |s, b| b.extend_from_slice(&s.units.frozen));
        field!("units.boosted", bool, [n], |s, b| b.extend_from_slice(&s.units.boosted));
        field!("units.home", i32, [n], |s, b| b.extend_from_slice(&s.units.home));
        field!("units.created", i32, [n], |s, b| b.extend_from_slice(&s.units.created));
        field!("units.converted", bool, [n], |s, b| b.extend_from_slice(&s.units.converted));
        field!("units.attacks_performed", i32, [n], |s, b| b.extend_from_slice(&s.units.attacks_performed));
        field!("units.prev_x", i32, [n], |s, b| b.extend_from_slice(&s.units.prev_x));
        field!("units.prev_y", i32, [n], |s, b| b.extend_from_slice(&s.units.prev_y));
        field!("units.last_attack", i32, [n], |s, b| b.extend_from_slice(&s.units.last_attack));
        field!("units.passenger", i32, [n], |s, b| b.extend_from_slice(&s.units.passenger));
        field!("units.invisible", bool, [n], |s, b| b.extend_from_slice(&s.units.invisible));
        field!("units.direction", i32, [n], |s, b| b.extend_from_slice(&s.units.direction));
        field!("cities.owner", i32, [n], |s, b| b.extend_from_slice(&s.cities.owner));
        field!("cities.level", i32, [n], |s, b| b.extend_from_slice(&s.cities.level));
        field!("cities.population", i32, [n], |s, b| b.extend_from_slice(&s.cities.population));
        field!("cities.progress", i32, [n], |s, b| b.extend_from_slice(&s.cities.progress));
        field!("cities.production", i32, [n], |s, b| b.extend_from_slice(&s.cities.production));
        field!("cities.border", i32, [n], |s, b| b.extend_from_slice(&s.cities.border));
        field!("cities.rewards", i32, [n, 9], |s, b| b.extend_from_slice(&s.cities.rewards));
        field!("cities.territory", bool, [n, n], |s, b| b.extend_from_slice(&s.cities.territory));
        field!("cities.connected", bool, [n], |s, b| b.extend_from_slice(&s.cities.connected));
        field!("cities.order", i32, [n], |s, b| b.extend_from_slice(&s.cities.order));
        field!("cities.infiltrated", i32, [n], |s, b| b.extend_from_slice(&s.cities.infiltrated));
        field!("cities.riot", bool, [n], |s, b| b.extend_from_slice(&s.cities.riot));
        field!("players.stars", i32, [np], |s, b| b.extend_from_slice(&s.players.stars[..s.np]));
        field!("players.score", i32, [np], |s, b| b.extend_from_slice(&s.players.score[..s.np]));
        field!("players.tech", bool, [np, 128], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.tech[k]); });
        field!("players.tech_turn", i32, [np, 128], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.tech_turn[k]); });
        field!("players.known", bool, [np, np], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.known[k][..s.np]); });
        field!("players.kills", i32, [np], |s, b| b.extend_from_slice(&s.players.kills[..s.np]));
        field!("players.casualties", i32, [np], |s, b| b.extend_from_slice(&s.players.casualties[..s.np]));
        field!("players.pacifist", i32, [np], |s, b| b.extend_from_slice(&s.players.pacifist[..s.np]));
        field!("players.attacked", bool, [np], |s, b| b.extend_from_slice(&s.players.attacked[..s.np]));
        field!("players.built", bool, [np, 128], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.built[k]); });
        field!("players.killed_turn", i32, [np], |s, b| b.extend_from_slice(&s.players.killed_turn[..s.np]));
        field!("players.resigned_turn", i32, [np], |s, b| b.extend_from_slice(&s.players.resigned_turn[..s.np]));
        field!("players.killer", i32, [np], |s, b| b.extend_from_slice(&s.players.killer[..s.np]));
        field!("players.peace", bool, [np, np], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.peace[k][..s.np]); });
        field!("players.embassy", bool, [np, np], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.embassy[k][..s.np]); });
        field!("players.peace_offer", bool, [np, np], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.peace_offer[k][..s.np]); });
        field!("players.truce_breaker", i32, [np, np], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.truce_breaker[k][..s.np]); });
        field!("players.hostile_until", i32, [np, np], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.hostile_until[k][..s.np]); });
        field!("players.peace_broken", i32, [np, np], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.peace_broken[k][..s.np]); });
        field!("players.monuments", bool, [np, 128], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.monuments[k]); });
        if np > 2 {
            field!("players.out_order", i32, [np], |s, b| b.extend_from_slice(&s.players.out_order[..s.np]));
        }
        // observation history (history.rs); optional on import
        let ht = crate::state::HIST_TURNS;
        field!("history.seen_unit", bool, [np, ht, n], |s, b| for k in 0..s.np { for v in &s.history[k].seen_unit { b.extend_from_slice(v); } });
        field!("history.prev_code", i32, [np, n], |s, b| for k in 0..s.np { b.extend_from_slice(&s.history[k].prev_code); });
        field!("history.built", bool, [np, n], |s, b| for k in 0..s.np { b.extend_from_slice(&s.history[k].built); });
        field!("history.trained", bool, [np, n], |s, b| for k in 0..s.np { b.extend_from_slice(&s.history[k].trained); });
        field!("history.enemy_tech", bool, [np, 128], |s, b| for k in 0..s.np { for i in 0..128 { b.push(s.history[k].enemy_tech >> i & 1 == 1); } });
        field!("current_player", i32, [], |s, b| b.push(s.current_player));
        field!("turn", i32, [], |s, b| b.push(s.turn));
        field!("version", i32, [], |s, b| b.push(s.version));
        field!("state_version", i32, [], |s, b| b.push(s.state_version));
        field!("actions", i32, [], |s, b| b.push(s.actions));
        field!("terminated", bool, [], |s, b| b.push(s.terminated));
        field!("truncated", bool, [], |s, b| b.push(s.truncated));
        field!("fog", bool, [], |s, b| b.push(s.fog));
        field!("reference_max_turns", i32, [], |s, b| b.push(s.reference_max_turns));
        field!("core_supported", bool, [], |s, b| b.push(s.core_supported));
        field!("seed", u32, [], |s, b| b.push(s.seed));
        field!("initial_seed", u32, [], |s, b| b.push(s.initial_seed));
        Ok(d)
    }

    /// Inverse of export: overwrite every game from arrays with a leading game axis (missing keys are an error).
    fn import_(&mut self, py: Python<'_>, fields: &Bound<'_, PyDict>) -> PyResult<()> {
        let n = self.size * self.size;
        let gm = self.games;
        fn take<'py, T: numpy::Element + Copy>(py: Python<'py>, d: &Bound<'py, PyDict>, key: &str, expect: usize) -> PyResult<Vec<T>> {
            let item = d.get_item(key)?.ok_or_else(|| err(&format!("missing field {key}")))?;
            let arr: PyReadonlyArrayDyn<T> = item.extract()?;
            let v: Vec<T> = arr.as_array().iter().copied().collect();
            if v.len() != expect {
                return Err(err(&format!("field {key}: expected {expect} values, got {}", v.len())));
            }
            let _ = py;
            Ok(v)
        }
        macro_rules! load {
            ($name:expr, $ty:ty, $per:expr, $set:expr) => {{
                let v: Vec<$ty> = take(py, fields, $name, gm * $per)?;
                for g in 0..gm { let s = &mut self.states[g]; let f: &dyn Fn(&mut State, &[$ty]) = &$set; f(s, &v[g * $per..(g + 1) * $per]); }
            }};
        }
        load!("tiles.terrain", i32, n, |s, v| s.tiles.terrain.copy_from_slice(v));
        load!("tiles.owner", i32, n, |s, v| s.tiles.owner.copy_from_slice(v));
        load!("tiles.capital", i32, n, |s, v| s.tiles.capital.copy_from_slice(v));
        load!("tiles.city", i32, n, |s, v| s.tiles.city.copy_from_slice(v));
        load!("tiles.road", bool, n, |s, v| s.tiles.road.copy_from_slice(v));
        load!("tiles.structure", i32, n, |s, v| s.tiles.structure.copy_from_slice(v));
        load!("tiles.resource", i32, n, |s, v| s.tiles.resource.copy_from_slice(v));
        let np = self.players;
        load!("tiles.explored", bool, np * n, |s, v| for k in 0..s.np { s.tiles.explored[k].copy_from_slice(&v[k * n..(k + 1) * n]); });
        load!("tiles.structure_level", i32, n, |s, v| s.tiles.structure_level.copy_from_slice(v));
        load!("tiles.structure_turn", i32, n, |s, v| s.tiles.structure_turn.copy_from_slice(v));
        // optional (gpu/state.py has no TileData.hasRoute): without it, the routes MapDataExtensions.UpdateRoutes
        // finds on the loaded position (computed below, once every field is in)
        let routes_given = fields.contains("tiles.route")?;
        if routes_given {
            load!("tiles.route", bool, n, |s, v| s.tiles.route.copy_from_slice(v));
        }
        load!("units.owner", i32, n, |s, v| s.units.owner.copy_from_slice(v));
        load!("units.kind", i32, n, |s, v| s.units.kind.copy_from_slice(v));
        load!("units.health", f32, n, |s, v| s.units.health.copy_from_slice(v));
        load!("units.veteran", bool, n, |s, v| s.units.veteran.copy_from_slice(v));
        load!("units.moved", bool, n, |s, v| s.units.moved.copy_from_slice(v));
        load!("units.attacked", bool, n, |s, v| s.units.attacked.copy_from_slice(v));
        load!("units.kills", i32, n, |s, v| s.units.kills.copy_from_slice(v));
        load!("units.poison", bool, n, |s, v| s.units.poison.copy_from_slice(v));
        load!("units.frozen", bool, n, |s, v| s.units.frozen.copy_from_slice(v));
        load!("units.boosted", bool, n, |s, v| s.units.boosted.copy_from_slice(v));
        load!("units.home", i32, n, |s, v| s.units.home.copy_from_slice(v));
        load!("units.created", i32, n, |s, v| s.units.created.copy_from_slice(v));
        load!("units.converted", bool, n, |s, v| s.units.converted.copy_from_slice(v));
        load!("units.attacks_performed", i32, n, |s, v| s.units.attacks_performed.copy_from_slice(v));
        load!("units.prev_x", i32, n, |s, v| s.units.prev_x.copy_from_slice(v));
        load!("units.prev_y", i32, n, |s, v| s.units.prev_y.copy_from_slice(v));
        load!("units.last_attack", i32, n, |s, v| s.units.last_attack.copy_from_slice(v));
        load!("units.passenger", i32, n, |s, v| s.units.passenger.copy_from_slice(v));
        load!("units.invisible", bool, n, |s, v| s.units.invisible.copy_from_slice(v));
        // optional (states saved before units had a facing): toward the map centre, as at creation
        if fields.contains("units.direction")? {
            load!("units.direction", i32, n, |s, v| s.units.direction.copy_from_slice(v));
        } else {
            for s in self.states.iter_mut() {
                for t in 0..n {
                    s.units.direction[t] = if s.units.owner[t] != 0 { crate::military::spawn_direction(s.size, t) } else { 0 };
                }
            }
        }
        load!("cities.owner", i32, n, |s, v| s.cities.owner.copy_from_slice(v));
        load!("cities.level", i32, n, |s, v| s.cities.level.copy_from_slice(v));
        load!("cities.population", i32, n, |s, v| s.cities.population.copy_from_slice(v));
        load!("cities.progress", i32, n, |s, v| s.cities.progress.copy_from_slice(v));
        load!("cities.production", i32, n, |s, v| s.cities.production.copy_from_slice(v));
        load!("cities.border", i32, n, |s, v| s.cities.border.copy_from_slice(v));
        load!("cities.rewards", i32, n * 9, |s, v| s.cities.rewards.copy_from_slice(v));
        load!("cities.territory", bool, n * n, |s, v| s.cities.territory.copy_from_slice(v));
        load!("cities.connected", bool, n, |s, v| s.cities.connected.copy_from_slice(v));
        load!("cities.order", i32, n, |s, v| s.cities.order.copy_from_slice(v));
        load!("cities.infiltrated", i32, n, |s, v| s.cities.infiltrated.copy_from_slice(v));
        load!("cities.riot", bool, n, |s, v| s.cities.riot.copy_from_slice(v));
        load!("players.stars", i32, np, |s, v| s.players.stars[..s.np].copy_from_slice(v));
        load!("players.score", i32, np, |s, v| s.players.score[..s.np].copy_from_slice(v));
        load!("players.tech", bool, np * 128, |s, v| for k in 0..s.np { s.players.tech[k].copy_from_slice(&v[k * 128..(k + 1) * 128]); });
        load!("players.tech_turn", i32, np * 128, |s, v| for k in 0..s.np { s.players.tech_turn[k].copy_from_slice(&v[k * 128..(k + 1) * 128]); });
        load!("players.known", bool, np * np, |s, v| for k in 0..s.np { s.players.known[k][..s.np].copy_from_slice(&v[k * s.np..(k + 1) * s.np]); });
        load!("players.kills", i32, np, |s, v| s.players.kills[..s.np].copy_from_slice(v));
        load!("players.casualties", i32, np, |s, v| s.players.casualties[..s.np].copy_from_slice(v));
        load!("players.pacifist", i32, np, |s, v| s.players.pacifist[..s.np].copy_from_slice(v));
        load!("players.attacked", bool, np, |s, v| s.players.attacked[..s.np].copy_from_slice(v));
        load!("players.built", bool, np * 128, |s, v| for k in 0..s.np { s.players.built[k].copy_from_slice(&v[k * 128..(k + 1) * 128]); });
        load!("players.killed_turn", i32, np, |s, v| s.players.killed_turn[..s.np].copy_from_slice(v));
        load!("players.resigned_turn", i32, np, |s, v| s.players.resigned_turn[..s.np].copy_from_slice(v));
        load!("players.killer", i32, np, |s, v| s.players.killer[..s.np].copy_from_slice(v));
        load!("players.peace", bool, np * np, |s, v| for k in 0..s.np { s.players.peace[k][..s.np].copy_from_slice(&v[k * s.np..(k + 1) * s.np]); });
        load!("players.embassy", bool, np * np, |s, v| for k in 0..s.np { s.players.embassy[k][..s.np].copy_from_slice(&v[k * s.np..(k + 1) * s.np]); });
        load!("players.peace_offer", bool, np * np, |s, v| for k in 0..s.np { s.players.peace_offer[k][..s.np].copy_from_slice(&v[k * s.np..(k + 1) * s.np]); });
        load!("players.truce_breaker", i32, np * np, |s, v| for k in 0..s.np { s.players.truce_breaker[k][..s.np].copy_from_slice(&v[k * s.np..(k + 1) * s.np]); });
        // per pair since 2026-10-31; states saved before hold one stamp per attacker (applied to every victim)
        let per_attacker = {
            let item = fields.get_item("players.hostile_until")?.ok_or_else(|| err("missing field players.hostile_until"))?;
            let arr: PyReadonlyArrayDyn<i32> = item.extract()?;
            arr.as_array().len() == gm * np
        };
        if per_attacker && np > 1 {
            load!("players.hostile_until", i32, np, |s, v| for k in 0..s.np { s.players.hostile_until[k] = [v[k]; crate::state::MAXP]; });
        } else {
            load!("players.hostile_until", i32, np * np, |s, v| for k in 0..s.np { s.players.hostile_until[k][..s.np].copy_from_slice(&v[k * s.np..(k + 1) * s.np]); });
        }
        // optional (states saved before it existed): a CoolDown relation is taken as broken this turn (set below,
        // once the turn is loaded)
        let has_broken = fields.contains("players.peace_broken")?;
        if has_broken {
            load!("players.peace_broken", i32, np * np, |s, v| for k in 0..s.np { s.players.peace_broken[k][..s.np].copy_from_slice(&v[k * s.np..(k + 1) * s.np]); });
        }
        load!("players.monuments", bool, np * 128, |s, v| for k in 0..s.np { s.players.monuments[k].copy_from_slice(&v[k * 128..(k + 1) * 128]); });
        if np > 2 {
            load!("players.out_order", i32, np, |s, v| s.players.out_order[..s.np].copy_from_slice(v));
        }
        // optional (states saved before the observation history existed, and gpu/state.py States without it): empty
        let ht = crate::state::HIST_TURNS;
        if fields.contains("history.seen_unit")? {
            load!("history.seen_unit", bool, np * ht * n, |s, v| for k in 0..s.np { for j in 0..ht { s.history[k].seen_unit[j].copy_from_slice(&v[(k * ht + j) * n..(k * ht + j + 1) * n]); } });
            load!("history.prev_code", i32, np * n, |s, v| for k in 0..s.np { s.history[k].prev_code.copy_from_slice(&v[k * n..(k + 1) * n]); });
            load!("history.built", bool, np * n, |s, v| for k in 0..s.np { s.history[k].built.copy_from_slice(&v[k * n..(k + 1) * n]); });
            load!("history.trained", bool, np * n, |s, v| for k in 0..s.np { s.history[k].trained.copy_from_slice(&v[k * n..(k + 1) * n]); });
            load!("history.enemy_tech", bool, np * 128, |s, v| for k in 0..s.np { s.history[k].enemy_tech = (0..128).fold(0u128, |b, i| b | (v[k * 128 + i] as u128) << i); });
        } else {
            for s in self.states.iter_mut() {
                for h in s.history.iter_mut() {
                    h.clear();
                }
            }
        }
        load!("current_player", i32, 1, |s, v| s.current_player = v[0]);
        load!("turn", i32, 1, |s, v| s.turn = v[0]);
        load!("version", i32, 1, |s, v| s.version = v[0]);
        load!("state_version", i32, 1, |s, v| s.state_version = v[0]);
        load!("actions", i32, 1, |s, v| s.actions = v[0]);
        load!("terminated", bool, 1, |s, v| s.terminated = v[0]);
        load!("truncated", bool, 1, |s, v| s.truncated = v[0]);
        load!("fog", bool, 1, |s, v| s.fog = v[0]);
        load!("reference_max_turns", i32, 1, |s, v| s.reference_max_turns = v[0]);
        load!("core_supported", bool, 1, |s, v| s.core_supported = v[0]);
        load!("seed", u32, 1, |s, v| s.seed = v[0]);
        load!("initial_seed", u32, 1, |s, v| s.initial_seed = v[0]);
        for s in self.states.iter_mut() {
            s.monuments_settled = false;
            if !has_broken {
                s.players.peace_broken = [[s.turn; crate::state::MAXP]; crate::state::MAXP];
            }
            if !routes_given {
                structures::update_routes(s, &self.rules);
            }
        }
        self.forget_masks();
        Ok(())
    }

    /// Names of the exported fields, in export order.
    fn field_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let d = self.export(py)?;
        Ok(d.keys())
    }
}

/// Seat ids (1-based) other than the mover, in turn order after it: with two seats the single opponent.
impl GameBatch {
    /// Step every game once (sequentially, or on the worker pool with the GIL released) into the given buffers.
    #[allow(clippy::too_many_arguments)]
    fn step_games(&mut self, py: Python<'_>, f: &[i32], s: &[i32], t: &[i32], seeds: &[i32],
                  lay: Option<(&[i32], &[i32], &[i32], &[i32])>, both_views: bool, packed: bool, masks: &mut [u8],
                  view: &mut [i16], opp: &mut [i16], final_view: &mut [i16])
                  -> PyResult<(Vec<(i32, bool, bool, bool)>, Vec<f32>)> {
        let n = self.size * self.size;
        let games = self.games;
        if f.len() != games || s.len() != games || t.len() != games || seeds.len() != games {
            return Err(err("choice and seed arrays must have one entry per game"));
        }
        let np = self.players;
        let vw = view_width_np(n, np);
        let others = np - 1;
        let mw = if packed { packed_width(n) } else { mask_width(n) };
        let ow = if both_views { others * vw } else { 0 };
        let mut flags = vec![(0i32, false, false, false); games];   // status, done, terminated, truncated
        let mut rewards = vec![0f32; games * np];
        let ctx = TickCtx { rules: &self.rules, geom: &self.geom, max_actions: self.max_actions, max_turns_opt: self.max_turns,
                            max_turns: self.max_turns.unwrap_or(mapgen::NO_ROUND_LIMIT), np, n, vw, packed,
                            both_views, opts: self.mask_opts(), f, s, t, seeds, lay };
        let threads = self.pool.as_ref().map(|p| p.current_num_threads()).unwrap_or(1);
        if threads <= 1 || games < 2 {
            for (g, ((((((st, cache), m), v), o), fv), (fl, rw))) in self.states.iter_mut().zip(self.legal.iter_mut())
                .zip(masks.chunks_mut(mw)).zip(view.chunks_mut(vw)).zip(chunk_or_empty(opp, ow, games))
                .zip(final_view.chunks_mut(np * vw)).zip(flags.iter_mut().zip(rewards.chunks_mut(np))).enumerate() {
                *fl = tick_game(&ctx, g, st, cache, &mut self.scratch, &mut self.mscratch, m, v, o, fv, rw);
            }
        } else {
            // one task per contiguous run of games, each with its own scratch buffers
            let per = games.div_ceil(threads);
            while self.pool_scratch.len() < threads {
                self.pool_scratch.push((Scratch::new(n), MaskScratch::new(n)));
            }
            let mut tasks = Vec::with_capacity(threads);
            let (mut st, mut lg, mut mk, mut vi, mut op, mut fv, mut fl, mut rw) = (&mut self.states[..], &mut self.legal[..],
                &mut masks[..], &mut view[..], &mut opp[..], &mut final_view[..], &mut flags[..], &mut rewards[..]);
            let mut first = 0;
            for scratch in self.pool_scratch.iter_mut() {
                let k = per.min(games - first);
                if k == 0 { break; }
                let (a, b) = st.split_at_mut(k); st = b;
                let (c, d) = lg.split_at_mut(k); lg = d;
                let (e, h) = mk.split_at_mut(k * mw); mk = h;
                let (i, j) = vi.split_at_mut(k * vw); vi = j;
                let (l, q) = op.split_at_mut(k * ow); op = q;
                let (u, w) = fv.split_at_mut(k * np * vw); fv = w;
                let (x, y) = fl.split_at_mut(k); fl = y;
                let (z, zz) = rw.split_at_mut(k * np); rw = zz;
                tasks.push((first, a, c, e, i, l, u, x, z, scratch));
                first += k;
            }
            let pool = self.pool.as_ref().unwrap();
            let ctx = &ctx;
            py.allow_threads(|| pool.install(|| {
                use rayon::prelude::*;
                tasks.into_par_iter().for_each(|(first, st, lg, mk, vi, op, fv, fl, rw, scratch)| {
                    let (sc, ms) = (&mut scratch.0, &mut scratch.1);
                    for k in 0..st.len() {
                        let o: &mut [i16] = if ow > 0 { &mut op[k * ow..(k + 1) * ow] } else { &mut [] };
                        fl[k] = tick_game(ctx, first + k, &mut st[k], &mut lg[k], sc, ms, &mut mk[k * mw..(k + 1) * mw],
                                          &mut vi[k * vw..(k + 1) * vw], o, &mut fv[k * np * vw..(k + 1) * np * vw],
                                          &mut rw[k * np..(k + 1) * np]);
                    }
                });
            }));
        }
        Ok((flags, rewards))
    }

    /// done / terminated / truncated / rewards / status and the light post-tick scalars into `d`.
    fn small_outputs<'py>(&self, py: Python<'py>, d: &Bound<'py, PyDict>, flags: &[(i32, bool, bool, bool)], rewards: Vec<f32>)
                          -> PyResult<()> {
        let games = self.games;
        let np = self.players;
        d.set_item("done", flags.iter().map(|x| x.1).collect::<Vec<bool>>().into_pyarray(py))?;
        d.set_item("terminated", flags.iter().map(|x| x.2).collect::<Vec<bool>>().into_pyarray(py))?;
        d.set_item("truncated", flags.iter().map(|x| x.3).collect::<Vec<bool>>().into_pyarray(py))?;
        d.set_item("rewards", PyArray1::from_vec(py, rewards).reshape([games, np])?)?;
        d.set_item("status", flags.iter().map(|x| x.0).collect::<Vec<i32>>().into_pyarray(py))?;
        // The few post-tick fields an actor reads (RustGames(light_states=True)), so it needs no full export():
        // current player and turn [G], score / killed_turn / resigned_turn [G, players].
        let sts = &self.states;
        d.set_item("current_player", sts.iter().map(|s| s.current_player).collect::<Vec<i32>>().into_pyarray(py))?;
        d.set_item("turn", sts.iter().map(|s| s.turn).collect::<Vec<i32>>().into_pyarray(py))?;
        let per_seat = |pick: fn(&State) -> &[i32; crate::state::MAXP]| -> Vec<i32> {
            let mut out = Vec::with_capacity(games * np);
            for st in sts { out.extend_from_slice(&pick(st)[..np]); }
            out
        };
        d.set_item("score", PyArray1::from_vec(py, per_seat(|s| &s.players.score)).reshape([games, np])?)?;
        d.set_item("killed_turn", PyArray1::from_vec(py, per_seat(|s| &s.players.killed_turn)).reshape([games, np])?)?;
        d.set_item("resigned_turn", PyArray1::from_vec(py, per_seat(|s| &s.players.resigned_turn)).reshape([games, np])?)?;
        Ok(())
    }
}

type Layouts<'a> = Option<(&'a [i32], &'a [i32], &'a [i32], &'a [i32])>;

fn layout_slices<'a, 'py>(layouts: &'a Option<(PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>,
                                                 PyReadonlyArray2<'py, i32>)>, games: usize, n: usize, np: usize)
                          -> PyResult<Layouts<'a>> {
    match layouts {
        None => Ok(None),
        Some((a, b, c, d)) => {
            if a.shape() != [games, n] || b.shape() != [games, n] || c.shape() != [games, n] || d.shape() != [games, np] {
                return Err(err("layout arrays must be [games, n] and capitals [games, players]"));
            }
            Ok(Some((a.as_slice()?, b.as_slice()?, c.as_slice()?, d.as_slice()?)))
        }
    }
}

fn other_seats(st: &State) -> impl Iterator<Item = i32> + '_ {
    let np = st.np as i32;
    (1..np).map(move |k| (st.current_player - 1 + k) % np + 1)
}

impl GameBatch {
    /// The games' states changed outside tick(): the cached masks no longer describe them.
    fn forget_masks(&mut self) {
        for c in &mut self.legal {
            c.clear();
        }
    }
}

/// Sparse row deltas for the coded-view wire format (rl/view_codec.py), straight from the arrays the rows live in.
/// arrays: 2-d i16 arrays of one width; src / base: i64 [m, streams, 2] = (array index, row) of output row r's
/// stream s and of the row its delta is taken from ((-1, -1): zeros). Returns the nonzero entries of src - base
/// (wrapping int16, as view_codec's int16 deltas) in row-major order: (row i32, stream i8, col i16, val i16).
#[pyfunction]
#[pyo3(signature = (arrays, src, base, threads=4))]
fn coo_deltas<'py>(py: Python<'py>, arrays: Vec<PyReadonlyArray2<'py, i16>>, src: numpy::PyReadonlyArray3<'py, i64>,
                   base: numpy::PyReadonlyArray3<'py, i64>, threads: usize)
                   -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i16>>, Bound<'py, PyArray1<i16>>)> {
    let shape = src.shape().to_vec();
    if shape.len() != 3 || shape[2] != 2 || base.shape() != shape.as_slice() {
        return Err(err("src and base must be [rows, streams, 2] of the same shape"));
    }
    let (m, streams) = (shape[0], shape[1]);
    let width = if arrays.is_empty() { 0 } else { arrays[0].shape()[1] };
    let mut views: Vec<(&[i16], usize)> = Vec::with_capacity(arrays.len());
    for a in &arrays {
        if a.shape()[1] != width {
            return Err(err("arrays must share one width"));
        }
        views.push((a.as_slice()?, a.shape()[0]));
    }
    let (src, base) = (src.as_slice()?, base.as_slice()?);
    // bounds-check every reference up front
    for v in src.chunks(2).chain(base.chunks(2)) {
        if v[0] >= 0 && (v[0] as usize >= views.len() || v[1] < 0 || v[1] as usize >= views[v[0] as usize].1) {
            return Err(err("row reference out of range"));
        }
    }
    let zeros = vec![0i16; width];
    let chunk = m.div_ceil(threads.max(1) * 4).max(1);
    let parts: Vec<(Vec<i32>, Vec<i8>, Vec<i16>, Vec<i16>)> = py.allow_threads(|| {
        use rayon::prelude::*;
        let run = || (0..m.div_ceil(chunk)).into_par_iter().map(|k| {
            let (mut r_, mut s_, mut c_, mut v_) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for r in k * chunk..((k + 1) * chunk).min(m) {
                for st in 0..streams {
                    let at = (r * streams + st) * 2;
                    let row = |x: &[i64]| -> &[i16] {
                        if x[at] < 0 { &zeros[..] } else {
                            let (a, i) = (views[x[at] as usize].0, x[at + 1] as usize);
                            &a[i * width..(i + 1) * width]
                        }
                    };
                    let (cur, prev) = (row(src), row(base));
                    for c in 0..width {
                        let d = cur[c].wrapping_sub(prev[c]);
                        if d != 0 {
                            r_.push(r as i32);
                            s_.push(st as i8);
                            c_.push(c as i16);
                            v_.push(d);
                        }
                    }
                }
            }
            (r_, s_, c_, v_)
        }).collect::<Vec<_>>();
        match rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build() {
            Ok(pool) => pool.install(run),
            Err(_) => run(),
        }
    });
    let total: usize = parts.iter().map(|p| p.0.len()).sum();
    let (mut r, mut s_, mut c, mut v) = (Vec::with_capacity(total), Vec::with_capacity(total), Vec::with_capacity(total),
                                         Vec::with_capacity(total));
    for p in parts {
        r.extend(p.0);
        s_.extend(p.1);
        c.extend(p.2);
        v.extend(p.3);
    }
    Ok((r.into_pyarray(py), s_.into_pyarray(py), c.into_pyarray(py), v.into_pyarray(py)))
}

#[pymodule]
fn polytopia_engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GameBatch>()?;
    m.add_function(wrap_pyfunction!(coo_deltas, m)?)?;
    m.add("OK", core::OK)?;
    Ok(())
}

#[allow(dead_code)]
fn _unused(_: &PyArray3<i16>) {}


/// Read-only inputs of one tick, shared by the worker threads.
struct TickCtx<'a> {
    rules: &'a Rules,
    geom: &'a Geom,
    max_actions: i32,
    max_turns_opt: Option<i32>,
    max_turns: i32,
    np: usize,
    n: usize,
    vw: usize,
    packed: bool,
    both_views: bool,
    opts: crate::masks::MaskOpts,
    f: &'a [i32],
    s: &'a [i32],
    t: &'a [i32],
    seeds: &'a [i32],
    lay: Option<(&'a [i32], &'a [i32], &'a [i32], &'a [i32])>,
}

fn chunk_or_empty(buf: &mut [i16], width: usize, games: usize) -> Box<dyn Iterator<Item = &mut [i16]> + '_> {
    if width == 0 {
        Box::new((0..games).map(|_| -> &mut [i16] { &mut [] }))
    } else {
        Box::new(buf.chunks_mut(width))
    }
}

/// One game of a tick (GameBatch::tick): step game g, reset it if it ended, then its masks and views.
/// Returns (status, done, terminated, truncated).
#[allow(clippy::too_many_arguments)]
fn tick_game(c: &TickCtx, g: usize, st: &mut State, cache: &mut LegalCache, sc: &mut Scratch, ms: &mut MaskScratch,
             masks: &mut [u8], view: &mut [i16], opp: &mut [i16], final_view: &mut [i16], rewards: &mut [f32])
             -> (i32, bool, bool, bool) {
    let (np, n, vw) = (c.np, c.n, c.vw);
    let a = decode(st, c.f[g], c.s[g], c.t[g]);
    let v = st.state_version;
    // the masks the previous tick returned describe exactly this state (until it is stepped below)
    let legal = if cache.valid { Some(&cache.bits[..]) } else { None };
    let status = prof::time(prof::TICK_STEP, || step_with(st, a, v, c.rules, c.geom, c.max_actions, sc, legal));
    cache.clear();
    let (terminated, truncated) = (st.terminated, st.truncated);
    let done = terminated || truncated;
    let rw = terminal_rewards(st);
    rewards.copy_from_slice(&rw[..np]);
    if truncated {
        prof::time(prof::TICK_FINAL, || {
            for seat in 0..np {
                observe(st, seat as i32 + 1, c.geom, &mut final_view[seat * vw..(seat + 1) * vw]);
            }
        });
    }
    if done {
        *st = prof::time(prof::TICK_RESET, || match &c.lay {
            None => mapgen::reset_players(c.seeds[g], c.geom, c.max_turns_opt, np),
            Some((a, b, cc, d)) => mapgen::initial_state(c.seeds[g] as u32, &a[g * n..(g + 1) * n], &b[g * n..(g + 1) * n],
                                                         &cc[g * n..(g + 1) * n],
                                                         &d[g * np..(g + 1) * np].iter().map(|&x| x as usize).collect::<Vec<usize>>(),
                                                         c.geom, c.max_turns),
        });
    }
    let st = &*st;
    masks.fill(0);   // masks_into only sets the true entries; tick_into reuses its buffers
    prof::time(prof::TICK_MASKS, || if c.packed {
        masks_into(st, c.rules, c.geom, c.opts, ms, &mut Tee { out: &mut Packed(masks), cache });
    } else {
        masks_into(st, c.rules, c.geom, c.opts, ms, &mut Tee { out: &mut Dense(masks), cache });
    });
    cache.valid = true;
    prof::time(prof::TICK_OBSERVE, || observe(st, st.current_player, c.geom, view));
    if c.both_views {
        prof::time(prof::TICK_OBSERVE_OPP, || for (k, seat) in other_seats(st).enumerate() {
            observe(st, seat, c.geom, &mut opp[k * vw..(k + 1) * vw]);
        });
    }
    (status, done, terminated, truncated)
}
