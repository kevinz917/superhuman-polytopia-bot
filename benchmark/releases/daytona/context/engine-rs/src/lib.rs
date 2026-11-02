//! PyO3 bindings: a batch of independent games mirroring rl/cpu_engine.py::CpuGames.

pub mod capture;
pub mod core;
pub mod cov;
pub mod diplomacy;
pub mod discovery;
pub mod economy;
pub mod geom;
pub mod hash;
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

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
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
    max_actions: i32,
    scratch: Scratch,
    mscratch: MaskScratch,
    /// per game: the masks tick() last returned, while they describe the game's state (step validation)
    legal: Vec<LegalCache>,
}

fn err(msg: &str) -> PyErr {
    PyValueError::new_err(msg.to_string())
}

#[pymethods]
impl GameBatch {
    #[new]
    #[pyo3(signature = (size, games, rules_json, engine_revision=None, max_turns=None, no_resign=true, max_actions=10000, players=2))]
    fn new(size: usize, games: usize, rules_json: &str, engine_revision: Option<&str>, max_turns: Option<i32>, no_resign: bool,
           max_actions: i32, players: usize) -> PyResult<Self> {
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
        Ok(GameBatch { size, games, players, rules, geom, states, max_turns, no_resign, max_actions, scratch: Scratch::new(size * size),
                     mscratch: MaskScratch::new(size * size), legal: (0..games).map(|_| LegalCache::new(size * size)).collect() })
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
                                                   self.max_turns.unwrap_or(mapgen::round_limit(self.size)));
        }
        self.forget_masks();
        Ok(())
    }

    /// Only the mapgen layout of one seed (terrain, structure, resource, capitals), for map checks.
    fn layout<'py>(&self, py: Python<'py>, seed: i32) -> PyResult<Bound<'py, PyDict>> {
        let l = mapgen::layout_players(seed, &self.geom, self.players);
        let d = PyDict::new(py);
        d.set_item("terrain", l.terrain.into_pyarray(py))?;
        d.set_item("structure", l.structure.into_pyarray(py))?;
        d.set_item("resource", l.resource.into_pyarray(py))?;
        d.set_item("capitals", l.capitals.iter().map(|&c| c as i32).collect::<Vec<i32>>().into_pyarray(py))?;
        Ok(d)
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
    #[pyo3(signature = (family, source, target, seeds, both_views=true, layouts=None, packed_masks=false))]
    fn tick<'py>(&mut self, py: Python<'py>, family: PyReadonlyArray1<i32>, source: PyReadonlyArray1<i32>,
                 target: PyReadonlyArray1<i32>, seeds: PyReadonlyArray1<i32>, both_views: bool,
                 layouts: Option<(PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>, PyReadonlyArray2<'py, i32>)>,
                 packed_masks: bool)
                 -> PyResult<Bound<'py, PyDict>> {
        let n = self.size * self.size;
        let (f, s, t, seeds) = (family.as_slice()?, source.as_slice()?, target.as_slice()?, seeds.as_slice()?);
        if f.len() != self.games || seeds.len() != self.games {
            return Err(err("choice and seed arrays must have one entry per game"));
        }
        let layouts = match &layouts {
            None => None,
            Some((a, b, c, d)) => {
                let (a, b, c, d) = (a.as_array(), b.as_array(), c.as_array(), d.as_array());
                if a.shape() != [self.games, n] || b.shape() != [self.games, n] || c.shape() != [self.games, n] || d.shape() != [self.games, self.players] {
                    return Err(err("layout arrays must be [games, n] and capitals [games, players]"));
                }
                Some((a, b, c, d))
            }
        };
        let max_turns = self.max_turns.unwrap_or(mapgen::round_limit(self.size));
        let np = self.players;
        let vw = view_width_np(n, np);
        let others = np - 1;
        let mw = if packed_masks { packed_width(n) } else { mask_width(n) };
        // The views are written in full, so they start uninitialised (numpy's PyArray::new contract); the masks
        // only write their true entries and the final views only for truncated games, so those start zeroed.
        let (mut masks, view_arr, opp_arr, mut final_view) = prof::time(prof::TICK_ALLOC, || (
            vec![0u8; self.games * mw],
            unsafe { PyArray2::<i16>::new(py, [self.games, vw], false) },
            unsafe { PyArray2::<i16>::new(py, [if both_views { self.games * others } else { 0 }, vw], false) },
            vec![0i16; self.games * np * vw]));
        // SAFETY: fresh contiguous arrays owned here; every element is written in the loop below before they are
        // returned (view: every game; opp: every game when both_views, else empty).
        let (view, opp) = unsafe { (view_arr.as_slice_mut()?, opp_arr.as_slice_mut()?) };
        let mut done = vec![false; self.games];
        let mut terminated = vec![false; self.games];
        let mut truncated = vec![false; self.games];
        let mut rewards = vec![0f32; self.games * np];
        let mut status = vec![0i32; self.games];
        for g in 0..self.games {
            let st = &mut self.states[g];
            let a = decode(st, f[g], s[g], t[g]);
            let v = st.state_version;
            // the masks the previous tick returned describe exactly this state (until it is stepped below)
            let cache = &mut self.legal[g];
            let legal = if cache.valid { Some(&cache.bits[..]) } else { None };
            status[g] = prof::time(prof::TICK_STEP, || step_with(st, a, v, &self.rules, &self.geom, self.max_actions, &mut self.scratch, legal));
            cache.clear();
            terminated[g] = st.terminated;
            truncated[g] = st.truncated;
            done[g] = st.terminated || st.truncated;
            let rw = terminal_rewards(st);
            rewards[g * np..(g + 1) * np].copy_from_slice(&rw[..np]);
            if st.truncated {
                prof::time(prof::TICK_FINAL, || {
                    for seat in 0..np {
                        let off = (g * np + seat) * vw;
                        observe(st, seat as i32 + 1, &self.geom, &mut final_view[off..off + vw]);
                    }
                });
            }
            if done[g] {
                *st = prof::time(prof::TICK_RESET, || match &layouts {
                    None => mapgen::reset_players(seeds[g], &self.geom, self.max_turns, np),
                    Some((a, b, c, d)) => mapgen::initial_state(seeds[g] as u32, &a.row(g).to_vec(), &b.row(g).to_vec(), &c.row(g).to_vec(),
                                                                &d.row(g).iter().map(|&c| c as usize).collect::<Vec<usize>>(), &self.geom, max_turns),
                });
            }
            let st = &self.states[g];
            let row = &mut masks[g * mw..(g + 1) * mw];
            let (rules, geom, no_resign, ms, cache) = (&self.rules, &self.geom, self.no_resign, &mut self.mscratch, &mut self.legal[g]);
            prof::time(prof::TICK_MASKS, || if packed_masks {
                masks_into(st, rules, geom, no_resign, ms, &mut Tee { out: &mut Packed(row), cache });
            } else {
                masks_into(st, rules, geom, no_resign, ms, &mut Tee { out: &mut Dense(row), cache });
            });
            self.legal[g].valid = true;
            prof::time(prof::TICK_OBSERVE, || observe(st, st.current_player, geom, &mut view[g * vw..(g + 1) * vw]));
            if both_views {
                prof::time(prof::TICK_OBSERVE_OPP, || for (k, seat) in other_seats(st).enumerate() {
                    let off = (g * others + k) * vw;
                    observe(st, seat, geom, &mut opp[off..off + vw]);
                });
            }
        }
        prof::time(prof::TICK_OUTPUT, || {
        let d = PyDict::new(py);
        d.set_item("masks", PyArray1::from_vec(py, masks).reshape([self.games, mw])?)?;
        d.set_item("view", view_arr)?;
        if both_views {
            if np > 2 {
                d.set_item("opp_view", opp_arr.reshape([self.games, others, vw])?)?;
            } else {
                d.set_item("opp_view", opp_arr)?;
            }
        }
        d.set_item("final_view", PyArray1::from_vec(py, final_view).reshape([self.games, np, vw])?)?;
        d.set_item("done", done.into_pyarray(py))?;
        d.set_item("terminated", terminated.into_pyarray(py))?;
        d.set_item("truncated", truncated.into_pyarray(py))?;
        d.set_item("rewards", PyArray1::from_vec(py, rewards).reshape([self.games, np])?)?;
        d.set_item("status", status.into_pyarray(py))?;
        Ok(d)
        })
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
        time("full_count", &mut |st| masks_into(st, r, g, self.no_resign, &mut ms.borrow_mut(), &mut c))?;
        time("full", &mut |st| legal_masks(st, r, g, self.no_resign, &mut ms.borrow_mut(), &mut flat))?;
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
                masks_into(&self.states[g], &self.rules, &self.geom, self.no_resign, &mut self.mscratch, &mut Packed(row));
            } else {
                masks_into(&self.states[g], &self.rules, &self.geom, self.no_resign, &mut self.mscratch, &mut Dense(row));
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
        field!("players.hostile_until", i32, [np], |s, b| b.extend_from_slice(&s.players.hostile_until[..s.np]));
        field!("players.monuments", bool, [np, 128], |s, b| for k in 0..s.np { b.extend_from_slice(&s.players.monuments[k]); });
        if np > 2 {
            field!("players.out_order", i32, [np], |s, b| b.extend_from_slice(&s.players.out_order[..s.np]));
        }
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
        load!("players.hostile_until", i32, np, |s, v| s.players.hostile_until[..s.np].copy_from_slice(v));
        load!("players.monuments", bool, np * 128, |s, v| for k in 0..s.np { s.players.monuments[k].copy_from_slice(&v[k * 128..(k + 1) * 128]); });
        if np > 2 {
            load!("players.out_order", i32, np, |s, v| s.players.out_order[..s.np].copy_from_slice(v));
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

#[pymodule]
fn polytopia_engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GameBatch>()?;
    m.add("OK", core::OK)?;
    Ok(())
}

#[allow(dead_code)]
fn _unused(_: &PyArray3<i16>) {}
