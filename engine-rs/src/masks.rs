//! gpu/actions.py::legal_masks, written into one flat boolean buffer in catalog order
//! (move, attack, summon, build, research, harvest, forest, reward, capture, recover, promote, disband, heal,
//! destroy, diplomatic, end, resign).
//!
//! The mask is mostly zeros (a few hundred true entries out of 130,868 on 14x14), so every family only emits its
//! true entries into a zeroed sink: dense 0/1 bytes, or bits in numpy.packbits order (the inference server's wire
//! format). Nothing is written for false entries.

use crate::capture;
use crate::core::family_widths;
use crate::diplomacy;
use crate::geom::Geom;
use crate::kernels::{max_health, movement_flags, reach, research_bits, MoveScratch};
use crate::military::{self, SummonCtx};
use crate::rules::Rules;
use crate::state::State;
use crate::structures::{self, BuildCtx};

pub fn mask_width(n: usize) -> usize {
    family_widths(n).iter().sum()
}

/// Bytes per game of the packed masks: ceil(mask_width / 8).
pub fn packed_width(n: usize) -> usize {
    (mask_width(n) + 7) / 8
}

pub fn family_offsets(n: usize) -> [usize; 18] {
    let w = family_widths(n);
    let mut o = [0usize; 18];
    for i in 0..17 {
        o[i + 1] = o[i] + w[i];
    }
    o
}

/// Receives the true entries of one game's flat mask.
pub trait Sink {
    fn set(&mut self, i: usize);
}

/// 0/1 bytes; the buffer must be zeroed.
pub struct Dense<'a>(pub &'a mut [u8]);

impl Sink for Dense<'_> {
    #[inline]
    fn set(&mut self, i: usize) {
        self.0[i] = 1;
    }
}

/// Bits, most significant first (numpy.packbits' default order); the buffer must be zeroed.
pub struct Packed<'a>(pub &'a mut [u8]);

impl Sink for Packed<'_> {
    #[inline]
    fn set(&mut self, i: usize) {
        self.0[i >> 3] |= 0x80 >> (i & 7);
    }
}

/// Buffers reused across legal_masks calls (one per GameBatch).
pub struct MaskScratch {
    pub mv: MoveScratch,
    pub build: BuildCtx,
    pub summon: SummonCtx,
}

impl MaskScratch {
    pub fn new(n: usize) -> MaskScratch {
        MaskScratch { mv: MoveScratch::new(n), build: BuildCtx::new(n), summon: SummonCtx::new(n) }
    }
}

/// Dense masks into `out` (any contents; the first mask_width bytes are overwritten).
pub fn legal_masks(s: &State, r: &Rules, g: &Geom, no_resign: bool, ms: &mut MaskScratch, out: &mut [u8]) {
    out[..mask_width(s.n)].fill(0);
    masks_into(s, r, g, no_resign, ms, &mut Dense(out));
}

/// Forwards to another sink and records every entry in a persistent packed bitset (LegalCache).
pub struct Tee<'a, S: Sink> {
    pub out: &'a mut S,
    pub cache: &'a mut LegalCache,
}

impl<S: Sink> Sink for Tee<'_, S> {
    #[inline]
    fn set(&mut self, i: usize) {
        self.out.set(i);
        self.cache.bits[i >> 3] |= 0x80 >> (i & 7);
        self.cache.list.push(i as u32);
    }
}

/// One game's legal masks from the last tick, kept for the next step's validation (core::step_with): packed bits
/// (masks::Packed layout), the list of set entries (to clear them cheaply) and whether they belong to the game's
/// current state.
pub struct LegalCache {
    pub bits: Vec<u8>,
    list: Vec<u32>,
    pub valid: bool,
}

impl LegalCache {
    pub fn new(n: usize) -> LegalCache {
        LegalCache { bits: vec![0; packed_width(n)], list: Vec::new(), valid: false }
    }
    /// Forget the cached masks (they no longer describe the state).
    pub fn clear(&mut self) {
        for &i in &self.list {
            self.bits[(i >> 3) as usize] = 0;
        }
        self.list.clear();
        self.valid = false;
    }
}

/// Counts the true entries (profiling: the mask computation without its memory traffic).
pub struct Count(pub usize);

impl Sink for Count {
    #[inline]
    fn set(&mut self, _: usize) {
        self.0 += 1;
    }
}

/// Emit every true entry of actions.legal_masks (CpuGames: resign forced False when no_resign).
pub fn masks_into<S: Sink>(s: &State, r: &Rules, g: &Geom, no_resign: bool, ms: &mut MaskScratch, out: &mut S) {
    let active = s.core_supported && !s.terminated && !s.truncated;
    if !active {
        return;
    }
    let o = family_offsets(s.n);
    if diplomacy::peace_pending(s) {
        // a pending peace request is answered before anything else (accept or decline; resigning stays)
        for k in [1, 2] {
            out.set(o[14] + k);
        }
        if !no_resign && s.np == 2 {
            out.set(o[16]);
        }
        return;
    }
    // reward is gated by `active` only; any reward entry is also State::pending_reward, which gates the rest
    if reward_rows(s, out) {
        return;
    }
    move_rows(s, r, g, &mut ms.mv, out);
    attack_rows(s, r, g, out);
    summon_rows(s, r, &mut ms.summon, out);
    build_rows(s, r, g, &mut ms.build, out);
    let mut bits = research_bits(s, r);
    while bits != 0 {
        out.set(o[4] + bits.trailing_zeros() as usize);
        bits &= bits - 1;
    }
    economy_rows(s, &ms.build.econ_list, out);
    unit_rows(s, r, g, out);
    let other = 3 - s.current_player;
    let capital = diplomacy::opponent_capital(s);
    let abilities = [18, 19, 19, 20, 21, 22];
    let targets = [-1, 0, 1, capital.0 as i32, -1, -1];
    for k in 0..6 {
        if diplomacy::legal_with(s, other, targets[k], abilities[k], capital) {
            out.set(o[14] + k);
        }
    }
    out.set(o[15]);
    // resigning exists only in two-seat games
    if !no_resign && s.np == 2 {
        out.set(o[16]);
    }
}

/// City rewards: the two kinds of the next slot of every own city with one pending. Returns whether any is
/// pending (State::pending_reward).
pub fn reward_rows<S: Sink>(s: &State, out: &mut S) -> bool {
    let o = family_offsets(s.n);
    let (c, cp) = (&s.cities, s.current_player);
    let mut pending = false;
    for i in 0..s.n {
        if c.owner[i] == cp {
            let count = s.reward_count(i);
            if count < c.level[i] - 1 {
                pending = true;
                let (a, b) = match count + 1 { 1 => (3, 4), 2 => (1, 7), 3 => (5, 8), _ => (2, 6) };
                out.set(o[7] + i * 9 + a);
                out.set(o[7] + i * 9 + b);
            }
        }
    }
    pending
}

/// Move rows of the mover's unmoved units (kernels.movement_targets' remaining gate: a supported kind).
pub fn move_rows<S: Sink>(s: &State, r: &Rules, g: &Geom, mv: &mut MoveScratch, out: &mut S) {
    let (n, u, cp) = (s.n, &s.units, s.current_player);
    let o = family_offsets(n);
    let movable = |i: usize| u.owner[i] == cp && !u.moved[i] && r.unit_supported[u.kind[i] as usize];
    let Some(first) = (0..n).find(|&i| movable(i)) else {
        return;
    };
    movement_flags(s, cp, g, &mut mv.flags);
    for i in first..n {
        if movable(i) {
            let base = o[0] + i * n;
            reach(s, i, r, g, mv, |j| out.set(base + j));
        }
    }
}

pub fn attack_rows<S: Sink>(s: &State, r: &Rules, g: &Geom, out: &mut S) {
    let (n, u, cp) = (s.n, &s.units, s.current_player);
    let o = family_offsets(n);
    for i in 0..n {
        if u.owner[i] == cp && !u.attacked[i] {
            let base = o[1] + i * n;
            military::attack_targets(s, i, r, g, |j| out.set(base + j));
        }
    }
}

pub fn summon_rows<S: Sink>(s: &State, r: &Rules, ctx: &mut SummonCtx, out: &mut S) {
    let (n, u, cp) = (s.n, &s.units, s.current_player);
    let o = family_offsets(n);
    ctx.prepare(s, r);
    for i in 0..n {
        if s.cities.owner[i] == cp || u.owner[i] == cp {
            let mut bits = military::summon_row(s, ctx, i) & r.supported_bits;
            while bits != 0 {
                out.set(o[2] + i * 128 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
    }
}

/// Build rows; leaves economic_tiles(s) in `ctx.econ` for the economy rows.
pub fn build_rows<S: Sink>(s: &State, r: &Rules, g: &Geom, ctx: &mut BuildCtx, out: &mut S) {
    let o = family_offsets(s.n);
    ctx.prepare(s, r, g);
    // outside the economy only the neutral road and the neutral bridge can be built
    {
        for i in 0..s.n {
            if !ctx.econ[i] && structures::neutral_road(s, r, ctx, i) {
                out.set(o[3] + i * 128 + 3);
            }
            if !ctx.econ[i] && structures::neutral_bridge(s, r, ctx, i) {
                out.set(o[3] + i * 128 + 48);
            }
        }
    }
    for &i in &ctx.econ_list {
        let i = i as usize;
        let mut bits = structures::build_row(s, r, g, ctx, i) & !(1u128 << 37);
        while bits != 0 {
            out.set(o[3] + i * 128 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}

/// Harvest, forest and destroy (the economy tiles, as listed by BuildCtx::prepare).
pub fn economy_rows<S: Sink>(s: &State, econ: &[u16], out: &mut S) {
    let o = family_offsets(s.n);
    let t = &s.tiles;
    let pi = s.p();
    let tech = &s.players.tech[pi];
    let stars = s.players.stars[pi];
    for &i in econ {
        let i = i as usize;
        let (resource, structure, terrain) = (t.resource[i], t.structure[i], t.terrain[i]);
        let unlocked = (resource == 1 && tech[15]) || (resource == 3 && tech[10]) || (resource == 6 && tech[6]);
        if structure == 0 && unlocked && stars >= 2 {
            out.set(o[5] + i);
        }
        // forest: clear (3 stars), burn, grow
        if structure == 0 || structure == 3 {
            if terrain == 5 && tech[9] && stars >= 3 {
                out.set(o[6] + i * 3);
            }
            if terrain == 5 && tech[16] {
                out.set(o[6] + i * 3 + 1);
            }
            if terrain == 3 && tech[19] && stars >= 5 {
                out.set(o[6] + i * 3 + 2);
            }
        }
        if structures::destructible(structure) && tech[3] {
            out.set(o[13] + i);
        }
    }
}

/// Capture, recover, promote, disband, heal (the mover's units only).
pub fn unit_rows<S: Sink>(s: &State, r: &Rules, g: &Geom, out: &mut S) {
    let o = family_offsets(s.n);
    let (u, cp) = (&s.units, s.current_player);
    let tech2 = s.players.tech[s.p()][2];
    for i in 0..s.n {
        if u.owner[i] != cp {
            continue;
        }
        let idle = !u.moved[i] && !u.attacked[i];
        if idle && capture::capturable(s, i) {
            out.set(o[8] + i);
        }
        if idle && u.health[i] < max_health(s, r, i) {
            out.set(o[9] + i);
        }
        if crate::military::promotable(s, r, i) {
            out.set(o[10] + i);
        }
        if idle && tech2 {
            out.set(o[11] + i);
        }
        if idle && u.kind[i] == 10 && g.adj[i].iter().any(|&j| {
            let j = j as usize;
            u.owner[j] == cp && u.health[j] < max_health(s, r, j)
        }) {
            out.set(o[12] + i);
        }
    }
}
