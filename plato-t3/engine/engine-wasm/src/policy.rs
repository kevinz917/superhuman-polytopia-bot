//! The hierarchical policy over the engine's flat action catalog (rl/actions.py, rl/network.py::sample).

use polytopia_engine_rs::core::{family_row_widths, family_widths};
use polytopia_engine_rs::state::State;

use crate::net::{argmax, log_distribution, sample, Buffers, Net, Trunk, FAMILIES};
use crate::search::Rng;
use crate::view::{features, packed};
use crate::Engine;

/// rl/actions.py::shapes and offsets: family f is a [rows, width] matrix at offset[f] of the flat catalog.
pub struct Catalog {
    pub n: usize,
    pub offset: [usize; FAMILIES + 1],
    pub rows: [usize; FAMILIES],
    pub width: [usize; FAMILIES],
}

impl Catalog {
    pub fn new(n: usize) -> Catalog {
        let sizes = family_widths(n);
        let mut width = family_row_widths();
        width[0] = n;
        width[1] = n;
        let mut offset = [0; FAMILIES + 1];
        let mut rows = [0; FAMILIES];
        for f in 0..FAMILIES {
            offset[f + 1] = offset[f] + sizes[f];
            rows[f] = sizes[f] / width[f];
        }
        Catalog { n, offset, rows, width }
    }

    pub fn unflatten(&self, index: usize) -> (usize, usize, usize) {
        let f = (0..FAMILIES).rfind(|&f| self.offset[f] <= index).unwrap();
        let at = index - self.offset[f];
        (f, at / self.width[f], at % self.width[f])
    }

    pub fn flatten(&self, f: usize, s: usize, t: usize) -> usize {
        self.offset[f] + s * self.width[f] + t
    }

    pub fn family_mask(&self, m: &[u8]) -> Vec<bool> {
        (0..FAMILIES).map(|f| m[self.offset[f]..self.offset[f + 1]].iter().any(|&b| b != 0)).collect()
    }

    /// Rows of family f with a legal entry, padded to n.
    pub fn source_mask(&self, m: &[u8], f: usize) -> Vec<bool> {
        let mut out = vec![false; self.n];
        for r in 0..self.rows[f] {
            let at = self.offset[f] + r * self.width[f];
            out[r] = m[at..at + self.width[f]].iter().any(|&b| b != 0);
        }
        out
    }

    /// Row min(s, rows - 1) of family f, padded to max(n, 128).
    pub fn target_mask(&self, m: &[u8], f: usize, s: usize) -> Vec<bool> {
        let mut out = vec![false; self.n.max(128)];
        let at = self.offset[f] + s.min(self.rows[f] - 1) * self.width[f];
        for (o, &b) in out.iter_mut().zip(&m[at..at + self.width[f]]) {
            *o = b != 0;
        }
        out
    }
}

pub struct Decision {
    pub index: usize,
    pub family: usize,
    pub value: f32,
    pub logp: f32,
}

/// How the policy draws each level: the argmax, or a sample at a temperature.
#[derive(Clone, Copy)]
pub enum Draw {
    Greedy,
    Sample(f32),
}

pub struct Policy {
    pub net: Net,
    size: usize,
    buffers: Buffers,
    board: Vec<f32>,
    global: Vec<f32>,
}

impl Policy {
    pub fn new(net: Net, size: usize) -> Policy {
        let n = size * size;
        let buffers = Buffers::new(n, net.width);
        let board = vec![0.; n * net.board_channels()];
        let global = vec![0.; net.global_channels()];
        Policy { net, size, buffers, board, global }
    }

    /// Buffers for a `size` x `size` board (lib.rs select).
    pub fn resize(&mut self, size: usize) {
        if size != self.size {
            let n = size * size;
            self.size = size;
            self.buffers = Buffers::new(n, self.net.width);
            self.board = vec![0.; n * self.net.board_channels()];
        }
    }

    /// Trunk of `player`'s packed public view of `s`.
    pub fn trunk(&mut self, e: &Engine, s: &State, player: i32) -> Trunk {
        let v = packed(s, player, &e.geom);
        features(&e.layout, &v, self.size, &e.structure_ids, &e.unit_ids, &mut self.board, &mut self.global);
        self.net.trunk(self.size, &self.board, &self.global, &mut self.buffers)
    }

    /// The served bot's decision (Studio's StructuredPolicy with no_resign): resignation is masked out.
    pub fn act(&mut self, e: &mut Engine, s: &State, deterministic: bool, rng: &mut Rng) -> Result<Decision, String> {
        self.act_mode(e, s, if deterministic { 1 } else { 0 }, rng)
    }

    /// mode 0: ancestral sample at temperature 1; 1: hierarchical argmax; 2: global joint top-1 (the T3 reference chooser).
    pub fn act_mode(&mut self, e: &mut Engine, s: &State, mode: i32, rng: &mut Rng) -> Result<Decision, String> {
        if s.terminated || s.truncated {
            return Err("the game is over".into());
        }
        let trunk = self.trunk(e, s, s.current_player);
        // The candidates come from the mover's public view only (public.rs): an unseen enemy cannot
        // change them. The caller referees the one proposed action against the true state; there is no retry.
        let mask = e.legal(&crate::public::public_state(s, s.current_player), true).to_vec();
        Ok(match mode {
            2 => {
                let mask = joint_catalogue(e, s, mask);
                self.decide_joint(&e.catalog, &trunk, &mask)
            }
            1 => self.decide(&e.catalog, &trunk, &mask, Draw::Greedy, rng),
            _ => self.decide(&e.catalog, &trunk, &mask, Draw::Sample(1.), rng),
        })
    }

    /// The T3 reference chooser: over every legal
    /// flat id in ascending order, logp = log P(f) + log P(s | f) + log P(t | f, s) in float32 (left to right), and
    /// the first maximum of exp(logp). Hierarchical argmax (Draw::Greedy) can pick a different action.
    pub fn decide_joint(&self, c: &Catalog, t: &Trunk, mask: &[u8]) -> Decision {
        let family = log_distribution(&self.net.family_logits(t), &c.family_mask(mask));
        let mut sources: Vec<Option<Vec<f32>>> = vec![None; FAMILIES];
        let mut targets: std::collections::HashMap<(usize, usize), Vec<f32>> = std::collections::HashMap::new();
        let (mut best, mut best_p, mut best_lp) = (usize::MAX, f32::NEG_INFINITY, 0f32);
        for (index, _) in mask.iter().enumerate().filter(|(_, &m)| m != 0) {
            let (f, s, tg) = c.unflatten(index);
            let ls = sources[f].get_or_insert_with(|| log_distribution(&self.net.source_logits(t, f), &c.source_mask(mask, f)))[s];
            let lt = targets.entry((f, s)).or_insert_with(|| log_distribution(&self.net.target_logits(t, f, s), &c.target_mask(mask, f, s)))[tg];
            let lp = family[f] + ls + lt;
            let p = lp.exp();
            if best == usize::MAX || p > best_p {
                (best, best_p, best_lp) = (index, p, lp);
            }
        }
        let (f, _, _) = c.unflatten(best);
        Decision { index: best, family: f, value: t.value, logp: best_lp }
    }

    /// Every legal candidate's joint log probability (ascending flat id), for the parity test and telemetry.
    pub fn joint_logp(&self, c: &Catalog, t: &Trunk, mask: &[u8]) -> Vec<(usize, f32)> {
        let family = log_distribution(&self.net.family_logits(t), &c.family_mask(mask));
        mask.iter().enumerate().filter(|(_, &m)| m != 0).map(|(index, _)| {
            let (f, s, tg) = c.unflatten(index);
            let ls = log_distribution(&self.net.source_logits(t, f), &c.source_mask(mask, f))[s];
            let lt = log_distribution(&self.net.target_logits(t, f, s), &c.target_mask(mask, f, s))[tg];
            (index, family[f] + ls + lt)
        }).collect()
    }

    /// network.sample / search.tempered_sample from a computed trunk and flat legal mask.
    pub fn decide(&self, c: &Catalog, t: &Trunk, mask: &[u8], draw: Draw, rng: &mut Rng) -> Decision {
        let pick = |logits: &[f32], legal: &[bool], rng: &mut Rng| -> (usize, f32) {
            let lp = log_distribution(logits, legal);
            let i = match draw {
                Draw::Greedy => argmax(&lp),
                Draw::Sample(temp) => {
                    let scaled: Vec<f32> = lp.iter().map(|l| l / temp).collect();
                    sample(&log_distribution(&scaled, legal), rng.uniform())
                }
            };
            (i, lp[i])
        };
        let (f, lf) = pick(&self.net.family_logits(t), &c.family_mask(mask), rng);
        let (s, ls) = pick(&self.net.source_logits(t, f), &c.source_mask(mask, f), rng);
        let (tg, lt) = pick(&self.net.target_logits(t, f, s), &c.target_mask(mask, f, s), rng);
        Decision { index: c.flatten(f, s, tg), family: f, value: t.value, logp: lf + ls + lt }
    }
}

/// Raw kind-3 (Diplomacy) parameters the T3 reference chooser never offers: it drops the diplomacy
/// family parameters (18, 19, 20, 21, 22), which are not qualified against the real game,
/// and the frozen evaluation chooser applies the same filter.
pub const DIPLOMACY_KIND: i32 = 3;
pub const DIPLOMACY_DROPPED: [i32; 5] = [18, 19, 20, 21, 22];

/// The joint chooser's catalogue (12 Nov: the site's T3 chooser equals the frozen public chooser and
/// filter): the mover's public legal ids minus Diplomacy parameters 18-22. Each id is decoded on the mover's PUBLIC
/// state, so nothing hidden decides what is dropped. Resignation is already off (legal(.., no_resign = true)); the
/// family/source/target masks are rebuilt from what remains (decide_joint), so the distributions renormalise as in the reference.
///
/// One case the frozen chooser does not define: a pending peace request must be answered before anything else
/// (diplomacy.rs), so the public catalogue holds only Accept/Decline (kind 3, parameter 19). The filter then leaves
/// nothing and the reference chooser raises ('invalid or empty public catalogue'); it never meets that case (no peace
/// requests in its games). The site lets a player offer peace, so there the joint top-1 runs over the unfiltered
/// public catalogue (the answer is still the network's own joint choice). `joint_catalogue_fallback` reports it.
pub fn joint_catalogue(e: &Engine, s: &State, mask: Vec<u8>) -> Vec<u8> {
    joint_catalogue_fallback(e, s, mask).0
}

/// joint_catalogue, and whether the filter left nothing so the unfiltered public catalogue was kept.
pub fn joint_catalogue_fallback(e: &Engine, s: &State, mask: Vec<u8>) -> (Vec<u8>, bool) {
    let filtered = diplomacy_filtered(e, s, mask.clone());
    if filtered.iter().all(|&m| m == 0) && mask.iter().any(|&m| m != 0) {
        (mask, true)
    } else {
        (filtered, false)
    }
}

fn diplomacy_filtered(e: &Engine, s: &State, mut mask: Vec<u8>) -> Vec<u8> {
    let public = crate::public::public_state(s, s.current_player);
    for index in 0..mask.len() {
        if mask[index] == 0 {
            continue;
        }
        let (f, src, t) = e.catalog.unflatten(index);
        let a = polytopia_engine_rs::core::decode(&public, f as i32, src as i32, t as i32);
        if a.kind == DIPLOMACY_KIND && DIPLOMACY_DROPPED.contains(&a.parameter) {
            mask[index] = 0;
        }
    }
    mask
}

