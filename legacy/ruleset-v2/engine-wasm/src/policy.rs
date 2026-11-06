//! The hierarchical policy over the engine's flat action catalog (rl/actions.py, rl/network.py::sample).

use polytopia_engine_rs::core::{family_row_widths, family_widths};
use polytopia_engine_rs::state::State;

use crate::net::{argmax, log_distribution, sample, Buffers, Net, Trunk, BOARD, FAMILIES, GLOBAL};
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
        Policy { net, size, buffers: Buffers::new(n), board: vec![0.; n * BOARD], global: vec![0.; GLOBAL] }
    }

    /// Trunk of `player`'s packed public view of `s`.
    pub fn trunk(&mut self, e: &Engine, s: &State, player: i32) -> Trunk {
        let v = packed(s, player, &e.geom);
        features(&e.layout, &v, self.size, &e.structure_ids, &e.unit_ids, &mut self.board, &mut self.global);
        self.net.trunk(self.size, &self.board, &self.global, &mut self.buffers)
    }

    /// The served bot's decision (Studio's StructuredPolicy with no_resign): resignation is masked out.
    pub fn act(&mut self, e: &mut Engine, s: &State, deterministic: bool, rng: &mut Rng) -> Result<Decision, String> {
        if s.terminated || s.truncated {
            return Err("the game is over".into());
        }
        let draw = if deterministic { Draw::Greedy } else { Draw::Sample(1.) };
        let trunk = self.trunk(e, s, s.current_player);
        let mask = e.legal(s, true).to_vec();
        Ok(self.decide(&e.catalog, &trunk, &mask, draw, rng))
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
