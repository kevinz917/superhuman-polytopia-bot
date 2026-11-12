//! rl/network.py (architecture normalized-v2, any number of residual blocks, width 64 or 128: the 2x64 site bots
//! and hero1's 4x128) on the CPU: the conv trunk, the value head and the masked autoregressive family -> source ->
//! target heads. fp32 throughout; sums run in a different order than XLA's, so logits match the JAX policy to
//! rounding, not bitwise. Width and depth are read from the weight file.

pub const BOARD: usize = 138;
pub const GLOBAL: usize = 397;
pub const FAMILIES: usize = 17;
const PARAMETER_TARGETS: usize = 128;
/// Trunk widths the convolution is compiled for (its accumulator is a fixed-size array per width).
pub const WIDTHS: [usize; 2] = [64, 128];

pub struct Net {
    v3: Option<crate::v3::V3>,
    /// Channels of the trunk (64 or 128).
    pub width: usize,
    blocks: Vec<(Vec<f32>, Vec<f32>)>,
    critic_b: Vec<f32>,
    critic_w: Vec<f32>,
    family_b: Vec<f32>,
    family_w: Vec<f32>,
    family_embedding: Vec<f32>,
    global_b: Vec<f32>,
    global_w: Vec<f32>,
    norm_bias: Vec<f32>,
    norm_scale: Vec<f32>,
    parameter_embedding: Vec<f32>,
    embedding_rows: usize,
    query_b: Vec<f32>,
    query_w: Vec<f32>,
    source_b: Vec<f32>,
    source_w: Vec<f32>,
    stem: Vec<f32>,
    value_b: Vec<f32>,
    value_w: Vec<f32>,
}

/// The trunk's outputs for one position: per-tile features, the global vector and the critic value.
pub struct Trunk {
    pub tiles: Vec<f32>,
    pub g: Vec<f32>,
    pub value: f32,
}

/// Reusable activation buffers (one per worker).
pub struct Buffers {
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
}

impl Buffers {
    pub fn new(n: usize, width: usize) -> Buffers {
        Buffers { a: vec![0.; n * width], b: vec![0.; n * width], c: vec![0.; n * width] }
    }
}

impl Net {
    /// Parses scripts/export_browser_weights.py output: b"PTB1", count, shapes, then float32 data.
    pub fn parse(bytes: &[u8]) -> Result<Net, String> {
        let u32_at = |at: usize| -> Result<usize, String> {
            bytes.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize).ok_or("truncated header".into())
        };
        let v3 = bytes.get(..4) == Some(b"PTB3");
        if !v3 && bytes.get(..4) != Some(b"PTB1") {
            return Err("not a browser weight file".into());
        }
        let count = u32_at(4)?;
        if count > 256 { return Err("too many tensors".into()); }
        let mut at = 8;
        let mut shapes = Vec::with_capacity(count);
        for _ in 0..count {
            let rank = u32_at(at)?;
            at += 4;
            if rank > 4 { return Err("invalid tensor rank".into()); }
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                dims.push(u32_at(at)?);
                at += 4;
            }
            shapes.push(dims);
        }
        let mut tensors = Vec::with_capacity(count);
        for dims in &shapes {
            let len = dims.iter().try_fold(1usize, |a, &b| a.checked_mul(b)).ok_or("tensor too large")?;
            if len > 16_000_000 { return Err("tensor too large".into()); }
            let raw = bytes.get(at..at + 4 * len).ok_or("truncated tensor data")?;
            tensors.push(raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<f32>>());
            at += 4 * len;
        }
        if at != bytes.len() {
            return Err("trailing bytes after tensors".into());
        }
        if tensors.iter().flatten().any(|x| !x.is_finite()) { return Err("non-finite weights".into()); }
        if v3 {
            crate::v3::validate(&shapes)?;
            let (vb, vg) = crate::v3::widths(&shapes).ok_or("unsupported Plato observation widths")?;
            let mut take = |i: usize| std::mem::take(&mut tensors[i]);
            return Ok(Net {
                width: 128, blocks: Vec::new(), critic_b: Vec::new(), critic_w: Vec::new(),
                family_b: take(0), family_w: take(1), family_embedding: take(2),
                global_b: Vec::new(), global_w: Vec::new(), norm_bias: take(5), norm_scale: take(6),
                parameter_embedding: take(7), embedding_rows: 196, query_b: take(8), query_w: take(9),
                source_b: take(10), source_w: take(11), stem: Vec::new(), value_b: Vec::new(), value_w: Vec::new(),
                v3: Some(crate::v3::V3::new(tensors, vb, vg)),
            });
        }
        // leaf order (jax.tree.flatten, sorted keys): blocks.0.a, blocks.0.b, ..., then the 17 head leaves; the
        // stem [3, 3, BOARD, w] is the 3rd from last and gives the width
        if count < 19 || (count - 17) % 2 != 0 {
            return Err(format!("unexpected tensor count {count}"));
        }
        let nblocks = (count - 17) / 2;
        let w = shapes[count - 3].get(3).copied().unwrap_or(0);
        if !WIDTHS.contains(&w) {
            return Err(format!("unsupported trunk width {w} (supported: {WIDTHS:?})"));
        }
        let pooled = 2 * w + GLOBAL;
        let h = 2 * nblocks;
        let rows = shapes.get(h + 9).and_then(|s| s.get(1)).copied().unwrap_or(0);
        let mut expect: Vec<Vec<usize>> = (0..h).map(|_| vec![3, 3, w, w]).collect();
        expect.extend([
            vec![w], vec![pooled, w], vec![FAMILIES], vec![w, FAMILIES], vec![FAMILIES, w], vec![w], vec![pooled, w],
            vec![w], vec![w], vec![FAMILIES, rows, w], vec![w], vec![2 * w, w], vec![FAMILIES], vec![w, FAMILIES],
            vec![3, 3, BOARD, w], vec![1], vec![w, 1],
        ]);
        if shapes != expect || rows < PARAMETER_TARGETS {
            return Err(format!("unexpected tensor shapes {shapes:?}"));
        }
        let mut t = tensors.into_iter();
        let mut next = || t.next().unwrap();
        Ok(Net {
            v3: None,
            width: w,
            blocks: (0..nblocks).map(|_| (next(), next())).collect(),
            critic_b: next(),
            critic_w: next(),
            family_b: next(),
            family_w: next(),
            family_embedding: next(),
            global_b: next(),
            global_w: next(),
            norm_bias: next(),
            norm_scale: next(),
            parameter_embedding: next(),
            embedding_rows: rows,
            query_b: next(),
            query_w: next(),
            source_b: next(),
            source_w: next(),
            stem: next(),
            value_b: next(),
            value_w: next(),
        })
    }

    pub fn board_channels(&self) -> usize { self.v3.as_ref().map_or(BOARD, |v| v.board) }
    pub fn global_channels(&self) -> usize { self.v3.as_ref().map_or(GLOBAL, |v| v.global) }
    /// The observation version these widths are (rl/observation.py at c7e34799; 6 at b81241e1): 1 (138, 397),
    /// 3 (151, 525), 4 (153, 525), 5 (153, 555) or 6 (174, 556). view::features encodes exactly that version; for 6 the
    /// special-tribe inputs (21 board channels, the Convert task entry) are zero, as in every game of two regular
    /// tribes (the only games the site plays).
    pub fn obs_version(&self) -> u32 {
        match (self.board_channels(), self.global_channels()) {
            (151, 525) => 3,
            (153, 525) => 4,
            (153, 555) => 5,
            (174, 556) => 6,
            _ => 1,
        }
    }

    pub fn trunk(&self, size: usize, board: &[f32], global: &[f32], buf: &mut Buffers) -> Trunk {
        if let Some(v3) = &self.v3 { return v3.trunk(size, board, global, buf); }
        let n = size * size;
        let w = self.width;
        let Buffers { a, b, c } = buf;
        conv(size, BOARD, w, board, &self.stem, a);
        a[..n * w].chunks_exact_mut(w).for_each(silu_norm);
        for (k1, k2) in &self.blocks {
            conv(size, w, w, a, k1, b);
            b[..n * w].chunks_exact_mut(w).for_each(silu_norm);
            conv(size, w, w, b, k2, c);
            for (x, y) in a[..n * w].iter_mut().zip(c.iter()) {
                *x += *y;
            }
            a[..n * w].chunks_exact_mut(w).for_each(silu_norm);
        }
        let tiles = a[..n * w].to_vec();
        let mut pooled = vec![0f32; 2 * w + GLOBAL];
        {
            let (mean, rest) = pooled.split_at_mut(w);
            let (max, glob) = rest.split_at_mut(w);
            max.fill(f32::NEG_INFINITY);
            for row in tiles.chunks_exact(w) {
                for k in 0..w {
                    mean[k] += row[k];
                    max[k] = max[k].max(row[k]);
                }
            }
            mean.iter_mut().for_each(|m| *m /= n as f32);
            glob.copy_from_slice(global);
        }
        let mut g = dense(&pooled, &self.global_w, &self.global_b);
        layer_norm(&mut g);
        for k in 0..w {
            g[k] = silu(g[k] * self.norm_scale[k] + self.norm_bias[k]);
        }
        let mut critic = dense(&pooled, &self.critic_w, &self.critic_b);
        silu_norm(&mut critic);
        let value = dense(&critic, &self.value_w, &self.value_b)[0];
        Trunk { tiles, g, value }
    }

    pub fn family_logits(&self, t: &Trunk) -> Vec<f32> {
        dense(&t.g, &self.family_w, &self.family_b)
    }

    pub fn source_logits(&self, t: &Trunk, family: usize) -> Vec<f32> {
        let w = self.width;
        t.tiles.chunks_exact(w).map(|row| {
            let mut z = self.source_b[family];
            for k in 0..w {
                z += row[k] * self.source_w[k * FAMILIES + family];
            }
            z
        }).collect()
    }

    /// Logits over max(n, 128) targets: tiles for move/attack (families 0, 1), else parameter embeddings.
    pub fn target_logits(&self, t: &Trunk, family: usize, source: usize) -> Vec<f32> {
        let w = self.width;
        let n = t.tiles.len() / w;
        let mut input = Vec::with_capacity(2 * w);
        input.extend_from_slice(&t.tiles[source * w..(source + 1) * w]);
        input.extend_from_slice(&t.g);
        let mut q = dense(&input, &self.query_w, &self.query_b);
        for k in 0..w {
            q[k] += self.family_embedding[family * w + k];
        }
        layer_norm(&mut q);
        let rows = n.max(PARAMETER_TARGETS);
        let scale = (w as f32).sqrt();
        (0..rows).map(|r| {
            let key = if family < 2 {
                if r < n { &t.tiles[r * w..(r + 1) * w] } else { return 0.0 }
            } else if r < PARAMETER_TARGETS {
                let at = (family * self.embedding_rows + r) * w;
                &self.parameter_embedding[at..at + w]
            } else {
                return 0.0;
            };
            key.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() / scale
        }).collect()
    }
}

/// 3x3 "SAME" convolution, NHWC input [n, cin], HWIO kernel [3, 3, cin, width], output [n, width].
pub(crate) fn conv(size: usize, cin: usize, width: usize, x: &[f32], kernel: &[f32], out: &mut [f32]) {
    match width {
        64 => conv_w::<64>(size, cin, x, kernel, out),
        128 => conv_w::<128>(size, cin, x, kernel, out),
        _ => unreachable!("width checked at load"),
    }
}

/// conv at a compile-time width W (a stack accumulator per output tile). Zero inputs are skipped (the stem's
/// one-hot planes are mostly zeros); adding 0 * w changes nothing.
fn conv_w<const WIDTH: usize>(size: usize, cin: usize, x: &[f32], kernel: &[f32], out: &mut [f32]) {
    let n = size * size;
    out[..n * WIDTH].fill(0.);
    for y in 0..size {
        for xx in 0..size {
            let o = (y * size + xx) * WIDTH;
            let mut acc = [0f32; WIDTH];
            for ky in 0..3 {
                let yy = y as isize + ky as isize - 1;
                if yy < 0 || yy >= size as isize {
                    continue;
                }
                for kx in 0..3 {
                    let xs = xx as isize + kx as isize - 1;
                    if xs < 0 || xs >= size as isize {
                        continue;
                    }
                    let input = &x[(yy as usize * size + xs as usize) * cin..][..cin];
                    let w = &kernel[(ky * 3 + kx) * cin * WIDTH..][..cin * WIDTH];
                    for (c, &v) in input.iter().enumerate() {
                        if v == 0. {
                            continue;
                        }
                        let row: &[f32; WIDTH] = w[c * WIDTH..(c + 1) * WIDTH].try_into().unwrap();
                        for k in 0..WIDTH {
                            acc[k] += v * row[k];
                        }
                    }
                }
            }
            out[o..o + WIDTH].copy_from_slice(&acc);
        }
    }
}

pub(crate) fn dense(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let out = b.len();
    let mut y = b.to_vec();
    for (i, &v) in x.iter().enumerate() {
        if v == 0. {
            continue;
        }
        let row = &w[i * out..(i + 1) * out];
        for j in 0..out {
            y[j] += v * row[j];
        }
    }
    y
}

pub(crate) fn layer_norm(x: &mut [f32]) {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    x.iter_mut().for_each(|v| *v -= mean);
    let var = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1. / (var + 1e-5).sqrt();
    x.iter_mut().for_each(|v| *v *= r);
}

#[inline]
pub(crate) fn silu(z: f32) -> f32 {
    z / (1. + (-z).exp())
}

pub(crate) fn silu_norm(x: &mut [f32]) {
    layer_norm(x);
    x.iter_mut().for_each(|v| *v = silu(*v));
}

/// log_softmax over the legal entries (rl/network.py::log_distribution: illegal logits sit at -1e9, and an
/// all-illegal row makes entry 0 legal).
pub fn log_distribution(logits: &[f32], mask: &[bool]) -> Vec<f32> {
    let any = mask.iter().any(|&m| m);
    let masked: Vec<f32> = logits.iter().zip(mask).enumerate()
        .map(|(i, (&z, &m))| if m || (!any && i == 0) { z } else { -1e9 }).collect();
    let max = masked.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = masked.iter().map(|z| (z - max).exp()).sum();
    let log_total = total.ln() + max;
    masked.iter().map(|z| z - log_total).collect()
}

/// jnp.argmax: the first maximum.
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}

/// Draws from exp(logp) with a uniform u in [0, 1).
pub fn sample(logp: &[f32], u: f32) -> usize {
    let mut acc = 0f32;
    let mut last = 0;
    for (i, &l) in logp.iter().enumerate() {
        let p = l.exp();
        if p > 0. {
            last = i;
            acc += p;
            if u < acc {
                return i;
            }
        }
    }
    last
}
