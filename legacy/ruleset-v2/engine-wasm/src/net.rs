//! rl/network.py (architecture normalized-v2, 2 blocks x 64 channels) on the CPU: the conv trunk, the value head
//! and the masked autoregressive family -> source -> target heads. fp32 throughout; sums run in a different order
//! than XLA's, so logits match the JAX policy to rounding, not bitwise.

pub const WIDTH: usize = 64;
pub const BOARD: usize = 138;
pub const GLOBAL: usize = 397;
pub const FAMILIES: usize = 17;
const POOLED: usize = 2 * WIDTH + GLOBAL;
const PARAMETER_TARGETS: usize = 128;

pub struct Net {
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
    pub fn new(n: usize) -> Buffers {
        Buffers { a: vec![0.; n * WIDTH], b: vec![0.; n * WIDTH], c: vec![0.; n * WIDTH] }
    }
}

impl Net {
    /// Parses scripts/export_browser_weights.py output: b"PTB1", count, shapes, then float32 data.
    pub fn parse(bytes: &[u8]) -> Result<Net, String> {
        let u32_at = |at: usize| -> Result<usize, String> {
            bytes.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize).ok_or("truncated header".into())
        };
        if bytes.get(..4) != Some(b"PTB1") {
            return Err("not a browser weight file".into());
        }
        let count = u32_at(4)?;
        let mut at = 8;
        let mut shapes = Vec::with_capacity(count);
        for _ in 0..count {
            let rank = u32_at(at)?;
            at += 4;
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                dims.push(u32_at(at)?);
                at += 4;
            }
            shapes.push(dims);
        }
        let mut tensors = Vec::with_capacity(count);
        for dims in &shapes {
            let len: usize = dims.iter().product();
            let raw = bytes.get(at..at + 4 * len).ok_or("truncated tensor data")?;
            tensors.push(raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<f32>>());
            at += 4 * len;
        }
        if at != bytes.len() {
            return Err("trailing bytes after tensors".into());
        }
        let w = WIDTH;
        let rows = shapes.get(13).and_then(|s| s.get(1)).copied().unwrap_or(0);
        let expect: [&[usize]; 21] = [
            &[3, 3, w, w], &[3, 3, w, w], &[3, 3, w, w], &[3, 3, w, w], &[w], &[POOLED, w], &[FAMILIES], &[w, FAMILIES],
            &[FAMILIES, w], &[w], &[POOLED, w], &[w], &[w], &[FAMILIES, rows, w], &[w], &[2 * w, w], &[FAMILIES],
            &[w, FAMILIES], &[3, 3, BOARD, w], &[1], &[w, 1],
        ];
        if count != expect.len() || shapes.iter().zip(expect).any(|(s, e)| s.as_slice() != e) || rows < PARAMETER_TARGETS {
            return Err(format!("unexpected tensor shapes {shapes:?}"));
        }
        let mut t = tensors.into_iter();
        let mut next = || t.next().unwrap();
        Ok(Net {
            blocks: vec![(next(), next()), (next(), next())],
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

    pub fn trunk(&self, size: usize, board: &[f32], global: &[f32], buf: &mut Buffers) -> Trunk {
        let n = size * size;
        let Buffers { a, b, c } = buf;
        conv(size, BOARD, board, &self.stem, a);
        a.chunks_exact_mut(WIDTH).for_each(silu_norm);
        for (k1, k2) in &self.blocks {
            conv(size, WIDTH, a, k1, b);
            b.chunks_exact_mut(WIDTH).for_each(silu_norm);
            conv(size, WIDTH, b, k2, c);
            for (x, y) in a.iter_mut().zip(c.iter()) {
                *x += *y;
            }
            a.chunks_exact_mut(WIDTH).for_each(silu_norm);
        }
        let tiles = a[..n * WIDTH].to_vec();
        let mut pooled = vec![0f32; POOLED];
        {
            let (mean, rest) = pooled.split_at_mut(WIDTH);
            let (max, glob) = rest.split_at_mut(WIDTH);
            max.fill(f32::NEG_INFINITY);
            for row in tiles.chunks_exact(WIDTH) {
                for k in 0..WIDTH {
                    mean[k] += row[k];
                    max[k] = max[k].max(row[k]);
                }
            }
            mean.iter_mut().for_each(|m| *m /= n as f32);
            glob.copy_from_slice(global);
        }
        let mut g = dense(&pooled, &self.global_w, &self.global_b);
        layer_norm(&mut g);
        for k in 0..WIDTH {
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
        t.tiles.chunks_exact(WIDTH).map(|row| {
            let mut z = self.source_b[family];
            for k in 0..WIDTH {
                z += row[k] * self.source_w[k * FAMILIES + family];
            }
            z
        }).collect()
    }

    /// Logits over max(n, 128) targets: tiles for move/attack (families 0, 1), else parameter embeddings.
    pub fn target_logits(&self, t: &Trunk, family: usize, source: usize) -> Vec<f32> {
        let n = t.tiles.len() / WIDTH;
        let mut input = Vec::with_capacity(2 * WIDTH);
        input.extend_from_slice(&t.tiles[source * WIDTH..(source + 1) * WIDTH]);
        input.extend_from_slice(&t.g);
        let mut q = dense(&input, &self.query_w, &self.query_b);
        for k in 0..WIDTH {
            q[k] += self.family_embedding[family * WIDTH + k];
        }
        layer_norm(&mut q);
        let rows = n.max(PARAMETER_TARGETS);
        let scale = (WIDTH as f32).sqrt();
        (0..rows).map(|r| {
            let key = if family < 2 {
                if r < n { &t.tiles[r * WIDTH..(r + 1) * WIDTH] } else { return 0.0 }
            } else if r < PARAMETER_TARGETS {
                let at = (family * self.embedding_rows + r) * WIDTH;
                &self.parameter_embedding[at..at + WIDTH]
            } else {
                return 0.0;
            };
            key.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>() / scale
        }).collect()
    }
}

/// 3x3 "SAME" convolution, NHWC input [n, cin], HWIO kernel [3, 3, cin, WIDTH], output [n, WIDTH].
/// Zero inputs are skipped (the stem's one-hot planes are mostly zeros); adding 0 * w changes nothing.
fn conv(size: usize, cin: usize, x: &[f32], kernel: &[f32], out: &mut [f32]) {
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

fn dense(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
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

fn layer_norm(x: &mut [f32]) {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    x.iter_mut().for_each(|v| *v -= mean);
    let var = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1. / (var + 1e-5).sqrt();
    x.iter_mut().for_each(|v| *v *= r);
}

#[inline]
fn silu(z: f32) -> f32 {
    z / (1. + (-z).exp())
}

fn silu_norm(x: &mut [f32]) {
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
