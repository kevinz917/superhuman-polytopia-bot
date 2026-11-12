//! Plato's transformer-v3-p2-h4 policy. Tensor order is JAX's sorted pytree order;
//! validates the complete layout before evaluating. No privileged critic is loaded.
use crate::net::{conv, dense, layer_norm, silu, silu_norm, Buffers, Trunk};

const W: usize = 128;
/// The relative-position tables' own grid (13 x 13 offsets: the 14 x 14 base board, POLYTOPIA_V3_BASE_SIZE=14).
const TABLE_GRID: usize = 7;

/// (board channels, global entries) of a Plato tensor layout: observation version 3 (151, 525), 4 (153, 525) or 5
/// (153, 555) (rl/observation.py at c7e34799); 6 (174, 556) at b81241e1 (T3 u96070).
pub fn widths(shapes: &[Vec<usize>]) -> Option<(usize, usize)> {
    let g = shapes.get(4)?.first().copied()?;
    let b = shapes.get(103)?.get(2).copied()?;
    matches!((b, g), (151, 525) | (153, 525) | (153, 555) | (174, 556)).then_some((b, g))
}

pub fn validate(shapes: &[Vec<usize>]) -> Result<(), String> {
    let (board, global) = widths(shapes).ok_or("unsupported Plato observation widths")?;
    let mut expected = vec![vec![17], vec![128,17], vec![17,128], vec![128], vec![global,128],
        vec![128], vec![128], vec![17,196,128], vec![128], vec![256,128], vec![17], vec![128,17]];
    for _ in 0..6 {
        expected.extend([vec![128],vec![128],vec![512],vec![128,512],vec![128],vec![128],
            vec![128],vec![512,128],vec![128],vec![128,128],vec![384],vec![128,384],vec![4,13,13]]);
    }
    expected.extend((0..8).map(|_|vec![3,3,128,128]));
    expected.extend([vec![128],vec![128],vec![128],vec![512,128],vec![128],vec![3,3,board,128],
        vec![512],vec![128,512]]);
    if shapes != expected { return Err("unsupported Plato tensor layout".into()); }
    Ok(())
}

pub struct V3 { p: Vec<Vec<f32>>, pub board: usize, pub global: usize }

fn rows_dense(x: &[f32], input: usize, weights: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len() / input * bias.len());
    for row in x.chunks_exact(input) { out.extend(dense(row, weights, bias)); }
    out
}

fn norm(x: &mut [f32], bias: &[f32], scale: &[f32]) {
    for row in x.chunks_exact_mut(W) {
        layer_norm(row);
        for i in 0..W { row[i] = row[i] * scale[i] + bias[i]; }
    }
}

impl V3 {
    pub fn new(p: Vec<Vec<f32>>, board: usize, global: usize) -> Self { Self { p, board, global } }

    /// transformer_v3.encode: a side that is not a multiple of the patch (2) is zero-padded up to one (tiles cropped
    /// back); grid = padded / 2 patch tokens plus the global token; attention offsets beyond the tables' +-6 use
    /// their edge entries (_table_bias). At 14 x 14 both are the identity.
    pub fn trunk(&self, size: usize, board: &[f32], global: &[f32], _buf: &mut Buffers) -> Trunk {
        let p = &self.p;
        let (real, cb) = (size, self.board);
        let side = real + real % 2;
        let grid = side / 2;
        let cells = side * side;
        let patches = grid * grid;
        let mut padded = vec![0.; cells * cb];
        for y in 0..real {
            let row = &board[y * real * cb..(y + 1) * real * cb];
            padded[y * side * cb..y * side * cb + real * cb].copy_from_slice(row);
        }
        let mut tiles = vec![0.; cells * W];
        let mut b = vec![0.; cells * W];
        let mut c = vec![0.; cells * W];
        conv(side, cb, W, &padded, &p[103], &mut tiles);
        for row in tiles.chunks_exact_mut(W) {
            for i in 0..W { row[i] += p[102][i]; }
            silu_norm(row);
        }
        for k in 0..4 {
            conv(side,W,W,&tiles,&p[90+2*k],&mut b);
            b.chunks_exact_mut(W).for_each(silu_norm);
            conv(side,W,W,&b,&p[91+2*k],&mut c);
            for (x,y) in tiles.iter_mut().zip(&c) { *x += y; }
            tiles.chunks_exact_mut(W).for_each(silu_norm);
        }
        let mut patch = vec![0.; patches * 4 * W];
        for y in 0..grid { for x in 0..grid { for dy in 0..2 { for dx in 0..2 {
            let src = ((2*y+dy)*side+2*x+dx)*W;
            let dst = ((y*grid+x)*4+dy*2+dx)*W;
            patch[dst..dst+W].copy_from_slice(&tiles[src..src+W]);
        }}}}
        let mut tokens = rows_dense(&patch,4*W,&p[101],&p[100]);
        tokens.extend(dense(global,&p[4],&p[3]));
        for k in 0..6 { self.block(&mut tokens,12+13*k,grid); }
        norm(&mut tokens,&p[98],&p[99]);
        let decoded = rows_dense(&tokens[..patches*W],W,&p[105],&p[104]);
        for y in 0..grid { for x in 0..grid { for dy in 0..2 { for dx in 0..2 {
            let dst = ((2*y+dy)*side+2*x+dx)*W;
            let src = ((y*grid+x)*4+dy*2+dx)*W;
            for k in 0..W { tiles[dst+k] = silu(tiles[dst+k]+decoded[src+k]); }
        }}}}
        let out = if side == real { tiles } else {
            let mut out = Vec::with_capacity(real * real * W);
            for y in 0..real { out.extend_from_slice(&tiles[y * side * W..(y * side + real) * W]); }
            out
        };
        Trunk { tiles: out, g: tokens[patches*W..].to_vec(), value: 0. }
    }

    fn block(&self, x: &mut [f32], at: usize, grid: usize) {
        let p = &self.p;
        let mut h = x.to_vec();
        norm(&mut h,&p[at],&p[at+1]);
        let qkv = rows_dense(&h,W,&p[at+11],&p[at+10]);
        let patches = grid * grid;
        let n = patches + 1;
        let reach = (TABLE_GRID - 1) as isize;
        let mut mixed = vec![0.;n*W];
        let mut scores = vec![0f32;n];
        let scale = 1. / (32f32).sqrt();
        for head in 0..4 { for q in 0..n {
            for key in 0..n {
                let mut s = 0.;
                for d in 0..32 {
                    s += qkv[q*3*W+head*32+d] * qkv[key*3*W+W+head*32+d];
                }
                let relative = if q < patches && key < patches {
                    let dy = ((q/grid) as isize - (key/grid) as isize).clamp(-reach, reach) + reach;
                    let dx = ((q%grid) as isize - (key%grid) as isize).clamp(-reach, reach) + reach;
                    p[at+12][head*169+dy as usize*13+dx as usize]
                } else { 0. };
                scores[key] = s*scale+relative;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);
            let mut sum = 0.;
            for s in &mut scores { *s = (*s-max).exp(); sum += *s; }
            for key in 0..n { for d in 0..32 {
                mixed[q*W+head*32+d] += scores[key]/sum * qkv[key*3*W+2*W+head*32+d];
            }}
        }}
        let projected = rows_dense(&mixed,W,&p[at+9],&p[at+8]);
        for (a,b) in x.iter_mut().zip(projected) { *a += b; }
        h.copy_from_slice(x);
        norm(&mut h,&p[at+4],&p[at+5]);
        let mut ff = rows_dense(&h,W,&p[at+3],&p[at+2]);
        ff.iter_mut().for_each(|x|*x=silu(*x));
        let projected = rows_dense(&ff,4*W,&p[at+7],&p[at+6]);
        for (a,b) in x.iter_mut().zip(projected) { *a += b; }
    }
}
