//! Board geometry shared by every module: Chebyshev distances and the 8-neighbour table
//! (same order as gpu/kernels.py: dx=[-1,0,1,-1,1,-1,0,1], dy=[-1,-1,-1,0,0,1,1,1]).

pub const DX: [i32; 8] = [-1, 0, 1, -1, 1, -1, 0, 1];
pub const DY: [i32; 8] = [-1, -1, -1, 0, 0, 1, 1, 1];

#[derive(Clone)]
pub struct Geom {
    pub size: usize,
    pub n: usize,
    /// dist[i*n+j] Chebyshev distance.
    pub dist: Vec<u8>,
    /// nb[i*8+k] neighbour index (clipped into the board like the JAX table) and valid flag.
    pub nb: Vec<u16>,
    pub nb_valid: Vec<bool>,
    /// Per tile: list of in-board neighbour indices (compact form).
    pub adj: Vec<Vec<u16>>,
    pub corners: [usize; 4],
}

impl Geom {
    pub fn new(size: usize) -> Geom {
        let n = size * size;
        let mut dist = vec![0u8; n * n];
        for i in 0..n {
            let (xi, yi) = ((i % size) as i32, (i / size) as i32);
            for j in 0..n {
                let (xj, yj) = ((j % size) as i32, (j / size) as i32);
                dist[i * n + j] = ((xi - xj).abs().max((yi - yj).abs())) as u8;
            }
        }
        let mut nb = vec![0u16; n * 8];
        let mut nb_valid = vec![false; n * 8];
        let mut adj = vec![Vec::with_capacity(8); n];
        let s = size as i32;
        for i in 0..n {
            let (x, y) = ((i % size) as i32, (i / size) as i32);
            for k in 0..8 {
                let (nx, ny) = (x + DX[k], y + DY[k]);
                let valid = nx >= 0 && nx < s && ny >= 0 && ny < s;
                let idx = (ny * s + nx).clamp(0, (n - 1) as i32) as u16;
                nb[i * 8 + k] = idx;
                nb_valid[i * 8 + k] = valid;
                if valid {
                    adj[i].push(idx);
                }
            }
        }
        Geom { size, n, dist, nb, nb_valid, adj, corners: [0, size - 1, n - size, n - 1] }
    }

    #[inline]
    pub fn d(&self, i: usize, j: usize) -> i32 {
        self.dist[i * self.n + j] as i32
    }

    /// Calls f(t) for every tile t with d(center, t) <= radius (row-major order), visiting only that box.
    #[inline]
    pub fn for_ball(&self, center: usize, radius: i32, mut f: impl FnMut(usize)) {
        if radius < 0 {
            return;
        }
        let size = self.size as i32;
        let (cx, cy) = ((center % self.size) as i32, (center / self.size) as i32);
        for y in (cy - radius).max(0)..=(cy + radius).min(size - 1) {
            for x in (cx - radius).max(0)..=(cx + radius).min(size - 1) {
                f((y * size + x) as usize);
            }
        }
    }
}
