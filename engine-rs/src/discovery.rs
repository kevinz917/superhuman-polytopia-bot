//! gpu/discovery.py::explorer, the reference's deterministic city explorer.

use crate::geom::Geom;
use crate::hash::hash_word;
use crate::state::State;

pub fn explorer(s: &State, source: usize, g: &Geom) -> Vec<bool> {
    let n = s.n;
    let p = s.p();
    let initial = &s.tiles.explored[p];
    let tech = &s.players.tech[p];
    let mut visible: Vec<bool> = initial.clone();
    let mut current = source;
    let mut previous = -1i32;
    let mut score = vec![0i32; n];
    let mut spread = vec![0i32; n];
    for _ in 0..12 {
        for t in 0..n {
            if visible[t] {
                score[t] = 1000;
            } else {
                let mut count = 0;
                let mut light = false;
                for j in 0..n {
                    if g.d(t, j) <= 2 && !visible[j] {
                        count += 1;
                        if s.tiles.structure[j] == 47 {
                            light = true;
                        }
                    }
                }
                let count = count.clamp(1, 4);
                score[t] = 183 - count * 10 - if light { 33 } else { 0 };
            }
        }
        for _ in 0..3 {
            for t in 0..n {
                let mut best = 1000;
                for &j in &g.adj[t] {
                    best = best.min(score[j as usize] + 100);
                }
                spread[t] = score[t].min(best);
            }
            std::mem::swap(&mut score, &mut spread);
        }
        let mut allowed = [false; 8];
        let mut values = [10000i32; 8];
        let mut best = 10000i32;
        for k in 0..8 {
            if !g.nb_valid[current * 8 + k] {
                continue;
            }
            let idx = g.nb[current * 8 + k] as usize;
            let terrain = s.tiles.terrain[idx];
            let passable = terrain != 0 && (terrain != 4 || tech[20]) && (terrain != 1 || tech[13]) && (terrain != 2 || tech[14]);
            allowed[k] = !visible[idx] || passable;
            if allowed[k] {
                values[k] = score[idx] + (idx as i32 == previous) as i32;
                best = best.min(values[k]);
            }
        }
        let candidates: Vec<usize> = (0..8).filter(|&k| allowed[k] && values[k] == best).collect();
        let count = candidates.len() as u32;
        if count == 0 {
            break;
        }
        let draw = hash_word(s.seed, current as u32) % count;
        let target = g.nb[current * 8 + candidates[draw as usize]] as usize;
        if target == current {
            break;
        }
        for j in 0..n {
            if g.d(target, j) <= 1 {
                visible[j] = true;
            }
        }
        previous = current as i32;
        current = target;
    }
    (0..n).map(|t| visible[t] && !initial[t]).collect()
}
