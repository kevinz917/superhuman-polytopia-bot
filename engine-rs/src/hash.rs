//! xxHash32 short-input path (gpu/discovery.py) and the .NET System.Random stream (gpu/random.py).

pub fn hash_words(seed: u32, words: &[u32]) -> u32 {
    let mut h = seed.wrapping_add(374761393).wrapping_add(4 * words.len() as u32);
    for &w in words {
        h = h.wrapping_add(w.wrapping_mul(3266489917));
        h = h.rotate_left(17).wrapping_mul(668265263);
    }
    h = (h ^ (h >> 15)).wrapping_mul(2246822519);
    h = (h ^ (h >> 13)).wrapping_mul(3266489917);
    h ^ (h >> 16)
}

#[inline]
pub fn hash_word(seed: u32, word: u32) -> u32 {
    hash_words(seed, &[word])
}

const MAX: i32 = 2147483647;

pub struct DotNetRandom {
    values: [i32; 56],
    current: usize,
    lag: usize,
}

impl DotNetRandom {
    pub fn new(seed: i32) -> DotNetRandom {
        let absolute = if seed == i32::MIN { MAX } else { seed.abs() };
        let first = 161803398i32.wrapping_sub(absolute);
        let mut values = [0i32; 56];
        values[55] = first;
        let (mut previous, mut next_value) = (first, 1i32);
        for i in 1..55usize {
            let at = (21 * i) % 55;
            values[at] = next_value;
            let delta = previous.wrapping_sub(next_value);
            previous = next_value;
            next_value = if delta < 0 { delta.wrapping_add(MAX) } else { delta };
        }
        for _ in 0..4 {
            for i in 1..56usize {
                let delta = values[i].wrapping_sub(values[1 + (i + 30) % 55]);
                values[i] = if delta < 0 { delta.wrapping_add(MAX) } else { delta };
            }
        }
        DotNetRandom { values, current: 0, lag: 21 }
    }

    pub fn next_int(&mut self) -> i32 {
        let i = if self.current == 55 { 1 } else { self.current + 1 };
        let j = if self.lag == 55 { 1 } else { self.lag + 1 };
        let mut value = self.values[i].wrapping_sub(self.values[j]);
        if value == MAX {
            value -= 1;
        }
        if value < 0 {
            value = value.wrapping_add(MAX);
        }
        self.values[i] = value;
        self.current = i;
        self.lag = j;
        value
    }

    /// Exact Next(0, upper) with the reference's 32-bit limb arithmetic.
    pub fn bounded(&mut self, upper: i32) -> i32 {
        let a = self.next_int() as u32;
        let b = upper as u32;
        let lower = (a & 65535).wrapping_mul(b);
        let middle = (a >> 16).wrapping_mul(b).wrapping_add(lower >> 16);
        let high = middle >> 16;
        let low = (middle << 16) | (lower & 65535);
        let mut quotient = (high << 1) | (low >> 31);
        let remainder = (low & 2147483647).wrapping_add(quotient);
        quotient = quotient.wrapping_add((remainder >= 2147483647) as u32);
        quotient as i32
    }

    /// Float32 cast of NextDouble, including the upward-shifted halfway cases.
    pub fn unit_float(&mut self) -> f32 {
        let value = self.next_int();
        let bits = value as u32;
        let shift = (8 - bits.leading_zeros() as i32).max(0) as u32;
        let quantum = 1u32 << shift;
        let tie = shift > 0 && (bits & (quantum - 1)) == (quantum >> 1) && ((bits / quantum) & 1) == 0;
        let result = (value as f32) * (1.0f32 / 2147483648.0f32);
        if tie {
            f32::from_bits(result.to_bits() + 1)
        } else {
            result
        }
    }
}
