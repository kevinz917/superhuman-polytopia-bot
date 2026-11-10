/* hash.h - port of engine-rs/src/hash.rs: the xxHash32 short-input path (gpu/discovery.py).
 *
 * DotNetRandom (the .NET System.Random stream) is mapgen-only and is not ported here.
 * All arithmetic is wrapping u32, exactly as the Rust (`wrapping_add/mul`, `rotate_left`). */
#ifndef PC_HASH_H
#define PC_HASH_H

#include "pc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* u32::rotate_left */
PC_FN uint32_t pc_rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

/* XXHash.GetHash(int[]) for fewer than 4 words (XXH32 short input). With `seed` = GameState.Seed (our
 * `initial_seed`) this is GameState.RandomHash: HashFunction.GetHash(x[, y[, z]]) and XXHash.GetHash(int) are this.
 * `count` < 4 (debug_assert in the Rust; dropped here). */
PC_FN uint32_t pc_hash_words(uint32_t seed, const uint32_t *words, int count) {
    uint32_t h = seed + 374761393u + 4u * (uint32_t)count;
    for (int i = 0; i < count; i++) {
        uint32_t w = words[i];
        h = h + w * 3266489917u;
        h = pc_rotl32(h, 17) * 668265263u;
    }
    h = (h ^ (h >> 15)) * 2246822519u;
    h = (h ^ (h >> 13)) * 3266489917u;
    return h ^ (h >> 16);
}

PC_FN uint32_t pc_hash_word(uint32_t seed, uint32_t word) { return pc_hash_words(seed, &word, 1); }

/* HashFunction.Value(x, y, z) on an XXHash: the hash of the int triple times 2^-32, in float.
 * Rust: `hash_words(seed, &[x as u32, y as u32, z as u32]) as f32 * 2.3283064e-10f32` (u32 -> f32 round to nearest,
 * then one f32 multiply; keep the constant and the f32 math). */
PC_FN float pc_hash_value3(uint32_t seed, int32_t x, int32_t y, int32_t z) {
    uint32_t w[3];
    w[0] = (uint32_t)x; w[1] = (uint32_t)y; w[2] = (uint32_t)z;
    float h = (float)pc_hash_words(seed, w, 3);
    return h * 2.3283064e-10f;
}

/* HashFunction.Range(min, max, x, y, z): the triple's hash modulo (max - min), unsigned, plus min.
 * Rust: `(hash_words(...) % (max - min) as u32) as i32 + min`. */
PC_FN int32_t pc_hash_range3(uint32_t seed, int32_t min, int32_t max, int32_t x, int32_t y, int32_t z) {
    uint32_t w[3];
    w[0] = (uint32_t)x; w[1] = (uint32_t)y; w[2] = (uint32_t)z;
    uint32_t span = (uint32_t)pc_wsub(max, min);
    uint32_t m = pc_hash_words(seed, w, 3) % span;
    return pc_wadd((int32_t)m, min);
}

/* HashFunction.Range(int min, int max, ...): hash % (max - min) + min, for max > min (u32 throughout). */
PC_FN uint32_t pc_hash_range(uint32_t seed, uint32_t min, uint32_t max, const uint32_t *words, int count) {
    return pc_hash_words(seed, words, count) % (max - min) + min;
}

#ifdef __cplusplus
}
#endif
#endif /* PC_HASH_H */
