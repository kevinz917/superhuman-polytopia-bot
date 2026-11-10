# GPU-resident Polytopia engine (engine-cu): design and porting contract (2026-11-06)

**Goal.** Run the whole training loop on one GPU: tens of thousands of games stepped by CUDA kernels, legal masks
and observations written straight into the policy's device tensors, no CPU actors, no sockets, no batch assembly.
Model: Netherite (Minecraft 1.11.2 in C/CUDA, `~/Documents/Codex/2026-07-29/.../netherite`): one source compiled
twice (CPU oracle + CUDA), one thread per environment for the serial game logic, bitwise CPU == CUDA as the gate.

**Why it pays.** The engine is ~5% of actor time today; the win is deleting the plumbing between actors and
learner (egress 30% of the bill, batch assembly and H2D upload were the walls of every hero run). Projected
~$3–6 per billion samples on one H100/PRO 6000 against ~$50–60 on Modal today.

## Layout

```
engine-cu/
  Makefile                 CPU: cc -O2 -ffp-contract=off -fwrapv ; CUDA: nvcc -O2 --fmad=false -arch=sm_XX
  src/pc.h                 shared types (PcState, PcRules, PcGeom, bitsets, Rust cast helpers)   [written]
  src/hash.h               xxHash32 short-input + hash_range/hash_value (hash.rs; DotNetRandom is mapgen-only)
  src/rules.h              PcRulesInput (primary tables) + pc_rules_finalize (rules.rs from_json derivations)
  src/kernels.h            combat, max_health, research, movement flags / reach / targets, update_exploration, observe
  src/history.h            history.rs
  src/military.h           military.rs      src/capture.h  capture.rs     src/infiltration.h  infiltration.rs
  src/structures.h         structures.rs    src/economy.h  economy.rs
  src/diplomacy.h          diplomacy.rs     src/discovery.h discovery.rs  src/pathfinding.h   pathfinding.rs
  src/masks.h              masks.rs (packed sink only)
  src/core.h               core.rs (decode, step_with, terminal_rewards, Scratch) + mapgen::initial_state
  src/engine.h             includes everything in dependency order (the one translation unit)
  src/pc_api.h             the C ABI below
  src/pc_cpu.c             CPU driver: loop over games
  src/pc_cuda.cu           CUDA driver: k_reset, k_tick (1 thread/game), same ABI, host or device output pointers
src/polytopia_bot/rl/gpu_engine.py   ctypes wrapper (RustGames-shaped), rules packing, layouts from engine-rs
tests/test_gpu_engine.py             lockstep harness vs RustGames (CLI like the old test_rust_engine.py)
experiments/modal_gpu_engine.py      Modal: build (nvidia/cuda devel image), CPU lockstep gate, GPU gate, bench
```

## Porting rules (every module)

1. **Line-by-line port of the Rust module of the same name.** Same function names (`module::f` → `pc_<module>_<f>`),
   same argument order, same semantics, same comments where they explain a rule. Read the Rust; do not redesign.
2. **No allocation, no recursion, no function pointers.** Rust `Vec`s bounded by `n` become fixed arrays of
   `PC_MAXN` (324) in a scratch struct the caller passes (or on the stack when small). `Vec::push` → array +
   count. `impl FnMut(usize)` callbacks (e.g. `reach`'s `emit`) → write into an out array / bitset the caller scans.
3. **Types.** `&State` → `const PcState*`, `&mut State` → `PcState*`, `&Rules` → `const PcRules*`, `&Geom` →
   `const PcGeom*`, `bool` → `uint8_t` (0/1), `usize` → `int`, `i32` → `int32_t`, `f32` → `float`, `u128` →
   `pc_u128` (helpers in pc.h), `Option<T>` → documented sentinel (`-1`) or an `int* found` out-param, tuples →
   out-params in order. `Vec<bool>` return → `uint8_t out[PC_MAXN]` parameter.
4. **Numerics.** f32 only, never double (no `pow`, `sqrt` of doubles; use `floorf`, `sqrtf`...). Rust `as i32` on a
   float → `pc_f2i`, `as i16` → `pc_f2i16`. `wrapping_add/sub/mul` → `pc_wadd/wsub/wmul`. `div_euclid`/`rem_euclid`
   → `pc_div_euclid/rem_euclid`. `/` and `%` on i32 are C's. `clamp/min/max` → `pc_clamp/min/max`. Keep the
   operation order of every float expression exactly (the combat formula is `af/divisor*aa*4.5 + 1e-6 + 0.5` then
   floor). Never reorder, factor or simplify arithmetic.
5. **Sorting / ordering.** Rust `sort`/`sort_by` is stable → insertion sort (bounded n). Heap orderings
   (pathfinding, structures routes) are ported as the exact comparator the Rust uses.
6. **Per-tile territory** is a bitset: `pc_territory(s, city, tile)` / `pc_set_territory`.
7. **Bitsets of 128** (`u128` tech/unit/structure sets) use the `pc_u128_*` helpers; iterate set bits with
   `pc_u128_lowest` + clear.
8. **Device-safe.** Every function `PC_FN` (static inline, `__host__ __device__` under nvcc). No `printf`, no
   `assert` (use `(void)` or return codes), no libc beyond `memset`/`memcpy`/`math.h` f32 functions. `static const`
   tables are fine. Skip `prof::time` and `cov::` (no-ops).
9. **Each module = one header** `src/<module>.h` guarded `PC_<MODULE>_H`, including `pc.h` and the module headers it
   calls. Put the module's structs (e.g. `PcMoveScratch`, `PcBuildCtx`, `PcSummonCtx`, `PcPathScratch`) in its
   header. If two modules call each other, forward-declare the callee prototype at the top of the caller's header
   (`PC_FN` prototypes are fine before the definition in the same translation unit).
10. **Compile check only, locally:** `clang -fsyntax-only -x c -std=c11 -I engine-cu/src <file>` (and `-x c++` to
    mimic nvcc). Never run engines or tests on the Mac; all runs go to Modal.
11. **Rust constructs that need care.** `.at[-1]` / negative gathers → `pc_pwrap`. `iter().any/all/filter.count`
    → loops. `matches!` → switch. Closures capturing state → inline loops. Iterator chains over `0..n` → for loops
    in the same order (order matters for every "first match" and tie-break). `debug_assert!` → drop.

## The C ABI (pc_api.h; both drivers export it)

```c
typedef struct PcBatch PcBatch;
PcBatch *pc_create(int size, int games, int players, int max_turns /* <=0 none */, int no_resign, int no_peace,
                   int max_actions, int device /* -1 CPU, else CUDA device */);
void pc_destroy(PcBatch *);
int  pc_set_rules(PcBatch *, const PcRulesInput *);        /* 0 ok; finalizes derived tables */
int  pc_mask_bytes(const PcBatch *);                       /* packed bits per game = ceil(mask_width/8) */
int  pc_view_width(const PcBatch *);                       /* int16 entries per view */
int  pc_sizes(const PcBatch *, int *n, int *np, int *games);
/* fresh games from layouts (engine-rs GameBatch.layout(seed)): terrain/structure/resource [G*n], caps [G*np] */
int  pc_reset(PcBatch *, const int32_t *seeds, const int32_t *terrain, const int32_t *structure,
              const int32_t *resource, const int32_t *caps);
typedef struct {
    uint8_t *masks;        /* [G, mask_bytes] packed, numpy.packbits order */
    int16_t *view;         /* [G, vw] mover's view */
    int16_t *opp_view;     /* [G, np-1, vw] other seats in turn order after the mover (both_views) */
    int16_t *final_view;   /* [G, np, vw] every seat's view of a truncated final state (zero otherwise) */
    uint8_t *done, *terminated, *truncated;   /* [G] */
    float *rewards;        /* [G, np] terminal rewards */
    int32_t *status;       /* [G] */
    int device_ptrs;       /* CUDA driver: 1 = all pointers are device memory (no copies) */
} PcTickOut;
/* step every game with (family, source, target), auto-reset finished games from the given layouts, then masks and
   views of the new state. Same contract as engine-rs GameBatch.tick(packed_masks=True). */
int  pc_tick(PcBatch *, const int32_t *family, const int32_t *source, const int32_t *target, const int32_t *seeds,
             const int32_t *terrain, const int32_t *structure, const int32_t *resource, const int32_t *caps,
             int both_views, PcTickOut *out);
/* full State of one game <-> host buffers, field order = engine-rs export() keys (gpu_engine.py EXPORT_FIELDS) */
int  pc_export(PcBatch *, int game, void **field_ptrs);
int  pc_import(PcBatch *, int game, void **field_ptrs);
int  pc_legal_masks(PcBatch *, uint8_t *packed /* [G, mask_bytes] */);   /* masks of the current states */
int  pc_observe(PcBatch *, int player /* 0 = mover */, int16_t *view /* [G, vw] */);
```

Export field order and dtypes follow `engine-rs/src/lib.rs::export` exactly (tiles.* , units.*, cities.*,
players.*, history.*, then the scalars); `cities.territory` is exported dense `[n, n]` u8 and `history.enemy_tech`
as `[128]` u8 like the Rust.

## Rules input

`PcRulesInput` (rules.h) carries the primary tables of `gpu/rules.json` (units: health, attack, defense, movement,
range, cost, 10 skill flags, exists; technologies: tier, requires or -1, next as u128, structures as u128, exists;
structures: cost/population/score/stars or -1, resource or -1, terrain bits [16], adjacent u128, per_city,
per_tribe, tribe or -1, exists; engine_revision string). `gpu_engine.py::pack_rules(rules_json)` fills it;
`pc_rules_finalize` ports the derivation code of `Rules::from_json` (rules.rs). The lockstep gate is the check.

## Lockstep harness (tests/test_gpu_engine.py)

Both engines reset from the same `GameBatch.layout(seed)` layouts; each tick samples a legal action per game from
the Rust packed masks (uniform, end-turn boosted, a forced resignation every k ticks when allowed), feeds the same
(family, source, target, seeds, layouts) to both, then asserts bitwise equality of status, done/terminated/
truncated, rewards, packed masks, mover and other-seat views, final views; every K ticks exports every State
field of every game from both and compares. First difference: field, game, index, both values. CLI flags like
the old `test_rust_engine.py`: `--size --games --ticks --seed --players --end-boost --resign-every --export-every`.
Prints `RESULT ok ...` or `RESULT FAIL ...`.

`--policy /results/distributed/<run>/latest.npz [--policy-temp 1.0 --epsilon 0.05]` (2026-11-06): policy play. The actions
come from `rl.network.sample` on CPU JAX over the Rust tick's packed view (`cpu_engine.view_features`) and packed masks
(`rust_engine.split_masks`), vmapped over the games, one PRNG key per tick; an epsilon share of the games plays a uniform-random
legal action (end-turn boosted) and the forced resignation applies on top, so the gate sees the positions a trained bot
reaches (long games, late techs, many units) and odd actions. The checkpoint loads through the production path
(`distributed.network_spec` / `checkpoint_template`, `runner.load_checkpoint`); `network.trunk` fits the newest features to the
checkpoint's observation version, so version 1-3 checkpoints play. `modal_gpu_engine.py::equality --policy <path>` mounts the
run volumes read-only (`/results`, `/retained`, modal_distributed_rl.py's paths) and prints the checkpoint's config first.
RESULT rows carry `policy=<run>/<stem>` (`random` otherwise), `policy_actions`, `max_turn`, `policy_ms_per_tick`.

## CUDA driver (pc_cuda.cu)

- `PcState states[G]` and `PcScratch scratch[G]` device arrays, `PcRules`/`PcGeom` in global memory (one copy).
- `k_tick`: one thread per game, 32 threads per block (latency-bound serial code; spread across SMs as Netherite
  found), runs decode → step_with (validated by the previous tick's packed masks, kept on device) → rewards →
  final views → reset from layout → masks → observe. `k_reset` likewise.
- Output pointers host or device (`device_ptrs`); host mode copies through staging buffers (harness).
- Later: split masks/observe into per-tile kernels only where the profile says so; the GPU budget is generous
  (the network forward is the cap at ~200k decisions/s).

## Gates and budget

1. C-CPU vs Rust: Modal CPU container (no GPU), millions of lockstep steps across 11/14/18 boards, 2 seats then 4.
2. CUDA vs C-CPU: bitwise on the same harness, small GPU (T4/L4), minutes. Compile in the CPU container (nvcc
   needs the toolkit, not a GPU); the GPU container only runs.
3. Throughput: samples/s of tick alone at N = 4k/16k/64k; then the PPO loop (torch buffers → policy → actions).
Testing budget: $1–2 total on Modal GPU; CPU gates are cents.

## Results (2026-11-06, first runs, zero divergences)

Lockstep vs engine-rs (Modal CPU, `::equality`, `tests/test_gpu_engine.py`, full-state export compared every 100 ticks):

| board | seats | runs | steps | games finished | note |
|---|---|---|---|---|---|
| 14x14 | 2 | 8 x 2000 ticks x 24 games | 384,000 | 68 (7 eliminations, 61 round cap) | |
| 18x18 | 2 | 4 x 1200 x 24 | 115,200 | 0 | |
| 14x14 | 4 | 4 x 1500 x 16 | 96,000 | 0 | four-seat views, out_order |
| 11x11 | 2 | 4 x 1500 x 24, `--no-resign 0 --resign-every 53` | 144,000 | 20,509 (all by elimination/resign; 108 resignations) | |

CUDA vs engine-rs AND CUDA vs the C CPU driver, bitwise (Modal T4, `::gpu_equality`): 11x11 2p 1,200 steps; 14x14 2p
14,400 steps; 18x18 4p 4,800 steps. The CUDA driver compiled clean with nvcc on the first try.

Throughput: one thread ticks a 14x14 game in ~17-21 ms (T4 / L4) and an 18x18 four-seat game in ~32 ms (T4); with
4,096 resident games an L4 ticks the batch in 17.5 ms (234k ticks/s), with 8,192 in 26 ms (315k ticks/s, residency
starting to bind). GPU-resident PPO smoke (`experiments/gpu_ppo_smoke.py`, torch, width 64 x 2 blocks, bf16, L4):
18.2k samples/s at 4,096 games x 32 steps; per iteration of 131k decisions: env 32 x 17.5 ms, policy forward
32 x 80 ms, PPO update 4.0 s. The engine is ~15% of the loop; the rest is torch-side glue (unpacking the 131k-bit
mask per game, building 178 feature channels from the int16 view, re-encoding in the update). No invalid actions
(status != 0 fraction 0.0). 16k games OOM an L4: 275 KB engine state + scratch per game, plus the 64 KB per-thread
stack reservation (~5.7 GB on an L4) and torch's buffers.

Spend: ~$1 on Modal (CPU gates dominate; T4 + L4 time ~10 minutes).

Next levers, in order: (1) a device-side feature kernel (encode_view in CUDA, bf16, straight into the policy
tensor) and factored masks (per-family any-bit, per-row any-bit, target row gathered after the source choice) so
torch never touches the packed mask; (2) shrink per-game scratch (two open lists of 64 + 48 KB) and measure the
real stack frame (`-Xptxas -v`) to drop PC_CUDA_STACK_KB; (3) wire the production JAX network through dlpack;
(4) H100 run for the samples/s-per-GPU number the cost case rests on.

## Production stack, GPU-resident (2026-11-06 evening): measured

`experiments/gpu_resident_train.py` (`::jax_train`): the C engine's device buffers go to JAX through dlpack (zero copy,
`--check-dlpack` passed), the production `view_features` / `network.sample` / `ppo.loss` run on the same card
(normalized-v2, 4 blocks x 128, bf16 trunk, DEFAULT matmul, HIGHEST heads, turn clock, 1 epoch: hero-1v1's learner
settings; 2.02M params on obs v3 = 151 board + 525 global channels). Pure self-play, policy value head as critic.

| GPU | games | minibatch | steady samples/s | tick ms | encode ms | sample ms | update s / iter | decisions | finished |
|---|---|---|---|---|---|---|---|---|---|
| L4 | 4,096 | 2,048 | 10.9k | 24.4 | 5.9 | 95 | 11.1 (131k) | 3.9M | 140 |
| H100 | 8,192 | 4,096 | **91.7k** | 18.8 | 2.2 | 22.1 | 1.85 (262k) | 31.7M in 7 min | 13,216 |

Zero invalid actions in both. On the H100 an iteration (262k decisions) is rollout 1.38 s (32 steps x 43 ms: engine
18.8, encode 2.2, policy 22.1) + GAE 0.12 s + update 1.85 s: the update is 55%, the policy forward 21%, the engine
18%. The first iteration carries ~3 minutes of XLA compile (gae scan, train_step); `samples_per_s_steady` excludes it.
The fatbin needs sm_90 for H100s (added); PC_CUDA_STACK_KB defaults to 32 after ptxas reported a 26,624-byte k_tick
frame.

Cost: 91.7k samples/s on one Modal H100 ($3.95/h) = 330M samples/h = **~$12 per billion samples**, against ~$50-60/B
for the Oct 28 hero pipeline (208k/s on 4 H100 + 17 A10G actor containers + egress) — **4-5x cheaper per sample**,
and 1.8x the per-H100 rate of that pipeline's learners (52k/s each) with no actor fleet. A 7 B-sample hero run is
~21 h on one H100, ~$85, versus $470-1,150. Wall-clock per card is below today's 4-card fleet; N cards scale
linearly (independent game sets, periodic weight sync).

Where the next 2x is (all orthogonal to the engine): overlap the engine tick with the policy forward (two game
sets double-buffered; hides the 18.8 ms engine entirely -> ~115k/s), more resident games (tick latency amortizes:
17 -> 26 ms from 4k -> 8k on L4), and the learner's own kernel efficiency (the update runs ~1.8 GFLOP/sample at
~10-15% of H100 bf16 peak; cuDNN NHWC convs / fused heads are the lever, and they would help the current pipeline
equally).

### Second H100-class burst (served as an H200): two game sets, mb 8192, DEFAULT heads

`--games 16384 --sets 2 --minibatch 8192 --head-precision DEFAULT --bf16-trunk`, 420 s: **97.3k samples/s steady**,
35.1M decisions, 8,711 games finished, 0 invalid actions. Per 524k-decision iteration: rollout 2.6 s (step 81 ms x 32),
GAE 0.22 s, update 3.45 s. Findings: DEFAULT heads sped the forward 25% (22.1 -> 17.6 ms per 8192) but heads +
minibatch 8192 sped the update only 7% (142k -> 152k samples/s update-only) — the update is the trunk convs, running
at roughly 27% of bf16 peak (~1.8 GFLOP/sample fwd+bwd), so the learner is not grossly inefficient; the network is
what it costs. The overlap did not materialize: step 81 ms = 2 x engine + 2 x policy and engine_wait = 0, i.e. the
engine kernel ran before, not beside, the JAX work. Cause: the engine stream was created blocking (implicit
synchronization with the legacy default stream torch uses); fixed with cudaStreamNonBlocking, re-measured below.

### Third burst (real H100, engine stream non-blocking): 97.5k samples/s

Same configuration with the engine stream created `cudaStreamNonBlocking`: **97.5k samples/s steady**, 35.7M decisions in
425 s, 14,045 games finished, 0 invalid actions. Step 75.8 ms (was 81.2) with engine_wait still 0: the engine kernel
now runs beside the JAX work, but the JAX work slows by about the same amount (8,192 resident engine threads with
26 KB stacks occupy SM slots and bandwidth), so the engine's ~15-20% is real GPU occupancy, not hidden latency. Per
524k-decision iteration: rollout 2.43 s, GAE 0.23 s, update 3.56 s (the trunk convs; ~27% of bf16 peak).

### Final before / after (14x14 two-seat, the hero 4x128 network, 2.02M params)

| | hardware | samples/s | $/h | $/B samples | 10 B-sample run |
|---|---|---|---|---|---|
| Oct 28 hero pipeline (measured) | 4 H100 + 17 A10G actor containers + egress, Modal | 208k | ~$35-40 | **$50-60** | 13.4 h, ~$500-600 |
| GPU-resident, 1 H100 (measured) | 1 H100, Modal | 97.5k | $3.95 | **$11.3** | 28.5 h, ~$113 |
| GPU-resident, 4 H100 (projected, data-parallel) | 4 H100, Modal | ~390k | $15.8 | $11.3 | 7.1 h, ~$113 |
| GPU-resident, 1 H100-class card at a single-card vendor (projected) | Verda / Daytona H100 or PRO 6000, $1.5-2.5/h | ~90-100k | $1.5-2.5 | **$4.5-7** | ~$45-70 |

Measured on Modal: **4.4-5.3x cheaper per sample** and 1.9x the per-H100 rate of the old pipeline's learners, with no
actor fleet. The further 2x to ~$5/B comes from the fact that a GPU-resident run needs one card and no cluster
networking, so the cheaper single-card vendors (Verda, Daytona; H100 at ~$2/h, PRO 6000 at ~$1.5-2/h) apply, which
the old actor-fleet design could not use well; that row is a projection from the measured per-card rate and the
Oct 23-28 vendor prices, not a run. Horizontal scaling is data-parallel with gradient allreduce of 2M params
(the existing learner's multi-GPU path); keep the recipe's 262k batch by shortening segments as cards are added.

What did not move the number: head precision (forward +25%, update +7%), minibatch 8192 (+7% update), overlapping
engine and policy (engine occupancy is real). What would: the update is 55% of the loop at ~27% of bf16 peak (cuDNN
NHWC convs, fused heads, larger per-card batch), and the engine's SM footprint (warp-per-game fan-out of the mask
build, smaller scratch so more games fit per SM).

Total Modal spend for the project: ~$4 (CPU gates ~$1, GPU bursts ~$2.5, image builds).

### Single-card vendor row: not measured (2026-11-06)

An attempt to measure the ~$5-7/B row on Daytona was abandoned: the H100 spot box was reclaimed within minutes of
creation, and on-demand GPUs need prepaid GPU credits the organization's wallet does not hold. The user asked to keep
testing on Modal only. The vendor row therefore stays a projection from the measured per-card rate (97.5k samples/s on
an H100) and the Oct 26 list prices (experiments/daytona/launch.py PRICES: PRO 6000 $1.74/h spot, H100 $2.27/h spot;
plus ~$0.46/h for 4 vCPU / 16 GiB). Modal spend for the whole project: ~$4.

### Policy-play lockstep gate (2026-11-06, after the main merge)

`::equality --policy /results/distributed/hero-1v1-cont1h/latest.npz` (update 11650, 4x128 normalized-v2, obs v1, the
strongest 1v1 checkpoint on the volume; 95% of actions from the policy at temperature 1, 5% uniform legal, forced resigns
never legal under no_resign): 8 processes x 2000 ticks x 24 games = **384,000 steps, 0 divergences**, 87 games finished by
elimination, games reaching turn 49-54, 168 full-state exports compared. C-CPU vs Rust; the CUDA == C-CPU gate is the
same code path (re-run after the ABI change: 14x14 T4, 0 divergences). The policy forward on one pinned CPU core costs
~190 ms per 24-game tick, so the gate is policy-bound (~10 min per 2000 ticks).

## Recipe-faithful trainer (2026-11-06 evening): first measurements

`rl/gpu_resident.py` (map: `docs/hero-recipe-map-2026-11-06.md`) continuing hero-1v1-cont1h with its critic on one Modal
H100, 600 s each, run folders `/results/distributed/gpu-hero-smoke-1106{,b}` (production checkpoint layout: latest /
latest-critic / snapshots / final, training.jsonl):

| run | games x sets | ring | steady samples/s | per 262k update | forced cuts | finished | kl / clip / ent |
|---|---|---|---|---|---|---|---|
| 1106 | 4096 x 2 | 96 | 34.9k | rollout 1.8 s + value 0.84 + assemble 0.75 + train 2.89 | 7.2% | 7,271 | .002 / .016 / 2.93 |
| 1106b | 4096 x 4 | 160 | 36.4k | rollout 2.0 s + value 0.45 + assemble 0.44 + train 2.92 | 1.3% | 8,810 | .001 / .009 / 3.00 |

Both with 0 invalid actions, maps drawn per the proposed-v1 weights, the lr at its decayed end (the schedule follows
`--seconds`). In-rollout critic values match the assembly pass to 2e-6 (`--value-check`, 200 s run). At the ledger's
$4.33/h one H100 gives **~$33/B** today against ~$45/B (main's vector-actor rig, 4 H100 + 6 A10G) and $83-90/B (Oct 28).
Where the time goes: the PPO update (policy + critic, micro-batch 4096 x 4) is 2.9 s = 40%; the rollout 2.0 s = 28%, now
launch-bound at 1,024 games per set (sampling 1,024 rows costs 5.9 ms for ~1.5 ms of math); assembly 0.4 s; the engine
is hidden (5.8 ms wait per step). Next: fuse each set's encode + critic + sample into one jit, then micro-batch 8192.

## Equal-sample A/B: GPU-resident vs main's pipeline (2026-11-06 evening)

Both arms continued `hero-1v1-cont1h` (latest.npz + latest-critic.npz) for exactly 400 updates = 104.9M samples under the
tp-smoke hero recipe (lr schedule over updates): `ab-main-1106` on main's vector-actor rig (4 H100 + 6 A10G, 534 s,
$33.56/h) and `ab-gpu-1106` on the GPU-resident trainer (1 H100, 3128 s, $4.33/h; commit 528afbf). Ladders: `::arena`,
deterministic, 32 maps x 4 games per seed set, seed sets 302000 and 303000 (64 games each).

| match | set a | set b | combined |
|---|---|---|---|
| GPU-resident vs parent | 28-34-2 (44%) | 35-28-1 (55%) | **63-62-3, 49%** |
| main vs parent | 20-33-11 (31%) | 10-36-18 (16%) | **30-69-29, 23%** |
| **GPU-resident vs main** (head-to-head) | 34-17-13 (53%) | 34-12-18 (53%) | **68-29-31: 70% of decided games** |

Training metrics over the 400 updates: GPU-resident approx_kl .004-.005 / clip .035-.055, main .008-.011 / .10-.13 (the
on-policy gap: main's actors run weights ~8 versions stale and dropped 22k rows); critic loss .002-.005 vs .004-.014;
main's entropy rose to 4.2 and its return std fell to .29 in the last 200 updates. Verdict: at equal samples the
GPU-resident trainer learned at least as well as production's pipeline (held the parent's strength where main's
continuation lost ground, and beat main's result 68-29 head-to-head). Parity holds; the per-sample cost comparison
($34/B vs $45/B measured; ~$24/B projected with the engine footprint work) stands on learning-equivalent samples.
Caveat: one parent, one recipe, 105M samples, 64-game deterministic ladders (CI about +-12 points per set).

## Engine SM footprint: profile, smaller per-game memory, warp-per-game tick (2026-11-06, late; gates pending)

Problem: in the GPU-resident trainer the rollout step sits at ~65 ms per 4,096 games however the JAX side is arranged;
the engine's resident threads hold SM slots and memory bandwidth beside the policy forward (fused JAX work 8.7-13 ms for
~3-4 ms of math). Thread-per-game also makes every engine load uncoalesced: the 32 lanes of a warp read 32 different
games' state, 32 transactions per load. Three changes, all bit-for-bit (nothing below changes a result; each is gated).

**1. Phase profile (`-DPC_PROFILE`, Makefile `PROFILE=1` -> `lib*_prof.so`).** Per game and phase, clock64 cycles of the
thread (the warp kernel: lane 0, after a lane sync) on the GPU, nanoseconds on the CPU driver, summed on the host by the
new ABI call `pc_profile_read` (pc_api.h: 16 phase counters, ticks, games, counter unit, k_tick time from CUDA events,
kernel, threads/block, PC_MAXSIZE; `pc_profile_reset`, `pc_build_info`). Phases (pc.h `PC_PH_*`): step, outputs, final
views, reset, mask prelude (active / peace / rewards), mask move / attack / summon / build / research / economy / unit /
diplomacy, legal-cache copy, observe mover, observe others. `GpuGames.profile_read()` turns it into us per game-tick and
shares; the harness prints `PROFILE {json}` rows with `--profile`. Profile builds synchronize after every tick.

**2. Per-game memory (bytes; `sizeof` from clang).**

| struct | before | full build now | `SIZE=14` build (PC_MAXN 196) |
|---|---|---|---|
| PcState | 83,184 | 83,184 | 48,816 |
| PcScratch | 174,904 | 113,864 | 68,936 |
| - PcPathScratch | 90,836 | 54,468 | 32,964 |
| - PcStructScratch | 66,816 | 42,144 | 25,504 |
| PcMaskScratch | 16,280 | 14,336 | 8,704 |
| **per game** | **274,368** | **211,384 (-23%)** | **126,456 (-54%)** |
| PcGeom (one copy) | 118,284 | 118,284 | 46,476 |

- Unit-search open list: capacity min(1 + 8 x 1000, 18 n + 1) = 5,833 (3,529) entries, tiles u16: 35 KB (21 KB) instead
  of 64 KB. Bound: a tile is pushed when unvisited or when its cost strictly improves; step costs are 5 / 10 / 1000 and a
  tile expands only while cost < 10 x (movement + boost) <= 40, so a pushed cost takes one of 17 values (+1 for the
  source). `pc_set_rules` refuses a rule table with a supported unit moving more than 3 (`pc_pathfinding_open_fits`), and
  the push compacts instead of dropping if the bound were ever missed (time only, never an entry).
- Router open list: min(8001, 12 n + 1) = 3,889 (2,353) entries: 23 KB (14 KB) instead of 48 KB (costs 5 / 10, expansion
  below range x 10 <= 50: 11 values per tile + the origin).
- u16 where the value is a tile index without a negative sentinel: the search_ref heap ids, `PcPathScratch.path`, the
  attack-target lists. i32 kept for `parent` / `came_from` (-1 sentinels), costs and distances (PC_INF). The mask
  scratch's emit lists moved to the stack (`reached`, `targets`).
- `-DPC_MAXSIZE=14` (Makefile `SIZE=14` -> `libpc_cpu_14.so`, `libpc_cuda_14.so`): every array sized for 196 tiles;
  `pc_create` rejects 18x18 there. `rl/gpu_engine.default_so_path` takes it for boards <= 14 when it exists
  (`POLYTOPIA_PC_VARIANT=full` keeps the full build, `=14` insists); the trainer at 14x14 picks it up automatically.
- Stack frame (reasoning; ptxas numbers come from `::build`): the deep chain is the step's structures code, whose locals
  are PC_MAXN-sized int32 arrays, so the 26.6 KB of k_tick should scale to ~16 KB in the 14 build; observe lost its five
  per-tile stack arrays (2.3 KB); the warp kernel's mask path adds one lane-private PcMoveScratch (6.2 / 3.7 KB), below
  the step chain. `PC_CUDA_STACK_KB` stays 32 by default; with the 14 build's frame confirmed, 20 frees ~3 GB of an
  H100's stack reservation (2,048 threads x 132 SMs x 12 KB).

**3. Warp-per-game tick (`PC_CUDA_WARP_TICK=1` at pc_create; `k_tick_warp`, 128 threads = 4 games per block,
`PC_CUDA_WARP_TPB`).** One game per warp: lane 0 runs decode, step_with, the outputs and the reset (the serial game logic
unchanged); then all 32 lanes run the masks and the views. The engine code is single-source: the hot functions take
`PcLanes {lane, count}` (pc.h) and the serial engine passes `{0, 1}`, so the CPU gate checks the same code the warp
runs. Per family: reward, summon, build (neutral road/bridge + build_row per economy tile), economy and unit rows,
research's city count, movement_flags and every observe field loop are strided over the tiles; move and attack rows run
one unit per lane (reach on a lane-private scratch holding a copy of the flags; attack_targets is pure);
BuildCtx::prepare visits the own cities in index order by ballot, ORs each city's present-structure bits across lanes
and compacts econ_list in tile order, so its lists equal the serial ones. movement_flags' ZOC scatter becomes a gather
(the in-board neighbour relation is symmetric). Mask bits are a set, so the lanes set them with atomic ORs into the
legal-cache row, which is now word-aligned per game (stride round4(mask_bytes)); the warp zeroes it, fills it, and copies
it byte for byte to the output row (the output memset is skipped for this kernel). Both kernels stay selectable and
both are gated. `-DPC_WARP_MIN_BLOCKS=k` (NVEXTRA) caps k_tick_warp's registers if ptxas shows occupancy is the limit.

Trade-off to measure, not assume: a warp holds one game instead of 32, so for the same registers per thread an SM holds
32x fewer games at once (at 255 registers: 8 warps per SM = 1,056 games on an H100, so 4,096 games run in ~4 waves),
against coalesced per-tile loads and 32-way parallel masks/views. The profile decides where it pays.

What is not measured yet (this section was written before any GPU run): the profile split, the tick time at 4,096
games for each kernel, and the effect in the trainer. Commands:

    modal run --env staging experiments/modal_gpu_engine.py::build        # ptxas registers / stack frame per kernel, both builds
    modal run --env staging experiments/modal_gpu_engine.py::equality --size 14 --games 24 --ticks 2000 --procs 8
    modal run --env staging experiments/modal_gpu_engine.py::equality --size 14 ... --variant full
    modal run --env staging experiments/modal_gpu_engine.py::equality --size 18 --games 24 --ticks 1200 --procs 4
    modal run --env staging experiments/modal_gpu_engine.py::gpu_equality --size 14 --ticks 500 [--warp 1] [--variant full]
    modal run --env staging experiments/modal_gpu_engine.py::gpu_equality --size 18 --players 4 --games 16 --ticks 300 --warp 1
    modal run --env staging experiments/modal_gpu_engine.py::gpu_equality --size 14 --ticks 400 --warp 1 \
        --policy /results/distributed/hero-1v1-cont1h/latest.npz
    modal run --env staging experiments/modal_gpu_engine.py::gpu_equality --gpu H100 --size 14 --ticks 400 --profile --modes rust [--warp 1] [--policy ...]
    modal run --env staging experiments/modal_gpu_engine.py::tick_bench --gpu H100 --games 4096 [--profile]

`::tick_bench` (engine-cu/tools/tick_bench.py) plays 4,096 games with random legal actions sampled on the GPU, through the
thread kernel then the warp kernel with the same seed, and compares every tick's output checksums between the two
(`TICKBENCH_CMP checksums_equal`), so it is also a bitwise check at trainer scale.

## Engine footprint + data-parallel results (2026-11-06 night, all on Modal H100s, hero recipe continuing hero-1v1-cont1h)

Engine tick alone (`::tick_bench`, 14x14 2p, random play, both views, output checksums thread == warp): 4,096 games
thread kernel 13.5 ms, **warp-per-game kernel 4.1 ms (3.3x)**; 8,192 games 14.4 vs 6.7 ms. Gates for every change: CPU
vs Rust 14x14 384k steps on the 14 build, 18x18 115k, policy play 384k; CUDA thread and warp vs Rust and CPU on T4
(14x14, 18x18 four seats): 0 divergences.

| trainer configuration | samples/s | per card | $/B (ledger $4.33/h per H100) |
|---|---|---|---|
| 1 H100, thread kernel, 4 sets (before) | 36.4k | 36.4k | $33.1 |
| 1 H100, warp kernel + 14x14 build, 4 sets | **40.5k** | 40.5k | **$29.7** |
| 1 H100, warp, 2 sets | 38.8k | 38.8k | $31.0 |
| 4 H100, one host thread for all devices (thread or warp kernel, 1-4 sets) | 102-108k | 26-27k | $45 |
| **4 H100, one driver thread per device, warp, 2 sets** | **155.3k** | 38.8k | **$31.0** |

Single card: the rollout step (4 sets x 1,024 games) is 50.7 ms whether 2 or 4 sets, i.e. GPU-bound (fused encode +
policy + critic + the engine kernel), and the update is 3.3 s per 262k; the two halves are now about equal. Four
cards: per 262k update rollout 0.83 s, assembly 0.20 s, train 0.72 s; 96% of the single-card rate per card. With one
host thread for all four devices the fused calls, bookkeeping and assembly serialised and the GPUs waited.

Against main's vector-actor pipeline (same recipe, measured live, same ledger formula): 4 H100 + 6 A10G 206-215k/s
at $33.56/h = ~$45/B; 8 H100 + 8 A10G 360-410k/s at $54.61/h = ~$40/B. A 10 B-sample run: main 4+6 13.2 h / ~$444,
main 8+8 7.2 h / ~$394; GPU-resident 4 H100 17.9 h / ~$310, 8 H100 ~9.4 h / ~$310 (projected at the measured 96%
scaling). The equal-sample A/B (above) showed the GPU-resident samples learn at least as well. Remaining levers: bf16
activations (+14% measured on the learner, needs a learning check, helps both designs) and cheaper single-card
vendors (the loop needs one container and no cluster networking).
