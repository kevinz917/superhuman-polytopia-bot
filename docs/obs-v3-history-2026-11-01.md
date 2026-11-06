# Observation version 3: leak-free opponent history (2026-11-01)

Humans beat the bot on the public site with a Riding -> Chivalry knight rush it does not anticipate
(docs/site-games-analysis-2026-10-31.md). Until now the policy saw a snapshot of the current turn only: no memory of
where enemy units were, nothing about the opponent's techs. Version 3 adds engine-computed history inputs. They are
built from the viewing seat's own past views only, so they leak nothing a human player could not know, and every
older network keeps working on the new features (`observation.fit_board`); a version-2 network widens to version 3 with
zero input rows and computes exactly what it computed before (`network.widen_inputs`), so a live run can be swapped
mid-training.

## Features

Engine: `engine-rs/src/history.rs`, per seat `State.history` (`state.rs::History`), exported as `history.*` and optional
on import (states saved before it start with an empty history). Updated twice per turn, O(n) each: at the end of a
seat's turn (`end_of_turn`: a snapshot of what it sees) and at the start of its next turn (`start_of_turn`: the diff).
View fields (`kernels::VIEW_FIELDS`, sorted; the packed view grows from 38n + 397 to 43n + 525 int16):

| view field | shape | meaning (for the viewing seat; "enemy" = any other seat) |
|---|---|---|
| `hist_enemy_unit` | [3, n] | a visible enemy unit stood here at the end of my turn 1 / 2 / 3 of my turns ago |
| `hist_enemy_built` | [n] | an enemy structure, road or structure level appeared or changed here since the end of my last turn (tile explored then) |
| `hist_enemy_trained` | [n] | a visible enemy unit stands on an enemy city tile where none stood at the end of my last turn |
| `enemy_tech` | [128] | techs the enemy must have: the tech of every enemy unit kind, structure and road I have seen, a unit seen on a mountain (Climbing) or shallow water (Fishing), a Dagger (Diplomacy), each with its prerequisites (`Rules::tech_closure`), accumulated over the game |

Encoder (`rl/observation.py`, `OBS_VERSION = 3`): the five history planes go at the end of each view's board channels
(after the two-seat channels, the unit facing and, with more than two seats, the multi-seat one-hots), the 128 enemy-tech
entries at the end of the global vector. Widths for two seats: board 138 / 146 / 151 channels and global 397 / 397 / 525
entries for versions 1 / 2 / 3. `board_layout` / `global_layout` / `channel_index` describe the blocks; `fit_board`
decodes which (groups, seats, version) a stem reads from its width (unique by construction, tested) and gathers those
channels, so version-1 and version-2 networks (the ladder references champion, c3, hero1, bn4, hero u11650 and the run's own
earlier snapshots) read version-3 features correctly. Public demonstrations (`view_from_public`) and the retired JAX
`observe` have no history: zeros.

Determinization (`rl/belief.py`) carries `State.history` through unchanged: the searcher's history is part of its own
view, so `observe(determinize(s)) == observe(s)` still holds (tests/test_search.py).

## Tests (Modal, `experiments/modal_distributed_rl.py::tests`)

`tests/test_obs_history.py`:
- history is a function of the seat's own views: random games, and at the start of every turn each history field is
  recomputed in Python from the views the seat was given (before each of its end-turn actions and at each turn start)
  and must be equal, including the tech closure; constant through the turn;
- export/import round trip keeps the history; a State without it imports empty;
- version 1 and 2 networks read version-3 features bitwise (`fit_board`), two and four seats, one view and the critic's two;
- `widen_inputs` version 2 -> 3 gives identical policy and critic outputs; the new rows are read;
- width decoding is unique; config / checkpoint spec round trip; a tick-cost probe (200 ticks x 24 games).
Updated: `tests/test_rp2_actions.py` (newest = version 3; four-seat conversion for versions 1-3).

Modal 2026-11-01 (main-equivalent tree adcacdf): test_obs_history 10/10, test_block_pool 3/3, test_four_player 13/13,
test_distributed_rl 17/17, test_search 4/4, test_platform_profiles 8/8. The four-seat relabeling test needed its helper to
rotate the per-seat `history.*` arrays with the other per-seat state (a test fix, not an engine change).

## Global pooling bias in the trunk blocks (`block_pool`)

Per-tile source and target logits came from a trunk whose 4 blocks give a 19x19 receptive field: short of corner to
corner on 14x14 (27), and nothing board-wide reached a tile before the heads (the mean/max pooling fed only `g` and the
critic). `network.init(block_pool=True)` adds KataGo's global pooling bias inside every existing residual block: after
the block's first convolution, each channel gets a bias computed by a dense layer from the mean and max of that
convolution over the board (the layer the `extra_blocks` path already had, but on the existing blocks, so no extra
convolution: +170 k parameters on 4x128, 1.98 M -> 2.15 M). The layer starts at zero, so `network.add_block_pool` turns
any checkpoint into a pooled one with identical outputs, and the pooled features are nonzero, so it gets gradients from
the first step (tests/test_block_pool.py). `Config.block_pool` / spec key `block_pool` version the parameter layout; older
checkpoints load unchanged. `widen_obs.py --block-pool` widens a run's checkpoints (history inputs + pooling, Adam first
moments zero, second moments seeded with the median of the existing tensors so the new weights do not take 3-6x steps
while Adam's running averages fill); `random_parent.py --block-pool` builds a from-scratch parent.

## Swapping a live run (Curio)

`experiments/curio/widen_obs.py` widens a run in place (latest.npz with its Adam state over (policy, critic),
latest-critic.npz, every pool snapshot, and writes a version-3 parent), after copying the originals to a backup directory,
then checks old vs new networks on random positions. `curio.py publish --stage-only` unpacks the new commit beside the
live code and builds its engine wheel without moving `code/current` (a live process may import through it lazily);
`curio.py switch --sha SHA` installs the wheel and moves the symlink on every box, with every process stopped.

## Run `farsight` (2026-11-01)

The widen was verified on bighero's 16 checkpoints (worst output difference 2.4e-6 over 64 positions), but the user chose
a clean from-scratch run instead: `farsight` started 12:07 CST from `ckpt/scratch-init/farsight-init-4x128-obs3-pool-s1.npz`
(random init, obs version 3, block_pool, seed 1) with the bighero recipe (hero-1v1-scratch.json: no_peace, a capped game
is a loss for both seats), 29.6 h of training (ends Mon Nov 2 17:43 CST) followed by the keeper chain. bighero (same
recipe, obs version 2, no pooling) stopped at update 3880 / 1.02 B samples is the control curve: its ladder at u2060 was
19% vs c3 and ~1.5% vs champion, hero1 and bn4; `bighero_u3880` is a ladder reference. The end-of-run reliance probes
(history inputs zeroed, pooling weights zeroed) measure what the trained net uses, not a trained-without counterfactual.

## farsight's pooling bias took over the trunk, and layout 2 (farsight v2, 2026-11-01 evening)

**Result.** farsight reached 1.5 B samples (update 5660) clearly behind the control run bighero at equal samples:
ladder vs c3 0 / 1.6 / 12.5% at updates 1750 / 3580 / 5360 where bighero had 19% at update 2060; vs `bighero_u3880`
0 / 12.5 / 23%. Its per-update KL ramped 0.010 -> 0.03 (clip fraction 0.18) twice at a fixed learning rate, which the
adaptive scale (`LearnerConfig.kl_adapt_target`) then fought all the way down to 0.44 with the KL still at 0.017 when
the run was stopped. bighero resumed on the cluster for the rest of the window; farsight's checkpoints stay in
`runs/farsight` on the head.

**Diagnosis** (measured on real positions from farsight's latest network). In every block the pooling bias has an RMS
3-5x that of the convolution output it is added to: block 0 13.3 vs 2.8, block 1 20.1 vs 3.7, block 2 21.0 vs 3.9,
block 3 13.1 vs 4.2. The pooled inputs are un-normalized (|mean| ~2, |max| ~3 and growing with the trunk), the dense
layer's weight gradient sums over all 196 tiles, and nothing bounds its scale, so a shared board-wide vector dominates
the per-tile LayerNorm that follows it: after the norm every tile looks like the normalized bias, and the spatial
pathway is starved. Three departures from KataGo made this possible: the pooled statistics were not normalized, the
bias joined the convolution before the block's normalization, and it did so in all 4 blocks on all 128 channels.
The same mechanism explains the KL ramps: one pooling weight shifts every tile of every position, so a fixed Adam step
moves the policy further than in bighero.

**Layout 2** (`network.BLOCK_POOL_LAYOUTS`, spec `block_pool: 2`; farsight's checkpoints keep `block_pool: true` =
layout 1 and load unchanged):

```
h  = conv(x, k0)                                    # the block's first convolution
hn = layer_norm(h)                                  # per tile, as before
p  = concat(layer_norm(mean_tiles(hn[:, :32])), layer_norm(max_tiles(hn[:, :32])))   # 32 of 128 channels pooled
h' = silu(hn + gate * dense(p))                     # the gated bias joins AFTER the norm; gate is a scalar at 0
x  = activate(x + conv(h', k1))
```

in blocks 1 and 3 of 4 only (`block_pool_indices`). The pooled statistics are normalized, so the bias cannot track the
trunk's scale; it is added after the per-tile LayerNorm, so it can never dominate that normalization (it shifts the
pre-activation per channel, which is what a bias should do); and its scale is a single learnable gate that starts at
0, so the block computes exactly what it computes without pooling at init and the bias can only grow at about one
learning rate per update. The dense weights are randomly initialized under the zero gate: the gate's gradient is
nonzero from the first step, the dense layer's from the second. The critic (omniscient trunk) carries the same
layers. Parameters: +2 x (64 x 128 + 128 + 1) = 16.6 k on 4x128 (layout 1 added 170 k).

**Conversion of farsight.** No parameter setting of layout 2 reproduces layout 1's nonzero bias: layout 1 computes
`layer_norm(h + b(h))` and layout 2 `layer_norm(h) + g * b'(h)`, and farsight's `b` is 3-5x `h`. A layout-2 network
starts from the switched-off function, i.e. farsight with its pooling weights zeroed, which `pool_probe.py` measures
as a large change of farsight's policy. On farsight's final network (update 5660, `pool_probe.py`, 8 self-play
games, 178 positions) the ratio bias_rms / conv_rms per block was 4.7 / 7.5 / 5.5 / 4.0 on average (max 9.8; bias RMS
18 / 30 / 24 / 16 against convolutions of ~4), i.e. still growing since the mid-run measurement above, and with the
pooling switched off the family distribution moved by 1.29 nats of KL on average (max 11.6) and the tiles by 133% of
their RMS: the trunk had been rebuilt around the bias. So farsight v2 starts from a random parent on the fixed
architecture (`random_parent.py --block-pool`, `node.py hero_random_parent` with `block_pool: 2`), and a
converted-but-perturbed farsight is only a fallback.

**Tests** (`tests/test_block_pool.py`, Modal): zero-gate identity for policy and critic, the gate gets a gradient at
step 1 and the dense layer none until the gate moves, the bias is linear in the gate and stays within a small factor
of the normalized tiles on real positions, both layouts round-trip through `network_spec` / `checkpoint_template`,
load their own checkpoints and refuse each other's, `random_parent.py --block-pool` writes a layout-2 parent.

**Monitoring.** `trunk(params, features, pool_stats=[...])` reports `(conv_rms, bias_rms, gate)` per pooling layer;
`experiments/daytona/pool_probe.py` (and `hero.py pool-probe RUN`) plays sampled self-play with a checkpoint and
reports bias_rms / conv_rms per layer, the gates, and the family-KL / value change when the pooling is switched off.
Stop rule for the run: a bias past ~0.5x the normalized tiles, or the KL-adaptive scale below 0.8.

## Run `farsight-v2-1v1` (Daytona)

Plan `farsight-v2-1v1` in `experiments/daytona/hero.py`, config `experiments/hero/farsight-v2-1v1.json` = the bighero
recipe (no peace, a capped game is a loss for both seats, lr 2e-4 -> 1e-4 over the last 25%, `kl_adapt_target` 0.01)
on Daytona's platform knobs, seed 821; 4x RTX PRO 6000 spot learner + 5 PRO 6000 GPU-inference actors (~$30/h, the
lookout topology), random layout-2 parent written on the hub. Ladder every 2 h on 32 proposed-v1 maps x 2 seats vs c3,
bn4, champion, hero1, `bighero_u3880` (`hero.py sync-ckpt` from the head's `ckpt/bighero-u3880`) and farsight's final
checkpoint (`hero.py sync-farsight`, update 5660). Read against bighero's curve: 19% vs c3 at update 2060; farsight
0 / 1.6 / 12.5% vs c3 and 0 / 12.5 / 23% vs bighero_u3880 at updates 1750 / 3580 / 5360.
