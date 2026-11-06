# KataGo, read against our Polytopia bot (2026-10-23)

Question: which of KataGo's techniques ([lightvector/KataGo](https://github.com/lightvector/KataGo),
[KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md), Wu 2019
[arXiv:1902.10565](https://arxiv.org/abs/1902.10565)) would help the scratch champion lineage, given what we have
already measured? Companion to [credit-assignment-review-2026-10-21.md](credit-assignment-review-2026-10-21.md) and
[alphastar-review-2026-10-21.md](alphastar-review-2026-10-21.md).

## What KataGo is, and the one structural difference

KataGo is AlphaZero (MCTS in the self-play loop, the net trained on search targets) plus about a dozen efficiency
tricks. The 2019 paper claims ~50x less compute than ELF to reach the same strength. The paper's ablations, each
measured as "compute needed to match the full recipe":

| Removed component | Compute factor | Applies to us? |
|---|---:|---|
| Ownership + score auxiliary targets | **1.65x** | **Yes, directly** |
| Global pooling inside the trunk | **1.60x** | **Yes, directly** |
| Game-specific input features | 1.55x | Already have them |
| Playout cap randomization | 1.37x | Only with search in the loop |
| Opponent next-move auxiliary policy | **1.30x** | **Yes, directly** |
| Forced playouts + policy target pruning | 1.25x | MCTS only |

The factors multiply to ~9x. The difference that matters is that we are model-free PPO with test-time search.
Our d=1 turn searcher is a weak improvement operator: +50 Elo on the champion, depth flat, and one-round
distillation gave nothing because 87% of its overrides fell within noise. KataGo's MCTS gives hundreds of Elo. So the
search-target half of KataGo (playout cap randomization, forced playouts, policy surprise weighting, the soft policy
target) does not transfer until we have a search worth distilling. The network and data half transfers almost
unchanged, and it hits our known gaps.

## The gap in our current recipe that KataGo points at

Two facts from main (`rl/network.py`, `rl/observation.py`, `rl/distributed.py`):

1. **The policy trunk gets no supervised signal.** With `omniscient_critic`, the critic is a separate network. The
   policy's own value head is never trained: scratch lineages need a post-hoc fog head before they can search. So the
   champion's conv trunk learns only from the PPO advantage, one noisy scalar per decision. KataGo's trunk is shared by
   policy, value, ownership, score and opponent-policy heads. Its biggest ablation, 1.65x, is exactly "give the trunk
   dense per-tile targets".
2. **Tile features never see global state.** `global_state` (turn, stars, tech, relations) is concatenated only after
   pooling, into `g`. The source head (`dense(tiles, params["source"])`) uses tile features alone. Which unit or city
   acts first therefore cannot condition on stars or tech except through the family choice. The default trunk is the
   stem plus 2 blocks, 5 3x3 convs, an 11x11 receptive field. A tile on a 14x14 or 18x18 board does not see the far
   side of the map. KataGo's fix is global pooling bias layers inside 2-3 blocks (ablation: 1.60x).

## Recommendations, ranked by expected value per dollar

### 1. Auxiliary heads on the policy trunk (KataGo: ownership/score 1.65x, opponent policy 1.30x)

These heads are used in training only. The actor's `sample` path never computes them, so inference cost stays
unchanged. All the targets exist in the actor already, because `both_views` gives us omniscient final state.

| Head | Target | KataGo analog | Why for us |
|---|---|---|---|
| Final ownership | per tile: me / opponent / neutral at game end (tile `owner`, plus cities) | ownership, w = 1.5/b² | localises the ±1 outcome onto tiles: credit assignment, the 18×18 plateau |
| Future ownership | the same at +3 and +10 own turns | `futurepos` | survives long and capped games, where the game-end target is far |
| Score margin | final Polytopia score difference, as a categorical or atan-scaled regression | score belief pdf/cdf | gives signal in seat-decided games (see 3) |
| Opponent next turn | the opponent's next-turn summons by unit type, plus the tiles it attacks or moves to | aux opponent policy, w = 0.15 | cheap; teaches threat reading |
| Hidden state (fog) | enemy unit type and owner on fogged tiles, enemy stars bucket | none (Go has no fog) | belief learning; the policy sees only its view, so this is not a leak |
| Short-term value | exponential average of the omni critic's future values, horizons ≈ 2 / 6 / 15 own turns | short-term value targets | trains the own value head jointly, so no post-hoc fog-head step before search |

Implementation notes:

- Add the heads as an `architecture` version, like `normalized-v2`, so old checkpoints still load. `grow`-style zero
  init keeps a continuation from the champion exact at step 0.
- Attach targets once per turn, not per decision. There are ~15 decisions per turn, so per-decision rows would cost
  ingestion bandwidth, and the assembler is our wall. A 196-tile int8 ownership map per turn is small after zstd.
- Loss weights: start from KataGo's, scaled to our loss magnitudes (ownership ≈ 1.5/tiles summed per tile, opponent
  policy 0.15), and log each head's loss separately.
- **Test:** 3-h continuation of scratch-cont-2 with aux heads vs a same-length plain continuation (the control the
  B1 pilot lacked), 512 seat-paired games on `map_mix='proposed-v1'`. Cost ≈ 2 arms × 3 h × ~$20/h ≈ $120. A from-scratch
  1-h arm would show the early-learning-speed effect more cleanly (KataGo's claim is about speed), ≈ $40 for the pair.

### 2. Global context in the trunk (KataGo: global pooling, 1.60x)

- Broadcast `global_state` as constant input planes into the stem (KataGo feeds rules and komi this way).
- Add a KataGo global pooling bias in each block: pool the block's first conv output (masked mean, mean × size/14,
  max), pass it through a dense layer to per-channel biases, and add those before the second conv. Zero-init the
  dense layer so a grown checkpoint computes the same thing at step 0.
- Feed `g` into the source head as well: `dense(concat(tiles, broadcast(g)))`.
- Compute: a few dense layers per block, negligible next to the convs. It touches `network.trunk`, so the Rust/JAX
  bitwise tests and the actor `encode_view` path need re-validating.
- Can share an arm with 1 (one "KataGo-trunk" arm vs control), at the cost of not separating the two effects. Given
  the budget, I would run them together first and ablate only if the combined arm wins.

### 3. Score margin in the reward, and fair starts (KataGo: score utility, komi and handicap compensation)

The seat-map study found 61% of mixed-map boards are decided by seat. In those games the ±1 says nothing about play
quality. KataGo has the same problem: a won or lost position gives no signal. It handles it in two ways.

- **Score utility.** utility = win/loss + c · (2/π)·atan((score − x0)/b), with c = 0.4–0.5 and x0 centred on the
  predicted score, so it never trades win probability for margin. The PPO analog is a second advantage stream: a
  margin head (from 1) as its own critic, A = A_win + c · A_margin with c ≈ 0.1–0.2. This is AlphaStar's
  "pseudo-reward with its own value head" form, which we already cited in the credit review. It is terminal-only, so
  it is not PBRS, which hurt. Polytopia score weights economy and tech, which is the axis the human beat OMNI on.
  It pulls the opposite way from the winner speed bonus. If both are used, keep both small.
- **Fair starts.** In 90% of KataGo's handicap games, komi compensates the weaker side. Komi itself is randomised
  (N(7, 1), and 5% of the time σ = 10), and the net sees komi as an input. The Polytopia analog: give the
  disadvantaged seat a star bonus at turn 0, sized from the omni critic's turn-0 value on that map, and expose the
  bonus as a global input. Randomise it in both directions in some games, so the bot also learns to play from behind
  and to close out from ahead. Engine work: a start-of-game star grant in the Rust engine, plus one more global feature.
- **Branching.** KataGo branches 2.5% of positions to a policy-sampled alternative, and in 5% of games picks the best
  of 3–10 openings. This is the "mid-game start states" item from the diversity plan, already costed there. KataGo
  does it in the actor by copying the game state. The Rust engine already clones state for search, so this is a
  small amount of actor code.

### 4. Optimistic policy head for search candidates (KataGo v1.13: +40–90 Elo with search)

We have measured that our search is limited by candidate quality, not depth or breadth: it rescues the same 16
blunders under every setting, and more replies did not help. KataGo's optimistic head is a second policy head trained
on the ordinary policy target, weighted by `clamp(sigmoid((z_value − 1.5)·3) + sigmoid((z_score − 1.5)·3), 0, 1)`,
where z is the surprise in the short-term outcome divided by its predicted error. It is used only inside search,
never in self-play. For us: sample half of the 16 lanes from an optimistic head trained on positive-advantage
decisions (z from the critic plus the error head in 5). UPGO was neutral, but UPGO changed the self-play policy, and
this does not. It needs heads 1 and 5 first, then only arena runs (~$15 per 200-game search arm).

### 5. Uncertainty-aware lane selection (KataGo: uncertainty-weighted playouts, LCB move choice)

The sweep showed the optimizer's curse: deeper search inflates the chosen-minus-greedy gap by the added noise.
KataGo trains heads that predict the squared error of the short-term value, then down-weights uncertain evaluations,
and picks moves by a lower confidence bound. For us, choose lane = argmax(mean − k·SE). SE can come from reply
sampling across determinizations, or from a trained error head. Override greedy only when the margin clears the
bound. An honest prior: de-noising by 64 replies bought only +16 [−65, +100], so expect small gains. Its real use is
making d=2 worth re-testing. Arena-only cost, so do it after 4 or skip it.

### 6. Bigger nets: KataGo's size schedule needs GPU actor inference first

KataGo grew b6c96 → b10c128 → b15c192 → b20c256 over 19 days. Each larger net trained concurrently on the same
self-play data and took over when its loss caught up. The run ended with a 10x LR drop. Our evidence:

- the 4x128 pilot was undertrained at 325M samples and actor-bound;
- the B1 low-LR polish gave +41 (KataGo's 10x end-of-run LR drop supports it, but our B1 still needs its control arm).

The concurrent-training trick does not transfer directly to PPO, because the larger net's data is off-policy. Our
`grow()` (function-preserving extra blocks) is the right PPO substitute. The blocker is actor cost: network forwards
are ~60% of the actor tick. KataGo's answer is the one in the 10x plan's Stage B: batch many games' forwards onto one
GPU per node (Curio's 4090s fit this). KataGo v1.17 is also moving its main run to transformers. With 196–324 tiles
and our receptive-field gap, attention is a natural later bet, once inference runs on a GPU.

### 7. Robustness: KataGo's adversarial-policy history is our human game

Wang et al. 2023 ("Adversarial Policies Beat Superhuman Go AIs") trained a cheap adversary that beat superhuman
KataGo with a "cyclic group" trick that weak humans can then copy. KataGo patched that exploit by training on
adversarial positions. Follow-up work (Tseng et al. 2024, "Can Go AIs be adversarially robust?") found that new
exploits still appeared after each defence. The human who out-built OMNI c3 on economy is the same phenomenon: a
self-play-only lineage with one style of opponent. The KataGo lesson: run exploiter agents against the frozen champion
as a standing eval (`league.py` already has the machinery), feed exploit games back as starting positions (branching,
3), and don't treat Elo against our own lineage as robustness.

### Lower value or not applicable

- **Only with search in the loop** (revisit if expert iteration comes back): playout cap randomization (cheap searches
  to play, expensive ones for labels, 1.37x), policy surprise weighting (oversample states where search disagrees
  with the prior), and the auxiliary soft policy target (T = 4, 8x weight). Together these are the fix for "labels
  are mostly noise" in our distillation failure, if we retry it.
- **MCTS-tree specific:** subtree value bias correction, variance-scaled cPUCT, graph search, shaped Dirichlet noise,
  root softmax temperature. Our searcher has no tree.
- **Mixed board sizes in one net** (masking; KataGo trains 7×7–19×19 together, with early learning at least as fast).
  Our net is already size-agnostic, apart from unmasked mean pooling. Relevant only if 18×18 comes back: train on a
  14/18 mix rather than separate runs.
- **Normalization and blocks** (fixed-variance init, one batch norm, nested bottleneck blocks): tuned for 20–40-block
  nets. At 2–4 blocks, layer norm is fine.
- **Human SL model** (v1.15 `humanv0`, rank-conditioned imitation): later, once the public site has enough human games
  for human-style opponents and a "humanlike" difficulty.

## Suggested order

1. One "KataGo-trunk" arm, aux heads (1) plus global context (2), against a plain control, both 3-h continuations of
   scratch-cont-2 with the critic loaded. ≈ $120. This is the highest-confidence transfer, and it also hands the
   searcher a jointly trained own value head.
2. If it wins: optimistic-head candidates (4), arena-only.
3. Fair starts, margin advantage and branching (3), folded into the planned 8-h diversity run rather than a run of
   their own.
4. Bigger nets (6) only after GPU actor inference lands.
