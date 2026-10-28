# Next small run, and how strong the bot really is (2026-10-20)

Combines three threads: the 18×18 runs ([rl-18x18-2026-10-19.md](rl-18x18-2026-10-19.md), the turn-clock pilot in
[long-horizon-credit-2026-10-20.md](long-horizon-credit-2026-10-20.md)), the throughput diagnosis of the same pilot
([rl-18x18-bottlenecks-2026-10-20.md](rl-18x18-bottlenecks-2026-10-20.md)), and new research on (a) why self-play
PPO plateaus and what to change, (b) what "superhuman" evidence looks like and what we can credibly claim.

## Part 1 — where we stand on strength

### What we have measured (all against our own bots, frozen engine, both seats)

| Step | Opponent | Result | ≈ Elo gap |
|---|---|---|---|
| league96 (imitation + RL) | scripted teacher (`ScriptedTeacher('balanced')`) | 64–0 | > +500 |
| u3142 | league96, untouched final maps | 117–10–1 (91%) | +410 |
| c3 (u4854) | u3142, `proposed-v1` mix | 179–34–43 (70%; 84% of decisive) | +150–290 |
| c3 + leak-free robust search | raw c3, mix | 114–49–37 | +117 [+87, +149] |
| c3 + leaky search (upper bound, cheats on fog) | raw c3, classic | — | +200 |

Chained, c3 sits roughly 1,000+ Elo above a hand-written heuristic. That number has no external meaning:
every rung is our own lineage, and ladder Elo among related self-play checkpoints is inflated and can be
non-transitive ([Balduzzi et al., Re-evaluating Evaluation](https://arxiv.org/abs/1806.02643)). We have never
played a human, an independent bot, or the commercial game.

### Internal evidence that c3 is strong but far from optimal

- **One-ply search adds +100 Elo.** A policy near optimal gains little from a shallow lookahead with its own
  critic; AlphaGo's raw policy net was hundreds of Elo below the searched system. +117 from d=1 means c3 still
  makes tactical mistakes a one-turn look catches.
- **Fog reasoning is weak.** Letting search peek through fog adds another +80–150 Elo on classic maps.
- **It cannot close games.** 20–24% of 14×14 self-play games hit the 100-round cap (31% at 18×18); on ridge and
  highland maps a third stall.
- **Narrow repertoire.** Warrior/Catapult/Cloak are ~75% of recruits; tier-3 economy techs (Construction,
  Trade, Philosophy, Chivalry) appear in 11–18% of games; Knights, Swordsmen, Archers, Giants are nearly absent.
- **Exploitability untested.** A ~97%-winning adversary beat superhuman KataGo and the exploit transferred to
  amateurs ([Wang et al.](https://arxiv.org/abs/2211.00241)); we have never trained an exploiter against c3.

So the honest statement today: *c3 is the strongest agent in our environment by a wide margin, beats every
scripted and learned baseline we have, and is demonstrably sub-optimal.* Nothing supports "superhuman", and
nothing refutes "quite good".

### How published systems earned the claim

| System | Evidence | Scale |
|---|---|---|
| AlphaGo / Master | anonymous online games vs named pros, then formal matches | 60–0 online; Lee Sedol 4–1; Ke Jie 3–0 |
| AlphaZero | vs Stockfish on tournament hardware | 1,000 games, +155 −6 =839 |
| AlphaStar | anonymous Battle.net ladder, all races, human-like interface | above 99.8% of players |
| OpenAI Five | TI champions + open public arena | 7,215–42 |
| DeepNash (Stratego) | Gravon platform rating | 84% vs top experts, top-3 all time |
| Cicero (Diplomacy) | anonymous webDiplomacy league | 40 games, top 10% |
| Pluribus (poker) | 10,000 hands vs pros, AIVAT variance reduction | p = 0.028 |
| Generals.io ([2606.23348](https://arxiv.org/abs/2606.23348)) | public 1v1 ladder, 2 months, native latency; rating recomputed for 14,616 players under several K; 189–67 vs the best-ever human; simulator audited cell-for-cell on 10,000 real replays | 946/1,147 rated games (82.5% ± 2.2%) |

Common standard: blind play against *rated* humans on the venue humans use, hundreds of games, a human-equivalent
interface, head-to-head against named top players, and an audit that the simulator matches the real game.

### The Polytopia landscape

- No public 1v1 ladder. Midjiwan's tournaments run on Challengermode (Polysseum; a $10k 2025 World Championship),
  points-based. [PolyChampions](https://polychampions.com/) is a Discord team league (2v2/3v3) rated by the
  PolyElo bot. An in-game Elo exists but is not a public ladder.
- The built-in AI's difficulty only changes its capital income (+1 … +5 stars/turn) plus aggression; "Crazy" is
  routinely beaten by experienced players. Beating Crazy would say nothing about human-relative strength.
- Prior bots: Tribes (AIIDE 2020, MCTS/RHEA, "far from human strength"); Polyfish (the engine we derived ours
  from; AlphaZero-style, its network still loses ~2:1 to its own greedy heuristic per its ladder notes, no human
  claims); DeepMind's 2021 AndroidEnv integration, unpublished.
- No official API; multiplayer runs through Midjiwan's servers, so a ladder bot would almost certainly breach
  their terms (text not retrievable; treat as unconfirmed). The sanctioned route is asking Midjiwan directly;
  they have partnered with DeepMind and Challengermode before.
- Human data exists: Polyfish's scraper collected ~5,800 PolyChampions game ids (Challengermode) and exports
  "professional" replays with outcomes (`games_pro_*`, source `PolytopiaProfessional`); none of it is in the
  public repo. HenBOMB (the Polyfish author the user intends to contact) holds it.

### What we can credibly do, ranked

1. **A human evaluation on our own ruleset.** Recruit 5–10 strong players (PolyChampions/Polysseum regulars)
   to play the replay studio ([replay-studio.md](replay-studio.md)) blind against c3, c3+search and each other on
   14×14 Bardur mirrors; ≥100 bot–human games (±10% CI), plus head-to-head vs the single best player. This is
   the only thing that supports any human-relative claim for *our* rules. Caveats to state up front: Bardur
   mirror only, research rules (documented deviations in [native-rules-exceptions.md](native-rules-exceptions.md)),
   14×14 and 100-round cap.
2. **Human-replay checks, if HenBOMB shares the PolyChampions data.** Our `gpu/importer.py` reads Polyfish
   states (version 115, 2 players). On Bardur-vs-Bardur positions we can measure (a) the critic's outcome
   prediction (AUC) on pro positions and (b) policy agreement with pro moves. Agreement is not strength, but a
   critic that cannot tell who is winning in pro games is a clear negative result.
3. **Pre-registered pitfall audit, cheap and internal:** train an exploiter against frozen c3 (an adversary
   winning > 60% ends any "superhuman" talk); report the search-gain ceiling (if one-ply still adds > 50 Elo,
   say "strong"); transfer to unseen sizes and map types; rule-quirk sensitivity.
4. **Ask Midjiwan for sanctioned play.** If granted, port to commercial rules and run a Generals.io-style
   longitudinal ladder deployment with recomputed ratings. Months, not weeks.
5. **Stop quoting internal Elo as strength.** Port Tribes' MCTS/RHEA agents and a Crazy-equivalent
   income-cheat opponent as external floors; both are clearly sub-human, so beating them is a floor, not a claim.

## Part 2 — why self-play stalled on 18×18, and the next small run

### What the three threads agree on

| Finding | Source |
|---|---|
| Credit assignment was broken on 18×18 (per-decision discount, flat critic) and is now fixed; the policy still does not improve | turn-clock pilot: value std 0.25–0.28, KL to parent 0.02, parity vs parent |
| The learner is the wall-clock bottleneck (89% of each 7.5 s update is GPU time, fp32 on CUDA cores, tensor cores idle); actors are the cost bottleneck (67% of $/h) | bottlenecks doc |
| 22% of learner time recomputes the frozen anchor's log-probs every epoch; 11% is checkpoint/status bookkeeping on the training thread; warm-up updates cost full price | bottlenecks doc |
| 32-vCPU actor containers deliver 83 steps/s per paid vCPU; 8-vCPU ones 139 (1.66×) | bottlenecks doc |
| Trained samples are 5–6 policy versions old and reused 3× (3 epochs). OpenAI Five reports staleness ≈ 8 "causes significant slowdowns" and reuse 2–3× ≈ 2× slower learning; Sample Factory tolerates lag 5–10 only with V-trace corrections | research, [OpenAI Five](https://ar5iv.labs.arxiv.org/html/1912.06680), [Sample Factory](https://ar5iv.labs.arxiv.org/html/2006.11751) |
| We measure progress only against the parent. A child at parity with its parent can still have learned (beating siblings, or cycling through counter-strategies); non-transitive cycles are widest at intermediate strength | [Czarnecki et al.](https://ar5iv.labs.arxiv.org/html/2004.09468), [Balduzzi 2019](https://arxiv.org/abs/1901.08106v1) |
| The network is small: 4 residual blocks × 64 channels, 0.5M parameters, about the budget Jones (2021) found necessary for 9×9 Hex; KataGo's strength tracked network size (b6c96 → b20c256: −1276 → +76 Elo) | [Jones](https://ar5iv.labs.arxiv.org/html/2104.03113), [KataGo](https://ar5iv.labs.arxiv.org/html/1902.10565) |

### What the Generals.io result used that we do not

The closest published success (plain PPO self-play, sparse ±1, ~730-turn games, superhuman on a public ladder;
[2606.23348](https://arxiv.org/html/2606.23348)):

- **Potential-based shaping early**: Φ(s) = 0.3·clip(log(army/army′)/log 1.6) + 0.3·clip(log(land/land′)/log 1.3),
  F = Φ(s′) − Φ(s), Φ(terminal) = 0; +150 Elo at 24 h fading to +90 at 48 h; their final agent trained without it.
  In zero-sum self-play an antisymmetric potential (Φ_B = −Φ_A) keeps the game zero-sum and, by Devlin & Kudenko
  (AAMAS 2011), leaves the Nash equilibria unchanged.
- **Top-25% |advantage| filtering**: normalise Â over the batch, keep the quarter with the largest |Â|, same
  minibatch → ¼ the optimizer steps. Learning per frame is unchanged; the win is ~2× more iterations per hour on a
  compute-bound learner (+150–250 Elo at 24 h, pooled p = 0.002).
- **1 epoch** per batch (reuse ≈ 1), decaying entropy coefficient (0.05 → 1e-3), **EMA of the parameters**
  (τ 0.999) as the deployed policy (+65 Elo early, +32 late).
- Pure self-play, no opponent pool; a 15M-parameter transformer.

Three of those four (shaping, filtering, 1 epoch + EMA) are cheap changes to our learner. The fourth (model
size) is the expensive one and the likely medium-term ceiling.

### Staged plan, about $80 total

**Stage 0 — diagnosis, ≈ $8, CPU only, 40 min.** Round-robin on fresh 18×18 maps (64 maps × 2 seats per pair,
sampled play as in training) among grown c3, run 2 u220, run 3 u300, pilot u200, pilot final. Readout:
- children beat each other non-transitively → cycling: add a PFSP-style pool of the last ~8 snapshots at 30–50%
  of games and evaluate against the pool, not the parent;
- children beat siblings but not the parent → the eval was hiding progress: keep the recipe, measure against the
  pool;
- flat everywhere → no learning signal: Stage 2 is the answer.

**Stage 1 — learner and fleet efficiency, ≈ $10, no training.** Already on this branch: the
`POLYTOPIA_MATMUL_PRECISION` hook (TF32 torso) and `more_actors --memory-gib`. To add: fp32 policy/value
heads under TF32, anchor log-probs computed once per batch, checkpoint and status writes off the training
thread, trunk gradient stopped during critic warm-up, 8- or 16-vCPU actor containers. Gate for TF32 (bench on
one H100 with real actor rows): epoch-0 importance ratio exactly centred, k3 = ρ − log ρ − 1 mean < 1e-4,
max |log ρ| < 0.02; otherwise keep the heads in fp32 and TF32 only in the convolutions. Expected: learner
~3 s/update instead of 7.5, $0.20 per million samples instead of $0.40. This pays for itself in the first hour
of any later run.

**Stage 2 — the recipe test on 14×14 from c3, ≈ $45, 1.5 h.** Test the learning changes where self-play is
known to work (c3 gained 65–75% over its parent in 4 h), so a null result means the recipe, not the board:
- zero-sum PBRS on end-turn steps: Φ = Σ 0.3·clip(log(x/x′)/log k, −1, 1) over stars, unit count and territory,
  antisymmetric between seats, scale annealed linearly to 0 by 60% of the run, Φ = 0 at game end;
- 1 epoch, top-25% |Â| filtering, EMA 0.999 of the policy for evaluation and publishing;
- turn clock (γ_turn 0.97, λ 0.92), 40-update critic warm-up, no KL anchor (c3 did not need one), entropy
  0.005 → 0.001 decayed;
- 2×H100 learner + 12 × 16-vCPU actors (the filtered learner needs a quarter of the GPU work, so lag should fall
  to 1–2 versions on its own).
Pass: deterministic ≥ 58% vs c3 on fresh 14×14 `proposed-v1` dev maps (256 games, CI above 50%) after the anneal
ends, and ≥ c3's 70% vs u3142. Fail: no gain by the end, or a gain that vanishes once Φ → 0 (Generals' fade). A
pass goes straight to 18×18 with the same recipe (≈ $60 for 1.5 h at the Stage-1 rates). A fail says the
remaining lever is capacity: 6 blocks × 128 channels (~2M parameters), which needs either 2× the actor fleet or
a GPU inference service for the actors — a separate project.

**Stage 3 — strength evaluation, in parallel, mostly people time.** Train an exploiter against frozen c3
(≈ $20 of the 14×14 recipe with c3 frozen as the only opponent); set up the replay studio for invited players
and start collecting bot–human games (Part 1, item 1). Both tell us where c3 is independent of any new run.

### What this plan does not do

- It does not change the network or move actor inference to GPUs; both are the next projects if Stage 2 passes
  on 14×14 and the 18×18 recipe still stalls.
- It does not touch the frozen engine. The build-mask proposal (45% of the 18×18 actor tick) belongs in
  `proposals/fast_engine.py` with a bitwise test, as a separate change.

