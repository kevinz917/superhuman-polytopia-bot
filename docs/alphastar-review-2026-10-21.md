# AlphaStar through the lens of our Polytopia RL program (2026-10-21)

Source: Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning", Nature 575,
350–354 (2019), main text plus Methods. Numbers below are from the paper's Fig. 3 ablations and Methods unless
marked. The second half summarises where this project stands as of the morning of 2026-10-21 and reads the two
against each other.

## Part 1 — what AlphaStar did, with the ablation numbers that matter

The ablations run on a cut-down setting (one map, Protoss vs Protoss, 10^10 steps, main agents only, 50/50
self-play and PFSP unless stated), so they rank levers rather than give absolute effects.

### 1. Opponent observations in the value function (training only)

The critic receives both players' observations; the policy never does. Purpose stated as variance reduction.
Fig. 3k, measured as best-response win rate against fixed opponents: **22% without opponent information, 82% with
it.** This is the largest single RL-algorithm effect in the paper. It is exactly our omniscient-critic result
(the only training lever that beat c3; readout 2026-10-20, c5 6-h runs 77% vs c3).

### 2. Human data as the exploration mechanism

Fig. 3e, test Elo: **no human data 149; supervised agent alone 936; human init only 1,020; + KL toward the
supervised policy 1,400; + strategy statistic z 1,540.** Three separate mechanisms, all kept at once:

- initialise from the supervised policy (971k replays, MMR > 3,500; fine-tuned on 16k winning replays with MMR
  > 6,200, which lifted the SL agent from 87% to 96% vs the elite bot);
- a continual KL penalty toward the frozen supervised policy during RL;
- condition the policy on z (first 20 buildings/units built, plus the cumulative set of units/buildings/upgrades),
  sampled from human games, with pseudo-rewards for following it (edit distance on build order, Hamming distance
  on cumulative stats), each active in 25% of games, each with its own value head. Unconditional play (z = 0,
  10% of SL data) is what was evaluated.

The authors' argument: naive exploration cannot discover a precise thousand-step sequence that makes a new unit
type useful, so strategy modes have to be injected and protected.

### 3. League training: PFSP + exploiters

- **PFSP** samples a frozen opponent with probability proportional to f(P[A beats B]); f_hard(x) = (1 − x)^p
  focuses on opponents the learner cannot beat (and gives zero games to ones it already beats); f_var(x) = x(1 − x)
  is the curriculum for struggling agents and for exploiters.
- Fig. 3c/d: **FSP 1,143 Elo, PFSP 1,273, self-play 1,519, PFSP + self-play 1,540.** Forgetting proxy (minimum
  win rate against all past versions): FSP 69%, PFSP 70%, **self-play 46%**, PFSP + SP 71%. Plain self-play
  learns fastest but forgets; the mixture keeps most of the speed and fixes the forgetting.
- Main agents: 35% self-play, 50% PFSP over the whole league, 15% PFSP against forgotten main players and past
  main exploiters. Snapshot every 2 × 10^9 steps; never reset.
- **Main exploiters** train only against the current main agents (with an f_var curriculum when under 20% win
  rate); frozen into the league when they beat all three main agents > 70% or after 4 × 10^9 steps; then reset to
  the supervised parameters. **League exploiters** run PFSP against the whole league; added when > 70% vs
  everyone or after 2 × 10^9 steps; 25% chance of reset.
- Fig. 3a/b: **main agents only 1,540 Elo → + main exploiters 1,693 → + league exploiters 1,824**; relative
  population performance (exploitability of the whole league) **6% → 35% → 62%.** Fig. 4d: exploiters, not main
  agents, are where new unit compositions appear first.
- Diagnostics: a round-robin of all players over time, a held-out set of 17 "validation" strategies (main agents
  trained with hand-picked z), and the Nash distribution over the league to show that recent players dominate and
  the process is not cycling.

### 4. Off-policy learning in a large structured action space

Asynchronous actors + replay buffer (replayed twice) made the data off-policy. Fig. 3i, best-response win rate:
**V-trace 49% → + TD(λ) 73% → + UPGO 82%.** The hybrid: V-trace (clipped IS) for the policy, uncorrected TD(λ)
for the value (biased, lower variance); V-trace applied per action component (type, delay, arguments treated as
independent) so the trace is not truncated early; UPGO bootstraps the return with V(s_{t+1}) whenever the taken
action looks worse than average, using the one-step Q = r + V(s_{t+1}).

### 5. Reward and credit

Terminal ±1 (0 for draws), **undiscounted** ("to accurately reflect the true goal of winning games"). No shaping
on the win signal; the only pseudo-rewards are the z-following ones, each with a separate value head.

### 6. Architecture

Fig. 3f, supervised win rate vs the elite bot: **baseline 0% → + action delays 7% → + pointer network 36% →
+ transformer over units 71% → + scatter connections 87%.** Deep LSTM core for partial observability;
autoregressive action heads (type → delay → queue → selected units → target) conditioned on the LSTM and
encoders; one value head per reward. 139M weights, 55M used at inference. Components were chosen by supervised
performance, not RL.

### 7. Interface constraints and a surprising inverted U

22 non-duplicate actions per 5 s; camera-restricted view (costs 96% → 87% SL win rate). Fig. 3g: lowering the
APM limit hurts, but **raising it also hurts** (1,540 at 100%, 1,411 at 200%, 1,392 unlimited) — the agent spends
its capacity refining micro instead of learning strategy. Read: a constraint on the action economy acted as a
regulariser toward strategy.

### 8. Scale

Per training agent: 16,000 concurrent games, 16 TPU-v3 actor tasks for batched inference, a 128-core TPU learner
at ~50,000 agent steps/s, batch 512 sequences, actors refreshed every 10 s. Twelve such agents (3 main, 3 main
exploiters, 6 league exploiters) × 32 TPU-v3 × 44 days; ~900 frozen players. External anchor: blind Battle.net
play, above 99.8% of ranked players; the supervised-only agent was already at the 84th percentile.

## Part 2 — where this project stands (Oct 16 → Oct 21, 2026)

**Simulator.** JAX GPU-native Polytopia 1v1, frozen as `bardur-1v1-v1` (14×14) and `bardur-1v1-v2` (18×18, cap
150), validated against the pinned Polyfish oracle (1,977 tests CPU and L4). A CPU single-game build with real
branching (`rl/cpu_engine.py`) is bitwise-equal to the batched engine and is what actors run; the vmapped GPU engine
executes all 17 action branches per lane and is unusable for search (14.7 s/turn on an L4 vs ~1 s on 16 CPU cores).

**Language-model track (parked, Oct 17–19).** Qwen3.5-4B "thinking" is not a viable policy (2 wins in 30 games
for ~18 H100-hours). Expert SFT of Qwen3-4B on 39.6k moves of the RL bot league96 went 0 → 45% vs scripted bots
but inherited the teacher's 99.7%-Warrior play and lost 0/10 to u3142; win rate still rising with data; imitation
accuracy 36%. Compact prompt v2.1 cuts tokens 69% (one regression: self-demolition). Trainer at 75k tok/s on
4×B200 with cuDNN. Next step written up but not run: ~160k-example SFT imitating the newest RL bot.

**Structured RL bot, lineage and strength (all internal, deterministic both seats, frozen engine):**

| Stage | Date | Recipe | Result |
|---|---|---|---|
| teacher-imitation + mixed PPO → league96 | Oct 17–18 | 0.5M-param CNN (4 res blocks × 64 ch), masked autoregressive family/source/target heads, synchronous PPO, 25% frozen learned rivals | 64/64 vs scripted teacher, 64/64 vs random; 99% Warriors. Pure RL from scratch failed (5/27 vs teacher) |
| campaign 1 → u3142 | Oct 19 | Duper-style CPU actors (32-core containers, pinned processes) + 2×H100 learner, 35–50k samples/s, unit-choice KL for diversity, frozen-rival + snapshot league | 117–10–1 vs league96 on untouched final maps (91%); 2.9 unit types/game |
| campaign 3 → c3 u4854 | Oct 19 | pure self-play on the `proposed-v1` map mix | 70% vs u3142 on new maps, 67% classic; served on the public site |
| test-time search | Oct 19 | d=1 lookahead over whole turns, K=16, policy-played reply, critic at next own turn; leak-free via determinised fog; parallel planner ~1 s/turn | +125–141 Elo leaky; **+56–104 leak-free** (about half the leaky gain was hidden information); robust (64 replies) +117 on the mix |
| c4: 18×18 | Oct 19–20 | grown c3, three self-play runs, turn-clock pilot | all failed to beat grown c3 (best 47% of decisive). Diagnosis: per-decision γ 0.995 over ~1,500 decisions flattened the critic and pushed advantages under the normalisation floor; turn-clock discounting fixed the critic but the policy stayed at parity; round-robin transitive, no cycling |
| c4: search distillation | Oct 19–20 | 23.4M hard labels from c3 + search | **null**: 7 students within ±3 Elo; students lose most of the search gain (+40 vs +104). Frozen-critic refit (Oct 21) ruled out the critic: the loss is policy-side |
| c4: $200 ablation (five 1-h arms from c3) | Oct 20 | omniscient critic / PBRS / cap-score / top-25% advantage filter + 1 epoch + EMA / plain continuation | **only the omniscient critic won** (54% of all, 65% of decisive vs c3, on 30% fewer samples); PBRS hurt (37%); the rest neutral; plain continuation 40%. TF32 convs pass the parity gate (1.9×/3.3× learner speed) |
| c5: omni 6 h × 2 seeds | Oct 20–21 | abl-omni-2 recipe, seeds 201/202, 1×H100 + 12×16-vCPU | **both 77.3% of all games vs c3 [72.7, 82.0], +255 Elo**; seeds within 2–3 points throughout; curve still rising at 6 h (82–87% of decisive sampled); Warrior share 39% → 20% |
| c5: omni + leak-free d=1 search | Oct 21 | own stale value head as scorer | **+338 to +354 Elo vs c3**; search gain (+81–100) undiminished by the stronger policy; scorer choice irrelevant |
| c5: UPGO | Oct 20–21 | upgo_coef 1.0 on the omni recipe, 3 h | neutral (+196 vs +185 over c3; head-to-head +10 [−33, +52]) |
| c5: 18×18 omni + turn clock | Oct 20–21 | grown c3, γ 0.97 / λ 0.92 at end-turn only, omni critic, 4×H100 + 24×8-vCPU | **passed**: 159–49–48 = 62% of all / 76% of decisive vs grown c3; first 18×18 bot to beat its parent; Cloak 37% of recruits |

Strength chain: hand heuristic → scripted teacher → league96 → u3142 (+410) → c3 (+150–290) → omni-6h (+255)
→ + search (+80–100). Chained that is well over 1,000 Elo, but every rung is our own lineage; no human,
independent bot, or exploiter has played any of these checkpoints.

**Cost and operations.** Roughly $1.6–1.8k of Modal spend across campaigns 1–5 and the search studies (campaign 1
~$345, c4 ~$560, c5 ~$457, search/distillation studies ~$200, smaller pilots). A 1×H100 + 12 actor arm bills
~$22/h. Preempted containers are replayed (an orchestrator replay once spawned a duplicate learner and fleet);
non-detached `modal run` evaluations die with the laptop; the `runs/` ledger is a shared git-excluded symlink.

**What has been falsified** (readout and next-experiments report): scoring-time explanation of d=0 failure; "the
search gain is mostly real skill"; one-round hard-label distillation; c3 being tier-3 tech-starved; 18×18 as a
drop-in; credit assignment alone restoring 18×18 learning; 18×18 cycling (so PFSP as a plateau fix); potential
shaping as safe; TF32 breaking parity; GPUs for search.

**Open as of this morning.** Nothing from c5 is merged (four `claude/c5-*` branches, nothing pushed to main);
the ship candidate for 14×14 is `c5-omni-s1-final` + leak-free d=1 search; the 18×18 candidate is
`c5-l18-omni-final`. Dropped from c5 for budget: a 3-h control, a PFSP pool, the exploiter run.

## Part 3 — reading the two against each other

### Where we already match AlphaStar, and the paper says why it worked

1. **The omniscient critic is their biggest RL lever and ours.** 22% → 82% for them; 40% → 54% in one hour and
   77% at six hours for us. Their framing (variance reduction, training only, the policy stays blind) matches our
   form exactly: a fresh critic on the union of both seats' views, policy input unchanged. The lesson is not to
   touch this; the 6-h curve has not flattened, so a longer run is the plain next lever.
2. **Undiscounted terminal reward, no shaping.** AlphaStar uses γ = 1 on ±1, and Generals.io did the same. Our
   18×18 collapse was a per-decision γ = 0.995 over 1,500 decisions; the turn clock (γ 0.97 at end-turn only) is
   our workaround. PBRS hurt us, and AlphaStar never shaped the win signal either. An arm with γ = 1 and a
   lower λ on the turn clock is cheap and literature-aligned.
3. **Mixed self-play + frozen pool.** Our pools (50–75% self-play, 10–20% own snapshots, 5–15% frozen rivals) are
   the "SP + FSP" mixture that scored best in their Fig. 3c/d. Their data also explain why pure self-play looked
   fine for us: it has the highest single-method Elo; its cost is forgetting, which we have never measured.
4. **UPGO neutral is consistent, not contradictory.** UPGO's 73% → 82% sits on top of V-trace in an
   asynchronous replay setting with heavy policy lag. We run near-on-policy PPO (max lag 6–10 updates, samples
   beyond that dropped), so the off-policy problem UPGO/V-trace address barely exists. Do not chase V-trace either
   unless actor lag grows.
5. **Supervised bootstrap, then RL.** league96 came from teacher imitation + PPO, pure RL failed (5/27). AlphaStar's
   "no human data" arm sits at 149 Elo against 1,540. Same shape.

### Where AlphaStar has something we have not used

6. **Exploiters, and exploitability as a first-class metric.** +284 Elo and 6% → 62% population robustness came
   from main and league exploiters, not from more main-agent training. We have never trained a best response
   against any of our checkpoints; the $45 exploiter run was dropped from c5. It is doubly valuable here: it is
   AlphaStar's own method for measuring the critic effect cleanly (best response vs a fixed opponent, no
   multi-agent drift), and it answers the exploitability question the strength doc raises. Their reference
   threshold is > 70% for "clearly exploitable". If omni-s1 is exploitable, the fix is AlphaStar's 15% slot:
   add the exploiter's frozen checkpoints to the main agent's pool and keep training.
7. **PFSP weighting instead of uniform history.** Our snapshot pool is sampled uniformly. f_hard = (1 − x)^p over
   the pool, with the win-rate table the ladder already produces, is a small change. The 18×18 diagnosis found no
   cycling, so the expected gain is forgetting-protection rather than a plateau fix; their numbers (46% vs 71%
   minimum win rate against past versions) say that protection is real. Cheap enough to ride on the next run.
8. **Strategy conditioning (z) as the answer to narrow unit use.** Our recurring weakness is strategic narrowness:
   99% Warriors in league96, Catapult/Warrior in u3142, 37% Cloaks on 18×18. AlphaStar's cure was not an entropy
   bonus but a sampled strategy target with a pseudo-reward and its own value head, active in a quarter of games,
   plus a KL to a diverse prior. We lack human replays, but we have a diverse checkpoint history (E3, diverse u176,
   the 18×18 Cloak bot). A self-generated z (first-N recruit/tech order, cumulative tech/unit set) sampled from
   those games, conditioned on and rewarded 25% of the time, evaluated unconditionally, is the direct analogue.
   The unit-KL term we did use is the weaker, label-free cousin of their supervised KL; it held diversity in
   training but not deterministic play, which is the same gap AlphaStar closed with z.
9. **Exploiters are also the diversity engine.** Fig. 4d shows new compositions appearing in exploiters first.
   Even one periodically reset exploiter would seed the pool with styles the main line never explores.
10. **Validation strategies and a Nash check.** They evaluated against 17 held-out hand-curated strategies and
    computed the league Nash to show no regression. We have the infrastructure (round-robin ladder, Bradley–Terry
    fit) and the frozen roster (teacher, league96, u3142, E3, diverse u176, c3, omni, 18×18 bot). A fixed
    validation roster with per-opponent win rates reported every run is the cheapest robustness signal we are not
    collecting, and it is the only thing that would catch forgetting.
11. **Architecture scale and components.** 139M weights vs our 0.5M. Their components were selected by supervised
    accuracy, where each step was large (0% → 87%). We have an SFT pipeline and imitation-accuracy proxy on the
    LLM side but nothing comparable for the structured net; a supervised-accuracy screen of architecture changes
    (global pooling, wider trunk, per-unit attention) against c3/omni labels would be a cheap way to pick
    capacity changes before paying for RL. Our one capacity experiment (global pooling) is still unrun.

### Where the analogy breaks

12. **Fog is different here.** AlphaStar's LSTM exists because StarCraft re-hides seen units. In our engine
    `visible = explored`: once a tile is explored it stays visible, enemy units on explored tiles are always
    shown, and the hidden information is only unexplored tiles, Cloaks, and the opponent's stars/tech. A
    recurrent core is therefore low priority for us; the value-function fix already covers the opponent's bank
    and tech, and the belief-prior search upgrade covers the unexplored map at inference.
13. **Compute.** Per learner, our 30–50k samples/s is the same order as their 50k agent steps/s. The difference is
    12 learners × 44 days versus 1 learner × 6 hours, about three orders of magnitude. A full league is out of
    reach; one main exploiter at a time, run in series, is the affordable slice.
14. **Search.** AlphaStar is deliberately model-free; our best bot is policy + one-ply search, and search gain is
    undiminished at every strength we have reached. That is evidence we are still far from the ceiling AlphaStar's
    budget bought, and it is the one lever AlphaStar offers no guidance on.
15. **External anchor.** They proved strength on Battle.net. We have no external anchor at all; every number
    above is relative to our own lineage. Nothing in the AlphaStar toolkit substitutes for a human evaluation.

### What the paper would have us do next, in order of evidence per dollar

| # | Action | AlphaStar basis | Our status | Cost |
|---|---|---|---|---|
| 1 | Longer omniscient-critic run from omni-s1 (12 h+) | critic ablation; our curve still rising | ready | ~$22/h |
| 2 | Main-exploiter run vs frozen omni-s1 (symmetric and omni critic) | Fig. 3a/b; Fig. 3k measurement method | designed, dropped from c5 | ~$50 |
| 3 | PFSP f_hard over the snapshot pool, forgetting metric on a fixed validation roster | Fig. 3c/d | not implemented | small code change, rides on #1 |
| 4 | γ = 1 / lower-λ arm on the 18×18 turn clock | undiscounted ±1 | not tested | ~$35/h × 2 h |
| 5 | Self-generated strategy statistic z + pseudo-reward, 25% of games | Fig. 3e (+140 of 1,540) | not implemented | engineering + 3-h arm |
| 6 | Supervised-accuracy screen for architecture changes before RL | Fig. 3f method | SFT proxy exists for LLM only | cheap |
| — | Not worth repeating: UPGO, V-trace, reward shaping on the win signal, a full multi-agent league | tested neutral / off-policy only / never used by them / unaffordable | | |
