# Proven levers for self-play RL in long-horizon turn-based strategy games (4X / Generals.io / Freeciv / Polytopia-like)

Scope note: evidence collected 2026-10-20 from primary papers (arXiv/ar5iv full text where available), project READMEs/docs and one author blog. Peer-reviewed vs. preprint vs. blog status is marked per item. "Our setup" below means the Polytopia bot team's: 4×H100 learner, CPU actor fleet, ~35–50k samples/s, ~0.5M-parameter CNN, a few hundred dollars per run on Modal.

---

## Q1. Reward shaping for finishing games: per-step penalties, potential-based shaping, discounting, draws/truncation — what did OpenAI Five, AlphaStar, Generals.io and Freeciv projects do?

### Takeaway
The strongest recent evidence (Generals.io, superhuman agent, 2026) says: keep the reward sparse win/loss, use γ = 1.0 (undiscounted), truncate at a hard turn cap with reward 0 to both sides, and bootstrap the win signal with a *difficulty curriculum* (spawn distance) rather than with shaping; shaping buys an early lead (~+150 Elo at 24 h) that shrinks with training (~+90 Elo at 48 h). The explicit, authored rationale against discounting in a zero-sum game is that a discounted agent "stalls when losing and rushes to bank wins when ahead." Where shaping was used (OpenAI Five, Generals.io v1, μRTS, Lux AI), the recurring patterns are: (a) make non-terminal rewards decay with game time (OpenAI Five's 0.6^(T/10 min)), (b) anneal shaping to zero or use it only to bootstrap a small net, and (c) expect agents to overfit the shaped reward instead of the sparse one.

### Cited Findings

**Generals.io, superhuman self-play agent (Straka, Lisý & Schmid 2026, arXiv preprint; ICML 2026 listing)**
- Reward is "sparse and terminal": "Our agent is trained on the sparse win/loss reward alone and reaches superhuman play." No time penalty. — [arXiv 2606.23348](https://arxiv.org/abs/2606.23348)
- Draw/truncation rule: "Games that reach 2,048 turns are truncated with reward 0 to both players and counted as draws in evaluation." Self-play games are "about 730 turns on average." — [arXiv 2606.23348](https://arxiv.org/html/2606.23348v2)
- Discount and GAE: γ = 1.0, GAE λ = 0.9 (Table S.III); PPO clip 0.2, minibatch 1024, 128 optimizer steps per iteration, 1 epoch, value-loss coefficient 0.5, grad-norm clip 0.267; categorical HL-Gauss value loss; EMA of parameters (τ = 0.999) is the deployed policy. — [arXiv 2606.23348](https://arxiv.org/html/2606.23348v2)
- Why γ = 1.0 (author blog, not peer-reviewed): "a discounted agent stalls when losing and rushes to bank wins when ahead." λ = 0.9 beat 0.7 after ~70 h; "the best λ depends on training length." A value-loss coefficient of 1.0 "completely broke training"; 0.5 was kept. — [Straka blog](https://kam.mff.cuni.cz/~straka/blog/generals.html)
- Shaping ablation: "shaped runs led by about 150 Elo at 24 hours, but the lead shrank to about 90 Elo at 48 hours"; the authors hypothesize the gap narrows further with more frames and argue the sparse reward "needs no hand-designed game knowledge, even at this episode length." — [arXiv 2606.23348](https://arxiv.org/html/2606.23348v2)
- Bootstrapping the sparse signal is done with a *spawn-distance curriculum* (BFS distance windows 2–6, 4–8, 6–13, 11–17, 17–28 cells); a stage advances at ≥60% wins vs. a random-move opponent over 512 games. "Without the curriculum, the win signal is too sparse for a random policy to bootstrap." — [arXiv 2606.23348](https://arxiv.org/html/2606.23348v2)
- Advantage filtering: keeping the top 25% of transitions by |advantage| "completes twice as many iterations" in 24 h; filtered runs ended 150–250 Elo higher at equal wall-clock but were level at equal frames. — [arXiv 2606.23348](https://arxiv.org/html/2606.23348v2)
- Two 25%-filter seeds were "still on the early plateau" at ~4,400 iterations, leaving it after ~16 and ~20 h — i.e., multi-hour early plateaus occurred even in the final recipe. — [arXiv 2606.23348](https://arxiv.org/html/2606.23348v2)

**Generals.io v1 agent "zero v3" (Straka & Schmid 2025, arXiv preprint; Master's thesis)**
- Default reward +1 win / −1 loss plus potential-based shaping φ(s) = 0.3·φ_land + 0.3·φ_army + 0.4·φ_castle, each term a normalized log-ratio of agent vs. enemy values. — [arXiv 2507.06825](https://arxiv.org/html/2507.06825v2)
- Effect: "pure self-play converged to an overly aggressive, exploitable strategy"; with shaping the agent "focused on material advantage and avoided unnecessary risk." Table 2 (2,000 games each): shaped vs. naive self-play 71.9% ± 2.1%; population-trained vs. self-play 78.4% ± 1.8%; population vs. shaped 70.2% ± 2.0%; self-play vs. behavior cloning 68.8% ± 2.0%. — [arXiv 2507.06825](https://arxiv.org/html/2507.06825v2)
- Author's blog assessment of v1: shaping "was judged to be most of the climb, not the pool"; failure modes included "behavior collapse into all-in castle capture, neglecting defense" and convergence to "an aggressive style, a local optimum that compute and exploration were insufficient to escape." — [Straka blog](https://kam.mff.cuni.cz/~straka/blog/generals.html)
- Training data hygiene mattered for stalling: idle replays were removed because otherwise the pretrained agent "learned to do nothing"; games over 500 turns were filtered. — [Straka blog](https://kam.mff.cuni.cz/~straka/blog/generals.html); [arXiv 2507.06825](https://arxiv.org/html/2507.06825v2)
- Self-play opponents in v1 were past checkpoints acting *greedily* (argmax); "stochastic opponents led agents to exploit noise rather than play robustly." — [arXiv 2507.06825](https://arxiv.org/html/2507.06825v2)

**OpenAI Five (Berner et al. 2019, arXiv preprint; not peer-reviewed)**
- Shaped reward weights (Table 5): Win = 5 (team); Hero Death = −1; Courier Death = −2 (team); XP Gained = 0.002; Gold Gained = 0.006; Health Changed = 2; Killed Hero = −0.6; Last Hit = −0.16. Rewards are symmetrized by subtracting the opposing team's reward. — [ar5iv 1912.06680](https://ar5iv.labs.arxiv.org/html/1912.06680)
- Time decay on shaping: every non-win/loss reward is scaled ρ_i ← ρ_i × 0.6^(T/10 min), T = time since game start (Eq. 11). Rationale: without it "the learning procedure focuses entirely on the later stages of the game" because late-game rewards have much larger magnitude. — [ar5iv 1912.06680](https://ar5iv.labs.arxiv.org/html/1912.06680)
- Team spirit schedule: 0.3 (baseline) → 0.8 (Rerun) → 1.0 (OpenAI Five). — [ar5iv 1912.06680](https://ar5iv.labs.arxiv.org/html/1912.06680)
- Horizon (H = T/(1−γ), T = 0.133 s/step): baseline 180 s (γ = 0.9993); OpenAI Five moved horizon from 60 s to 840 s over training; Rerun 180 s → 360 s. A sparse-reward run used a 1-hour horizon (γ = 0.99996). In resume-training experiments "longer horizons helped up to 6–12 minutes, with diminishing returns at high horizons" (Sec. 4.5). Games are ~45 min ≈ 20,000 steps; early agents produced "long, drawn-out losses over 45 minutes." — [ar5iv 1912.06680](https://ar5iv.labs.arxiv.org/html/1912.06680)
- Opponent mix: 80% games vs. latest parameters, 20% vs. past versions; past versions sampled by softmax over quality scores q_i; current agent added to pool every 10 iterations with q = max existing; on beating a past opponent q_i decreases by η/(N·p_i), η = 0.01; purpose is to avoid "strategy collapse in which the agent forgets how to play against a wide variety of opponents." — [ar5iv 1912.06680](https://ar5iv.labs.arxiv.org/html/1912.06680)

**Gym-μRTS (Huang et al. 2021, IEEE CoG; peer-reviewed)**
- Shaped reward: "+10 for winning, 0 for drawing, −10 for losing" plus +1 per harvested resource, +1 per worker, +0.2 per building, +1 per valid attack, +4 per combat unit; weights "picked by hand with very little tuning." Episodes truncated at 2,000 ticks in training (4,000 in evaluation). — [ar5iv 2105.13807](https://ar5iv.labs.arxiv.org/html/2105.13807)
- Failure mode: "agents sometimes overfit to the shaped rewards instead of sparse rewards"; "little difference in the shaped return when the sparse (win/loss) return could be drastically different." Plain self-play variants "do not perform particularly well," attributed to the need for league-style training. — [ar5iv 2105.13807](https://ar5iv.labs.arxiv.org/html/2105.13807)

**Lux AI 2021 winner, Toad Brigade (Kaggle write-up/GitHub; not peer-reviewed)**
- Shaped reward (cities/units built or lost, research, fuel) for the first 20M steps only, used to train an 8-block net; 16- and 24-block nets then trained on sparse ±1 win/loss with the smaller net as a KL teacher. Sparse reward: −1 loss, +1 win, 0 otherwise. A frozen-teacher KL loss was used "to stabilize behavior and prevent strategic cycles." Game length fixed at 360 turns. — [Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021)

**Bansal et al. 2018 "Emergent Complexity via Multi-Agent Competition" (ICLR 2018; peer-reviewed)**
- Exploration curriculum r_t = α_t·s_t + (1−α_t)·I[t=T]·R with α_t linearly annealed to 0 in 500 iterations (1,000 for Kick-and-Defend); dense reward used for roughly 10–15% of training. Without it, Sumo agents "learned to stand and move toward the center" and the defender "learned to stand without defending." — [ar5iv 1710.03748](https://ar5iv.labs.arxiv.org/html/1710.03748)

**KataGo (Wu 2019, arXiv preprint; widely replicated)**
- Score as a tie-break/auxiliary utility: score-utility weight c_score started at 0.5 and was lowered to 0.4 after two days; the value head includes a "no result" outcome for long-cycle draws; "a tiny bias is added to favor passing when passing and continuing play would lead to identical scores"; "games are played to completion without resignation," but when the losing side's winrate stays below 5% for 5 turns, visit caps shrink and positions are downweighted. — [ar5iv 1902.10565](https://ar5iv.labs.arxiv.org/html/1902.10565)

**Potential-based shaping theory**
- Devlin & Kudenko, AAMAS 2012 (peer-reviewed): extends PBRS to *dynamic* potentials while keeping single-agent policy invariance and, in the multi-agent case, consistent Nash equilibria; the 2011 AAMAS result shows static PBRS does not modify the Nash equilibria of the underlying stochastic game. — [White Rose eprints 75121](https://eprints.whiterose.ac.uk/75121/2/p433_devlin.pdf); [AAMAS 2011](https://eprints.whiterose.ac.uk/id/eprint/75111/1/aamas11.pdf)
- A 2026 ESANN paper integrates PBRS into AlphaZero (Othello only so far), motivated by "dependence on sparse terminal rewards limits learning efficiency." — [ESANN 2026-104](https://www.esann.org/sites/default/files/proceedings/2026/ES2026-104.pdf)

**CivRealm / Freeciv (Qi et al. 2024, ICLR 2024; peer-reviewed)**
- Full-game reward is the Freeciv *score*, "a multi-facet metric calculated from a weighted sum of important game stats" (population, techs, units built/killed). Result: RL agent's scores rose but by "primarily focusing on unit production to enhance its score," which the authors call myopic; "both RL- and LLM-based agents struggle to make substantial progress in the full game." Stated challenges: "Rewards are sparse, delayed, and asynchronous"; "greedy moves can easily be non-optimal in the long term." ~2M steps for full game. — [arXiv 2401.10568](https://arxiv.org/html/2401.10568v2)

### Inferences
- For our zero-sum 1v1 setting the literature-consistent default is: terminal ±1, γ = 1.0 (or ≥0.999 if variance forces it), hard turn cap with 0/0 draw, GAE λ ≈ 0.9, and bootstrap via curriculum rather than shaping. The Generals.io 2026 recipe is the closest published analogue to our hardware class (4 GPUs, 7 days) and reports no time penalty.
- A per-step penalty is *not* potential-based unless written as a time-indexed potential; the safe forms are (i) PBRS with a material/territory potential (Generals v1, policy-invariant in theory per Devlin & Kudenko), (ii) a time-decayed shaping multiplier (OpenAI Five), or (iii) KataGo-style score as a small secondary utility (c_score 0.4–0.5) which acts as a tie-break that pushes toward finishing without changing who wins.
- The documented failure modes of shaping are: agents optimizing the shaped proxy while sparse win-rate diverges (μRTS), aggressive local optima (Generals v1), and "do nothing" behavior learned from idle data (Generals v1 BC). The documented failure mode of discounting is stall-when-losing / rush-when-winning (Straka blog).
- If draws are a large fraction of games, CivRealm's experience says a score-based reward will be gamed toward the cheapest score source (unit production). Use score only as a tie-break with small weight, or as an auxiliary head (KataGo ownership/score heads gave the single largest ablation gain, see Q4).

### Gaps
- No paper found that directly A/B tests per-step time penalties vs. none in a zero-sum self-play strategy game; the Straka blog's γ = 1.0 rationale is the only explicit, authored statement and is not peer-reviewed.
- AlphaStar's draw/time-limit handling could not be read (Nature paywall).
- CivRealm's full-game numeric scores were not in the fetched text (figures only).

---

## Q2. Scaling the board/horizon: map-size curricula, architectures that generalize across sizes, transfer results

### Takeaway
Fully-convolutional / size-agnostic policies trained on mixed sizes transfer upward in practice (KataGo trains on 9×9–19×19 simultaneously; Lux AI winner pads 12–32 maps to 32×32 with masking; the GNN "Train on Small, Play the Large" beat an AlphaZero trained ~10× longer on the large board), but models with size-specific dense layers or purely local receptive fields degrade most as boards grow. Compute needed for a fixed strength level grows steeply with board size (Jones 2021: ~7× per Hex board-size increment for perfect play, ~4× to beat random), so a small→large curriculum is mostly about *bootstrapping the win signal* (Generals spawn-distance curriculum) and *not* a free lunch on compute.

### Cited Findings
- KataGo trains one net on mixed sizes: "37.5% of games on 19x19" early, rising to "50% of games after two days of training," the rest spread across 9×9–18×18 weighted toward larger boards. Global pooling (which lets a conv net aggregate board-wide context) was worth 1,329 → 1,153 Elo when removed (1.60× training time). — [ar5iv 1902.10565](https://ar5iv.labs.arxiv.org/html/1902.10565)
- Ben-Assayag & El-Yaniv 2021 "Train on Small, Play the Large" (arXiv preprint): GNN-AlphaZero trained 3 days on Othello 5×5–8×8 scored 0.54 ± 0.02 on 16×16 and 0.84 ± 0.01 on 20×20 against AlphaZero trained on the tested board "about 10× longer"; Gomoku 17×17/19×19: 1.00. Board-size sampling during training: {n, n−1, n−2, n−3} with probabilities 0.4/0.3/0.2/0.1. A small CNN baseline scored only 0.10/0.33 (Othello 16/20) and "declined most as boards grew, suggesting global board relations matter." Go: ~68% vs. AZ on 9×9 (AZ trained 20 days), ~77.5% on 15×15 (AZ trained 10 days). Removing rotation/reflection augmentation "sped up training about 5×." — [ar5iv 2107.08387](https://ar5iv.labs.arxiv.org/html/2107.08387)
- AlphaViT (2024 preprint): standard AlphaZero "cannot function properly even with small variations in board size" because of fixed-size dense layers; fine-tuning from small-board pre-trained weights "accelerates convergence and improves performance, particularly in Gomoku." — [arXiv 2408.13871](https://arxiv.org/pdf/2408.13871)
- Jones 2021 (arXiv preprint, Hex 3×3–9×9 with AlphaZero): "The minimum compute needed for perfect play increases 7× for each increment in board size"; minimum compute to beat random "increases by 4× for each increment of board size"; "the distance between random and perfect play increases by 500 Elo for each increment of board size." Fitted frontier error "starts small and decays exponentially as more small boards are added," i.e., small-board runs predict large-board frontiers. — [ar5iv 2104.03113](https://ar5iv.labs.arxiv.org/html/2104.03113)
- Lux AI 2021 winner (hobby, non-peer-reviewed): fully convolutional ResNet, 24 blocks, 5×5 convs, 128 channels, ~20M params; "All boards are zero-padded to 32x32," outputs "masked after every convolution so information doesn't leak through the padded edges," plus a 4-way discrete board-size feature and a "distance from center" feature. Trained overnight on a personal dual-GPU PC with IMPALA + UPGO + TD(λ). Curriculum was in *network size* (8 → 16 → 24 blocks with KL to the previous net), not map size. — [Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021)
- Generals.io 2026: training maps 17–23 cells per side padded to 24×24; horizon handled by a *spawn-distance* curriculum (2–6 → 17–28 BFS steps) rather than a map-size curriculum. — [arXiv 2606.23348](https://arxiv.org/html/2606.23348v2)
- OpenAI Five grew the *reward horizon* over training (60 s → 840 s) as an explicit curriculum; "longer horizons helped up to 6–12 minutes, with diminishing returns." — [ar5iv 1912.06680](https://ar5iv.labs.arxiv.org/html/1912.06680)
- XLand (DeepMind 2021, preprint): dynamic task generation keeps tasks where the agent "has a low probability of scoring high" yet beats a random control by a margin, with thresholds (e.g., m_cont = 5, m_s = 450, m_solved = 0.1) themselves made dynamic "so training doesn't stall"; generational training distills from the best agent of the previous generation and adds it to the co-player pool. — [ar5iv 2107.12808](https://ar5iv.labs.arxiv.org/html/2107.12808)
- CivRealm: mini-games use "a smaller map size and fewer players" than the full game; RL solved CityTileWonder at ~90% but "most other mini-tasks remained harder" and full-game progress was minimal. — [arXiv 2401.10568](https://arxiv.org/html/2401.10568v2)

### Inferences
- A 0.5M-param CNN with only local receptive fields is the architecture class that transferred worst in Ben-Assayag's ablation; adding global pooling (KataGo: +176 Elo / 1.6× speed) or a few attention/global tokens is the cheapest architectural change before scaling maps.
- Train on a mixture of map sizes from the start (KataGo/Ben-Assayag style, weighted toward the target size) rather than staged small→large; with padding+masking (Lux) one net serves all sizes and the per-size frontier can be measured in one run.
- Expect large-map strength per dollar to fall sharply (Jones: 4–7× per size step in Hex). Use small maps to *predict* the large-map compute frontier (Jones) and to bootstrap, not as a substitute.
- A difficulty curriculum on *distance between starting positions* (Generals) is a direct analogue to Polytopia capital spacing and is the lever that converts a random policy into one that finishes games.

### Gaps
- No controlled study of a staged map-size curriculum (train small, then move to large) vs. mixed-size training for a strategy game was found; Ben-Assayag uses mixed sampling, Lux uses padding, KataGo uses mixing.
- AlphaViT numbers beyond the abstract were not fetched.

---

## Q3. Self-play plateaus: diagnostics (Nash averaging, spinning tops) and league remedies (PFSP, exploiters, past-self pools)

### Takeaway
Diagnostics: compute the round-robin logit matrix among checkpoints and decompose it (Hodge): if the cyclic component ("curl", triangle log-odds sums) is near zero and Elo vs. a *fixed* reference stops rising, you have a transitive plateau (no learning signal); if curl is large and Elo vs. recent checkpoints oscillates, you are cycling. Czarnecki's spinning-top theory predicts cycling is widest at *mid* skill and that a population at least as large as the current "layer" is required; fixed-memory fictitious play converges once the population covers it. Remedies with numbers: OpenAI Five's 80/20 past-pool with quality-weighted sampling, Bansal's δ-window sampling (δ=0.5 raised Humanoid win rate from 0.25 to 0.63), Generals v1's small 3-checkpoint pool (+78% vs. plain self-play), and AlphaStar's PFSP (outperforms FSP "under all measures") + exploiters (whose matrix contained ~3,000,000 RPS cycles vs. ~200 among main agents alone).

### Cited Findings

**Diagnostics**
- Balduzzi et al. 2018 "Re-evaluating evaluation" (NeurIPS 2018; peer-reviewed): "Elo bakes-in the assumption that relative skill is transitive"; "Elo is meaningless – it has no predictive power – in cyclic games like rock-paper-scissors." Any antisymmetric logit matrix splits into a transitive part (gradient of a rating vector = row averages) and a cyclic part; Elo reproduces win probabilities exactly only when the curl is zero, i.e., "the sum of log-odds around every triangle of players must vanish." Nash averaging = payoff against the max-entropy Nash of the meta-game; invariant to adding redundant agents. On Go agents, Elo's Frobenius error 0.85 / logistic loss 1.41 vs. mElo2 0.35 / 1.27. — [ar5iv 1806.02643](https://ar5iv.labs.arxiv.org/html/1806.02643)
- Czarnecki et al. 2020 "Real World Games Look Like Spinning Tops" (NeurIPS 2020; peer-reviewed): non-transitivity "first rapidly increases, and then, after reaching a potentially very large quantity... quickly reduces as we approach the strongest strategies." Proposition 3: fixed-memory fictitious play converges in layered Games of Skill if the population is at least the size of the lowest layer it occupies. Theorem 3: if the population contains a full Nash cluster, training against it guarantees transitive improvement. For small populations "training cycles in all games"; above a critical size it "converges to the best strategies in almost all experiments." Recommendation: "require each new agent to beat the population while being as weak as possible"; strong priors (search, imitation, architecture) "start agents higher on the top, reducing the multi-agent machinery needed." — [ar5iv 2004.09468](https://ar5iv.labs.arxiv.org/html/2004.09468)
- Balduzzi et al. 2019 "Open-ended learning in symmetric zero-sum games" (ICML 2019; peer-reviewed): self-play "cycles through agents without improving overall agent strength" in nontransitive games because it "assumes transitivity"; in transitive/monotonic games self-play is "well-suited." PSRO_rN trains each Nash-supported agent "against the Nash-weighted mixture of agents that it beats or ties" and "outperforms the other approaches across a wide range of allowed compute budgets" on Blotto/differentiable Lotto; "self-play performs the worst" in those non-transitive toys. — [ar5iv 1901.08106](https://ar5iv.labs.arxiv.org/html/1901.08106)

**Remedies with numbers**
- OpenAI Five: 80% latest / 20% past, softmax over quality scores, η = 0.01 decrement on wins; motivation: prevent "strategy collapse." — [ar5iv 1912.06680](https://ar5iv.labs.arxiv.org/html/1912.06680)
- Bansal 2018: opponents sampled Uniform(δv, v). Humanoid Sumo E[Win]/E[Loss]: δ=1.0 (latest only) 0.25/0.53; δ=0.8 0.40/0.40; δ=0.5 **0.63/0.17**; δ=0.0 0.35/0.54. Ant Sumo: δ=0.0 best (0.50/0.32). Latest-only "caused imbalance: one agent became much stronger early and the other is unable to recover"; long training vs. one opponent gave "win-rate oscillations." — [ar5iv 1710.03748](https://ar5iv.labs.arxiv.org/html/1710.03748)
- Generals v1: pool of N = 3 greedy past checkpoints; candidate replaces the oldest after ≥45% win rate vs. pool (blog says 60%); population-trained beat plain self-play 78.4% ± 1.8%. — [arXiv 2507.06825](https://arxiv.org/html/2507.06825v2); [Straka blog](https://kam.mff.cuni.cz/~straka/blog/generals.html)
- Generals 2026 dropped the pool entirely: "In plain self-play the pool contains only one policy, the current one"; opponent is the current iterate; the EMA policy was ahead of the last iterate "by 65 Elo on average" from 4,000 iterations on (32 Elo at 39,000). — [arXiv 2606.23348](https://arxiv.org/html/2606.23348v2)
- AlphaStar (Nature 2019; peer-reviewed; only captions/abstract accessible): "PFSP-based training outperforms FSP under all measures considered" (population strength, exploitability, final agent performance; Extended Data Fig. 5). League payoff matrix contained "around 3,000,000 rock–paper–scissor cycles involving at least one exploiter, and around 200 involving only main agents" (≥70% win-rate threshold; Extended Data Fig. 8). — [Nature s41586-019-1724-z](https://www.nature.com/articles/s41586-019-1724-z)
- AlphaStar reset rules as restated by TStarBot-X (Tencent 2020, preprint): main and league exploiters reset to the supervised model at a max step count or "a win-rate above 70%"; main exploiter resets "every 24 hours approximately (4×10^9 steps)"; league exploiter resets with 25% probability at 2×10^9 steps. TStarBot-X's own main agent used "25% Self-Play, 60% PFSP and 15% of PFSP matches against forgotten main players, similar to AlphaStar." Compute: 144 V100 + 13,440 CPU cores; 583 agents generated. — [ar5iv 2011.13729](https://ar5iv.labs.arxiv.org/html/2011.13729)
- Lux AI winner used a frozen-teacher KL loss "to stabilize behavior and prevent strategic cycles." — [Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021)
- Survey (2024): self-play methods taxonomy (vanilla, PFSP, PSRO, league). — [arXiv 2408.01072](https://arxiv.org/html/2408.01072v3)

### Inferences
- Minimal plateau diagnostic for our Elo ladder: (1) round-robin the last K checkpoints + fixed anchors, (2) compute per-triangle log-odds sums (curl) and the Nash support; (3) "no signal" = Elo vs. anchors flat AND curl ≈ 0 AND policy entropy/advantage magnitude collapsing; "cycling" = Elo vs. anchors flat but curl large and Nash support spread across several recent checkpoints. The Generals 2026 "early plateau" lasting 16–20 h before take-off warns against declaring a plateau too early.
- Czarnecki implies the needed population is largest at mid-skill; a cheap tiered recipe is: plain self-play + EMA (Generals 2026) first; add a small quality-weighted past pool (OpenAI Five 20%, or Bansal δ≈0.5) if curl grows; add a main-exploiter only if a specific checkpoint is exploitable at ≥70% by a fresh learner (AlphaStar reset criterion).
- Greedy (argmax) past opponents (Generals v1) are a cheap fix if the learner starts exploiting opponent sampling noise.

### Gaps
- AlphaStar's exact main-agent split (commonly cited as 35% self-play / 50% PFSP vs. league / 15% PFSP vs. forgotten) and its ablation Elo numbers could not be verified from an accessible primary source this session; TStarBot-X's 25/60/15 is the only fetched numeric restatement and is their own setting.
- No paper found that gives an *operational threshold* on the Hodge cyclic component that separates "stalled" from "cycling"; Balduzzi provides the decomposition, not a cutoff.

---

## Q4. Network size vs. strength at fixed compute on small boards (Jones 2021, Neumann & Gros 2022, KataGo)

### Takeaway
On small boards strength scales smoothly: Jones finds ~500 Elo per 10× compute and that a 500k-parameter fully-connected net suffices for perfect 9×9 Hex; Neumann & Gros find Elo ∝ N^0.88 in parameters and compute-optimal size N_opt ∝ C^0.62, i.e., bigger budgets should buy bigger nets, roughly 4× params per 10× compute. KataGo's run shows large per-size jumps at fixed visits (b10→b15 ≈ +800 Elo, b15→b20 ≈ +400 Elo) and switched sizes at 0.75 / 1.75 / 7.5 days; its biggest single-ablation wins were auxiliary ownership/score targets (1.65×) and global pooling (1.60×).

### Cited Findings
- Jones 2021 (preprint): Hex 3×3–9×9; "The slope of the incline is 500 Elo per order of magnitude increase in compute"; "you will need about 2× as much compute as your opponent to beat them 2/3 of the time"; train/test tradeoff: "for each additional 10× of train-time compute, about 15× of test-time compute can be eliminated, down to a floor of a single-node tree search." Architectures ranged "from 1 layer of 1 neuron through to 8 layers of 1024 neurons"; the efficient 9×9 net was "a fully-connected residual net with two layers of 512 neurons" with "500k parameters and 500 KF-s for a forward pass"; perfect 9×9 play took "a little under 3 hours on a single RTX 2080 Ti"; ~200 models, ~500 GPU-h training; batch 32k, replay 2M, 64 MCTS nodes. Fitted incline coefficients 510 (log-flops), −430 (board size), −4400 (intercept). — [ar5iv 2104.03113](https://ar5iv.labs.arxiv.org/html/2104.03113)
- Neumann & Gros 2022 (preprint, AlphaZero on Connect Four / Pentago): Elo ∝ N^α_N with α_N = 0.88 (Connect Four), 0.87 (Pentago), 0.75 (Oware); Elo ∝ C^α_C with α_C = 0.55; compute-optimal size N_opt(C) ∝ C^0.62 (Connect Four) / C^0.63 (Pentago). Widths 4–256 neurons, 6 seeds each; 300 MCTS sims per move; at fixed inference compute, "smaller agents improved steadily with size." — [ar5iv 2210.00849](https://ar5iv.labs.arxiv.org/html/2210.00849)
- KataGo paper: size schedule b6c96 → b10c128 → b15c192 → b20c256 "at roughly 0.75 days, 1.75 days, and 7.5 days"; at 1,600 nodes vs. LZ/ELF reference: b20c256 +76, b15c192 −329, b10c128 −850 Elo (Table 7). Ablation (2-day runs, Elo at 2.5G self-play queries): baseline 1329; −playout-cap randomization 1242 (1.37×); −forced playouts/target pruning 1276 (1.25×); −global pooling 1153 (1.60×); −aux policy targets 1255 (1.30×); −aux ownership/score 1139 (1.65×); −game-specific features 1168 (1.55×); product ≈ 9.1×. Main run: 19 days, ≤28 V100. — [ar5iv 1902.10565](https://ar5iv.labs.arxiv.org/html/1902.10565)
- KataGo README mirror (project doc; fixed 1,200 visits, "NOT fixed computation time"): g170 run Elo by net — b6c96 −1176 (0.75 d), b10c128 −281 (1.75 d), b15c192 514 (7.5 d), b20c256 955 (15.5 d) … 1293 (38.5 d), b30c320 1409 vs. b40c256 1405 (≈47 d), b30 1545 vs. b40 1554 (78 d); extended-training b15 933 vs. extended b20 1346 at 47.5 d (413 Elo gap). June 2019 g104 run: b6 −1133, b10 −482, b15 331, b20 904. — [zeta1999/katago mirror README](https://github.com/zeta1999/katago)
- Current KataGo distributed run (katagotraining.org): Elo (own scale) b11c768 14602 ±16; b40c768nbt 14521; b10c512 14167; b28c512nbt 14092–14156; b18c384nbt 13621; b60c320 13542 — note wide-and-shallow nets (b10/b11 with 512–768 channels) now top the table. — [katagotraining.org/networks](https://katagotraining.org/networks/)
- Generals 2026 used a 15.35M-param transformer (depth 7, d=448, 8 heads) at 4 GPUs × 7 days; blog: "wider shapes trained faster at fixed parameter count" and "trading network size for more games played might help." — [arXiv 2606.23348](https://arxiv.org/html/2606.23348v2); [Straka blog](https://kam.mff.cuni.cz/~straka/blog/generals.html)
- Jones also reports larger nets were not needed on small boards: "far smaller - and faster - networks could make it to perfect play." — [ar5iv 2104.03113](https://ar5iv.labs.arxiv.org/html/2104.03113)

### Inferences
- With ~35–50k samples/s on 4×H100 a 0.5M-param CNN is almost certainly far below the compute-optimal size (Neumann: N_opt grows as C^0.62; KataGo moved from ~b6 to b10 within the first day of a 16–24 GPU run). The expected ordering of cheap wins, from KataGo's ablation, is: auxiliary ownership/score-style heads (1.65×) ≈ global pooling (1.60×) > game-specific features (1.55×) > a width increase. Try ~2–4× width (not depth) first, consistent with the "wider trains faster" observation and KataGo's current wide-shallow leaders.
- Jones's 500 Elo per 10× compute gives a budget rule of thumb: a 2× compute run should move the Elo ladder by ~150 Elo if nothing else is broken; a flat ladder at 2× compute means a plateau in the pipeline, not insufficient compute.

### Gaps
- No 4X-specific net-size vs. Elo study exists; the closest are Hex/Connect-Four/Go. KataGo's per-size numbers are at fixed visits, not fixed wall-clock, and are project docs, not peer-reviewed.

---

## Q5. Published RL on Polytopia or very similar 4X games (2020–2026)

### Takeaway
No academic or hobby publication with reported results for an RL agent on The Battle of Polytopia was found; only two unofficial environments exist (polyterra-env, PolyEnv). The nearest reported results are Generals.io (superhuman, 2025–2026), CivRealm/Freeciv (RL fails in the full game, 2024), Lux AI (RL winner, 2021), and μRTS (2021).

### Cited Findings
- polyterra-env: PettingZoo multi-agent Polytopia-style environment; README points to an RLlib self-play example; macOS ARM only. — [PyPI polyterra-env](https://pypi.org/project/polyterra-env/)
- PolyEnv: unofficial engine (C++ core, Python bindings) "intended for AI training, external bots, MCTS, and inspecting saved games"; docs mention batched neural MCTS and belief-MCTS self-play; no Aquarion/Elyrion/Polaris/Cymanti. — [polyenv.readthedocs.io](https://polyenv.readthedocs.io/)
- Generals.io 2026: #1 "in all ten weekly standings of a two-month deployment on the public 1v1 ladder"; 946/1,147 rated wins (82.5%); 69.6% vs. humans rated 100+; 189–67 vs. Mithraaaa; 32–18 vs. shimatetsu; 13–0 vs. Human.exe; 672 GPU-hours (4×H200, 7 d). — [arXiv 2606.23348](https://arxiv.org/html/2606.23348v2)
- Generals.io 2025 (v1): top 0.003% of 1v1 leaderboard "among the top 25 players"; 54.82% vs. Human.exe (529 games); ~10% vs. top-5 humans; 36 h on one H100; 142.56M samples. — [arXiv 2507.06825](https://arxiv.org/html/2507.06825v2); [Straka blog](https://kam.mff.cuni.cz/~straka/blog/generals.html)
- CivRealm (ICLR 2024): "canonical RL-based agents exhibit reasonable performance in mini-games" (~90% CityTileWonder) but "struggle to make substantial progress in the full game." — [arXiv 2401.10568](https://arxiv.org/abs/2401.10568)
- Earlier Freeciv/Civ work: KB-RL (Voss et al. 2019, preprint) "is able to play and complete the full FreeCiv game, and to win against the computer players"; Ulam/Jones/Goel (AIIDE 2021) city-defense sub-task; Wender & Watson (CIG 2008) Q-learning for Civ IV city placement. — [arXiv 1908.05472](https://arxiv.org/pdf/1908.05472); [AIIDE 2021](https://ojs.aaai.org/index.php/AIIDE/article/view/18685); [Wender & Watson CIG08](https://www.cs.auckland.ac.nz/research/gameai/publications/Wender_Watson_CIG08.pdf)
- Gym-μRTS (CoG 2021): best agent 91% cumulative win rate vs. 13 competition bots after ~60 h on one GPU, 300M steps; weakest vs. MCTS worker-rush bot Droplet. — [ar5iv 2105.13807](https://ar5iv.labs.arxiv.org/html/2105.13807)

### Inferences
- The project is in unpublished territory for Polytopia specifically; the Generals.io 2026 paper is the best template (same genre: fog-of-war, grid, 1v1, long episodes, sparse reward, comparable GPU budget).

### Gaps
- No Polytopia RL results, theses or write-ups with numbers found (two searches, different phrasings).

---

## Q6. Cost/throughput engineering (brief): CPU actors vs. GPU-resident simulators

### Takeaway
Published reference points: Sample Factory ~130k FPS on one GPU + CPU envs (asynchronous PPO); Podracer/Sebulba 200k FPS on an 8-core TPU with CPU-hosted envs; Generals.io 2026 moved the whole environment onto the GPU in JAX (≈42.8M frames/s on one H200 at 2^18 envs vs. <10k steps/s on a 64-core CPU box) and attributes superhuman play within 4 GPU × 7 days to this. Our ~35–50k samples/s with CPU actors is in the Sample-Factory/Sebulba regime, three orders of magnitude below a GPU-resident simulator.

### Cited Findings
- Sample Factory (ICML 2020; peer-reviewed): "throughput as high as 130000 FPS" on a single node with one GPU, APPO with asynchronous GPU sampler and off-policy correction. — [arXiv 2006.11751](https://arxiv.org/pdf/2006.11751.pdf)
- Podracer/Sebulba (DeepMind 2021, preprint): environments on CPU host, acting/learning on TPU cores; raising actor batch from 32 to 128 "allowed the same V-trace agent to reach 200K frames per second on an 8 core TPU"; 200M Atari frames "in just ~1 hour." — [arXiv 2104.06272](https://arxiv.org/pdf/2104.06272)
- Generals.io 2026: JAX simulator "About 42.8M frames/s on one H200 at 2^18 environments, and 38.1M frames/s for 1v1 at 65,536 environments"; "CPU actors: None"; per iteration 512 envs × 512 turns per device, ~99 s/iteration with 25% advantage filter (~20 s at 100%); blog: 260B timesteps in 7 days on 4×H200, vs. "under 10,000 steps/s for a 64-core CPU machine." — [arXiv 2606.23348](https://arxiv.org/html/2606.23348v2); [Straka blog](https://kam.mff.cuni.cz/~straka/blog/generals.html)
- OpenAI Five scale for reference: Rerun used 512 optimizer GPUs, 512 rollout GPUs, 51,200 rollout CPUs; 150 ± 5 PFlops/s-days over 2 months; surgery avoided ~40 months of from-scratch retraining but "surgery-trained agents plateaued at a weaker skill level than the from-scratch model could reach." — [ar5iv 1912.06680](https://ar5iv.labs.arxiv.org/html/1912.06680)
- Jones 2021 train/test tradeoff: 10× train compute ≈ 15× test compute, down to a single-node floor. — [ar5iv 2104.03113](https://ar5iv.labs.arxiv.org/html/2104.03113)
- TStarBot-X compute split: of 144 V100s, 96 for training and 48 for inference, with 13,440 CPU cores. — [ar5iv 2011.13729](https://ar5iv.labs.arxiv.org/html/2011.13729)

### Inferences
- The highest-leverage engineering change in the genre's best result was a GPU-resident simulator (≈1,000× sample throughput), which removed the need for pools, BC and shaping. Short of that, Generals's advantage filtering (2× iterations per wall-clock, +150–250 Elo at equal time) and EMA policy (+30–65 Elo) are cheap learner-side wins compatible with CPU actors.

### Gaps
- "Duper-style setups": no system or paper by that name was found; it may be a misremembering (Podracer/Sebulba, EnvPool, or Sample Factory are the standard references).
- No source reports a direct $/Elo comparison between CPU-actor and GPU-simulator pipelines.
