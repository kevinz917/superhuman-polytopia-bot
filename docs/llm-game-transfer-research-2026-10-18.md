# Long-horizon game post-training for transfer: literature and implementation plan

October 18, 2026. Literature scan plus a proposed recipe for the Qwen 4B game-transfer study in the [post-training plan](post-training-plan.md). This note proposes work only. It runs no experiments and claims no results. The figures below come from each paper's abstract or HTML page unless marked **(unverified)**. Several papers report different numbers in different arXiv versions.

## 1. What the literature says, in one paragraph

No published work RL-trains an LLM on a full 4X or RTS game (Civilization, Freeciv, Unciv, StarCraft II) and then measures transfer outside the game. Polytopia has no papers at all, so this study would be the first. Every transfer result comes from short games, where the model reasons in text and is trained with RL: SPIRAL, MARSHAL, Stratagem, lmgame-Bench and ViGaL. SFT on expert game trajectories transfers less and can reduce general scores: SPIRAL's SFT baseline, card games, and Huan et al. Action-only policies have essentially no transfer evidence. Rationales help held-out tasks more than in-domain fit (AgentBank). Long horizons are handled with dense per-step signals and resettable states, not with longer whole-game rollouts.

## 2. Papers most relevant to a long-horizon strategy game

Ranked by how much we should copy from each.

| # | Paper | Why it matters here | Copy this |
|---|---|---|---|
| 1 | **Supervised RL (SRL)**, Google, ICLR 2026, [2510.25992](https://arxiv.org/abs/2510.25992) | Expert trajectories become per-step tasks. The model writes its own reasoning, and the reward is only how similar its action is to the expert's. That removes the need to supervise the reasoning. Running SRL before RLVR was best. Reported (unverified): math SFT 16.6, RLVR 24.5, SRL 27.6, SRL then RLVR 28.3. SWE-bench: SRL 8.6 vs SFT 4.2. | Per-decision reward from the PPO teacher, with the model's own reasoning. |
| 2 | **Master Distillation / C1**, Tang et al., [2603.20510](https://arxiv.org/abs/2603.20510) | Same base model as ours (Qwen3-4B-Instruct-2507). The expert is an engine, Stockfish, and has no reasoning. An LLM writes rationales while shown the engine's answer and told to reason "as if the solution is unknown". Result: SFT 40.9%, then RLVR 48.1%, beating its teacher. Outputs are about 180 tokens. Removing the feigned-discovery prompt cut accuracy from 20.1% to 16.3% in the 8k-sample ablation. | Generating rationales for action-only expert data, kept short. |
| 3 | **DipLLM**, ICML 2025, [2506.09655](https://arxiv.org/abs/2506.09655) | Closest action structure to ours: a multi-unit turn factored into autoregressive per-unit decisions. Fine-tuned toward an equilibrium policy, it beats Cicero with 1.5% of Cicero's data. Reported (unverified): Llama-3-8B LoRA, targets weighted by exp(Q). | Distil the teacher's whole action distribution, not its single top move. |
| 4 | **SPIRAL**, ICLR 2026, [2506.24119](https://arxiv.org/abs/2506.24119), and **MARSHAL**, ICLR 2026, [2510.15414](https://arxiv.org/abs/2510.15414) | Qwen3-4B self-play produces transfer. SPIRAL raised its 8-benchmark average from 34.0 to 44.5, while SFT on 25k expert games reached 39.7. MARSHAL gained up to +10 AIME and +7.6 GPQA. Both document reasoning collapse and how they stopped it. | Per-role baselines (RAE). MARSHAL's per-turn length penalty (cap 2048). Their transfer evaluation suite. |
| 5 | **Chess RLVR**, Hwang et al., [2507.00726](https://arxiv.org/abs/2507.00726) | GRPO with a dense per-move reward from an action-value network beat a sparse reward. Every model plateaus because base models lack chess knowledge. | A value network as dense reward, plus the warning that the teacher sets a ceiling. |
| 6 | **GiGPO / verl-agent**, NeurIPS 2025, [2505.10978](https://arxiv.org/abs/2505.10978); **VinePPO**, [2410.01679](https://arxiv.org/abs/2410.01679) | Per-step advantages without a critic: group actions taken from the same state, or estimate values by resetting to intermediate states. GiGPO reports +12% ALFWorld and +9% WebShop over GRPO. Our snapshot/restore makes both exact. | Fork from snapshots and compare branches. |
| 7 | **Diplomacy GRPO**: [2602.05183](https://arxiv.org/abs/2602.05183) (Qwen3-235B) and the [Glickenhaus blog](https://www.benglickenhaus.com/blog/diplomacy_rl_part_1) (Qwen3-14B on Modal, not peer-reviewed) | The most practical long multi-call game RL recipe published. GRPO groups come from forking one mid-game state. Legal moves are enforced with a trie, and opponents are chosen by PFSP league. Shaped rewards were reward-hacked. | Mid-game forks and a league of opponents. Audit shaped rewards. |
| 8 | **Game-TARS**, ByteDance, [2510.23691](https://arxiv.org/abs/2510.23691) | "Sparse thinking": reason only at hard steps. Reported (unverified, hard to read from the table): Minecraft success 55% with no thinking, 45% thinking every step, 63% sparse. Thoughts were kept only if they led to the expert's action. | Think at turn start and at hard decisions only. Keep only thoughts that agree with the expert. |
| 9 | **Horizon/curriculum:** R³ [2402.05808](https://arxiv.org/abs/2402.05808), ScalingInter-RL [2509.08755](https://arxiv.org/abs/2509.08755), horizon study [2605.02572](https://arxiv.org/abs/2605.02572) | Start near the outcome and move the start earlier. Training on short horizons is more stable and generalizes to long ones. | Mid-game snapshot starts, with segments lengthened over training. |
| 10 | **Diversity collapse**, [2607.19523](https://arxiv.org/abs/2607.19523) | SFT on one demonstrated move collapses action diversity, and reasoning mode narrows moves further. Training on every good action partly fixes it. Relevant given our ~99% Warrior issue. | Soft or multi-action targets. Track action diversity. |

**Negative and cautionary evidence to cite:**
- *Playing with Fire* (Hanabi, ICML 2026): with a 4B model, transfer depends on how competent the model is before RL.
- lmgame-Bench: GSM8K was flat and BIRD regressed after game RL.
- Card games ([2509.01328](https://arxiv.org/abs/2509.01328)): SFT erodes general capability.
- CivBench: over hundreds of turns, agents fail to follow through on their own plans.

### Which papers actually measured transfer

Numbers below were checked against each paper's own tables on 2026-10-18. The chess papers C1 and Hwang et al. evaluate only on chess, and so does DipLLM. They are method sources, not transfer evidence.

**Tier 1: transfer shown with numbers and a baseline**

| Paper (game → base model) | Out-of-domain result | Strength of evidence |
|---|---|---|
| **CAST** [2607.25308](https://arxiv.org/abs/2607.25308) (Sokoban / Minesweeper / Rush Hour; RL where the turn-level advantage is the change in a game solver's value; **Qwen3-4B-Instruct-2507**) | Zero-shot: ALFWorld **30.2 → 37.9**, WebShop **18.8 → 22.7** | Our base model, 3 runs, but no variance reported for transfer. On WebShop, all four outcome/GiGPO baselines average below base (16.6–17.9 vs 18.8). On ALFWorld they average about the same as base (27.2–32.1), and 7 of their 12 per-game cells beat it. CAST beats DAPO, its own backbone, on WebShop by about 6 points from every training game. That is the most robust result. The closest template to our PPO value network. Full reading below. |
| **SPIRAL** [2506.24119](https://arxiv.org/abs/2506.24119) (Kuhn Poker / TicTacToe / Negotiation self-play) | Qwen3-4B-Base: 8-benchmark average **34.0 → 44.5**. AIME24 9.6 → 19.7, GPQA 30.6 → 40.1, MMLU-Pro 47.2 → 58.5. Qwen3-8B-Base 39.5 → 49.6. | Beats SFT on 25k expert games (39.7) on all 8 benchmarks. 3 seeds (±0.5). But on post-trained models the gain is small: Llama-3.1-8B-Instruct 23.9 → 25.9, R1-Distill-7B 60.4 → 61.8. No equal-compute math-RL control. |
| **Stratagem** [2604.17696](https://arxiv.org/abs/2604.17696) (same games, transferability-weighted advantage) | Qwen3-4B-Base: MATH500 65.8 → 76.0, MMLU-Pro 47.2 → 57.8, HumanEval 67.9 → 77.9, AIME24 10 → 20 | Beats SPIRAL. On Qwen3-4B-Instruct it reports only SPIRAL vs Stratagem, with no untrained instruct row, so the instruct-model transfer is unproven. No seeds reported. |
| **ViGaL** [2506.08011](https://arxiv.org/abs/2506.08011) (Snake + Rotation RL; Qwen2.5-VL-7B) | Math avg 47.7 → 50.6, MMMU 54.3 → 58.0, geometry 44.8 → 57.1 | General visual benchmarks preserved (72.4 vs 72.2). Matches MM-Eureka, a math-RL model (50.1). Multimodal, so a weaker analogue for us. |
| **lmgame-Bench** [2505.15146](https://arxiv.org/abs/2505.15146) (Sokoban or Tetris RL; Qwen2.5-7B-Instruct) | WebShop 7.0 → 19.1, Blocksworld-2D 9.0 → 29.5 | No transfer to math (GSM8K 89.5 → 89.0). BIRD regressed 25.0 → 17.5. No error bars. |
| **ChessArena** [2509.24239](https://arxiv.org/abs/2509.24239), Table 20 (chess SFT → GRPO with Stockfish reward; Qwen3-8B) | ZebraLogic **25.9 → 48.0** | Everything else flat or down (AIME25 18.6 → 19.3, LiveCodeBench unchanged, BigCodeBench, CruxEval and DROP slightly down). Chess SFT alone hurt. Math-only RL gets the better AIME (21.3). |

**Tier 2: claimed, weaker evidence**
- **MARSHAL** ([2510.15414](https://arxiv.org/abs/2510.15414)): the headline +10 AIME / +7.6 GPQA is measured inside multi-agent systems. Single-agent rows look mixed.
- **Orak**: its abstract makes no transfer claim. The 3B transfer numbers are unverified.
- **Good Start Labs' Diplomacy RL**: blog claims only.

**Negative evidence**
- **Paying Less Generalization Tax** ([2601.18217](https://arxiv.org/abs/2601.18217), Llama-3.1-8B-Instruct): RL on one agent environment usually *lowers* the others. For example, Sokoban RL took WebShop 38.3 → 34.0 and SciWorld 18.0 → 13.0. Turning thinking off during RL collapsed out-of-domain scores, e.g. after ALFWorld RL, WebShop scored 30.5 with thinking vs 1.0 without. SFT warmup protects domains in its data mix and hurts domains outside it.
- **Playing with Fire** (Hanabi RL): at 4B the gains don't transfer, and a 120B model is needed.

What this pattern implies:
- Transfer is narrow. It mostly appears on tasks shaped like the game: logic grids, planning, and agentic tasks.
- Gains are largest on Base models, which RL also teaches to reason step by step.
- Gains shrink on already-tuned models. Our Instruct-2507 is one, so expect smaller effects.
- SFT-only game training usually costs general ability.
- RL directly on the target domain usually beats game training on that domain. This is why the post-training plan's equal-compute C2 control matters.

### CAST, read in full (2026-10-18)

**Setup.** Each agent is trained on a single game (Sokoban, Minesweeper or Rush Hour; 6×6 boards; ≤20–40 turns). The policy writes ReAct reasoning, then one action, every turn. CAST is DAPO plus `α·asinh(ΔN)/RMS` added to every turn's advantage, with α = 0.1 and ΔN = the drop in the solver's cost-to-go. Training runs 200/400/300 steps, batch 16 × 8 rollouts, and the final checkpoint is used. Transfer is evaluated zero-shot, with the same ReAct loop, on ALFWorld (140 `valid_seen` games, 30 turns) and WebShop (500 test instructions, success rate). Both use Avg@4. Table 2 is the mean of 3 runs, but reports **no std, CI or per-seed numbers**, and the repo releases no eval logs.

**Table 2, per training game (ALFWorld / WebShop; base ReAct 30.2 / 18.8):**

| | Sokoban | Minesweeper | Rush Hour | Avg |
|---|---|---|---|---|
| GRPO | 31.5 / 20.4 | 22.3 / 14.0 | 27.9 / 18.0 | 27.2 / 17.5 |
| GSPO | 34.2 / 18.0 | 22.0 / 13.1 | 40.0 / 20.9 | 32.1 / 17.3 |
| DAPO | 37.3 / 17.5 | 35.2 / 18.8 | 18.6 / 13.6 | 30.4 / 16.6 |
| GiGPO | 35.6 / 17.2 | 24.9 / 20.2 | 32.1 / 16.4 | 30.9 / 17.9 |
| CAST | 38.4 / 23.2 | 33.9 / 25.1 | 41.4 / 19.7 | 37.9 / 22.7 |

**What survives scrutiny:**
- **WebShop.** CAST − DAPO is +5.7, +6.3 and +6.1 from the three games. The same backbone, with only the solver term added, gives a consistent gain. Outcome-only cells are mostly at or below base (8 of 12 below, 1 tied).
- **CAST never regresses below base** in all 6 cells. Every baseline has at least one bad game.

**What is overstated:**
- **"Baselines failed to transfer" is false for ALFWorld.** Sokoban-trained DAPO reaches 37.3 vs CAST 38.4. GSPO from Rush Hour reaches 40.0 vs 41.4. CAST's +7.5 ALFWorld average over DAPO comes almost entirely from one DAPO Rush Hour cell, 18.6. The paper itself notes that DAPO/GSPO drop late in training on Rush Hour, and evaluation uses the final checkpoint, so this cell partly measures training instability.
- **Noise.** CAST is best in only 4 of 6 per-game cells, by +1.1 to +4.9. Binomial sampling SE alone is ~2–4 pt per run on ALFWorld (n = 140) and ~1–2 pt on WebShop (n = 500). Near-identical objectives also swing transfer by more than 10 pt (ALFWorld from Minesweeper: DAPO 35.2 vs GRPO 22.3). So seed variance is probably larger than most per-cell gaps.
- **No mechanism is tested for transfer.** Every ablation (α; asinh; RMS) is on Sokoban, in-domain only. The learned-value-network variant (DQN on Rush Hour) is shown only in-domain. The case for *why* is one cherry-picked WebShop trajectory. Nothing separates "dense credit teaches transferable skills" from "CAST is a better, more stable game policy (ID 62 vs 45) that degrades the ReAct format less".
- **The OPD equivalence is near-tautological.** Any Q defines a Boltzmann policy, and adding β·A per turn is a policy gradient on cross-entropy to that policy. The result is cross-entropy, not KL, so it is mode-seeking and entropy-reducing. The proof also linearises asinh and freezes visitation. It explains the shape of the update, not transfer.

**For us:**
- Our PPO bot has a **policy head over the same action IDs**. So we can do true on-policy distillation, using `log π_PPO` or a per-step reverse KL. We don't need CAST's logit-free trick.
- The CAST-style arm is `V(s′) − V(s)`, where s′ is an afterstate taken by snapshot/restore *before* opponents move, to avoid opponent noise. Our V is V^π_PPO (win probability against a league), not an optimal cost-to-go. Before relying on it, validate it per step: does ΔV predict snapshot-rollout outcomes?
- The biggest mismatch: CAST reasons before every action, and our serving grammar is action-only (`{"action_id":…}`). Generalization Tax shows action-only RL can collapse ALFWorld/WebShop transfer. A faithful replication needs the rationale/sparse-thinking format of Stages 2–3.
- To count as evidence, we need: ≥3 seeds per arm; an α = 0 (DAPO) control at equal compute; per-seed and paired-bootstrap CIs; multiple checkpoints, not just the final one; and transfer reported against in-domain win rate. Calibrate the harness first by reproducing the base 30.2 / 18.8 with CAST's prompts (Appendix C.1), 140 valid_seen games, 500 WebShop tests, temperature 0.6, top-p 0.95. An expected effect of +4–8 pt is close to the noise floor.

### Which paper transfers best? Deep read of all six (2026-10-18)

Every paper was read in full. "Over control" is the gain that remains after subtracting the paper's strongest in-paper alternative. The benchmarks differ, so compare rows by direction and evidence quality, not by raw magnitude.

| Paper | Headline transfer (raw gain over untrained base) | What remains over the strongest control | Seeds / variance | Evidence (1–10) |
|---|---|---|---|---|
| **ViGaL** | Math avg 47.7 → 50.6 (+2.9). Geo3K +17.7, CLEVR-Math +25.7 | **Random-reward RL gives nothing** (49.1 → 49.4 vs 62.3 real). SFT on the same data is worse (47.2). Only ≈ +0.5 over the math-RL model MM-Eureka, with data not matched | 3 seeds (±0.3) | **6** |
| **CAST** | ALFWorld +7.7, WebShop +3.9 (our base model) | Over DAPO (same backbone, α = 0): WebShop +5.7 / +6.3 / +6.1 across the three games, consistent. The ALFWorld margin comes mostly from one unstable DAPO cell | Mean of 3, no std | **5** |
| **SPIRAL** | Qwen3-4B-Base 8-bench avg 34.0 → 44.5 (+10.5) | +4.8 over SFT on the same games. **Math RLVR alone does better (46.0).** Entropy-only RL on Qwen3-4B-Base reproduces much of the math gain ([Co-rewarding](https://arxiv.org/abs/2508.00410)). **On Qwen3-4B-Instruct-2507 only +1.8 (74.1 → 75.9)** | 3 seeds on 4B only (avg ±0.5, GPQA ±3.1) | **5** |
| **ChessArena** | ZebraLogic 25.9 → 48.0 (+22.1) | Chess SFT then *math* RL already gives 42.4. Chess RL adds only +5.6. AIME +0.7 is noise. Math RL without chess wins on AIME, CruxEval and DROP | None | **4** |
| **lmgame-Bench** | WebShop 7.0 → 19.1, Blocksworld-2D 9 → 29.5 | WebShop is mostly a thinking-format effect: base without thinking scores 37.9. Measured in the same format the gain is +5.9. GSM8K flat, BIRD −7.5 | None | **3** |
| **Stratagem** | 9-bench avg 36.9 → 47.2 (+10.2) | +5.2 over its own SPIRAL re-run. Single sample per problem: one AIME problem is 3.3 points, and the sampling SE is ≈ 7 points. Baselines appear copied across protocols | None | **3** |

**Verdict:**
- **Most credible transfer: ViGaL.** It is the only paper with a placebo (random-reward) control, and that control gives zero. But its core-math gains are small, and it is multimodal.
- **Biggest effect: SPIRAL's +10.5.** About 60–80% of it is plausibly generic: RL draws out step-by-step reasoning and the answer format in a Qwen3 base model. Evidence for this reading:
  - SFT on the game transcripts, which contain no math, already adds +5.6.
  - The base model is evaluated few-shot on GPQA and MMLU-Pro, while the trained model is evaluated zero-shot.
  - Entropy-only RL matches much of the gain.
  - Qwen3-Base partially memorizes MATH-500 and AIME24 ([2507.10532](https://arxiv.org/abs/2507.10532)).
- **Most relevant to us: CAST**, our base model with a solver as teacher. Its result is consistent on WebShop but close to noise.
- **What survives after controls** is roughly +2 to +6 points per paper. No paper shows a game beating RL directly on the target domain. The best complementary result is SPIRAL's appendix: RLVR then SPIRAL scores 47.9 vs RLVR alone 46.0, from a single run.
- **For Qwen3-4B-Instruct-2507, expect small effects.** SPIRAL's own instruct run gained +1.8, and SFT lowered it by −2.2. Our study needs ≥3 seeds, placebo arms (shuffled-outcome RL, format-only RL) and an equal-compute target-domain RL arm. Without those, a positive result would not be distinguishable from these confounds.

## 3. Thinking vs non-thinking, and the trace mismatch

You are right that our expert data (the PPO bot and scripted teachers) has actions and no reasoning. Here is how each option fares.

| Option | Verdict | Evidence |
|---|---|---|
| A. Thinking model, SFT with empty `<think></think>` on expert actions | **Avoid.** It erases reasoning. | [Reasoning-Trace Collapse, 2605.21127](https://arxiv.org/abs/2605.21127): answer-only SFT rapidly suppresses valid reasoning while answer metrics hide it. Qwen3 itself mixes self-generated thinking samples into its no-think stage to prevent this ([2505.09388](https://arxiv.org/abs/2505.09388)). A vendor blog reported 100% → 0% valid traces on Qwen3-4B-Thinking (unverified). |
| B. Non-thinking model, action-only SFT/distillation | Fine for game strength and cheap. **Weak for transfer.** | SPIRAL's SFT baseline, AgentBank's ablation removing rationales, Huan et al. |
| C. Write rationales for the expert's actions, then SFT on rationale + action | **Promising.** Most direct evidence on our exact base model. | C1 / Master Distillation, STaR rationalization, Thought Cloning, AgentBank. Risk: the rationales may not reflect how the expert actually decided. Short rationales and later RL reduce this. |
| D. Do not supervise reasoning at all: the model writes its own reasoning, and RL rewards only its action, judged by the PPO teacher | **Most promising.** The mismatch disappears, because the reasoning never has to match anything. | SRL, on-policy distillation ([Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/): about 1/10 the GPU-hours of RL), Hwang et al. dense reward. |

Two further conclusions:
1. **Bounding the reasoning matters more than which model you pick.** C1 got nearly the same result from Instruct-2507 and from Qwen3-4B with reasoning. Our own measurements rule out open-ended thinking: Qwen3.5-4B thinking took a median of 196 s per request, and responses ran past 100k tokens, over thousands of decisions per game. The useful format is a short rationale of at most ~256 tokens, then the action. Use it at turn start and at hard decisions (Game-TARS), with a length penalty (MARSHAL) and per-role baselines (SPIRAL).
2. **Recommended base: Qwen3-4B-Instruct-2507** (our pinned base, and C1's), taught a bounded "brief plan, then `action_id`" format. This avoids Qwen3.5's native long-thinking prior, and that prior is exactly what blew up our latency. Revisit when the 30-vs-30 Qwen3.5 thinking comparison finishes. If thinking-on clearly wins games, Qwen3.5 with a hard thinking budget becomes the alternative arm.

## 4. Proposed recipe

> **Superseded for arms and evaluation** by [Transfer experiment design v2](transfer-experiment-design-2026-10-18.md), which folds in the deep-read lessons: placebo arm, α = 0 control, an equal-compute target-domain arm, segment training and format-matched baselines. The stages below remain as background.

Stages build on each other. Each stage has a gate before scaling. Arm labels extend the post-training plan's B0–C2.

### Stage 0: teacher API (prerequisite, small)
- Add a method to `StructuredPolicy` ([policy.py](../src/polytopia_bot/rl/policy.py)), e.g. `distribution(observation)`. It returns the probability of **every** legal action plus V(s), by scoring each catalog entry with `network.evaluate` ([network.py:95](../src/polytopia_bot/rl/network.py)). `select` today samples only one action.
- Add `action_values(env, observation)`: `snapshot()` → for each candidate (or the top-k by probability) `restore` → step → V(s′) → restore ([environment.py:41](../src/polytopia_bot/environment.py)). Check carefully whose turn and perspective V(s′) is from after end-turn or after an opponent reply. Test the signs on known winning and losing states.
- Map prompt-local IDs (`a17`) to simulator IDs. `StructuredPolicy` rejects prompt-local IDs today.

### Stage 1: soft-target distillation, action-only (arm **D1**, cheap baseline)
- Collect states from games played by the student, the teacher and a mix of both (DAgger-style). Label each state with the PPO teacher's top-k distribution (k ≈ 8, renormalized), as in DipLLM, instead of the one sampled action.
- Loss: cross-entropy against the soft target, summing over each candidate's action-ID token sequence. This reuses `training.py`'s response masking.
- Purpose: in-game competence and valid formatting. Following *Playing with Fire*, being competent before RL is what gates transfer. It also avoids training a whole corpus onto ~99% Warrior choices from one-hot labels.

### Stage 2: rationale SFT (arm **R1**)
- For a stratified subset of states (turn starts, and high-entropy or high-ΔV decisions), prompt a strong LLM with:
  - the public prompt,
  - the teacher's top-3 actions with probabilities and V(s′) values,
  - the scripted teacher's per-action `reason` strings (`TeacherDecision.scores`, [teacher.py](../src/polytopia_bot/learning/teacher.py)).
- The instruction is C1's "feigned discovery": write ≤150 tokens of reasoning that reaches the chosen action, without mentioning the teacher or the numbers.
- Filter: keep a rationale only if a fresh model, given the rationale alone, picks the same action (Game-TARS-style consistency).
- Train with the rationale at hard steps and an empty plan at easy steps (sparse thinking). Output format: `{"plan": "...", "action_id": "a17"}`. Replace the current action-only grammar with a schema that caps the plan length. Keep the grammar mask identical in the learner and in serving.

### Stage 3: dense-reward RL (arms **S1** from D1 and **S2** from R1; the main transfer arms)
- Per-decision reward, SRL / on-policy-distillation style:
  - r_t = log π_PPO(a_t | s_t), or the advantage V(s′_{a_t}) − V(s_t) from Stage 0;
  - plus a terminal ±1 with per-seat baselines (RAE);
  - plus a length penalty above the plan budget.
- Decay the dense weight over training so the win/loss outcome becomes the objective and the teacher's ceiling isn't permanent (the Chess RLVR lesson).
- Step-level credit: at selected decisions, preferably where teacher ΔV is spread widely across actions, fork K = 4–8 branches from the snapshot. Roll out H = 20–100 decisions and bootstrap with teacher V. Advantage = branch return minus group mean (GiGPO / VinePPO). This replaces the post-training plan's whole-game Monte Carlo outcome, which gives the same advantage to all ~1,000+ decisions.
- Curriculum: start from mid- and late-game snapshots of teacher and student games, then move earlier (R³, horizon generalization).
- Opponents: a PFSP league of PPO checkpoints, scripted bots and past LLM checkpoints, as already planned.
- Collapse monitors:
  - plan length;
  - whether the plan depends on the state (RAGEN-2 "template collapse");
  - action and unit-type diversity;
  - KL to the starting policy.

### Stage 4: outcome-only RL continuation (optional)
Once dense-reward training plateaus, continue with sparse outcome rewards and snapshot forks only. The aim is to test whether the student can exceed its teacher, as C1 did.

## 5. Evaluation additions

- Keep the post-training plan's preregistered B0/B1/B2/B3/C1/C2 comparisons. Add D1 (soft action-only), R1 (rationale SFT), S1 and S2 (dense RL).
- The key contrasts:
  - **S2 − S1:** does supervised reasoning help transfer beyond self-generated reasoning?
  - **S1 − D1:** what does RL add?
  - **R1 − D1:** what do rationales add without RL?
- Transfer suite:
  - SPIRAL's 8 reasoning benchmarks (math, GPQA, MMLU-Pro);
  - agentic held-out tasks (WebShop, ALFWorld, τ²-Bench), which matter most given the plan's "trajectory" success definition;
  - every regression reported, as lmgame-Bench does.

## 6. Main risks

- **Teacher ceiling and bias:** the PPO bot recruits ~99% Warriors ([investigation](unit-diversity-investigation-2026-10-18.md)). Soft targets help, decaying the dense weight helps, and Stage 4 tests whether the student can go beyond the teacher.
- **Value exploitation:** the student may find states the PPO value head misjudges. Keep the terminal reward, and audit replays where shaped reward is high but the game was lost.
- **Unfaithful rationales:** keep them short and consistency-filtered, then let RL reshape them.
- **Throughput:** reasoning at every decision is unaffordable. Sparse thinking and a ≤256-token budget are requirements, not options.
- **Evidence gap:** no paper directly compares answer-only SFT → RL with rationale SFT → RL. Our S1 vs S2 contrast would be a contribution in itself.
