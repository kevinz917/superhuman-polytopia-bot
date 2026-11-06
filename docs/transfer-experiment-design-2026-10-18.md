# Transfer experiment design v2 (from the literature review)

October 18, 2026. This is a proposed design. No part of it has run yet. It rebuilds the RL and evaluation parts of the [post-training plan](post-training-plan.md) around what the [literature review](llm-game-transfer-research-2026-10-18.md) found. Where the two disagree, this document wins for the RL recipe, the arms and the transfer protocol. The post-training plan still governs splits, the environment freeze, logprob accounting and the stop rules.

## What the papers teach, and the design rule each one implies

| Lesson | Source | Design rule |
|---|---|---|
| Game-specific transfer after controls is about +2 to +6 points, and only about +2 on instruct models | All six deep reads. SPIRAL on Instruct-2507: +1.8 | Power the study for roughly +4 points. That means seeds and paired statistics, not single runs. |
| Placebo RL (random or format-only rewards) can give large "gains" on Qwen models | Spurious Rewards, Co-rewarding, ViGaL's random-reward null | Include a placebo arm with the same format and the same compute. |
| Outcome-only RL often lowers transfer, while a dense solver signal kept it non-negative | CAST, Generalization Tax | The main arm uses a per-decision teacher signal. The control arm is the same algorithm with α = 0. |
| Reasoning text during RL is what keeps transfer alive | Generalization Tax (WebShop 30.5 with thinking vs 1.0 without), CAST format | The model writes a short rationale before every action. An action-only arm tests this directly. |
| Open-ended thinking is unaffordable, and length limits prevent collapse | Our Qwen3.5 run (196 s median per request), MARSHAL length penalty | Use prompt-level reasoning with a hard token cap, not a native thinking mode. |
| The untrained base model can look artificially bad in the trained model's format | lmgame-Bench Table 5 (WebShop 7.0 with think tags vs 37.9 without) | Evaluate the base model in the trained format **and** in its best format, and report both. |
| Game training rarely beats RL directly on the target domain | SPIRAL math-RLVR 46.0 vs 44.5; ChessArena math-RL wins on AIME | Add an equal-compute RL arm on the target domain as the ceiling and reference. |
| Long horizons destabilize RL, and short horizons generalize to longer ones | Horizon study 2605.02572, R³, CAST's 20–40-turn episodes | Train on game segments started from snapshots, not on whole games. |
| Transfer at 4B depends on in-game competence | Playing with Fire | Report transfer against in-domain skill, and test a warm-up arm. |

## Fixed choices (identical in every arm)

- **Base model:** Qwen3-4B-Instruct-2507 at revision `cdbee75f…`. This is our pinned base, and the same model CAST and C1 used. It has no native thinking mode. A Qwen3.5 thinking arm is deferred until the 30-vs-30 comparison finishes, and even then would need a hard budget.
- **Decision format:** "reason, then act", as in CAST. The prompt asks the model to think briefly, then give the action in a final block. The output is `<rationale ≤ 192 tokens> ```{"action_id":"a17"}``` `.
  - The grammar constrains only the final block.
  - Going over the cap counts as invalid. Scored decisions are first-attempt only, and any repair attempts are logged separately.
  - Every token gets that decision's advantage, so the rationale is trained along with the action.
- **Context:** a fresh prompt for every decision, with the existing compact public state and seat-isolated factual memory. No accumulated chat history, no earlier rationales.
- **Training episodes are segments, not whole games.** Each episode starts from a snapshot drawn from teacher, student or league games (stratified by game phase). It runs H = 40 learner decisions, then truncates with a bootstrapped value. H starts at 20 and grows to 40 → 80 if training stays stable.
- **Teacher signal:** the deterministic PPO bot `low256-league-rivals-v1` u96.
  - V is the learner seat's win probability.
  - ΔV is measured at the afterstate: restore the snapshot, apply the action, read V *before* any opponent move.
  - Our end-turn action moves play to the opponent. For it, the afterstate is the state just before the opponent's first action.
- **Backbone algorithm:** DAPO-style (token-mean loss, clip-higher, dynamic sampling), the same backbone as CAST. Group of 8 rollouts from each snapshot.
- **Opponents:** a frozen league: PPO checkpoints, the scripted teacher and the weak baseline. It is identical across arms and fixed before the pilot.
- **Budget per arm:** the same number of optimizer steps, rollout decisions and generated tokens. A run stops at a preregistered step count, not at whichever checkpoint looks best.

## Arms

| Arm | Training | Question it answers |
|---|---|---|
| **A0** | None (base model) | Reference. Evaluated in the trained format and in its best format. |
| **A1** | Segment RL, outcome only. Reward is the segment-end bootstrapped V, or ±1 if the game resolves inside the segment. α = 0. | Does plain game RL transfer? (CAST's DAPO row) |
| **A2 (main)** | A1 + α·asinh(ΔV)/RMS per decision, α = 0.1 | Does the dense teacher signal add transfer over A1? |
| **A3 (placebo)** | A2 with ΔV shuffled across decisions in the batch, and segment outcomes shuffled across episodes | Is any gain just RL drawing out reasoning or formatting in Qwen? |
| **A4** | A2 with no rationale (action-only grammar) | Is the reasoning text necessary for transfer? |
| **A5 (ceiling)** | Equal-compute RL on the target-domain *training* splits (ALFWorld train, WebShop train), same format | How does game training compare with training directly on the target domain? |
| A6 (stage 2) | Light soft-target distillation from the PPO bot (5–10K decisions, top-k distribution), then A2 | Does warm-up competence increase transfer? |

Seeds: three per arm for A0–A5. A0 has no training seeds, so its evaluation is repeated with four sampling seeds. A6 and other extensions are added only if A2 > A1 in the pilot.

Confirmatory comparisons, declared before the pilot, with a Holm correction:
1. A2 − A0
2. A2 − A1
3. A2 − A3

Everything else is exploratory.

## Evaluation

**Calibrate the harness before any training.**
- Reproduce CAST's untrained-base numbers for Qwen3-4B-Instruct-2507 with their prompts: ALFWorld 30.2 on 140 `valid_seen` games, WebShop 18.8 on 500 test instructions, temperature 0.6, top-p 0.95, Avg@4.
- If we can't get within about 2 points, fix the harness before going further.

**Primary transfer outcome:** the mean success of ALFWorld and WebShop, equally weighted.
- Zero-shot, using each task's own reason-then-act prompt, with no Polytopia content.
- Avg@4 on every instance.
- ALFWorld is also reported on the unseen split (134 games) as a held-out check.

**Secondary outcomes:**
- **Planning:** Blocksworld, the lmgame-Bench variants.
- **Logic:** ZebraLogic.
- **Math and knowledge:** MATH500, AIME24/25 at avg@32, GPQA-Diamond, MMLU-Pro. These are diagnostics only.
- **Retention:** IFEval and one coding set. Every regression is reported.

**In-domain outcomes:**
- Win rate against the scripted teacher and the PPO league on held-out maps, both seats.
- Mean per-decision ΔV, and invalid-action rate.

**Statistics:**
- Per-seed scores, plus paired bootstrap intervals over instances, clustered by task template, with seeds as an outer resampling level.
- Report each arm's transfer **against its in-domain win rate**, so a better game policy that degrades the format less can be told apart from transferred skill.

**Checkpoints:**
- Evaluate transfer at 25%, 50% and 100% of training, and report all three.
- Choose the headline checkpoint by **in-domain validation only**.

**Monitors to log on every run:**
- Rationale length, and whether the rationale depends on the input (RAGEN-2 template collapse).
- Action and unit-type diversity (the Warrior issue).
- KL to the base model, entropy, clip fraction.
- Fraction of decisions with |ΔV| below a threshold.

## Gates, in order

1. **Teacher API** (task card already open): full action distribution, V, and afterstate ΔV. Tests check the sign on near-win and near-loss states.
2. **ΔV validity:** on 500 sampled states, check that ΔV ranks actions the same way as snapshot Monte Carlo rollouts do (H = 40, against the league). Require a positive rank correlation before using it as a reward. If it fails, fall back to an MC-branch advantage computed with GiGPO-style grouping.
3. **Format pilot:** at A0, the 4B model in reason-then-act format must reach at least 95% first-attempt validity on Polytopia, with median rationale length under the cap. Otherwise adjust the prompt, not the weights.
4. **Harness calibration:** reproduce the CAST base numbers, as above.
5. **Pilot:** A1 vs A2, one seed each, about 300 steps. Continue only if:
   - A2 learns in-domain (ΔV and win rate rise);
   - logprobs reproduce;
   - the rationale doesn't collapse.

   Evaluate pilot transfer, but don't treat it as confirmatory.
6. **Main study:** A0–A5 × 3 seeds. Then A6 if A2 > A1.

## Cost anchor

CAST used 200–400 updates of 16 prompts × 8 rollouts, at about 190 s per update (GPU count not stated). Our segments are longer: about 40 decisions × ≤ 220 generated tokens each. Profile one segment batch before sizing a budget. Do not extrapolate from CAST's wall-clock time.
