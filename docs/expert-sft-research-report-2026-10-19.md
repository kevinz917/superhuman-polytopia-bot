# Imitating an RL bot makes a 4B LLM win at Polytopia — research report (2026-10-19)

**Headline.** We fine-tuned `Qwen/Qwen3-4B-Instruct-2507` with a small LoRA, about 2 hours on 4 B200s, on 39,626 moves from our structured RL bot. Results, both with the same readable prompt and legal-action grammar as the base model:

- **Development maps:** the model went from **0 wins to 18 of 30 games** against the scripted `balanced` opponent.
- **Held-out test maps:** it won **77 of 170 resolved games (45%)** across three opponent styles. On the same games the base model won **0 of 180**.
- **Paired difference against base, clustered by map:** **+0.43** (95% interval [0.37, 0.49]). Expert-SFT was better on all 30 held-out maps and worse on none.
- **Gap to the expert:** the RL bot itself still wins 179/180, so imitation closes about 45% of the gap.

Checkpoint record: [`docs/expert-sft/checkpoint-expert-sft-v1.json`](expert-sft/checkpoint-expert-sft-v1.json). Data: [`docs/expert-sft/`](expert-sft/).

## 1. Question and context

Earlier work (Oct 16–18) tried three ways to get a 4B LLM to play the 14×14, 1v1 Bardur ruleset:

| Approach | Best result vs scripted `balanced` |
|---|---|
| Bare Qwen3-4B-Instruct, readable prompt + grammar | 0/30 |
| Qwen3.5-4B thinking off | 0/30 |
| Qwen3.5-4B thinking on (about 7.5k reasoning tokens per move, about 17.9 H100-hours) | 2 wins / 16 losses / 11 unfinished |
| SFT on scripted-teacher labels (`movement-epoch2-v1`) | 1/30 in today's harness |

A structured RL bot trained in the same environment, `league96` (`low256-league-rivals-v1` update 96), wins 64/64 against the scripted teacher on final maps. **Question:** does supervised imitation of that RL bot, in exactly the prompt format the LLM is evaluated in, transfer its strength to the LLM?

## 2. Method

### 2.1 Expert data

- **Expert:** `league96`, deterministic, SHA `f998d15e…`. It plays through the public `GPUEnvironment` interface (`StructuredPolicy`), with no private state.
- **Games:** all 500 maps of the **training split** (seeds 100041–100540) × both seats, against scripted `balanced`. The expert won **999/1,000** (median round 20), producing 356,727 expert decisions in 8.3 minutes on CPU containers.
- **Rendering and labels:** each expert decision is rendered with `render_readable_decision`, the exact evaluation prompt. The label is the prompt-local `aN` of the expert's legal action, checked by a `parse_response` round trip. Target string: `{"action_id":"aN"}`.
- **Sampling:** up to 40 decisions per game (seed 17), giving 39,626 examples: about 370M tokens, mean 9,329 tokens.
  - Examples were capped at 24,576 tokens, and longer ones were skipped and counted, never truncated. The cap comes from smoke-test memory: 145 GB peak on a 23,664-token batch with the qualified no-checkpointing kernel.
  - There were no duplicates and no conflicting labels.
- **Move mix:** move 47%, build 10%, attack 10%, recruit 8%, end turn 6%, harvest 6%, city reward 6%, capture 3%, research 3%, ability 1%.

Code: [`expert_data.py`](../src/polytopia_bot/learning/expert_data.py), [`prepare_expert_sft.py`](../scripts/prepare_expert_sft.py).

### 2.2 Training

- **Recipe:** the SFT-0 LoRA recipe, unchanged:
  - rank 32, alpha 64, on q/k/v/o/gate/up/down;
  - learning rate 1e-4 with cosine decay and 3% warmup;
  - effective batch 64, 1 epoch, response-only loss.
- **Runtime:** the qualified packed "cast-free segmented SDPA" fast-trainer variant on 4 × B200.
- **Steps and time:** 620 updates in 6,997 s of training time. Loss fell from 1.90 at step 1 to about 0.25 by mid-run.
- **Integrity:** a mandatory longest-input stress test before the first update passed at 150 GB peak. The frozen base weights were unchanged and the adapter was saved.

Receipt: [`training-receipt.json`](expert-sft/training-receipt.json). Code: [`expert_trainer.py`](../src/polytopia_bot/learning/expert_trainer.py).

### 2.3 Evaluation harness (identical for every LLM arm)

- **Prompt and grammar:** readable named-field prompt plus the legal-action JSON grammar (xgrammar, from the first token). One call per move, no repairs, no token cap.
- **Sampling:** temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5. The per-decision seed is `17 + rng_index·100000 + decision`, identical across arms.
- **Serving:** vLLM 0.19.0 on H100 with **LoRA enabled for every arm**; the base arm simply sends no adapter. Adapters were hash-checked before play.
- **RL bot reference:** the RL bot itself, deterministic, on the same jobs.
- **Statistics:** wins, losses, round-limit stalls, forfeits and unfinished games are counted separately; unfinished games are never wins or losses. Uncertainty is a paired bootstrap clustered by map (20,000 resamples).

Code: [`benchmark_expert_sft.py`](../scripts/benchmark_expert_sft.py), [`modal_expert_sft.py`](../experiments/modal_expert_sft.py), [`modal_expert_heldout.py`](../experiments/modal_expert_heldout.py).

## 3. Results

### 3.1 Development maps (15 validation maps × 2 seats, vs `balanced`), run `expert-sft-v1`

| Arm | W | L | Median last round | Peak cities |
|---|---:|---:|---:|---:|
| Base | 0 | 30 | 21.5 | 1.0 |
| Old SFT (scripted-teacher labels) | 1 | 29 | 28.5 | 2.9 |
| **Expert-SFT** | **18** | 12 | 34.5 | 7.4 |
| RL bot (`league96`) | 28 | 2 | 19 | 8.3 |

- Expert-SFT − base: **+0.60** [0.40, 0.77], better on 12 of 15 maps and worse on 0.
- Expert-SFT − old SFT: +0.57 [0.40, 0.73].

### 3.2 Held-out test maps (30 maps × 2 seats × 3 opponents = 180 games per arm), run `expert-sft-heldout-v1`

- **Maps:** seeds 100741–100770, the first 30 of the reserved test split in seed order, never used for data, prompts or evaluation; overlap was checked in code.
- **Opponents:** `balanced` (the opponent in the training data), `pressure` (combat × 1.35) and `economy` (economy × 1.35). These are mild variants of the same scripted bot.
- **Stopped early:** the run was stopped at the user's request with **10 expert-SFT games still in progress** (1 against `balanced`, 9 against `economy`). They are reported as unfinished.

| Arm | Opponent | W | L | Unfinished | Median last round | Peak cities |
|---|---|---:|---:|---:|---:|---:|
| Base | all | **0** | 180 | 0 | 20.5 | 1.1 |
| **Expert-SFT** | `balanced` | 26 | 33 | 1 | 36 | 7.1 |
| **Expert-SFT** | `pressure` | 28 | 32 | 0 | 35.5 | 7.2 |
| **Expert-SFT** | `economy` | 23 | 28 | 9 | 33 | 7.1 |
| **Expert-SFT** | **all** | **77** | 93 | 10 | 35 | 7.1 |
| RL bot (`league96`) | all | 179 | 1 | 0 | 21 | — |

All resolved games ended natively: 169 eliminations plus 1 expert-SFT resignation. There were no forfeits and no round-limit stalls.

**Paired differences, clustered by map** (unfinished counted as non-wins, which is conservative):

| Comparison | Opponents | Difference | 95% interval | Maps better / worse |
|---|---|---:|---:|---:|
| Expert-SFT − base | all | **+0.43** | [0.37, 0.49] | **30 / 0** |
| | `balanced` | +0.43 | [0.35, 0.52] | 24 / 0 |
| | `pressure` | +0.47 | [0.35, 0.58] | 22 / 0 |
| | `economy` | +0.38 | [0.27, 0.50] | 19 / 0 |
| RL bot − expert-SFT | all | +0.57 | [0.50, 0.63] | 30 / 0 |
| RL bot − base | all | +0.99 | [0.98, 1.00] | 30 / 0 |

**Generalization:**
- **To new maps:** the gain carries over. The win rate drops from 60% (18/30) on development maps to 45% of resolved games on unseen maps. The development estimate came from only 30 games, so part of that drop may be sampling noise.
- **To new opponent styles:** the model does about as well against `pressure` (47%) and `economy` (45% of resolved games) as against `balanced` (44%).

### 3.3 Behavior (held-out, LLM model moves)

| | Base | Expert-SFT |
|---|---:|---:|
| Model moves (180 games) | 17,555 | 103,670 |
| End turn | 21.2% | 6.2% |
| Attack | 2.4% | 10.9% |
| Captures (villages and cities) | 36 | 2,053 |
| Enemy cities taken / own cities lost | 0 / 198 | **808** / 978 |
| Kills / units lost | 89 / 319 | 2,504 / 2,645 |
| Own units disbanded | **1,017** | **15** |
| Peak cities (mean) | 1.1 | 7.1 |

The base model barely leaves its capital and scraps its own army for stars. Expert-SFT reproduces the RL bot's style: many actions per turn, aggressive capturing, sustained fighting, almost no disbanding, and above all **taking enemy cities, which is how it finishes games.** It fights roughly even trades, with 2,504 kills to 2,645 losses. Its games are also longer than the RL bot's (median round 35 against 21), so it converts its leads more slowly.

### 3.4 Head-to-head against the latest RL bot (run `expert-sft-h2h-v1`)

- **Opponent:** `dist-c460div-v1` update 3142 (SHA `04cbb474…`), delivered 2026-10-19. It beat `league96` 117–10–1 on untouched final maps.
- **Games:** 5 fresh test-split maps (100771–100775) × both seats = 10 games, with the RL bot deterministic.
- **Reference:** `league96`, the bot expert-SFT imitated, against the same opponent on the same 10 games.

| Player vs u3142 | W | L | Unresolved | Last rounds |
|---|---:|---:|---:|---|
| **Expert-SFT** (LLM) | **0** | 10 | 0 | 19–31 |
| `league96` (its teacher) | 1 | 9 | 0 | 17–43 |

- **Expert-SFT's games:** peak cities ranged 1–6 (median 4). Across the 10 games it made 106 kills against 230 units lost, and took 1 enemy city against 40 lost.
- **The opponent's dominance:** the newer bot beats both the student and its teacher. The student's 0/10 is in line with the teacher's 1/10, so 10 games can't separate the two. As expected, imitating `league96` doesn't produce a player stronger than `league96`.
- **Implication:** to compete with u3142, the LLM would need to imitate u3142 itself, or go beyond imitation with DAgger or RL. See §7.

Summary: [`h2h-u3142-summary.json`](expert-sft/h2h-u3142-summary.json).

## 4. Cost

| Stage | Hardware | Time |
|---|---|---|
| Expert data, 1,000 games | CPU containers | 8.3 min wall |
| Tokenize | CPU | 5.5 min |
| Training | 4 × B200 | 2.0 h, about 8 B200-hours, roughly $50 at the published rate |
| Development evaluation | 3 × H100 | 15 / 32 / 95 min (base / old SFT / expert-SFT) |
| Held-out evaluation | 6 × H100 | base about 25 min per opponent; expert-SFT about 3 h per opponent (stopped) |

For comparison, the Qwen3.5 thinking arm used about 17.9 H100-hours for 30 unfinished games, and it won 2.

## 5. Integrity and incidents

- **Smoke tests before launch:** expert data (739 round-trip-checked labels), LoRA preflight in vLLM, and a 2-update smoke train on 4 B200s with stress and integrity checks.
- **Development-run summary bug:** the summary first crashed on JSON tuple keys. It was fixed and re-run without replaying any game, so that run's `collection.json` says `failed`; `summary-small.json` is authoritative.
- **Coordinator restart (held-out run):** Modal restarted the held-out coordinator mid-run at 09:33 PDT. The restarted copy spawned duplicate expert-SFT evaluations, which failed in 20 s on the guard against overwriting an arm, and it wrote a premature summary.
  - The original workers were unaffected and kept running.
  - The final summary was regenerated after the stop and overwrites the premature one.
  - The head-to-head launcher now records its calls and reattaches to them on restart.
- **No leakage:** the training data uses only training-split maps. The development evaluation used 15 validation maps and the held-out evaluation used test-split maps.

## 6. Limitations

- **Opponents:** the held-out test used scripted opponents only (three mild variants). The only learned-opponent test is the 10-game head-to-head (§3.4).
- **Sampling:** one LLM sampling sequence per game. Map clusters capture map variance, not sampling variance.
- **Unfinished games:** 10 held-out expert-SFT games were unfinished at the stop. Counting all 10 as wins would raise the expert-SFT total to 87/180; counting them as losses would leave it at 77.
- **Old SFT comparison:** the old SFT adapter was trained on a different, older prompt format, so its weak result partly reflects that mismatch.
- **Imitation gap:** expert-SFT still loses to the scripted bot about 55% of the time, where the RL bot almost never does. It plays the expert's style but imperfectly.

## 7. Next steps

1. **Imitate the current best RL bot** (u3142) instead of `league96`. In the head-to-head (§3.4), both expert-SFT (0/10) and `league96` (1/10) lost to it.
2. **DAgger:** have the RL bot relabel the states the SFT model actually reaches, and retrain. This targets the imitation gap.
3. **More or better data:** imitate the stronger `u3142` bot, include more games and opponents, and add the round-limit score as an auxiliary metric.
4. **RL fine-tuning** of the LLM, starting from this checkpoint.
5. **Confirmatory test:** another untouched test slice with several sampling seeds.
