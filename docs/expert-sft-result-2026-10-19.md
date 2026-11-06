# Expert-SFT: imitating the structured RL bot (2026-10-19)

> Full research report, including the 180-game held-out test: [expert-sft-research-report-2026-10-19.md](expert-sft-research-report-2026-10-19.md).

**Result: SFT on the RL bot's moves turned Qwen3-4B-Instruct-2507 from a 0/30 player into an 18/30 player** on the same 30 games against the scripted `balanced` opponent, in the same harness. The win difference against the base model, clustered by map, is **+0.60** (bootstrap 95% interval [0.40, 0.77]). The expert-SFT model is better on 12 of 15 maps and worse on none. The earlier teacher-imitation SFT (`movement-epoch2-v1`) won only 1/30.

Run `expert-sft-v1`, all on Modal (env `staging`). Protocol: `runs/expert-sft-v1/protocol.json` (SHA256 `4c98e7df…132e7c`). Summary: `runs/expert-sft-v1/summary-small.json`.

## Results (same 30 games: 15 validation maps × 2 seats, vs `balanced`)

| Arm | Wins | Losses | Stalled / failed | Median last round | Peak cities (mean) |
|---|---:|---:|---:|---:|---:|
| Base Qwen3-4B-Instruct-2507 | 0 | 30 | 0 | 21.5 | 1.0 |
| Old SFT (`movement-epoch2-v1`, teacher imitation) | 1 | 29 | 0 | 28.5 | 2.9 |
| **Expert-SFT (new)** | **18** | 12 | 0 | 34.5 | **7.4** |
| RL bot (the expert itself, reference) | 28 | 2 | 0 | 19 | 8.3 |

Every game in every arm ended natively by elimination. There were no forfeits, no round-limit stalls and no operational failures.

**Paired, map-clustered win differences** (20,000 bootstrap samples over 15 maps):

| Comparison | Difference | 95% interval | Maps better / worse |
|---|---:|---:|---:|
| Expert-SFT − base | **+0.60** | [0.40, 0.77] | 12 / 0 |
| Expert-SFT − old SFT | +0.57 | [0.40, 0.73] | 12 / 0 |
| Old SFT − base | +0.03 | [0, 0.10] | 1 / 0 |
| RL bot − base | +0.93 | [0.83, 1.00] | 15 / 0 |

The expert-SFT model closes about two thirds of the gap between the base model and the RL bot (18 of 28 wins).

## How it plays (LLM arms, model moves only)

| | Base | Old SFT | Expert-SFT |
|---|---:|---:|---:|
| Model moves (30 games) | 3,023 | 8,025 | 17,898 |
| End turn | 21.5% | 11.0% | **6.1%** |
| Attack | 2.7% | 8.3% | **11.0%** |
| Captures (villages and cities) | 3 | 114 | **365** |
| Enemy cities taken / own cities lost | 0 / 31 | 13 / 91 | **140** / 145 |
| Kills / units lost | 14 / 53 | 157 / 212 | **431** / 400 |
| Own units disbanded | **168** | 119 | **4** |

- **Base model:** it barely expands (3 captures), disbands its own units (168) and ends turns early.
- **Old SFT (teacher imitation):** it expands and fights somewhat more, but still disbands units and loses 29/30.
- **Expert-SFT:** it plays like the RL bot. It takes many actions per turn, captures aggressively, fights, almost never disbands, and **takes enemy cities (140), which is how it finishes games.** None of the earlier LLM arms did that.
- **Game length:** its games run longer (median round 34.5) than the RL bot's (19), so it finishes opponents more slowly.

## What ran

- **Expert:** structured RL bot `low256-league-rivals-v1` update 96, deterministic, SHA `f998d15e…6225d`, from `polytopia-structured-rl-retained`. It plays through the public `GPUEnvironment` interface.
- **Data:**
  - The expert played all 500 **training** maps (seeds 100041–100540) × 2 seats against `balanced`. It won 999 of 1,000 games (median round 20), producing 356,727 expert decisions in 8.3 minutes.
  - Each decision was rendered with the evaluation prompt (`render_readable_decision`, named fields). The label is the prompt-local `aN` of the expert's legal action, checked by a parse round trip.
  - Up to 40 decisions were sampled per game (seed 17), giving **39,626 examples**: about 370M tokens, mean 9,329 tokens.
  - Examples were capped at 24,576 tokens: longer ones are skipped and counted, never truncated. There were no duplicates and no conflicting labels.
  - Move mix: move 18,748, build 4,046, attack 3,893, recruit 3,271, end turn 2,402, harvest 2,314, city reward 2,264, capture 1,214, research 1,066, ability 408.
  - Validation and test maps were never used for data.
- **Training:**
  - Recipe: the SFT-0 LoRA recipe (rank 32, alpha 64, all attention and MLP projections; learning rate 1e-4 cosine with 3% warmup; effective batch 64; 1 pass; response-only loss). It used the qualified packed fast-trainer variant on 4 B200s, with a mandatory longest-input stress test, which passed at 150 GB peak.
  - Steps: 620 updates. Loss fell from 1.90 to about 0.25.
  - Time: 1.94 h of training time (2.0 h occupied).
  - Integrity: the base weights were unchanged and the adapter was saved.
- **Evaluation:**
  - Games: the same 30 jobs as `qwen3-instruct-2507-30paired-v2` and `qwen35-thinking-30paired-v1`.
  - Harness: the readable-prompt plus legal-action-grammar harness, with the same sampling (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5), per-decision seeds and one call per move with no repairs.
  - Serving: all three LLM arms ran an identical LoRA-enabled vLLM 0.19.0 engine on one H100 each; the base arm sends no adapter. Every adapter was hash-checked before play.
  - RL reference: the RL bot itself, deterministic, on the same games, using `scripts/evaluate_structured_bot.py`.

## Cost (resource-rate estimates, not billing)

| Stage | Hardware | Time |
|---|---|---|
| Expert data | CPU containers | about 8 min wall |
| Tokenize | CPU | about 5.5 min |
| Training | 4 × B200 | 2.0 h, about 8 B200-hours |
| Evaluation | 3 × H100 | base 15 min, old SFT 32 min, expert-SFT about 95 min |
| RL reference | CPU | 12 min |

Training is the dominant cost. At the ledger's published B200 rate it's roughly $50. Container startup, image builds and storage are excluded.

## Pipeline checks before launch

All on Modal:
- **Expert-data smoke test:** 2 games, 739 round-trip-checked labels.
- **LoRA preflight:** the old SFT adapter loaded through vLLM and produced 4 valid 8-token answers.
- **Smoke train:** 2 updates on 4 B200s, stress test at a 23,664-token input, integrity checks passed.

The coordinator's summary stage first failed on a JSON key bug in the summary script, after every game had finished. It was fixed and re-run with `--action summarize`, without replaying any game. The coordinator's `collection.json` therefore records `failed`. `summary-small.json` is the authoritative result.

Code (new files): [expert_data.py](../src/polytopia_bot/learning/expert_data.py), [expert_trainer.py](../src/polytopia_bot/learning/expert_trainer.py), [prepare_expert_sft.py](../scripts/prepare_expert_sft.py), [expert_train_distributed.py](../scripts/expert_train_distributed.py), [benchmark_expert_sft.py](../scripts/benchmark_expert_sft.py), [summarize_expert_sft.py](../scripts/summarize_expert_sft.py), [modal_expert_sft.py](../experiments/modal_expert_sft.py).

## Caveats

- 30 games on 15 validation maps, with one sampling sequence and one opponent. The interval reflects map resampling of this run only.
- The old SFT adapter was trained on a different, older prompt format. Its weak result partly reflects that mismatch, not just teacher-quality labels.
- The expert data used training maps against the same opponent style as the evaluation (`balanced`). Strength against other opponents, including learned ones, is untested.
- The reserved test split (seeds 100741–101240) was not used and remains available for a confirmatory run.
- The expert-SFT model loses 12/30 where the RL bot loses 2/30, so imitation is incomplete. Natural next steps: more data, DAgger-style relabeling of the SFT model's own states by the expert, or RL from this checkpoint.
