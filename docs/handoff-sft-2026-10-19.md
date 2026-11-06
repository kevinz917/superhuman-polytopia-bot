# Handoff: LLM Polytopia bots via expert SFT (state as of 2026-10-19, 22:30 PDT)

This document lets a new thread pick up all LLM-bot work from Oct 18–19 without reading the old sessions. Everything below is on branch **`qwen35-thinking-continuation`**, pushed to GitHub. The trainer speedups from PR #1 have been merged into it.

## 1. Where things stand (one paragraph)

Letting a 4B LLM "think" is not a viable policy: about 18 H100-hours for 2 wins in 30 games. **Imitating a strong RL bot is.** `Qwen/Qwen3-4B-Instruct-2507` plus a LoRA trained for 2 h on 39.6k moves of the RL bot `league96` went from 0 to 45% wins on unseen maps against the scripted bots. The student copied its teacher exactly, including `league96`'s 99.7%-Warrior army, so it lost 0/10 to the newer, more diverse RL bot u3142.

The pipeline is now configurable, resumable and much cheaper:
- a compact prompt cuts tokens by 69% and trains 4.1× faster at equal strength, with one regression to guard;
- cuDNN attention gives +41% training speed;
- evaluation sharding is built.

**Next step:** a larger SFT run (about 160k examples) imitating the newest RL bot, using the template config in §5.

## 2. Results

| Experiment | Result | Doc |
|---|---|---|
| Qwen3.5-4B thinking on/off, 30 paired games vs scripted `balanced` | off 0/30. On: 2 W, 16 L, 1 loop failure, 11 unfinished (stopped after about 9 h). About 74.8M tokens, about 17.9 H100-h | `docs/qwen35-thinking-comparison-final-2026-10-18.md` |
| Qwen3-4B-Instruct-2507 bare, same games | 0/30 | `docs/qwen3-instruct-baseline-2026-10-18.md` |
| **Expert-SFT v1** (imitates `league96`), 30 dev games | **18/30** (base 0/30; old teacher-SFT 1/30) | `docs/expert-sft-result-2026-10-19.md` |
| Expert-SFT v1, 180 held-out games (30 test maps × 2 seats × balanced/pressure/economy) | **77 W / 93 L / 10 unfinished**. Base 0/180; RL bot 179/180. vs base: +0.43 [0.37, 0.49], better on 30/30 maps | `docs/expert-sft-research-report-2026-10-19.md` |
| Expert-SFT v1 vs u3142 (10 games) | **0/10** (`league96` itself 1/10). Lost to Catapults: 57% of u3142's kills | report §3.4 |
| Data scaling (v1 checkpoints at 25/50/75/100% of the data) | 9 / 9 / 14 / **18** wins, **still rising** | `docs/expert-sft-scaling-curve-2026-10-19.md` |
| Imitation accuracy (2,000 held-out expert positions) | **36% overall**: move 15%, build 19%, research 34%, attack 47%, recruit 52%, capture 83–90%, end turn 92% | `docs/compact-prompt-v2-2026-10-19.md` |
| Compact prompt v2.1 (full state every move) | 0.31× tokens, 64% prefix shared between moves (was 14%), 4.1× faster training. 34/60 vs v1 18/30, paired −0.03 [−0.23, +0.17]. **Regression:** demolishes its own buildings in about half its games (0 such labels in training) | `docs/compact-prompt-v2-2026-10-19.md` |
| Evaluation speed | Engine tuning: TTFT 7.0 → 2.5 s, per-move time only −8%. Sharding games across GPUs is linear | `docs/sft-eval-speedup-2026-10-19.md` |
| Trainer speed | About 16% MFU. **cuDNN attention +41%** (52.6k → 75.2k tok/s on 4 × B200), Liger SwiGLU +1.5%, H100 never cheaper | `docs/sft-trainer-benchmark-2026-10-19.md` |
| Pipeline | One JSON config, resumable stages, restart-safe coordinators, trainer resume from checkpoint (bit-identical step after resume). Smoke passed | `docs/sft-pipeline-config-2026-10-19.md` |

## 3. Lessons to apply

1. **The teacher defines the strategy.** Check the expert's unit mix and win rate in the generated data before training on it.
2. **The run is data-limited.** The win curve still rises at 40k examples and imitation is only 36%, so scale data (about 160k) and diversify the states (learned opponents, more maps).
3. **A prompt format can fail silently.** v2.0 lost about 8 wins in 30 to a representation bug. Always compare per-action-type frequencies against the training labels, and always confirm with games.
4. **Cheap proxy first:** held-out imitation accuracy takes about 72 s with the compact prompt; game evaluation takes 1–3 h.
5. **Cost is prompt length:** about 99.9% of compute reads the prompt to emit 8 tokens. The compact prompt, prefix caching and cuDNN stack.
6. **Expect platform failures.** Modal 4-hour limits and coordinator restarts both happened. Use the resumable pipeline, always launch with `--detach`, and launch with module form.
7. **Evaluate against learned opponents as well as scripted ones.** The scripted styles are mild variants of one bot.

## 4. Artifacts

**Modal** (env `staging`):

| What | Where |
|---|---|
| Volumes | `polytopia-sft0-training` (runs, adapters, HF cache), `polytopia-structured-rl-retained` (RL checkpoints) |
| Expert-SFT v1 adapter | `expert-sft-v1/training/model/adapter-final`. sha256: config `51de236e…`, safetensors `d79addf5…`. Record: `docs/expert-sft/checkpoint-expert-sft-v1.json` |
| v1 intermediate checkpoints | `expert-sft-v1/training/model/checkpoint-{157,314,471,620}` |
| Compact v2.1 adapter | `compact-prompt-v2.1/training/model/adapter-final` (see the compact-prompt doc) |
| RL bots | `league96`: `/retained/checkpoints/low256-league-rivals-v1/update-96-f998d15e220c/latest.npz` (sha `f998d15e…`). u3142: `/retained/checkpoints/distributed-final-v1/dist-c460div-v1-update-3142/final.npz` (sha `04cbb474…`) |
| Run IDs | `qwen35-thinking-30paired-v1`, `qwen3-instruct-2507-30paired-v2`, `expert-sft-v1`, `expert-sft-heldout-v1`, `expert-sft-h2h-v1`, `compact-prompt-v2`, `compact-prompt-v2.1`, `trainer-qualification/qualify-b200x4-1792452187`, `eval-speedup-v1`, `sftp-smoke-*` |

**Code map:**

| Piece | Files |
|---|---|
| Configurable pipeline | `experiments/modal_sft_pipeline.py`, `src/polytopia_bot/learning/sft_pipeline.py`, `scripts/sft_pipeline_{prepare,eval,summarize}.py`, `configs/sft-pipeline/*.json` |
| Expert data and trainer | `src/polytopia_bot/learning/expert_data.py`, `src/polytopia_bot/learning/expert_trainer.py` (resume + cuDNN/Liger/parity gates), `scripts/expert_train_distributed.py` |
| Training runtime | `src/polytopia_bot/learning/fast_training.py`, `src/polytopia_bot/learning/trainer_qualification.py`, `experiments/modal_qualify_expert_trainer.py` |
| Evaluation | `scripts/benchmark_expert_sft.py` (opt-in `serving` speedups, RL opponents), `experiments/modal_eval_shards.py` |
| Compact prompt | `src/polytopia_bot/learning/compact_prompt.py` (registered as `compact-v2` in the pipeline), `experiments/modal_compact_prompt.py` |
| Older one-off launchers | `experiments/modal_expert_sft.py`, `modal_expert_heldout.py`, `modal_expert_h2h.py`, `modal_thinking_continuation.py` |

**Local:** `runs/` (git-ignored) holds small receipts for every run.

## 5. How to run the next large SFT run

1. **Pick the expert.** Use the newest RL bot from the RL thread. The `external-website` branch serves `c3-maps-ctl` update 4854, described there as beating u3142 67–75%. Add it under `players` in a copy of `configs/sft-pipeline/next-expert-160k.json` (path under `/retained/`, plus sha256), and list it in `generation.experts` and `evaluation.rl_reference`.
2. **Prompt.** `prompt.renderer`: `compact-v2` (recommended) or `readable-v1` (safe).
   - **Before using compact:** implement and verify the demolish guard, which removes the `demolish_own` ability from the model's action catalog in training and inference alike. The compact-prompt doc §5 has a 1-hour evaluation-only check. Not done yet.
3. **Trainer.** The default is the strict-gated runtime. For cuDNN (+41%):
   - **The configurable pipeline doesn't expose it yet.** `modal_sft_pipeline.py` always trains with the old benchmark receipt and the strict gate. It needs two small config fields, `training.benchmark_receipt` and `training.parity_gate`, passed into `training/input.json` (`expert_train_distributed.py` already reads `parity_gate`).
   - Only the older `experiments/modal_expert_sft.py` launcher accepts `--benchmark-receipt` and `--parity-gate revised` today.
   - See `docs/sft-trainer-benchmark-2026-10-19.md` §7.
   - It's qualified on 4 × B200 only. Re-qualify first if using `gpus: 8`: `experiments.modal_qualify_expert_trainer`.
   - No full run or game evaluation has used cuDNN yet. A cheap check: re-train the compact equal-data run with cuDNN (about 20 min) and evaluate it on 30 games (about 1 h).
4. **Estimate, then launch:**
   ```bash
   ~/.local/share/uv/tools/modal/bin/python -m experiments.modal_sft_pipeline estimate configs/sft-pipeline/NAME.json
   modal run --detach --env staging -m experiments.modal_sft_pipeline --action smoke --config configs/sft-pipeline/smoke.json --run-id sftp-smoke-N
   modal run --detach --env staging -m experiments.modal_sft_pipeline --action launch --config configs/sft-pipeline/NAME.json --run-id RUN
   modal run --env staging -m experiments.modal_sft_pipeline --action status --run-id RUN
   modal run --detach --env staging -m experiments.modal_sft_pipeline --action resume --run-id RUN [--retry STAGE]
   ```
5. **Expected time and cost** for 160k examples (estimator plus measurements; generation about 25 min / $11, prepare about 15 min / $1.50):

| Setup | Training | Training cost | Whole run |
|---|---|---:|---|
| Readable, 4 × B200 | about 8 h | about $200 | about 10–11 h, about $230 |
| Readable, cuDNN, 8 × B200 | about 3 h | about $150 | — |
| Compact v2.1, 4 × B200 | about 2 h | about $45 | about 3.5–4 h, about $60–80 |
| Compact + cuDNN | about 45 min | about $40 | about 2.5 h |

   Evaluation is 20 fresh test maps (seeds 100776–100795) × 2 seats vs `balanced` and `league96`, plus head-to-head vs the RL bots, sharded with prefix caching: about 45–120 min.
6. **Gates while it runs:**
   - after generation: expert win rate and **unit mix**;
   - after prepare: no conflicting labels, a sane action-type mix, and few drops at the token cap;
   - after training: held-out imitation accuracy and the action-frequency check;
   - then games.
7. **What it should show:**
   - **H1:** a better teacher gives a better student (beats v1; unit mix changes);
   - **H2:** more data helps (checkpoints every 500 steps give a 160k-scale curve);
   - **H3:** varied positions help against learned opponents (wins vs `league96` and u3142);
   - **H4:** compact holds strength at about ⅓ cost;
   - **H5:** the run finishes unattended near the estimate.

   It can't attribute gains between expert, data and opponents, because all change at once. The v1 arm and the mid-run checkpoints are the partial controls.

## 6. Open items

1. **Demolish guard for compact v2.1:** not implemented. The compact-prompt session proposed it as a separate task.
2. **cuDNN decision:** accept the revised parity gate? If so:
   - add `benchmark_receipt` and `parity_gate` to the pipeline config (§5.3);
   - re-qualify on 8 × B200;
   - run the end-to-end check in §5.3.
3. **PRs:** PR #1 (`claude/compassionate-kirch-61e470`) is merged into this branch but still open on GitHub against `main`. Close it as superseded or merge it. No PR exists yet from `qwen35-thinking-continuation` to `main`.
   - This branch is 15+ commits ahead of `main` and also carries other threads' commits (Studio, map variants, distributed RL). Four of the RL commits already exist on `main` under other hashes.
4. **Resume after the PR #1 merge:** the trainer merge (resume + new runtime) passed the 38 trainer tests and the pipeline smoke on Modal. The kill-and-resume test (`configs/sft-pipeline/resume-test.json`) was not repeated after the merge.
5. **Uncommitted files from other threads** are in this checkout: Studio/web UI and `.claude/launch.json`. They were left untouched.

## 7. Rules carried over from the user

- All model inference, training, simulation and full-trace analysis runs on Modal, never locally. Never download checkpoints locally.
- Never stop Modal apps you didn't create. Many unrelated RL jobs share `staging`.
- Unfinished, timed-out or infrastructure-failed games are never counted as wins or losses.
- Use the Modal SDK python `~/.local/share/uv/tools/modal/bin/python` and the CLI `~/.local/bin/modal`.
