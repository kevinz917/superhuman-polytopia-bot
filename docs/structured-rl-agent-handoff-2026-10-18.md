# Polytopia structured RL campaign: agent handoff

Prepared October 18, 2026, from local campaign records observed through approximately 10:19 a.m. PDT (17:19 UTC). This is a handoff snapshot, not a live status page. The main agent may continue updating the repository and cloud jobs. Reconcile current records and ownership before acting.

Repository: `/Users/K.Z./Desktop/projects/polytopia-bot`.

This document was produced in a side conversation. Preparing it did not launch, stop, restart, or modify training/evaluation jobs, change the budget ledger, promote a checkpoint, or consume final-test maps. Only this handoff document was created. Cloud state below comes from saved observations; the handoff author did not independently poll Modal.

## 1. Objective and authorization inherited by the next agent

Build a strong 1v1 reinforcement-learning Polytopia bot alongside the existing language-model bot, sharing the simulator and public environment interface. The user asked for Duper-inspired PPO with structured action heads, demonstration bootstrap if useful, and evolving self-play. The campaign plan records a total **$500 Modal ceiling** and subsequent direction to use the available budget toward the best bot possible. Passing minimum acceptance gates is a milestone, not the entire optimization objective.

Latest training direction: **new allocations use learned opponents only**, including current and historical neural policies and optionally immutable learned rivals. Scripted opponents remain evaluation benchmarks. Existing mixed-opponent jobs are controls and potential initialization sources. Strong current candidates have imitation and mixed-PPO ancestry; disclose that ancestry. Learned-only continuation does not make them from-scratch pure RL.

The user asked whether this resembles Flagship Neuro's async PPO. Current implementation is synchronous collect/optimize PPO with independently advancing game lanes. It does not overlap actor collection with learner optimization. True distributed async PPO is not implemented; do not imply otherwise. The Neuro reference publishes latest weights periodically, every four trainer iterations, rather than only publishing evaluation-approved champions.

Keep Duper/Curio/Flagship references read-only. Do not interfere with concurrent Qwen/LLM jobs or charge them to this campaign. Preserve unrelated dirty files. Do not infer authorization to delete retained historical bundles, change simulator mechanics, or exceed the budget.

## 2. Read these records first

All paths in this document are repository-relative unless explicitly absolute or described as cloud paths.

| Record | Purpose |
|---|---|
| `docs/structured-rl-current.md` | Current architecture, candidate, usage and acceptance status |
| `docs/structured-rl-repair-plan.md` | Authorized objective, budget, experiments, final-test contract |
| `docs/structured-rl-repair-progress.md` | Chronological evidence and decisions; newest entries supersede older status |
| `runs/structured-ppo-v3/budget.json` | Authoritative run allocations, app/call IDs, terminal accounting |
| `runs/structured-ppo-v3/current-development-candidate.json` | Selected incumbent, checkpoint hash and evidence |
| `docs/cloud-comparisons-2026-10-18.md` | Cloud-only inference/retention migration and preserved local artifacts |
| `docs/ruleset-v1.md`, `configs/ruleset-v1.json` | Frozen simulator contract and source hashes |
| `docs/structured-rl-postmortem-2026-10-17.md` | Early failure analysis; distinguish old bugs from repaired current code |

At handoff inspection, pre-existing tracked modifications were `docs/qwen35-current-benchmark.md`, `docs/structured-rl-current.md`, `docs/structured-rl-repair-progress.md`, and `experiments/modal_thinking_parallel_games.py`. Do not reset/stash/overwrite these as cleanup. Inspect current git status again because the main thread is active.

## 3. Implemented system and correctness boundaries

**Simulator:** frozen `bardur-1v1-v1`, with 14×14 maps for current training and acceptance. It is a research ruleset, not a claim of complete commercial Polytopia parity. The handoff author reran `python3 scripts/verify_ruleset.py`: all 22 source hashes matched, validation passed. Preserve the ruleset and map generator. Training starts around seed 200000; development partitions are declared separately below 400000.

**Policy:** `normalized-v2`, 519,267 parameters, 64 channels, two normalized residual blocks, SiLU, mean/max spatial pooling and a separate critic projection. Masked autoregressive action-family, source and target/parameter decisions select from the public legal catalog. Voluntary resignation is masked by policy configuration; other legal engine actions remain available. No scripted action fallback or search is inserted at inference. Both actor and critic use public observations, not privileged hidden opponent state.

**PPO:** current large runs use 512 lanes × 64 learner decisions = 32,768 decisions/update, two optimization epochs, minibatches of 256 and a 2,048-map training pool. Standard comparisons use learning rates 1e-4 and 3e-5, gamma .995, entropy coefficient .005. Gamma discounts learner transitions, not rounds. Learner transitions span opponent responses. Lanes advance independently, but collection and optimization are sequential.

**Self-play:** ordinary learned-only runs sample current and historical policies 50/50. History refreshes every 16 PPO updates and retains four snapshots. Opponent selection occurs per rollout, so an ongoing game's opponent can change across rollouts. Training win counts grouped by rollout opponent are therefore not isolated head-to-head strength estimates. Frozen-rival pilots use 25% immutable learned rivals and 37.5% each current/history. Continuations restore actor, critic and Adam; they do not restore all live episode state, RNG state and historical pool. Matching parent and seed is not evidence of bitwise-identical training: short and long lower-rate pilots diverged numerically after their first update.

**Reward:** native resolved win +1 / loss -1. Earlier `own-potential-v1` experiments are separate, explicitly shaped branches. They must not be conflated with current native-outcome results or reported as native victories.

**Repairs already implemented:** correct joint entropy gradients, normalized features, separate critic projection, advantage-normalization divisor floor .05, and behavior-log-probability replay checks. Actor-sized replay addresses measured batch-shape roundoff without relaxing the 1e-5 mismatch threshold. Nonfinite parameters or genuine log-probability mismatches fail the run. Do not reintroduce old architecture/reward work merely because an early report describes a failure.

**Truncation:** an engine safety limit ends collection without a native winner. It is not a draw or a win. Distinguish it from an evaluator wall-time cutoff. Keep unresolved games in the requested-game denominator and report them separately. A worker's `complete: true` can mean all requested candidates were evaluated even when a game truncated. Earlier pure RL suffered weak exploration/expansion and recruit/disband cycling; tactical learning and finite losses alone did not establish full-game competence.

## 4. Selected incumbent and strongest established references

**Incumbent remains champion-512 update 96**, deterministic inference, with teacher-imitation + native-outcome PPO ancestry. It is the selected development benchmark, not necessarily the opponent in each self-play rollout.

Cloud checkpoint: `/training/champion51296-05c414466823/final.npz`.

SHA-256: `05c41446682389972373e1478ed1ed4e139262a3c69c54f17964ef94b78ce5f3`.

Qualified development evidence: teacher **63/64**, random **32/32**, common learned references **70/96**, previous update-12 incumbent beaten **46–18/64**. Source/checkpoint identities and 256 benchmark trace hashes were checked; representative win/loss replay covered 3,714 actions. `champion51296-qualification.json` and the current-candidate record hold the evidence. No final acceptance has passed.

**Low144 is a qualified comparison reference, not the incumbent or final delivery.** Cloud bundle `/training/h25-low144-8150ea3b8468/final.npz`; SHA `8150ea3b8468771c4ebf552158fbda5cfc38f82db00066e68ae6a758fae126b1`. Consolidated evidence: 448 audited development games, teacher **63/64**, random **32/32**, common references **80/96**, fresh incumbent replication **46–18**. Deterministic beat stochastic **25–7**. Its latest training phase included scripted opponents. See `h25-low144-qualified-reference.json`.

**Low176 is a promising mixed-trained parent, not a promoted champion.** Bundle `/training/h25-low176-565eb6e315c8/final.npz`; SHA `565eb6e315c82b970c0d0694f448aaf5a798de26c41deb7ad9b5436de4489052`. Historical GPU screen scored **54–10** against incumbent96. Broader public qualification and its learned-only continuation were pending at the saved observation.

**Short incumbent-origin learned-only update16** has SHA `808f05cc623638ecbd8a4739d75f1f64c281fdd49943ff77ff74eede6cc3bbe5`. It completed 524,288 phase decisions with no new imitation labels and current/history opponents only. Public results include teacher **63/64**, random **32/32**, parent96 **43–21**, mixed low32 **24–40**, common native96 **23–9**, common gamma **25–7**, stochastic-vs-deterministic **9–23**. Low32 had twice as many phase decisions, so that loss is not a sample-matched recipe comparison. Its no-entropy common-reference job was still pending in the snapshot. The independent public audit had covered 256 games; newer result rows require audit expansion.

The separate from-scratch pure-RL control remains weak: `pure-async32-aa8505267dcd` scored 5/32 teacher wins. A later continuation scored 1 win / 30 losses / 1 truncation at update128 and 0/8 at update144 and was stopped. Its existence should be reported, but the strong line is bootstrapped PPO followed by learned-only self-play.

## 5. Most recent completed screens and the pending decision

Historical GPU screen `gpu-historical-dev301000-v1`: 12 candidates × 64 games, **768 audited games; 767 native and 1 unresolved engine truncation**. Stage1 `gpu-selfplay-stage1-dev300800-v1`: 10 × 64, **640 audited native games, zero truncations/cutoffs**. Both have complete independent JSON identity/coverage/outcome/bootstrap audits and stopped/tasks=0 receipts. GPU trace hashes are cloud receipts, not locally recomputed neural action selections. These are development screens, separate from public-interface qualification and final acceptance.

Historical leaders against incumbent96 on paired maps 301000–301031: low176 54–10, low192 50–14, low128 48–16, low160 46–18. Low144 scored 38–26 on that partition; its different scores on other map partitions are not contradictions. The old noentropy816 candidate had **33 wins / 30 losses / 1 unresolved**, not 33/63: seed301017, seat2, 1,748 actions, reward0, truncated=true and terminated=false.

Stage1 on 300800–300831: low144 parent 47–17; control8 47–17, rivals8 46–18, control16 42–22, rivals16 44–20; incumbent-origin short slow16 35–29, long slow16 36–28, fast16 34–30. Eight-update vs sixteen-update paired-map differences include zero in their confidence intervals. Retaining early candidates is sensible; claiming established regression is not.

**Stage2 is already dispatched; do not launch it again.** Run `gpu-selfplay-stage2-dev301400-v1`, app `ap-kQ8jCp5QYxmhsigm0rfEb0`, call `fc-01M5807DE52VQE9ATYNYWM4FFH`. It was pending at 10:18 a.m. PDT. Its frozen plan covers 11 candidates × 64 = **704 games**, on fresh paired maps 301400–301431, against fixed short self16 SHA808f.... It includes low176 parent, low176 self8/16, low144 control/rivals8/16, incumbent-origin long32/48, fast16 and long16. Candidate/model metadata and source hashes were frozen before outcomes. All candidates except the explicitly labeled mixed parent control are learned-only descendants.

Use `gpu-selfplay-stage2-dev301400-v1-plan.json` as the exact roster authority. The original nine-candidate intent was versioned to eleven before any new-map results; do not change the roster after seeing outcomes. The helper `prepare-stage2.py` refuses an existing plan/allocation and should not be rerun. A detached CLI log timeout or “App completed” message is not proof that the durable remote function finished.

## 6. Budget and resource accounting

At the inspected ledger snapshot: **$460.37 reserved/accounted; $39.63 headroom** under $500. This includes pending allocations; it is not actual spend. Both completed GPU screens were reconciled to terminal bounds ($2.56 historical, $2.33 stage1), releasing $5.11 against their original $10 reservations; stage2 holds $5.

Latest inspected provider report, `modal-billing-owned-20261018T170112Z.json`, reports **$271.65229396** for 79 campaign-owned apps through **10:00 a.m. PDT (17:00 UTC)**. It does not include the partial current hour or imply that the difference from $500 is uncommitted. Storage/egress are not included in that compute report. The ledger includes a shared overhead allocation. `spend-observation-current.json` is misleadingly named: the inspected copy was old (03:16 a.m. PDT); use timestamps and newer billing/ledger records.

Recorded resource assumptions, not a newly fetched price quote: L40S + 8 CPU + 32 GiB ≈ $2.584224/hour; public CPU qualifier 6 CPU + 16 GiB ≈ $.42/hour; small retainer 1 CPU + 2 GiB ≈ $.07/hour. Hard caps, requested runtime, terminal app-wall estimates and provider charges are distinct. Reserve before dispatch; release only against confirmed terminal state/tasks=0 and the accounting procedure. Keep capacity for final qualification and delivery. Do not double-count dispatch placeholders or interpret a released reservation as a refund.

The stopped H25 standard mixed run `scale-champion96-ratecontrol-v1` has a $13.39 terminal bound; update208 remains retained. It was stopped to prioritize lower-rate/learned-only work, not because every standard-rate checkpoint was proven inferior. Do not restart it or other deliberately stopped old branches.

## 7. Operating the existing cloud work

Modal environment is **staging**. CLI: `/Users/K.Z./.local/bin/modal`. Python with Modal SDK: `/Users/K.Z./.local/share/uv/tools/modal/bin/python`. Repository `.venv/bin/python` has JAX/pytest but not Modal.

Volumes: `polytopia-structured-rl` mounted at `/training`; `polytopia-structured-rl-retained` at `/retained`. Retained checkpoints normally live under `/retained/checkpoints/<run>/update-<n>-<12-character-sha>/latest.npz`, with adjacent JSON metadata. Public qualifier results live below `/retained/qualifications/<run>/`; use published manifests for exact paths rather than repeatedly listing entire volumes.

All new checkpoint copies, model loading, games, evaluation and replay inference run in Modal. Local work may inspect code and small JSON/provenance reports. Bulk local checkpoint caches were removed after upload/hash verification. Historical qualified-parent bundles were intentionally preserved; their local presence is not a migration failure or authorization to delete them.

For an initial non-mutating refresh, inspect the ledger and poll its existing function handles. This does not launch work:

```bash
cd /Users/K.Z./Desktop/projects/polytopia-bot
MODAL_ENVIRONMENT=staging /Users/K.Z./.local/share/uv/tools/modal/bin/python - <<'PY'
import json
from pathlib import Path
import modal
ledger = json.loads(Path('runs/structured-ppo-v3/budget.json').read_text())
for row in ledger['runs']:
    if row.get('status') != 'submitted' or not row.get('function_call_id'):
        continue
    try:
        result = modal.FunctionCall.from_id(row['function_call_id']).get(timeout=0)
        print(row['run_id'], 'returned', type(result).__name__)
    except TimeoutError:
        print(row['run_id'], 'pending')
    except Exception as exc:
        print(row['run_id'], type(exc).__name__, str(exc)[:300])
PY
```

Then check fresh app/task receipts before terminal accounting. `scripts/poll_structured_run.py` is **not read-only**: when a call returns, it writes a local summary and changes ledger status. It also assumes a training-shaped result, so inspect applicability before using it for evaluation workers. Do not equate a polling exception or missing old manifest entry with remote failure. Retainers should be checked separately from producers; they retain snapshots but do not control training.

The cloud comparison service `cloud-screens-20261018` already retains/small-screens the older 15-run roster; do not duplicate it. Its teacher/incumbent screens are eight-game selection probes, not broad qualification. New learned-only producers have independent retainers. Status paths include `checkpoints/watch-cloud-status.json`, `checkpoints/match-watch-cloud-champion12-status.json`, and `checkpoints/match-watch-cloud-champion96-status.json` on the retained volume.

## 8. Final acceptance and delivery still required

No final partition had been reserved or consumed at the inspected snapshot. Development seeds must remain below 400000. The initial final partition is **400000–400031**, 32 maps with both seats, independently yielding 64 teacher games and 64 random games.

Before running it, select one qualified learned-only self-play descendant and freeze checkpoint SHA, adjacent config, training protocol/ancestry, inference mode, sampling seed, runtime versions and the evaluator/policy/host/simulator source set (39 files). Use the existing explicit `final_acceptance` plan and worker, not an ad hoc development invocation. The plan reserves the partition atomically in the ledger and claims it in cloud storage before gameplay. A failed/interrupted test does not silently reopen those maps for selection.

Minimum native-win thresholds are **50% teacher (32/64)** and **80% random (at least 52/64)**. Unresolved games remain in the denominator; wall-time cutoffs make the evaluation incomplete. Report engine truncations separately, confidence intervals, expansion/action diagnostics, verified trace identities and representative complete win/loss replays. The acceptance module distinguishes integrity from strength. A gate pass does not prove broad commercial-game strength or finish full-budget optimization automatically.

Relevant implementation: `scripts/structured_qualification_plan.py`, `experiments/modal_structured_qualification.py`, `src/polytopia_bot/rl/acceptance.py`, `scripts/summarize_structured_acceptance.py`. Relevant tests include `tests/test_structured_qualification_plan.py`, `tests/test_structured_acceptance.py`, `tests/test_structured_budget.py`, and replay tests. Prior work recorded a passing targeted test set; this handoff only reran the frozen-rules hash verifier, not the full tests. Run checks appropriate to any new code changes.

Final deliverable should identify an immutable cloud checkpoint/config/source archive, demonstrate `StructuredPolicy` working through `GPUEnvironment`, attach the acceptance and replay evidence, document ancestry and remaining limitations, summarize actual spend versus reservations, and close out or explicitly account for campaign jobs. Update the current candidate/docs only after justified selection. Preserve useful earlier checkpoints and unrelated work.

## 9. Recommended continuation sequence

1. Establish that this agent owns the campaign; avoid concurrent ledger writes/dispatches with the main thread. Read newest progress, ledger and dispatch records. Refresh existing calls and app states, not launch commands.
2. Collect stage2 and pending public qualifiers. Complete independent audits of new result rows: exact checkpoint/metadata/source identity, map/seat coverage, modes, native outcomes, unresolved denominators and replay receipts. The prior two GPU screens are already audited; do not redo them without cause.
3. Check health and retained checkpoints of both learned-only long producers. Low176 had reached update16/524,288 decisions with finite metrics, 736 native training completions and zero illegal/resign/truncation outcomes; incumbent-origin long update48 was retained. These are last observed milestones, not guaranteed current progress or strength.
4. Use stage2 to choose promising learned-only candidates for public-interface replication. Compare direct learned-opponent performance, parent performance, broader references and mode behavior; teacher scores are near ceiling. Use fresh predeclared development maps, report paired-map uncertainty, and do not make causal claims across unequal ancestry/sample counts.
5. Reconcile completed allocations, refresh owned provider spend, and use remaining budget for evidence-backed learned-only improvement while retaining final-test capacity. Existing mixed controls may finish or be stopped deliberately under the campaign's authorization; do not launch new mixed training. Never spend merely because a reservation was released.
6. Freeze the selected candidate and protocol, execute untouched final acceptance once, inspect complete evidence, package the bot and report outcome/spend/limitations. If further iteration follows a failed final test, use a new untouched partition.

## 10. Code map

| Area | Files |
|---|---|
| Network/actions/observations | `src/polytopia_bot/rl/network.py`, `actions.py`, `observation.py` |
| Host inference | `src/polytopia_bot/rl/policy.py`; shared `src/polytopia_bot/learning/adapter.py` |
| PPO and collection | `src/polytopia_bot/rl/ppo.py`, `league.py`, `async_league.py`, `runner.py` |
| Curricula/bootstrap/reward | `src/polytopia_bot/rl/curriculum.py`, `bootstrap.py`, `reward.py`, `repair.py` |
| Budget/provenance | `src/polytopia_bot/rl/budget.py`, `replay_audit.py` |
| Long runs/pilots | `experiments/modal_structured_scale.py`, `modal_structured_repair.py` |
| GPU screens/public qualification | `experiments/modal_structured_arena_screen.py`, `modal_structured_qualification.py` |
| Retention/bundles | `experiments/modal_structured_retainer.py`, `modal_structured_comparisons.py`, `modal_structured_bundle.py` |
| Public evaluators/replay | `scripts/evaluate_structured_bot.py`, `evaluate_structured_match.py`, `verify_structured_replays.py` |

Names containing `async` refer to the implemented collection arrangement and do not prove distributed asynchronous actor/learner training. Inspect behavior, not filenames. Similarly, a `teacher_probability` field alone does not establish imitation in a `mode='ppo'` run; inspect the actual protocol, BC coefficient and public/random opponent probabilities.

The appendices below are extracted from local JSON records when this document was written. Their timestamps and hashes identify the snapshot; refresh their source records before operating cloud resources.

## Appendix A. Ledger snapshot and existing durable handles

Extraction time: 2026-10-18T17:20:33.276483+00:00. Ledger total: $460.37. `submitted` is a saved ledger state, not a fresh cloud liveness check.

| Run | App | Function call | Reservation | Hard timeout |
|---|---|---|---:|---:|
| `scale-champion512-lowrate-v1` | `ap-028SEQ2HDzAVwfsGDu4ArB` | `fc-01M5762R9RZJS491BVQHBV26GT` | $35.00 | 43200 s |
| `scale-champion512-lowrate-frozen-v1` | `ap-qKuhaNCASdVYAttlNeBrLe` | `fc-01M577Y4172NPW6V2G955Z8FNR` | $35.00 | 43200 s |
| `scale-champion512-lowrate-seed2-v1` | `ap-GxmhzPSd38YiE7mmfs1wA3` | `fc-01M578PC3692E7R6Y6KEW99YZ0` | $35.00 | 43200 s |
| `scale-champion512-lowrate-noentropy-v1` | `ap-zZIBkEXWycJdyfNeUdMk5I` | `fc-01M579GY280AQHFDBPN36NYQ8N` | $35.00 | 43200 s |
| `scale-champion96-lowrate-v1` | `ap-tZitdTqDCE61XdmtkGhR3C` | `fc-01M57F18W68W05XNMC6F2FM3A3` | $35.00 | 43200 s |
| `cloud-screens-20261018` | `ap-sxb7cXXxImeRcDEgiSo5dP` | `fc-01M57QNFVJM7KSJ58PE1HGCFCE` | $12.00 | 43200 s |
| `selfplay16-cloud-qualification-v1` | `ap-ykdt9vyqDbvlwCmBhnuXOc` | `fc-01M57VWXBX03YE0EX04CYVK90C` | $3.00 | 10800 s |
| `scale-champion96-selfplay-v1` | `ap-rEmqnLKLw7ykJUBS6W2SKd` | `fc-01M57W075W7FF5384AR3HFJCSC` | $23.00 | 28800 s |
| `selfplay-checkpoint-retainer-v1` | `ap-IWTgBSjoKN3MIkAf2S2sn0` | `fc-01M57W0QE1ZPP71F86W0F254BA` | $2.00 | 43200 s |
| `portfolio-finalists-cloud-qualification-v1` | `ap-LV6grxqdKBqNrf9lyFAsdb` | `fc-01M57W4Y76TZZARJC59539SAA5` | $3.00 | 10800 s |
| `selfplay-fast16-qualification-v2` | `ap-qUSeHDqUDIXwWKrH7ioSrC` | `fc-01M57Y76V0BJ9B8ZK4455PKHPH` | $3.00 | 10800 s |
| `selfplay32-matched-qualification-v1` | `ap-7XN0d2x0FeHw1EPuyX7YPN` | `fc-01M57YPVRP0JBW2X4F5F096VFV` | $3.00 | 10800 s |
| `low176-cloud-qualification-v1` | `ap-jsdbxHGeiuhDYb1zHaIHKk` | `fc-01M57YW0DM2WNY8N2H4VZ7KZP4` | $3.00 | 10800 s |
| `scale-low176-selfplay-v1` | `ap-FVuibq1kFrYF1uNco6L4T6` | `fc-01M57YWR36K51X4N08H028KZRA` | $12.00 | 14400 s |
| `low176-selfplay-retainer-v1` | `ap-4cpXQM3mNEaKpO54JHOgt5` | `fc-01M57YX8ENPC40KM5SMN9HJAPZ` | $2.00 | 43200 s |
| `low144-selfplay-pair-qualification-v1` | `ap-nMeaJIm0NQnMhBRXsW2hfH` | `fc-01M57Z8P84XM76PWKTGMDP80WK` | $3.00 | 10800 s |
| `gpu-selfplay-stage2-dev301400-v1` | `ap-kQ8jCp5QYxmhsigm0rfEb0` | `fc-01M5807DE52VQE9ATYNYWM4FFH` | $5.00 | 3600 s |

The two long learned-only producers are `scale-champion96-selfplay-v1` (seven requested hours, eight-hour hard cap) and `scale-low176-selfplay-v1` (three requested hours, four-hour hard cap). Five older mixed producers remain listed as submitted; only a fresh receipt can establish whether they still run.

## Appendix B. Stage2 immutable checkpoint roster

Plan status: `frozen_before_any301400_outcomes`; seed base 301400; 32 paired maps per candidate. Paths below are **inside Modal containers**, not local paths. Metadata SHA values and archived protocol/source checks are in the frozen plan.

### champion96-selfplayonly-v1-update16

Role: fixed reference; phase updates: 16; phase learner decisions: 524288.

- Checkpoint: `/retained/checkpoints/champion96-selfplayonly-v1/update-16-808f05cc6236/latest.npz`
- SHA-256: `808f05cc623638ecbd8a4739d75f1f64c281fdd49943ff77ff74eede6cc3bbe5`
- Metadata SHA-256: `d03b0c4d2c6c7ec90da387028dc1cf406c1e4e05ae0428039f21cf5f626cc1c8`

### scale-champion96-lowrate-v1-update176

Role: mixed-trained parent control; phase updates: 176; phase learner decisions: 5767168.

- Checkpoint: `/retained/checkpoints/scale-champion96-lowrate-v1/update-176-565eb6e315c8/latest.npz`
- SHA-256: `565eb6e315c82b970c0d0694f448aaf5a798de26c41deb7ad9b5436de4489052`
- Metadata SHA-256: `74674fe28189a3ef88f2b24c75cd86a09f51350dd2a1e36a9e86a9a1cfa8c089`

### scale-low176-selfplay-v1-update16

Role: learned-only descendant; phase updates: 16; phase learner decisions: 524288.

- Checkpoint: `/retained/checkpoints/scale-low176-selfplay-v1/update-16-0db27757a262/latest.npz`
- SHA-256: `0db27757a2620e54362acb42c5f4108c21af624080f80f0569d7ed3534d098a1`
- Metadata SHA-256: `f2c2b9a1a78ee03b262f8c4dab3a727551b3f493877f10335b1350bf107cb289`

### scale-low176-selfplay-v1-update8

Role: learned-only descendant; phase updates: 8; phase learner decisions: 262144.

- Checkpoint: `/retained/checkpoints/scale-low176-selfplay-v1/update-8-a0cd39c7182d/latest.npz`
- SHA-256: `a0cd39c7182d5536f7891252e3d0a5bd0284e8cfb498121ed8882a2fe0ea2bfb`
- Metadata SHA-256: `3d3073b661046f6d977b6e6af62b0059caf3fa5b899f9dd82e69c640ac399c67`

### low144-selfplay-rivals-v1-update8

Role: learned-only descendant; phase updates: 8; phase learner decisions: 262144.

- Checkpoint: `/retained/checkpoints/low144-selfplay-rivals-v1/update-8-a4b930f2ed18/latest.npz`
- SHA-256: `a4b930f2ed180d24690bfb9492ddf0cc329e14c5806a3a76ee61055d5eee0dfa`
- Metadata SHA-256: `bf6e2d8191f76353dc14c6be87c5efc41d1829ec11f4069751ebeece0bf39fa5`

### low144-selfplayonly-v1-update8

Role: learned-only descendant; phase updates: 8; phase learner decisions: 262144.

- Checkpoint: `/retained/checkpoints/low144-selfplayonly-v1/update-8-b214774d9a1f/latest.npz`
- SHA-256: `b214774d9a1f6ad4639296682950ce32f8cbb18a87cd87413f71ac12a88ed8db`
- Metadata SHA-256: `8d385a308cf73b5e9fe85ac47904a10762a662faa3ac95a8705a5f285a40912a`

### low144-selfplay-rivals-v1-update16

Role: learned-only descendant; phase updates: 16; phase learner decisions: 524288.

- Checkpoint: `/retained/checkpoints/low144-selfplay-rivals-v1/update-16-71fef239712c/latest.npz`
- SHA-256: `71fef239712c992471bbb061e1e36344547ad48815a522b6a29c938933c4c33c`
- Metadata SHA-256: `10889b80e4d4b2f672f493a84fb98165230c45587188acaa7d7a9e21d7d377fa`

### low144-selfplayonly-v1-update16

Role: learned-only descendant; phase updates: 16; phase learner decisions: 524288.

- Checkpoint: `/retained/checkpoints/low144-selfplayonly-v1/update-16-d5e689880670/latest.npz`
- SHA-256: `d5e6898806700cabf45b094527e494d06201b8734fc1242bd0f4084893df37c2`
- Metadata SHA-256: `5a282dae7de5696aed9284b0e063f5ad8b2191262c0d83fe249a909108de98d6`

### scale-champion96-selfplay-v1-update32

Role: learned-only descendant; phase updates: 32; phase learner decisions: 1048576.

- Checkpoint: `/retained/checkpoints/scale-champion96-selfplay-v1/update-32-1aec3543b1d1/latest.npz`
- SHA-256: `1aec3543b1d18dd18a6fd0c967d42e75eafe328ebc7338be97ab622b3fd0710c`
- Metadata SHA-256: `c3554a03b814c26518397140af1a98ffdf8ebe6ca657c814858d56d59eebaedc`

### scale-champion96-selfplay-v1-update48

Role: learned-only descendant; phase updates: 48; phase learner decisions: 1572864.

- Checkpoint: `/retained/checkpoints/scale-champion96-selfplay-v1/update-48-75c499e33a7b/latest.npz`
- SHA-256: `75c499e33a7b71dbaef15b5366685fab29fe3de4e7cc8899960ecfb09ebb6140`
- Metadata SHA-256: `7ec96a1e542e474a7372ebc1b7cf901030605e08dea2ed08441241e69c3bf2fb`

### champion96-selfplay-fastlr-v1-update16

Role: learned-only descendant; phase updates: 16; phase learner decisions: 524288.

- Checkpoint: `/retained/checkpoints/champion96-selfplay-fastlr-v1/update-16-dc39b052def0/latest.npz`
- SHA-256: `dc39b052def08a5ad3e260526c50f2da48c53d61011815d4fd1a838abb230941`
- Metadata SHA-256: `04525fef486cb2d03a39c4fd1867a3e198bf8330c6c02e803694511f12fb0ceb`

### scale-champion96-selfplay-v1-update16

Role: learned-only descendant; phase updates: 16; phase learner decisions: 524288.

- Checkpoint: `/retained/checkpoints/scale-champion96-selfplay-v1/update-16-542976069a84/latest.npz`
- SHA-256: `542976069a84bf7598edf45117f849521f213752fec9617ca4a74efacdc79347`
- Metadata SHA-256: `a2b1fa4b13f93d0ee511831d0e0be002925a950d0d62b2f71de4d179ebed183b`

## Appendix C. Published public-qualification result rows

These are published partial result rows from saved JSON reports, not a declaration that the overall job has finished. Worker trace-verification receipts are present; inspect each independent-audit record to see which rows have received the additional local identity/coverage audit. Ratios use all played games, including unresolved games.

| Qualification / job | Games | Wins | Losses | Engine truncations | Evaluation cutoffs |
|---|---:|---:|---:|---:|---:|
| `selfplay16-cloud-qualification-v1` / `self16-teacher-dev` | 32 | 32 | 0 | 0 | 0 |
| `selfplay16-cloud-qualification-v1` / `self16-vs-native96-common` | 32 | 23 | 9 | 0 | 0 |
| `selfplay16-cloud-qualification-v1` / `self16-vs-low32-fresh300700` | 64 | 24 | 40 | 0 | 0 |
| `selfplay16-cloud-qualification-v1` / `self16-vs-parent96-fresh300700` | 64 | 43 | 21 | 0 | 0 |
| `selfplay16-cloud-qualification-v1` / `self16-random-dev` | 32 | 32 | 0 | 0 | 0 |
| `selfplay16-cloud-qualification-v1` / `self16-stochastic-vs-deterministic` | 32 | 9 | 23 | 0 | 0 |
| `selfplay16-cloud-qualification-v1` / `self16-teacher-replication` | 32 | 31 | 1 | 0 | 0 |
| `selfplay16-cloud-qualification-v1` / `self16-vs-gamma48-common` | 32 | 25 | 7 | 0 | 0 |
| `selfplay32-matched-qualification-v1` / `self32-teacher-dev` | 32 | 31 | 1 | 0 | 0 |
| `selfplay-fast16-qualification-v2` / `fast16-random-dev` | 32 | 32 | 0 | 0 | 0 |
| `selfplay-fast16-qualification-v2` / `fast16-teacher-dev` | 32 | 32 | 0 | 0 | 0 |
| `low176-cloud-qualification-v1` / `low176-teacher-dev` | 32 | 32 | 0 | 0 | 0 |
| `low176-cloud-qualification-v1` / `low176-vs-native96-common` | 32 | 28 | 4 | 0 | 0 |
| `portfolio-finalists-cloud-qualification-v1` / `frozen240-teacher-dev` | 32 | 32 | 0 | 0 | 0 |
| `portfolio-finalists-cloud-qualification-v1` / `standard112-teacher-replication` | 32 | 31 | 1 | 0 | 0 |
| `portfolio-finalists-cloud-qualification-v1` / `frozen240-vs-parent96-fresh300600` | 64 | 37 | 27 | 0 | 0 |
| `portfolio-finalists-cloud-qualification-v1` / `standard112-vs-parent96-fresh300600` | 64 | 41 | 23 | 0 | 0 |
| `portfolio-finalists-cloud-qualification-v1` / `frozen240-vs-native96-dev` | 32 | 27 | 5 | 0 | 0 |
| `portfolio-finalists-cloud-qualification-v1` / `standard112-teacher-dev` | 32 | 32 | 0 | 0 | 0 |
| `portfolio-finalists-cloud-qualification-v1` / `frozen240-teacher-replication` | 32 | 32 | 0 | 0 | 0 |
| `portfolio-finalists-cloud-qualification-v1` / `frozen240-random-dev` | 32 | 32 | 0 | 0 | 0 |
| `portfolio-finalists-cloud-qualification-v1` / `standard112-vs-gamma48-dev` | 32 | 24 | 8 | 0 | 0 |
| `portfolio-finalists-cloud-qualification-v1` / `standard112-random-dev` | 32 | 32 | 0 | 0 | 0 |
| `low144-selfplay-pair-qualification-v1` — no local results file at extraction | — | — | — | — | — |

For a mode match, wins belong to the named challenger (for example stochastic), not automatically the preferred deterministic mode. Missing rows are pending/unpublished in this local snapshot, not zero scores. Low144 pair plans 448 games, short self16 352, self32 256, fast16-v2 192, low176 352 and portfolio finalists 576; use their frozen plan files to enumerate expected jobs.

## Appendix D. Completed GPU screen audit inventories

### gpu-historical-dev301000-v1

Audit observed at `2026-10-18T17:17:40.773684+00:00`: 768 verified games, 767 native, 1 engine truncations, 0 cutoffs.

| Candidate | Wins | Losses | Unresolved | Paired-map 95% interval |
|---|---:|---:|---:|---|
| `scale-champion96-lowrate-v1-update144` | 38 | 26 | 0 | 48.44%–70.31% |
| `scale-champion96-lowrate-v1-update128` | 48 | 16 | 0 | 65.62%–84.38% |
| `scale-champion96-lowrate-v1-update160` | 46 | 18 | 0 | 64.06%–81.25% |
| `scale-champion96-lowrate-v1-update176` | 54 | 10 | 0 | 75.00%–92.19% |
| `scale-champion96-lowrate-v1-update192` | 50 | 14 | 0 | 68.75%–87.50% |
| `scale-champion96-ratecontrol-v1-update128` | 43 | 21 | 0 | 57.81%–76.56% |
| `scale-champion96-ratecontrol-v1-update160` | 29 | 35 | 0 | 35.94%–54.69% |
| `scale-champion512-lowrate-frozen-v1-update272` | 41 | 23 | 0 | 54.69%–73.44% |
| `scale-champion512-lowrate-noentropy-v1-update224` | 39 | 25 | 0 | 51.56%–70.31% |
| `scale-champion512-v1-update304` | 26 | 38 | 0 | 31.25%–50.00% |
| `scale-bootstrap-longhorizon1-update688` | 27 | 37 | 0 | 31.25%–53.12% |
| `scale-bootstrap-noentropy1-update816` | 33 | 30 | 1 | 42.19%–62.50% |

Independent JSON checkpoint/metadata/reference identities, paired coverage, outcomes and bootstrap audit. GPU trace hashes are cloud receipts, not locally recomputed. Complete means all requested candidates evaluated; unresolved games remain in every win-rate denominator. Screening only, not public qualification or final acceptance.

### gpu-selfplay-stage1-dev300800-v1

Audit observed at `2026-10-18T17:17:40.886712+00:00`: 640 verified games, 640 native, 0 engine truncations, 0 cutoffs.

| Candidate | Wins | Losses | Unresolved | Paired-map 95% interval |
|---|---:|---:|---:|---|
| `low144-parent-update144` | 47 | 17 | 0 | 64.06%–82.81% |
| `low144-selfplayonly-v1-update16` | 42 | 22 | 0 | 56.25%–75.00% |
| `low144-selfplay-rivals-v1-update16` | 44 | 20 | 0 | 57.81%–79.69% |
| `champion96-selfplay-fastlr-v1-update16` | 34 | 30 | 0 | 42.19%–64.06% |
| `champion96-selfplayonly-v1-update16` | 35 | 29 | 0 | 45.31%–64.06% |
| `scale-champion96-selfplay-v1-update16` | 36 | 28 | 0 | 48.44%–64.06% |
| `low144-selfplayonly-v1-update8` | 47 | 17 | 0 | 64.06%–82.81% |
| `low144-selfplay-rivals-v1-update8` | 46 | 18 | 0 | 60.94%–82.81% |
| `champion96-selfplay-fastlr-v1-update8` | 33 | 31 | 0 | 42.19%–60.94% |
| `scale-champion96-selfplay-v1-update8` | 33 | 31 | 0 | 39.06%–64.06% |

Independent JSON checkpoint/metadata/reference identities, paired coverage, outcomes and bootstrap audit. GPU trace hashes are cloud receipts, not locally recomputed. Complete means all requested candidates evaluated; unresolved games remain in every win-rate denominator. Screening only, not public qualification or final acceptance.

## Appendix E. Snapshot provenance

The following SHA-256 values identify the exact local JSON bytes used to assemble the appendices. These files are expected to evolve; a later mismatch does not by itself indicate corruption. Prefer newer verified receipts over this snapshot.

| Local record under `runs/structured-ppo-v3/` | SHA-256 |
|---|---|
| `budget.json` | `ca42f56f32736ab9a4fb9d294fe2215667613102b4dc814adb683a66615834d1` |
| `gpu-selfplay-stage2-dev301400-v1-plan.json` | `fa9f7e5784256a7cccea2f1c2f854671ae589b59626f87e569239097248b7189` |
| `campaign-observation-20261018T1719Z.json` | `3eb4e940ee09a9ce8097721a78e63a8de52a0f4a4ffc053bace0e6d0f1aef2cf` |
| `selfplay16-cloud-qualification-v1-results.json` | `70915b4a4c0587e5d588e7d474d55faca60e0fc2aa92688924427a5138a2fad3` |
| `selfplay32-matched-qualification-v1-results.json` | `75eb0b3de60469b19a3e63141c0fab909ed52e09c1edb7b50dfb2165744fd43f` |
| `selfplay-fast16-qualification-v2-results.json` | `3da76eaf97360e5e042c3fc1272bd15727566ac2377d358d93898fa00a7ed948` |
| `low176-cloud-qualification-v1-results.json` | `11769859be56684f4357dca3646e3b62f26c985dd2a780c843abbd4cc5b10cb8` |
| `portfolio-finalists-cloud-qualification-v1-results.json` | `193ae2d30f64ba4c6da80c4a7cb23b36c6071382b58de0113828524016967cf0` |
| `gpu-historical-dev301000-v1-independent-audit.json` | `fa8ea1040eea52f25070b3691afb6d6c311876bb6ebee772208ad62611eb5a33` |
| `gpu-selfplay-stage1-dev300800-v1-independent-audit.json` | `1f8726f8e66f375bb1b6a2ca7af658e9ee9a3c801f21055b9b3bc5c916117fe3` |
