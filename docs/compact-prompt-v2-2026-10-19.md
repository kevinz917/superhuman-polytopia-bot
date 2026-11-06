# Compact full-state prompt v2 — format, savings and equal-data ablation (2026-10-19)

**Bottom line.** A compact prompt that still carries the complete public state on every move cuts prompt tokens by 69%, raises the prefix shared between consecutive moves from 14% to 64%, and makes the identical expert-SFT run train 4.1× faster.

- **Imitation:** unchanged. On 2,000 held-out expert decisions, both adapters pick the expert's move 36% of the time.
- **Games:** the final version (v2.1) won 34/60 dev games against v1's 18/30. The pooled paired difference is −0.03 (95% CI [−0.23, +0.17]).
- **One regression:** in about half the games the compact model sometimes demolishes its own buildings, which the expert never does. That is where all of the remaining win gap sits: in games without demolition it plays at v1 level.

**Recommendation: GO for the next large SFT run with compact v2.1, plus one guard.** Remove the `demolish_own` ability from the model's action catalog, in training and inference alike; the expert chose it in 0 of 39,626 training labels. First confirm the guard with a 1-hour eval-only check (§5).

| | Readable (expert-sft-v1) | Compact v2.1 | Change |
|---|---:|---:|---:|
| Prompt tokens per decision (mean, incl. system; full corpus) | 10,124 | 3,175 | **0.31×** |
| Prompt tokens, p90 / max | 17,273 / 39,323 | 4,622 / 10,527 | 0.27× |
| Shared token prefix with the previous decision | 14% | **64%** | 4.6× |
| New (uncached) tokens per decision with prefix caching | 9,022 | **1,244** | **0.14×** |
| Training: tokens / wall time (4×B200, same recipe, same 39,626 examples) | 370M / 6,997 s | 120M / 1,702 s | **0.32× / 0.24×** |
| Training peak memory (longest-batch stress test) | 150 GB | 51 GB | 0.34× |
| Eval: prompt tokens per move / latency per call (H100, no prefix caching) | 11,764 / 6.3 s | 4,113 / 1.7 s | 0.35× / 0.26× |
| Eval: GPU time per 30-game dev run (no prefix caching) | 5,726 s | 2,999 s | 0.52× |
| Held-out imitation accuracy (2,000 decisions, greedy) | 36.0% | 36.3% | +0.35 pp [−1.6, +2.3] |
| Dev-map wins vs `balanced` (30 jobs) | 18/30 | 14/30 and 20/30 (two arms) | pooled −0.03 [−0.23, +0.17] |

## 1. What changed

`render_compact_decision(obs)` in [`compact_prompt.py`](../src/polytopia_bot/learning/compact_prompt.py) returns the same `Prompt` as `render_readable_decision`: the same `render(obs)` catalog, the same prompt-local `a0…aN` IDs in the same order, the same `public` state and `state_version`. Only `messages` differ. So `parse_response`, the expert labels (`{"action_id":"aN"}`) and the label mapping carry over unchanged. `compact_prompt.action_schema(messages)` builds the legal-action grammar from either format, so the IDs and enum are identical.

Design rules:

1. **Full state on every move.** Every message carries the complete public observation: all tiles, units, cities, technology, relations, achievements, memory (recent actions), stars, turn and `state_version`. There are no deltas and no transcript. `parse_compact` inverts the user text exactly (§3).
2. **Drop derived helpers.** `decision_facts` (27% of old tokens), `action_names` (4%) and the verbose encoding instructions are gone, and the per-observation `rules` glossary (6%) moved into the static system prompt. The only derived values kept are the star cost of recruit, build, research and harvest items (`n*`, which replaces the old `spending` table) and, in v2.1, the name of the structure a `demolish_own` would remove.
3. **Compact encoding.** Tiles are `x,y` coordinates. Terrain and territory are 14×14 one-token-per-cell grids. Features are grouped by kind. Legal actions are grouped by kind and acting tile. Units, technologies and structures appear by name.
4. **Slow content first, fast content last.** The order is: system prompt (static rules) → terrain → territory → features → cities → technology → enemy units → your units → recent actions → turn/stars/version → legal actions. Within a turn, a unit move or a purchase changes only text from the unit lists onward.

**v2.0 → v2.1.** The only change is that ability 4 is shown as `demolish_own <Structure> x,y` instead of `destroy x,y`, plus one system-prompt line: "demolish_own permanently removes one of YOUR OWN structures (named after it); it never harms the enemy." See §4.3 for why, and for the finding that it did not stop the behavior. The v2.0 renderer that produced the v2.0 run is kept verbatim (its hash matches that run's protocol) at `runs/compact-prompt-v2/compact_prompt-v2.0.py.txt`.

### 1.1 System prompt (static, 1,078 tokens)

The system prompt is a short contract plus a rule glossary covering every unit, technology and structure that occurs anywhere in the 356,727-decision corpus: 15 units, 26 techs and 23 structures (see [`corpus-census.json`](compact-prompt-v2/corpus-census.json)). An observation that references any other ID gets an `extra rules:` line in its user text, so the glossary is never lossy. For comparison, the old prompt had a 1,008-token system prompt plus a per-observation glossary of about 560 tokens in the user text.

```
You play Polytopia (1v1). Each message is the complete public state of your current decision.
Choose exactly one listed legal action and answer only {"action_id":"aN"}.
Tiles are x,y (x = column left to right, y = row top to bottom, from 0). Y = you, E = enemy, N = unowned.
terrain grid: w water o ocean f field m mountain t forest i ice b wetland g mangrove, ? unknown. territory grid: ...
...
demolish_own permanently removes one of YOUR OWN structures (named after it); it never harms the enemy.
Rules:
units (cost* hp attack defense movement range skills): Warrior 2* hp10 atk2 def2 mov1 rng1 Dash Fortify; Rider 3* ...
technologies (tier, prerequisite, unlocked structures): Basic T0; Riding T1; FreeSpirit T2 after Riding builds Temple; ...
structures (cost*, base population, base stars): ... LumberHut 3* pop1 stars0; ... Market 5* pop0 stars1
```

### 1.2 User message (one real training-split decision, turn 18; grids and lists shortened)

```
polytopia-compact-v2.1 player 1 board 14x14
terrain:
y0 t t m t t t t t t f t t w ?
...
y13 ? ? ? ? ? ? ? ? ? ? ? ? ? ?
territory:
y0 . . . . . . . . . . . . . ?
y1 . . . . . . Y Y Y . Y Y Y ?
...
resources: animal 5,0 5,2 2,3 ...; fish 12,0 12,2 7,10; metal 6,8; fruit 10,1 ...; starfish 11,7
structures: Village 7,2 11,2L0 4,4L4 9,6L2 ...; Ruin 3,2L0; LumberHut 4,3 5,3 ...; Sawmill 10,9L2; Lighthouse 0,0L0
road: 1,3 0,7 4,7 ...
capital: 4,4 3,10=2
monuments: ParkOfFortune | built: ParkOfFortune
relations: known_players=2 peace=- embassies_sent=- embassies_received=- peace_offers=-
cities:
Y 4,4 L5 pop15 prod6 prog1 rewards workshop superunit resources population
Y 11,2 L1 pop0 prod1 prog0
E 3,10
technology: Basic Organization Fishing Hunting Forestry Mathematics Climbing Mining
enemy units:
your units:
Warrior 4,0 hp10 home 4,4
Warrior 11,2 hp10 home 11,2 moved attacked
Giant 3,9 hp40 home 4,4 k1
recent actions: t17 r356 move 4,8>3,9; t17 r357 move 10,1>11,1; t17 r358 end_turn>; t18 r360 capture 11,5>; t18 r361 capture 11,2>
turn 18 stars 41 version 362
legal actions:
move from 4,0: a0 3,0; a1 5,0; a2 3,1; a3 4,1; a4 5,1
...
ability: a64 clear_forest 11,1; a65 clear_forest 12,1; a66 clear_forest 10,3
ability from 4,9: a67 recover
recruit from 7,2: a69 Warrior 2*; a70 Catapult 8*
harvest: a81 2* 10,1; a82 2* 12,2; a83 2* 12,3
build: a84 Port 7* 10,2; ... a88 LumberHut 3* 11,1; ... a91 Sawmill 5* 3,3
research: a97 Riding 11*; a98 Strategy 18*; ...
capture from 3,10: a105
end_turn: a106
resign: a107
```

### 1.3 Format spec

| Section | Encoding | Inverse |
|---|---|---|
| Header | `polytopia-compact-v2.1 player P board NxN` | player, board size |
| `terrain:` | N rows `yR c0 … c(N−1)`. Cells: `w o f m t i b g` are terrain IDs 1–8, `?` unknown, `#k` other ID, `*` irregular tile (see `tile json`) | `known`, `terrain` per tile |
| `territory:` | Same grid. `Y` you, `E` other player, `.` owner 0, `?` unknown | `owner` |
| `resources:` | `name x,y x,y; name …` (IDs 1, 2, 3, 5–9 by name, else `resource#k`) | `resource` (unlisted = 0) |
| `structures:` | `Name x,y[Lk]`. No suffix = `structure_level` 1; `Lk` = level k (`L0` included) | `structure`, `structure_level` |
| `levels without structure:` / `road:` / `capital:` | `x,y[=v]` (v shown when ≠ 1) | `structure_level`, `road`, `capital` |
| `tile json:` | Canonical JSON of any known tile without exactly the 9 standard keys (never occurs in the corpus) | verbatim |
| `extra rules:` | Glossary lines for rule IDs outside the static set (never occurs in the corpus) | derived; ignored |
| `monuments: … \| built: …` | Structure names | `achievements` |
| `relations:` | `key=players-with-true` (`-` none) for the 5 relation pairs | `relations` |
| `cities:` | Yours first: `Y x,y Lk popN prodN progN [connected] [riot] [rewards name…]` (a name repeats for counts > 1). Others: `E x,y` / `N x,y`. `city json:` for any other shape | `cities` |
| `technology:` | Names in ID order | `technology` |
| `enemy units:` | `E Kind x,y hpH [veteran]`; `unit json:` for other shapes | `units` |
| `your units:` | `Kind x,y hpH home x,y\|- [true flags…] [kN] [cargo Kind]`. `home` is the decoded tile+1 value; hp is a float printed without `.0` (`hpNi` for an integer) | `units` |
| `recent actions:` | `tT rR <command>` joined by `; ` (`-` none). `memory json:` / `remembered json:` for non-standard memory | `remembered` |
| `turn T stars S version V` | | `turn`, `stars`, `state_version` |
| `legal actions:` | One line per (kind, acting tile): `kind[ from x,y\|#p]: aN [param] [cost*] [annotation] [target]; …`. Diplomatic abilities (params 18–22) use raw `#n` values. The only annotation is the structure name after `demolish_own` | full catalog `aN → (kind, source, target, parameter)` |

## 2. Token and prefix savings (all 356,727 corpus decisions)

Measured on Modal (`--action measure`) with the Qwen3-4B-Instruct-2507 chat template; results in [`measure-v2.1.json`](compact-prompt-v2/measure-v2.1.json). v2.0 numbers are within 1% ([`measure-v2.0.json`](compact-prompt-v2/measure-v2.0.json)).

| | Readable | Compact v2.1 |
|---|---:|---:|
| System prompt tokens | 1,008 | 1,078 |
| Prompt tokens: mean / median / p90 / max | 10,124 / 9,041 / 17,273 / 39,323 | 3,175 / 2,975 / 4,622 / 10,527 |
| User text characters (mean) | 24,587 | 3,743 |
| Shared prefix with previous decision, same turn (335,737 pairs) | 14.1% | 64.2% |
| Shared prefix, first decision of the next turn (19,990 pairs) | 12.1% | 60.3% |
| Uncached tokens per decision, all pairs (same turn / next turn) | 9,022 (8,907 / 10,952) | 1,244 (1,227 / 1,528) |

Where the compact user tokens go (mean of about 2,078 per decision; sampled every 10th decision):

| Section | Tokens | Share |
|---|---:|---:|
| Legal actions | 773 | 37% |
| Terrain + territory grids | 467 | 22% |
| Features (resources, structures, roads, capitals, relations) | 340 | 16% |
| Your units | 227 | 11% |
| Cities | 116 | 6% |
| Recent actions | 80 | 4% |
| Enemy units, header, status, technology | 75 | 4% |

Why the old prompt shared so little: in sorted-key JSON, `action_names` and `decision_facts`, which change every move, came right after the system prompt. In the compact prompt the system prompt, grids and features (about 60% of tokens) usually survive a move unchanged. A newly revealed tile breaks the prefix at its grid row; a build breaks it at the structures line.

## 3. Equivalence

[`tests/test_compact_prompt.py`](../tests/test_compact_prompt.py), run on Modal (`--action test`): 7 passed, for both v2.0 and v2.1.

- **Catalog identity.** For every action, the readable prompt's `aN → command` equals the compact prompt's parsed `aN → command`, which equals `Prompt.catalog`. `parse_response` returns the same engine action for every ID under both prompts. The compact grammar's enum equals the readable one, and out-of-catalog IDs are rejected.
- **State round trip.** `parse_compact(user) == public − rules` (canonical JSON). This is tested on a synthetic edge-case position covering:
  - diplomatic raw IDs that collide with tile numbers;
  - an out-of-glossary unit;
  - integer and fractional health, cargo;
  - repeated city rewards, relations, achievements, memory;
  - a `demolish_own` annotation;
  - plus the irregular-shape fixture (exercises every JSON fallback).
- **Prefix order.** Moving a unit leaves everything before the unit lists byte-identical.
- **Real decisions.** Every decision in 40 expert games is checked. Beyond the tests, the measurement and both corpus regenerations repeat the checks on **all 356,727 decisions, with 0 failures**.

The expert game files store the readable user text, not observations. `public_from_readable` recovers the full public state from that text, and `readable_from_public` re-renders it **byte-identically** (checked on every decision). So the corpus was converted without replaying the environment.

## 4. Equal-data ablation

### 4.1 Protocol

Only the prompt changes relative to `expert-sft-v1`.

- **Data:** the same 39,626 (game, decision, label) rows from `expert-sft-v1/prepared/index.jsonl`, in the same shard order with the same `aN` labels. The combine step asserts that ID and label lists are identical to v1. Each row is re-rendered from its stored readable prompt with the compact renderer, after the equivalence checks above.
- **Training:** the identical config, seed 17 and qualified packed runtime (`castfree_segmented_8192`), via `expert_trainer.py` on 4×B200: LoRA r32/α64 on all projections, lr 1e-4 cosine, 3% warmup, effective batch 64, 1 epoch, 620 updates.
- **Evaluation:** the same 30 development jobs (15 validation maps × 2 seats, vs scripted `balanced`), sampling, per-decision seeds, vLLM settings, LoRA serving and no-repair single call as v1, using the compact prompt and grammar. [`benchmark_compact_sft.py`](../scripts/benchmark_compact_sft.py) is a copy of `benchmark_expert_sft.py` at 5cde6dc (the code that produced v1's numbers) with only the renderer swapped.
- **Two arms per adapter:**
  - `compact_sft` serves exactly like v1, with prefix caching off.
  - `compact_sft_prefix_cache` serves the same adapter with vLLM prefix caching on. It measures the serving gain and doubles the game sample. Its games differ from the first arm only through engine numerics and scheduling.
- **Statistics:** paired difference against the v1 adapter's 30 games, with a map-cluster bootstrap (20,000 resamples), exactly as in the expert-SFT report.
- **Smoke tests first** (run `compact-prompt-smoke-1792443835`): data round trip (80 rows; the supervised suffix decodes to the label), a 2-update train on 4×B200 with stress and integrity checks, and a LoRA preflight in vLLM with the compact prompt and grammar (4/4 valid).

### 4.2 Results

Data and training ([`ablation-v2.1.json`](compact-prompt-v2/ablation-v2.1.json)):

| | expert-sft-v1 | compact v2.0 | compact v2.1 |
|---|---:|---:|---:|
| Examples / duplicates / label conflicts | 39,626 / 0 / 0 | 39,626 / 0 / 0 | 39,626 / 0 / 0 |
| Tokens: total / mean / max | 369.7M / 9,329 / 24,574 | 119.0M / 3,003 / 7,193 | 119.9M / 3,027 / 7,423 |
| Train time, 620 updates | 6,997 s (7.8 B200-h) | 1,630 s | 1,702 s (1.9 B200-h) |
| Stress-test peak memory | 150 GB | 50 GB | 51 GB |
| Loss: step 1 / mean of last 20 | 1.898 / 0.260 | 1.466 / 0.246 | 1.467 / 0.242 |

The loss curves match closely (for example 0.299 vs 0.302 at step 38).

Held-out imitation ([`imitation.json`](compact-prompt-v2/imitation.json)): 40 fresh expert games on validation maps 100541–100560 (never used for training), with 50 decisions sampled per game. Each adapter makes a greedy grammar-constrained choice in its own prompt format, scored against the expert's label.

| | Readable v1 | Compact | Compact − v1 [game-cluster 95% CI] |
|---|---:|---:|---:|
| v2.0 adapter | 36.1% | 36.6% | +0.55 pp [−1.25, +2.4] |
| v2.1 adapter | 36.0% | 36.3% | +0.35 pp [−1.55, +2.3] |
| Scoring time, 1 H100, prefix caching on in both | 434 s (19.2M tokens) | 73 s (6.1M tokens) | 5.9× faster |

Per-family accuracies match within a few points: end turn 92% vs 92%, attack 47% vs 49%, move 15% vs 15%. Absolute accuracy is low because many moves are interchangeable, such as the order in which units move.

Games on the 30 development jobs ([`ablation-v2.0.json`](compact-prompt-v2/ablation-v2.0.json), [`ablation-v2.1.json`](compact-prompt-v2/ablation-v2.1.json)):

| Arm | W | L | Round-limit | Median last round | Peak cities | Δ wins vs v1 [map 95% CI] | Maps better / worse |
|---|---:|---:|---:|---:|---:|---:|---:|
| expert-sft-v1 (readable) | 18 | 12 | 0 | 34.5 | 7.4 | — | — |
| v2.0, no cache | 10 | 19 | 1 | 40.5 | 7.0 | −0.27 [−0.50, −0.03] | 2 / 8 |
| v2.0, prefix cache | 13 | 16 | 1 | 38.0 | 7.2 | −0.17 [−0.43, +0.10] | 4 / 7 |
| **v2.1, no cache** | 14 | 16 | 0 | 36.0 | 6.9 | −0.13 [−0.40, +0.13] | 4 / 7 |
| **v2.1, prefix cache** | 20 | 9 | 1 | 34.5 | 7.7 | +0.07 [−0.10, +0.23] | 4 / 2 |
| **v2.1, pooled (60 games)** | 34 | | | | | **−0.03 [−0.23, +0.17]** | 6 / 8 |
| v2.0, pooled (60 games) | 23 | | | | | −0.22 [−0.42, −0.02] | 4 / 9 |
| All four compact arms (120 games) | 57 | | | | | −0.13 [−0.31, +0.06] | 5 / 9 |

- **Against the base model:** every compact arm beats the base (0/30), for example v2.1 by +0.47 [0.27, 0.67] and +0.67 [0.50, 0.80].
- **Sampling noise:** the two arms of the same v2.1 adapter differ by 6 wins (14 vs 20). That is the size of the per-30-game sampling noise; v1's 18/30 is itself a single 30-game sample.
- **Game endings:** all games ended by elimination except three round-limit stalls. There were no forfeits, invalid outputs or operational failures, across 98,604 compact model calls in total.

Evaluation cost (one H100 per arm, 30 games, concurrency 64):

| Arm | Model calls | Prompt tokens/call | Total prompt tokens | Latency/call | Worker time |
|---|---:|---:|---:|---:|---:|
| expert-sft-v1 | 17,898 | 11,764 | 210.6M | 6.34 s | 5,726 s |
| v2.1, no cache | 21,157 | 4,113 | 87.0M | 1.67 s | 2,999 s |
| v2.1, prefix cache | 22,580 | 4,400 | 99.4M | 1.21 s | 3,075 s |

Total wall time for a 30-game run depends on game length and the last few long games, not only on per-call cost. At 4k tokens the evaluation is no longer prefill-bound: its tail is a handful of games stepping serially, which is why prefix caching barely changes v2.1's total time while cutting per-call latency by another 28%. With the long-prompt v1 format, caching would gain little, because only 14% of each prompt repeats.

### 4.3 The one regression: self-demolition

**v2.0.** The compact model used ability 4 ("destroy") 719 and 870 times, in 12 of 30 games per arm. v1 used it twice in its 30 games, and the expert never chose it in the 39,626 training labels, although it is offered 37,917 times across the corpus. Traces show the pattern ([`demolition.json`](compact-prompt-v2/demolition.json)): late in a game, after spending its stars to 0, the model demolished its own lumber huts, sawmills, temples and market one after another, about 40 in one turn, before ending the turn. Our hypothesis was that the bare word "destroy" reads as an attack to a pretrained model, whereas the readable prompt showed only `kind 3, parameter 4`.

**v2.1 renamed it `demolish_own <Structure>` and added a system-prompt warning. The behavior did not go away:** 547 and 534 demolitions, in 14 and 16 of 30 games ([`abilities-v2.1.json`](compact-prompt-v2/abilities-v2.1.json)). The wording hypothesis is therefore not confirmed. The v2.1 win rate is higher than v2.0's, but the two v2.1 arms alone differ by 6 wins, so most of the v2.0 → v2.1 change is within sampling noise.

The demolition games are where the gap lives. Pooled over all four compact arms (120 games):

| Games | Compact wins | v1 wins on the same jobs |
|---|---:|---:|
| With any self-demolition (54) | 19 (35%) | 32 (59%) |
| Without (66) | 38 (58%) | 40 (61%) |

v1 won the same jobs, so these were not harder maps. Demolition may be partly a symptom of an aimless late game rather than only a cause, but it is compact-specific: v1 did it twice in 30 games, the compact adapters in 12–16 of 30. Why the compact model generalizes into it is still open. The action is never in the labels, so its probability comes entirely from how the model reads unfamiliar, fully legal options in its own late-game states. Those states are the ones the held-out imitation test does not reach.

## 5. Recommendation for the next large run

**GO: use compact v2.1 for the next large SFT run, with the demolition guard.** Expected savings at the same data size:

- about 3.1× fewer training tokens and about 4× less training time, with a peak-memory stress test at a third of v1's;
- about 3× fewer prompt tokens per evaluation move, with per-call latency 3.8× lower (5× with prefix caching);
- room to raise the 24,576-token example cap or the examples per game for free.

Guard (the user's call, since it restricts the action space):
- Omit ability 4 (`demolish_own`) from the model's catalog in both data rendering and gameplay, for example by filtering those entries out of `obs.legal_actions` before `render_compact_decision` assigns IDs.
- The current RL expert never takes this action, so no training label is lost.
- If the next expert, u3142, does use it, keep the action and instead investigate further.

Before the large run, confirm the guard cheaply:
- Evaluate the existing v2.1 adapter on the same 30 jobs with demolition masked: one eval-only run, about 1 h on 2 H100s, no retraining.
- If wins return to roughly v1's level (around 18/30, within noise) and the "with demolition" gap disappears, proceed.
- If not, keep v1's readable prompt for the large run and treat the compact format as a training-cost optimization still to be fixed.

Limitations:
- Development maps only (15 maps × 2 seats) against one scripted opponent.
- One sampling sequence per arm, and v1 itself is a single 30-game sample.
- The held-out test-split protocol (`expert-sft-heldout-v1`) was not repeated here.

## 6. Files and runs

- Renderer: [`compact_prompt.py`](../src/polytopia_bot/learning/compact_prompt.py).
- Tests: [`test_compact_prompt.py`](../tests/test_compact_prompt.py).
- Data prep: [`prepare_compact_sft.py`](../scripts/prepare_compact_sft.py), which reuses `prepare_expert_sft.combine`.
- Evaluation arm: [`benchmark_compact_sft.py`](../scripts/benchmark_compact_sft.py).
- Launcher: [`modal_compact_prompt.py`](../experiments/modal_compact_prompt.py), with actions `scan`, `test`, `measure`, `smoke`, `launch`, `compare`, `imitation`, `abilities`, `destroy-trace`.
- Result files: [`docs/compact-prompt-v2/`](compact-prompt-v2/).
- Runs on volume `polytopia-sft0-training` (env `staging`), with local receipts under `runs/` (gitignored):
  - `compact-prompt-smoke-1792443835`: smoke run.
  - `compact-prompt-v2`: app `ap-ZOvDNgvRa7FAfm9lD1nCAC`.
  - `compact-prompt-v2.1`: app `ap-iTSnS9Uf5jo84TuE3xublH`. Its adapter is `compact-prompt-v2.1/training/model/adapter-final`.
  - `compact-prompt-v2-imitation`: the 40 held-out expert games and per-decision rows.
- No checkpoints were downloaded. `thinking_readable.py`, `benchmark_expert_sft.py` and `modal_expert_sft.py` were not modified. `expert-sft-v1/` was only read.
