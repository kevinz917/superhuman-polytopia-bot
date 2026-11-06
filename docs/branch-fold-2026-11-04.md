# Branch fold into main (2026-11-04)

Decision: experiment work lands on main as soon as its run is over; a branch lives only while a run is live.
Both gameplay environments are kept: the root `engine-rs/` is the 2.17.3 engine every current run uses, and
`legacy/ruleset-v2/` is the research-ruleset engine + wasm crate the public site and the champion still run on.

## Merged into main (in this order)

| Branch | What came in |
|---|---|
| `claude/farsight-v2-e0f1fe` | Daytona hero harness (keeper, ladders, pool probe, tests on Daytona), obs v3 + trunk pooling layouts 1 and 2, KL-adaptive LR, farsight-v2 / Arm B / Arm C plans, the series analysis and ledger archive (`docs/run-archive/`). Contains `claude/obs-v3-history-03c82d`, `claude/daytona-hero`, `claude/isoflop-sweep` (committed part), `claude/daytona`. |
| `claude/lookout` | Hybrid attention trunk (dormant behind its spec key), live ladder references, sync-farsight, lookout close-out. |
| `claude/hero-run-v3-d53e9f` | Curio from-scratch config: loss at cap; `selfplay_probe --no-peace`. |
| `claude/rust-engine-c74afb` (remote) | Rust engine report addendum. |
| `claude/trailer-evalbar-cb4fa3` | Hands trailer (contains `claude/trailer-hands`): 15/20 s cuts, leaderboard close, ElevenLabs soundtrack. |
| `claude/gracious-wright-bb6f22` | scratch-cont-2 write-up, studio-main launch entry. |
| `claude/verda-big-run` | Verda controller/pilot tooling, bn-main write-ups, long-run plan, bn4 replays, `map_mix` for the reference round robin. Its learner changes were already on main. |
| `claude/isoflop-sweep` (uncommitted work, committed first) | `training_updates` / `snapshot_at`, update-driven schedules, cooldown forks, FLOP accounting, sweep arms, `tests/test_sweep.py`. Never launched. |
| stray local work | Launch configurations (18 entries), docs that were only untracked (blog draft, Arm B brief, human games, KataGo review, bench post, site-games analysis), bench scripts, the omni2 replay set, qwen35's actor-profile scripts and a friend's game record. |

## Preserved outside main's tree (annotated tags)

| Tag | Why not merged |
|---|---|
| `legacy/ruleset-v2-browser-bot` (= `claude/browser-bot`) | Copied verbatim to `legacy/ruleset-v2/` instead; merging would have put the old ruleset back into the root engine. |
| `archive/ffa4-value-pass` | 4p `tight_values` lever rewrites the value pass that main later reworked (split value mesh, unrolled loops); needs its bitwise gate rerun before it can land. |
| `archive/trailer-chips`, `archive/trailer-leaderboard` | Alternative cuts of the same trailer page; the evalbar cut is the one on main. |
| `archive/inference-server-rust` | Superseded by per-node GPU inference in the hero runs. |
| `archive/structured-rl-*` | Work on the single-machine RL stack removed in 3ac7ce8. |
| `archive/qwen35-studio-ui` | Oct 24 Studio UI pass; `web/` since rewritten (real-art modes). Everything else from it is on main. |
| `archive/open-source-staging` | Staging for the public repo; the public repo is branch `open-source-release`. |

## Kept as separate branches on purpose

- `external-website` (deployed site, unrelated history), `claude/site-browser-play`, `claude/bench-viewer`: deploy
  branches of the site; they vendor the built wasm and champion weights.
- `open-source-release`: the public repository's history.
- `codex/polytopia-skirmish`, `codex/polytopia-bench-v2`: the benchmark harness, another agent's active work
  (updated Nov 3); main carries the archived benchmark under `benchmark/`.

## Not committed

- `docs/launch-video/` (791 MB renders) and `docs/bench-tweet/` (28 MB): media, keep out of git.
- `reports/` and `research_notes/` on the main checkout except the Polytopia ones: unrelated research notes.
- `runs/` everywhere: git-ignored ledgers; copy a finished series into `docs/run-archive/` (done for this one).
