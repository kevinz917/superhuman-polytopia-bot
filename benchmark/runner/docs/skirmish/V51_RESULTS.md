# Completed Polytopia benchmark — October 30–31, 2026

Eleven OpenRouter models each produced three independent bot builds on both tracks: up to 40 uninterrupted coding minutes, or four 10-minute stages with three bounded expert-feedback batches. The coding harness is Mini-SWE-agent 2.4.6 in offline Modal sandboxes. Every model received the same rules and engine tools; no opponent code or weights were supplied.

All 66 final submissions contain changed, compiling Rust code. Every selected stage was independently rebuilt before hidden maps were allocated. The primary score uses three frozen references, fifteen shared hidden layouts, and both seats. All builds count; there is no best-of selection.

## Primary results

Points per 100 (win 1, draw 0.5, loss 0), averaged over the three fixed opponents and all three builds. These scores are **not expert win rates**. Rows are ordered by the revision-track mean. Each expert-win count is out of 90 final-stage matches per track.

| Model | Independent 40m | Three revisions, 40m | Expert wins: independent / revised | Saved API per scored two-track pair |
|---|---:|---:|---:|---:|
| GPT-6 Astra | 67.59 | 66.48 | 2/90 / 1/90 | $31.3033 |
| GPT-6.1 Sol | 66.67 | 65.74 | 1/90 / 1/90 | $4.5644 |
| Gemini 4 Argon | 60.93 | 60.37 | 0/90 / 0/90 | $4.6152 |
| Claude Opus 5.5 | 50.00 | 48.89 | 0/90 / 0/90 | $35.0939 |
| GLM 5.3 | 27.59 | 44.81 | 0/90 / 0/90 | $7.2988 |
| Kimi K3.1 Pro | 43.52 | 43.52 | 0/90 / 0/90 | $2.7052 |
| GPT-6 Luna | 34.63 | 40.00 | 0/90 / 0/90 | $0.6810 |
| DeepSeek V4.1 Flash | 31.85 | 35.37 | 0/90 / 0/90 | $0.2320 |
| DeepSeek V4 Pro | 42.04 | 34.26 | 0/90 / 0/90 | $1.2142 |
| Qwen3.8 Max | 16.30 | 25.74 | 0/90 / 0/90 | $2.4567 |
| MiniMax M3.1 Flash | 19.26 | 10.19 | 0/90 / 0/90 | $0.4652 |

Astra has the highest primary point estimates in both tracks. Its 95% bootstrap intervals overlap Sol’s: independent Astra 66.67–69.81 versus Sol 65.19–67.78; revised Astra 65.00–68.15 versus Sol 63.15–67.78. Three builds remain a preliminary comparison.

The expert remained dominant in the 1,980 final-stage expert matches: 1,943 wins, 32 draws and five losses. Astra accounts for three model wins and Sol for two.

The secondary pooled Elo ladder puts Astra first among models in the independent track and Sol first in the revision track. The expert leads both. These are roster-dependent ratings, anchored to the expert at zero; see the full report for all ratings and outcomes.

## Revision findings

27/33 feedback builds improved from their initial 10-minute snapshot; six worsened. The mean final-minus-initial change was 15.27 points and the median was 10.56.

At equal final 40-minute coding budgets, 4 model means were higher with feedback, 6 lower and 1 unchanged. Within-run improvement includes extra coding time and does not establish a causal feedback benefit.

## Integrity and reliability

- 16,830 scheduled outcomes: 14,850 primary and 1,980 peer.
- 16,380 played games; 450 virtual compilation losses from five intermediate snapshots.
- All 12,254,391 recorded actions passed streaming Rust-engine replay audits.
- All 16,830 unique job indices match the frozen evaluation plan; zero unresolved outcomes.
- All 165 selected source-stage receipts match independent rebuild proofs. The complete preserved-attempt check contains 190 matching proofs: 182 compiled outputs and eight sources that remain invalid.
- Overall outcomes: 14,899 native conclusions, 1,184 engine-cap draws and 297 guest forfeits (289 guest exits, eight fuel exhaustion).
- Final fixed-reference runtime faults: MiniMax’s revision track had 19 forfeits out of 270 outcomes; all other final model/track entries had none. Compiling code is not a guarantee of valid play.

## Recorded API receipts

- Scored authoring builds: $271.8896.
- Excluded infrastructure attempts: $74.9970.
- Compatibility probes: $0.0576.
- This cohort’s preserved-call subtotal: **$346.9442**.
- Earlier invalidated transport-bug cohort: $51.2342 additional; subtotal including that batch: $398.1784.

These are lower bounds from saved completed-call receipts, not an invoice reconciliation. Unsaved/replayed or in-flight requests, other pilots/smokes and Modal compute are additional. Costs never gate a run. The table’s pair average covers the selected independent and revision builds, averaged over three replicates; infrastructure exclusions are separate.

## Reproduction and scope

Authoring source is frozen at `44d3cb9`, protocol `polytopia-mini-v5.1`; final evaluation is `referee-900-stream-v1`. The evaluated finalizer was frozen before map allocation; its source digest is `cdf8ceeb667f9cbcdf20c36abfd892fc010e76d594490f9dc237205b8469fadc`. The source is retained on Modal and in commit `9e2cb59`. Subsequent operator guards and export improvements do not change these scoring records.

All sources, Wasm, complete model trajectories, referee chains and immutable plans remain on Modal volume `polytopia-skirmish`, under `mini-v5/mini-v51-20261030a` and its referenced `mini-runs` directories. The 64-worker smoke matched all 192 outcomes and public trace hashes of the 32-worker baseline; one private expert action sequence varied. Engine replay is verified; repeated policy inference is not claimed bitwise identical across hosts.

- [Full rankings, uncertainty intervals, expert breakdown, Elo, every revision and timing](v51-2026-10-30.md)
- [Compact machine-readable report](v51-2026-10-30.json)
- [Final integrity checks](v51-2026-10-30-integrity.json)
- [Fresh source-rebuild proofs](v51-2026-10-30-rebuild-proofs.json)
- [Run the complete pipeline](PIPELINE.md)
- [Recovery and source-freeze procedures](RECOVERY.md)
