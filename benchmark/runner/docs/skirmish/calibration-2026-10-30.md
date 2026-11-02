# Harness calibration — 2026-10-30

These are calibration trials, separate from the final v3 pilot in
[mini-2026-10-30.md](mini-2026-10-30.md). Raw sources and complete trajectories
remain on Modal; the adjacent JSON contains compact evidence only.

## Resume protocol

The first full cohort exposed an OpenRouter/OpenAI rejection after an early
submission: mini-SWE-agent exits on its terminal command before emitting a tool
result. Resuming that conversation left an unanswered tool call. The adapter now
adds an explicit unknown-outcome stage-boundary receipt for pending calls while
preserving the original trajectory and reasoning fields. A regression test and
a live two-stage GPT-6 Luna high early-submit/resume probe passed. Affected trials
are infrastructure failures, not model losses. The first cohort's recorded games
still passed replay verification: 48 games, 14,429 actions.

## Cumulative fuel

Cohort `mini-20261030b` used 100 billion fuel units/game. Its feedback-track
DeepSeek submission was the unchanged starter. Luna's custom bot beat that starter
twice, but exhausted fuel in three long games; one game reached the engine cap.
Counting forfeits gave the starter a misleading strength ranking for ordinary play.

We reran **all frozen submissions, unchanged, on their exact original maps and
seats**, increasing only fuel to one trillion. The two-second decision limit,
180-second game limit, 100-round cap, 10,000-action cap and memory limit stayed fixed.

| Track | 100-billion head-to-head, DeepSeek perspective | One-trillion head-to-head, DeepSeek perspective |
|---|---|---|
| 20-minute one-shot | 4 wins, 2 losses | 4 wins, 2 losses |
| Three feedback revisions | 3 forfeit wins, 2 losses, 1 unresolved | 0 wins, 2 losses, 4 unresolved |

No new model calls or code changes were used in this controlled comparison.
All 18 one-shot action chains were identical. In the feedback track, 15 action
chains were identical; the other three preserved the complete original action
prefix and continued past the former fuel trap to the engine's turn cap. All
36 rerun games passed replay audits. There were no fuel faults at one trillion.
The expert still won all its games. The new default uses one trillion fuel as a
fail-safe, with decision and game time limits retained.

## Model response allowance

The 12,000-token v2 response ceiling was reached by two saved DeepSeek responses
in the revision track and three in its one-shot track. One Luna one-shot response
also reached it. DeepSeek's revision trial never changed the starter; long requests
were also interrupted by the five-minute window. This does not establish a single
cause for its failure to finish edits.

V3 raises the response allowance to 32,768 tokens, states the allowance to the
agent, and explicitly identifies truncation in format-error feedback. Both API
models support that allowance. The final report uses a **fresh four-cell v3 cohort**,
not a selection of the best calibration outputs. It is still a small integration
pilot, not a causal study of feedback or psychological pressure.

## Reproducibility

The Rust dependency lockfile was recovered from the actual v3 coding image and
committed to the template. Its manifest and starter hashes matched the repository;
future builds fetch those locked versions before entering the offline sandbox.
The locked image passed the live compilation, protected-build and background-cutoff
preflight. Compiler, rules, image, source, transcript and artifact receipts remain
with each trial on Modal staging.
