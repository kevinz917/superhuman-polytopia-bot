# Arm B brief — farsight-v2 architecture + score-at-cap (Daytona)

Paste into the "Farsight Run" session (the cross-session message expired undelivered because of a permission-mode mismatch).

---

User decision (Nov 2 09:30 local, via session "Cap reward design in 1v1 runs"): launch Arm B of the score-at-cap A/B on Daytona — the farsight-v2 architecture with the cap rule changed — and manage it in this thread. Do not extend farsight-v2-1v1 itself; let its final ladder (u9779, 17 boxes) finish and record it.

Context: all three runs (bighero, lookout, farsight-v2) share truncation_reward -1 / cap_score_reward 0. Trace arenas showed bighero never learned mid-game city growth (levels freeze ~t12, 6 builds/game vs the champion's 24); hypothesis: loss-at-cap during the first ~1,000 updates (4-8% capped games then) made elimination the only path to +1. hero-1v1 (cap_score_reward 1.0, old rules) was 2-3x more sample-efficient vs champion. Arm A (bighero's obs-v2 recipe + the fix) is being launched on Curio by "BigHero Run"; lookout is being stopped by "Alternative Arch Run" to free Daytona CPUs (org limit 250 — round 5 of farsight-v2 was refused by it).

Steps:
1. Code: worktree .claude/worktrees/obs-v3-history-03c82d, branch claude/farsight-v2-e0f1fe (published HEAD 01255bc; hero.py has uncommitted keeper fixes + bighero u5250/u7750/u10000 ladder refs — commit them and publish).
2. Config: copy experiments/hero/farsight-v2-1v1.json with exactly one change — cap_score_reward 1.0 and truncation_reward null. Keep no_peace true (score-at-cap without no_peace produced v2's peace collapse), block_pool 2, obs v3, lr 2e-4 + kl_adapt_target 0.01, same random parent /vol/parents/farsight-v2-1v1/random-init.npz (seed may differ; note it). From scratch. New PLANS entry (e.g. farsight-v2-capfix), 4x PRO 6000 learner + 5 PRO 6000 actors as before (~$30/h).
3. Launch only after lookout's sandboxes are deleted (check `hero.py status lookout-1v1` shows no alive learner/actors; "Alternative Arch Run" is doing the stop). If the Daytona API still refuses on the CPU limit, wait for the farsight-v2 final-ladder boxes to finish and retry. Plan 12 h training (~$380 + ladders; user is fine extending if it is ahead), keeper on the Mac under caffeinate with the supervisor-restart pattern, --max-usd ~500.
4. Ladder refs every 2 h: c3, champion, hero1, bighero_u3880/u5250/u7750/u10000, and farsight-v2's own snapshots nearest to its ladder rounds (u2620, 3940, 5140, 6860 — the /vol snapshot cadence is 250, so use the nearest) so every round is an equal-update comparison against the SAME architecture under the old rule. Add farsight-v2 final (u9779) too.
5. Early health: selfplay_truncations share and turns/game over u0-1000 should fall to ~1-3% as under hero-1v1; if the capped share climbs (score race) report at once. Pool-probe as before.
6. Pre-registered gate at ~u5,140: (a) >55% head-to-head vs farsight-v2's u5140-ish snapshot and (b) growth signature in a trace arena (builds/game > 12, city levels still rising at turn 16). Both = cap rule confirmed on the best architecture. Neither = the gap is the rules change.

Please message "Cap reward design in 1v1 runs" at launch (run name, samples/s) and after each ladder round, and record the farsight-v2 final-ladder result when it lands.
