# farsight-v2 / Arm B hand-off (rebuilt Nov 2 15:15 local after the original worktree was deleted)

The original runs/hero-daytona/farsight-v2-handoff.md was lost with worktree obs-v3-history-03c82d. This file is the
surviving record, assembled from the memory notes; the volume keeps /vol/runs/<run>/{plan,eval-history}.json, pool-probe/,
and the arena JSONs. Worktree now: .claude/worktrees/farsight-v2-capfix (branch claude/farsight-v2-e0f1fe).

## farsight-v2-1v1 (done)

Branch `claude/farsight-v2-e0f1fe` (worktree `.claude/worktrees/obs-v3-history-03c82d`) = farsight's obs-v3 branch + merge
of `claude/lookout` (Daytona harness, from-scratch plans, sync-farsight; the attention trunk comes along dormant) +
`block_pool` layout 2 (`network.BLOCK_POOL_LAYOUTS`): spec `block_pool: 2`; LayerNorm'd (mean, max) of 32 channels of the
block's normalized first conv -> dense -> scalar gate (init 0) -> added AFTER the block's LayerNorm, blocks 1 and 3 of 4.
farsight's `block_pool: true` = layout 1 still loads. `trunk(params, x, pool_stats=[])` reports bias_rms/conv_rms/gate;
`experiments/daytona/pool_probe.py` / `hero.py pool-probe RUN` measure it on self-play positions. Doc section in
docs/obs-v3-history-2026-11-01.md. Tests (60) pass on Daytona (`hero.py tests`, the user's choice over Modal).

Why no conversion: farsight's final net (u5660) has bias/conv ratios 4.7/7.5/5.5/4.0 per block (max 9.8); switching the
pooling off moves the family policy by 1.29 nats KL. farsight ended with lr_scale 0.44 and KL still 0.017.

Daytona state (Nov 1 ~19:45 CST): code 01255bc published; layout-2 random parent /vol/parents/farsight-v2-1v1/random-init.npz
(2,036,773 params, seed 1); refs on the volume: /results/distributed/ladder/bighero-u3880.npz, farsight-latest.npz (= u5660).
Plan `farsight-v2-1v1` (hero.py): 4x PRO 6000 spot + 5 PRO 6000 actors, $30.21/h; config experiments/hero/farsight-v2-1v1.json
(bighero recipe, lr 2e-4, kl_adapt_target 0.01, seed 821, send_thread off, serialize_gpu on). User decision (via the parent
session): 12 h (~$380), from scratch; extension later if ahead of bighero at equal updates. LAUNCHED Nov 1 19:06 CST (training_seconds 43200 -> ends ~07:06 CST Nov 2, ladders after); learner
farsight-v2-1v1-learner-30792 + actors 0-4 all PRO 6000, GPU checks ok. Keeper: `caffeinate -i hero.py keep farsight-v2-1v1
--max-usd 450` on the Mac, log runs/hero-daytona/keep-farsight-v2.log; launch log launch-farsight-v2.log.

**How to apply:** compare ladder rounds with bighero u2060 (c3 19%) and farsight u1750/3580/5360 (c3 0/1.6/12.5%,
bighero_u3880 0/12.5/23%); run `hero.py pool-probe farsight-v2-1v1` at each ladder; stop rule bias > ~0.5x conv or
lr_scale < 0.8. Related: [[farsight-run]], [[lookout-run]], [[daytona-hero]], [[tests-on-daytona]].

Nov 1 19:35 CST decision (user granted monitoring autonomy, see [[autonomous-run-monitoring]]): the run was actor-bound at
~50k samples/s (learner ready_wait 3.5 s of a 6 s update, actor GPUs 15%, actor boxes pinned at their 16-vCPU quota;
~10k/s per PRO 6000 actor vs ~29.5k/s for hero-1v1 obs v2 on the same hosts on Oct 28, so obs v3 encoding costs ~3x on
the actor side). Raised the plan ledger to 8 actors (keeper creates missing actors itself; actors 5-7 up 19:33) and
restarted the keeper with --max-usd 560; projected total ~$510 at $41/h. Pool-probe at update 200 (layout 2 baseline):
ratios 0.17 / 0.03 (blocks 1 / 3), gates 0.11 / 0.03, switched-off family KL 0.0015 nats (farsight final: 1.29).
Learner at u160: kl 0.0046, lr_scale 1.0, clip 0.046, grad 0.12-0.18.

Nov 1 20:50 CST correction: the 8-actor change was WRONG and was reverted (actors 5/7/9 deleted, plan back to 5, keeper
--max-usd 450). Throughput stayed ~47k/s with 8 actors. Diagnosis (py-spy on actors + learner): all actor workers block on
`outbox.put` (queue maxsize 2) at distributed.py:1232; sender threads wait in `exchange` for the learner's reply; the
learner's 392 handler threads sit in `_recv_exact`/`inbox.put` with `inbox` qsize 128 = full; the learner process uses
~1-2 cores (GIL): receiver zstd decompress 1.5-2.8 s + assembler DeviceRows streaming (asm_stream 0.8-2.8 s) per update,
asm_wait 2.4-6 s, train 2.25 s. Actor boxes: 48 workers pinned one core each over host CPUs 2-15, server on 0-1 (the
"worker affinity 0,1" reading was the server process), host CPUs 25% busy, GPU 0-15%: actors idle, not CPU-bound. Oct 28
hero-1v1 (obs v2) ingested 120k/s on the same recipe, so obs v3's bigger views roughly halve the GIL wall. Levers for a
future run, not mid-run: coded_views (Curio 1.8x; test view_codec with obs v3 first), send_thread stays off on PRO 6000
(hangs, docs/platforms.md), lower-level receiver work (A6 memory: 117k/s capacity with host pages). 12 h at 47k/s ≈ 2.0 B
samples ≈ 7,700 updates, enough for the bighero u3880 / farsight u5360 comparisons. Spent ~$30 by 20:50.

Nov 1 21:55 (local, UTC+8) ladder round 0 (u1240, 21/24 jobs, 64 games/ref with 2 seats): vs c3 0 W / 21 L / 43 capped,
bighero_u3880 2 W / 17 L / 45 capped, farsight(u5660) 3 W / 8 L / 53 capped, bn4 0/43, champion 0/48, hero1 0/46 (48 of 64
in). Same as farsight u1750 (0% vs c3); bighero's 19% reference is at u2060, so round 1 (~u2500) is the first real read.
Pool-probe trend (ratio = bias_rms over the unit-RMS normalized conv, layout 2 adds the bias AFTER LayerNorm): u200
0.17/0.03 gates 0.11/0.03; u1740 0.49/0.45 (max 0.55/0.51) gates 0.235/0.21, switched-off family KL 0.071 nats (farsight
1.29). Training metrics unaffected (KL 0.0057, lr_scale 1.0, clip 0.06, grad 0.16, 49k/s). My decision: the hand-off's
0.5 stop rule was a heuristic for layout 1's failure mode; post-LN the ratio cannot hijack normalization, so continue
with hourly probes; stop/rethink only if ratio_mean > ~1.5, lr_scale < 0.8, KL ramps, or round 1 is clearly behind
bighero u2060 (19% vs c3). Note: `date` on this Mac prints CST = China time (UTC+8), not Chicago.

Nov 1 23:45 local: keeper crashed on a ~10-min Daytona API outage (http() 5 retries exhausted; cmd_keep has no
per-iteration try/except) -> now runs under runs/hero-daytona/keep-farsight-v2-supervisor.sh (restart on non-zero exit).
Eval rounds launched during API flakiness can leave untracked boxes that still run their jobs (ttl 180 min cleans them).
Probe u2480: 0.52/0.44 (plateau vs u1740). Round 1 = u2620 launched ~23:10 local.
Nov 1 23:50 local: keeper cap raised 450 -> 500 (supervisor script): $171 spent at 4.7 h once 48 eval boxes are
counted; projection $440-460 for the full 12 h + final ladder, and a cap trip would cut the run short for a few dollars.

Nov 2 00:12 local, ladder round 1 (u2620, 23/24 jobs): vs c3 17.2% (11 W / 6 L / 47 capped of 64), bighero_u3880 46.9%
(30 W / 15 L / 19 capped), farsight u5660 60.9% (39 W / 7 L), bn4 2.1%, champion 3.1%, hero1 7.8%. References: bighero
u2060 = 19% vs c3; farsight u1750/3580 = 0/1.6% vs c3 and 0/12.5% vs bighero_u3880. => layout 2 fixed the pooling
failure: v2 at u2620 beats farsight's final net 39-7 and is near parity with bighero's u3880 checkpoint, at bighero pace
vs c3. Probe u3200: 0.54/0.43, gates .254/.210 (flat since u1740). Spend $188 at 5 h ($46/h while 48 eval boxes live).

Nov 2 02:57 local, ladder round 2 (u3940, 23/24): vs c3 15.6% (10 W / 5 L / 49 capped), bighero_u3880 56.2% (36 W / 8 L /
20 capped) = ahead of bighero at EQUAL updates head-to-head, farsight u5660 78.1% (50-1), bn4 6%, champion 4%, hero1 3%.
vs c3 is flat (17 -> 16%) because ~75% of those games hit the cap (both lose under this recipe). Spend $297 at 8 h.
Probe u3880 unchanged (0.54/0.43). Keeper fixes (uncommitted hero.py): eval box create retried 3x, collect timeout 100 min.
Nov 2 03:10: bighero at u12300 on Curio; synced its u5250 and u7750 snapshots to the volume
(/results/distributed/ladder/bighero-u5250.npz, -u7750.npz) and added them as ladder references (hero.py, uncommitted) so
round 3 and the final ladder compare at equal updates. Extension decision at ~07:10 after the final ladder.

Nov 2 03:59 local, ladder round 3 (u5140, 32/32 parts): vs c3 31.2% (20 W / 4 L / 40 capped), bighero_u5250 (EQUAL updates)
71.9% (46 W / 7 L), bighero_u7750 65.6% (42 W / 15 L), bighero_u3880 73.4%, farsight u5660 87.5% (56-4), champion 9.4%,
hero1 7.8%, bn4 4.7%. => layout 2 on obs v3 is clearly ahead of bighero at equal updates (and beats bighero 2,600 updates
older). Extension condition met; decision deferred to the user at the 12-h mark (resumable from final.npz, no cost to wait).

Nov 2 04:25: learner spot-reclaimed ~03:30; keeper resumed it on the 9th host (GPU check rejects psum_8mb_ms > ~100 ms)
at 03:51. FRESH learner + actors run ~90k samples/s vs ~47k/s before -> the learner process degrades over hours (57 GB RSS,
900 threads); periodic learner restart = cheap lever (see [[rl-sampling-bottlenecks]]). Learner clock ends ~07:42 local,
so plan until_unix was moved to 08:15 and actor nodes restarted with it. bighero u10000 synced as a reference for the final.

Nov 2 05:46 local, ladder round 4 (u6860, 36/36): vs c3 48.4% (31-5, 28 capped), bighero_u10000 68.8% (44-11), u7750 60.9%,
u5250 76.6%, u3880 82.8%, farsight final 96.9% (62-1), hero1 14.1%, bn4 9.4%, champion 7.8%. Spend $384 at 10.8 h.
Nov 2 ~06:50 local: u8860, $418 spent; projected ~$485-500 incl. round 5 + final ladder, so the keeper cap went 500 -> 560
(a budget_stop would cut the LR-decay tail). Probe u8860: 0.59/0.44, gates .254/.196 (flat all run).

Nov 2 09:30 local: DONE at u9779 (12 h learner clock, ~2.6 B samples). Partial final ladder (128 games/ref): c3 56.2%,
champion 21.1%, hero1 18.0%, bn4 10.9%, bighero_u3880 81% (32 games). The equal-update bighero refs + farsight were missing
(Daytona org CPU cap ~250 vCPU: round 5's 36 boxes blocked 19 of the final's 36 creates -> "Total CPU limit exceeded");
round 5 (u9400) itself yielded 0 outputs. Relaunched the 19 missing jobs by hand at 09:27; tally pending. Keeper stopped,
fleet gone, spend $473 + ~$10. Lesson: never launch a ladder round while the previous round's boxes are alive.

FINAL (u9779, 128 games/ref, Nov 2 10:20 local): bighero_u10000 79.7% (102-17), u7750 74.2%, u5250 82.8%, u3880 85.9%,
farsight(v1) 96.9%, c3 56.2% (72-3, 53 capped), champion 21.1%, hero1 18.0%, bn4 10.9%. => v2 beats bighero at equal
updates 80-20. Total ~$483. Everything on Daytona deleted. Next: Arm B 'farsight-v2-capfix' (same recipe, cap_score_reward
1.0 / truncation_reward null) per docs/capfix-arm-b-brief-2026-11-02.md.
Nov 2 15:10: local ledgers and the hand-off doc of this run were lost with the deleted worktree; the volume keeps
/vol/runs/farsight-v2-1v1/eval-history.json (rounds 0-5) + arena JSONs (final 36 files) + pool-probe dir; the memory notes
above are the surviving narrative. New worktree for this branch: .claude/worktrees/farsight-v2-capfix.

## Arm B: farsight-v2-capfix (running)

Arm B (brief docs/capfix-arm-b-brief-2026-11-02.md, decided in session "Cap reward design in 1v1 runs",
local_a9471176-6075-4ebd-8992-50d9b40d41c4): `farsight-v2-capfix` = farsight-v2-1v1's config with exactly one change,
cap_score_reward 1.0 / truncation_reward null (a capped game pays the score reward instead of a loss for both seats), seed
821, byte-identical random parent. Hypothesis: loss-at-cap in the first ~1,000 updates taught the from-scratch runs that
elimination is the only path to +1 (bighero never learned mid-game city growth). Arm A = bighero's obs-v2 recipe + the fix
on Curio (session "BigHero Run", commit 394b0cf capfix-a.json).

Launched Nov 2 10:30 local; the first learner died and the second deadlocked after update 1 (4-GPU page-streaming signature) ->
stream_views off (commit d43c531 published; plan ledger patched too), fleet restarted 11:50 from the random parent: learner
farsight-v2-capfix-learner-90760 (4x PRO 6000) + actors 0-4, 93k samples/s (fresh learner), training ends ~23:50 local,
12 h, ~$30/h; keeper runs/hero-daytona/keep-capfix-supervisor.sh (--max-usd 500). Ladder every 2 h, 16 maps/box (24 boxes,
fits the org's 250-vCPU cap) vs c3, champion, hero1, bighero u3880/5250/7750/10000, v2 u2620/3940/5140/6860/final.

**How to apply:** early health = capped-game share and turns/game over u0-1000 should fall to ~1-3% (a climbing share =
score race, report at once); pool-probe hourly; every ladder round compare against the v2 snapshot nearest the update.
Pre-registered gate at ~u5140: (a) > 55% vs v2_u5140 and (b) growth signature in a trace arena (builds/game > 12, city
levels rising at turn 16). Message the cap-reward session at launch, after each round, and at the end. Related:
[[farsight-v2-run]], [[autonomous-run-monitoring]], [[daytona-hero]].
Early health: u40 capped share 47% (18/38), 159 turns/game (random-policy start). Actor nodes carry until_unix 22:45;
plan says 00:30 Nov 3 -> `restart --actors-only` before ~22:00. Gotchas: cmd_restart and keeper resumes take the config
from the plan ledger, not the JSON; never pkill -f a pattern that appears in a background shell's own command text.
Nov 2 13:00: early health PASSED: capped share 47% (u40) -> 0% (u380) -> 12% (u700, 25 games) -> 1.8% (u1020) -> 0% (u1360);
turns/game ~52; KL 0.004-0.006; 88k samples/s. Probe ~u900: 0.37/0.40, gates 0.19/-0.20. Round 0 launched ~12:30 (~u800).
Round 0 (u940): 0% vs everything (expected at this age). Growth metric added to cpu_arena.py (commit 7010fa1): `growth`
= builds/harvests per side + cities/mean level at t16 and end; plan sha -> 7010fa1. The cap-reward session is archived
(messages bounce) -> report to the user directly.
Nov 2 15:10: the worktree obs-v3-history-03c82d was DELETED by another session/app cleanup while the run was live:
local ledgers, keeper logs, pool-probe jsonl and runs/hero-daytona/farsight-v2-handoff.md are gone (runs/ there was a
real dir, not the shared symlink). Rebuilt in a new worktree .claude/worktrees/farsight-v2-capfix (branch
claude/farsight-v2-e0f1fe, now at 5471a83 incl. another session's Arm C commit): plan from /vol/runs/<run>/plan.json
(sha 7010fa1), state from the live sandboxes (round 1 = update-3540 pending on 24 boxes, next_eval 16:37), eval-history
from the volume; keeper restarted 15:08. Spend counter reset to an estimate ($150 at 15:05).

## Keeper / commands (new worktree)

cd .claude/worktrees/farsight-v2-capfix; keeper: runs/hero-daytona/keep-capfix-supervisor.sh (caffeinate, --max-usd 500),
logs runs/hero-daytona/keep-capfix.log; hero.py status|pool-probe|restart farsight-v2-capfix. Plan ledger sha 7010fa1,
until_unix 00:30 Nov 3; actor nodes still carry 22:45 -> restart --actors-only before ~22:00 (keeper paused around it).

## Arm B round 1 (u3540, 24/24, 15:24 local)

| ref | W | L | capped | win% | decided% | v2 at nearby update |
|---|---|---|---|---|---|---|
| c3 | 20 | 19 | 25 | 31.2 | 51 | v2 u3940: 15.6% (10-5-49 capped, 67% decided); u5140: 31.2% (20-4-40, 83%) |
| bighero_u3880 | 30 | 18 | 16 | 46.9 | 62 | v2 u3940: 56.2% (36-8, 82%) |
| bighero_u5250 | 29 | 22 | 13 | 45.3 | 57 | v2 u5140: 71.9% (46-7) |
| bighero_u7750 | 23 | 25 | 16 | 35.9 | 48 | v2 u5140: 65.6% |
| bighero_u10000 | 15 | 31 | 18 | 23.4 | 33 | v2 u6860: 68.8% |
| v2_u2620 | 11 | 30 | 23 | 17.2 | 27 | |
| v2_u3940 (equal updates) | 9 | 31 | 24 | 14.1 | 22 | |
| v2_u5140 (gate ref) | 9 | 41 | 14 | 14.1 | 18 | gate: > 55% at ~u5140 |
| v2_u6860 / v2_final | 7 / 5 | 45 / 47 | 12 / 12 | 10.9 / 7.8 | | |
| champion / hero1 | 9 / 7 | 53 / 56 | 2 / 1 | 14.1 / 10.9 | | v2 u5140: 9.4 / 7.8 |

Read: Arm B caps far fewer games (25 vs 49 vs c3) but loses most decided games to its loss-at-cap sibling at equal updates
(9-31 vs v2_u3940). Caveat: the arena scores a capped game as no-result, i.e. the old rule's framing; under Arm B's own
objective a capped game it leads on score is a win. Round 2 (~u6100, boxes on sha 7010fa1) adds the growth fields.

Arm B transient at u~4780-4930 (15:35-15:45): kl_ema peaked ~0.019 (target 0.01) -> lr_scale cut to 0.25 at u4840, recovered to
1.0 by u4928 as KL fell to 0.003; entropy, grad norm and value loss unremarkable. The adaptive scale did its job; unlike
farsight (sustained ramp, lr_scale stuck at 0.44) this resolved in ~90 updates. Watch for recurrence at pool rotations
(historical_every 250). Health at u5060: kl 0.0067, lr 1.0, capped 0/70, 98k samples/s.

## Arm B round 2 = GATE read (u6200, 24/24, 17:29 local)

| ref | W | L | capped | win% | decided% | v2 reference point |
|---|---|---|---|---|---|---|
| v2_u5140 (gate a: > 55%) | 14 | 32 | 18 | 21.9 | 30 | FAIL |
| v2_u6860 (equal updates) | 15 | 40 | 9 | 23.4 | 27 | |
| v2_u3940 / v2_u2620 | 18 / 28 | 24 / 17 | 22 / 19 | 28 / 44 | 43 / 62 | |
| v2_final | 10 | 40 | 14 | 15.6 | 20 | |
| c3 | 26 | 8 | 30 | 40.6 | 76 | v2 u5140 31%, u6860 48% |
| champion | 15 | 48 | 1 | 23.4 | 24 | v2 u6860 7.8%, final 21% |
| hero1 | 15 | 48 | 1 | 23.4 | 24 | v2 u6860 14%, final 18% |
| bighero_u3880 | 43 | 9 | 12 | 67.2 | 83 | v2 u5140 73% |
| bighero_u5250 | 44 | 15 | 5 | 68.8 | 75 | v2 u5140 72% |
| bighero_u7750 | 29 | 18 | 17 | 45.3 | 62 | v2 u5140 66%, u6860 61% |
| bighero_u10000 | 24 | 21 | 19 | 37.5 | 53 | v2 u6860 69% |

Read: Arm B at u6200 is as good as v2 against bighero's early snapshots and clearly better against the mature Modal bots
(champion/hero1 23% vs v2's 8-14% at the same age), but loses the direct head-to-head to v2 at equal updates (15-40) and
to v2_u5140 (14-32). Gate (a) fails under the arena's no-result scoring of capped games.
Gate (b) growth, round 2 (cand = Arm B u6200 | opp): vs v2_u5140 builds 49.6 | 24.6, harvests 13.2 | 13.7, t16 cities 5.42 | 5.19,
t16 level 2.18 | 2.34, end cities 4.89 | 6.47, end level 3.30 | 3.13; vs v2_u6860 builds 36.9 | 25.7, t16 level 2.17 | 2.34,
end cities 4.06 | 7.30. vs c3: builds 51 | 17, t16 level 2.23 | 1.84, end cities 7.67 | 3.53. vs champion: builds 30.8 | 26.3,
t16 level 2.13 | 2.53. Read: Arm B issues 1.5-2x more build actions than v2 but its cities are slightly LOWER level at turn 16
(2.18 vs 2.34) and it ends with fewer cities when it loses; no growth advantage over the same architecture under the old rule.
Gate verdict (both parts): (a) fail, (b) no. Per the brief: the v2-vs-bighero gap is not explained by the cap rule.
Nuance: Arm B is the better bot against the mature Modal references at this age (champion 23% vs 8%), equal vs bighero.

User decision (18:10 local): stop Arm B at 22:00 tonight; summary afterwards; next steps are the user's call. Stop armed
for 21:58 (`hero.py stop farsight-v2-capfix`; the keeper then runs the final ladder and tears down). Actor refresh not needed.

20:35 local: Arm B round 3 (u8750) was written with 0/24 parts: all 24 boxes created, frozen checkpoint fine, zero arena
outputs, boxes already deleted by collect (cause unverified; another session's Arm C now runs on the same org, 48 vCPU, and
an external cleanup of eval boxes is possible). With 196/250 vCPU in use, round 4 would get ~13 boxes and then crowd the
final, so round 4 is skipped (state next_eval +10 h) and the keeper now deletes the fleet BEFORE launching the final
(commit on the branch; keeper restarted 20:36). Stop at 21:58 stands; the final ladder is the last measurement.

## Arm B FINAL ladder (stopped 21:58 at u13000; final checkpoint u13014; 128 games per reference; 22:45 local)

| ref | W | L | capped | win% |
|---|---|---|---|---|
| c3 | 111 | 5 | 12 | 86.7 |
| champion | 59 | 66 | 3 | 46.1 |
| hero1 | 53 | 74 | 1 | 41.4 |
| bighero_u3880 / u5250 / u7750 / u10000 | 113 / 113 / 106 / 99 | 10 / 13 / 16 / 22 | 5 / 2 / 6 / 7 | 88 / 88 / 83 / 77 |
| v2_u2620 / u3940 / u5140 / u6860 | 91 / 85 / 83 / 76 | 17 / 32 / 36 / 47 | 20 / 11 / 9 / 5 | 71 / 66 / 65 / 59 |
| v2_final (u9779) | 61 | 57 | 10 | 47.7 |

Compared with v2's final (u9779): c3 56% -> 87%, champion 21% -> 46%, hero1 18% -> 41%, bighero_u10000 80% -> 77%; head-to-head
with v2's final: parity (61-57). Arm B had 33% more updates (13,014 vs 9,779; 3.41 B vs 2.6 B samples in 10 h thanks to
90k samples/s), so this is not an equal-update comparison; v2's own trajectory was champion 8% (u6860) -> 21% (u9779).
Revised read: the gate at u6200 failed on the sibling head-to-head, but score-at-cap learns faster against opponents it never
trained on and catches v2 head-to-head by the end. Spend ~$385 (keeper-estimated).

# Arm B segment 2 (Nov 3, resumed from u13014 for 12 h)

User decision (Nov 3 morning; launched 07:30 local): continue Arm B to ~6.9 B samples as the equal-sample test against hero-1v1 (the plain
4x128 trunk under the same score-at-cap rule). Commit 63722ab published. Resume: final.npz (u13014) promoted to latest.npz,
STOP + learner-done.json cleared, `launch farsight-v2-capfix --hours 12`: training_seconds 79,643 (elapsed 36,444 + 43,200),
LR 2e-4 again, decaying over the segment's last 25%; stream_views off; cap_score_reward 1.0. Ladder every 2 h, 16 boxes:
c3, champion, hero1 (hero-1v1 final, 3.9 B), hero1_u11500 (continuation ~6.9 B), hero1_final (u17840), bighero_u10000,
v2_final, self_u13014. Keeper: keep-capfix-supervisor.sh --max-usd 450 (segment-2 spend only; segment 1 was ~$385).
Plan: restart the learner once at ~6 h (hero.py restart, keeper paused) to keep intake at ~90k samples/s.
Success = above 68% vs champion and > 50% vs hero1_u11500 at ~6.9 B; then this is the strongest 1v1 network.

Segment 2, 08:30 local: resumed cleanly from u13014 (KL 0.005 at the first updates, 105k samples/s), but Daytona reclaimed
spot actors 8 times between 07:40 and 08:22 (no "lost" lines: evicted sandboxes vanish from the list). Each replacement
coincides with a KL spike (u13080 ema 0.026, u13480-13520 kl 0.035-0.038, u14100 kl 0.024) that cuts lr_scale to 0.25 for
~60 updates: a fresh actor's 1,152 games all start at turn 0 and flood the batch with opening positions. Mitigation: the plan
ledger's actor spec now has spot: false, so each further eviction is replaced by an on-demand box (the ledger's price table
has no on-demand rate, so the keeper's $ will under-count; expect roughly 2x the GPU share, ~+$2/h per on-demand actor).

Segment 2, 08:50 local: ROOT CAUSE of the actor churn = the Daytona org has NO GPU CREDITS ("Organization doesn't have GPU
credits. Add more by visiting the Wallet page", HTTP 400 on every create since ~08:27). The "evictions" from 07:40 on were
most likely sandboxes being torn down as the wallet ran dry; replacements worked until ~08:25, then failed. The keeper
exhausted its replacement budget (15) on failed creates and stopped; fleet = learner + 2 spot actors (~40k samples/s).
Fixes: actor spec back to spot (on-demand was never the issue), state.replaced reset, keeper patched (commit on the branch)
so refused creates don't consume the budget and a credits error ends the loop's attempts. NEEDS THE USER: top up Daytona
GPU credits on the Wallet page; the keeper then refills the fleet by itself. If the learner itself gets torn down, the
keeper resumes it from latest.npz once credits exist (checkpoint every 20 updates). Snapshots u13250..u14500 exist in the
run dir if the 2-actor / spike window (u14100-14600, kl to 0.05 at lr 0.25) turns out to have hurt: roll back to u14000.

Segment 2, 09:05 local: with 2 actors the learner sat at kl 0.04-0.11 / lr_scale 0.25 for 150+ updates (correlated batches),
so the fleet was taken down at u14660 and the resume point rolled back to snapshot u14000 (latest.npz/json + critic =
update-14000; training.jsonl trimmed to u14000, the damaged tail kept as training-seg2-damaged-u14660.jsonl). Daytona is
empty. A 10-minute probe retries `launch farsight-v2-capfix --hours 11.3` (remaining clock from u14000) until the org has
GPU credits again, then starts the keeper. Segment-2 spend so far ~$30 (keeper) for ~1,000 net updates kept (13014 -> 14000).
Monitoring cadence from now: every 10 minutes while a fleet is live.
08:56 local: GPU quota back (org usage shows totalGpuQuota 10 in region 'earth'; a probe GPU box created and deleted fine).
Relaunch from u14000 for 11.3 h started 08:57; keeper after; then `python3 runs/hero-daytona/check-capfix.py` every 10 min.
09:00 local: relaunched from u14000 (commit ab353bd published): learner farsight-v2-capfix-learner-67494 + actors 0-4, keeper
--max-usd 450. 10-minute checks via runs/hero-daytona/check-capfix.py; first ladder ~11:00; learner restart reminder 13:30.
12:20-12:30 local: keeper hardened (cmd_keep -> _keep_once; an iteration's API failure logs keeper_error and retries after
60 s; commits 5dba9f5 + 4d4d93e, NOT published: the running fleet's plan sha stays ab353bd). The shared hero-hub sandbox had
disappeared (TTL from Nov 1 04:13 + 36 h default); recreated with `hero.py hub --ttl-hours 120` (id b0c48e16). Lesson:
the hub's TTL must outlive every run that depends on it; check `hero.py status` for the hub before launches.
13:15 local: segment-2 round 0 (u16580) was deleted at the 100-min collect timeout with 0/16 outputs, like u8750 yesterday.
Live inspection of the next round's boxes (u19460, launched 13:02): arena processes running, code ab353bd, each box getting
2.9-3.6 of its 4 cores, but the hosts carry load averages of 450-600 on 64 cores. Hypothesis: on busy hosts 16-map boxes
take > 100 min; collect timeout raised to 180 min (commit ece0795, keeper restarted 13:14). Lost rounds cannot be recovered.

Segment 2 round 1 (u19460, 5.1 B, 16/16, 13:45): c3 70.3% (45-1-18 capped), champion 42.2% (27-32), hero1 34.4% (22-34),
hero1_u11500 20.3% (13-45), hero1_final 14.1%, bighero_u10000 62.5%, v2_final 40.6% (26-24), self_u13014 39.1% (25-28-11).
Every comparable column is below the u13014 final ladder (champion 46, hero1 41, c3 87, bighero_u10000 77, v2_final 48; those
had 128 games vs 64 here) and the head-to-head with its own segment start is parity. Possible erosion from the warm restart
at lr 2e-4 (segment 1 had decayed to ~1.6e-4), or noise. Decision pending the self-play-pool trend and the next round.
13:55 local: self-play pool win rate (current policy vs its recent snapshots) was 52.7-54.5% through segment 1's climb and
50.9-52.7% since the resume, with KL/entropy normal: the warm restart at 2e-4 churns more than it improves. Switched the
learner to lr 1.5e-4 -> 4e-5 annealed over the remaining 6.7 h (plan ledger + volume plan; learner-only restart on the same
sandbox at u20540, actors kept). The final ladder then measures "Arm B + polish" at ~6.9 B, comparable to hero-1v1's
continuation, which also ended on a decayed LR.
14:00-14:25 local: after the learner-only restart the actors did not reconnect (tunnel_stale/tunnel_down cycling; the
learner listened on 17070 with no connections for 15 min). Remedy: fresh ssh token + restart_actor_nodes; actors 4/5 took it,
actors 0/2/3 hit TLS errors mid-restart, lost their node, and were deleted + replaced by the keeper (6, 7, +1). Cost: ~25 min
without training plus the replacement compile. Lesson: a learner-only node restart needs restart_actor_nodes right after
(as cmd_restart does); do not skip the actor step.
14:50 local: Daytona's API is dropping TLS on a large share of calls today; keeper launches leave boxes bootstrapped but
without a started node (actors 6/7 earlier, 8/9 now), which the keeper only notices after 15 min. Started 8/9 by hand.
Learner (71279) fine: u21000, lr 1.34e-4 annealing, KL 0.004, 2 actors connected (64k/s). The 14:10 learner_stalled line
was a false alarm during the actor outage (the delete failed on TLS and the learner kept running).

Segment 2 round 2 (u21540, 5.6 B, 16/16, 16:01; frozen ~1,000 updates into the anneal): c3 78.1% (50-1), champion 42.2%
(27-32), hero1 45.3% (29-29), hero1_u11500 31.2% (20-37), hero1_final 20.3%, bighero_u10000 81.2% (52-9), v2_final 45.3%
(29-23), self_u13014 59.4% (38-24-2). Up on every column vs u19460 (self 39 -> 59, hero1 34 -> 45, bighero 62 -> 81, c3
70 -> 78); champion flat at 42. The segment is now clearly ahead of its own start; the equal-sample target (> 50% vs
hero1_u11500) is still far at 31%.

Segment 2 round 3 (u24460, 6.4 B, 16/16, 18:00): c3 82.8% (53-0), champion 57.8% (37-25), hero1 50.0% (32-30),
hero1_u11500 34.4% (22-38), hero1_final 34.4% (22-38), bighero_u10000 87.5% (56-6), v2_final 57.8% (37-16), self_u13014
71.9% (46-17). Champion 42 -> 58 in one round; hero1 (3.9 B) now level; hero1_u11500 still 34%.

Segment 2 round 4 (u27400, 7.2 B, 16/16, 19:50): c3 82.8% (53-3), champion 50.0% (32-31), hero1 45.3% (29-32), hero1_u11500
35.9% (23-37), hero1_final 37.5% (24-36), bighero_u10000 79.7%, v2_final 54.7% (35-22), self_u13014 60.9% (39-24). Within
64-game noise of u24460 (champion 58 -> 50, self 72 -> 61); plateau at ~50% champion / 45% hero1 / 36% hero1_u11500.

20:35 local: training ended at u28957 (7.6 B samples); the learner wrote final.npz + learner-done and the keeper tore the
fleet down, but it SKIPPED the final ladder: the state file still had final_eval from segment 1 (launch updates the state
instead of replacing it) and the keeper exited "finished". Fixed in hero.py (launch unlinks the state file first; commit on
the branch), final_eval cleared by hand and the keeper restarted at 20:47 to launch + collect the final ladder (16 boxes,
128 games/ref). Segment-2 keeper spend $368 (+ ~$30 pre-rollback), total Arm B ~$780 across both segments.

## Arm B segment 2 FINAL (u28957, ~7.6 B samples, 128 games per reference, 21:50 local Nov 3)

| reference | W | L | capped | win% |
|---|---|---|---|---|
| champion | 72 | 47 | 9 | 56.2 |
| hero1 (hero-1v1 final, 3.9 B) | 61 | 56 | 11 | 47.7 |
| hero1_u11500 (continuation ~6.9 B) | 50 | 73 | 5 | 39.1 |
| hero1_final (u17840) | 46 | 73 | 9 | 35.9 |
| bighero_u10000 | 109 | 14 | 5 | 85.2 |
| v2_final (u9779) | 58 | 39 | 31 | 45.3 |
| self_u13014 (segment-1 end) | 88 | 39 | 1 | 68.8 |
| c3 | 108 | 4 | 16 | 84.4 |

Growth (final round, Arm B | opp): vs hero1_u11500 t16 cities 5.97 | 5.37, level 2.49 | 2.68, end cities 4.88 | 7.02; vs champion
t16 6.17 | 5.16, level 2.47 | 2.47; vs self_u13014 t16 5.94 | 5.33, level 2.48 | 2.32. Arm B now out-expands everything but
the hero-1v1 continuation. Verdict: the pooling+history trunk under score-at-cap matches the plain trunk at equal samples
near 4 B (48% vs hero1) and trails it at 7 B (39% vs hero1_u11500); the segment gained +20 pts vs its own start and +10 vs
the champion (46 -> 56). Total Arm B spend ~$780 (seg 1 ~$385, seg 2 ~$400 incl. false start). Nothing left on Daytona.
