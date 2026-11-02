# Rust bot protocol

Bardur, 1v1, 14x14, polytopia-2.17.3. Each game starts a fresh WASI process.
The bot reads a JSON Observation on stdin, writes {"action_id":"<legal ID>"} plus newline on stdout,
and flushes. Process memory persists between decisions within one game.

Observation fields: player, state_version, visible_state, remembered_state, legal_actions.
Each legal action has a string action_id and command {kind, source, target, parameter}.
Choose an ID from the current legal_actions list, never synthesize one. Tile index=y*14+x.
The visible and remembered state schemas and serializer are included in RULES.md.
Only stdout protocol responses are allowed; debugging belongs on stderr.

Kinds: 1 move, 2 attack, 3 ability, 4 recruit, 5 harvest, 6 build, 7 research,
8 capture, 9 mandatory city reward, 10 end turn, 11 resign.
Ability codes, unit/tech/terrain/structure/resource IDs and all mechanics are in RULES.md.
Unknown tiles and opponent bank/technologies/hidden units are not available to your bot.

Edit /bot/src/main.rs only; helper functions must stay in this file. The dependency set is
Rust std plus serde_json 1.0.145. Compile with:

    cd /bot && cargo build --release --offline --locked --target wasm32-wasip1

The output is target/wasm32-wasip1/release/skirmish-bot.wasm. The controller independently
compiles your frozen source using the original Cargo.toml/Cargo.lock. Agent-produced binaries,
manifest changes, external files and modifications to cached dependencies are not submissions.
Use `skirmish compile`, `maps`, `inspect`, `replay`, `example`, and `check` during coding;
see TOOLS.md. The CLI uses the real engine on nine supplied public maps. It offers
short caller-authored mechanics checks and persistent WASI protocol checks, not an
opponent or automatic practice-match runner. No expert checkpoint is in the sandbox.

Runtime: 256 MiB Wasm memory, one trillion fuel per game, 2 seconds/decision plus a one-time
5-second startup allowance. No preopened filesystem, environment variables or networking.
Game caps: 100 rounds, 10000 actions, 180 referee seconds. Guest faults are forfeits;
v5 engine round/action caps are explicitly benchmark draws. Wall-clock referee timeouts remain unresolved and prevent a complete primary score. Legacy v4 results retain their original unresolved-cap policy.

Maps: drylands (mostly land), lakes (mixed connected land and inland water), continents
(separated landmasses needing naval play). Both seats are evaluated per type.
Feedback games and final games use disjoint seeds. Full feedback files contain each of your
public observations and actions, including public memory, in order. They deliberately omit
private opponent observations; complete trusted action-chain logs are archived by the referee.
No game feedback is available in oneshot-20m. feedback-3r supplies six complete game traces
after each of the first three coding stages. Read /feedback/round-N/report.json and the
referenced JSONL files. Evaluation time is outside your coding clock.


## Protocol v5 timing and evaluation

`independent-40m`: up to 2400 uninterrupted coding seconds, no supplied game feedback.
`feedback-3r-40m`: initial 600 seconds plus three 600-second revisions. Six expert
matches (three map families, both seats) follow stages 0, 1, 2: 18 feedback games.
No extra opponent-match endpoint is supplied. Compilation and self-written tests
consume the clock. Setup, model-client initialization, and scheduled evaluation do
not. API wait/retries count, as do local tests. There is no dollar limit or tool-call
budget. Each shell invocation has a declared 60-second timeout. Each model response
permits 32768 output tokens including reasoning. The agent may explicitly submit
before cutoff; this ends that stage and unused time is not transferred.

Frozen source has no extra byte limit. The independently compiled Wasm must have a
valid header and be at most 8 MiB. The source is captured and hashed before the
trusted compiler runs. Only exact current source is accepted; no automated repair,
best-checkpoint selection or rollback takes place at cutoff.

All three independent builds count. Every stage is evaluated after the whole cohort
is frozen, using identical hidden maps, seats and a fixed evaluation-only opponent
panel. Those results never reach an active authoring agent. Compiler failures score
zero on their scheduled held-out games; executable unchanged starters are explicitly
flagged and receive their actual match results. Infrastructure failures are excluded
and may receive one fresh full-budget retry, with both attempts recorded.

The primary score is mean match points, win=1/draw=0.5/loss=0, equally weighted across
references, map families, seats and builds. Five fresh hidden layouts per family are
used. Infrastructure-unresolved results produce score bounds and block a complete
point estimate. Secondary peer games use matched independent replicate numbers and
one held-out layout per family, both seats. Elo is roster-dependent and secondary.
All actually played scored games are replay-audited in the real engine.


Protocol v5.1 preserves these budgets and scoring rules. Commands are transferred as
files and executed as bash scripts, avoiding Modal's 64 KiB argv limit and Linux's
per-argument limit. The original cutoff receipt and hash are archived before compilation,
including on worker-error exits. Cohorts with the earlier argv transport are marked
invalid infrastructure data and never mixed with corrected runs.
