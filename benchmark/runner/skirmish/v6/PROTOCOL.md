# Local v6 coding and game protocol

This is `polytopia-local-v6`, implemented with Mini-SWE-agent 2.4.6, host CLI-proxy inference, and offline containers in the local VM. No paid API fallback is allowed. The task receipt specifies the active coding duration; defaults are 40 minutes, while explicitly labeled pilots can be shorter.

There are three equal-total-coding-time tracks:

- `independent`: one continuous coding stage without supplied opponent feedback.
- `segmented-control`: four equal coding stages, retained conversation/workspace, no opponent feedback.
- `feedback`: four equal coding stages. Six matches follow each of stages 0, 1 and 2, balanced across drylands/lakes/continents and both seats, using a preregistered rotation of frozen mixed-style opponents and hero. Full public observations/actions are supplied; private reference state is withheld.

Scheduler CPU queue waits, setup, compilation after cutoff, and feedback generation are outside the coding clock. API latency/retries, tool execution, feedback reading, and self-written experiments consume coding time. The per-response maximum is 32768 output tokens including reasoning. A model request can use the remaining stage budget; it has no separate 60-second timeout. The controller cancels the request process group at cutoff and atomically preserves the conversation, including an initial receipt before the first request. There is no tool-call count, model-step or dollar gate. Providers' reasoning settings do not imply equal inference compute. CPU scheduler waits are separately recorded.

Edit only `/bot/src/main.rs`, using Rust std and serde_json. Compile offline with the original Cargo manifest. Current source is captured at the deadline, independently rebuilt, and scored even if broken; no rollback or model repair. All authored builds count. Read `/reference/TOOLS.md` and the implementation-defined rule book. The rule book's embedded legacy protocol is superseded by this file and the active task receipt.

The bot reads one JSON Observation per stdin line and writes exactly `{"action_id":"<current legal ID>"}` plus newline and flushes stdout. Debugging belongs on stderr. Memory persists within one game. Observations contain `player`, `state_version`, `visible_state`, `remembered_state`, and `legal_actions`. Each action has a command with kind/source/target/parameter. Never invent an action ID. Runtime has no filesystem, networking, environment variables, or access to authoring tools.

Core play is Bardur 1v1 with fog, on 14x14 drylands/lakes/continents. Always derive coordinates from `visible_state.board_size`; separate transfer diagnostics use sizes 11 and 18 and pangea/archipelago/highlands. Transfer is reported separately and never changes the core score. The core reference panel is fixed before authoring. No reference implementation or weights enter the coder container.

Default v6 runtime: exactly 200 full rounds, 40000 shared game actions, 4 trillion fuel per submitted bot/game, 256 MiB Wasm memory, 2 seconds per decision and a one-time 5-second startup allowance. A full round means both seats have ended their turn; rounds are internally numbered from zero. The referee stops before turn 200 starts. Feedback and final evaluation both have a uniform 1800-second infrastructure guard. An explicit immutable evaluation receipt may set a diagnostic horizon of 100/150/200/300 rounds or another disclosed diagnostic safety allowance.

Win=1, benchmark round/action-cap draw=0.5, loss=0. Guest faults (including fuel, invalid output, decision timeout) forfeit. Referee/infrastructure timeouts stay unresolved, with score bounds and no complete point estimate. Draws are benchmark outcomes, not claims that unlimited Domination would draw. No contestant's own value head or game score adjudicates capped games.

All models/stages share hidden layouts and both seats. Five builds and at least ten maps/family are required for a non-pilot run. Peer comparisons use at least five maps/family, matched replicate indices, and remain secondary. Pilot runs are explicitly preliminary. Sources and plans are immutable; every played game is replay-audited. Replay proves engine consistency, not equivalence to the commercial game.

Independent-track source is also sampled at 25%, 50%, and 75% of the active coding budget for descriptive budget curves. No sampled result is returned during authoring. Staged tracks use their stage snapshots at the same fractions. These are sampled current files, not best-of selection. A malformed sampled file scores compilation losses.
