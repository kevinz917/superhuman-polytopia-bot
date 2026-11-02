# Polytopia Skirmish

Build the strongest Bardur Polytopia 1v1 Rust bot you can within **40 minutes of active authoring time**. This is one continuous session. You decide when to read, edit, compile, run practice matches, inspect traces, and revise. There are no scheduled feedback rounds or conversation resets.

Read `/reference/PROTOCOL.md`, `/reference/RULES.md` and `/reference/TOOLS.md`. Edit `/bot/src/main.rs`, using Rust std and serde_json. The original Cargo manifest is used by the independent verifier. Derive coordinates from `visible_state.board_size`.

Available commands:

```sh
skirmish maps
skirmish inspect --map drylands-0
skirmish compile
skirmish practice --map drylands-0 --seat 1
skirmish practice --map lakes-1 --seat 2
skirmish trace --path /bot/practice/GAME_ID/game.jsonl --offset 0 --limit 1
skirmish check
```

Practice automatically compiles your current source and plays a full match against the public `public-greedy-v1` baseline. Its result names the real trace path. You may inspect the public baseline implementation and write your own tests. Practice results are development feedback, not final grading. Use your time to improve strategy; do not stop after merely making the starter compile.

API latency/retries, reasoning, editing, compilation, practice play, replay checking, and trace reading all consume your authoring budget. Environment setup, explicitly logged CPU scheduler waits, and the final hidden evaluation are excluded. Each model response allows 32,768 output tokens including reasoning. There is no fixed number of tool calls or practice matches. Long commands are bounded by the remaining budget.

Your bot receives JSON observations on stdin and must output exactly `{"action_id":"CURRENT_LEGAL_ID"}` followed by a newline and flush. It retains memory during each game. Runtime: 4 trillion fuel/game, 256 MiB Wasm memory, 2 seconds/decision plus a one-time 5-second startup allowance. Core games use exactly 200 complete rounds and 40,000 shared actions. Guest faults forfeit; round/action caps are draws. Infrastructure failures are not model losses.

The final score is 100 times mean match points: win 1, draw 0.5, loss 0. Your final bot plays a fixed six-opponent panel including hero-1v1, equally weighted across drylands/lakes/continents and both seats, on held-out maps. Standard evaluation has ten maps/family (360 matches); an explicitly labeled integration demo may use one map/family (36 matches). Champion results are also reported separately. Hidden evaluation never returns feedback during authoring.

When finished, call bash with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Otherwise the controller captures your current source at cutoff. Broken or missing current source scores compilation losses; there is no rollback, best-of selection, or post-deadline repair. Only the root-owned cutoff snapshot is transferred to a fresh verifier and independently rebuilt.
