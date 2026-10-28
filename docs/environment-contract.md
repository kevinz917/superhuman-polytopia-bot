# Headless worker and trajectory contract

Status: GPU-first implementation. The device core in `src/polytopia_bot/gpu/` implements the frozen `bardur-1v1-v1` research environment; commercial-game equivalence is not certified. The CPU process protocol below is retained as an optional future policy/debug transport, not the simulator runtime architecture.

## Device boundary

`State` is a PyTree of fixed-shape arrays. `vmap(step)` consumes one action per game and `jit` compiles the transitions; `lax.scan` supports device-resident sequences. A batch's game states never pass through Polyfish during normal execution. The Rust executable under `tools/polyfish-oracle` is exclusively a validation tool.

The current action ABI is `(kind, source, target, parameter)` with pinned Polyfish type IDs. Status codes distinguish success, invalid action, unsupported rule/state, stale version and finished episode. Rejections do not mutate state. Reset supports procedural GPU generation and device gathers from an initial-state pool. The factored legal catalog, device decoder, native terminal rewards, and private semantic snapshot format are implemented. Snapshots preserve every state field and reject a different schema, rules profile or implementation digest. The full-game gate refers to the frozen research contract in `ruleset-v1.md`.

## Process boundary

One long-lived engine process manages multiple game handles. Start with a small process pool for fault isolation. JSON Lines on stdin/stdout is the reference transport; stdout contains protocol messages only and stderr contains diagnostics. Requests and responses carry request IDs. A worker serializes mutations for each game handle; independent games may run concurrently.

Operations:

| Operation | Inputs | Output |
| --- | --- | --- |
| `describe` | protocol version | engine commit, rules version, capabilities, unsupported settings |
| `reset` | game settings and seed | handle, observation, state version, player, legal catalog |
| `observe` | handle, player | that player's visible state and observation-derived memory |
| `step` | handle, expected state version, action ID | actor's post-action observation, next actor, terminal/truncation flags, rewards, new state version |
| `snapshot` | handle | opaque private snapshot with all RNG state |
| `restore` | private snapshot | new handle and matching deterministic state |
| `close` | handle | acknowledgment |

An action can leave the same player active. The rollout controller explicitly requests the next actor's observation, keeping each seat's history separate. Mandatory sub-decisions such as a city reward must have an explicit active decision owner.

Action IDs are response-local, mapped to semantic commands, and invalid after a state-version change. Reject stale or invalid commands without mutating the game. Never choose a fallback action silently. Stable semantic commands, not temporary IDs, are the replay format. The legal catalog includes end-turn when genuinely allowed and every other legal choice; do not cap it at an arbitrary fixed number.

The expected state version is a monotonic counter, not an omniscient state hash sent to the policy. Keep full-state hashes, seeds, private snapshots, hidden troop counts, and secret opponent fields exclusively in trusted logs. Even pseudorandom state hashes can become side channels or memorization keys.

## Observation boundary

The model sees only player-visible facts and explicitly marked remembered observations. Public rules can be included. Unknown cells must not carry terrain/resource values. Hidden opponent technologies, stars, units, and unexplored tile ownership must not leak through entity lists, ordering, IDs, errors, or suggested actions. Only include enemy aggregate statistics if the pinned reference game exposes them to that player.

An unseen cell can still be a legal exploration destination; enumerate only the information the real player may know about the attempted command. Do not infer impossible travel or safe combat from omniscient terrain. If hidden information legitimately affects public legality, match the reference game's observable behavior precisely.

## SFT-0 memory lifecycle

The SFT checkpoint plan specifies fresh model context per action plus compact runner-maintained factual memory. This is a required extension to the current tile-memory schema. Maintain one deterministic history per episode/player; begin with five recent own actions and up to 32 historical public records, freezing measured limits before prompt optimization. Mark historical facts with observation timestamps/revisions and uncertainty. Current public observations supersede stale facts; unseen state changes cannot update memory. Repeated reads of the same observation must be idempotent.

Reset clears both seats' memory. Episode snapshot/restore must restore the associated public histories as well as engine state. The teacher, base model and SFT model receive the same information contract. Save the exact memory at each decision for reconstruction without future information. Bounded repair exchanges exist only within the current decision; full-game dialogue and model-written strategic plans are excluded from SFT-0.

## Saved records

Each decision stores episode ID, seat, decision index, observation schema version, exact rendered prompt and token IDs, action catalog and order, semantic action, generated response and token IDs, sampled token log probabilities, behavior-policy adapter hash, opponent hash, sampling settings, response loss mask, rewards, and terminal versus truncation status. An optional value estimate is versioned with the critic checkpoint.

Snapshots and omniscient diagnostics live in a separate private record, never mixed into model datasets. Source traces retain rejected outputs. Finished trajectories can be compressed; expensive full snapshots are needed at episode starts and selected debug checkpoints rather than every action.

SFT uses exact observation-to-command examples. RL uses complete episode linkage even when decisions are processed in separate model contexts. Adapter sync occurs at rollout-batch boundaries in the first implementation. Opponent data is not an on-policy learner sample unless it was actually sampled from the same behavior policy under the same configuration.

## Required validation

- Exact reset/restore/replay determinism, including all stochastic events.
- Same observable history plus different hidden world states produces identical policy-visible serialization unless the real rules reveal a difference. Recompute public legality appropriately when constructing such pairs.
- Complete legal-action enumeration and agreement with execution; action IDs cannot execute against a newer state version.
- Correct income/capture/turn/terminal semantics and distinct truncation handling.
- Per-player memory stays isolated when the same LM serves both seats.
- Recorded sampling log probabilities match recomputation within numerical tolerance. Test unchanged adapters before any RL update; include tokenizer, EOS, temperature, and masking behavior.
- Actual generated token IDs, rather than retokenized or normalized responses, are used for PPO probability ratios.
- Protocol smoke tests run in CI without a GPU. Authoritative replay fixtures gate engine changes; a small GPU integration run gates changes to weight synchronization or probability accounting.
