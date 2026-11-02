# Offline authoring tools

All models get these same tools and nine fixed public maps. Calls consume the coding
clock. There is no supplied opponent, checkpoint, match runner, win-rate evaluator,
or internet connection. Expert matches remain the three scheduled feedback batches
in feedback-3r; oneshot-20m receives no expert feedback during authoring.

```sh
skirmish --help
skirmish compile
skirmish maps
skirmish inspect --map lakes-0 > /tmp/map.json
skirmish example
skirmish check
```

`compile` builds your current source offline for wasm32-wasip1. The final referee
still independently rebuilds frozen main.rs using its original manifest and lockfile.

`maps` lists three maps each of drylands, lakes and continents. `inspect` gives the
full PUBLIC TEST layout and seat 1's initial redacted observation. Full layouts are
for these public tests only; actual match observations retain fog. Files are also
in `/reference/examples/<map-id>.json`. Public test seeds (410000..410022), expert
feedback seeds (500000..599999), and hidden final seeds are separate.

`example` prints `/reference/examples/worked-match.jsonl`: an annotated, static
three-action wire example. Seat 1 ends its turn, seat 2 ends, and seat 1 resigns.
It demonstrates player transitions, income and termination, not a competitive strategy.
It contains no opponent implementation and does not execute one.

`check` feeds recorded positions to YOUR compiled bot using the same persistent
WASI transport and limits as evaluation: 256 MiB, one trillion fuel per process,
two seconds per decision plus five seconds at startup. It checks JSON responses
and legal IDs. The default checks the two seat-1 positions of the worked example.
To check positions from a supplied feedback trace:

```sh
skirmish check --observations /feedback/round-0/00-drylands-seat1.jsonl --seat 1
```

All selected observations use one process, preserving bot memory. This is a
PROTOCOL TEST, not a new game: later positions follow the recorded actions even
if your new bot chooses differently. There is no counterfactual win rate.

To verify mechanics in the REAL Rust engine, provide every action explicitly:

```sh
skirmish inspect --map drylands-0 > /tmp/map.json
# Write a JSON array of current legal action-ID strings into /tmp/actions.json.
skirmish replay --map drylands-0 --actions /tmp/actions.json
```

An empty array returns the initial observation. Each invocation starts from the
selected public map, applies at most 32 explicitly supplied actions, and returns
the final actor observation and transition events. Both seats' actions must be
supplied by you. It never chooses a move for another player or runs an automated
match. Invalid action IDs fail visibly. This is a convenience limit for short
mechanics checks, not a security quota on self-written code: engine bindings are
available offline, and all self-written experiments remain inside the coding clock.

During actual games the submitted Wasm has only stdin/stdout. It cannot call
these authoring tools, inspect test layouts, access files, or make network requests.
