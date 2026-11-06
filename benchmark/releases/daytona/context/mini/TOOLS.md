# Authoring tools

Use `skirmish --help`. `maps`, `inspect`, `replay`, `compile`, `check` and `example` retain the real-engine SDK behavior.

`skirmish practice --map drylands-0 --seat 1` compiles current source and plays public-greedy-v1. Both seats and all nine named public maps are supported. The returned report names `/bot/practice/<id>/game.jsonl`. `skirmish trace --path PATH --offset 0 --limit 1` prints your recorded observation/action entries. You may also use Python to summarize those files. Compilation, practice play, replay checks and inspection consume coding time. Practice uses 200 rounds, 40,000 actions and four trillion fuel, with the same Wasm limits as final evaluation. No hidden-map/champion query exists.
