# Plato T3 (update 96,070)

`plato.bin` is the policy of **T3**, the latest variant of Plato, trained by Jack Yuan. It is the file the website
serves as "Plato": the browser weight format, loaded by the engine in [`plato-t3/`](../../plato-t3).

- Architecture: `transformer-v3-p2-h4`, 6 transformer blocks × 128 channels, the same family as
  [`weights/plato`](../plato)
- Parameters: 3,243,386 (106 tensors, float32)
- Observation version: 6
- Ruleset: `polytopia-2.17.3`, two players, resignation disabled
- Training update: 96,070

## What it was trained on

From `plato.json` (the `trained` block):

- The bot always plays **Bardur**.
- Board sizes 11, 14, 16, 18 and 20.
- Map types: dryland, lakes, continents, archipelago, waterworld, pangea, from a port of the real game's map
  generator.
- Two players, conquest, 200-round cap.
- Opponent tribes: the 12 regular tribes (Ai-Mo, Bardur, Hoodrick, Imperius, Kickoo, Luxidoor, Oumaji, Quetzali,
  Vengir, Xin-xi, Yadakk, Zebasi) and the four special tribes (Aquarion, Elyrion, Polaris, Cymanti). The special
  tribes went through a work-in-progress engine path; their rules are not qualified against the real game.
- Not covered: 30×30 boards, perfection (score) mode, more than two players.

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `plato.bin` | 12,974,688 | `f1c61ed405decdf3473cbcb70aa166098f97863351179b46f1147cd90366f149` |
| `plato.json` | 4,493 | `b75c9bd58549ae33402014da4889b34cc5d9a1e832edd3fe8ebbd4db496ae143` |

`plato.bin` is byte-identical to the website's file. `plato.json` is the website's manifest with three edits for
release: the map-generator description and one caveat were reworded, and the trainer's own training-time evaluation
block was removed (those panels ran on the training simulator and are not the measurements reported in
[`docs/plato-t3.md`](../../docs/plato-t3.md)).

**The training checkpoint (`.npz`) is not included.** It is not available where this release was prepared. The
manifest's `source_sha256` (`9b31b8d7…`) is that checkpoint's hash and cannot be checked from the files here. What
`plato.bin` does hold is every policy tensor in float32, in checkpoint leaf order;
[`plato-t3/tools/bin_to_npz.py`](../../plato-t3/tools/bin_to_npz.py) writes them back out as `p0…p105` arrays. The
privileged critic and the optimizer state are not recoverable from it.

## Hidden information

T3 was trained with legal-move masks computed from the true game state, so units the mover could not see could
affect which moves were offered, and with two observation leaks that have since been fixed in the engine. The
website plays the bot from its public view only, with a referee that checks each proposed move against the real
state. Read the strength numbers with that in mind; details are in [`docs/plato-t3.md`](../../docs/plato-t3.md).
