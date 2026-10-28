# Champion checkpoint

`final.npz` holds the policy of `scratch-cont-2`, the strongest bot with turn search: about 20 hours of self-play PPO from random
weights (three runs, 7.7 billion samples). It also holds a fog-of-war value head, fitted afterwards with the policy
frozen, which turn search uses. `final.json` records the training config, the parameter count (519,267), the
checkpoint's SHA-256 and the simulator digest it was trained against. Loaders refuse the file if either does not match.

- Trained under: `bardur-1v1-v2`, 14×14 (the retired research rules). It loads through `ruleset.LEGACY_DIGESTS` and
  plays under the game's rules, `polytopia-2.17.3`.
- Architecture: `normalized-v2`, 2 residual blocks × 64 channels
- SHA-256: `90618f5a24d44f76987eb2c002858a9e8c34e3f2bf1ba78e9a19de354ca4fc2c`

The training history is in [docs/scratch-check-2026-10-22.md](../../docs/scratch-check-2026-10-22.md),
[docs/scratch-continuations-2026-10-23.md](../../docs/scratch-continuations-2026-10-23.md) and
[docs/scratch-search-2026-10-23.md](../../docs/scratch-search-2026-10-23.md).
