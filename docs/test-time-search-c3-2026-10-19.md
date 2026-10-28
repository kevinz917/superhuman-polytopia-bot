# Test-time search on c3 u4854 (2026-10-19)

Does inference-time turn search (docs/test-time-search-2026-10-19.md) still make the newest RL bot,
`c3-maps-ctl` update 4854 ("c3", docs/map-variants-2026-10-19.md), meaningfully stronger, and does the
leak-free version keep the gain?

## Answer

- **Yes, on the map mix c3 was trained for.** c3 with leak-free robust search (K=16 candidate turns,
  64 averaged opponent replies, determinized fog) beats raw c3 **114–49–37, +117 Elo [+87, +149]**
  over 100 fresh `proposed-v1` mix maps × both seats. Plain leak-free d=1 already gets +104 [+69, +141].
- **On classic maps the leak-free gain is smaller**: +53 [+17, +89] with robust search, and plain leak-free
  d=1 (+26 [−9, +61]) is not distinguishable from zero. Against raw u3142 robust leak-free adds +76 [+14, +141]
  on top of c3's 73% raw score.
- **Hidden information is most of the classic-map gain.** Leaky search is the upper bound: +108 (d=1) and
  +200 (robust) vs raw c3 on classic maps. Removing the leak costs −81 [−132, −34] for d=1 and −148 [−197, −99]
  for robust. On the map mix it costs less (−58 [−101, −16] for d=1).
- **Robust beats plain d=1 by +13 to +29 Elo leak-free in every block, never significantly**, at about 2.5×
  the compute (≈10.5 s vs ≈4 s per searched turn on one Modal core, ≈3–4 s vs ≈1.5 s on an M-series core).
  Leaky, robust adds +93 [+51, +138]: averaging replies mostly helps when the replies are real.
- Recommendation: for a fair (human-facing) stronger c3, use `determinize=True`. Plain d=1 is the better
  value on mixed maps; robust only if ~10 s/turn is acceptable. Speeding up the searcher (batching all
  candidates × replies into one vectorized call) is the obvious next step before serving it.

## Setup

- Candidate: c3 = `/retained/checkpoints/distributed-final-v1/c3-maps-ctl-update-4854/final.npz`, playing every
  turn through `search.TurnSearcher`. Opponent: raw greedy policy (c3 or u3142), never searching.
- Arms: 100 development maps × 2 seats = 200 games, frozen engine, 100-turn cap. Seeds: c3 mirror classic
  364000–364099, vs u3142 classic 365000–365099, c3 mirror on the `proposed-v1` mix 366000–366099
  (map type per seed from `maps.stream.MapSource`; 14–30 games per type).
- Search settings (all arms): K=16, d=1 (play the opponent's reply, score with the critic at the next own
  turn), T=1 candidates, T=1 rollout replies. Robust = `replies=64` with sequential halving. Leak-free =
  `determinize=True` (belief.py; re-plans after reveals, ≈1.9–2.2 plans per turn).
- Raw c3 vs raw c3 scores exactly 0.5 on every map (same deterministic policy in both seats), so mirror gains
  are measured against 0.5; the u3142 block is paired against a raw-c3 arm on the same maps. CIs are a
  paired-map bootstrap (4000 resamples).
- Parity check: with K=1 the search arena reproduces `cpu_arena.match` outcomes exactly on classic and on
  mix maps (`runs/test-time-search-c3/smoke.json`); `tests/test_search.py`, `test_belief.py`, `test_maps.py`
  pass on Modal (12 passed).

## Results

### c3 + search vs raw c3, `proposed-v1` map mix

| arm | W–L–trunc | score [95% CI] | Elo vs raw c3 [95% CI] | s / searched turn | $ |
|---|---|---|---|---:|---:|
| leaky d=1 | 126–39–35 | 0.718 [0.68, 0.76] | +162 [+127, +200] | 1.70 | 1.11 |
| leak-free d=1 | 110–52–38 | 0.645 [0.59, 0.69] | **+104 [+69, +141]** | 3.72 | 2.07 |
| **leak-free robust** | 114–49–37 | 0.662 [0.62, 0.70] | **+117 [+87, +149]** | 10.77 | 5.81 |

Leak-free robust minus leak-free d=1: +13 [−26, +54]. Score by map type, leak-free robust (n games):
lakes 0.75 (14), waterworld 0.75 (16), continents 0.71 (26), archipelago 0.70 (22), classic 0.68 (14),
ridge 0.68 (30), pangea 0.66 (16), jungle 0.65 (24), highlands 0.58 (18), drylands 0.47 (20). Per-type
samples are small (±0.2); drylands is the only type below 0.5 for both leak-free arms.

### c3 + search vs raw c3, classic maps

| arm | W–L–trunc | score [95% CI] | Elo vs raw c3 [95% CI] | s / searched turn | $ |
|---|---|---|---|---:|---:|
| leaky d=1 | 116–56–28 | 0.650 [0.60, 0.70] | +108 [+72, +145] | 2.05 | 1.36 |
| leak-free d=1 | 95–80–25 | 0.537 [0.49, 0.59] | +26 [−9, +61] | 4.38 | 2.42 |
| leaky robust | 139–35–26 | 0.760 [0.72, 0.80] | +200 [+162, +244] | 4.43 | 2.08 |
| **leak-free robust** | 100–70–30 | 0.575 [0.52, 0.62] | **+53 [+17, +89]** | 10.47 | 4.38 |

### c3 + search vs raw u3142, classic maps

| arm | W–L–trunc | score [95% CI] | Elo vs raw c3 [95% CI] | s / searched turn | $ |
|---|---|---|---|---:|---:|
| raw c3 (no search) | 132–41–27 | 0.728 [0.67, 0.78] | — | 0.10 | 0.79 |
| leaky d=1 | 162–19–19 | 0.858 [0.82, 0.90] | +141 [+85, +204] | 1.73 | 1.29 |
| leak-free d=1 | 140–29–31 | 0.777 [0.73, 0.82] | +47 [−10, +106] | 5.28 | 3.39 |
| **leak-free robust** | 148–26–26 | 0.805 [0.76, 0.85] | **+76 [+14, +141]** | 10.77 | 4.56 |

### Compared with u3142 (earlier study, classic maps, vs raw self)

| | u3142 | c3 |
|---|---|---|
| leaky d=1 (T=1 replies) | +125 [+87, +168] | +108 [+72, +145] |
| leak-free d=1 | +56 [+21, +92] | +26 [−9, +61] |
| leak-free robust | not run | +53 [+17, +89] |

The stronger policy leaves about as much for leaky search but less for leak-free plain d=1 on classic maps.

## Caveats

- The determinized fog is filled from the classic generator (`mapgen.layout`), also on variant maps. That is
  still leak-free but a poor prior on water maps, so the mix-map leak-free numbers are, if anything, low.
- Opponents never search, and every game uses the frozen engine and 14×14 boards.
- Robust leak-free games average ≈10 min of one Modal core each; one 100-turn mix game took ≈65 min of search
  time, which held up its whole chunk.

## Reproduce

```bash
SEARCH_LEDGER=runs/test-time-search-c3/budget.json modal run --detach --env staging experiments/modal_test_time_search.py::sweep --tag c3-v2 --spec-json runs/test-time-search-c3/spec-c3.json --per-container 64
modal run --env staging experiments/modal_test_time_search.py::fetch --tag c3-v2 --out-dir runs/test-time-search-c3
.venv/bin/python scripts/analyze_c3_search.py runs/test-time-search-c3/c3-v2
```

Per-game rows (with critic traces and map types) are in `runs/test-time-search-c3/c3-v2/` and on the
`polytopia-structured-rl` volume under `/results/test-time-search/c3-v2/`.

## Cost

| item | $ |
|---|---:|
| c3-v2 sweep (11 arms, 44 × 32-core containers) | 29.25 |
| c3-v1 sweep, failed (workers crashed on an unresolved `c3` path; 44 containers idle ~25 min) | ≈ 35 |
| smoke, diagnostic, Modal tests | ≈ 0.5 |
| **total** | **≈ 65** |

The c3-v1 failure came from uncommitted harness edits being reset in the worktree before launch; the arena now
fails a chunk when a worker's setup raises (instead of hanging on the empty core queue) and `sweep` rejects
unknown checkpoint names.
