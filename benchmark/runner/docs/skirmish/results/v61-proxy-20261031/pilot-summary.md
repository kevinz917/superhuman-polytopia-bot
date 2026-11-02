# Four-model CLI-proxy pilot

Local VM/Docker execution. One authored build/model/track, up to ten coding minutes. These are panel match points, not expert win rates. No defensible model ranking or confidence interval follows from this sample.

Original fixed feedback preserved; every frozen submission/stage scored under the corrected evaluation-only panel. No authoring repair or best-of selection.

| Model | Independent | Segmented, no feedback | Feedback |
|---|---:|---:|---:|
| openai/gpt-6-astra | 80.56 | 77.78 | 72.22 |
| anthropic/claude-opus-5.5 | 30.56 | 36.11 | 33.33 |
| openai/gpt-6.1-sol | 66.67 | 2.78 | 63.89 |
| anthropic/claude-sonnet-5.5 | 36.11 | 25.00 | 4.17 |

Strongest-reference outcomes across all final builds: {'loss': 72}. Panel points measure broader opposition; they are not hero win rates.

## Integrity

```json
{
  "core_scheduled": 1692,
  "diagnostic_scheduled": 1260,
  "actual_episodes": 2916,
  "full_games": 2880,
  "tactical_checks": 36,
  "virtual_compile_losses": 36,
  "replay_actions": 1958455,
  "unresolved": 0,
  "independent_rebuilds": 44
}
```

## Final-core cap draws

| Family | Cap draws | Games |
|---|---:|---:|
| drylands | 2 | 144 |
| lakes | 2 | 144 |
| continents | 7 | 144 |

## Horizon sensitivity

Separate diagnostic games against siege, naval-50 and hero, on reserved calibration maps. Results do not replace the fixed 200-round primary score.

### independent

- 100 rounds: astra6 61.11, sol61 38.89, sonnet55 16.67, opus55 11.11
- 150 rounds: astra6 61.11, sol61 38.89, sonnet55 16.67, opus55 11.11
- 200 rounds: astra6 61.11, sol61 38.89, sonnet55 16.67, opus55 11.11
- 300 rounds: astra6 61.11, sol61 38.89, sonnet55 16.67, opus55 11.11

### segmented-control

- 100 rounds: astra6 33.33, opus55 16.67, sonnet55 11.11, sol61 2.78
- 150 rounds: astra6 33.33, opus55 16.67, sonnet55 11.11, sol61 2.78
- 200 rounds: astra6 33.33, opus55 16.67, sonnet55 11.11, sol61 2.78
- 300 rounds: astra6 33.33, opus55 16.67, sonnet55 11.11, sol61 2.78

### feedback

- 100 rounds: sol61 50.00, astra6 36.11, opus55 16.67, sonnet55 8.33
- 150 rounds: sol61 50.00, astra6 33.33, opus55 16.67, sonnet55 8.33
- 200 rounds: sol61 50.00, astra6 33.33, opus55 16.67, sonnet55 8.33
- 300 rounds: sol61 50.00, astra6 33.33, opus55 16.67, sonnet55 8.33

## Transfer and tactical diagnostics

Transfer points against scripted siege across 11/14/18 boards and pangea/archipelago/highlands. Tactical probes test only three declared one-decision situations. These are separate from primary scores.

| Model / track | Transfer points /100 | Tactical passes |
|---|---:|---:|
| astra6 / feedback | 77.78 | 1/3 |
| astra6 / independent | 88.89 | 1/3 |
| astra6 / segmented-control | 66.67 | 1/3 |
| opus55 / feedback | 47.22 | 3/3 |
| opus55 / independent | 41.67 | 3/3 |
| opus55 / segmented-control | 38.89 | 3/3 |
| sol61 / feedback | 72.22 | 1/3 |
| sol61 / independent | 72.22 | 1/3 |
| sol61 / segmented-control | 0.00 | 0/3 |
| sonnet55 / feedback | 5.56 | 3/3 |
| sonnet55 / independent | 38.89 | 3/3 |
| sonnet55 / segmented-control | 19.44 | 3/3 |

Matched diagnostic outcome changes across horizons (same frozen bot, opponent, map and seat):

- 100_vs_300: 1/216 outcomes changed; cap draws 5 → 4.
- 150_vs_300: 0/216 outcomes changed; cap draws 4 → 4.
- 200_vs_300: 0/216 outcomes changed; cap draws 4 → 4.

## Reliability

- sol61/segmented-control: unchanged_starter; 0 runtime forfeits in 36 final-panel outcomes.

## Original authoring caveats

- Original protocol-check helper used legacy one-trillion fuel; actual feedback and final evaluation used four trillion. A scoped future-authoring fix does not change these frozen bots.
- Original feedback reports named host archive paths; public traces were available under /feedback/round-N/00.jsonl etc. Future reports explicitly name these sandbox paths.

Raw plans, all outcomes, source rebuilds, authoring origin and the complete report remain beside this summary.
