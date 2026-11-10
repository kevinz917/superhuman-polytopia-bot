# Claude Fable 5.2: Polytopia Bench follow-up

**Restart disclosure.** Two original sessions were stopped during unbounded filesystem searches and replaced. The interrupted sessions are archived and excluded. This differs from the original uninterrupted five-attempt procedure. See [restart record](restart-record.json) and `interrupted/` for the original conversations and cutoff sources. These interruptions are not reported as model losses or as spontaneous harness failures.

Five independent local CLI-proxy attempts: **54.33 ± 11.96 panel points /100** (mean ± sample SD).
Champion results: 4 wins, 0 draws, 296 losses across 300 games.

Each completed attempt received one continuous 40-minute authoring allowance, followed by the original six-opponent, three-map-family, ten-layout, both-seat schedule (360 games per bot). All 1,800 outcomes and replay audits from these five completed attempts are retained. No post-cutoff source repair.

Follow-up on the original fixed panel. Maps and references were published after the original campaign; they remain unavailable inside this offline authoring sandbox.

The immutable original coder/toolchain image and all game/reference hashes match the October 31 campaign. Independent final games now run in four spawned processes per verifier, with up to three local verifiers. No OpenRouter, Daytona, or paid API fallback was used; subscription dollar cost and quota percentage are unavailable.

| Submission | Panel points | Champion W–D–L | Active minutes | Practice games | Submission quality |
|---|---:|---:|---:|---:|---|
| [task__cPjVoRk](sources/task__cPjVoRk.rs) | 56.67 | 1–0–59 | 28.79 | 82 | custom_submission |
| [task__fyG35eh](sources/task__fyG35eh.rs) | 67.36 | 1–0–59 | 26.86 | 75 | custom_submission |
| [task__x3guP3E](sources/task__x3guP3E.rs) | 61.11 | 1–0–59 | 27.93 | 84 | custom_submission |
| [task__Q4zSJ9K](sources/task__Q4zSJ9K.rs) | 50.56 | 1–0–59 | 31.36 | 143 | custom_submission |
| [task__zSeiYjB](sources/task__zSeiYjB.rs) | 35.97 | 0–0–60 | 35.65 | 17 | custom_submission |

Scores, hashes and provenance: [results.json](results.json). Article-ready aggregates: [article-model.json](article-model.json).

`runs/` contains each exact authoring conversation, compact final replay, and practice archive. Completed practice observations can be reconstructed from the private action log and report using the pinned engine. `harness-source.tar.gz` contains the exact follow-up harness sources; `build-provenance.json` pins their hashes. The original benchmark archive is unchanged.

Report generator: [exporter-source.py](exporter-source.py), originally run as `scripts/export_skirmish_extension.py` from the benchmark worktree with its retained native Harbor artifacts.
