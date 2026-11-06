# Corrected performance review (2026-10-30)

The two tested LLMs were `deepseek/deepseek-v4-flash-0731` and
`openai/gpt-5.6-terra` through OpenRouter. Each had one five-minute offline Rust
bot-writing run, with practice and trace feedback. The reference, hero-1v1, is a
previously trained 1.97M-parameter PPO policy using greedy inference.

The original tournament's 2-billion-fuel allowance produced five DeepSeek
forfeits. To remove that confound, the same frozen Wasm binaries were evaluated
on the same four maps from both seats with 100 billion fuel per game. The model
APIs were not called again and neither generated bot was revised. Engine/rule
hashes and frozen binary hashes were checked. This is a controlled reevaluation,
not a new independent sample of model-written bots.

[Corrected machine-readable report](results-review-2026-10-30.json):

| Rank | Entrant | Wins | Losses | Unresolved | Elo, hero=0 |
|---|---|---:|---:|---:|---:|
| 1 | hero-1v1 | 16 | 0 | 0 | 0 |
| 2 | DeepSeek V4 Flash 0731 | 5 | 10 | 1 | −551 |
| 3 | GPT-5.6 Terra | 2 | 13 | 1 | −688 |

Both LLM-written bots lost 0–8 to hero. DeepSeek won their head-to-head **5–2**;
one game reached the engine's round limit (reported turn 101) and remains
unresolved. All seven resolved head-to-head games ended by elimination; there
were no runtime forfeits in this reevaluation. Elo is regularized and relative
to this three-bot roster. Four maps and one authoring run per model do not
establish a general model ranking.

## What the models built

Both wrote stateless, single-position greedy heuristics over legal actions.
Neither implemented multi-step tree search or trained a neural network. The
following is based on hash-verified source inspection and action summaries
computed on Modal, not solely on the models' descriptions of their code.

**DeepSeek: mobile expansion and Rider production.** Its scorer prioritizes
captures, favors recruiting Riders/Archers and maintaining an army, rewards
moves into villages and near unexplored tiles, and moves toward a visible enemy
capital or the opposite corner from its own capital. An analytic combat estimate
scores damage, retaliation, kill value and survival. It chooses Workshop,
Resources, Population growth and Giant rewards when those pairs are available.

Across its 16 corrected games it made 381 unit-purchase/upgrade actions: **289
Riders, 91 Warriors, one Scoutship**, plus 90 capture actions. It researched
Archery in every game but bought no Archers: a clear mismatch between the tech
priority and actual recruitment. It also made 405 structure-destruction actions,
so its positive scores for miscellaneous legal actions permit substantial
unproductive churn. These are weaknesses despite the better head-to-head result.

**GPT-5.6 Terra: Defender-heavy fixed priorities.** Its cascade resolves rewards,
captures and promotions, attacks a low-health target, harvests, builds selected
economic structures, researches a fixed tech list, recruits, then moves. It
prioritizes Organization/Strategy and Forestry/Mathematics; its recruitment list
ranks Defenders above Riders and Warriors. Movement scores proximity to known
villages/ruins and unexplored tiles. Its attack score does not estimate damage
or retaliation.

Across its 16 corrected games it made **227 Defender and 12 Warrior purchases**,
37 capture actions, and built **104 Lumber Huts and 15 Sawmills**. This was much
more defensive and less expansion-oriented than DeepSeek. Raw action counts
are descriptive and can be dominated by a long game; they are not normalized
measures of strategic efficiency.

The five-minute authoring costs reported by the API were $0.0566 for DeepSeek
and $0.8896 for GPT, excluding Modal compute. GPT used more source edits (six vs
three) and practice games (16 vs ten), but its frozen bot was weaker in this
particular matchup.

## Pressure results

The pressure runs were short plumbing checks, not a replicated effect study.
The early 90-second pair yielded one unchanged starter and one failed final
edit. In the updated 150-second/two-edit pair, both neutral and evaluative runs
ended without a compiled current source. Their failed submissions are retained;
there is no valid paired strength comparison or supported pressure-effect claim.
The larger randomized study remains unrun.

All full game logs and source stay in Modal staging, on `polytopia-skirmish`
under `results-review-20261030/`; the review driver is retained there as well.
