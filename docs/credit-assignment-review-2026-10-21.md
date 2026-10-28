# Credit assignment in long-horizon game agents, read against our 14×14 / 18×18 runs (2026-10-21)

Question: how did AlphaStar, OpenAI Five, AlphaZero/KataGo and DeepNash handle credit assignment over long games,
and how do their decision counts compare with ours? Companion to alphastar-review-2026-10-21.md
and the 18×18 diagnosis in [rl-18x18-2026-10-19.md](rl-18x18-2026-10-19.md).

## Decisions per game and what each system did about the horizon

| System | Decisions per agent per game | Reward | Discount / return | Variance reduction that made it work |
|---|---|---|---|---|
| AlphaZero (Go / chess) | Go ≈100–150 moves per player, chess ≈40 | terminal ±1 | undiscounted Monte Carlo outcome z | 800-sim MCTS targets; KataGo adds ownership + score auxiliary heads ("predicting subcomponents of the target") |
| DeepNash (Stratego) | hundreds, draw at 2,000 total moves or 200 without an attack | terminal win/loss | full-episode returns "without any bootstrapping", V-trace ρ̄ = c̄ = 1 | R-NaD regularisation toward a reference policy; 10^9+ games |
| AlphaStar (SC2) | "thousands of actions" over ≈10 min, ≤22 actions per 5 s (≤ ~2,600) | terminal −1/0/+1 | **undiscounted** ("to accurately reflect the true goal of winning"); TD(λ) for the value, V-trace + UPGO for the policy | opponent observations in the critic (22% → 82%), KL to the supervised policy, z pseudo-rewards with their own value heads, 50k agent steps/s |
| OpenAI Five (Dota 2) | ≈20,000 steps (every 4th frame, ≈45 min) | **dense shaped** (Table 5: win 5, towers 2.25–4.5, barracks 6, hero death −1, gold 0.006 …), zero-sum symmetrised, non-win terms decayed 0.6^(T/10 min) | GAE λ 0.95; horizon H = T/(1−γ) annealed 60 s → 840 s (γ ≈ 0.9978 → 0.9998) i.e. ≤ ⅓ of a game | shaping "to simplify the credit assignment problem"; team spirit 0.3 → 1.0. Sparse-reward ablation (γ = 0.99996, 1 h horizon) still learned, "with a large penalty to sample efficiency" |
| Honor of Kings (JueWu) | ≈150 macro decisions (1v1 table), 9,000 frames | shaped | γ 0.997 (1v1, "half-life ≈ 46 s") / 0.998 (full game), dual-clip PPO | shaping + scale |
| **Ours, 14×14** | ≈785 decisions per seat, 100 rounds | terminal ±1 | γ 0.995 per decision (horizon 200 decisions ≈ ¼ game) | omniscient critic, KL anchor |
| **Ours, 18×18** | ≈1,500 decisions per seat | terminal ±1 | γ 0.995 per decision (⅛ game) failed; γ/λ only at end-turn (0.97/0.92) + critic warm-up + anchor 0.05 passed with the omni critic | same |

## Reading

1. Our decision counts are ordinary: between AlphaStar and Stratego, an order of magnitude below Dota. Nobody with a
   sparse win reward discounts per decision at a horizon shorter than the game. AlphaZero, DeepNash and AlphaStar all use
   γ = 1 on the terminal outcome; the two systems that discount hard (Five, JueWu) only do so because they also pay a
   dense shaped reward every step, and Five's sparse-reward ablation needed a horizon longer than the game.
2. The 18×18 failure was therefore the discount, which the turn clock fixed; the plateau after that is the variance of
   an undiscounted outcome over 1,500 decisions with ~10^6–10^7 samples per run, which the others solved with (a) a
   critic that sees everything (AlphaStar, now ours), (b) search targets (AlphaZero), (c) auxiliary targets that
   localise the outcome (KataGo ownership/score heads), (d) priors that stop the policy drifting (KL to SL / R-NaD),
   and (e) 10^3–10^4× our sample count.
3. Levers in order of cost for 18×18 / 20×20 / multiplayer: γ = 1 with TD(λ) over turns (an arm never run, see the
   AlphaStar review table); KataGo-style auxiliary heads (final city ownership per tile, final score, units alive at
   end) on the existing critic; search-improved soft targets in the actors; dense shaping last, since our PBRS arm hurt
   and Five needed many symmetric, time-decayed terms to make it work.

## Sources

- Vinyals et al. 2019, AlphaStar (Nature 575) — Methods, "Reinforcement Learning" and "Value and policy updates".
- Berner et al. 2019, Dota 2 with Large Scale Deep RL, arXiv:1912.06680 — §3.2, Table 2, Appendix G Table 5, Figure 16.
- Perolat et al. 2022, DeepNash, arXiv:2206.15378 — Methods, Table 3.
- Silver et al. 2017, AlphaZero, arXiv:1712.01815; Wu 2019, KataGo, arXiv:1902.10565 (auxiliary targets ablation).
- Ye et al. 2020, JueWu, arXiv:1912.09729 and arXiv:2011.12692.
