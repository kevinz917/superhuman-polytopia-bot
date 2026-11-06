# Polytopia blog post — editable text

I trained a 520k parameter bot network to play the popular mobile game Poytopia. The bot was trained for two modes, a 1v1 14x14 map size mode, and a 4-player 18x18 map mode.

The replay here is a pretty interesting complete game between the champion network (orange) and an earlier checkpoint (blue). Blue hides behind a catapult wall, and for 30 turns the champion rates its own position as losing, but then uses Cloaks to track the enemy. Then its economy pulls ahead and used cloaks to track the wall

> **[Figure 1 — replay viewer, wide]**
> Caption: Champion + search (orange) vs an earlier self-play checkpoint (blue), archipelago map. Press play, drag the slider, or jump to a moment. The dashed line is orange's own win estimate.

A few goals on why I wanted to do this project:

- Learn reinforcement learning by doing it - which RL tricks matter in practice, and gain intuition to guess why, as well as how to build the stack end to end, from data prepration, building the environment, managing a compute budget, and knowing when to stop.
- I wanted to see how far frontier models can go at vibe-researching a relatively trivial RL task. There are tons of literature of game-playing RL out there, so Fable should be pretty good at this.
- Finish an end-to-end bot in a span of ~1 week on a fun game that I understand well.

A few highlights of this project

- A curriculum and a scripted teacher got the first policy off the ground, but pure self-play from random weights overtook that within a few GPU hours.
- Letting the critic see the full, unfogged state was the biggest single win, worth +125 Elo after one hour. Thought this was super cool.
- Discounting once per turn lines credit assignment up with game time.
- A one-turn lookahead over sampled hidden states adds about +100 Elo; searching deeper or wider adds nothing measurable once the critic is strong due to confusion and unpredictability of future state.
- Five rounds of shifting work between CPU and GPU took training from 488 to 208k samples per second. This was where the main optimizations happened.
- The whole project cost about $7,000 of total cloud compute (a lot wasted).

Where it ended up: a residual CNN with two 64-channel blocks and factored, masked action heads, trained by PPO against a pool of its own snapshots. A classic critic with the whole board, rewards discounted per turn. Games are simulated on fleets of CPU actors running a Rust rules engine 1:1 verified with official game rules as of October 2026, policy inference is batched on small GPUs, and one H100 trains. When it plays, a one-turn search over 16 sampled candidate turns chooses the move. Here's the rough sketch of the training loop.

> **[Figure 2 — recipe diagram]**
> Caption: The training and play loop. Boxes link to the sections that explain them.
>
> Box text:
>
> - **CPU actor fleet** (→ #balance): Rust rules engine, bit-exact / Hundreds of games per container / Policy forward on a small GPU
> - **GPU learner** (→ #critic): PPO, pool of past snapshots / Omniscient critic (full board) / Per-turn discounting
> - **Self-play from random weights** (→ #scratch): No teacher, no human data / Mixed map types every game / Win or lose: +1 / −1
> - **Play time** (→ #search): Policy plays under fog / One-turn search, 16 candidates / Hidden state sampled, never peeked
> - Arrow labels: games, weights, checkpoint

## Timeline {#timeline}

1. `Environment`**.** An open-source Polytopia engine, frozen to one tribe (Bardur) on a 14×14 board, rewritten in JAX so simulation and training could share one GPU. The policy got factored, masked action heads from the start. I initially set off to keep the mall small and fix 1 tribe to get something going without overcomplicating the game.

   _488 samples/s on one L40S_

2. **A first learning signal.** Pure PPO from random weights learned only to stop resigning. A capture curriculum, imitation of a scripted teacher and a small league produced league96. However, this strategy produced almost all warriors, so clearly suboptimal and just brute-forcing.

   _league96: 64–0 vs the teacher, 99% Warriors_

3. **Scaling self-play.** Simulation moved to fleets of CPU actors feeding one GPU learner due to the branching logic of the game being unsuited to GPUs. Ten map types replaced the single classic map to increase data diversity without being under-trained.

   _u3142: 117–10–1 vs league96 · c3: the reference bot (Elo 0)_

4. **Search at play time.** A one-turn lookahead over sampled candidate turns, then a leak-free version that samples the hidden board instead of peeking. This test-time bootsted the ELO significantly. A trick we learn from the AlphaZero family / Go bots.

   _+104 Elo on c3_

5. **One-hour ablations.** Five arms from c3, one change each. Only the omniscient critic beat the parent. A stalled 18×18 run exposed the per-decision discount.

   _omniscient critic: +125 in one hour_

6. **The omniscient critic's long run.** Six hours, two seeds, tracking within a few points of each other.

   _c5-omni: 77% vs c3, +255_

7. **Throughput.** Lean actors, a bit-exact Rust engine, GPU inference for actors and a faster batch assembler. This should've been done much earlier to save time and compute (thousands of $ could've been saved).

   _23k → 208k samples/s_

8. **Starting over from random weights.** Same recipe, no curriculum, no teacher. It passed the whole inherited lineage within hours.

   _champion: +442, 92% vs c3_

9. **The real game.** Decompiled the Steam build, replicated its AI, and aligned the engine to its rules.

   _48–0 vs the real Crazy AI_

10. **Bigger nets and four players.** A 4×128 network trained from scratch on the real rules, plus an 18×18 four-player free-for-all to cap off this fun experiment.

    _hero: 68% vs the champion_

## Measuring progress with Elo {#results-guide}

To measure the relative strength of the bot against its past self, I had bots play against each other in tournament style every checkpoint and use ELO to quantify its strength. We pair up bots to play and use Bradley-Terry ratings to score the result, with the initial model pinned at 0 ELO. This gave me a pretty clear view of how did different arms of parallel experiments perform relative to each other, given same amount of compute.

Relative ratings suit the way the research ran. Every experiment asked whether one change helped, and a match between the arm and its control at equal compute answers that directly. Ratings also compare bots that never met, through shared references, and they put every idea on one axis (Figure 7).

Their limits, each of which bit me at some point:

- **Elo measures beating relatives.** A self-play lineage can get better at beating its own ancestors without getting better at the game. The real game's AI was the outside anchor.
- **Chains drift.** Errors add up along a chain of matches, so a +500 at the far end of a long chain carries wide error bars. I re-anchored with round robins of five to seven reference bots.
- **The map sample matters.** The same two bots read +125 on one set of 100 maps and about +30 on another. I fixed the evaluation maps per campaign and never reused the maps I selected on.

Every idea got an arm with the change and a control arm, both started from one parent and given equal compute. Each then played a tournament against a fixed reference bot on unseen maps, with both sides choosing moves greedily.

- **Maps.** Training and evaluation use a mix of ten map types: 15% classic, the rest split among drylands, lakes, pangea, continents, archipelago, waterworld, highlands, ridge and jungle. Water maps make the bot build ports and fight at sea, and they make games longer.
- **Seats.** Every map is played twice with seats swapped. The map and turn order decide a large share of games between near-equal bots, and pairing cancels that out.
- **Metric.** Win rate over all games, plus win rate over decisive games. A 14×14 game is capped at 100 rounds, and mirror matches between close bots hit the cap up to a quarter of the time. For Elo, +100 is about a 64% win rate.
- **Size.** Usually 256–512 games per comparison, with the pass bar written down before the run (for example: at least 58% of all games, with the 95% interval above 50%).

Elo across the whole project is chained match to match, so trust comparisons between neighbors more than long chains.[^2]

## A bootstrapped start {#bootstrap}

The hardest part of Polytopia for RL is the first win. A game is about 800 decisions per player, the reward is a single +1 or −1 at the end, and random play almost never finishes a game: eight random games all ran to the round limit. With no decisive games, PPO has nothing to learn from.

So the first policy got a curriculum. Lessons started from positions where a single action captures a city, then two (move, then capture), then three (attack, move, capture). Each lesson shortens the horizon enough that the reward shows up. After the curriculum came imitation of a scripted teacher and a small league of rivals. The result, _league96_, beat the teacher 64–0 on untouched maps.

It also recruited Warriors 99% of the time. The teacher scored a recruit as `20·demand + 20·threat − 2·cost`, and cost was the only unit-specific term, so it always picked the cheapest unit. PPO inherited the habit and never left it behind even in games where catapults were available.

Scaling PPO across a CPU actor fleet (see [Balancing CPU and GPU](#balance)) produced _u3142_, which beat league96 117–10–1. A further four hours of self-play on the map mix gave _c3_, the reference for the rest of this post. It beat u3142 in 70% of games on the mix and won on every one of the ten map types.

> **[Aside] Dropped: scoring the turn cap as a draw**
>
> Long water-map games often hit the 100-round cap. One arm scored those games as draws. The other let the critic bootstrap from its own estimate at the cap. The draw arm trailed at both checkpoints (71% vs 85% of decisive games) and was stopped.
>
> A draw label says nothing about who was ahead, so a large share of the hardest games contributed no signal. Bootstrapping keeps a gradient in those games: the side that built the stronger position still gets credit for it.

## Omniscient critic {#critic}

In Polytopia, the policy plays under fog of war, so its value head has to guess what it can't see: where enemy units are, how many stars the opponent has, which techs they own.

To fix this, we implement an asymmetric actor-critic. The value function used for training receives the full game state from both players' perspectives, effectively cheating while training. The policy stays fogged, and since the critic is used only in training, the deployed bot sees nothing it shouldn't. This was taken directly from AlphaStar as one of the best RL tricks.

I test ran it in a one-hour ablation with four other arms, each a single change from c3's recipe:

| Arm (1 h from c3)             | W–L–cap vs c3 | Win rate | Of decisive |
| ----------------------------- | ------------: | -------: | ----------: |
| Control (c3's recipe)         |    102–116–38 |      40% |         47% |
| Cap-score result (±0.5)       |    103–114–39 |      40% |         47% |
| 1 epoch + top-25% \|A\| + EMA |    105–101–50 |      41% |         51% |
| **Omniscient critic**         | **138–75–43** |  **54%** |     **65%** |
| Potential-based shaping       |     95–116–45 |      37% |         45% |

The omniscient critic was the only arm that beat its parent, and it won on all ten map types. It did so on fewer samples than the control (80M vs 113M), because its actors also shipped the opponent's view. Its recruit mix also broadened: Warriors fell from 43% to 34% (intuitively from a player's perspective this is correct, nobody should be producing this many warriors). Inuitively, this omni critic gave meaningful credit to a defender or archer at the right moment. Fog critic could not see this.

Run for six hours, the same recipe won 77% of games against c3 (+255 Elo), and two seeds tracked within 2–3 points of each other throughout. The curve was still rising at the end. Every bot after this point uses it.

> **[Aside] Dropped: potential-based shaping**
>
> Shaping adds a potential term to the reward, annealed to zero over training. While the potential was strong, the arm fell to 38–40% of decisive games against c3, and it recovered only as the shaping faded. The potential rewarded owning units, so the bot built more units, and every star spent on units was a star missing from city upgrades. Its final score was 37%, below the control.

> **[Aside] Dropped: continuing with a fresh critic**
>
> Continuing c5-omni for an hour with its own recipe, but a freshly initialized critic and optimizer, lost to its own parent (43% of decisive games). The new critic spent that hour relearning values the parent's critic already knew, and the policy trained on its noisy advantages meanwhile. Loading the parent's critic on every continuation removes the warm-up and the early dip.

## Discounting per turn {#credit}

I started with a discount of γ = 0.995 applied at every decision, which sounds like a long horizon until you count decisions. A typical turn has about 15 of them, so the discount per turn is 0.995^15 ≈ 0.93. A reward three turns away is then worth only about 80% of its face value. Most of the economy in Polytopia pays off over several turns, so the bot was undervaluing exactly the moves that win long games.

In one game against a human, the bot won the opening and had five cities by turn 11. Then it cleared 23 forests at one star each and spent 118 stars on units, while the human spent 67. The human took the lead on income at turn 15 and on score at turn 17.

So I moved the discount from decisions to turns. With a turn clock, the discount applies once at the end of each turn, so a turn with three moves and a turn with thirty count the same. I ran matched one-hour continuations from the same parent: a per-decision control, and turn-clock arms at γ = 0.99 and γ = 1.0, all with λ = 0.95. Against the parent, the turn-clock arms won 8–10 percentage points more games than the control. Their critics also spread their value predictions much wider, with a standard deviation of 0.51–0.62 against 0.29. I take that to mean the critic got better at telling good positions from bad ones.

---

> **[Aside] Dropped: UPGO**
>
> UPGO bootstraps advantages along trajectories that beat the critic's expectation, which helps when the critic is weak. With an omniscient critic and λ = 0.99 GAE over 800 decisions, the UPGO advantage was a weaker signal (raw std 0.020 vs 0.044 for GAE, correlation 0.41). Normalizing it to unit scale roughly doubled the policy-gradient weight and dropped entropy from 2.3 to 1.6 in the first hour. After three hours it was at head-to-head parity with its plain sibling (+10 Elo, interval −33 to +52).

## Search at play time {#search}

The policy chooses one action per forward pass. A Polytopia turn consists of many moves: move this unit, then capture, then train, then research, etc. Could be many dozens per turn. So the search works at the level of whole turns.

At each turn, the bot samples 16 candidate turns from its policy and plays each one out. It then lets the opponent's policy reply, and scores the position at the start of its next turn with the critic. The candidate with the best score is the move. This represents depth = 1.

This is similar to AlphaZero - AlphaZero searches with two learned networks: a policy that proposes which moves are worth looking at, and a value function that judges a position without playing the game out to the end. My search keeps that division of labor and drops the tree. A Polytopia turn is a sequence of ten or more actions, and the opponent's hidden units make a deep tree mostly guesswork, so it searches one turn and one reply. The policy narrows the candidates to 16 plausible turns, and the critic replaces the rollout to the end of the game.

One key difference is that AlphaZero's search is for a game with no hidden information. For the fog, I borrowed from imperfect-information games such as bridge, where a standard approach is to sample a full deal consistent with what you can see and search that as if it were known (determinization). We tested a combination of different depths and widths to see what's best:

| Search variant                                 | Elo vs no search |
| ---------------------------------------------- | ---------------: |
| Best of K turns by critic, no reply (K = 4–64) |              ≈ 0 |
| Same, sampling at temperature 0.5              |       +61 to +96 |
| Depth 1, leak-free, 16 candidates (c3)         |             +104 |
| Depth 1, leak-free (stronger c5-omni)          |      +83 to +100 |
| Depth 1, leak-free (champion)                  |            ≈ +50 |
| Depth 2 vs depth 1                             |              +12 |
| Depth 3 vs depth 1                             |              −19 |
| 64 averaged replies vs 1 (strong policy)       |              +16 |

Going deeper didn't help once the critic was good. Every extra ply adds a sampled rollout turn and another sample of the hidden state, and those errors compound faster than the extra lookahead pays. Although this could simply be due to the fact that we didn't scale up the models. Maybe if we had 100x data and bigger model, it would predict more accurately.

Search is also a CPU job. Spread across 16 cores, with each lane's executable on a pinned thread, a searched turn takes about one second. On a GPU the same search took 14.7 seconds, because the engine's branches serialize.

> **[Aside] Dropped: distilling search into the policy**
>
> AlphaZero's answer is distillation: fit the policy to the searcher's choices, so the raw policy inherits the search gain for free. I labeled 23.4 million positions and trained seven students. All of them landed within ±3 Elo of the raw parent.
>
> In 87% of turns, the searcher's preferred turn beat the policy's by less than the spread in its own scores, so most labels were noise. Search also picks whole turns, and a better turn doesn't break down into a clean target for each sub-action. Search on top of the students gained only +40 Elo against +104 on the parent, and freezing the critic didn't restore it. The policy drifted, and the search lost the candidates it had been choosing between.

## Starting over from random weights {#scratch}

By this point, every strong bot descended from league96 and still carried its old (bad) habits. To see how much of the style was inherited, I trained from random weights with the same recipe: no curriculum, no teacher, pure self-play with 80% of games against the current policy and 20% against its own past snapshots.

The curriculum turned out to be unnecessary once the rest of the recipe was in place. A fresh network's policy is already close to uniform (98.8% of maximum entropy), so it explores everything. Within 100 updates, decisive games went from 1% to 98%. Its strategy then went through phases: a Warrior rush peaked at 61% of recruits, and archers came back as the rush stopped working. This matches how humans would play as well.

The overnight run passed c3 at 2.3 hours and matched c5-omni, the best inherited bot, at 3.8 hours. After about 20 hours in total, the "champion" beat c3 in 92% of games and c5-omni in 76%.

> **[Figure 3 — Elo line chart]**
> Caption: Approximate Elo of each successive best bot, with c3 fixed at 0. Hover a point for what changed.[^2]
> X label: successive best bots →
>
> Hover notes — Raw policy:
>
> - league96 (−557): Curriculum + teacher bootstrap + league. Recruits Warriors 99% of the time.
> - u3142 (−147): First distributed PPO run. Beat league96 117–10–1.
> - c3 (0): Self-play on the map mix. The fixed reference for everything after.
> - OMNI (+125): One hour with an omniscient critic.
> - c5-omni (+255): The same recipe for six hours.
> - Scratch (+288): Random weights, seven hours, 2.3 B samples.
> - Scratch, continued (+378): Six more hours.
> - Champion (+442): Seven more hours. Wins 92% against c3 over 512 games.
> - Hero 4×128 (+573): Twice the blocks and twice the width, real game rules, ~7 B samples.
>
> Hover notes — With turn search:
>
> - c3 + search (+104): Leak-free one-turn lookahead over 16 sampled turns.
> - c5-omni + search (+338): The search gain held up as the policy got stronger.
> - Champion + search (+492): The version served for public play.

Why did scratch win? Most likely the inherited prior narrowed exploration. The bootstrapped line kept its Warrior-heavy openings because PPO only makes local updates to a confident policy. A uniform policy tries every opening and keeps what wins. The scratch bot settled on Climbing, then Archery by turn 3–5, and fields archers. Its catapult share started near zero and climbed to 7% as its games got longer and catapults had time to pay off.

## Balancing CPU and GPU {#balance}

One training sample involves three kinds of work: simulating the game (branchy, integer-heavy, irregular), running the network (dense matrix math), and assembling batches (moving memory). CPUs are good at the first and GPUs at the second, so moving the batches becomes the bottleneck once the other two are fast.

Most of the engineering went into one question: where does each training sample spend its time? The bottleneck moved every time one stage got faster, so I kept profiling one actor tick and one learner update and fixing whichever was slower.

Essentially, much of the auto-research-styled-engineering-loop becomes doing a short run, identifying which of the 3 is the bottleneck, optimizing, repeat, and see where the bottleneck shifts, until you hit some sort of hardware / architectural limit.

| Stage         | What was slow                                                     | Fix                                             | Samples/s |
| ------------- | ----------------------------------------------------------------- | ----------------------------------------------- | --------: |
| GPU simulator | Branchy rules on a batched GPU; 30–50% of lanes doing useful work | CPU actor fleet, one GPU learner                | 488 → 23k |
| CPU actors    | Actors ran the critic; legal-action masks rebuilt twice per step  | Values on the learner, masks reused             |     → 40k |
| Lean actors   | JAX engine at 45–60 ms per 24-game tick                           | Bit-exact Rust port                             |    → 117k |
| Rust engine   | Policy forward pass at 93–97% of actor CPU                        | GPU inference server per actor container        |    → 135k |
| GPU inference | Learner was waiting on host-side batch assembly                   | Index gathers, whole-frame reads, upload thread |    → 208k |

> **[Figure 4 — throughput step chart, log scale]**
> Caption: Measured end-to-end training throughput at each stage, log scale. Hover for what changed.
> Stage labels: GPU sim · CPU actors · Lean actors · Rust engine · GPU inference · Assembler
>
> Hover notes:
>
> - Simulator on one GPU (488): The whole loop on one L40S. Collecting games took 91–96% of each update.
> - CPU actors + GPU learner (23,200): Game simulation moves to fleets of CPU containers. One GPU trains.
> - Lean actors (40,000): The learner scores values and the engine reuses masks. Actor tick 2.2× faster.
> - Rust engine (117,000): A bit-exact Rust port of the rules. The network forward pass becomes the actor's main cost.
> - GPU inference for actors (135,000): Actors batch their forward passes on A10G inference servers.
> - Faster batch assembly (208,000): Stale samples gathered by index, frames read whole, uploads on their own thread.

- **Put the game on CPUs.** The first simulator was a batched JAX engine on the GPU. Polytopia's rules branch heavily (pathfinding, combat, city growth, a dozen unit special cases), and batching branchy code on a GPU wastes most of it. Each step paid about 90 ms of fixed cost, and only 30–50% of lanes did useful work. Moving simulation to CPU actor containers with one GPU learner gave 48–95× more samples per second at a seventh of the cost per sample.
- **Move value estimation to the learner.** Actors were running the critic on every step, and the learner recomputed those values anyway. Now actors run only the policy, and the learner scores a whole batch in one GPU pass.
- **Never compute a mask twice.** Every engine step rebuilt all legal-action masks, which was 65% of engine time. A trusted-step path reuses the masks it already built and spot-checks against a full rebuild every 64 steps.
- **Port the engine to Rust.** The rules came to 3,900 lines of Rust, run in lockstep beside the original JAX engine for 12.5 million steps and compared on all 72 state fields and 17 mask families. A 24-game tick went from 45–60 ms to 3.5 ms, then to about 0.6 ms with preallocated buffers and bit-packed masks.
- **Move inference back to a GPU.** With the game that cheap, the policy forward pass was 93–97% of actor CPU. Each actor container got a small A10G inference server that batches forward passes across hundreds of games. Simulation stays on CPU; inference and training run on GPU.
- **Feed the learner.** Past about 100k samples per second, the GPU sat waiting on host-side batch assembly. Shared-memory batch pages and pinned buffers raised the learner's ceiling to 117k. Gathering stale samples by index on the copy threads and uploading on a dedicated thread then took the 1v1 trainer from about 120k to 208k. The trunk also trains in bf16 with fp32 heads, 1.66× faster at the same KL.
- **Cut the bytes.** Sending a 1 MB policy to every actor process every two updates made network egress 30% of the bill. A per-container relay sends weights once per machine. Running 32 processes on each 16-vCPU container gave 1.6× more samples per container.
- **Size the fleet to the learner.** Past the learner's intake, extra actors only produce stale samples that PPO drops. I size the fleet from the learner's measured throughput.

The pattern generalizes: put each kind of work where it is cheap, then size each side to the other. Below the learner's intake the GPU idles; above it, extra actors only add stale samples. On every run I watched two numbers, GPU idle time and actor wait time, and moved hardware toward whichever side was starved.

Every one of these changes shipped with a bitwise gate: the previous path and the new one had to produce identical bits on the same inputs. That made it safe to change the training system freely, because a speedup could never quietly change the experiment.

> **[Figure 5 — cost per million samples, columns]**
> Caption: Cost per million training samples at each stage of the stack. Hover for the setup.
>
> - GPU sim (one L40S) — $1.47: Everything on one GPU.
> - CPU actors (first fleet) — $0.225: 1×H100 learner and 12 CPU actor containers.
> - Lean actors + Rust — $0.09: 1×H100 and 6 actor containers.
> - Relay, 32 procs per container — $0.044 (highlighted): 1×H100 and 8 containers of 32 processes. Estimated from measured per-container rates.

## Going bigger - scaling law? {#scaling}

With cheap samples, the next lever was model size. A network with 4 blocks of 128 channels has about 8× the trunk compute of the champion's 2×64.

The first big-net run fell behind the small one at equal samples (−85 Elo). Its minibatch was 4× smaller, which meant 4× more Adam steps per batch and a 1.5× higher KL per update: the bigger network was taking more, noisier steps. With the minibatch restored to 16,384, it pulled ahead of the small net at matched samples (+26 Elo at 3B samples, about +50 at 6B).

The "hero" run trained that network from scratch under the real game's rules (see [Against the real game](#results)). It drew level with the champion at about 4 billion samples, and after a lower-learning-rate continuation it wins 68% of games against it.

> **[Figure 6 — hero win rate line chart]**
> Caption: Win rate of the 4×128 hero network against the 520k-parameter champion, by training samples. The last point is after a lower-LR continuation on 4×H100.
> Reference line: parity (50%)
>
> Hover notes:
>
> - 0.45 B samples (0%): update 1,700
> - 1.5 B samples (25%): update 5,700
> - 2.0 B samples (38%): update 7,800
> - 3.3 B samples (47%): update 12,650
> - 3.9 B samples (52%): end of the first run, 192 games
> - 7.0 B samples (68%): after the continuation, 128 games (+131 Elo)

Two other scaling directions use the same recipe. On 18×18 maps, a network grown from c3 (its embedding remapped to 324 tiles, with extra zero-initialized blocks) only started improving once it had both the omniscient critic and per-turn discounting. It then won 62% of games against its parent (76% of decisive ones). A four-player free-for-all version, trained from the champion with rank-based rewards, reached +158 Elo over its starting point. A seat-relabeling symmetry test (permute the players, check that the game is identical) keeps the four-seat engine honest.

A final polish (three more hours at LR 1e-4) added +41 Elo over the champion.

> **[Figure 7 — ablation delta bars]**
> Caption: Effect of each idea in Elo, against its own control or parent. Whiskers are 95% intervals where computed. Hover for the intuition.
>
> Hover notes:
>
> - Omniscient critic (+125, 92 to 160): Less noise in every advantage estimate, since the critic no longer guesses at hidden state.
> - One-turn search at play time (+104): +83 to +100 on stronger policies, about +50 on the champion.
> - Per-turn discounting (+77): From 31.6% to 41.8% against the parent. Treat as an upper bound.
> - Bigger net (4×128), same samples (+50): Holds once the minibatch scales with the network.
> - Low-LR polish (3 h) (+41, 19 to 62): Lower LR and extra training are confounded here.
> - Search depth 2 vs 1 (+12, −66 to 94): 1.75× the compute for no measurable gain.
> - 64 opponent replies vs 1 (+16, −65 to 100): Worth +72 on a weaker policy. Gone on a strong one.
> - UPGO (+10, −33 to 52): A weaker signal than GAE once the critic is omniscient.
> - Distilling search into the policy (0, −13 to 13): Search labels were mostly noise at the level of single actions.
> - Search depth 3 vs 1 (−19, −94 to 53): Rollout noise compounds with depth.
> - Potential-based shaping (−22): The potential rewarded material, so the bot overbuilt units.
> - Continuing with a fresh critic (−49): Resetting the critic and Adam costs more than an hour of training wins back.

## The RL tricks, collected {#tricks}

Here's a collection of RL tricks and techniques to sum everything up.

### Getting a learning signal

- **Factored, masked action heads.** The flat action space has 130,868 entries per decision. Choosing family, then source tile, then target, with illegal options masked at each step, keeps every head small and every sampled action legal. No probability is wasted on moves that can't happen.
- **A curriculum that starts near the reward.** With one ±1 at the end of 800 decisions, a random policy never sees a win. Starting episodes one, two, then three actions from a capture puts the reward inside the horizon. Once the rest of the recipe worked, from-scratch self-play no longer needed it.
- **Bootstrap at the turn cap.** Scoring capped games as draws throws away who was ahead. Bootstrapping from the critic's estimate keeps a gradient in exactly the longest, hardest games.

### Credit assignment

- **Omniscient critic.** The critic is used only in training, so it can see the whole board. A fogged critic has to guess the enemy's army and bank, and every bad guess becomes advantage noise.
- **Discount once per turn.** γ per decision compounds about 15 times a turn. Discounting at turn boundaries matches the horizon to game time, so multi-turn economy moves get their credit.
- **Check the advantage floor.** Advantages are divided by max(std, floor). When the raw std sat below a floor of 0.05, the effective step size silently changed between arms, and on 18×18 it fed an entropy runaway. A floor of 0.01 fixed it.
- **Carry the critic across continuations.** A fresh critic spends its first hour relearning values while the policy trains on its noisy advantages. Loading the parent's critic and optimizer state removes the dip.

### Self-play and evaluation

- **A pool of past selves.** About 80% of games against the current policy and 20% against older snapshots, so strategies the bot has moved past stay in the training data and can't quietly come back as blind spots.
- **Seat-paired maps and decisive games.** Turn order and the map decide many games between close bots, so every map is played from both seats, and capped games are read separately.
- **A live record against the parent.** Logging the training-time win rate against the frozen parent every 50 updates predicted every collapse within about 30 minutes, far sooner and cheaper than a tournament.

### Optimization

- **Big batches.** 262k samples per update with minibatches of 16,384. Sparse rewards over long games make gradients noisy, and large batches average the noise down.
- **Scale the minibatch with the network.** A bigger net at a 4× smaller minibatch took 4× more, noisier Adam steps and looked worse than the small net. Restoring the minibatch reversed the result.
- **Polish at a lower learning rate.** Decaying the LR from 3e-4 to 1e-4 over the last quarter of a run gave about +40 Elo, though extra training time is mixed into that number.
- **Gate mixed precision.** bf16 and TF32 convolutions with fp32 heads trained 1.7–3.3× faster. Each switch went in only after PPO ratios and KL matched the fp32 run.
- **Cap staleness.** Actors play with weights a few updates old. Samples past a maximum lag are dropped so the PPO ratio still means something.

### Search

- **AlphaZero's split of the work.** The policy proposes candidates and the value function scores positions, so the search never plays a game out to the end. That cuts both the breadth and the depth the search has to cover.
- **Search whole turns and let the opponent reply.** Scoring candidate turns by the critic alone picks its biggest overestimate. Simulating the reply first scores consequences.
- **Determinize.** Sample a hidden state consistent with what the bot has seen before each lookahead. Peeking through the fog looked twice as good, and half of that was cheating.
- **Train a fogged value head afterwards.** From-scratch bots only trained an omniscient critic, which can't run at play time. Freezing the policy and fitting a fogged value head for an hour gave the search a scorer.

## Against the real game {#results}

Elo only shows the bot beats its relatives inside a reimplementation of the rules.

**Rules parity.** I ran the game's own engine headless and replayed 27,785 real transitions through the bot's engine, comparing every field. After fixing every difference, the legal action sets match in 685 of 686 real positions. One gap is kept on purpose: the real game allows roads on unexplored tiles, which would leak hidden information. All training since then uses the matched rules.

**The built-in AI.** A Python replica of the game's AI matches 28,640 of 28,641 of its real decisions. Against the real thing at its hardest difficulty, the champion won every game:

| Matchup                               | Result | Games |
| ------------------------------------- | -----: | ----: |
| Champion + search vs real Crazy AI    |   48–0 |    48 |
| Champion (raw) vs real Crazy AI       |   24–0 |    24 |
| Champion + search vs OMNI c3 + search |    81% |   200 |
| Champion (raw) vs c5-omni (raw)       |    76% |   512 |
| Champion (raw) vs c3 (raw)            |    92% |   512 |
| Hero 4×128 (raw) vs champion (raw)    |    68% |   128 |

Against the Crazy AI, the average win came on turn 21.

> **[Figure 8 — video slot]**
> Caption: Video slot: a 30-second capture of the champion against the real Crazy AI. [draft: record in Mock art mode]

## Things I noticed watching it play {#observations}

- **The 1v1 bot rushes.** It clears forests early for the quick stars and blitzes the enemy, and most of its wins come before turn 30. That pays off against everything it has met so far, but I suspect longer training with per-turn discounting will push it toward longer-term play.
- **The four-player bot builds.** It spends longer on its economy, then clears its forests late to harvest the stars and fund the frontier. I rarely see human players do that.
- **The 1v1 champion likes Cloaks.** In the archipelago replay above, Cloaks and the Daggers they spawn made 7 of the champion's last 11 captures.
- **The hero fails at Diplomacy.** The bigger 4×128 bot researched Diplomacy in 2 of 106 land games. When its opponent did, it won 1 of 25 against the champion and 2 of 35 against another big net. The replays show the same shape: a city standoff broken by a Dagger swarm. My explanation is the opponent pool. Nothing in the hero's own lineage played Cloaks, so it never saw the threat, and self-play has no gradient toward defending against something that never happens. The fix is to seed the pool with Cloak players, maybe.
- **Unit choice carries the bootstrap's fingerprint.** To break league96's Warrior habit, I bootstrapped the next policies from rewritten teachers with hand-written unit preferences, and PPO from those starting points settled on catapults (30–42% of u3142's army against strong opponents). Every bot descended from them kept the lean: the c3 line built catapults for 20–27% of its units and archers for 2–3%, run after run, because each run started from the previous checkpoint. The from-scratch line went the other way. Archers made up 24–36% of its units, and catapults started near 1% and only reached 7% as its games got longer. That matches the economics: Bardur starts with Hunting, so Archery is a single tier-2 tech, while catapults need Forestry and Mathematics (about 2.3× the stars), cost 8 stars against 3, have no defense, and can't move and shoot in one turn. Biased starting data shaped every run that descended from it.

I'm curious how all of this shifts with bigger networks and longer runs.

## What it cost {#cost}

All in, the project cost about $7,000 of cloud compute, mostly on Modal. I tried a few other GPU clouds but nothing as stabile or UX friendly. A lot of platforms turned out quite flakey and ended up disrupting runs, costing more money.

| Phase                                                     | Approx. cost |
| --------------------------------------------------------- | -----------: |
| Bootstrap, first CPU fleet, search studies                |       $1,400 |
| Ablations, omniscient-critic runs, search sweeps          |       $1,350 |
| From scratch, engine and actor speedups, real-game checks |         $800 |
| Big-net runs (4×128)                                      |         $500 |
| Hero runs (1v1 and four-player), GPU inference            |       $2,850 |
| **Total**                                                 | **≈ $6,900** |

The champion's own training, three runs adding up to about 20 hours, cost about $500. Everything else went into finding out what to train. Most of it went early: a million samples cost $1.47 at the start and $0.044 at the end (Figure 5), so the first experiments paid about 30× more per sample than the hero runs did.

## What I'd do differently {#lessons}

- **Start from the budget.** I should have run a Chinchilla-style tradeoff first: fix the compute (say, a few thousand dollars of credits), sweep a few network sizes against sample counts at that budget, then spend the rest on the winner. I learned only late that the 4×128 network beats the 2×64 one at equal samples once its minibatch scales. A small sweep up front would have told me which size to bet on before most of the money was gone.
- **Build the environment once.** The JAX simulator was a detour. Branchy game rules belong on CPUs, and the Rust engine that replaced it ran many times faster. I also started from an open-source engine, and only later decompiled the real game and aligned the rules one-to-one, which meant retraining under the real rules. Doing that first would also have handed me the game's own AI, at every difficulty, as an absolute yardstick from day one. Elo alone told me how much better each bot was than the last, but not how good it was.
- **Decide how much of the loop to automate.** Most rounds went the same way. The agent finished a run, I watched replays and played against the result, I asked for research directions, the agent gathered papers and tricks (AlphaStar, KataGo, the top generals.io bot), I reviewed them, and the next run went out. I could have automated far more of that loop. I'd have learned less.

## Final thoughts {#final}

The overall project was done with roughly 1 week of $200 max plan Claude with Fable and Opus 5.5. Overall it was both quite a good researcher with taste and a good teacher. Training a game bot, with verifiable rewards, in some ways is probably pretty trivial to Fable at this point.

A more structured rerun would require me to map out the initial compute budget and goals clearer, optimize earlier for speed and cost, and run more of an auto reserch style, letting it rip with a daily budget instead of baby sitting the runs more often.

Some more ideas.

- KataGo-style auxiliary heads (territory ownership, score margin, the opponent's next move) and global pooling in the trunk. This idea seems super promising with a bigger network. Intuitively it's like teaching a human which parts of the map they should be paying attention and "checking" every turn.
- An opponent pool seeded with the styles the hero struggles against, starting with Diplomacy and Cloak rushes. This can be auto built up adversarially as the network is built up as well.
- Strong human players. If you play Polytopia seriously and want to try the bot, [get in touch](mailto:zkw917@gmail.com). In this run i had very limited human testing. Tested the bot on a friend (a self-proclaimed Polytopia GOAT) and the 1v1 forest clearing rush strategy overwhelmed him a bit but was unclear how far it would go.
- One can imagine a fully autoresearched, human in the loop setting of this - the model can self research, release it into prod periodically, get feedback, organize real world experiments, and come back to iterate. A fully enclosed loop is totally possible.

## Notes {#appendix}

### The network

The board is encoded as a 14×14 grid of feature planes from the acting player's fogged view: terrain, ownership, cities, units, and what has been explored. A vector of global features (turn, stars, techs) joins after the convolutional trunk is pooled. The trunk is two residual blocks of 64 channels, which gives an 11×11 receptive field.

The action space flattens to 130,868 entries, and nearly all of them are illegal at any moment. Masking the full space would make the output layer enormous, so the policy factors it into three heads: an action family (move, attack, train, build, research, end turn…), then a source tile, then a target. Each head is masked to legal options given the earlier choices, and the log-probability of the action is the sum of the three. About 210k of the 520k parameters are the target embedding table.

The critic is a separate network. Its input stacks both players' views of the full board, and it outputs a single value in [−1, 1].

### More dead ends

- **KL anchors for unit diversity.** To fix the Warrior habit, I added a unit-choice KL term and a unit-entropy bonus. The diverse policies played worse (one went 2–30 against league96). Diversity forced by a loss term costs strength. Diversity found by self-play from scratch came with strength.
- **Best-of-K with Gumbel noise at the root.** Forcing the K candidate turns to start with distinct first actions cost −413 Elo. Probably most good turns share an obvious first move, so forcing distinct ones fills the candidate set with bad turns.
- **Top-25% advantage filtering with EMA weights.** Training only on the quarter of samples with the largest |A| helped the strongest generals.io bot, which is learner-bound with a free simulator. Here it was neutral (41% vs control 40%), and with 493 updates at EMA 0.999 the deployed weights were still mostly the parent's, so EMA itself went untested.
- **Bigger batches.** A 65k-sample batch matched the default per sample. It neither helped nor hurt.
- **A big net on a short budget.** At 325M samples, a 4×128 network won 13 of 14 decisive games against the small one, but 114 of 128 games hit the turn cap. Bigger networks need billions of samples before they finish games decisively.
- **Self-play on 18×18 without the new critic.** Three runs from the grown c3 produced no checkpoint that beat their parent (best: 47% of decisive games). A round-robin of the plateaued checkpoints showed a clean transitive ordering with no cycles: the runs lacked a learning signal. The omniscient critic plus per-turn discounting fixed it.

---

[^1]: Both seats received the hardest difficulty's income bonus (+3 stars a turn over a human player), so these were even-economy games.
[^2]: Each bot was measured against one or two neighbors, usually in 256–512 seat-paired games, and Elo was chained through those matches. The hero was measured under the real game's rules, which the earlier bots also play under when loaded.
