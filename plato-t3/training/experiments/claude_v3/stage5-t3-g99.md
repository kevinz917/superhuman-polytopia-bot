# stage5-t3-g99.json (T3 recipe) note

turn_gamma 0.99 instead of 0.98: 0.98 trades about a third of relative win probability for winning about 20 turns
sooner, and Elo only counts wins. This file is the stage-5 T3 recipe with turn_gamma .99 (the previous variant's
value). critic_warmup 89090 (critic-only updates until u88890 + 200) is kept: its original reason was the gamma
change (the critic was fitted at .99), so with .99 it is optional; it also lets the critic adapt to the
observation-6 widened inputs before policy updates.
