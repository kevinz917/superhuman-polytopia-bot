"""Observation-only policies and SFT tooling, independent of the JAX runtime.

Frozen legacy (2026-10-27): the expert-SFT / language-model line here plays the retired research rules on the JAX
engine (gpu/, adapter.GPUEnvironment) and is not maintained. The game's rules are engine-rs's
(polytopia_bot.ruleset, docs/rules-alignment-2026-10-27.md). Studio reuses only adapter.serialize, interface and
memory, which are rules-free.
"""
