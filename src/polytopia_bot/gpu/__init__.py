"""The retired JAX engine of the research rules (bardur-1v1-v2), kept for what is not game rules.

The game's rules are engine-rs's (polytopia_bot.ruleset, docs/rules-alignment-2026-10-27.md); nothing current steps
this engine. Still used from here: the State layout (state.py), engine-rs's rule table (rules.json), the Polyfish
importer (importer.py), the map generator (mapgen.py, for determinized search and map variants), the view
schema (kernels.observe, read by rl.cpu_engine.view_layout) and state snapshots (snapshot.py).
"""

ENGINE_REVISION = "a9f90ab54607f5633556d02422d7457f548cc58f"
RULESET_ID = "bardur-1v1-v2"   # the retired research rules (the frozen learning line); see polytopia_bot.ruleset
FULL_GAME_READY = True
BOARD_SIZES = (11, 14, 18)
# Round cap per board size; v1 boards keep 100. Reaching it truncates (never a win).
ROUND_LIMITS = {11: 100, 14: 100, 18: 150}


def round_limit(size):
    return ROUND_LIMITS[size]
