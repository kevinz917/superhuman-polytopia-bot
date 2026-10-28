"""The game every bot here trains and plays: Polytopia's own rules (Steam 2.17.3, game version 123, game data
v28), Bardur, 2 or 4 seats, as implemented by engine-rs (docs/rules-alignment-2026-10-27.md). There is one ruleset;
the engine has no other rules to choose from.

Checkpoints record RULESET_ID and RULES_SHA256. Checkpoints from before 2026-10-27 were trained under the retired
research rules (the JAX engine, gpu/); their weights load unchanged (LEGACY_DIGESTS) and they play under these rules,
which is how the old champions serve as evaluation references.
"""
import hashlib
from importlib.resources import files

RULESET_ID = 'polytopia-2.17.3'
BOARD_SIZES = (11, 14, 18)
# Round cap per board size. Reaching it truncates the game (never a win).
ROUND_LIMITS = {11: 100, 14: 100, 18: 150}

# simulator_sha256 values of checkpoints trained under the retired research rules (bardur-1v1-v1/v2, the four-seat
# research rules): the digest of the JAX engine package (gpu/*.py + rules.json) of their time.
LEGACY_DIGESTS = {
    '7b76e65b74afcd8d4732712b12a2f9e001fafc1a5b2899982b971a9fd78cb134': 'bardur-1v1-v1 (JAX engine, through 2026-10-19)',
    'e4b656d34cb8721e059837980365a5cbc606e52b9d10876e0eb9596ef012d6b9': 'bardur-1v1-v2 / four-seat research rules',
}


def round_limit(size):
    return ROUND_LIMITS[size]


def rules_json():
    """engine-rs's rule table (units, techs, structures): the single source of the game data it uses."""
    return files('polytopia_bot.gpu').joinpath('rules.json').read_text()


def rules_digest():
    """Digest of the rules a checkpoint was trained under: the ruleset name and the rule table."""
    digest = hashlib.sha256(RULESET_ID.encode())
    digest.update(rules_json().encode())
    return digest.hexdigest()


def compatible(meta):
    """A checkpoint (its .json metadata) whose weights this code can load: trained under these rules, or a legacy
    reference from the research rules."""
    return meta.get('ruleset') == RULESET_ID or meta.get('simulator_sha256') in LEGACY_DIGESTS
