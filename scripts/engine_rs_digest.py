"""Digest of the game engine: sha256 over engine-rs's sources, manifest and lock file, and the rule table it loads.

    python scripts/engine_rs_digest.py

Checkpoints record polytopia_bot.ruleset.RULESET_ID and rules_digest() (the name and the rule table); this digest
pins the engine code itself (docs/rules-alignment-2026-10-27.md).
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def engine_rs_digest(root=ROOT):
    digest = hashlib.sha256()
    files = sorted((root / 'engine-rs/src').glob('*.rs')) + [root / 'engine-rs/Cargo.toml', root / 'engine-rs/Cargo.lock',
                                                             root / 'src/polytopia_bot/gpu/rules.json']
    for p in files:
        digest.update(str(p.relative_to(root)).encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


if __name__ == '__main__':
    print(engine_rs_digest())
