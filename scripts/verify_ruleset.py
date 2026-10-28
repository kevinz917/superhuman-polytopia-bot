"""Verify that the working GPU simulator matches the current ruleset manifest (v2; pass a path for another)."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / (sys.argv[1] if len(sys.argv) > 1 else "configs/ruleset-v2.json")).read_text())
    changed = []
    for relative, expected in manifest["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(relative)
    if changed:
        raise SystemExit("Frozen ruleset differs:\n" + "\n".join(changed))
    print(f"{manifest['ruleset_id']}: all {len(manifest['source_sha256'])} source hashes match")
    print(f"Validation: {manifest['validation_status']}")


if __name__ == "__main__":
    main()
