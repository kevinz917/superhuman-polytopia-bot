#!/usr/bin/env python3
"""Fetch the pinned rules reference and build the validation-only executable."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = json.loads((ROOT / "configs/research-sources.json").read_text())["polyfish"]
repo = ROOT / "vendor/Polyfish"
if not repo.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", SOURCE["url"], str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", SOURCE["commit"]], check=True)
rev = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
if rev != SOURCE["commit"] or dirty:
    raise SystemExit("Oracle source differs from the pinned clean revision; refusing to certify fixtures")
subprocess.run(["cargo", "build", "--locked", "--manifest-path", str(ROOT / "tools/polyfish-oracle/Cargo.toml")],
               env={**os.environ, "CARGO_TARGET_DIR": str(repo / "polyfish-rs/target")}, check=True)
