#!/usr/bin/env python3
"""Run from repo root: uv run --project training python scripts/sft.py --help."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polytopia_bot.learning.data import observation_from_dict, write_fixtures
from polytopia_bot.learning.training import prepare, read_config, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.json")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("fixture", help="write explicitly synthetic public-state demonstrations")
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=8)
    p = commands.add_parser("prepare", help="audit and tokenize response-only examples without truncation")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-fixtures", action="store_true")
    p.add_argument("--split-manifest")
    p.add_argument("--prompt-manifest")
    p = commands.add_parser("train", help="run one seed of BF16 LoRA on a CUDA GPU")
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--stage", choices=["fixture", "fit", "smoke", "main"], required=True)
    p.add_argument("--readiness")
    p = commands.add_parser("tiny-smoke", help="verify actual LoRA updates and reload locally, without 4B weights")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=80)
    p = commands.add_parser("teacher", help="score one public observation")
    p.add_argument("--observation", required=True)
    p.add_argument("--style", choices=["balanced", "economy", "pressure"], default="balanced")
    p = commands.add_parser("predict", help="run the base or an adapter on one public observation")
    p.add_argument("--observation", required=True)
    p.add_argument("--adapter")
    p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = read_config(args.config)
    read = lambda path: json.loads(Path(path).read_text()) if path else None
    if args.command == "fixture":
        result = {"synthetic_records": write_fixtures(args.output, args.count)}
    elif args.command == "prepare":
        result = prepare(args.input, args.output, config, allow_fixtures=args.allow_fixtures, split_manifest=read(args.split_manifest), prompt_manifest=read(args.prompt_manifest))
    elif args.command == "train":
        result = train(args.prepared, args.output, config, seed=args.seed, stage=args.stage, readiness=read(args.readiness))
    elif args.command == "tiny-smoke":
        from polytopia_bot.learning.smoke import tiny_smoke
        result = tiny_smoke(args.output, args.steps)
    elif args.command == "teacher":
        from polytopia_bot.learning.teacher import ScriptedTeacher
        result = asdict(ScriptedTeacher(args.style).select(observation_from_dict(read(args.observation))))
    else:
        from polytopia_bot.learning.inference import LocalGenerator
        from polytopia_bot.learning.interface import choose_action
        action, attempts = choose_action(observation_from_dict(read(args.observation)), LocalGenerator(config, adapter=args.adapter, device=args.device))
        result = {"action": asdict(action), "attempts": attempts}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
