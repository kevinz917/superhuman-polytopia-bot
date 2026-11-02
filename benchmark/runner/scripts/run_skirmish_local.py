"""Run Mini-SWE-agent locally with many sessions and bounded offline Docker work."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from polytopia_bot.skirmish.local import cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--manifest', type=Path, default=Path('configs/skirmish/local-v1-top12.json'))
    parser.add_argument('--results', type=Path, default=Path('artifacts/skirmish-local'))
    parser.add_argument('--references', type=Path, default=Path('artifacts/skirmish-references'))
    parser.add_argument('--concurrency', type=int, default=24)
    parser.add_argument('--cpu-slots', type=int, default=6)
    parser.add_argument('--smoke', action='store_true', help='30-second stages, one build/map per family; not a leaderboard')
    parser.add_argument('--evaluate-only', action='store_true', help='Resume frozen evaluation, never call authoring models again')
    args = parser.parse_args()
    report = cohort(args.prefix, json.loads(args.manifest.read_text()), args.results, args.references,
        concurrency=args.concurrency, cpu_slots=args.cpu_slots, smoke=args.smoke, evaluate_only=args.evaluate_only)
    print(json.dumps({k:report[k] for k in ('status','smoke','games','ranking')}, indent=2))


if __name__ == '__main__':
    main()
