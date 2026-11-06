#!/usr/bin/env python3
"""SFT-0 prerequisite execution; real game traces are not qualified labels."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def pilot(output, manifest, *, maps=2, opponent='weak', max_decisions=10000):
    from polytopia_bot.learning.adapter import GPUEnvironment
    from polytopia_bot.learning.interface import digest
    from polytopia_bot.learning.teacher import ScriptedTeacher, WeakBaseline
    from polytopia_bot.gpu.snapshot import implementation_digest
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    teacher = ScriptedTeacher()
    other = WeakBaseline() if opponent == 'weak' else ScriptedTeacher(opponent)
    protocol = dict(schema='teacher-development-v1', map_manifest_hash=manifest['manifest_hash'],
        split='teacher_dev', maps=maps, seats=[1, 2], teacher_hash=teacher.teacher_hash,
        opponent_hash=other.teacher_hash, opponent=opponent, simulator_sha256=implementation_digest(),
        horizon_actions=max_decisions, qualification=False)
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    results, started = [], time.monotonic()
    env = GPUEnvironment(manifest['size'])
    try:
        for board in manifest['maps']['teacher_dev'][:maps]:
            for seat in (1, 2):
                obs = env.reset(seed=board['seed'], settings={})
                name = f"seed-{board['seed']}-seat-{seat}"
                private = root / 'private'
                private.mkdir(exist_ok=True)
                (private / f'{name}.snapshot.json').write_bytes(env.snapshot())
                families = Counter()
                game_start = time.monotonic()
                with (root / f'{name}.jsonl').open('x') as trace:
                    for decision in range(max_decisions):
                        policy = teacher if obs.player == seat else other
                        selected = policy.select(obs)
                        trace.write(json.dumps(dict(observation=asdict(obs), input_digest=digest(asdict(obs)),
                            selection=asdict(selected)), separators=(',', ':')) + '\n')
                        families[str(selected.action.command['kind'])] += 1
                        transition = env.step(action_id=selected.action.action_id, expected_state_version=obs.state_version)
                        if transition.terminated or transition.truncated:
                            break
                        obs = env.observe(transition.next_player)
                        if decision % 250 == 249:
                            print(f'{name}: decisions={decision+1} round={obs.visible_state["turn"]}', flush=True)
                result = dict(seed=board['seed'], board_hash=board['board_hash'], seat=seat,
                    decisions=decision+1, round=int(env.state.turn), terminated=transition.terminated,
                    truncated=transition.truncated or not transition.terminated,
                    reason=transition.reason or 'pilot_action_limit', reward=transition.rewards.get(seat, 0),
                    families=dict(families), seconds=time.monotonic()-game_start)
                results.append(result)
                (root / 'games.json').write_text(json.dumps(results, indent=2))
                print(json.dumps(result), flush=True)
        report = dict(protocol=protocol, games=results, elapsed_seconds=time.monotonic()-started,
            native_completion=sum(r['terminated'] for r in results)/len(results),
            wins=sum(r['reward'] > 0 for r in results),
            teacher_qualified=False, note='Development pilot only; no qualification or training authorization inferred from its outcome.')
        (root / 'report.json').write_text(json.dumps(report, indent=2))
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('reserve-splits')
    p.add_argument('--output', required=True)
    p = sub.add_parser('qualify-teacher-parallel')
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', required=True)
    p = sub.add_parser('qualify-teacher')
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', required=True)
    p = sub.add_parser('teacher-pilot')
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--maps', type=int, default=2)
    p.add_argument('--opponent', choices=['weak', 'economy', 'pressure'], default='weak')
    p.add_argument('--max-decisions', type=int, default=10000)
    args = parser.parse_args()
    if args.command == 'reserve-splits':
        from polytopia_bot.learning.splits import reserve
        result = reserve(args.output)
        print(json.dumps({k: v for k, v in result.items() if k != 'maps'}, indent=2))
    elif args.command == 'qualify-teacher-parallel':
        from polytopia_bot.learning.research import qualify_parallel
        qualify_parallel(args.output, json.loads(Path(args.manifest).read_text()))
    elif args.command == 'qualify-teacher':
        from polytopia_bot.learning.research import qualify
        qualify(args.output, json.loads(Path(args.manifest).read_text()))
    else:
        pilot(args.output, json.loads(Path(args.manifest).read_text()), maps=args.maps,
              opponent=args.opponent, max_decisions=args.max_decisions)


if __name__ == '__main__':
    main()
