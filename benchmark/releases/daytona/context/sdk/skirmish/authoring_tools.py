"""Offline authoring tools: real mechanics, public maps, no opponent or match runner."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import canonical, sha
from .wasi import BotFault

MAP_TYPES = ('drylands', 'lakes', 'continents')
PUBLIC_MAPS = {f'{kind}-{i}': (kind, 410000 + j * 10 + i)
               for j, kind in enumerate(MAP_TYPES) for i in range(3)}
MAX_ACTIONS = 32
DEFAULT_EXAMPLES = Path('/reference/examples')


def environment(map_id):
    """Exactly the evaluator's layout generator, Rust engine and public serializer."""
    import numpy as np
    from polytopia_bot.maps.variants import generate
    from polytopia_bot.studio.environment import RustEnvironment
    if map_id not in PUBLIC_MAPS:
        raise ValueError('choose a supplied public map from `skirmish maps`')
    kind, seed = PUBLIC_MAPS[map_id]
    layout, _ = generate(seed, kind, size=14)
    env = RustEnvironment(14)
    row = lambda a: np.asarray(a, np.int32)[None]
    env.games.batch.reset_layouts(np.array([seed], np.int32), row(layout.terrain),
                                 row(layout.structure), row(layout.resource), row(layout.capitals))
    env._sync()
    env.observe(2)
    env.observe(1)
    return env


def inspect_map(map_id):
    from polytopia_bot.maps.variants import generate
    kind, seed = PUBLIC_MAPS[map_id]
    layout, _ = generate(seed, kind, size=14)
    env = environment(map_id)
    try:
        return dict(map_id=map_id, map_type=kind, seed=seed, board_size=14,
                    note='Public test layout, not a hidden evaluation map.',
                    layout={k: getattr(layout, k).tolist() for k in layout._fields},
                    observation=asdict(env.observe(1)))
    finally:
        env.close()


def replay(map_id, actions):
    """Check a short caller-authored action sequence. Never choose an opponent move."""
    if not isinstance(actions, list) or len(actions) > MAX_ACTIONS or any(type(a) is not str for a in actions):
        raise ValueError(f'actions must be a JSON array of at most {MAX_ACTIONS} action-ID strings')
    env = environment(map_id)
    try:
        events = []
        for action_id in actions:
            obs = env.observe(int(env.state.current_player))
            action = next((a for a in obs.legal_actions if a.action_id == action_id), None)
            if action is None:
                raise ValueError(f'illegal action at index {len(events)}: {action_id}')
            transition = env.step(action_id=action_id, expected_state_version=obs.state_version)
            events.append(dict(observation=asdict(obs), action_id=action_id, command=dict(action.command),
                               terminated=transition.terminated, truncated=transition.truncated,
                               reason=transition.reason))
        obs = env.observe(int(env.state.current_player))
        return dict(map_id=map_id, events=events, observation=asdict(obs),
                    terminated=bool(env.state.terminated), truncated=bool(env.state.truncated))
    finally:
        env.close()


def compile_bot(bot_dir='/bot'):
    proc = subprocess.run(['cargo', 'build', '--manifest-path', str(Path(bot_dir)/'Cargo.toml'),
                           '--release', '--offline', '--locked', '--target', 'wasm32-wasip1'],
                          stdout=sys.stderr, stderr=sys.stderr, timeout=60)
    if proc.returncode:
        raise ValueError(f'compilation failed (exit {proc.returncode})')
    artifact = Path(bot_dir)/'target/wasm32-wasip1/release/skirmish-bot.wasm'
    return dict(wasm=str(artifact), sha256=sha(artifact.read_bytes()),
                note='Local compile check; the controller independently rebuilds frozen main.rs.')


def check_bot(wasm, observations, seat=1):
    """Replay public positions through ONE persistent process, not simulated games."""
    from polytopia_bot.environment import LegalAction, Observation
    from .wasi import WasiBot
    rows = []
    for line in Path(observations).read_text().splitlines():
        raw = json.loads(line)
        value = raw.get('observation', raw)
        if value['player'] == seat and value['legal_actions']:
            value['legal_actions'] = [LegalAction(**a) for a in value['legal_actions']]
            rows.append(Observation(**value))
    if not rows or len(rows) > 10000:
        raise ValueError('expected 1..10000 public observations for the selected seat')
    fuel = int(os.environ.get('POLYTOPIA_SKIRMISH_CHECK_FUEL', '1000000000000'))
    if fuel <= 0:
        raise ValueError('protocol-check fuel must be positive')
    bot = WasiBot(wasm, fuel=fuel, seconds=2)
    try:
        outputs = [dict(state_version=o.state_version, action_id=bot.select(o)) for o in rows]
        return dict(passed=True, decisions=len(outputs), outputs=outputs,
                    runtime_limits=dict(fuel=fuel, move_seconds=2),
                    note='Protocol/legality check on recorded positions, NOT a counterfactual match or win rate.')
    finally:
        bot.close()


def build_examples(destination):
    """Image-build fixture generation only: fixed maps and a scripted wire example."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for map_id in PUBLIC_MAPS:
        (destination/f'{map_id}.json').write_text(canonical(inspect_map(map_id))+'\n')
    env = environment('drylands-0')
    try:
        lines = []
        # A worked three-action protocol example, not a strategy or callable opponent:
        # seat 1 ends, seat 2 ends, seat 1 resigns. This reaches a real terminal state.
        for kind in (10, 10, 11):
            obs = env.observe(int(env.state.current_player))
            action = next(a for a in obs.legal_actions if a.command['kind'] == kind)
            result = env.step(action_id=action.action_id, expected_state_version=obs.state_version)
            lines.append(canonical(dict(observation=asdict(obs), action_id=action.action_id,
                annotation={10:'End turn: income and player transition.',
                            11:'Resign: demonstrate a terminal result, not strategic play.'}[kind],
                transition=dict(next_player=result.next_player, terminated=result.terminated,
                                truncated=result.truncated, reason=result.reason))))
        (destination/'worked-match.jsonl').write_text('\n'.join(lines)+'\n')
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('compile', help='Offline Rust-to-WASI compile check.')
    commands.add_parser('maps', help='List the nine supplied public maps.')
    view = commands.add_parser('inspect', help='Full public map layout and initial bot observation.')
    view.add_argument('--map', choices=PUBLIC_MAPS, required=True)
    step = commands.add_parser('replay', help='Execute up to 32 explicitly supplied action IDs; no automatic opponent.')
    step.add_argument('--map', choices=PUBLIC_MAPS, required=True)
    step.add_argument('--actions', type=Path, required=True, help='JSON array of action-ID strings.')
    check = commands.add_parser('check', help='Protocol-check your WASI bot on recorded public positions.')
    check.add_argument('--wasm', default='/bot/target/wasm32-wasip1/release/skirmish-bot.wasm')
    check.add_argument('--observations', type=Path, default=DEFAULT_EXAMPLES/'worked-match.jsonl')
    check.add_argument('--seat', type=int, choices=(1, 2), default=1)
    if os.environ.get('POLYTOPIA_SKIRMISH_CONTINUOUS') == '1':
        practice = commands.add_parser('practice', help='Compile and play a public practice match; charged to coding time.')
        practice.add_argument('--map', choices=PUBLIC_MAPS, default='drylands-0')
        practice.add_argument('--seat', type=int, choices=(1, 2), default=1)
        trace = commands.add_parser('trace', help='Inspect your practice observations and selected actions.')
        trace.add_argument('--path', type=Path, required=True)
        trace.add_argument('--offset', type=int, default=0)
        trace.add_argument('--limit', type=int, default=1)
    commands.add_parser('example', help='Print the annotated fixed protocol example; no opponent is executed.')
    args = parser.parse_args(argv)
    try:
        if args.command == 'compile': out = compile_bot()
        elif args.command == 'maps': out = [dict(map_id=k, map_type=v[0], seed=v[1]) for k,v in PUBLIC_MAPS.items()]
        elif args.command == 'inspect': out = inspect_map(args.map)
        elif args.command == 'replay': out = replay(args.map, json.loads(args.actions.read_text()))
        elif args.command == 'check': out = check_bot(args.wasm, args.observations, args.seat)
        elif args.command == 'practice':
            from .practice import practice_match
            out = practice_match(args.map, args.seat)
        elif args.command == 'trace':
            from .practice import inspect_trace
            out = inspect_trace(args.path, args.offset, args.limit)
        else:
            print((DEFAULT_EXAMPLES/'worked-match.jsonl').read_text(), end='')
            return
        print(canonical(out))
    except (BotFault, ValueError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f'skirmish: {error}\n')


if __name__ == '__main__':
    main()
