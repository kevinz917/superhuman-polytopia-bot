"""Real-engine checks run on Modal; no generated bots or research games run locally."""
import json
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get('MODAL_TASK_ID'), reason='engine executes on Modal')


def test_public_maps_match_referee_observations_and_remain_outside_feedback_partition():
    from dataclasses import asdict
    from polytopia_bot.skirmish.authoring_tools import PUBLIC_MAPS, environment
    from polytopia_bot.skirmish.common import canonical, PRACTICE_START
    from polytopia_bot.skirmish.game import reset_map
    from polytopia_bot.studio.environment import RustEnvironment
    assert len(PUBLIC_MAPS) == 9
    for name, (kind, seed) in PUBLIC_MAPS.items():
        assert seed < PRACTICE_START
        tool, referee = environment(name), RustEnvironment(14)
        try:
            reset_map(referee, seed, kind)
            for _ in range(3):
                for seat in (1, 2):
                    assert canonical(asdict(tool.observe(seat))) == canonical(asdict(referee.observe(seat)))
                obs = tool.observe(int(tool.state.current_player))
                action = next(a for a in obs.legal_actions if a.command['kind'] == 10)
                for env in (tool, referee):
                    env.step(action_id=action.action_id, expected_state_version=obs.state_version)
        finally:
            tool.close(); referee.close()


def test_replay_requires_explicit_legal_actions_and_does_not_choose_opponent_moves():
    from polytopia_bot.skirmish.authoring_tools import replay
    before = replay('lakes-0', [])
    obs = before['observation']
    action = next(a for a in obs['legal_actions'] if a['command']['kind'] == 10)
    after = replay('lakes-0', [action['action_id']])
    assert len(after['events']) == 1
    assert after['observation']['player'] == 2
    with pytest.raises(ValueError): replay('lakes-0', ['invalid'])
    with pytest.raises(ValueError): replay('lakes-0', ['x'] * 33)
    with pytest.raises(ValueError): replay('hidden-0', [])


def test_worked_example_replays_to_a_native_terminal_state(tmp_path):
    from polytopia_bot.skirmish.authoring_tools import build_examples, replay
    build_examples(tmp_path)
    rows = [json.loads(line) for line in (tmp_path/'worked-match.jsonl').read_text().splitlines()]
    result = replay('drylands-0', [r['action_id'] for r in rows])
    assert result['terminated'] and not result['truncated']
    assert rows[-1]['transition']['reason'] == 'resignation'
    assert len(list(tmp_path.glob('*.json'))) == 9


def test_protocol_check_preserves_one_bot_process_and_reports_guest_exit(tmp_path):
    import wasmtime
    from dataclasses import asdict
    from polytopia_bot.skirmish.authoring_tools import check_bot, environment
    from polytopia_bot.skirmish.wasi import BotFault
    env = environment('drylands-0')
    try:
        obs = asdict(env.observe(1))
    finally:
        env.close()
    action = obs['legal_actions'][0]['action_id']
    # This fixture emits one valid response, then exits on the next call. Restarting
    # per observation would incorrectly pass the two-position check.
    response = json.dumps({'action_id': action}, separators=(',', ':'))+'\n'
    data = ''.join(f'\\{b:02x}' for b in response.encode())
    wat = f'''(module
        (import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))
        (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
        (memory (export "memory") 1) (data (i32.const 16) "{data}")
        (func (export "_start")
          (i32.store (i32.const 0) (i32.const 128))
          (i32.store (i32.const 4) (i32.const 1))
          (block $done (loop $readline
            (drop (call $read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 8)))
            (br_if $done (i32.eq (i32.load8_u (i32.const 128)) (i32.const 10)))
            (br $readline)))
          (i32.store (i32.const 0) (i32.const 16))
          (i32.store (i32.const 4) (i32.const {len(response.encode())}))
          (drop (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))))'''
    wasm = tmp_path/'once.wasm'; wasm.write_bytes(wasmtime.wat2wasm(wat))
    trace = tmp_path/'obs.jsonl'; trace.write_text(json.dumps(obs)+'\n')
    assert check_bot(wasm, trace)['decisions'] == 1
    trace.write_text((json.dumps(obs)+'\n') * 2)
    with pytest.raises(BotFault): check_bot(wasm, trace)
