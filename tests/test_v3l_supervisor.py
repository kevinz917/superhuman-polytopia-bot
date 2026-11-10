"""Session L tiered error policy: supervisor paths with a fake trainer (no GPU, no Modal)."""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / 'claude_v3l'))
from policy import PolicyState, classify_exit  # noqa: E402
from supervisor import JsonStore, SigTerm, Supervisor  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / 'v3l_fake_trainer.py')


def manifest(tmp_path, faults=(), updates=5, cap=100., rate=36., **kw):
    m = dict(schema='v3l-manifest-1', mode='qualification', run_id='t-run', stage=1, stage_cap_usd=cap,
             usd_per_hour=rate, stop_reserve_usd=0., startup_allowance_s=0., harness_script=FAKE,
             trainer_args=['--updates', str(updates)], faults=list(faults), stop_dir=str(tmp_path))
    m.update(kw)
    return m


def sup(tmp_path, m, launch='L1', **kw):
    run_dir = tmp_path / 'run'
    store = JsonStore(tmp_path / 'store.json')
    return Supervisor(m, launch, run_dir, store, poll_s=kw.pop('poll_s', .05), heartbeat_s=.05, **kw), store, run_dir


def kinds(run_dir):
    return [e['kind'] for e in PolicyState(run_dir, dict(run_id='t-run', stage=1)).events()]


def test_classify_exit_table():
    assert classify_exit(0, None, True) == ('complete', 'training_seconds reached')
    assert classify_exit(0, 'pause_request', False) == ('pause', 'pause_request')
    assert classify_exit(70, None, False)[0] == 'rollback'
    assert classify_exit(71, None, False) == ('pause', 'leakage')
    assert classify_exit(72, None, False) == ('pause', 'illegal_rate')
    assert classify_exit(0, None, False) == ('pause', 'unclassified_clean_exit')
    assert classify_exit(-9, None, False)[0] == 'resume'


def test_complete(tmp_path):
    s, store, run_dir = sup(tmp_path, manifest(tmp_path))
    assert s.run()['state'] == 'complete'
    assert store.get('t-run/status')['state'] == 'complete'
    assert kinds(run_dir) == ['stage_start', 'complete']


def test_one_nan_rolls_back_and_continues(tmp_path):
    s, _, run_dir = sup(tmp_path, manifest(tmp_path, faults=[dict(kind='nan', update=3, times=1)]))
    assert s.run()['state'] == 'complete'
    assert kinds(run_dir) == ['stage_start', 'nan_rollback', 'complete']
    ev = PolicyState(run_dir, dict(run_id='t-run', stage=1)).events()[1]
    assert ev['rolled_back_to'] == 'update-2' and ev['severity'] == 'INFO'


def test_three_nans_in_an_hour_pause(tmp_path):
    s, _, run_dir = sup(tmp_path, manifest(tmp_path, faults=[dict(kind='nan', update=3, times=9)]))
    out = s.run()
    assert out == dict(state='paused', reason='nan_repeat', nan_in_window=3, update=2)
    assert kinds(run_dir) == ['stage_start', 'nan_rollback', 'nan_rollback', 'paused']
    assert PolicyState(run_dir, dict(run_id='t-run', stage=1)).status()['reason'] == 'nan_repeat'


def test_old_nans_leave_the_window(tmp_path):
    m = manifest(tmp_path, faults=[dict(kind='nan', update=3, times=3)])
    st = PolicyState(tmp_path / 'run', m)
    for _ in range(2):
        st.record_nan(1)
    rows = [json.loads(x) for x in st.p('nan.jsonl').read_text().splitlines()]
    st.p('nan.jsonl').write_text(''.join(json.dumps(dict(r, t=r['t'] - 7200)) + '\n' for r in rows))
    s, _, run_dir = sup(tmp_path, m)
    assert s.run()['state'] == 'complete'          # the 3rd NaN is the only one in the last hour


def test_crashes_resume_up_to_three(tmp_path):
    s, _, run_dir = sup(tmp_path, manifest(tmp_path, faults=[dict(kind='crash', update=2, times=3)]))
    assert s.run()['state'] == 'complete'
    assert kinds(run_dir).count('trainer_crash_resume') == 3


def test_fourth_crash_pauses(tmp_path):
    s, _, run_dir = sup(tmp_path, manifest(tmp_path, faults=[dict(kind='crash', update=2, times=9)]))
    out = s.run()
    assert out['state'] == 'paused' and out['reason'] == 'resume_limit' and out['resumes'] == 4


def test_lost_container_counts_a_resume_and_the_fourth_pauses(tmp_path):
    m = manifest(tmp_path)
    st = PolicyState(tmp_path / 'run', m)
    for i in range(3):                     # three containers that started and vanished
        st.begin_attempt(f'L{i}')
    s, _, run_dir = sup(tmp_path, m, launch='L9')
    out = s.run()
    assert out['state'] == 'complete'      # attempts 0,1 lost were seen at starts 1,2? no: only the LAST is checked
    # each start checks only the previous attempt: here one lost attempt is seen -> one resume
    assert kinds(run_dir)[:2] == ['stage_start', 'container_lost_resume']


def test_resume_limit_across_container_losses(tmp_path):
    m = manifest(tmp_path, updates=50, faults=[dict(kind='slow', seconds=.02)])
    run_dir = tmp_path / 'run'
    st = PolicyState(run_dir, m)
    for i in range(3):
        st.begin_attempt(f'L{i}')
        st.count_resume('container_lost')
    s, _, _ = sup(tmp_path, m, launch='L9')
    out = s.run()
    assert out['state'] == 'paused' and out['reason'] == 'resume_limit' and out['resumes'] == 4


def test_pause_request_is_graceful(tmp_path):
    m = manifest(tmp_path, updates=500, faults=[dict(kind='slow', seconds=.02)])
    s, store, run_dir = sup(tmp_path, m)
    threading.Timer(.3, lambda: store.put('t-run/pause', dict(reason='test'))).start()
    out = s.run()
    assert out['state'] == 'paused' and out['reason'] == 'pause_request'
    assert list(run_dir.glob('paused-u*.json'))
    assert store.get('t-run/heartbeat')['stage'] == 1


def test_pause_flag_before_start(tmp_path):
    s, store, run_dir = sup(tmp_path, manifest(tmp_path))
    store.put('t-run/pause', dict(reason='x'))
    assert s.run()['reason'] == 'pause_request'


def test_stage_budget_stops(tmp_path):
    # $36/h -> 1 cent per second; cap 3 cents = 3 s of container time
    m = manifest(tmp_path, updates=10_000, cap=.03, faults=[dict(kind='slow', seconds=.01)])
    s, _, run_dir = sup(tmp_path, m)
    t = time.time()
    out = s.run()
    assert out['state'] == 'paused' and out['reason'] == 'stage_budget'
    assert 2.5 < time.time() - t < 6


def test_budget_counts_earlier_attempts(tmp_path):
    m = manifest(tmp_path, cap=.03)
    st = PolicyState(tmp_path / 'run', m)
    a, _ = st.begin_attempt('L0')
    rows = st.p('attempts.jsonl').read_text().splitlines()
    first = json.loads(rows[0])
    first['t'] -= 10                          # an attempt that ran 10 s = 10 cents
    st.p('attempts.jsonl').write_text(json.dumps(first) + '\n')
    st.end_attempt(a, 'paused:pause_request')
    s, _, _ = sup(tmp_path, m, launch='L1')
    assert s.run()['reason'] == 'stage_budget'


@pytest.mark.parametrize('kind,reason,severity', [('leak', 'leakage', 'ALERT'), ('illegal', 'illegal_rate', 'PAUSE')])
def test_leak_and_illegal_pause(tmp_path, kind, reason, severity):
    s, _, run_dir = sup(tmp_path, manifest(tmp_path, faults=[dict(kind=kind, update=2)]))
    out = s.run()
    assert out['state'] == 'paused' and out['reason'] == reason
    ev = PolicyState(run_dir, dict(run_id='t-run', stage=1)).events()[-1]
    assert ev['kind'] == 'paused' and ev['severity'] == severity


def test_duplicate_start_of_a_finished_launch_does_not_train(tmp_path):
    s, _, run_dir = sup(tmp_path, manifest(tmp_path))
    s.run()
    s2, _, _ = sup(tmp_path, manifest(tmp_path))
    assert s2.run()['reason'] == 'duplicate_start_of_finished_launch'


def test_source_mismatch_pauses(tmp_path, monkeypatch):
    monkeypatch.setenv('V3L_SOURCE_SHA', 'aaa')
    s, _, _ = sup(tmp_path, manifest(tmp_path, source_sha='bbb'))
    assert s.run()['reason'] == 'config_mismatch'


def test_sigterm_leaves_the_attempt_open(tmp_path):
    m = manifest(tmp_path, updates=500, faults=[dict(kind='slow', seconds=.02)])
    s, _, run_dir = sup(tmp_path, m)
    threading.Timer(.3, lambda: s._on_sigterm(15, None)).start()
    with pytest.raises(SigTerm):
        s.run()
    starts, ends = PolicyState(run_dir, m).stage_attempts()
    assert len(starts) == 1 and not ends      # the next start counts a container_lost resume
    s2, _, _ = sup(tmp_path, dict(m, faults=[]), launch='L1')
    s2.run()
    assert 'container_lost_resume' in kinds(run_dir)


def test_dollar_snapshots_are_requested_once_each(tmp_path):
    # $36/h = 1 cent/s; marks at 1 and 2 cents of cumulative campaign spend (0.5 cent spent before this stage)
    m = manifest(tmp_path, updates=10_000, cap=.04, faults=[dict(kind='slow', seconds=.01)],
                 campaign_spent_before_usd=.005, snapshot_at_usd=[.01, .02, 99])
    s, _, run_dir = sup(tmp_path, m)
    assert s.run()['reason'] == 'stage_budget'
    st = PolicyState(run_dir, m)
    assert [r['label'] for r in st.usd_snapshots()] == ['usd-0.01', 'usd-0.02']
    assert kinds(run_dir).count('dollar_snapshot') == 2


# ---------------------------------------------------------------- one process per GPU (session A's layout)
def test_combine_rcs():
    from supervisor import combine_rcs
    assert combine_rcs([0, 0, 0]) == 0
    assert combine_rcs([70, -9, 70]) == 70
    assert combine_rcs([71, -9, -9]) == 71
    assert combine_rcs([-9, 1, -9]) == 1
    assert combine_rcs([0, -9, -9]) == 0          # rank 0 stopped cleanly, stragglers killed after the grace period
    assert combine_rcs([0, -6, 0]) == 0           # shutdown-barrier abort after rank 0 finished (Q3c)


def test_multiprocess_complete_and_rank_args(tmp_path):
    s, _, run_dir = sup(tmp_path, manifest(tmp_path, processes=4))
    cmds = None
    s.attempt = 0
    s.stop_file = str(tmp_path / 'x')
    cmds = s._trainer_cmd()
    assert len(cmds) == 4 and cmds[2][-6:-2] == ['--num-processes', '4', '--process-id', '2']
    assert json.loads(cmds[3][cmds[3].index('--harness-json') + 1])['rank'] == 3
    s2, _, run_dir = sup(tmp_path, manifest(tmp_path, processes=4), launch='L2')
    assert s2.run()['state'] == 'complete'


def test_multiprocess_rank_crash_resumes_the_group(tmp_path):
    m = manifest(tmp_path, processes=4, updates=40, rank_grace_s=1,
                 faults=[dict(kind='crash', rank=2, update=3, times=1), dict(kind='slow', seconds=.02)])
    s, _, run_dir = sup(tmp_path, m)
    assert s.run()['state'] == 'complete'
    assert kinds(run_dir).count('trainer_crash_resume') == 1


def test_multiprocess_pause_and_nan(tmp_path):
    m = manifest(tmp_path, processes=3, updates=500, faults=[dict(kind='slow', seconds=.02)])
    s, store, run_dir = sup(tmp_path, m)
    threading.Timer(.4, lambda: store.put('t-run/pause', dict(reason='test'))).start()
    assert s.run()['reason'] == 'pause_request'
    m2 = manifest(tmp_path / 'b', processes=3, faults=[dict(kind='nan', update=3, times=1)])
    (tmp_path / 'b').mkdir()
    s2, _, run_dir2 = sup(tmp_path / 'b', m2)
    assert s2.run()['state'] == 'complete' and 'nan_rollback' in kinds(run_dir2)


# ---------------------------------------------------------------- two nodes (Modal clustered 16 GPUs), local stand-in
import subprocess as _sp
GROUP = str(ROOT / 'experiments' / 'claude_v3l' / 'group.py')


def _agent(tmp_path, run_dir=None):
    args = [sys.executable, GROUP, str(tmp_path / 'store.json'), 't-run', '1'] + ([str(run_dir)] if run_dir else [])
    return _sp.Popen(args, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)


def _group_manifest(tmp_path, **kw):
    return manifest(tmp_path, processes=4, nodes=2, rank_grace_s=1, node_start_grace_s=2, node_stale_s=3, **kw)


def test_two_nodes_complete(tmp_path):
    ag = _agent(tmp_path)
    s, store, run_dir = sup(tmp_path, _group_manifest(tmp_path, updates=8))
    assert s.run()['state'] == 'complete'
    assert ag.wait(timeout=30) == 0 and 'exit' in ag.stdout.read()
    gen = store.get('t-run/grp/gen')
    assert gen['action'] == 'exit' and json.loads(gen['nodes']['1'][1][gen['nodes']['1'][1].index('--harness-json') + 1])['rank'] == 3


def test_two_nodes_remote_rank_crash_resumes_the_group(tmp_path):
    ag = _agent(tmp_path)
    m = _group_manifest(tmp_path, updates=30, faults=[dict(kind='crash', rank=3, update=3, times=1), dict(kind='slow', seconds=.03)])
    s, store, run_dir = sup(tmp_path, m)
    assert s.run()['state'] == 'complete'
    assert kinds(run_dir).count('trainer_crash_resume') == 1 and store.get('t-run/grp/gen')['gen'] == 2
    ag.wait(timeout=30)


def test_two_nodes_pause(tmp_path):
    ag = _agent(tmp_path)
    m = _group_manifest(tmp_path, updates=500, faults=[dict(kind='slow', seconds=.03)])
    s, store, run_dir = sup(tmp_path, m)
    threading.Timer(1.0, lambda: store.put('t-run/pause', dict(reason='test'))).start()
    assert s.run()['reason'] == 'pause_request'
    ag.wait(timeout=30)


def test_two_nodes_lost_node_fails_the_input(tmp_path):
    from supervisor import NodeLost
    ag = _agent(tmp_path)
    m = _group_manifest(tmp_path, updates=2000, faults=[dict(kind='slow', seconds=.03)])
    s, store, run_dir = sup(tmp_path, m)
    threading.Timer(2.5, ag.kill).start()          # node 1's container vanishes (its ranks keep running locally here)
    with pytest.raises(NodeLost):
        s.run()
    assert 'node_lost' in kinds(run_dir)


def test_two_nodes_stale_checkpoint_view_is_refused(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    (run_dir / 'latest.npz').write_bytes(b'old view')
    ag = _sp.Popen([sys.executable, '-c', f"""
import sys, group; sys.path.insert(0, {str(ROOT / 'experiments' / 'claude_v3l')!r})
from supervisor import JsonStore
import time; group.time.sleep = lambda s: None
print(group.run_agent(JsonStore({str(tmp_path / 'store.json')!r}), 't-run', 1, run_dir={str(run_dir)!r}, poll_s=.1, report_s=.3, idle_exit_s=60))
"""], cwd=str(ROOT / 'experiments' / 'claude_v3l'), stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
    m = _group_manifest(tmp_path, updates=5)
    s, store, _ = sup(tmp_path, m, prepare_gen=lambda: 'f' * 64)
    out = s.run()
    assert out['state'] == 'paused' and out['reason'] == 'resume_limit'       # refused 4 times -> pause
    ag.wait(timeout=30)


def test_lost_attempt_billing_is_frozen_at_its_last_heartbeat(tmp_path):
    m = manifest(tmp_path, cap=100., rate=3600.)          # $1 per second
    st = PolicyState(tmp_path / 'run', m)
    a0, _ = st.begin_attempt('L0')
    rows = [json.loads(x) for x in st.p('attempts.jsonl').read_text().splitlines()]
    rows[0]['t'] -= 500
    st.p('attempts.jsonl').write_text(json.dumps(rows[0]) + '\n')
    st.heartbeat(a0)                                       # last heartbeat of attempt 0: now (500 s after its start)
    a1, lost = st.begin_attempt('L1')
    assert lost
    st.heartbeat(a1)                                       # the new attempt overwrites heartbeat.json
    spent = st.stage_spent_usd(a1)
    assert 560 - 1 <= spent <= 560 + 5                     # 500 s + 60 s for the lost attempt, ~0 for the live one
