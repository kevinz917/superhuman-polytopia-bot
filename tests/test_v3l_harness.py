"""Session L harness pieces that run without a GPU: checkpoint repair, illegal-action game dropping, the leakage
check on a real (small) transformer-v3 network, launcher validation and the monitor's relay/backstop logic."""
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / 'claude_v3l'))
import flagship_harness as H  # noqa: E402


# ---------------------------------------------------------------- checkpoint repair
def _ckpt(dirpath, name, payload):
    dirpath.mkdir(parents=True, exist_ok=True)
    npz = dirpath / f'{name}.npz'
    np.savez(npz, a=np.array(payload))
    (dirpath / f'{name}.json').write_text(json.dumps(dict(sha256=H.sha256(npz))))


def _good_set(root, u):
    d = root / f'ckpt/update-{u}'
    _ckpt(d, 'latest', [u])
    _ckpt(d, 'latest-critic', [u, u])
    (root / 'GOOD.json').write_text(json.dumps(dict(stage=f'update-{u}', dir=f'ckpt/update-{u}')))
    return d


def test_repair_restores_torn_latest(tmp_path):
    d = _good_set(tmp_path, 50)
    for f in H.LATEST_FILES:
        (tmp_path / f).write_bytes((d / f).read_bytes())
    (tmp_path / 'latest.npz').write_bytes(b'torn')               # a container lost mid-copy
    notes = H.repair_run_dir(tmp_path, log=lambda s: None)
    assert notes == ['latest restored from ckpt/update-50']
    assert H._ckpt_ok(tmp_path / 'latest.npz') and H.sha256(tmp_path / 'latest.npz') == H.sha256(d / 'latest.npz')


def test_repair_rolls_latest_back_to_good(tmp_path):
    _good_set(tmp_path, 50)
    _ckpt(tmp_path, 'latest', [99])        # complete but newer than GOOD (GOOD.json never landed): GOOD wins
    _ckpt(tmp_path, 'latest-critic', [99])
    assert H.repair_run_dir(tmp_path, log=lambda s: None) == ['latest restored from ckpt/update-50']


def test_repair_leaves_consistent_latest_and_quarantines_torn_snapshots(tmp_path):
    d = _good_set(tmp_path, 50)
    for f in H.LATEST_FILES:
        (tmp_path / f).write_bytes((d / f).read_bytes())
    _ckpt(tmp_path / 'snapshots', 'update-250', [1])
    _ckpt(tmp_path / 'snapshots', 'update-500', [2])
    (tmp_path / 'snapshots' / 'update-500.npz').write_bytes(b'torn')
    notes = H.repair_run_dir(tmp_path, log=lambda s: None)
    assert notes == ['quarantined torn snapshot update-500.npz']
    assert (tmp_path / 'snapshots-quarantine' / 'update-500.npz').exists()
    assert (tmp_path / 'snapshots' / 'update-250.npz').exists()


# ---------------------------------------------------------------- illegal-action game dropping
@pytest.fixture()
def tainting(tmp_path):
    sys.path.insert(0, str(ROOT / 'src'))
    from polytopia_bot.rl import gpu_resident as gr
    orig = gr.SegmentLog, gr.GpuResidentTrainer
    H.install(dict(run_dir=str(tmp_path), run_id='t', stage=1))
    yield gr
    gr.SegmentLog, gr.GpuResidentTrainer = orig


def test_tainted_game_rows_never_reach_a_batch(tainting):
    log = tainting.SegmentLog(4, 2, 3, 100)                   # 4 games, 2 seats, segment 3
    seats = np.zeros(4, np.int32)
    games = np.arange(4)
    log.record(0, games, seats)
    log.record(1, games, seats)
    log.taint(np.array([2]))                                   # game 2 hit an illegal action after step 1
    assert log.dropped_rows == 2 and log.dropped_games == 1
    for t in (2, 3, 4):
        log.record(t, games, seats)                            # game 2's rows are not recorded
    assert int(log.count[2, 0]) == 0
    closed = np.concatenate([e['game'] for e in log.events])
    assert 2 not in closed.tolist() and sorted(set(closed.tolist())) == [0, 1, 3]
    # the game ends; the lane's next game is clean again
    log.finish(np.array([2]), np.zeros((1, 2), np.float32), np.array([True]), np.ones((1, 2), bool), np.array([-1]))
    assert not log.tainted[2]
    log.record(5, games, seats)
    assert int(log.count[2, 0]) == 1


def test_taint_then_finish_in_the_same_tick_closes_nothing(tainting):
    log = tainting.SegmentLog(2, 2, 8, 100)
    log.record(0, np.arange(2), np.zeros(2, np.int32))
    log.taint(np.array([0]))
    log.finish(np.array([0, 1]), np.ones((2, 2), np.float32), np.array([True, True]), np.ones((2, 2), bool),
               np.array([-1, -1]))
    closed = np.concatenate([e['game'] for e in log.events])
    assert closed.tolist() == [1]


# ---------------------------------------------------------------- leakage check on a real small V3 network
@pytest.mark.parametrize('architecture', ['transformer-v3-p2', 'transformer-v3-p2-h4'])
def test_leak_check_passes_live_params_and_catches_a_planted_leak(architecture):
    sys.path.insert(0, str(ROOT / 'src'))
    import jax
    from polytopia_bot.rl.distributed import _critic_template, _template_params
    from polytopia_bot.rl.observation import OBS_VERSION
    spec = dict(size=14, blocks=2, width=32, architecture=architecture, obs_version=OBS_VERSION)
    trainer = types.SimpleNamespace(network=spec, config=types.SimpleNamespace(players=2), version=7,
                                    params=jax.tree.map(lambda a: a + .05, _template_params(**spec)),
                                    critic=_critic_template(**spec))
    res = H.leak_check(trainer, samples=2)
    assert res['result'] == 'pass' and res['max_grad'] == 0. and res['max_logp_change'] == 0.
    with pytest.raises(H.LeakCheckFailed):
        H.leak_check(trainer, planted=True, samples=1)


def test_leak_check_not_applicable_for_the_cnn():
    trainer = types.SimpleNamespace(network=dict(size=14, architecture='normalized-v2'))
    assert H.leak_check(trainer)['result'] == 'not_applicable'


# ---------------------------------------------------------------- launcher validation
def _qual_manifest(**kw):
    m = dict(schema='v3l-manifest-1', mode='qualification', run_id='q0-harness', stage=1, stage_cap_usd=5,
             usd_per_hour=round(3.95 + 4 * 0.0473 + 32 * 0.008, 4), worker='qual_worker_1', timeout_s=2700,
             trainer_args=['--run-root', '/flagship/runs', '--run-id', 'q0-harness', '--parent', '',
                           '--architecture', 'transformer-v3-p2', '--width', '128', '--blocks', '6', '--seconds', '900'])
    m.update(kw)
    return m


def test_validate_accepts_a_good_qualification_manifest():
    import launcher
    assert launcher.validate(_qual_manifest(), check_source=False) == []


@pytest.mark.parametrize('change,problem', [
    (dict(trainer_args=['--run-root', '/flagship/runs', '--run-id', 'q0-harness', '--architecture', 'transformer-v3-p2',
                        '--width', '128', '--blocks', '4', '--seconds', '9']), 'no network smaller than 6x128'),
    (dict(stage_cap_usd=40), 'capped at $25'),
    (dict(worker='stage_worker'), 'is for flagship manifests'),
    (dict(usd_per_hour=3.0), 'list rate'),
])
def test_validate_rejects(change, problem):
    import launcher
    bad = launcher.validate(_qual_manifest(**change), check_source=False)
    assert any(problem in b for b in bad), bad


def test_flagship_rules():
    import launcher
    m = _qual_manifest(mode='flagship', worker='stage_worker', usd_per_hour=launcher.RATES[8], stage=1, timeout_s=86400,
                       stage_cap_usd=150, faults=[dict(kind='nan')])
    m['trainer_args'] += ['--devices', '8']
    bad = launcher.validate(m, check_source=False)
    assert any('cannot inject faults' in b for b in bad) and any('$100 boundary' in b for b in bad)


def test_flagship_launch_refused_without_go_file(tmp_path, monkeypatch):
    import launcher
    monkeypatch.setattr(launcher, 'FLAGSHIP_DIR', tmp_path)
    m = _qual_manifest(mode='flagship', worker='stage_worker', usd_per_hour=launcher.RATES[8], stage_cap_usd=100)
    ok, why = launcher.go_file_ok(m)
    assert not ok and 'explicit go' in why
    (tmp_path / 'GO-stage1.json').write_text(json.dumps(dict(manifest_sha256='0' * 64, jack_words='go', relayed_by='infra')))
    ok, why = launcher.go_file_ok(m)
    assert not ok and 'approves manifest' in why
    (tmp_path / 'GO-stage1.json').write_text(json.dumps(dict(manifest_sha256=launcher.manifest_sha(m), jack_words='"go"',
                                                             relayed_by='infra')))
    assert launcher.go_file_ok(m)[0]


def test_ledger_refuses_over_cap(tmp_path):
    import launcher
    p = tmp_path / 'ledger.json'
    p.write_text(json.dumps(dict(cap_usd=60, entries=[])))
    launcher.ledger_reserve(p, 'a', 50, '')
    with pytest.raises(SystemExit):
        launcher.ledger_reserve(p, 'b', 11, '')
    launcher.ledger_reconcile(p, 'a', 3.0, 'measured')
    launcher.ledger_reserve(p, 'b', 11, '')
    assert json.loads(p.read_text())['spent_usd'] == 14.0


# ---------------------------------------------------------------- monitor
class FakeDict(dict):
    def put(self, k, v):
        self[k] = v


def test_monitor_relays_once_and_backstops(tmp_path, monkeypatch):
    import launcher
    import monitor
    monkeypatch.setattr(launcher, 'FLAGSHIP_DIR', tmp_path)
    monkeypatch.setattr(monitor, 'MON_DIR', tmp_path / 'mon')
    sent = []
    monkeypatch.setattr(monitor, 'notify', lambda sev, subj, body, dry=False, mode=None: sent.append((sev, subj)))
    monkeypatch.setattr(monitor, 'poll_call', lambda cid: ('running', None))
    ledger = tmp_path / 'l.json'
    ledger.write_text(json.dumps(dict(cap_usd=60, entries=[dict(name='L1', reserved_usd=7)])))
    m = _qual_manifest()
    st = dict(run_id='q0-harness', manifest=m, relayed=[], active=True,
              launches=[dict(launch_id='L1', call_id='fc-x', t=time.time() - 60, ledger=str(ledger))])
    ctl = FakeDict()
    ctl['q0-harness/events'] = [dict(id='a', kind='nan_rollback', severity='INFO', t=time.time()),
                                dict(id='b', kind='paused', severity='PAUSE', reason='nan_repeat', t=time.time()),
                                dict(id='c', kind='heartbeat', severity='DEBUG', t=time.time())]
    ctl['q0-harness/heartbeat'] = dict(t=time.time(), spent_usd=1.0, update=10)
    st = monitor.one_run(st, ctl)
    assert sorted(s[0] for s in sent) == ['INFO', 'PAUSE']
    sent.clear()
    st = monitor.one_run(st, ctl)
    assert sent == []                                            # relayed once
    ctl['q0-harness/heartbeat'] = dict(t=time.time(), spent_usd=10.5, update=11)    # budget 5 + 5 exceeded
    st = monitor.one_run(st, ctl)
    assert ctl['q0-harness/pause']['by'] == 'monitor' and sent[0][0] == 'ALERT'
    sent.clear()
    monkeypatch.setattr(monitor, 'poll_call', lambda cid: ('done', dict(state='paused', reason='pause_request')))
    st = monitor.one_run(st, ctl)
    assert not st['active'] and sent[0][0] == 'PAUSE'
    assert json.loads(ledger.read_text())['entries'][0]['actual_usd'] == 10.5


def test_control_manifest_rules():
    import launcher
    m = _qual_manifest(mode='control', worker='stage_worker', usd_per_hour=launcher.RATES[8], timeout_s=86400,
                       stage_cap_usd=100, leak_check=False)
    m['trainer_args'] = ['--run-root', '/flagship/runs', '--run-id', 'q0-harness', '--parent', '', '--architecture',
                         'normalized-v2', '--width', '128', '--blocks', '4', '--seconds', '9', '--devices', '8']
    assert launcher.validate(m, check_source=False) == []
    m['stage_cap_usd'] = 150
    assert any('cap <= $100' in b for b in launcher.validate(m, check_source=False))


def test_stage_five_is_not_authorized():
    import launcher
    m = _qual_manifest(mode='flagship', worker='stage_worker', usd_per_hour=launcher.RATES[8], stage=5, timeout_s=86400,
                       stage_cap_usd=1500)
    m['trainer_args'] += ['--devices', '8']
    assert any('stage 5 is NOT authorized' in b for b in launcher.validate(m, check_source=False))
    m4 = dict(m, stage=4)
    assert not any('NOT authorized' in b for b in launcher.validate(m4, check_source=False))
    assert any('above the $1500' in b for b in launcher.validate(dict(m4, stage_cap_usd=1500.5), check_source=False))


def test_per_process_manifest_rules():
    import launcher
    m = _qual_manifest(worker='qual_worker_8', usd_per_hour=launcher.RATES[8], timeout_s=7200, processes=8)
    assert launcher.validate(m, check_source=False) == []
    m2 = dict(m, trainer_args=m['trainer_args'] + ['--devices', '1'])
    assert any('leave --devices' in b for b in launcher.validate(m2, check_source=False))


def test_lr_restart_schedule():
    lr = H.lr_restart_schedule(dict(start_update=23270, warmup_updates=150, from_lr=1e-4, peak_lr=3e-4, end_lr=1e-4,
                                    planned_updates=40000, decay_fraction=.25))
    assert lr(23270) == pytest.approx(1e-4 + 2e-4 / 150)
    assert lr(23270 + 149) == pytest.approx(3e-4) and lr(23270 + 20000) == pytest.approx(3e-4)
    assert lr(23270 + 35000) == pytest.approx(2e-4) and lr(23270 + 40000) == pytest.approx(1e-4)
    assert lr(23270 + 50000) == pytest.approx(1e-4)
    assert all(lr(u) >= lr(u + 1) - 1e-12 for u in range(23270 + 150, 23270 + 41000, 97))
