"""Iso-FLOP sweep plumbing (docs/isoflop-sweep-plan-2026-10-28.md): a run stops at training_updates and schedules its
LR over updates; snapshot_at saves the fork points; fork_run builds a folder whose Learner resumes bitwise at the
fork point and follows the schedule of a run of the fork's own length; the log carries cumulative FLOPs."""
import json
import threading

import jax
import numpy as np
import pytest

from polytopia_bot.rl import distributed, training_log
from polytopia_bot.rl.distributed import (Learner, LearnerConfig, _template_params, conv_parameters, fork_run,
                                          sample_flops, scheduled_learning_rate, training_progress)
from test_levers import _config, _feed, _free_port, _parent


def test_a_fork_from_three_quarters_follows_the_schedule_of_its_own_length():
    """A trunk of U3 updates and a standalone run of U1 share every LR up to 0.75 U1 (both constant), so a fork
    resumed there with training_updates = U1 runs exactly the standalone schedule."""
    trunk, short = LearnerConfig(training_updates=400), LearnerConfig(training_updates=40)
    lr = lambda cfg, v: scheduled_learning_rate(3e-4, 1e-4, .25, training_progress(cfg, v, 0.))
    assert all(lr(trunk, v) == lr(short, v) == 3e-4 for v in range(31))
    assert lr(short, 35) == pytest.approx(2e-4) and lr(short, 40) == 1e-4
    assert training_progress(LearnerConfig(training_seconds=100), 7, 25.) == .25   # unset: the wall clock
    with pytest.raises(ValueError):
        LearnerConfig(training_updates=-1).validate()


def test_conv_parameters_count_the_trunk_convolutions():
    params = _template_params(14, width=128, blocks=4, architecture='normalized-v2')
    stem = int(np.prod(params['stem'].shape))
    assert conv_parameters(params) == stem + 4 * 2 * 9 * 128 * 128
    assert sample_flops(14, 10, 5, 1) == 2 * 196 * 4 * 15


def _train(root, cfg, parent, monkeypatch, snaps):
    from polytopia_bot.rl.assembly_bench import synthetic_messages
    learner = Learner(root, cfg, parent, log=lambda s: None, port=_free_port())
    width = sum(int(np.prod(s)) for _, s in learner.layout)
    messages = synthetic_messages(np.random.default_rng(1), width, 196, 16, rows=(40, 90), omni=True)
    stop = threading.Event()
    threading.Thread(target=_feed, args=(learner, messages, stop), daemon=True).start()
    try:
        history = learner.run(bind_host='127.0.0.1')
    finally:
        stop.set()
    learner.server.close()
    return learner, json.loads(json.dumps(history))


def test_trunk_stops_at_its_budget_and_a_fork_resumes_bitwise(tmp_path, monkeypatch):
    monkeypatch.setattr(distributed, 'VALUE_CHUNK', 64)
    snaps, original = {}, Learner.snapshot

    def snapshot(self):
        snap = original(self)
        snaps[(str(self.root), snap['version'])] = snap
        return snap
    monkeypatch.setattr(Learner, 'snapshot', snapshot)
    parent = _parent(tmp_path)
    common = dict(checkpoint_every=3, snapshot_every=4, training_seconds=900, historical_every=4, historical_keep=2,
                  max_lag=10 ** 6, learning_rate=3e-4, learning_rate_end=1e-4, lr_decay_fraction=.25)
    trunk_root, fork_root = tmp_path / 'trunk', tmp_path / 'fork'
    trunk, rows = _train(trunk_root, _config(training_updates=16, snapshot_at=(9,), **common), parent, monkeypatch, snaps)
    assert [r['update'] for r in rows] == list(range(1, 17))   # stopped by the budget, not the clock
    assert (trunk_root / 'final.npz').exists() and (trunk_root / 'snapshots' / 'update-9-critic.npz').exists()
    assert (trunk_root / 'snapshots' / 'update-4.npz').exists()   # not a checkpoint_every update: saved anyway
    assert rows[12]['learning_rate'] == 3e-4 and rows[13]['learning_rate'] == pytest.approx(2.5e-4)   # versions 12, 13
    per_update = trunk.config.batch * sample_flops(14, conv_parameters(trunk.params), conv_parameters(trunk.critic), 1)
    assert [r['flops'] for r in rows] == pytest.approx([per_update * u for u in range(1, 17)])

    fork_run(trunk_root, fork_root, 9)
    with pytest.raises(FileExistsError):
        fork_run(trunk_root, fork_root, 9)
    with pytest.raises(FileNotFoundError):
        fork_run(trunk_root, tmp_path / 'other', 5)
    cfg = _config(training_updates=12, **common)
    again = Learner(fork_root, cfg, parent, log=lambda s: None)
    want = snaps[(str(trunk_root), 9)]
    assert again.version == 9 and again.resume_history == rows[:9]
    for key, now in (('params', again.params), ('opt', again.opt), ('critic', again.critic)):
        for a, b in zip(jax.tree.leaves(want[key]), jax.tree.leaves(now), strict=True):
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), key
    assert [n for n, _ in again.historical] == ['hist-u4', 'hist-u8']   # the trunk's pool at update 9
    _, fork_rows = _train(fork_root, cfg, parent, monkeypatch, snaps)
    assert [r['update'] for r in fork_rows] == list(range(1, 13)) and fork_rows[:9] == rows[:9]
    assert fork_rows[9]['learning_rate'] == 3e-4 and fork_rows[10]['learning_rate'] == pytest.approx(3e-4 - 2e-4 / 3)
    assert fork_rows[-1]['flops'] == pytest.approx(per_update * 12)
    assert training_log.read(fork_root) == fork_rows and (fork_root / 'final.npz').exists()


def test_sweep_arms_fork_inside_their_trunks_constant_phase():
    """Every arm's config is a valid LearnerConfig; each fork point is 0.75 of the fork's budget, saved by its trunk
    (snapshot_at) while the trunk still runs at constant LR; the pool cadence has a snapshot at each entry; the arms of
    one budget train on the same compute to within the update rounding."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'daytona'))
    import sweep
    arms, plans = sweep.arms(), sweep.plans()
    assert len(arms) == 12
    for name, arm in arms.items():
        cfg = LearnerConfig(**plans[name]['run_config'])
        cfg.validate()
        assert cfg.training_updates == arm['updates'] and cfg.historical_every % cfg.snapshot_every == 0
        if arm['fork']:
            trunk, at = arm['fork']
            assert at == arm['updates'] * 3 // 4 and at in arms[trunk]['snapshot_at']
            assert at < .75 * arms[trunk]['updates']
    for budget in sweep.BUDGETS:
        flops = [sweep.conv(a['size']) * a['updates'] for a in arms.values() if a['budget'] == budget]
        assert max(flops) / min(flops) < 1.01
