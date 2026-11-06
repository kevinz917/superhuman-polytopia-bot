"""Tests for control.decide (run on Modal: modal run --env staging experiments/verda/modal_controller.py::selftest)."""
from control import actor_hostname, decide, setup_done

H = 3600.


def base(**kw):
    s = dict(run='r', status='armed', learner='r-learner-0', armed_at=0., learner_started=0., deadline=10 * H,
             cap_usd=100., start_balance=500., balance=500., hourly={'r-learner-0': 10.}, hourly_default=1.,
             desired_actors=2, spot=True)
    return {**s, **kw}


OK = dict(active=True, complete=False, log_age=10.)


def vms_for(state, status='running'):
    v = {state['learner']: dict(status='running', ip='1.1.1.1')}
    for a in state.get('actors', {}).values():
        v[a['hostname']] = dict(status=status, ip='2.2.2.2')
    return v


def test_first_tick_creates_every_actor_and_syncs():
    s, acts = decide(base(), {'r-learner-0': dict(status='running', ip='1.1.1.1')}, OK, 60.)
    assert ('sync',) in acts
    assert [a for a in acts if a[0] == 'create'] == [('create', '0', 'r-actor-0-g0'), ('create', '1', 'r-actor-1-g0')]
    assert s['actors']['0']['phase'] == 'requested'


def test_running_actor_gets_set_up_once_then_runs():
    s, _ = decide(base(), {'r-learner-0': dict(status='running', ip='1.1.1.1')}, OK, 60.)
    s, acts = decide(s, vms_for(s), OK, 240.)
    assert ('setup', '0', 'r-actor-0-g0') in acts and s['actors']['0']['phase'] == 'provisioning'
    s = setup_done(s, '0', 'r-actor-0-g0', True, 300.)
    s, acts = decide(s, vms_for(s), OK, 420.)
    assert not [a for a in acts if a[0] in ('setup', 'create', 'delete') and a[1] == '0']
    assert s['actors']['0']['phase'] == 'running'


def test_evicted_spot_actor_is_deleted_and_replaced_with_a_new_generation():
    s, _ = decide(base(desired_actors=1), {'r-learner-0': dict(status='running', ip='1.1.1.1')}, OK, 60.)
    s, _ = decide(s, vms_for(s), OK, 240.)
    s = setup_done(s, '0', 'r-actor-0-g0', True, 300.)
    v = vms_for(s, status='discontinued')
    s, acts = decide(s, v, OK, 600.)
    assert ('delete', 'r-actor-0-g0') in acts and ('create', '0', actor_hostname('r', '0', 1)) in acts
    assert s['replaced'][0]['status'] == 'discontinued'


def test_vanished_and_stuck_actors_are_replaced():
    s, _ = decide(base(desired_actors=1), {'r-learner-0': dict(status='running', ip='1.1.1.1')}, OK, 60.)
    lone = {'r-learner-0': dict(status='running', ip='1.1.1.1')}
    s2, acts = decide(s, lone, OK, 120.)                  # requested, not listed yet: wait
    assert not [a for a in acts if a[0] == 'create']
    s2, acts = decide(s, lone, OK, 60. + 21 * 60)          # still not listed after the request timeout
    assert ('create', '0', 'r-actor-0-g1') in acts
    s3, _ = decide(s, vms_for(s), OK, 240.)                 # provisioning ...
    s3, acts = decide(s3, vms_for(s3), OK, 240. + 36 * 60)  # ... past its timeout
    assert ('delete', 'r-actor-0-g0') in acts and ('create', '0', 'r-actor-0-g1') in acts
    s4 = setup_done(decide(s, vms_for(s), OK, 240.)[0], '0', 'r-actor-0-g0', False, 300.)
    _, acts = decide(s4, vms_for(s4), OK, 360.)              # a failed setup is replaced right away
    assert ('create', '0', 'r-actor-0-g1') in acts


def test_stray_vms_of_the_run_are_deleted():
    s, _ = decide(base(desired_actors=1), {'r-learner-0': dict(status='running', ip='1.1.1.1')}, OK, 60.)
    v = dict(vms_for(s), **{'r-actor-0-g7': dict(status='running', ip='3.3.3.3'), 'other-vm': dict(status='running')})
    _, acts = decide(s, v, OK, 120.)
    assert ('delete', 'r-actor-0-g7') in acts and not [a for a in acts if a[0] == 'delete' and a[1] == 'other-vm']


def test_teardown_reasons():
    v = {'r-learner-0': dict(status='running', ip='1.1.1.1')}
    assert decide(base(), v, OK, 10 * H)[1] == [('teardown', 'deadline')]
    assert decide(base(balance=390.), v, OK, 60.)[1][0][0] == 'teardown'          # balance dropped past the cap
    assert decide(base(), {}, OK, 60.)[1] == [('teardown', 'learner VM missing')]
    assert decide(base(), {'r-learner-0': dict(status='offline')}, OK, 60.)[1] == [('teardown', 'learner VM offline')]
    assert decide(base(), v, dict(OK, complete=True), 60.)[1] == [('teardown', 'training complete')]
    stalled = dict(OK, log_age=25 * 60.)
    assert decide(base(), v, stalled, 30 * 60.)[1][0][0] != 'teardown'             # still in the startup grace
    assert decide(base(), v, stalled, 50 * 60.)[1][0][0] == 'teardown'
    no_log = dict(OK, log_age=None)
    assert decide(base(), v, no_log, 30 * 60.)[1][0][0] != 'teardown'
    assert decide(base(), v, no_log, 50 * 60.)[1][0][0] == 'teardown'


def test_learner_misses_accumulate_and_reset():
    v = {'r-learner-0': dict(status='running', ip='1.1.1.1')}
    s, acts = decide(base(), v, None, 60.)
    s, acts = decide(s, v, None, 240.)
    assert not [a for a in acts if a[0] == 'teardown'] and s['learner_misses'] == 2
    s2, _ = decide(s, v, OK, 420.)
    assert s2['learner_misses'] == 0
    _, acts = decide(s, v, dict(OK, active=False), 420.)
    assert acts == [('teardown', 'learner service down')]


def test_spend_estimate_accumulates_hourly_rates():
    v = {'r-learner-0': dict(status='running', ip='1.1.1.1'), 'r-actor-0-g0': dict(status='running')}
    s, _ = decide(base(last_tick=0.), v, OK, H)
    assert abs(s['spent_estimate_usd'] - 11.) < 1e-9                                  # learner 10 + unknown actor 1
    s, acts = decide(dict(s, cap_usd=20.), v, OK, H + H)
    assert acts[0][0] == 'teardown'


def test_disarmed_run_does_nothing():
    assert decide(base(status='stopped'), {}, None, 60.)[1] == []
