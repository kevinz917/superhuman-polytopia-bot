"""Decisions for the Verda run controller: pure functions of the run's state, the VM list and a learner probe.

modal_controller.py runs `decide` every few minutes on Modal and executes the actions, so a run keeps its deadline,
spend cap, checkpoint sync and spot-actor replacement with the Mac asleep or off. The state is the run's
controller.json on the Modal volume (written by bigrun.py arm); decide() returns an updated copy and the actions:

    ('teardown', reason)          final sync if the learner answers, then delete every VM and volume of the run
    ('sync',)                     copy new learner files to /results/distributed/<run>/
    ('create', slot, hostname)    request an actor VM (spot when state['spot'])
    ('setup', slot, hostname)     provision a running actor VM and start its service
    ('delete', hostname)          delete one VM with its volumes (and its learner firewall rule)
"""
import copy

# VM states that will not come back on their own: spot discontinue, failed deploys, deletion in progress
DEAD = frozenset({'offline', 'error', 'discontinued', 'no_capacity', 'installation_failed', 'notfound', 'deleting'})
DEFAULTS = dict(stall_minutes=20, startup_grace_minutes=45, request_timeout_minutes=20, provision_timeout_minutes=35,
                learner_miss_limit=3, sync_every_minutes=15)


def actor_hostname(run, slot, gen):
    return f'{run}-actor-{slot}-g{gen}'


def learner_hostname(run):
    return f'{run}-learner-0'


def spend(state, vms, now):
    """Spend so far: the larger of the account balance drop and the running estimate (every listed VM of the run
    billed at its hourly rate since the last tick). Returns (state, usd)."""
    state = dict(state)
    last = state.get('last_tick', now)
    hourly = sum(state['hourly'].get(h, state['hourly_default']) for h in vms)
    state['spent_estimate_usd'] = state.get('spent_estimate_usd', 0.) + hourly * max(now - last, 0.) / 3600
    state['last_tick'] = now
    by_balance = (state['start_balance'] - state['balance']) if state.get('balance') is not None else 0.
    return state, max(state['spent_estimate_usd'], by_balance)


def decide(state, vms, learner, now):
    """state: controller.json; vms: {hostname: dict(status=..., ip=...)} for this run's VMs; learner: None if the
    learner did not answer over SSH, else dict(active=bool, complete=bool, log_age=seconds or None)."""
    s = copy.deepcopy(state)
    for k, v in DEFAULTS.items():
        s.setdefault(k, v)
    if s.get('status') != 'armed':
        return s, []
    s, spent = spend(s, vms, now)
    s['spent_usd'] = spent
    minute = 60.
    if now >= s['deadline']:
        return s, [('teardown', 'deadline')]
    if spent >= s['cap_usd']:
        return s, [('teardown', f'spend cap ${s["cap_usd"]:.0f} reached (${spent:.2f})')]
    lvm = vms.get(s['learner'])
    if lvm is None or lvm.get('status') in DEAD:
        return s, [('teardown', f'learner VM {"missing" if lvm is None else lvm.get("status")}')]
    if learner is not None and learner.get('complete'):
        return s, [('teardown', 'training complete')]
    healthy = learner is not None and learner.get('active')
    s['learner_misses'] = 0 if healthy else s.get('learner_misses', 0) + 1
    if s['learner_misses'] >= s['learner_miss_limit']:
        return s, [('teardown', 'learner unreachable' if learner is None else 'learner service down')]
    started = s.get('learner_started', s['armed_at'])
    if healthy and now - started > s['startup_grace_minutes'] * minute:
        if learner.get('log_age') is None:
            return s, [('teardown', f'learner wrote no training row in {(now - started) / minute:.0f} min')]
        if learner['log_age'] > s['stall_minutes'] * minute:
            return s, [('teardown', f'learner stalled: no log row for {learner["log_age"] / minute:.0f} min')]

    actions = []
    if healthy and now - s.get('last_sync', -1e18) >= s['sync_every_minutes'] * minute:
        actions.append(('sync',))
    slots = s.setdefault('actors', {})
    wanted = set()
    for slot in map(str, range(s['desired_actors'])):
        a = slots.get(slot)
        if a is None:
            a = slots[slot] = dict(gen=0, hostname=actor_hostname(s['run'], slot, 0), phase='requested', since=now)
            actions.append(('create', slot, a['hostname']))
            wanted.add(a['hostname'])
            continue
        vm = vms.get(a['hostname'])
        if vm is None and a['phase'] == 'requested' and now - a['since'] < s['request_timeout_minutes'] * minute:
            pass                                               # just requested; not listed yet
        elif (vm is None or vm.get('status') in DEAD
              or (a['phase'] == 'requested' and now - a['since'] >= s['request_timeout_minutes'] * minute)
              or (a['phase'] == 'provisioning' and now - a['since'] >= s['provision_timeout_minutes'] * minute)):
            if vm is not None:
                actions.append(('delete', a['hostname']))
            gen = a['gen'] + 1
            s.setdefault('replaced', []).append(dict(slot=slot, hostname=a['hostname'], at=now, phase=a['phase'],
                                                     status=None if vm is None else vm.get('status')))
            a = slots[slot] = dict(gen=gen, hostname=actor_hostname(s['run'], slot, gen), phase='requested', since=now)
            actions.append(('create', slot, a['hostname']))
        elif a['phase'] == 'requested' and vm.get('status') == 'running' and vm.get('ip'):
            a.update(phase='provisioning', since=now, ip=vm['ip'])
            actions.append(('setup', slot, a['hostname']))
        wanted.add(a['hostname'])
    for slot in [k for k in slots if int(k) >= s['desired_actors']]:      # desired_actors lowered
        del slots[slot]
    prefix = f'{s["run"]}-actor-'
    deleting = {x[1] for x in actions if x[0] == 'delete'}
    for h in sorted(vms):
        if h.startswith(prefix) and h not in wanted and h not in deleting and vms[h].get('status') != 'deleting':
            actions.append(('delete', h))                     # an older generation, or a slot no longer wanted
    return s, actions


def setup_done(state, slot, hostname, ok, now):
    """Apply a provisioning result (the setup function's marker file) to the state."""
    s = copy.deepcopy(state)
    a = s.get('actors', {}).get(slot)
    if a and a['hostname'] == hostname and a['phase'] == 'provisioning':
        if ok:
            a.update(phase='running', since=now)
        else:
            a['since'] = -1e18                                # past the provisioning timeout: replaced next tick
    return s
