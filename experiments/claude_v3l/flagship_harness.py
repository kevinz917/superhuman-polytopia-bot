"""Session L flagship harness: experiments/gpu_resident_train.py's recipe mode with the tiered-policy hooks
(LAUNCH-PLAN.md section 2). The trainer itself is unchanged; a subclass is swapped into rl.gpu_resident at start.

    python flagship_harness.py --harness-json '{...}' -- <gpu_resident_train.py arguments>

Hooks:
  * stop file (supervisor: pause request, stage budget, SIGTERM): graceful stop after the current update; `latest`
    and a `paused-u<N>` copy are written; exit 0.
  * nonfinite PPO metrics (the trainer's own check, before the weights are adopted): wait for the background
    checkpoint writer, exit 70 (EXIT_NAN). The supervisor restarts from latest.npz = the last good checkpoint.
  * illegal engine actions (step status != 0): that game is dropped from training (its open rows are discarded and
    no rows are recorded until it ends); counted; above `illegal_pause_fraction` of decisions (after
    `illegal_min_decisions`) or above `illegal_abs_pause` total: graceful stop, exit 72.
  * leakage: at start and at every checkpoint, on the live parameters, the policy log-prob must not change and its
    gradient w.r.t. the other seat's view must be exactly zero when only that view changes (transformer-v3 family;
    other architectures have a separate critic network and report not_applicable). Failure or error of the check:
    graceful stop, exit 71.
  * atomic checkpoints: `latest` is written into ckpt/<stage>/ first, copied over latest.* with os.replace, then
    GOOD.json names it. Before the trainer starts, latest.* is repaired from GOOD.json and torn snapshots are
    quarantined, so a container lost mid-write resumes from the last complete checkpoint.
  * fault injection for qualification only (`faults`; the launcher refuses faults in flagship mode).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from policy import EXIT_ILLEGAL, EXIT_LEAK, EXIT_NAN, EXIT_PEACE, PolicyState, write_json_atomic  # noqa: E402

LATEST_FILES = ('latest.npz', 'latest.json', 'latest-critic.npz', 'latest-critic.json')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _ckpt_ok(npz):
    """The trainer's own integrity rule: the .json next to the .npz records the npz's sha256."""
    meta = npz.with_suffix('.json')
    if not npz.exists() or not meta.exists():
        return False
    try:
        return json.loads(meta.read_text()).get('sha256') == sha256(npz)
    except (json.JSONDecodeError, OSError):
        return False


def repair_run_dir(root, log=print):
    """Make latest.* equal the GOOD.json checkpoint; quarantine snapshots whose npz/json disagree. Returns notes."""
    root = Path(root)
    notes = []
    good = root / 'GOOD.json'
    if good.exists():
        g = json.loads(good.read_text())
        src = root / g['dir']
        if all(_ckpt_ok(src / f) for f in ('latest.npz', 'latest-critic.npz')):
            if not (_ckpt_ok(root / 'latest.npz') and _ckpt_ok(root / 'latest-critic.npz')
                    and sha256(root / 'latest.npz') == sha256(src / 'latest.npz')):
                for f in LATEST_FILES:
                    tmp = root / f'.{f}.repair'
                    shutil.copyfile(src / f, tmp)
                    os.replace(tmp, root / f)
                notes.append(f'latest restored from {g["dir"]}')
        else:
            notes.append(f'GOOD dir {g["dir"]} failed integrity; left latest.* as is')
    elif (root / 'latest.npz').exists() and not (_ckpt_ok(root / 'latest.npz') and _ckpt_ok(root / 'latest-critic.npz')):
        notes.append('latest.* torn and no GOOD.json')
    snaps = root / 'snapshots'
    if snaps.exists():
        for npz in sorted(snaps.glob('*.npz')):
            if not _ckpt_ok(npz):
                bad = root / 'snapshots-quarantine'
                bad.mkdir(exist_ok=True)
                for p in (npz, npz.with_suffix('.json')):
                    if p.exists():
                        os.replace(p, bad / p.name)
                notes.append(f'quarantined torn snapshot {npz.name}')
    for n in notes:
        log(f'[v3l] repair: {n}')
    return notes


class LeakCheckFailed(RuntimeError):
    pass


_LEAK_FNS = {}


def _leak_fn(arch, size, board, glob, n_fam, planted):
    """One jitted check per (architecture, shapes, planted): compiled once per process, then ~milliseconds per call.
    (Flagship 2026-11-07: the eager version stalled every checkpoint update by ~7.8 s on rank 0 while the other ranks
    waited at the keep_going vote: ~4-5% of the run.)"""
    key = (arch, size, board, glob, n_fam, planted)
    if key in _LEAK_FNS:
        return _LEAK_FNS[key]
    import jax
    import jax.numpy as jnp
    from polytopia_bot.rl import network, transformer_v3
    from polytopia_bot.rl.actions import Choice, PathMasks
    from polytopia_bot.rl.observation import Features
    tiles = size * size

    def feats(k):
        a, b = jax.random.split(k)
        return Features(jax.random.bernoulli(a, .2, (size, size, board)).astype(jnp.float32), jax.random.normal(b, (glob,)))

    def full(own, opp):
        return Features(jnp.concatenate((own.board, opp.board), -1), jnp.concatenate((own.global_state, opp.global_state)))

    def evaluate(p, c, f, choice, masks):
        logp, value, ent = transformer_v3.evaluate(p, c, f, choice, masks)
        if planted:      # qualification only: a policy that reads the privileged view must be caught
            logp = logp + 1e-3 * jnp.sum(f.board[..., board:])
        return logp, value, ent

    def one(params, critic, k):
        ks = jax.random.split(k, 8)
        own, opp_a, opp_b = feats(ks[0]), feats(ks[1]), feats(ks[2])
        masks = PathMasks(jax.random.bernoulli(ks[3], .5, (n_fam,)).at[0].set(True),
                          jax.random.bernoulli(ks[4], .5, (tiles,)).at[3].set(True),
                          jax.random.bernoulli(ks[5], .5, (max(tiles, 128),)).at[5].set(True))
        choice = Choice(jnp.int32(0), jnp.int32(3), jnp.int32(5))
        la = evaluate(params, critic, full(own, opp_a), choice, masks)[0]
        lb = evaluate(params, critic, full(own, opp_b), choice, masks)[0]

        def pol(ob, og):
            return evaluate(params, critic, full(own, Features(ob, og)), choice, masks)[0]
        g = jax.grad(pol, argnums=(0, 1))(opp_a.board, opp_a.global_state)
        gmax = jnp.maximum(jnp.abs(g[0]).max(), jnp.abs(g[1]).max())
        la_actor = network.evaluate(params, own, choice, masks)[0]
        return jnp.abs(la - lb), gmax, jnp.abs(la_actor - la)

    fn = jax.jit(one)
    _LEAK_FNS[key] = fn
    return fn


def leak_check(trainer, *, planted=False, samples=4):
    """Runtime invariance check on the trainer's live parameters (module docstring). Returns a result dict; raises
    LeakCheckFailed on a failure."""
    import jax
    import numpy as np
    from polytopia_bot.rl import transformer_v3
    from polytopia_bot.rl.distributed import feature_shapes
    spec = trainer.network
    if not transformer_v3.is_v3(spec['architecture']):
        return dict(result='not_applicable', architecture=spec['architecture'])
    size = int(spec['size'])
    board, glob = feature_shapes(trainer.config.players, spec.get('obs_version', 1))
    host = getattr(trainer, '_host', np.asarray)       # multi-process: this process's local replica
    dev = jax.local_devices()[0]
    params = jax.device_put(jax.tree.map(host, trainer.params), dev)
    critic = jax.device_put(jax.tree.map(host, trainer.critic), dev)
    n_fam = int(params['family']['w'].shape[-1]) if 'family' in params and 'w' in params['family'] else 17
    fn = _leak_fn(spec['architecture'], size, board, glob, n_fam, bool(planted))
    t0 = time.perf_counter()
    worst = worst_grad = actor_gap = 0.
    for i in range(samples):
        d, g, a = (float(x) for x in jax.device_get(fn(params, critic, jax.random.key(1000 + i))))
        worst, worst_grad, actor_gap = max(worst, d), max(worst_grad, g), max(actor_gap, a)
    out = dict(result='pass', max_logp_change=worst, max_grad=worst_grad, actor_path_gap=actor_gap,
               architecture=spec['architecture'], update=trainer.version, planted=planted,
               seconds=round(time.perf_counter() - t0, 3))
    if not (worst <= 1e-5 and worst_grad == 0. and actor_gap <= 1e-3):
        out['result'] = 'FAIL'
        raise LeakCheckFailed(json.dumps(out))
    return out


def lr_restart_schedule(r):
    """Stage-3 LR warm restart (Jack ~21:10 PT, LAUNCH-PLAN decision log): from `from_lr` (the annealed stage-2 end) up
    to the recipe peak over `warmup_updates`, then the recipe's own shape over the stage's `planned_updates`: hold the
    peak, then linear to the recipe end over the last `decay_fraction`; the end LR afterwards. Update-based (identical
    on every rank and across resumes)."""
    u0, w, l0 = int(r['start_update']), int(r['warmup_updates']), float(r['from_lr'])
    peak, end, n, d = float(r['peak_lr']), float(r['end_lr']), int(r['planned_updates']), float(r['decay_fraction'])

    def lr(version):
        k = int(version) - u0
        if k < 0:
            return l0
        if k < w:
            return l0 + (peak - l0) * (k + 1) / w
        frac = k / n
        if frac >= 1.:
            return end
        if frac < 1. - d:
            return peak
        return peak + (frac - (1. - d)) / d * (end - peak)
    return lr


def install(h):
    """Swap the hooked trainer and segment log into rl.gpu_resident. `h` = the harness settings dict."""
    import numpy as np
    from polytopia_bot.rl import gpu_resident as gr

    state = PolicyState(h['run_dir'], dict(run_id=h['run_id'], stage=h['stage']))
    faults = list(h.get('faults') or [])
    lock = threading.Lock()
    rank, nprocs = int(h.get('rank', 0)), int(h.get('processes', 1))
    lead = rank == 0      # several processes: rank 0 alone decides stop reasons and does all file I/O

    def fault(kind):
        # injected faults act on rank 0 unless they name a rank; 'nan' acts on every rank (pmean'd metrics)
        return [f for f in faults if f['kind'] == kind and (kind == 'nan' or int(f.get('rank', 0)) == rank)]

    def global_sum(values):
        if nprocs == 1:
            return np.asarray(values, np.int64)
        from jax.experimental import multihost_utils
        return multihost_utils.process_allgather(np.asarray(values, np.int64)).reshape(nprocs, -1).sum(0)

    class TaintingSegmentLog(gr.SegmentLog):
        def __init__(self, games, *a, **k):
            super().__init__(games, *a, **k)
            self.tainted = np.zeros(games, bool)
            self.dropped_rows = 0
            self.dropped_games = 0

        def taint(self, games):
            games = np.unique(np.asarray(games, np.int64))
            fresh = games[~self.tainted[games]]
            self.dropped_rows += int(self.count[games].sum())
            self.count[games] = 0
            self.first[games] = -1
            self.tainted[games] = True
            self.dropped_games += len(fresh)

        def record(self, t, games, seats0):
            keep = ~self.tainted[games]
            if keep.all():
                return super().record(t, games, seats0)
            return super().record(t, games[keep], seats0[keep])

        def finish(self, games, rewards, ended, seats_mask, store_idx):
            super().finish(games, rewards, ended, seats_mask, store_idx)
            self.tainted[games] = False

    s4 = h.get('stage4') or None          # stage 4 (Session L, 2026-11-08): peace curriculum + opponent pools
    S4_KEYS = ('peace_finished', 'peace_truncated', 'peace_selfplay_games', 'peace_selfplay_truncations',
               'peace_set_steps', 'peace_steps', 'diplomatic', 'late_steps')

    class FlagshipTrainer(gr.GpuResidentTrainer):
        instance = None
        ring_extra_steps = int((s4 or {}).get('ring_extra_steps', 0))
        pool_batched = bool((s4 or {}).get('pool_batched', False))
        pool_fresh_draw = bool(s4)

        def __init__(self, *a, **k):
            if s4 and 'block' in s4:
                k['block'] = int(s4['block'])
            self._v3l_peace_hist = []          # [(update, global counter vector)] for the collusion monitors
            self._v3l_peace_warned = {}
            super().__init__(*a, **k)
            FlagshipTrainer.instance = self
            if s4:
                self._v3l_terminal_rule()
            self._v3l_stop = None              # (reason, exit code)
            self._v3l_illegal_seen = 0
            self._v3l_injected_illegal = 0
            self._v3l_last_leak = None

        def _v3l_terminal_rule(self):
            """Stage 4 terminal reward at the 200-round cap, checked against the live config and logged (receipt).
            'cap_loss_both' (Jack 19:20/20:16 PT, segment 4b): a game reaching the cap without an elimination is a LOSS
            for BOTH seats = cap_score_reward 0 (no score-at-cap winner) + truncation_reward -1 (episode_outcome: both
            seats -1, terminal, no critic bootstrap). 'score_at_cap': cap_score_reward > 0 (+/- for the score leader)."""
            cfg = self.config
            rule = s4.get('terminal_rule', 'score_at_cap')
            probe, probe_ended = gr.episode_outcome(np.zeros(2, np.float32), False, True, cfg.truncation_reward) \
                if not cfg.cap_score_reward else (None, True)
            if rule == 'cap_loss_both':
                ok = (not cfg.cap_score_reward and cfg.truncation_reward is not None
                      and float(cfg.truncation_reward) == -1.0 and probe_ended and probe.tolist() == [-1.0, -1.0])
            elif rule == 'score_at_cap':
                ok = float(cfg.cap_score_reward or 0.) > 0
            else:
                ok = False
            if not ok:
                raise ValueError(f'stage4 terminal_rule {rule!r} does not match the config (cap_score_reward '
                                 f'{cfg.cap_score_reward}, truncation_reward {cfg.truncation_reward})')
            if lead:
                state.event('terminal_rule', rule=rule, cap_score_reward=float(cfg.cap_score_reward or 0.),
                            truncation_reward=cfg.truncation_reward, update=int(self.version),
                            capped_game_rewards=None if probe is None else probe.tolist())

        # -- stage 4: rules per game set, frozen opponents of any architecture, per-rule pools
        def frozen_any_arch(self):
            return bool(s4) or super().frozen_any_arch()

        def extra_opponents(self):
            return sorted((s4 or {}).get('opponents', {}).items()) if s4 else super().extra_opponents()

        def set_no_peace(self, g, k):
            if not s4:
                return super().set_no_peace(g, k)
            return int(k) not in {int(x) for x in s4.get('peace_sets', ())}

        def pool_spec(self):
            if not s4:
                return super().pool_spec()
            hist = [n for n, _ in self.historical]
            out = {}
            for kind, rule in ((0, 'no_peace'), (1, 'peace')):
                spec = [(n, float(p)) for n, p in sorted(s4['pool'].get(rule, {}).items()) if float(p) > 0]
                hp = float(s4.get('historical', {}).get(rule, 0.))
                if hist and hp > 0:
                    spec += [(n, hp / len(hist)) for n in hist]
                out[kind] = spec
            return out

        # -- stop
        def _v3l_request(self, reason, code):
            if self._v3l_stop is None:
                self._v3l_stop = (reason, code)
                self.log(json.dumps(dict(stage='v3l_stop', reason=reason, update=self.version)))
            self.config.training_updates = max(int(self.version), 1)      # the run loop's finished() turns true

        def _v3l_local(self, reason, code):
            """A stop only rank 0 knows of (the supervisor's stop file, a rank-0 leak check). One process: now. Several:
            pending until the next global sum (_v3l_illegal_check, every illegal_check_every updates), where every rank
            adopts it at the same update. (Pilot 10:43 PT: with session A's sparse stop vote, a rank-0-only stop let
            the other ranks run into the next collective; the shutdown barrier timed out after 5 min, rc -6.)"""
            if nprocs == 1:
                return self._v3l_request(reason, code)
            if getattr(self, '_v3l_pending', None) is None:
                self._v3l_pending = (reason, code)
                self.log(json.dumps(dict(stage='v3l_stop_pending', reason=reason, update=self.version)))

        def _v3l_poll_stop_file(self):
            sf = h.get('stop_file')
            if lead and sf and os.path.exists(sf):
                self._v3l_local(Path(sf).read_text().strip() or 'stop_file', 0)

        # -- leakage
        def _v3l_leak(self, when):
            if not h.get('leak_check', True) or not lead:
                return
            planted = any(self.version >= f.get('update', 0) for f in fault('leak'))
            try:
                res = leak_check(self, planted=planted)
            except Exception as exc:       # a failed or broken check cannot vouch for invariance: pause
                res = dict(result='FAIL', error=repr(exc)[:2000], update=self.version, when=when)
                state.event('leakage', **res)
                self._v3l_local('leakage', EXIT_LEAK)
                return
            self._v3l_last_leak = res
            state.event('leak_check_pass', when=when, **res)

        # -- illegal actions
        def _outcomes(self, sh, gs, t):
            gs.stage_event.synchronize()
            status = gs.stage_out['status']
            for f in fault('illegal'):
                if self.version >= f.get('update', 0):
                    with lock:
                        budget = int(f.get('total', 1)) - self._v3l_injected_illegal
                        k = min(int(f.get('lanes', 1)), budget)
                        if k > 0:
                            status[:k] = 3          # an invalid-action code; counted as nonok by the trainer too
                            self._v3l_injected_illegal += k
            bad = np.flatnonzero(status != 0)
            if len(bad):
                sh.seglog.taint(gs.games[bad])
                with lock:
                    self._v3l_illegal_seen += len(bad)
            return super()._outcomes(sh, gs, t)

        def _v3l_illegal_check(self):
            # global counts over every process, so every rank reaches the same decision at the same update. Every
            # `illegal_check_every` updates (default 10; decided by the update number, identical on every rank): each
            # check is one cross-process host allgather (16-GPU diagnosis 23:35 PT). Games are still dropped every tick.
            every = int(h.get('illegal_check_every', 10))
            if nprocs > 1 and self.version % every:
                return
            st = self._stats()
            peace = self.peace_stats(st) if (s4 and self.mixed_rules) else {}
            pend = getattr(self, '_v3l_pending', None)
            sums = global_sum([
                int(st['nonok']), int(st['steps']), sum(getattr(s.seglog, 'dropped_games', 0) for s in self.shards),
                sum(getattr(s.seglog, 'dropped_rows', 0) for s in self.shards), self._v3l_illegal_seen,
                0 if pend is None else 1, 0 if pend is None else int(pend[1]) + 1]
                + [peace.get(k, 0) for k in S4_KEYS])
            nonok, steps, dropped, drows, seen = (int(x) for x in sums[:5])
            if int(sums[5]) > 0:      # a rank-0 stop: every rank stops at this update
                reason, code = pend if pend is not None else ('rank0_stop', int(sums[6]) - 1)
                self._v3l_request(reason, code)
            if peace:
                self._v3l_peace_monitor(np.asarray(sums[7:], np.int64))
            if lead and seen and seen != getattr(self, '_v3l_illegal_reported', 0):
                self._v3l_illegal_reported = seen
                state.event('illegal_game_dropped', nonok_steps=nonok, decisions=steps, dropped_games=dropped,
                            dropped_rows=drows, update=self.version)
            frac = nonok / max(steps, 1)
            if (steps >= int(h.get('illegal_min_decisions', 10_000_000)) and frac > float(h.get('illegal_pause_fraction', 1e-4))) \
                    or nonok > int(h.get('illegal_abs_pause', 1000)):
                if lead:
                    state.event('illegal_rate', nonok_steps=nonok, decisions=steps, fraction=frac, update=self.version)
                self._v3l_request('illegal_rate', EXIT_ILLEGAL)

        def _v3l_peace_monitor(self, vec):
            """Stage-4 collusion monitors (STAGE4-DESIGN.md section 1) over global counters, every check; decided on
            identical numbers on every rank. WARN = an event (once per window); PAUSE = a graceful stop, exit 74."""
            mon = s4.get('monitor', {})
            hist = self._v3l_peace_hist
            hist.append((int(self.version), vec))
            keep = int(mon.get('pause_window', 300)) + 50
            while hist and hist[0][0] < self.version - keep:
                hist.pop(0)

            def window(n):
                old = [v for u, v in hist if u <= self.version - n]
                if not old:
                    return None
                d = dict(zip(S4_KEYS, (vec - old[-1]).tolist()))
                return dict(
                    games=d['peace_finished'],
                    capped=d['peace_truncated'] / max(d['peace_finished'], 1),
                    selfplay_games=d['peace_selfplay_games'],
                    selfplay_capped=d['peace_selfplay_truncations'] / max(d['peace_selfplay_games'], 1),
                    at_peace=d['peace_steps'] / max(d['peace_set_steps'], 1),
                    diplomatic_per_game=d['diplomatic'] / max(d['peace_finished'], 1),
                    late=d['late_steps'] / max(d['peace_set_steps'], 1))
            w = window(int(mon.get('window', 100)))
            if w is None:
                return
            if lead and self.version % int(mon.get('window', 100)) == 0:
                state.event('peace_monitor', update=self.version, **{k: round(v, 5) for k, v in w.items()})
            min_games = int(mon.get('min_games', 200))
            warns = []
            if w['selfplay_games'] >= min_games and w['selfplay_capped'] > float(mon.get('warn_selfplay_capped', .10)):
                warns.append('selfplay_capped')
            if w['games'] >= min_games and w['capped'] > float(mon.get('warn_capped', .10)):
                warns.append('capped')
            if w['at_peace'] > float(mon.get('warn_at_peace', .25)):
                warns.append('at_peace')
            if w['late'] > float(mon.get('warn_late', .10)):
                warns.append('late_games')
            for name in warns:
                last = self._v3l_peace_warned.get(name, -10 ** 9)
                if lead and self.version - last >= int(mon.get('window', 100)):
                    self._v3l_peace_warned[name] = self.version
                    state.event('peace_warn', monitor=name, update=self.version, **{k: round(v, 5) for k, v in w.items()})
            p = window(int(mon.get('pause_window', 300)))
            if p is None:
                return
            if (p['selfplay_games'] >= min_games and p['selfplay_capped'] > float(mon.get('pause_selfplay_capped', .30))) \
                    or p['at_peace'] > float(mon.get('pause_at_peace', .50)) \
                    or p['late'] > float(mon.get('pause_late', .30)):
                if lead:
                    state.event('peace_collapse', update=self.version, **{k: round(v, 5) for k, v in p.items()})
                self._v3l_request('peace_collapse', EXIT_PEACE)

        # -- snapshots at matched cumulative dollars (the supervisor's budget clock decides when)
        def _v3l_dollar_snapshot(self):
            sf = h.get('snap_file')
            if not lead or not sf or not os.path.exists(sf):
                return
            labels = [x for x in Path(sf).read_text().split() if x]
            done = {r['label'] for r in state.usd_snapshots()}
            st = self._host_state()
            for label in labels:
                if label in done:
                    continue

                def job(label=label, st=st):
                    self._write(f'snapshots/{label}', st)
                    self.commit()
                self.ckpt.submit(job)
                state.record_usd_snapshot(label, self.version)
                state.event('dollar_snapshot', label=label, update=self.version)
            os.remove(sf)

        # -- the update (NaN path) and checkpoints
        def _update(self, started):
            for f in fault('nan'):
                if self.version + 1 >= f.get('update', 1) and len([r for r in state.p('nan.jsonl').read_text().splitlines()]
                                                                  if state.p('nan.jsonl').exists() else []) < f.get('times', 1):
                    if f.get('attempt') in (None, h.get('attempt')):
                        raise RuntimeError('nonfinite PPO metrics (injected by v3l qualification)')
            for f in fault('crash'):
                if self.version + 1 >= f.get('update', 1) and len(state.resumes()) < f.get('times', 1):
                    self.log(json.dumps(dict(stage='v3l_injected_crash', update=self.version)))
                    os._exit(1)
            for f in fault('container_kill'):
                lost = [r for r in state.resumes() if r['why'] == 'container_lost']
                if self.version + 1 >= f.get('update', 1) and len(lost) < f.get('times', 1):
                    self.log(json.dumps(dict(stage='v3l_injected_container_kill', update=self.version)))
                    self.ckpt.wait()
                    import signal
                    os.kill(os.getppid(), signal.SIGKILL)
                    os._exit(1)
            for f in fault('slow'):
                time.sleep(float(f.get('seconds', 0)))
            info = super()._update(started)
            self._v3l_dollar_snapshot()
            self._v3l_illegal_check()
            self._v3l_poll_stop_file()
            return info

        def _checkpoint(self, rows, names):
            self._v3l_leak(f'checkpoint u{self.version}')      # rank 0 only
            return super()._checkpoint(rows, names)

        def _write(self, name, st):
            if not lead:
                return
            if name != 'latest':
                return super()._write(name, st)
            d = f'ckpt/{st["stage"]}'
            (self.root / d).mkdir(parents=True, exist_ok=True)
            super()._write(f'{d}/latest', st)
            for f in LATEST_FILES:
                tmp = self.root / f'.{f}.tmp'
                shutil.copyfile(self.root / d / f, tmp)
                os.replace(tmp, self.root / f)
            write_json_atomic(self.root / 'GOOD.json', dict(stage=st['stage'], dir=d, t=time.time(),
                                                            sha256=sha256(self.root / d / 'latest.npz')))
            keep = int(h.get('ckpt_keep', 3))
            dirs = sorted((p for p in (self.root / 'ckpt').iterdir() if p.is_dir() and p.name.startswith('update-')),
                          key=lambda p: int(p.name.rsplit('-', 1)[1]))
            for old in dirs[:-keep]:
                shutil.rmtree(old, ignore_errors=True)

        def _v3l_pool_parity(self):
            """Stage-4 qualification: the grouped pool call (act_group) against the per-opponent act_subset with the
            same keys, lanes and padding, on shard 0 / set 0's initial state; every pool member of every structure."""
            import jax
            sh, gs = self.shards[0], self.shards[0].gsets[0]
            if not getattr(sh, 'pool_stacks', None):
                self._group_pool(sh)
            Gs, rows, agree, total = self.Gs, [], 0, 0
            key = jax.random.key(1234)
            for gi, (stacked, K) in enumerate(sh.pool_stacks):
                names = [n for n, (g, _) in sh.pool_group_of.items() if g == gi]
                ka = 1 << (len(names) - 1).bit_length()
                B = max(32, min(128, Gs // max(ka, 1)))
                which = np.zeros(ka, np.int32)
                idx2 = np.full((ka, B), Gs, np.int32)
                for r, n in enumerate(names):
                    which[r] = sh.pool_group_of[n][1]
                    idx2[r] = np.arange(r * B, (r + 1) * B) % Gs
                key, k = jax.random.split(key)
                grouped = np.asarray(self._act_group(stacked, jax.numpy.asarray(which), gs.view_j, gs.masks_j,
                                                     jax.numpy.asarray(idx2), k)).reshape(ka, B, 3)
                keys = jax.random.split(k, ka)
                for r, n in enumerate(names):
                    ref = np.asarray(self._act_subset(sh.pool_params[n], gs.view_j, gs.masks_j, jax.numpy.asarray(idx2[r]), keys[r]))
                    same = int((ref == grouped[r]).all(1).sum())
                    agree, total = agree + same, total + B
                    rows.append(dict(name=n, group=gi, agree=same, rows=B))
            # the two group variants (lax.map vs vmap over members): agreement with the per-opponent path and the
            # time of one call at a stage-4-like shape (8 members x 64 lanes), compile excluded
            bench = {}
            for vname, fn in self._act_group_variants.items():
                try:
                    stacked, K = sh.pool_stacks[0]
                    which = jax.numpy.asarray(np.arange(8) % K, dtype=np.int32)
                    idx2 = jax.numpy.asarray(np.arange(8 * 64).reshape(8, 64) % Gs, dtype=np.int32)
                    ref = np.asarray(self._act_group_variants['map'](stacked, which, gs.view_j, gs.masks_j, idx2, key))
                    out = np.asarray(fn(stacked, which, gs.view_j, gs.masks_j, idx2, key))
                    t0 = time.perf_counter()
                    for _ in range(20):
                        o = fn(stacked, which, gs.view_j, gs.masks_j, idx2, key)
                    jax.block_until_ready(o)
                    bench[vname] = dict(ms=round((time.perf_counter() - t0) / 20 * 1000, 2),
                                        agree_with_map=float((ref == out).all(1).mean()))
                except Exception as exc:
                    bench[vname] = dict(error=repr(exc)[:300])
            res = dict(agree=agree, rows=total, fraction=agree / max(total, 1), members=rows, bench=bench)
            self.log(json.dumps(dict(stage='v3l_pool_parity', **res)))
            if lead:
                state.event('pool_parity', **{k: v for k, v in res.items() if k != 'members'}, members=rows)
            return res

        def run(self):
            if s4 and s4.get('parity_check'):
                self._v3l_pool_parity()
            self._v3l_leak('start')
            if self._v3l_stop and self._v3l_stop[1] == EXIT_LEAK:
                raise LeakCheckFailed('leak check failed before training')
            return super().run()

    gr.SegmentLog = TaintingSegmentLog
    gr.GpuResidentTrainer = FlagshipTrainer
    if h.get('lr_restart'):
        lr = lr_restart_schedule(h['lr_restart'])
        # replaces the elapsed-based schedule for this stage; update-based, so every rank computes the same LR
        gr.warmed_learning_rate = lambda rate, version, warmup_updates: lr(version)
    return FlagshipTrainer, state


def finish_stop(trainer, root):
    """After a graceful stop: final.* becomes paused-u<N>.* and training-complete.json is removed (the run is not
    complete)."""
    root = Path(root)
    n = trainer.version
    for suffix in ('.npz', '.json', '-critic.npz', '-critic.json'):
        src = root / f'final{suffix}'
        if src.exists():
            os.replace(src, root / f'paused-u{n}{suffix}')
    done = root / 'training-complete.json'
    if done.exists():
        os.replace(done, root / f'paused-u{n}-stop.json')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' not in argv:
        sys.exit('usage: flagship_harness.py --harness-json JSON -- <trainer args>')
    cut = argv.index('--')
    ap = argparse.ArgumentParser()
    ap.add_argument('--harness-json', required=True)
    h = json.loads(ap.parse_args(argv[:cut]).harness_json)
    trainer_args = argv[cut + 1:]
    if int(h.get('rank', 0)) == 0:      # ranks > 0 block in jax.distributed.initialize until rank 0 joins after this
        repair_run_dir(h['run_dir'], log=lambda s: print(s, flush=True))
    import gpu_resident_train   # experiments/ on sys.path (PYTHONPATH=/workspace:/workspace/experiments)
    cls, state = install(h)
    code = 0
    try:
        gpu_resident_train.main(trainer_args)
    except LeakCheckFailed as exc:
        print(f'[v3l] leakage: {exc}', flush=True)
        code = EXIT_LEAK
    except RuntimeError as exc:
        if 'nonfinite' in str(exc):
            t = cls.instance
            if t is not None and getattr(t, 'ckpt', None) is not None:
                try:
                    t.ckpt.wait()            # the last good checkpoint finishes landing; the bad weights never do
                except Exception:
                    pass
            print(f'[v3l] nonfinite update at version {getattr(t, "version", None)}: {exc}', flush=True)
            sys.stdout.flush()
            os._exit(EXIT_NAN)
        raise
    t = cls.instance
    if t is not None and t._v3l_stop:
        if int(h.get('rank', 0)) == 0:
            finish_stop(t, h['run_dir'])
        code = code or t._v3l_stop[1]
    sys.stdout.flush()
    sys.exit(code)


if __name__ == '__main__':
    main()
