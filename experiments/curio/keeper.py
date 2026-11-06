"""Run keeper on the Curio head: keeps the fleet alive and drives the phases of a hero run without the Mac.

    python keeper.py --root ROOT --run RUN            # started by curio.py launch; reads ROOT/runs/RUN/plan.json

Every 60 s:
- learner: restart a dead learner (it resumes ROOT/runs/<run>/latest.npz); a learner whose status has not moved for
  10 min is dumped (SIGUSR1 -> faulthandler in its log) and restarted; five updates in a row with approx_kl > .3
  count as diverged and restart it too (the resume point is the last checkpoint).
- actors: restart dead actor nodes on the head and the workers (a node younger than 5 min is left alone).
- ladder: every eval.interval s freeze latest.npz to ROOT/ladder/<run>-update-N.npz, copy it to the eval host and
  play it against every reference there (arena.py match, detached, one job per reference); rows join
  ROOT/ladder/<run>-history.json in the Modal ladder's row format, so scripts/ladder_elo.py reads it.
- phases (plan['auto']): train -> [final ladder of final.npz, 64 maps] + foghead (fresh fleet, fog-head config,
  parent = final.npz) -> head_check + search arena (hero+search vs champion fog head+search) -> summary.md -> done.
State lives in ROOT/runs/<run>/keeper-state.json; the keeper is restartable at any point. Logs: ROOT/logs/keeper.log.
A STOP file in the run directory ends the current learner gracefully (final.npz) and the chain continues; a
STOP-ALL file in ROOT/runs/<run>/ tears the fleet down and exits.
"""
import argparse
import json
import shlex
import sys
import time
import traceback
from pathlib import Path

from common import Host, hosts_from, paths, learner_command, actor_command, LEARNER_ENV, ACTOR_ENV, parse_cores

LADDER_KEYS = ('wins', 'losses', 'draws_or_truncations', 'truncations', 'win_rate', 'ci95', 'candidate_unit_mix',
               'candidate_non_warrior_games', 'candidate_unit_types_per_game', 'candidate_first_non_warrior_turn_median',
               'by_map', 'map_mix')
STALL_SECONDS = 600
ACTOR_SETTLE = 300
ACTOR_STALL_SECONDS = 600
MAX_LEARNER_RESTARTS = 24


class Keeper:
    def __init__(self, root, run):
        self.root, self.run = Path(root), run
        self.run_dir = self.root / 'runs' / run
        self.plan = json.loads((self.run_dir / 'plan.json').read_text())
        self.cluster = self.plan['cluster']
        self.hosts = hosts_from(self.cluster, from_head=True)
        self.head = self.hosts['head']
        self.p = paths(self.cluster['head']['root'])
        self.state_path = self.run_dir / 'keeper-state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else dict(
            phase='train', learner_restarts=0, started={}, progress={}, ladder_round=0,
            next_ladder=self.plan['started_at'] + self.plan['eval']['interval'], jobs={}, events=[])
        self.log_path = self.root / 'logs' / 'keeper.log'

    # ----- bookkeeping -------------------------------------------------------------------------------------------
    def log(self, **row):
        line = json.dumps(dict(row, at=time.time(), phase=self.state['phase']))
        print(line, flush=True)
        self.state['events'] = (self.state.get('events', []) + [line])[-200:]

    def save(self):
        tmp = self.state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def current_run(self):
        return self.run if self.state['phase'] != 'foghead' else f'{self.run}-foghead'

    def run_dir_of(self, run):
        return self.root / 'runs' / run

    # ----- learner -----------------------------------------------------------------------------------------------
    def learner_cmd(self, run):
        if run == self.run:
            return learner_command(self.cluster, run, self.plan['config'], self.plan['parent'], self.plan['critic'],
                                   self.plan['hours'])
        fh = self.plan['foghead']
        return learner_command(self.cluster, run, fh['config'], str(self.run_dir / 'final.npz'), '', fh['hours'])

    def start_learner(self, run):
        env = dict(LEARNER_ENV, CUDA_VISIBLE_DEVICES=','.join(map(str, self.cluster['learner']['gpus'])),
                   JAX_COMPILATION_CACHE_DIR=self.p['cache'], PYTHONPATH=self.p['src'])
        out = self.head.start('learner', self.learner_cmd(run), env, self.p['code'], f'{self.p["logs"]}/learner-{run}.log',
                              f'{self.p["pids"]}/learner.pid')
        self.state['started']['learner'] = time.time()
        self.state['progress'] = {}
        self.log(event='learner_start', run=run, result=out)

    def stop_learner(self, reason):
        pid = self.head.read(f'{self.p["pids"]}/learner.pid')
        if pid:
            self.head.run(f'kill -USR1 {pid} 2>/dev/null; sleep 3; true', check=False)   # thread dump into its log
        self.head.stop(f'{self.p["pids"]}/learner.pid', grace=15)
        self.log(event='learner_stop', reason=reason)

    def learner_status(self, run):
        path = self.run_dir_of(run) / 'learner-status.json'
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def check_learner(self, run):
        """Returns 'done', 'alive', or 'restarted'."""
        rd = self.run_dir_of(run)
        if (rd / 'learner-done.json').exists():
            return 'done'
        alive = self.head.pid_alive(f'{self.p["pids"]}/learner.pid')
        status = self.learner_status(run)
        if alive and status:
            prog = self.state['progress']
            if status.get('update') != prog.get('update'):
                self.state['progress'] = dict(update=status.get('update'), at=time.time(), status_at=status.get('at'))
            elif time.time() - prog.get('at', time.time()) > STALL_SECONDS and time.time() - self.state['started'].get(
                    'learner', 0) > STALL_SECONDS + 900:
                self.stop_learner(f'stalled at update {prog.get("update")}')
                alive = False
            if alive and self.diverged(rd):
                self.stop_learner('diverged (approx_kl > .3 for 5 updates)')
                alive = False
        if alive:
            return 'alive'
        if (rd / 'STOP').exists() and self.state['started'].get('learner', 0) < time.time() - 60 and not alive:
            # a STOP ended training but no done marker (killed mid-final?): treat latest as final
            if not (rd / 'final.npz').exists() and (rd / 'latest.npz').exists():
                for ext in ('.npz', '.json', '-critic.npz', '-critic.json'):
                    if (rd / f'latest{ext}').exists():
                        (rd / f'final{ext}').write_bytes((rd / f'latest{ext}').read_bytes())
                self.log(event='latest_promoted_to_final', run=run)
            (rd / 'learner-done.json').write_text(json.dumps(dict(at=time.time(), promoted=True, run=run)))
            return 'done'
        if self.state['learner_restarts'] >= MAX_LEARNER_RESTARTS:
            self.log(event='learner_restart_limit')
            return 'dead'
        error = (rd / 'learner-error.txt')
        if error.exists():
            self.log(event='learner_error', tail=error.read_text()[-1500:])
            error.rename(rd / f'learner-error-{int(time.time())}.txt')
        self.state['learner_restarts'] += 1
        self.start_learner(run)
        return 'restarted'

    def diverged(self, rd):
        try:
            lines = (rd / 'training.jsonl').read_text().strip().splitlines()[-5:]
            rows = [json.loads(x) for x in lines]
            return len(rows) == 5 and all(float(r.get('approx_kl', 0)) > .3 for r in rows)
        except (OSError, ValueError):
            return False

    # ----- actors ------------------------------------------------------------------------------------------------
    def actor_nodes(self):
        """[(host, index, command, env, root)] for every actor node of the fleet."""
        nodes = []
        learner_ip = self.cluster['head']['data_ip']
        stop_file = str(self.run_dir / 'STOP-ALL')
        until = self.plan['fleet_until']
        for i, a in enumerate(self.cluster['head_actors']):
            cmd = actor_command(self.cluster, self.cluster['head']['root'], i, learner_ip, a['processes'], a['games'],
                                a['cores'], 'cuda', until, stop_file, self.cluster.get('serve_options'))
            env = dict(ACTOR_ENV, CUDA_VISIBLE_DEVICES=str(a['gpu']), JAX_COMPILATION_CACHE_DIR=self.p['cache'],
                       PYTHONPATH=self.p['src'])
            nodes.append((self.head, i, cmd, env))
        wa = self.cluster['worker_actors']
        for j, w in enumerate(self.cluster['workers'] if wa.get('processes') else []):
            host = self.hosts[w['name']]
            wp = paths(w['root'])
            cmd = actor_command(self.cluster, w['root'], 100 + j, learner_ip, wa['processes'], wa['games'], wa['cores'],
                                'cpu', until, None)
            env = dict(ACTOR_ENV, JAX_COMPILATION_CACHE_DIR=wp['cache'], PYTHONPATH=wp['src'])
            nodes.append((host, 100 + j, cmd, env))
        return nodes

    def start_actors(self, only_dead=True):
        for host, index, cmd, env in self.actor_nodes():
            hp = paths(host.root)
            pidfile = f'{hp["pids"]}/actor-{index}.pid'
            key = f'actor-{host.name}-{index}'
            if only_dead:
                if time.time() - self.state['started'].get(key, 0) < ACTOR_SETTLE:
                    continue
                if host.pid_alive(pidfile):
                    stalled = self.actor_stalled(host, index, key)
                    if not stalled:
                        continue
                    # alive but not producing (the Flagship outage pattern: processes up, event loop dead)
                    self.log(event='actor_stalled', host=host.name, index=index, reason=stalled)
                    host.stop(pidfile, pattern=f'{hp["python"]} .*(node_actor|multiprocessing)', grace=5)
            out = host.start(f'actor-{index}', cmd, env, hp['code'], f'{hp["logs"]}/actor-{index}.log', pidfile)
            self.state['started'][key] = time.time()
            self.log(event='actor_start', host=host.name, index=index, result=out)

    def actor_stalled(self, host, index, key):
        """None if the node's report shows progress; else a reason. A node reports every ~20 s (report file 'at');
        sent_samples must grow while the learner is up. Ten minutes without either counts as stalled."""
        if not self.head.pid_alive(f'{self.p["pids"]}/learner.pid'):
            return None   # no learner: actors are expected to wait
        text = host.read(f'{paths(host.root)["logs"]}/actor-{index}.json')
        try:
            report = json.loads(text) if text else None
        except ValueError:
            report = None
        progress = self.state.setdefault('actor_progress', {})
        seen = progress.get(key)
        now = time.time()
        if report is None:
            return 'no report file' if now - self.state['started'].get(key, 0) > ACTOR_STALL_SECONDS else None
        if now - float(report.get('at', 0)) > ACTOR_STALL_SECONDS:
            return f'report {int(now - float(report.get("at", 0)))} s old'
        sent = int(report.get('sent_samples') or 0)
        if seen is None or sent != seen['sent']:
            progress[key] = dict(sent=sent, at=now)
            return None
        if now - seen['at'] > ACTOR_STALL_SECONDS:
            return f'sent_samples stuck at {sent} for {int(now - seen["at"])} s'
        return None

    def stop_actors(self):
        for host, index, _, _ in self.actor_nodes():
            hp = paths(host.root)
            # Spawned actor workers and inference servers outlive their node; they run as our venv's python, so the
            # pattern is the interpreter path (shared boxes: never a bare 'multiprocessing' or 'python').
            host.stop(f'{hp["pids"]}/actor-{index}.pid', pattern=f'{hp["python"]} .*(node_actor|multiprocessing)', grace=5)
        self.log(event='actors_stopped')

    # ----- evaluation jobs (detached on the eval/search hosts) ---------------------------------------------------
    def job_start(self, name, host, command, cores):
        hp = paths(host.root)
        pidfile = f'{hp["pids"]}/{name}.pid'
        out = host.start(name, f'taskset -c {cores} {command}', dict(JAX_PLATFORMS='cpu', PYTHONPATH=hp['src']),
                         hp['code'], f'{hp["logs"]}/{name}.log', pidfile)
        self.state['jobs'][name] = dict(host=host.name, pidfile=pidfile, started=time.time(), done=False)
        self.log(event='job_start', job=name, host=host.name, result=out)

    def job_finished(self, name):
        job = self.state['jobs'][name]
        return not self.hosts[job['host']].pid_alive(job['pidfile'])

    def ref_paths(self, host):
        return {name: f'{paths(host.root)["ckpt"]}/{rel}' for name, rel in self.plan['refs'].items()}

    def freeze(self, run, stage_name=None):
        """Copy run/latest (or final) + json to ROOT/ladder/<run>-<stage>.npz and to the eval host; returns (tag, path there)."""
        rd = self.run_dir_of(run)
        src = rd / ('final.npz' if stage_name == 'final' else 'latest.npz')
        meta = json.loads(src.with_suffix('.json').read_text())
        tag = f'{run}-{meta["stage"]}' + ('-final' if stage_name == 'final' else '')
        dst = self.root / 'ladder' / f'{tag}.npz'
        dst.parent.mkdir(exist_ok=True)
        dst.write_bytes(src.read_bytes())
        dst.with_suffix('.json').write_text(src.with_suffix('.json').read_text())
        host = self.hosts[self.plan['eval']['host']]
        remote = f'{paths(host.root)["ladder"]}/{tag}.npz'
        self.push(host, [dst, dst.with_suffix('.json')], str(Path(remote).parent))
        return tag, remote, meta['stage']

    def push(self, host, files, remote_dir):
        host.run(f'mkdir -p {shlex.quote(remote_dir)}')
        target = (host.hops[-1] + ':' if host.hops else '') + remote_dir + '/'
        files = ' '.join(shlex.quote(str(f)) for f in files)
        self.head.run(f'scp -q -o BatchMode=yes {files} {shlex.quote(target)}' if host.hops else f'cp {files} {remote_dir}/',
                      timeout=900)

    def ladder_round(self, run, final=False):
        ev = self.plan['eval']
        host = self.hosts[ev['host']]
        tag, candidate, stage = self.freeze(run, 'final' if final else None)
        r = 99 if final else self.state['ladder_round']
        seeds = ev['seed_base'] + 1000 * r
        maps = ev['final_maps'] if final else ev['maps']
        refs = self.ref_paths(host)
        if final and 'start' not in refs and self.plan.get('start_ref'):
            refs['start'] = f'{paths(host.root)["ckpt"]}/{self.plan["start_ref"]}'
        hp = paths(host.root)
        for ref, path in refs.items():
            name = f'ladder-{tag}-vs-{ref}'
            out = f'{hp["arena"]}/{name}.json'
            cmd = (f'{hp["python"]} experiments/curio/arena.py match --candidate {shlex.quote(candidate)} --opponent '
                   f'{shlex.quote(path)} --seed-base {seeds} --maps {maps} --games 4 --processes {ev["processes"]} '
                   f'--map-mix {ev["map_mix"]} --size {ev["size"]} --out {shlex.quote(out)} --name {shlex.quote(name)}')
            self.job_start(name, host, cmd, ev['cores'])
            self.state['jobs'][name].update(kind='ladder', run=run, stage=stage, candidate=candidate, reference=ref,
                                            opponent=path, round=r, seed_base=seeds, out=out, final=final)
        if not final:
            self.state['ladder_round'] += 1
            self.state['next_ladder'] = time.time() + ev['interval']

    def collect_ladders(self):
        """Finished ladder jobs -> rows in ROOT/ladder/<run>-history.json (one json per run tag)."""
        for name, job in list(self.state['jobs'].items()):
            if job.get('kind') != 'ladder' or job['done'] or not self.job_finished(name):
                continue
            host = self.hosts[job['host']]
            text = host.read(job['out'])
            row = dict(run_id=job['run'], stage=job['stage'], candidate=job['candidate'], reference=job['reference'],
                       opponent=job['opponent'], round=job['round'], seed_base=job['seed_base'], finished=time.time(),
                       final=job['final'])
            if text:
                out = json.loads(text)
                row.update({k: out.get(k) for k in LADDER_KEYS})
                (self.root / 'arena').mkdir(exist_ok=True)
                (self.root / 'arena' / f'{name}.json').write_text(text)
            else:
                row['error'] = 'no output (see the job log on ' + job['host'] + ')'
            hist = self.root / 'ladder' / f'{self.run}-history.json'
            rows = json.loads(hist.read_text()) if hist.exists() else []
            rows.append(row)
            hist.write_text(json.dumps(rows, indent=1))
            job['done'] = True
            self.log(event='ladder_row', **{k: row.get(k) for k in ('stage', 'reference', 'wins', 'losses',
                                                                   'draws_or_truncations', 'win_rate', 'error')})

    def start_head_check(self):
        host = self.hosts[self.plan['eval']['host']]
        hp = paths(host.root)
        fog = self.root / 'runs' / f'{self.run}-foghead' / 'final.npz'
        pol = self.run_dir / 'final.npz'
        remote_dir = f'{hp["ladder"]}/final'
        self.push(host, [fog, fog.with_suffix('.json')], remote_dir + '/foghead')
        self.push(host, [pol, pol.with_suffix('.json')], remote_dir + '/policy')
        refs = {k: v for k, v in self.ref_paths(host).items() if k in ('c3', 'champion_foghead')}
        out = f'{hp["arena"]}/head-check-{self.run}.json'
        cmd = (f'{hp["python"]} experiments/curio/head_check.py --policy {remote_dir}/policy/final.npz --head '
               f'{remote_dir}/foghead/final.npz --references-json {shlex.quote(json.dumps(refs))} --games 32 --out {out}')
        self.job_start(f'head-check-{self.run}', host, cmd, self.plan['eval']['cores'])
        self.state['jobs'][f'head-check-{self.run}'].update(kind='head_check', out=out)

    def start_search(self):
        sp = self.plan['search']
        host = self.hosts[sp['host']]
        hp = paths(host.root)
        fog = self.root / 'runs' / f'{self.run}-foghead' / 'final.npz'
        remote_dir = f'{hp["ladder"]}/final/foghead'
        self.push(host, [fog, fog.with_suffix('.json')], remote_dir)
        refs = self.ref_paths(host)
        search = {k: sp[k] for k in ('lanes', 'depth', 'rollout_temperature', 'replies', 'determinize')}
        arms = {
            f'search-{self.run}-s-vs-champ-s': dict(candidate=f'{remote_dir}/final.npz', value=f'{remote_dir}/final.npz',
                                                   opponent=refs['champion_foghead'], opponent_search=search),
            f'search-{self.run}-s-vs-champ-raw': dict(candidate=f'{remote_dir}/final.npz', value=f'{remote_dir}/final.npz',
                                                     opponent=refs['champion_foghead']),
        }
        for name, arm in arms.items():
            cfg = dict(arm, name=name, maps=sp['maps'], seed_base=sp['seed_base'], map_mix=self.plan['eval']['map_mix'],
                       size=self.plan['eval']['size'], **search)
            cfg_path = f'{hp["arena"]}/{name}.cfg.json'
            host.write(cfg_path, json.dumps(cfg))
            out = f'{hp["arena"]}/{name}.json'
            cmd = f'{hp["python"]} experiments/curio/arena.py search --cfg {cfg_path} --out {out} --processes {sp["processes"]}'
            self.job_start(name, host, cmd, sp['cores'])
            self.state['jobs'][name].update(kind='search', out=out)

    def collect_outputs(self):
        for name, job in self.state['jobs'].items():
            if job.get('kind') in ('head_check', 'search') and not job['done'] and self.job_finished(name):
                text = self.hosts[job['host']].read(job['out'])
                if text:
                    (self.root / 'arena').mkdir(exist_ok=True)
                    (self.root / 'arena' / f'{name}.json').write_text(text)
                job['done'] = True
                self.log(event='job_done', job=name, output=bool(text))

    def jobs_pending(self):
        return [n for n, j in self.state['jobs'].items() if not j['done']]

    # ----- summary -----------------------------------------------------------------------------------------------
    def summary(self):
        sys.path.insert(0, str(Path(self.p['code']) / 'scripts'))
        lines = [f'# {self.run} on Curio: summary ({time.strftime("%Y-%m-%d %H:%M:%S %Z")})', '']
        hist = self.root / 'ladder' / f'{self.run}-history.json'
        rows = json.loads(hist.read_text()) if hist.exists() else []
        lines.append('## Ladder (greedy, both seats per map; score counts draws as half)')
        lines.append('| stage | reference | W-L-D | win rate | ci95 | round |')
        lines.append('|---|---|---|---|---|---|')
        for r in rows:
            if 'error' in r:
                lines.append(f'| {r.get("stage")} | {r["reference"]} | error | | | {r.get("round")} |')
                continue
            lines.append(f'| {r["stage"]}{" (final)" if r.get("final") else ""} | {r["reference"]} | '
                         f'{r["wins"]}-{r["losses"]}-{r["draws_or_truncations"]} | {r["win_rate"]:.3f} | {r.get("ci95")} | {r["round"]} |')
        try:
            from ladder_elo import fit
            anchor = 'c3' if 'c3' in self.plan['refs'] else next(iter(self.plan['refs']))
            lines += ['', f'## Bradley-Terry Elo (anchor {anchor} = 0)', '| player | elo | se | W-L-D |', '|---|---|---|---|']
            for e in fit(rows, anchor):
                lines.append(f'| {e["player"]} | {e["elo"]} | {e["se"]} | {"-".join(map(str, e["record"]))} |')
        except Exception as error:   # noqa: BLE001
            lines.append(f'(elo fit failed: {error!r})')
        for name, job in self.state['jobs'].items():
            if job.get('kind') in ('head_check', 'search'):
                path = self.root / 'arena' / f'{name}.json'
                lines += ['', f'## {name}']
                if path.exists():
                    out = json.loads(path.read_text())
                    keys = ('wins', 'losses', 'truncations', 'win_ci95', 'elo', 'elo_ci95', 'policy_unchanged', 'heads',
                            'wall_seconds')
                    lines.append('```'); lines.append(json.dumps({k: out.get(k) for k in keys if k in out}, indent=1)); lines.append('```')
                else:
                    lines.append('(no output)')
        (self.run_dir / 'summary.md').write_text('\n'.join(lines) + '\n')
        self.log(event='summary_written')

    # ----- phases ------------------------------------------------------------------------------------------------
    def tick(self):
        if (self.run_dir / 'STOP-ALL').exists():
            self.log(event='stop_all')
            self.stop_learner('STOP-ALL')
            self.stop_actors()
            self.state['phase'] = 'stopped'
            return False
        phase = self.state['phase']
        if phase == 'train':
            state = self.check_learner(self.run)
            self.start_actors(only_dead=True)
            self.collect_ladders()
            if state == 'alive' and time.time() >= self.state['next_ladder'] and (self.run_dir / 'latest.npz').exists():
                self.ladder_round(self.run)
            if state == 'alive' and time.time() > self.plan['until_unix'] + 900 and not (self.run_dir / 'STOP').exists():
                (self.run_dir / 'STOP').write_text(str(time.time()))   # backstop: the learner's own clock should end it
                self.log(event='backstop_stop')
            if state == 'done':
                self.log(event='training_done')
                self.ladder_round(self.run, final=True)
                if self.plan.get('auto') and self.plan.get('foghead'):
                    self.stop_actors()
                    self.state['learner_restarts'] = 0
                    self.state['phase'] = 'foghead'
                    self.start_learner(f'{self.run}-foghead')
                    time.sleep(90)   # let it bind the port before the fleet reconnects
                    self.start_actors(only_dead=False)
                else:
                    self.state['phase'] = 'final'
            elif state == 'dead':
                self.state['phase'] = 'final'
        elif phase == 'foghead':
            run = f'{self.run}-foghead'
            state = self.check_learner(run)
            self.start_actors(only_dead=True)
            self.collect_ladders()
            if state in ('done', 'dead'):
                self.log(event='foghead_done', state=state)
                self.stop_actors()
                if state == 'done':
                    self.start_head_check()
                    self.start_search()
                self.state['phase'] = 'final'
        elif phase == 'final':
            self.collect_ladders()
            self.collect_outputs()
            if not self.jobs_pending():
                self.summary()
                self.state['phase'] = 'done'
                return False
        return True

    def loop(self):
        self.log(event='keeper_start', run=self.run, plan=dict(until_unix=self.plan['until_unix'], auto=self.plan.get('auto')))
        while True:
            try:
                go = self.tick()
            except Exception as error:   # noqa: BLE001 - never die on a transient ssh failure
                self.log(event='tick_error', error=repr(error), trace=traceback.format_exc()[-1500:])
                go = True
            self.save()
            if not go:
                self.log(event='keeper_exit')
                return
            time.sleep(60)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--run', required=True)
    a = ap.parse_args()
    Keeper(a.root, a.run).loop()
