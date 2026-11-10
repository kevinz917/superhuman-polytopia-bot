"""Session L tiered error policy for the V3 flagship (LAUNCH-PLAN.md section 2), as pure functions over a state dir.

Everything here is plain Python + JSON files so it runs identically in the Modal container (state dir on the run's
volume, survives container loss) and in local tests. One writer: the stage supervisor of the live attempt.

State files under <run_dir>/v3l/:
  attempts.jsonl   one line per worker start / end: {"kind": "start"|"end", "attempt", "stage", "launch_id", "t", ...}
  heartbeat.json   the live attempt's last heartbeat {"attempt", "stage", "t", "update", ...}
  nan.jsonl        one line per NaN/Inf rollback {"t", "stage", "update"}
  resumes.jsonl    one line per counted resume {"t", "stage", "why"}
  events.jsonl     every notification-worthy event (the monitor relays these; ids are unique)
  status.json      {"state": "running"|"paused"|"complete", "reason", "stage", "t", ...}
"""
import json
import os
import time
import uuid
from pathlib import Path

# trainer exit codes set by flagship_harness.py
EXIT_OK = 0
EXIT_NAN = 70
EXIT_LEAK = 71
EXIT_ILLEGAL = 72
EXIT_CONFIG = 73
EXIT_PEACE = 74      # stage 4: the peace-collusion monitor's PAUSE rule

SEVERITY = {  # event kind -> notification severity
    'stage_start': 'INFO', 'container_lost_resume': 'INFO', 'trainer_crash_resume': 'INFO', 'nan_rollback': 'INFO',
    'illegal_game_dropped': 'INFO', 'leak_check_pass': 'DEBUG', 'checkpoint': 'DEBUG', 'heartbeat': 'DEBUG',
    'complete': 'INFO', 'peace_monitor': 'DEBUG', 'peace_warn': 'INFO',
    'paused': 'PAUSE',
    'leakage': 'ALERT', 'budget_overrun': 'ALERT', 'monitor_lost_contact': 'ALERT',
}

PAUSE_ALERT_REASONS = {'leakage', 'config_mismatch', 'budget_overrun'}


def now():
    return time.time()


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _read(path):
    """JSONL rows; a torn last line (container killed mid-append) is dropped."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex[:8]}.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


class PolicyState:
    """The persisted counters of one run (all stages)."""

    def __init__(self, run_dir, manifest):
        self.dir = Path(run_dir) / 'v3l'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.m = manifest
        self.stage = int(manifest['stage'])

    # ---- files
    def p(self, name):
        return self.dir / name

    def attempts(self):
        return _read(self.p('attempts.jsonl'))

    def stage_attempts(self):
        rows = [r for r in self.attempts() if r.get('stage') == self.stage]
        starts = [r for r in rows if r['kind'] == 'start']
        ends = {r['attempt']: r for r in rows if r['kind'] == 'end'}
        return starts, ends

    def status(self):
        path = self.p('status.json')
        return json.loads(path.read_text()) if path.exists() else None

    def set_status(self, state, reason='', **extra):
        write_json_atomic(self.p('status.json'), dict(state=state, reason=reason, stage=self.stage, t=now(), **extra))

    def event(self, kind, **fields):
        row = dict(id=uuid.uuid4().hex, kind=kind, severity=SEVERITY.get(kind, 'INFO'), stage=self.stage, t=now(),
                   run_id=self.m['run_id'], **fields)
        if kind == 'paused' and fields.get('reason') in PAUSE_ALERT_REASONS:
            row['severity'] = 'ALERT'
        _append(self.p('events.jsonl'), row)
        return row

    def events(self):
        return _read(self.p('events.jsonl'))

    # ---- attempts and resumes
    def begin_attempt(self, launch_id):
        """Register this worker start. Returns (attempt number, lost_previous): lost_previous = the previous attempt
        of this stage started but never ended (its container was lost / preempted / crashed hard)."""
        starts, ends = self.stage_attempts()
        lost = bool(starts) and starts[-1]['attempt'] not in ends
        if lost:
            # freeze the lost attempt's billed end NOW (its last heartbeat + 60 s), before this attempt's heartbeats
            # overwrite heartbeat.json (Q16 21:45 PT: the clock had fallen back to start + 60 s for a 9-min attempt)
            prev = starts[-1]
            hb_path = self.p('heartbeat.json')
            hb = json.loads(hb_path.read_text()) if hb_path.exists() else {}
            end_t = float(hb['t']) + 60. if hb.get('attempt') == prev['attempt'] and hb.get('t') else prev['t'] + 60.
            _append(self.p('attempts.jsonl'), dict(kind='end', attempt=prev['attempt'], stage=self.stage, t=end_t,
                                                    outcome='lost'))
        attempt = sum(1 for r in self.attempts() if r['kind'] == 'start')   # unique and monotone over the run
        _append(self.p('attempts.jsonl'), dict(kind='start', attempt=attempt, stage=self.stage, launch_id=launch_id,
                                                t=now(), container=os.environ.get('MODAL_TASK_ID', 'local'),
                                                usd_per_hour=float(self.m.get('usd_per_hour', 0.))))
        return attempt, lost

    def end_attempt(self, attempt, outcome):
        _append(self.p('attempts.jsonl'), dict(kind='end', attempt=attempt, stage=self.stage, t=now(), outcome=outcome))

    def resumes(self):
        return [r for r in _read(self.p('resumes.jsonl')) if r.get('stage') == self.stage]

    def count_resume(self, why):
        _append(self.p('resumes.jsonl'), dict(t=now(), stage=self.stage, why=why))
        return len(self.resumes())

    def resume_allowed(self):
        return len(self.resumes()) <= int(self.m.get('max_resumes_per_stage', 3))

    # ---- dollar-matched snapshots (written by the harness when it has scheduled the save)
    def usd_snapshots(self):
        return _read(self.p('usd-snapshots.jsonl'))

    def record_usd_snapshot(self, label, update):
        _append(self.p('usd-snapshots.jsonl'), dict(label=label, update=update, t=now(), stage=self.stage))

    # ---- NaN window
    def record_nan(self, update):
        _append(self.p('nan.jsonl'), dict(t=now(), stage=self.stage, update=update))
        window = float(self.m.get('nan_window_s', 3600))
        recent = [r for r in _read(self.p('nan.jsonl')) if r['t'] >= now() - window]
        return len(recent)

    def nan_should_pause(self, recent_count):
        return recent_count >= int(self.m.get('nan_pause_count', 3))

    # ---- budget clock (list rate x container wall-clock, every attempt of this stage, plus a startup allowance)
    def heartbeat(self, attempt, **fields):
        write_json_atomic(self.p('heartbeat.json'), dict(attempt=attempt, stage=self.stage, t=now(), **fields))

    def stage_seconds(self, live_attempt=None, live_start=None):
        return sum(sec for sec, _ in self._attempt_seconds(live_attempt))

    def _attempt_seconds(self, live_attempt=None):
        """[(container seconds incl. allowance, USD/h of that attempt)] for every attempt of this stage. The rate is the
        one the attempt started with (a stage can switch 8 -> 16 GPUs at a sub-boundary: stage 3, $500)."""
        starts, ends = self.stage_attempts()
        out = []
        hb_path = self.p('heartbeat.json')
        hb = json.loads(hb_path.read_text()) if hb_path.exists() else {}
        allowance = float(self.m.get('startup_allowance_s', 180))
        total = 0.
        for s in starts:
            a = s['attempt']
            if a == live_attempt:
                end_t = now()
            elif a in ends:
                end_t = ends[a]['t']
            elif hb.get('attempt') == a:
                end_t = hb['t'] + 60.          # lost attempt: last heartbeat + one minute of unobserved billing
            else:
                end_t = s['t'] + 60.
            out.append((max(0., end_t - s['t']) + allowance, float(s.get('usd_per_hour') or self.m['usd_per_hour'])))
        return out

    def stage_spent_usd(self, live_attempt=None):
        return sum(sec * rate / 3600. for sec, rate in self._attempt_seconds(live_attempt))

    def stage_budget_usd(self):
        """This stage's own tranche: the cumulative boundary minus the campaign spend measured before the stage."""
        return float(self.m['stage_cap_usd']) - float(self.m.get('campaign_spent_before_usd', 0.))

    def seconds_left(self, live_attempt=None):
        """Container seconds until the graceful-stop line (stage tranche minus the stop reserve)."""
        usable = self.stage_budget_usd() - float(self.m.get('stop_reserve_usd', 3.))
        left_usd = usable - self.stage_spent_usd(live_attempt)
        return left_usd * 3600. / float(self.m['usd_per_hour'])


def classify_exit(rc, stop_reason, complete):
    """Trainer exit -> (action, reason). action in {'pause', 'complete', 'rollback', 'resume'}."""
    if rc == EXIT_OK and complete and not stop_reason:
        return 'complete', 'training_seconds reached'
    if rc == EXIT_OK and stop_reason:
        return 'pause', stop_reason
    if rc == EXIT_NAN:
        return 'rollback', 'nonfinite update'
    if rc == EXIT_LEAK:
        return 'pause', 'leakage'
    if rc == EXIT_ILLEGAL:
        return 'pause', 'illegal_rate'
    if rc == EXIT_CONFIG:
        return 'pause', 'config_mismatch'
    if rc == EXIT_PEACE:
        return 'pause', 'peace_collapse'
    if rc == EXIT_OK:
        return 'pause', 'unclassified_clean_exit'     # neither complete nor asked to stop: do not guess
    return 'resume', f'trainer exit {rc}'
