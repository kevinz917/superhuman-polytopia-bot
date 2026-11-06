"""Poll a distributed run's registry entries (learner status + per-container actor summaries) into a JSONL file,
for per-container steady-state rates and memory high-water. Read-only (Modal Dict reads); no compute.

    MODAL_ENVIRONMENT=staging python scripts/levers_smoke_monitor.py <run_id> runs/levers/<run_id>-monitor.jsonl [--every 30]
    python scripts/levers_smoke_monitor.py --summarize runs/levers/<run_id>-monitor.jsonl [--skip 300]
"""
import json
import sys
import time


def poll(run_id, out, every):
    import modal
    registry = modal.Dict.from_name('polytopia-distributed-rl-registry')
    while True:
        entry = registry.get(run_id) or {}
        status = registry.get(f'{run_id}/status') or {}
        actors = {}
        for i in range(64):
            a = registry.get(f'{run_id}/actor/{i}')
            if a is None:
                if i > 8:
                    break
                continue
            actors[i] = a
        row = dict(t=time.time(), entry=entry, status={k: v for k, v in status.items() if k != 'actors'}, actors=actors)
        with open(out, 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(dict(t=row['t'], update=status.get('update'), relays=status.get('relays'),
                              connections=status.get('connections'),
                              actors={i: (a.get('steps'), a.get('alive'), a.get('memory_peak_gib'), a.get('pss_peak_gib'))
                                      for i, a in actors.items()})), flush=True)
        if 'stopped' in entry and actors and all(a.get('exitcodes') is not None for a in actors.values()):
            return
        time.sleep(every)


def summarize(path, skip):
    """Per container: steps/s between its first report after `skip` s of its own life and its last live report."""
    rows = [json.loads(l) for l in open(path)]
    by = {}
    for r in rows:
        for i, a in r['actors'].items():
            by.setdefault(i, []).append(a)
    for i, reps in sorted(by.items(), key=lambda kv: int(kv[0])):
        live = [a for a in reps if a.get('exitcodes') is None and a.get('elapsed') is not None]
        steady = [a for a in live if a['elapsed'] >= skip]
        if len(steady) < 2:
            print(i, 'not enough reports')
            continue
        a, b = steady[0], steady[-1]
        rate = (b['steps'] - a['steps']) / (b['elapsed'] - a['elapsed'])
        last = reps[-1]
        print(json.dumps(dict(container=i, processes=last.get('processes'), affinity=last.get('affinity'),
                              cpus=last.get('cpus'), cpu=last.get('cpu_model'), steps_per_s=round(rate),
                              per_process=round(rate / last['processes'], 1), window_s=round(b['elapsed'] - a['elapsed']),
                              memory_peak_gib=last.get('memory_peak_gib'), pss_peak_gib=last.get('pss_peak_gib'),
                              relay=last.get('relay'), restarts=last.get('restarts'), reconnects=last.get('reconnects'),
                              errors=len(last.get('errors') or []))))


if __name__ == '__main__':
    if sys.argv[1] == '--summarize':
        summarize(sys.argv[2], float(sys.argv[sys.argv.index('--skip') + 1]) if '--skip' in sys.argv else 300.)
    else:
        poll(sys.argv[1], sys.argv[2], float(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else 30.)
