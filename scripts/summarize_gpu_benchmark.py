"""Summarize bench-gpu-<tag>-* runs: steady-state PPO throughput and cost per GPU/lanes.

Downloads train.json, usage.json and log.txt for each run from the
polytopia-structured-rl volume, then writes a JSON table and prints a markdown one.
Cost per million learner decisions uses steady-state seconds per update (first
update, which includes compilation, excluded) at the job's recorded hourly rate.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fetch(remote, local):
    result = subprocess.run(['modal', 'volume', 'get', '--env', 'staging', '--force', 'polytopia-structured-rl',
                             remote, str(local)], capture_output=True, text=True)
    return result.returncode == 0 and Path(local).exists()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    ledger = json.loads((ROOT / 'runs/structured-ppo-v3/budget.json').read_text())
    rows = [r for r in ledger['runs'] if r['run_id'].startswith(f'bench-gpu-{args.tag}-')]
    table = []
    with tempfile.TemporaryDirectory() as tmp:
        for r in rows:
            run_id, local = r['run_id'], Path(tmp) / r['run_id']
            local.mkdir()
            entry = dict(run_id=run_id, gpu=r['gpu_type'], lanes=r['environments'], rate_usd_per_hour=r['rate_usd_per_hour'])
            if fetch(f'{run_id}/usage.json', local / 'usage.json'):
                usage = json.loads((local / 'usage.json').read_text())
                entry.update(device=usage['nvidia_smi'].split(',')[0], function_seconds=usage['function_seconds'],
                             job_estimate_usd=usage['estimate_usd'])
            if fetch(f'{run_id}/train.json', local / 'train.json'):
                result = next(iter(json.loads((local / 'train.json').read_text()).values()))
                seconds = result['seconds_per_update']
                entry.update(updates=result['updates'], seconds_per_update=seconds,
                    collection_seconds=result['collection_seconds'], phase_seconds=result['phase_seconds'],
                    decisions_per_update=result['decisions_per_update'],
                    learner_decisions_per_second=result['learner_decisions_per_second'],
                    usd_per_million_decisions=r['rate_usd_per_hour'] / 3600 * seconds / result['decisions_per_update'] * 1e6,
                    max_logp_error=result['max_logp_error'], atomic_steps=result['atomic_steps'])
            else:
                fetch(f'{run_id}/log.txt', local / 'log.txt')
                log = (local / 'log.txt').read_text() if (local / 'log.txt').exists() else ''
                entry['failure_tail'] = log.strip().splitlines()[-3:] if log else ['no log']
            table.append(entry)
    table.sort(key=lambda e: (e.get('usd_per_million_decisions', 1e9)))
    if args.out:
        args.out.write_text(json.dumps(table, indent=2) + '\n')
    print('| GPU | device | lanes | updates | s/update | collect s | learner dec/s | $/M decisions | max logp err |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for e in table:
        if 'seconds_per_update' in e:
            print(f"| {e['gpu']} | {e.get('device','?')} | {e['lanes']} | {e['updates']} | {e['seconds_per_update']:.2f} | "
                  f"{e['collection_seconds']:.2f} | {e['learner_decisions_per_second']:,.0f} | "
                  f"{e['usd_per_million_decisions']:.3f} | {e['max_logp_error']:.1e} |")
        else:
            print(f"| {e['gpu']} | {e.get('device','?')} | {e['lanes']} | failed | {' / '.join(e['failure_tail'])[:120]} | | | | |")


if __name__ == '__main__':
    main()
