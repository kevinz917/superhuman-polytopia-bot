"""Matched-update GPU comparison for bench-gpu-* runs.

Per-update time grows as games progress, so runs that fit fewer updates look
faster. Every benchmark uses the same seed and the deterministic fast engine, so
GPUs with the same (lanes, horizon) play nearly identical games: comparing the
same update indices is fair. Update 1 (compilation) is excluded.

Downloads each run's training.json into runs/structured-ppo-v3/gpu-benchmark-rows/
(skipping files already present) and prints one table per (lanes, horizon, window).
"""
import argparse
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / 'runs/structured-ppo-v3/gpu-benchmark-rows'


def load_runs(refresh):
    ledger = json.loads((ROOT / 'runs/structured-ppo-v3/budget.json').read_text())
    ROWS.mkdir(parents=True, exist_ok=True)
    groups = {}
    for r in ledger['runs']:
        if not r['run_id'].startswith('bench-gpu-v'):
            continue
        local = ROWS / f"{r['run_id']}.json"
        if refresh or not local.exists():
            subprocess.run(['modal', 'volume', 'get', '--env', 'staging', '--force', 'polytopia-structured-rl',
                            f"{r['run_id']}/train-decision-budget-fast-{r['environments']}/training.json", str(local)],
                           capture_output=True)
        try:
            rows = json.loads(local.read_text())
        except (OSError, ValueError):
            continue
        key = (r['environments'], r['horizon'])
        best = groups.setdefault(key, {}).get(r['gpu_type'])
        if best is None or len(rows) > len(best['rows']):
            groups[key][r['gpu_type']] = dict(run_id=r['run_id'], rate=r['rate_usd_per_hour'], rows=rows)
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--windows', type=int, nargs='+', default=[4, 10, 15])
    p.add_argument('--refresh', action='store_true')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    report = []
    for (lanes, horizon), runs in sorted(load_runs(args.refresh).items()):
        decisions = lanes * horizon
        for window in args.windows:
            eligible = {g: v for g, v in runs.items() if len(v['rows']) > window}
            if not eligible:
                continue
            table = []
            for gpu, v in eligible.items():
                seconds = statistics.mean(x['seconds'] for x in v['rows'][1:window + 1])
                table.append(dict(gpu=gpu, run_id=v['run_id'], lanes=lanes, horizon=horizon, window=window,
                    seconds_per_update=seconds, learner_decisions_per_second=decisions / seconds,
                    usd_per_million_decisions=v['rate'] / 3600 * seconds / decisions * 1e6,
                    rate_usd_per_hour=v['rate'], updates=len(v['rows'])))
            table.sort(key=lambda e: e['usd_per_million_decisions'])
            report += table
            print(f'\n## {lanes} lanes x {horizon} horizon ({decisions:,} decisions/update), updates 2..{window + 1}')
            print('| GPU | s/update | learner dec/s | $/M decisions | $/h |')
            print('|---|---:|---:|---:|---:|')
            for e in table:
                print(f"| {e['gpu']} | {e['seconds_per_update']:.2f} | {e['learner_decisions_per_second']:,.0f} | "
                      f"{e['usd_per_million_decisions']:.3f} | {e['rate_usd_per_hour']:.2f} |")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
