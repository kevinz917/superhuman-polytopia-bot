"""Finish this benchmark's artifact pipeline when its Modal collector exits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def finish(root, collector_pid):
    root=Path(root).resolve()
    while True:
        try:os.kill(collector_pid,0)
        except ProcessLookupError:break
        time.sleep(15)
    repository=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(repository/'scripts/summarize_unrestricted_games.py'),str(root)],
                   check=True,cwd=repository)
    s=json.loads((root/'summary.json').read_text())
    lines=['# Qwen3.5-4B unrestricted gameplay results','',
        'Completed artifact aggregation for the 80-game, unadapted-model comparison. '
        'The current harness was held fixed, with no separate output or thinking token budget.','',
        '| Metric | Thinking off | Thinking on |','|---|---:|---:|']
    a,b=s['modes']['off'],s['modes']['on']
    for title,key in [('Allocated games','allocated'),('Native wins','native_wins'),
        ('Native losses','native_losses'),('Invalid-output forfeits','forfeits'),
        ('Valid executed actions','valid_actions'),('Model calls','calls'),
        ('Total output tokens','output_tokens'),('Maximum response tokens','max_output_tokens')]:
        lines.append(f'| {title} | {a[key]} | {b[key]} |')
    for title,key in [('Mean output tokens per call','mean_output_tokens'),
        ('Median request seconds','median_request_seconds'),('Gameplay seconds','elapsed_seconds')]:
        values=[f'{x[key]:.2f}' if x[key] is not None else 'n/a' for x in (a,b)]
        lines.append(f'| {title} | {values[0]} | {values[1]} |')
    lines+=['','Full status counts:','',f"- Thinking off: `{a['status_counts']}`",
            f"- Thinking on: `{b['status_counts']}`",'',
            'Context exhaustion, simulator truncations and infrastructure failures are not native losses or draws. '
            'Forfeits are failures to return one legal action under the existing single-call contract.','']
    paired=s['paired_win_difference']
    if paired:
        lo,hi=paired['map_cluster_bootstrap_95_percent']
        lines += [f"Paired win-rate difference, thinking on minus off: {100*paired['on_minus_off']:.1f} percentage points; "
                  f"exploratory map-cluster bootstrap 95% interval [{100*lo:.1f}, {100*hi:.1f}]. "
                  'This uses twenty maps and one sampling sequence; it is not proof of a general model ranking.','']
    else:
        lines+=['A fully resolved paired win-rate comparison is unavailable because some games have no native/forfeit outcome.','']
    lines+=['| Map seed | Seat | Thinking off | Thinking on |','|---|---:|---|---|']
    bm={g['id']:g for g in b['games']}
    for g in sorted(a['games'],key=lambda x:(x['seed'],x['seat'])):
        def label(x):
            outcome=('win' if x['reward']>0 else 'loss' if x['reward']<0 else 'zero reward') if x['status']=='native' else x['status']
            return f"{outcome}; {x['reason']}; round {x['round']}"
        lines.append(f"| {g['seed']} | {g['seat']} | {label(g)} | {label(bm[g['id']])} |")
    lines+=['',f'Run artifacts: `{root}`.',
            'Protocol and limitations: [comparison protocol](qwen35-unrestricted-gameplay-2026-10-17.md).','']
    report=repository/'docs/qwen35-unrestricted-gameplay-result-2026-10-17.md'
    report.write_text('\n'.join(lines))
    (root/'finalization.json').write_text(json.dumps(dict(status='complete',report=str(report),
        finished_unix=time.time()),indent=2)+'\n')


if __name__=='__main__':
    try:finish(sys.argv[1],int(sys.argv[2]))
    except Exception:
        Path(sys.argv[1],'finalization-error.txt').write_text(traceback.format_exc())
        raise
