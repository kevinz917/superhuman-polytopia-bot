#!/usr/bin/env python3
"""Measure end-to-end rollout work and its model-call component from traces."""
import argparse,gzip,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def summarize(source,output):
    root=ROOT/source;report_path=root/'combined/games/report.json'
    report=json.loads(report_path.read_text());executions=json.loads((root/'execution.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    if not report['complete'] or len(executions)!=protocol['shards'] or any(r.get('exit_code')!=0 for r in executions):
        raise ValueError('complete successful rollout allocation required')
    calls=0;call_seconds=0.;inputs=[];outputs=[];trace_bytes=0
    for game in report['games']:
        paths=list(root.glob('*shard*/games/'+game['id']+'.jsonl.gz'))
        if len(paths)!=1:raise ValueError('unique trace required')
        trace_bytes+=paths[0].stat().st_size
        for line in gzip.open(paths[0],'rt'):
            row=json.loads(line)
            if row.get('policy')!='model':continue
            attempts=row.get('attempts') if row.get('forfeit') else row['diagnostic']
            for attempt in attempts:
                calls+=1;call_seconds+=attempt['batch_seconds']/attempt['batch_size']
                inputs.append(len(attempt['generation']['prompt_token_ids']))
                outputs.append(len(attempt['generation']['response_token_ids']))
    if calls!=sum(g['attempts'] for g in report['games']):raise ValueError('trace call count differs from report')
    occupied=sum(r.get('occupied_seconds_with_archive',r['elapsed_seconds']) for r in executions)
    lengths=sorted(inputs);games=len(report['games'])
    result=dict(schema='rollout-runtime-diagnostic-v1',games=games,calls=calls,
        model_decisions=sum(g['model_decisions'] for g in report['games']),
        calls_per_game=calls/games,model_decisions_per_game=sum(g['model_decisions'] for g in report['games'])/games,
        prompt_tokens=dict(total=sum(inputs),mean=statistics.mean(inputs),median=statistics.median(inputs),
                           p95=lengths[int(.95*len(lengths))],max=max(inputs)),completion_tokens=sum(outputs),
        occupied_function_seconds=occupied,model_call_path_seconds=call_seconds,
        other_function_seconds=occupied-call_seconds,model_call_path_fraction=call_seconds/occupied,
        model_call_note='Each batch duration is divided across its logged requests exactly once. This includes tokenization/decoding, not just GPU compute. Other function time includes simulation, public-state handling, loading and artifact work.',
        estimated_compute_usd=occupied*.00078192,estimated_compute_usd_per_game=occupied*.00078192/games,
        compressed_trace_bytes=trace_bytes,compressed_trace_bytes_per_game=trace_bytes/games,
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        timing_includes_archive=all('occupied_seconds_with_archive' in r for r in executions),
        limitation='Measured for this policy/runtime only. Faster training does not establish proportional rollout acceleration; revised policies, batches or budgets require new measurements. Published-rate compute excludes startup/idle, builds, storage and egress.')
    with (ROOT/output).open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();summarize(a.source,a.output)
