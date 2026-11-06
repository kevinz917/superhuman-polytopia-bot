#!/usr/bin/env python3
"""Measure pipelined rollout work using parent dispatch, not queued lane time."""
import argparse,gzip,hashlib,json,statistics
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def summarize(source,output,prefix_audit=None):
    root=ROOT/source;report_path=root/'combined/games/report.json'
    report=json.loads(report_path.read_text());executions=json.loads((root/'execution.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    if len(executions)!=protocol['shards'] or any(r.get('exit_code')!=0 for r in executions):
        raise ValueError('complete successful rollout allocation required')
    if not report['complete']:
        if prefix_audit is None:raise ValueError('incomplete evaluation requires an explicit matching prefix audit')
        audit=json.loads((ROOT/prefix_audit).read_text())
        if (audit.get('schema')!='incomplete-game-prefix-replay-audit-v1' or not audit.get('passed')
            or audit.get('evaluation_complete') is not False or audit['games']!=len(protocol['jobs'])
            or audit['report_sha256']!=hashlib.sha256(report_path.read_bytes()).hexdigest()):
            raise ValueError('matching passed prefix audit required')
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
    if protocol.get('runtime_version')!='pipelined-cpu-lanes-v1':raise ValueError('pipelined runtime required')
    dispatch_paths=list(root.glob('*shard*/games/dispatch.jsonl'))
    if len(dispatch_paths)!=protocol['shards']:raise ValueError('one dispatch log per GPU shard required')
    dispatch=[json.loads(line) for path in dispatch_paths for line in path.read_text().splitlines()]
    if sum(row['size'] for row in dispatch)!=calls:raise ValueError('dispatch/trace call count differs')
    lane_seconds=call_seconds
    call_seconds=sum(row['model_seconds'] for row in dispatch)
    coordinator=json.loads((root/'coordinator.json').read_text())
    occupied=sum(r.get('occupied_seconds_with_archive',r['elapsed_seconds']) for r in executions)
    lengths=sorted(inputs);games=len(report['games'])
    cost=occupied*.00079968+coordinator['elapsed_seconds']*.00001532
    result=dict(schema='rollout-runtime-diagnostic-v1',evaluation_complete=report['complete'],games=games,calls=calls,
        inference_batches=len(dispatch),mean_batch_size=calls/len(dispatch),
        batch_size_counts=dict(sorted(Counter(row['size'] for row in dispatch).items())),
        singleton_model_time_fraction=sum(row['model_seconds'] for row in dispatch if row['size']==1)/call_seconds,
        model_decisions=sum(g['model_decisions'] for g in report['games']),
        calls_per_game=calls/games,model_decisions_per_game=sum(g['model_decisions'] for g in report['games'])/games,
        prompt_tokens=dict(total=sum(inputs),mean=statistics.mean(inputs),median=statistics.median(inputs),
                           p95=lengths[int(.95*len(lengths))],max=max(inputs)),completion_tokens=sum(outputs),
        occupied_function_seconds=occupied,model_call_path_seconds=call_seconds,
        other_function_seconds=occupied-call_seconds,model_call_path_fraction=call_seconds/occupied,
        lane_call_seconds_including_queue=lane_seconds,queue_wait_seconds=sum(row['queue_wait_seconds'] for row in dispatch),
        model_call_note='Parent dispatch records each actual model call once, including tokenization/decoding. CPU lanes overlap this work. Lane durations include queue waits and cannot be summed as GPU time.',
        estimated_compute_usd=cost,estimated_compute_usd_per_game=cost/games,
        compressed_trace_bytes=trace_bytes,compressed_trace_bytes_per_game=trace_bytes/games,
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        timing_includes_archive=all('occupied_seconds_with_archive' in r for r in executions),
        limitation='Measured for this policy/runtime only. Faster training does not establish proportional rollout acceleration; revised policies, batches or budgets require new measurements. Published-rate compute excludes startup/idle, builds, storage and egress.')
    with (ROOT/output).open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix-audit',type=Path,help='Explicitly measure audited incomplete prefixes; never grants full-game acceptance.')
    a=p.parse_args();summarize(a.source,a.output,a.prefix_audit)
