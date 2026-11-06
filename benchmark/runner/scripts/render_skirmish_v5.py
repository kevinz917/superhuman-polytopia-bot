"""Compact source-free v5 report. Model artifacts and raw trajectories stay on Modal."""
import argparse
import json
from pathlib import Path


def compact_snapshot(data):
    """Export selected artifacts and compact receipts without changing the scoring report."""
    result=dict(data)
    if data.get('report',{}).get('status') in ('completed','provisional'):
        result['status']=data['report']['status']
    if result.get('status') in ('evaluating','completed','provisional') and len(data.get('runs',{}))==data.get('total'):
        result['original_attempt_statuses']=data.get('original_attempt_statuses',data.get('current_stages',{}))
        result['current_stages']={run:dict(status=receipt['status'],
            stage=3 if receipt.get('track')=='feedback-3r-40m' else 0,
            effective_run_id=receipt.get('effective_run_id') or run,
            submission_quality=receipt.get('submission_quality')) for run,receipt in data['runs'].items()}
    if data.get('infrastructure_exclusions'):
        fields=('status','reason','error','run_id','model','track','exclusion_reason')
        result['infrastructure_exclusions']={run:{key:value for key,value in receipt.items() if key in fields}
            for run,receipt in data['infrastructure_exclusions'].items()}
    if data.get('report',{}).get('all_attempt_api_cost_report') and data.get('api_cost_report'):
        result['api_cost_report']=dict(probe_completed_calls=data['api_cost_report']['probe_completed_calls'],
            coverage='compatibility probes only; comprehensive authoring receipts are report.all_attempt_api_cost_report')
    result['export_schema']='polytopia-mini-export-v1'
    return result


def render(data):
    if data.get('status') not in ('completed','provisional') or 'report' not in data:
        raise ValueError('cohort is not evaluated; never render pending work as results')
    report=data['report'];models={m['label']:m['model'] for m in data['manifest']['models']}
    number=lambda x:'pending' if x is None else f'{x:.2f}'
    lines=[f'# Polytopia benchmark v5 — {data["prefix"]}','',f'Status: {data["status"]}; smoke: {report["smoke"]}.',
        '',f'Protocol: {report["schema"]}.',report['scoring']+'.','',report['uncertainty']+'.','',
        f'Shared hidden map digest: `{report["maps_sha256"]}`. All stages and tracks use the same layouts.',
        '',f'Scheduled results: {report["games"]}; primary {report["primary_games"]}; peer {report["peer_games"]}.',
        f'Replay audits: {report["audit"]}.','',
        '## Fixed evaluation-only references','']
    if report.get('evaluation_revision'):
        lines[6:6]=[f'Evaluation revision: `{report["evaluation_revision"]}`; referee guard {report["referee_seconds"]} seconds. Guest decision/fuel/memory limits follow the authoring protocol.','']
    if report['smoke']:
        lines[2:2]=['Smoke integration test; these are not leaderboard measurements.','']
        lines=[line.replace(report['uncertainty']+'.','Smoke intervals are integration output only; one build cannot estimate variation between builds.') for line in lines]
    for ref in report['references']:
        lines.append(f'- {ref["label"]}: {ref.get("run_id","released hero-1v1 expert")}; digest `{ref.get("wasm_sha256",ref.get("checkpoint_sha256"))}`.')
    if data.get('integrity_audit'):
        audit=data['integrity_audit'];counts=audit['status_counts']
        lines += ['','## Integrity and runtime outcomes','',
            f'{audit["unique_job_indices"]:,} unique job indices match the frozen plan. Every accepted replay audit passed; no outcomes are unresolved.',
            f'Played games: {counts.get("native",0):,} native conclusions, {counts.get("draw",0):,} engine-cap draws, and {counts.get("forfeit",0):,} guest forfeits. '+
            f'Another {audit["virtual_compile_zeroes"]:,} scheduled outcomes are virtual compilation-failure losses.',
            f'Guest forfeit reasons: {audit["guest_forfeit_reasons"]}.','']
        failures=[(key,value['forfeit']) for key,value in audit['final_primary_statuses'].items() if value.get('forfeit')]
        lines.append('Final-stage fixed-reference runtime forfeits: '+('; '.join(key.replace('|',' / ')+f' = {count}/270' for key,count in failures) if failures else 'none')+'.')
    for track,ranking in report['ranking'].items():
        lines += ['',f'## {track} primary ranking','',
            '| Model | Mean points /100 | 95% bootstrap interval | All attempt scores | Submission quality |',
            '|---|---:|---|---|---|']
        for row in ranking:
            ci=row.get('score_ci95');interval='pending' if not ci else '–'.join(number(v) for v in ci)
            lines.append(f'| {models[row["model_label"]]} | {number(row["score"])} | {interval} | '+
                ', '.join(number(v) for v in row['attempt_scores'])+f' | {row["submission_quality"]} |')
    lines += ['','## Final-stage outcomes against each fixed reference','',
        'Each cell shows points per 100, then wins–losses–draws–unresolved across all builds.','',
        '| Model | Track | '+' | '.join(ref['label'] for ref in report['references'])+' |',
        '|---|---|'+'---|'*len(report['references'])]
    for track,ranking in report['ranking'].items():
        final_stage=3 if track=='feedback-3r-40m' else 0
        for row in ranking:
            parts=[stage for stage in report['stages'] if stage['model_label']==row['model_label'] and stage['track']==track and stage['stage']==final_stage]
            cells=[]
            for ref in report['references']:
                counts={key:sum(part['by_opponent'][ref['label']][key] for part in parts) for key in ('games','wins','losses','draws','unresolved')}
                points=None if counts['unresolved'] else 100*(counts['wins']+0.5*counts['draws'])/counts['games']
                cells.append(number(points)+' ('+'–'.join(str(counts[key]) for key in ('wins','losses','draws','unresolved'))+')')
            lines.append(f'| {models[row["model_label"]]} | {track} | '+' | '.join(cells)+' |')
    lines += ['','## Revision curves on identical hidden tests','',
              '| Model | Build | Initial | Revision 1 | Revision 2 | Revision 3 | Final − initial |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in report['revision_changes']:
        lines.append(f'| {models[row["model_label"]]} | {row["replicate"]} | '+
            ' | '.join(number(v) for v in row['stage_scores'])+f' | {number(row["final_minus_initial"])} |')
    for track,ladder in report['peer_ladders'].items():
        lines += ['',f'## {track} secondary Elo','',
            'Pooled matched-replicate peer games plus final-stage fixed-reference games. '+
            'Failed compilations stay zeroes in the primary score and have no invented Elo. '+
            'Ratings depend on this roster; they are not comparable with archived v4 ratings.','',
            '| Entrant | Elo | W–L–D–U |','|---|---:|---|']
        for row in ladder['ladder']:
            name=models.get(row['entrant'],row['entrant'])
            lines.append(f'| {name} | {number(row["elo"])} | {row["wins"]}–{row["losses"]}–{row.get("draws",0)}–{row["unresolved"]} |')
    lines += ['','## Authoring timing and reliability','',
              '| Run | Status | Quality | Coding minutes | API wait seconds | Tool seconds | Saved API USD |',
              '|---|---|---|---:|---:|---:|---:|']
    for run,receipt in sorted(data.get('runs_detail',{}).items()):
        stages=data['stage_receipts'].get(run,[])
        elapsed=sum(s.get('authoring_seconds') or 0 for s in stages)
        api=sum(s.get('timing',{}).get('api_seconds',0) for s in stages)
        tool=sum(s.get('timing',{}).get('tool_seconds',0) for s in stages)
        cost=sum(s.get('instance_cost') or 0 for s in stages)
        lines.append(f'| {run} | {receipt["status"]} | {receipt.get("submission_quality",stages[-1].get("submission_quality") if stages else "missing")} | {elapsed/60:.2f} | {api:.1f} | {tool:.1f} | ${cost:.4f} |')
    lines += ['','Costs are optional completed-call receipts. Interrupted requests, compatibility probes, '+
        'infrastructure exclusions and Modal compute can add charges. No dollar-based limit is enforced.','',
        'Sources, Wasm, complete model trajectories, private referee chains and immutable plans remain on '+
        f'Modal volume `polytopia-skirmish`, under `mini-v5/{data["prefix"]}` and its referenced `mini-runs` directories.','',
        f'This cohort uses {data["manifest"]["replicates"]} build(s) per model and track and {3 if report["smoke"] else 3*data["manifest"]["maps_per_family"]} hidden layouts. It measures performance under the declared '+
        'deadline and provider routing; it does not establish a universal model ordering or isolate a causal feedback effect.','']
    if report.get('all_attempt_api_cost_report'):
        costs=report['all_attempt_api_cost_report']
        lines += ['## Recorded authoring API receipts by model','',
            '| Model | Scored builds | Excluded infrastructure | All saved authoring calls | Mean per scored pair |',
            '|---|---:|---:|---:|---:|']
        for label,cost in costs['by_model'].items():
            lines.append('| '+models[label]+' | '+' | '.join(f'${cost[key]:.4f}' for key in ('scored_completed_calls','excluded_completed_calls','all_authoring_completed_calls','mean_scored_two_track_pair'))+' |')
        lines += ['',costs['coverage']+'.','']
    if data.get('api_cost_report'):
        if report.get('all_attempt_api_cost_report'):
            lines += [f'Saved compatibility-probe API receipts: ${data["api_cost_report"]["probe_completed_calls"]:.4f}, additional to the authoring table above.','']
        else:
            lines += ['## Recorded API charges','',str(data['api_cost_report']),'']
    if data.get('infrastructure_exclusions'):
        lines += ['## Infrastructure exclusions','']
        for name,r in data['infrastructure_exclusions'].items():
            lines.append(f'- `{name}`: {r["status"]}; {r.get("error",r.get("reason","see archived receipts"))}.')
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input');parser.add_argument('output');args=parser.parse_args()
    Path(args.output).write_text(render(json.loads(Path(args.input).read_text())))
