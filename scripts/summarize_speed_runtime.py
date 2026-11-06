"""Summarize completed speed-v2 receipts without reading private game traces."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.smoke_gate import position_comparison,game_comparison


def summarize(root):
    root=Path(root)
    read=lambda path:json.loads(Path(path).read_text())
    state=read(root/'status.json');protocol=read(root/'protocol.json')
    if state['status']!='completed':raise ValueError('completed workflow required')
    stages={s['name']:s for s in state['stages']}
    results={k:s['result'] for k,s in stages.items()}
    old=read(ROOT/'runs/sft0/efficient-smoke-result-v1/summary.json')
    smoke=results['smoke'];fit=results['fit'];world=protocol['world_size']
    reports={k:read(root/(k+'-positions-report.json')) for k in ['base','trained','previous']}
    quality=position_comparison(reports['base'],reports['trained'])
    control=position_comparison(reports['base'],reports['previous'])
    game_protocol=read(ROOT/'runs/sft0/efficient-games-allocation-v1.json')
    games={k:dict(games=[g for worker in results[k+'-games'] for g in worker['report']['games']],
        complete=all(worker['report']['complete'] for worker in results[k+'-games']),
        elapsed_seconds=sum(worker['occupied_seconds'] for worker in results[k+'-games'])) for k in ['base','trained']}
    gameplay=game_comparison(games['base'],games['trained'],
        {**game_protocol,'kind':'base','adapter':None},
        {**game_protocol,'adapter':'/workspace/runs/'+protocol['smoke']['run_id']+'/model/adapter-final'})
    # The old gate embeds historical per-function prices. CPU allocations differ
    # here, so retain measured resource time and omit stale dollar projections.
    gameplay.pop('projected_full_evaluation_usd',None);gameplay.pop('projection_note',None)
    for model in gameplay['models'].values():
        model.pop('estimated_compute_usd',None);model.pop('estimated_compute_usd_per_game',None)
    audit=results['audit']
    if not all(audit[k]['passed'] for k in ['base','trained']):raise ValueError('replay did not pass')
    early=[c for c in smoke.get('early_calls',[]) if c.get('result') and c.get('completed_seconds')]
    summary=dict(schema='speed-v2-result',workflow_seconds=state['finished_at']-state['started_at'],
        world_size=world,fit_seconds=fit['occupied_seconds'],fit_optimizer_seconds=fit['run']['train_seconds'],
        fit_accuracy=fit['run']['fit_after']['accuracy'],fit_reload_identical=fit['run']['reload_identical'],
        old_optimizer_seconds=old['train_seconds'],optimizer_seconds=smoke['run']['train_seconds'],
        optimizer_speedup=old['train_seconds']/smoke['run']['train_seconds'],
        optimizer_gpu_seconds_ratio=world*smoke['run']['train_seconds']/old['train_seconds'],
        training_invocation_seconds=smoke['occupied_seconds'],
        early_diagnostics=[dict(kind=c.get('kind','position'),ready_seconds_from_training_invocation=c['completed_seconds']) for c in early],
        position_quality=quality,previous_adapter_control=control,
        position_timings={k:results[k+'-positions']['occupied_seconds'] for k in reports},
        trained_game_group_wall_seconds=stages['trained-games']['finished_at']-stages['base-games']['finished_at'],
        gameplay=gameplay,replay={k:{f:audit[k][f] for f in ['games','passed','decisions','model_decisions']} for k in ['base','trained']},
        adapter=audit['adapter'],
        smoke_gate_passed=quality['format_passed'] and quality['learning_passed'] and gameplay['completion_passed'] and gameplay['execution_passed'],
        policy_promoted=False,
        limits=['Fixed-prompt inference benchmark is the controlled serving speed comparison.',
                'Full-game runtime also depends on changed policy trajectories.',
                'Workflow includes fresh base games and previous-adapter position controls beyond the prior trained-only evaluation.',
                'Passing a smoke diagnostic does not authorize main training or final-test spending.'])
    if (root/'early-progress-diagnostics.json').exists():summary['early_public_progress']=read(root/'early-progress-diagnostics.json')
    sharded=root.parent/'sharded-positions-v1/execution.json'
    if sharded.exists():
        candidate=read(sharded)
        summary['sharded_positions']=dict(passed=candidate['passed'],comparisons=candidate['comparisons'],
            timings={k:dict(wall_seconds=v['occupied_seconds'],gpu_seconds=v['gpu_seconds']) for k,v in candidate['models'].items()})
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    print(json.dumps(summarize(args.root),indent=2))
