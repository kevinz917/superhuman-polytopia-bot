"""Paired SFT smoke diagnostics. Passing these does not authorize main spending."""
from collections import Counter
from .freeze import inference_config


def position_comparison(base, smoke):
    for report in (base, smoke):
        rows = report['results']
        if not report['complete'] or report['allocated'] != 5000 or len(rows) != 5000:
            raise ValueError('both complete 5000-position reports required')
        if len({r['position_id'] for r in rows}) != 5000:
            raise ValueError('duplicate position IDs')
    left = {r['position_id']:r for r in base['results']}
    right = {r['position_id']:r for r in smoke['results']}
    if left.keys() != right.keys():
        raise ValueError('base/smoke position IDs differ')
    for key in left:
        for field in ('teacher_command','board_hash','episode_id','family','input_tokens'):
            if left[key][field] != right[key][field]:
                raise ValueError('base/smoke position inputs differ: '+field)
    valid = sum(r['first_attempt_valid'] for r in right.values())/5000
    before = sum(r['teacher_agreement'] for r in left.values())/5000
    after = sum(r['teacher_agreement'] for r in right.values())/5000
    return dict(base_agreement=before, smoke_agreement=after, agreement_gain=after-before,
                smoke_first_attempt_valid=valid, format_passed=valid>=.95,
                learning_passed=after-before>=.05-1e-12)


def game_comparison(base, smoke, base_protocol, smoke_protocol):
    for key in ('prompt_manifest_hash','map_manifest_hash','weak_opponent_hash','tokenizer_hash','chat_template_hash'):
        if base_protocol[key] != smoke_protocol[key]:
            raise ValueError('base/smoke protocols differ: '+key)
    if inference_config(base_protocol['config']) != inference_config(smoke_protocol['config']):
        raise ValueError('model or inference settings differ')
    if base_protocol['kind'] != 'base' or smoke_protocol['kind'] != 'smoke' or base_protocol['adapter'] is not None or not smoke_protocol['adapter']:
        raise ValueError('base and smoke model identity required')
    def identity(g):
        return tuple(g[k] for k in ('seed','board_hash','seat','candidate','strategy','opponent'))
    summaries = {}
    previous = None
    for name, report, protocol in [('base',base,base_protocol),('smoke',smoke,smoke_protocol)]:
        expected = {identity(g) for g in protocol['jobs']}
        rows = report['games']
        actual = {identity(g) for g in rows}
        if len(expected) != 50 or len(rows) != 50 or actual != expected or not report['complete'] or report.get('unstarted_jobs'):
            raise ValueError('all 50 allocated games required for each model')
        if previous is not None and expected != previous:
            raise ValueError('base/smoke maps, seats or opponents differ')
        previous = expected
        counts = dict(Counter(g['status'] for g in rows))
        calls = sum(g['attempts'] for g in rows)
        decisions = sum(g['model_decisions'] for g in rows)
        seconds = report.get('occupied_seconds_with_archive', report['elapsed_seconds'])
        summaries[name] = dict(games=50, statuses=counts,
            native_completion=counts.get('native',0)/50,
            wins=sum(g['status']=='native' and g['reward']>0 for g in rows),
            losses=sum(g['status']=='native' and g['reward']<0 for g in rows),
            calls=calls, calls_per_game=calls/50, decisions=decisions,
            repairs=calls-decisions, first_attempt_valid=sum(g['first_attempt_valid'] for g in rows)/max(1,decisions),
            occupied_function_seconds=seconds, estimated_compute_usd=seconds*.00078192,
            estimated_compute_usd_per_game=seconds*.00078192/50,
            integrity_errors=counts.get('infrastructure_error',0))
    return dict(models=summaries,
        completion_passed=summaries['smoke']['native_completion']>=.8,
        execution_passed=all(not s['integrity_errors'] for s in summaries.values()),
        # Frozen 400 base validation + 2400 trained validation + 6000 final.
        projected_full_evaluation_usd=2400*summaries['base']['estimated_compute_usd_per_game']+
                                      6400*summaries['smoke']['estimated_compute_usd_per_game'],
        projection_note='8800 planned games; smoke stands in for both main seeds. Opponent strength, game duration and checkpoint policy can change cost. Startup/storage/builds are excluded; this is not a spend authorization.')
