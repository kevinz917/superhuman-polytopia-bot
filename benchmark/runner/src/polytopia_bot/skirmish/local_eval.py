"""Trusted, offline local referee entrypoint. No model APIs or Modal dependency."""
import json
from pathlib import Path
import sys
import uuid

from .common import Budget, sha, write_json
from .game import ReferenceBot, WasiBot
from .local_referee import play_game_streamed, audit_trace_streamed, file_sha


def evaluate(payload):
    root = Path('/results')
    references = payload['references']
    expert = None
    results = []
    cfg=payload.get('evaluation',{})
    from types import SimpleNamespace
    budget=SimpleNamespace(game_seconds=payload.get('referee_seconds',cfg.get('game_seconds',900)),
        fuel=cfg.get('fuel',10**12),move_seconds=cfg.get('move_seconds',2.))

    def bot(spec):
        nonlocal expert
        if spec['kind'] == 'scripted':
            from .reference_panel import StyleBot
            return StyleBot(spec['style'])
        if spec['kind'] == 'hybrid':
            from .reference_panel import HybridReference
            if expert is None:expert=ReferenceBot('/refs/'+spec['file'],spec['checkpoint_sha256'])
            ref=HybridReference.__new__(HybridReference);ref.bot=expert.bot
            from .reference_panel import StyleBot
            ref.style_bot=StyleBot(spec['style']);ref.fraction=spec['hero_turn_fraction']
            return ref
        if spec['kind'] == 'expert':
            if expert is None:
                expert = ReferenceBot('/refs/'+spec['file'], spec['checkpoint_sha256'])
            return expert
        if spec['kind'] == 'frozen':
            path = Path('/refs')/spec['file']
        else:
            path = root/'runs'/spec['run_id']/(f'budget-{spec["snapshot"]}' if spec['kind']=='budget_snapshot' else f'stage-{spec["stage"]}')/'bot.wasm'
        if file_sha(path) != spec['wasm_sha256']:
            raise ValueError('Frozen Wasm digest mismatch')
        return WasiBot(path, budget.fuel, budget.move_seconds)

    for job in payload['jobs']:
        folder = root/job['output_relative']; summary = folder/'summary.json'
        if summary.exists():
            row = json.loads(summary.read_text())
            if row['job_sha256'] != job['job_sha256']:
                raise ValueError('Cached result belongs to a different evaluation plan')
            results.append(row); continue
        a, b = job['candidate'], job['opponent']
        metadata = dict(job_index=job['index'], job_sha256=job['job_sha256'],
            model_label=a['model_label'], track=a['track'], replicate=a['replicate'], stage=a.get('stage',0),
            submission_quality=a.get('submission_quality','frozen_reference'), opponent_label=b['label'])
        if job.get('invalid_submission'):
            row = dict(a=a['label'], b=b['label'], seat=job['seat'], seed=job['seed'],
                map_type=job['map_type'], phase=job['phase'], status='invalid_submission',
                reason='compile_failure', result='loss', winner=3-job['seat'], actions=0,
                audit=dict(passed=True, virtual=True, actions=0))
        else:
            a_bot = bot(a)
            try:
                b_bot = bot(b)
            except BaseException:
                a_bot.close(); raise
            trace = folder/uuid.uuid4().hex/'game.jsonl'
            row = play_game_streamed(a_bot, b_bot, job['seed'], job['seat'], budget, trace,
                phase=job['phase'], labels=[a['label'], b['label']], map_type=job['map_type'], cap_policy='engine_draw', evaluation=job.get('evaluation',cfg), initial_snapshot=job.get('initial_snapshot'))
            audited = audit_trace_streamed(row, trace)
            if job['phase']=='tactical':
                line=trace.open().readline()
                first=json.loads(line) if line else None
                chosen=next(a['command'] for a in first['observation']['legal_actions'] if a['action_id']==first['action_id']) if first else None
                row['tactical_pass']=chosen is not None and all(chosen.get(k)==v for k,v in job['expected_command'].items())
                row['tactical_name']=job['tactical_name']
                row['chosen_command']=chosen
            row.update(trace_relative=str(trace.relative_to(root)),
                       audit=dict(passed=True, virtual=False, actions=audited))
            if payload.get('compress_traces'):
                import gzip
                import shutil
                compressed = trace.with_suffix(trace.suffix+'.gz')
                with trace.open('rb') as source, compressed.open('wb') as target:
                    with gzip.GzipFile(fileobj=target,mode='wb',compresslevel=1,mtime=0) as output:
                        shutil.copyfileobj(source,output)
                trace.unlink()
                row.update(trace_relative=str(compressed.relative_to(root)),trace_compression='gzip')
        row.update(metadata)
        write_json(summary, row)
        results.append(row)
    return results


if __name__ == '__main__':
    print('SKIRMISH_RESULT='+json.dumps(evaluate(json.load(sys.stdin))))
