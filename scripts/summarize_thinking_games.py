"""Describe matched real-game outcomes without treating censored games as draws."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys


def describe(root):
    root = Path(root)
    protocol = json.loads((root/'protocol.json').read_text())
    expected = {j['id']:j for j in protocol['jobs']}
    summary = {}
    for mode in ['off','on']:
        folder = root/mode
        report = json.loads((folder/'report.json').read_text())
        games = report['games']
        if len(games)!=len(expected) or {g['id'] for g in games}!=expected.keys():
            raise ValueError('missing or duplicate game results')
        for g in games:
            if any(g[k]!=expected[g['id']][k] for k in ['seed','board_hash','seat','opponent']):
                raise ValueError('matched job changed')
        traces = {}
        for g in games:
            rows = list(map(json.loads,gzip.open(folder/'games'/(g['id']+'.jsonl.gz'),'rt')))
            decisions = [r for r in rows if r.get('policy')=='model']
            executed = [r for r in decisions if 'selected_command' in r]
            gens = [a['generation'] for r in decisions for a in r.get('diagnostic',r.get('attempts',[]))]
            counts = Counter(str(r['selected_command']['kind']) for r in executed)
            observations = [r['observation'] for r in rows if 'observation' in r]
            after = [o for r in rows for o in r.get('after',[]) if o['player']==g['seat']]
            first = next((o for o in observations if o['player']==g['seat']),None)
            last = after[-1] if after else first
            def state_stats(o):
                if o is None:return None
                v=o['visible_state']
                return dict(cities=sum(c['owner']==g['seat'] for c in v['cities']),
                    units=sum(u['owner']==g['seat'] for u in v['units']),
                    explored=sum(t['known'] for t in v['tiles']),stars=v['stars'],round=v['turn'])
            own_observations = [o for o in observations+after if o['player']==g['seat']]
            states = [state_stats(o) for o in own_observations]
            errors = Counter(a['error'] for r in decisions for a in r.get('diagnostic',r.get('attempts',[])) if 'error' in a)
            traces[g['id']] = dict(model_calls=len(gens),model_actions=len(executed),action_families=dict(counts),
                response_errors=dict(errors),
                input_tokens=sum(len(x['prompt_token_ids']) for x in gens),
                output_tokens=sum(len(x['response_token_ids']) for x in gens),
                responses_at_cap=sum(len(x['response_token_ids'])>=protocol['max_new_tokens'] for x in gens),
                peak_cities=max((s['cities'] for s in states),default=0),
                initial=state_stats(first),final=state_stats(last))
        timing = list(map(json.loads,(folder/'generation-timing.jsonl').read_text().splitlines()))
        outputs = [o for b in timing for o in b['outputs']]
        if sum(t['output_tokens'] for t in traces.values()) != sum(o['output_tokens'] for o in outputs):
            raise ValueError('trace and inference token counts disagree')
        request_seconds = [o['request_seconds'] for o in outputs if o['request_seconds'] is not None]
        runtime = json.loads((folder/'runtime.json').read_text())
        if runtime['revision']!=protocol['revision'] or runtime['adapter'] is not None or runtime['thinking']!=(mode=='on'):
            raise ValueError('model differs')
        summary[mode] = dict(allocated=len(games),
            native_wins=sum(g['status']=='native' and g['reward']>0 for g in games),
            native_losses=sum(g['status']=='native' and g['reward']<0 for g in games),
            forfeits=sum(g['status']=='forfeit' for g in games),
            incomplete=sum(g['status']=='budget_incomplete' for g in games),
            truncated=sum(g['status']=='truncated' for g in games),
            status_counts=dict(Counter(g['status'] for g in games)),
            finish_reasons=dict(Counter(o['finish_reason'] for o in outputs)),
            valid_actions=sum(g['first_attempt_valid'] for g in games),
            attempts=sum(g['attempts'] for g in games),
            total_model_calls=len(outputs),input_tokens=sum(o['input_tokens'] for o in outputs),
            output_tokens=sum(o['output_tokens'] for o in outputs),
            mean_output_tokens=statistics.mean(o['output_tokens'] for o in outputs) if outputs else None,
            mean_request_seconds=statistics.mean(request_seconds) if request_seconds else None,
            median_request_seconds=statistics.median(request_seconds) if request_seconds else None,
            model_batch_seconds=sum(b['seconds'] for b in timing),elapsed_seconds=report['elapsed_seconds'],
            runtime=runtime,execution=json.loads((folder/'execution.json').read_text()),games=games,trace_metrics=traces)
    for key in ['revision','tokenizer_hash','chat_template_hash','device','packages']:
        if summary['off']['runtime'][key]!=summary['on']['runtime'][key]:
            raise ValueError('runtime mismatch: '+key)
    result = dict(schema='qwen35-thinking-gameplay-result-v1',modes=summary,limitations=protocol['limitations'],
                  protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest())
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for mode, s in summary.items():
        print(mode,json.dumps({k:v for k,v in s.items() if k not in ['runtime','games','trace_metrics','execution']}))
    return result


if __name__=='__main__':
    describe(sys.argv[1])
