"""Aggregate unrestricted games and paired-map uncertainty, including failures."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys


def summarize(root):
    root=Path(root)
    p=json.loads((root/'protocol.json').read_text())
    expected={j['id']:j for j in p['jobs']}
    result=dict(schema='qwen35-unrestricted-summary-v1',modes={},
                protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest())
    for mode in ['off','on']:
        folder=root/mode
        report=json.loads((folder/'report.json').read_text())
        runtime=json.loads((folder/'runtime.json').read_text())
        games=report['games']
        assert len(games)==len(expected) and {g['id'] for g in games}==expected.keys()
        assert runtime['revision']==p['revision'] and runtime['adapter'] is None
        assert runtime['thinking']==(mode=='on') and runtime['max_tokens'] is None
        assert runtime['thinking_token_budget'] is None and runtime['max_model_len']==262144
        traces={}
        for g in games:
            assert all(g[k]==expected[g['id']][k] for k in ['seed','board_hash','seat','opponent','rng_index'])
            actions=Counter();errors=Counter();token_count=0;calls=0
            with gzip.open(folder/'games'/(g['id']+'.jsonl.gz'),'rt') as source:
                for line in source:
                    r=json.loads(line)
                    if r.get('policy')!='model':continue
                    if 'selected_command' in r:actions[r['selected_command']['kind']]+=1
                    for a in r.get('diagnostic',r.get('attempts',[])):
                        calls+=1;token_count+=len(a['generation']['response_token_ids'])
                        if 'error' in a:errors[a['error']]+=1
            traces[g['id']]=dict(action_families=dict(actions),errors=dict(errors),calls=calls,output_tokens=token_count)
        timing=[json.loads(l) for l in (folder/'generation-timing.jsonl').read_text().splitlines()]
        assert sum(x['output_tokens'] for x in traces.values())==sum(x['output_tokens'] for x in timing)
        for x in timing:
            assert x['max_tokens'] is None and x['thinking_token_budget'] is None
            assert x['seed']==p['seed']+expected[x['id']]['rng_index']*100000+x['decision']
        s=dict(allocated=len(games),status_counts=dict(Counter(g['status'] for g in games)),
            native_wins=sum(g['status']=='native' and g['reward']>0 for g in games),
            native_losses=sum(g['status']=='native' and g['reward']<0 for g in games),
            forfeits=sum(g['status']=='forfeit' for g in games),
            valid_actions=sum(g['first_attempt_valid'] for g in games),calls=len(timing),
            output_tokens=sum(x['output_tokens'] for x in timing),
            mean_output_tokens=statistics.mean(x['output_tokens'] for x in timing) if timing else None,
            max_output_tokens=max((x['output_tokens'] for x in timing),default=0),
            mean_request_seconds=statistics.mean(x['request_seconds'] for x in timing) if timing else None,
            median_request_seconds=statistics.median(x['request_seconds'] for x in timing) if timing else None,
            finish_reasons=dict(Counter(x['finish_reason'] for x in timing)),
            elapsed_seconds=report['elapsed_seconds'],runtime=runtime,games=games,traces=traces)
        result['modes'][mode]=s
    for k in ['revision','device','tokenizer_hash','chat_template_hash','packages','max_model_len']:
        assert result['modes']['on']['runtime'][k]==result['modes']['off']['runtime'][k]
    for k in ['serving','constrain_actions','harness']:
        assert result['modes']['on']['runtime'].get(k)==result['modes']['off']['runtime'].get(k)
    by_mode={m:{g['id']:g for g in s['games']} for m,s in result['modes'].items()}
    # Forfeits are operational failures; truncations/context/infrastructure are unobserved wins.
    all_resolved=all(g['status'] in ('native','forfeit') for s in result['modes'].values() for g in s['games'])
    if all_resolved:
        map_differences=[]
        for seed in sorted({j['seed'] for j in p['jobs']}):
            ids=[j['id'] for j in p['jobs'] if j['seed']==seed]
            map_differences.append(statistics.mean(
                int(by_mode['on'][i]['status']=='native' and by_mode['on'][i]['reward']>0)
                -int(by_mode['off'][i]['status']=='native' and by_mode['off'][i]['reward']>0) for i in ids))
        rng=random.Random(1701)
        boot=sorted(statistics.mean(rng.choices(map_differences,k=len(map_differences))) for _ in range(20000))
        result['paired_win_difference']=dict(on_minus_off=statistics.mean(map_differences),
            map_cluster_bootstrap_95_percent=[boot[499],boot[19499]],maps=len(map_differences),
            note='Exploratory bootstrap conditional on these maps and one sampling sequence; forfeits count as operational non-wins.')
    else:
        result['paired_win_difference']=None
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for m,s in result['modes'].items():
        print(m,json.dumps({k:v for k,v in s.items() if k not in ['runtime','games','traces']}))
    print('paired',result['paired_win_difference'])


if __name__=='__main__':summarize(sys.argv[1])
