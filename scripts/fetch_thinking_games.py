"""Read persisted gameplay progress without launching or resuming any jobs."""
import json
from pathlib import Path
import sys
import statistics
from collections import Counter
import modal


def main(root):
    root=Path(root)
    volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
    result={}
    for mode in ['off','on']:
        result[mode]={}
        for name in ['games/progress.json','games/games.json','runtime.json','execution.json','report.json','generation-timing.jsonl']:
            try:data=b''.join(volume.read_file(root.name+'/'+mode+'/'+name))
            except FileNotFoundError:continue
            path=root/'live'/mode/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data)
            if name=='generation-timing.jsonl':
                rows=[json.loads(line) for line in data.decode().splitlines()]
                outputs=[o for r in rows for o in (r['outputs'] if 'outputs' in r else [r])]
                result[mode]['generation']=dict(calls=len(outputs),output_tokens=sum(o['output_tokens'] for o in outputs),
                    limit_finishes=sum(o['finish_reason']=='length' for o in outputs),
                    longest_response=max((o['output_tokens'] for o in outputs),default=0))
            elif name=='games/progress.json' and 'generating' in json.loads(data):
                p=json.loads(data); active=p['active']; generating=p['generating']
                result[mode]['progress']=dict(elapsed_seconds=p['elapsed_seconds'],finished=p['finished'],
                    active_games=len(active),generating=len(generating),
                    round_range=[min((g['round'] for g in active),default=None),max((g['round'] for g in active),default=None)],
                    longest_pending_output=max((g['output_tokens'] for g in generating),default=0))
            elif name=='games/games.json' or name=='report.json':
                p=json.loads(data); games=p if isinstance(p,list) else p['games']
                result[mode][name]=dict(finished=len(games),statuses=dict(Counter(g['status'] for g in games)),
                    native_wins=sum(g['status']=='native' and g['reward']>0 for g in games),
                    native_losses=sum(g['status']=='native' and g['reward']<0 for g in games))
            elif name!='runtime.json':result[mode][name]=json.loads(data)
            else:result[mode]['loaded']=True
    for name in ['collection.json','summary.json','result.md']:
        try:data=b''.join(volume.read_file(root.name+'/'+name))
        except FileNotFoundError:continue
        path=root/'live'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        if name=='collection.json':result['collection']=json.loads(data)
        else:result[name]=str(path)
    protocol=json.loads((root/'protocol.json').read_text())
    if protocol.get('replicas_per_mode'):
        result['replicas']={}
        for mode in ['off','on']:
            all_rows=[]
            for shard in range(protocol['replicas_per_mode']):
                key=f'{mode}-{shard}';entry={};result['replicas'][key]=entry
                base=f'shards/{shard}/{mode}'
                for name in ['runtime.json','games/progress.json','generation-timing.jsonl']:
                    try:data=b''.join(volume.read_file(root.name+'/'+base+'/'+name))
                    except FileNotFoundError:continue
                    path=root/'live'/base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                    if name=='runtime.json':entry['runtime']=json.loads(data)
                    elif name=='games/progress.json':
                        row=json.loads(data);entry['progress']=dict(finished=row['finished'],active=len(row['active']),
                            pending=len(row['generating']),elapsed_seconds=row['elapsed_seconds'])
                    else:
                        rows=[json.loads(l) for l in data.decode().splitlines()];all_rows.extend(rows)
                        entry['calls']=len(rows);entry['finish_reasons']=dict(Counter(r['finish_reason'] for r in rows))
            result[mode]['generation']=dict(calls=len(all_rows),output_tokens=sum(r['output_tokens'] for r in all_rows),
                finish_reasons=dict(Counter(r['finish_reason'] for r in all_rows)),
                thinking_closed=sum(r['thinking_closed'] for r in all_rows),
                median_first_token_seconds=statistics.median(r['first_token_seconds'] for r in all_rows) if all_rows else None,
                median_request_seconds=statistics.median(r['request_seconds'] for r in all_rows) if all_rows else None)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main(sys.argv[1])
