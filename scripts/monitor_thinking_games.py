"""Read-only live watcher. Never launches, cancels, or changes model jobs."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo
import modal


def watch(root):
    root=Path(root)
    volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
    previous={}
    def read(name):
        try:data=b''.join(volume.read_file(root.name+'/'+name))
        except FileNotFoundError:return None
        local=root/'live'/name
        local.parent.mkdir(parents=True,exist_ok=True)
        local.write_bytes(data)
        return json.loads(data)
    with (root/'monitor.jsonl').open('a') as log:
        while True:
            try:
                row=dict(checked_at=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),modes={})
                for mode in ['off','on']:
                    p=read(mode+'/games/progress.json')
                    games=read(mode+'/games/games.json') or []
                    if p is None:
                        row['modes'][mode]=dict(status='initializing');continue
                    # Files are committed independently; a newly finished game can
                    # still appear in a slightly older progress snapshot.
                    finished_ids={g['id'] for g in games}
                    active=[g for g in p['active'] if g['id'] not in finished_ids]
                    pending=[g for g in p.get('generating',[]) if g['id'] not in finished_ids]
                    generated=sum(g['output_tokens'] for g in pending)
                    row['modes'][mode]=dict(finished=len(games),active=len(active),
                        native_wins=sum(g['status']=='native' and g['reward']>0 for g in games),
                        native_losses=sum(g['status']=='native' and g['reward']<0 for g in games),
                        statuses=dict(Counter(g['status'] for g in games)),
                        valid_actions=sum(g['first_attempt_valid'] for g in games+active),
                        elapsed_seconds=round(p['elapsed_seconds'],1),
                        round_range=[min((g['round'] for g in active),default=None),max((g['round'] for g in active),default=None)],
                        generating=len(pending),longest_pending_tokens=max((g['output_tokens'] for g in pending),default=0),
                        pending_tokens=generated)
                    before=previous.get(mode)
                    if before:
                        row['modes'][mode]['progress_seconds_delta']=round(p['elapsed_seconds']-before,1)
                    previous[mode]=p['elapsed_seconds']
                collection=read('collection.json')
                row['collection_status']=collection['status'] if collection else 'unknown'
                line=json.dumps(row)
                print(line,flush=True);log.write(line+'\n');log.flush()
                if row['collection_status'] in ('complete','completed_with_unresolved_games','failed'):
                    if row['collection_status']!='failed':
                        read('summary.json')
                        data=b''.join(volume.read_file(root.name+'/result.md'))
                        (root/'live/result.md').write_bytes(data)
                    return
            except Exception as exc:
                line=json.dumps(dict(monitor_error=f'{type(exc).__name__}: {exc}',at=time.time()))
                print(line,flush=True);log.write(line+'\n');log.flush()
            time.sleep(45)


if __name__=='__main__':watch(sys.argv[1])
