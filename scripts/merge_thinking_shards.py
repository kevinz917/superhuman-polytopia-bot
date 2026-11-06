"""Merge disjoint cloud game shards with exact coverage and runtime checks."""
import json
import os
from pathlib import Path

from polytopia_bot.learning.async_gameplay import atomic_json


def merge(root, *, final=False):
    root=Path(root)
    protocol=json.loads((root/'protocol.json').read_text())
    expected={j['id']:j for j in protocol['jobs']}
    for mode in ['off','on']:
        target=root/mode
        (target/'games').mkdir(parents=True,exist_ok=True)
        games=[];active=[];generating=[];runtimes=[];reports=[];elapsed=[]
        timings=[];allocated=set();worker_seconds=0.
        for shard in range(protocol['replicas_per_mode']):
            base=root/'shards'/str(shard)
            shard_protocol=json.loads((base/'protocol.json').read_text())
            ids={j['id'] for j in shard_protocol['jobs']}
            if allocated & ids:
                raise ValueError('overlapping shard jobs')
            allocated |= ids
            source=base/mode
            if (source/'runtime.json').exists():
                runtimes.append(json.loads((source/'runtime.json').read_text()))
            if (source/'games/progress.json').exists():
                progress=json.loads((source/'games/progress.json').read_text())
                active.extend(progress['active']);generating.extend(progress['generating'])
                elapsed.append(progress['elapsed_seconds'])
            if (source/'games/games.json').exists():
                rows=json.loads((source/'games/games.json').read_text())
                if any(g['id'] not in ids for g in rows):
                    raise ValueError('game in wrong shard')
                games.extend(rows)
            if final:
                report=json.loads((source/'report.json').read_text())
                if {g['id'] for g in report['games']} != ids or len(report['games']) != len(ids):
                    raise ValueError('incomplete shard report')
                reports.append(report)
                execution=json.loads((base/(mode+'-execution.json')).read_text())
                if execution['exit_code'] != 0:
                    raise ValueError('failed shard execution')
                worker_seconds+=execution['worker_seconds']
                for job_id in ids:
                    link=target/'games'/(job_id+'.jsonl.gz')
                    trace=source/'games'/link.name
                    if not trace.is_file():
                        raise ValueError('missing trace')
                    if not link.exists():
                        link.symlink_to(os.path.relpath(trace,link.parent))
                for line in (source/'generation-timing.jsonl').read_text().splitlines():
                    row=json.loads(line)
                    if row['id'] not in ids:
                        raise ValueError('timing in wrong shard')
                    timings.append(row)
        if allocated != expected.keys():
            raise ValueError('shard coverage does not match parent')
        if len({g['id'] for g in games}) != len(games):
            raise ValueError('duplicate game results')
        for game in games:
            if any(game[k] != expected[game['id']][k] for k in ['seed','seat','board_hash','opponent','rng_index']):
                raise ValueError('game identity changed')
        if runtimes:
            if any(r != runtimes[0] for r in runtimes[1:]):
                raise ValueError('worker runtime mismatch')
            atomic_json(target/'runtime.json',runtimes[0])
        atomic_json(target/'games/games.json',games)
        atomic_json(target/'games/progress.json',dict(allocated=len(expected),finished=len(games),
            active=active,generating=generating,elapsed_seconds=max(elapsed,default=0),
            replicas=protocol['replicas_per_mode']))
        if final:
            if len(games)!=len(expected) or len(runtimes)!=protocol['replicas_per_mode']:
                raise ValueError('incomplete final coverage')
            if len({(r['id'],r['decision']) for r in timings})!=len(timings):
                raise ValueError('duplicate generation timing')
            atomic_json(target/'report.json',dict(games=games,allocated_games=len(expected),
                finished_games=len(games),complete=all(r['complete'] for r in reports),
                elapsed_seconds=max(r['elapsed_seconds'] for r in reports),
                gpu_worker_seconds=worker_seconds,replicas=protocol['replicas_per_mode']))
            (target/'generation-timing.jsonl').write_text(''.join(json.dumps(r)+'\n'
                for r in sorted(timings,key=lambda r:(r['id'],r['decision']))))
