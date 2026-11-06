"""Durable metadata-only timing events. Never record prompts, commands or secrets."""
import json
from pathlib import Path
import time


class TimingLog:
    def __init__(self, path=None):
        self.path = Path(path) if path else None

    def event(self, kind, **fields):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as out:
                out.write(json.dumps(dict(event=kind, unix=time.time(), **fields), allow_nan=False)+'\n')
                out.flush()


def summarize_timing(path, cutoff=None):
    events = [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []
    totals = dict(api_seconds=0., tool_seconds=0., api_calls_started=0, api_calls_finished=0,
                  interrupted_api_calls=0, tool_calls_started=0, tool_calls_finished=0)
    pending = {}
    for e in events:
        kind=e['event']
        for prefix in ('api','tool'):
            if kind==prefix+'_start':
                totals[prefix+'_calls_started']+=1;pending[prefix,e['id']]=e['unix']
            elif kind==prefix+'_end':
                totals[prefix+'_calls_finished']+=1
                if prefix=='api' and e.get('outcome')=='interrupted':totals['interrupted_api_calls']+=1
                start=pending.pop((prefix,e['id']),None)
                if start is not None: totals[prefix+'_seconds']+=max(0,e['unix']-start)
    for (prefix,_),start in pending.items():
        if cutoff is not None: totals[prefix+'_seconds']+=max(0,cutoff-start)
        if prefix=='api':totals['interrupted_api_calls']+=1
    return totals
