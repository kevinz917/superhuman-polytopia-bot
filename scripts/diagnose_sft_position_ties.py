#!/usr/bin/env python3
"""Tie-aware teacher diagnostic alongside, never replacing, exact agreement."""
import argparse
from collections import defaultdict
import gzip,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.data import observation_from_dict
from polytopia_bot.learning.teacher import ScriptedTeacher


def diagnose(base,smoke,output):
    b=json.loads((ROOT/base/'report.json').read_text());s=json.loads((ROOT/smoke/'report.json').read_text())
    if not b['complete'] or not s['complete'] or b['decisions']!=5000 or s['decisions']!=5000:
        raise ValueError('complete paired 5000-position reports required')
    reports={'base':{r['position_id']:r for r in b['results']},'smoke':{r['position_id']:r for r in s['results']}}
    allocation=json.loads((ROOT/smoke/'allocation.json').read_text());teacher=ScriptedTeacher()
    qualified=json.loads((ROOT/'runs/sft0/qualified-teacher-v1.json').read_text())
    if teacher.teacher_hash!=qualified['teacher_hash']:raise ValueError('teacher changed')
    rows=[]
    for name in allocation['shard_files']:
        path=ROOT/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=allocation['uploads'][name]:raise ValueError('position file changed')
        for line in gzip.open(path,'rt'):
            position=json.loads(line);choice=teacher.select(observation_from_dict(position['observation']))
            if dict(choice.action.command)!=position['teacher_command']:raise ValueError('teacher target no longer reproduces')
            best=choice.scores[0]['score'];top=[r['command'] for r in choice.scores if r['score']==best]
            rows.append(dict(position_id=position['position_id'],family=position['teacher_command']['kind'],top_ties=len(top),
                **{model:report[position['position_id']]['selected_command'] in top for model,report in reports.items()}))
            if len(rows)%1000==0:print(json.dumps(dict(checked=len(rows))),flush=True)
    if len(rows)!=5000 or len({r['position_id'] for r in rows})!=5000:raise ValueError('position coverage differs')
    grouped=defaultdict(list)
    for row in rows:grouped[str(row['family'])].append(row)
    summary=dict(schema='teacher-score-tie-diagnostic-v1',teacher_hash=teacher.teacher_hash,count=len(rows),
        tied_positions=sum(r['top_ties']>1 for r in rows),
        base_top_score_agreement=sum(r['base'] for r in rows)/len(rows),
        smoke_top_score_agreement=sum(r['smoke'] for r in rows)/len(rows),
        families={k:dict(count=len(v),tied=sum(r['top_ties']>1 for r in v),base=sum(r['base'] for r in v)/len(v),
                         smoke=sum(r['smoke'] for r in v)/len(v)) for k,v in grouped.items()},
        rows=rows,limitation='This post-hoc diagnostic accepts any equal-highest heuristic score. The frozen primary metric remains exact semantic action agreement; heuristic score is not optimal play.')
    with (ROOT/output).open('x') as stream:json.dump(summary,stream,indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','smoke','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();diagnose(a.base,a.smoke,a.output)
