#!/usr/bin/env python3
"""Compare a fixed hash sample of live model decisions with the frozen teacher.

Validation diagnostics only: no model calls, no training data, no policy edits.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.data import observation_from_dict
from polytopia_bot.learning.interface import digest,canonical
from polytopia_bot.learning.teacher import ScriptedTeacher


def diagnose(source,output,count=500):
    source=ROOT/source
    protocol=json.loads((source/'protocol.json').read_text())
    report_path=source/'combined/games/report.json';report=json.loads(report_path.read_text())
    if protocol.get('final_test') or not report['complete']:
        raise ValueError('complete validation games only')
    teacher=ScriptedTeacher()
    qualified=json.loads((ROOT/'runs/sft0/qualified-teacher-v1.json').read_text())
    if teacher.teacher_hash!=qualified['teacher_hash']:
        raise ValueError('frozen teacher changed')
    candidates=[];traces={};families=Counter()
    for game in report['games']:
        paths=list(source.glob('*shard*/games/'+game['id']+'.jsonl.gz'))
        if len(paths)!=1:raise ValueError('unique game trace required')
        path=paths[0];traces[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for index,line in enumerate(gzip.open(path,'rt')):
            row=json.loads(line)
            if row.get('policy')!='model' or 'selected_command' not in row:continue
            families[str(row['selected_command']['kind'])]+=1
            key=digest(['live-teacher-diagnostic-v1',game['id'],index])
            candidates.append((key,str(path),index))
    chosen={(path,index) for _,path,index in sorted(candidates)[:count]}
    rows=[]
    for relative in traces:
        path=ROOT/relative
        for index,line in enumerate(gzip.open(path,'rt')):
            if (str(path),index) not in chosen:continue
            row=json.loads(line);obs=observation_from_dict(row['observation']);choice=teacher.select(obs)
            selected=row['selected_command'];label=dict(choice.action.command)
            rows.append(dict(episode_id=path.name.removesuffix('.jsonl.gz'),line=index,
                public_input_digest=digest(row['observation']),turn=obs.visible_state['turn'],
                model_command=selected,teacher_command=label,agreement=selected==label,
                model_ended_when_teacher_did_not=selected['kind']==10 and label['kind']!=10))
    result=dict(schema='live-teacher-diagnostic-v1',teacher_hash=teacher.teacher_hash,
        eligible_model_decisions=len(candidates),sampled=len(rows),
        selection='Lowest hashes of fixed diagnostic salt, episode ID and trace line; no outcome filtering.',
        model_action_families=dict(families),teacher_agreement=sum(r['agreement'] for r in rows)/max(1,len(rows)),
        model_ended_when_teacher_did_not=sum(r['model_ended_when_teacher_did_not'] for r in rows),
        disagreement_pairs=dict(Counter(f'{r["model_command"]["kind"]}->{r["teacher_command"]["kind"]}' for r in rows if not r['agreement'])),
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),trace_sha256=traces,rows=rows,
        limitation='A heuristic disagreement is not proof of a bad move. Validation states remain diagnostic and cannot enter training.')
    with (ROOT/output).open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','trace_sha256')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=500)
    a=p.parse_args();diagnose(a.source,a.output,a.count)
