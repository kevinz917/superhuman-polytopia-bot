#!/usr/bin/env python3
"""Parallel source scanning with exactly the original corpus.sample selection.

Merge results in source-file order, so duplicates, ties and output ordering are
unchanged. Model inputs and labels still use the frozen corpus/data modules.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip,hashlib,json,multiprocessing,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from polytopia_bot.learning.corpus import phase
from polytopia_bot.learning.data import example
from polytopia_bot.learning.interface import canonical


def scan(name):
    path=Path(name);sha=hashlib.sha256(path.read_bytes()).hexdigest();rows=[]
    for index,line in enumerate(gzip.open(path,'rt')):
        row=json.loads(line)['record']
        if row is None:continue
        if row['source']!='teacher':raise ValueError('real teacher source required')
        _,_,key=example(row)
        rows.append((canonical(row['selected_command']),dict(file=str(path),line=index,input_hash=key,
            episode_id=row['episode_id'],map_hash=row['map_hash'],decision_index=row['decision_index'],
            phase=phase(row),family=row['selected_command']['kind'])))
    return str(path),sha,rows


def sample(directories,output,count=50000,workers=8):
    if count<1 or not 1<=workers<=8:raise ValueError('positive count and 1..8 workers required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    paths=[]
    for directory in map(Path,directories):
        if directory.joinpath('invalidation.json').exists() or not json.loads((directory/'report.json').read_text())['complete']:
            raise ValueError('completed valid corpus required')
        paths.extend(str(p) for p in sorted(directory.glob('*.jsonl.gz')))
    groups=defaultdict(list);seen={};files={};duplicates=0
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for completed,(name,sha,rows) in enumerate(pool.map(scan,paths),1):
            files[name]=sha
            for target,row in rows:
                key=row['input_hash']
                if key in seen:
                    if seen[key]!=target:raise ValueError('conflicting labels for identical public input')
                    duplicates+=1;continue
                seen[key]=target;groups[(row['phase'],row['family'])].append(row)
            if completed%20==0:
                print(json.dumps(dict(files_scanned=completed,allocated_files=len(paths),unique_examples=len(seen))),flush=True)
    for group in groups.values():group.sort(key=lambda r:r['input_hash'],reverse=True)
    selected=[];ends=0
    while len(selected)<count:
        changed=False
        for group in sorted(groups):
            if not groups[group] or (group[1]==10 and ends>=int(count*.20)):continue
            selected.append(groups[group].pop());ends+=group[1]==10;changed=True
            if len(selected)==count:break
        if not changed:raise ValueError('insufficient unique examples under end-turn cap')
    chosen={(r['file'],r['line']) for r in selected};written=0
    with (root/'records.jsonl').open('x') as stream:
        for completed,path in enumerate(files,1):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=files[path]:raise ValueError('source changed during selection')
            for index,line in enumerate(gzip.open(path,'rt')):
                if (path,index) in chosen:
                    stream.write(canonical(json.loads(line)['record'])+'\n');written+=1
            if completed%100==0:print(json.dumps(dict(files_written=completed,records_written=written)),flush=True)
    assert written==count
    report=dict(schema='semantic-sample-v1',count=count,source_files=files,unique_available=len(seen),duplicates=duplicates,
        end_turn_fraction_cap=.20,maps=len({r['map_hash'] for r in selected}),episodes=len({r['episode_id'] for r in selected}),
        phases=dict(Counter(r['phase'] for r in selected)),families=dict(Counter(str(r['family']) for r in selected)),
        selection='Round-robin phase/action-family strata; lowest semantic input hashes first; no outcome filtering',
        selected=selected,records_sha256=hashlib.sha256((root/'records.jsonl').read_bytes()).hexdigest())
    (root/'sampling.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('selected','source_files')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=50000);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();sample(a.input,a.output,a.count,a.workers)
