#!/usr/bin/env python3
"""Sample distinct teacher/model-collected quotas for the single repair pass."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip,hashlib,json,multiprocessing,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.interface import canonical
from sample_sft_parallel import scan


def capacity(groups):
    ends=sum(len(rows) for key,rows in groups.items() if key[1]==10)
    other=sum(len(rows) for key,rows in groups.items() if key[1]!=10)
    return other+min(ends,other//4)


def choose(groups,count):
    groups={key:sorted(rows,key=lambda r:r['input_hash'],reverse=True) for key,rows in groups.items()}
    selected=[];ends=0
    while len(selected)<count:
        changed=False
        for key in sorted(groups):
            if not groups[key] or (key[1]==10 and ends>=int(count*.20)):continue
            selected.append(groups[key].pop());ends+=key[1]==10;changed=True
            if len(selected)==count:break
        if not changed:raise ValueError('source quota cannot be filled under the end-turn cap')
    return selected


def sample(teacher_dirs,relabel_dir,output,count,workers=8):
    if count not in (10000,50000) or not 1<=workers<=8:raise ValueError('registered repaired smoke/main sizes only')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    paths=[];sources={};provenance={}
    for directory,tag in [(Path(p),'teacher_trajectory') for p in teacher_dirs]+[(Path(relabel_dir),'smoke_on_policy_relabel')]:
        receipt=directory/'report.json';report=json.loads(receipt.read_text())
        if directory.joinpath('invalidation.json').exists() or not report['complete']:raise ValueError('complete valid collection required')
        if tag=='smoke_on_policy_relabel' and (report.get('schema')!='relabeled-training-rollouts-v1' or report.get('repair_pass')!=1):
            raise ValueError('registered model-collected teacher labels required')
        provenance[str(directory)]=dict(collection_source=tag,report_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
        if tag=='smoke_on_policy_relabel':
            provenance[str(directory)]['sidecar_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((directory/'provenance').glob('*.json'))}
            if not provenance[str(directory)]['sidecar_sha256']:raise ValueError('relabel source tags missing')
        for path in sorted(directory.glob('*.jsonl.gz')):paths.append(str(path));sources[str(path)]=tag
    seen={};files={};duplicates=Counter();groups=defaultdict(lambda:defaultdict(list))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for completed,(path,sha,rows) in enumerate(pool.map(scan,paths),1):
            files[path]=sha;tag=sources[path]
            for target,row in rows:
                key=row['input_hash']
                if key in seen:
                    if seen[key]!=target:raise ValueError('conflicting targets for identical public inputs')
                    duplicates[tag]+=1;continue
                seen[key]=target;row['collection_source']=tag
                groups[tag][(row['phase'],row['family'])].append(row)
            if completed%50==0:print(json.dumps(dict(scanned=completed,allocated=len(paths))),flush=True)
    available=capacity(groups['smoke_on_policy_relabel'])
    relabel_count=2000 if count==10000 else min(10000,available)
    if available<relabel_count:raise ValueError('insufficient unique repair labels; do not duplicate or relax the quota')
    # Each source's 20% end-turn cap also implies the combined 20% cap.
    selected=choose(groups['teacher_trajectory'],count-relabel_count)+choose(groups['smoke_on_policy_relabel'],relabel_count)
    chosen={(r['file'],r['line']) for r in selected};written=0
    with (root/'records.jsonl').open('x') as stream:
        for path,sha in files.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError('source changed')
            for index,line in enumerate(gzip.open(path,'rt')):
                if (path,index) in chosen:stream.write(canonical(json.loads(line)['record'])+'\n');written+=1
    assert written==count and relabel_count<=count*.20
    report=dict(schema='semantic-sample-v1',count=count,source_files=files,source_provenance=provenance,
        sampling_implementation_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),Path(__file__).with_name('sample_sft_parallel.py'))},
        unique_available=len(seen),duplicates=sum(duplicates.values()),duplicates_by_source=dict(duplicates),
        end_turn_fraction_cap=.20,maps=len({r['map_hash'] for r in selected}),episodes=len({r['episode_id'] for r in selected}),
        phases=dict(Counter(r['phase'] for r in selected)),families=dict(Counter(str(r['family']) for r in selected)),
        collection_sources=dict(Counter(r['collection_source'] for r in selected)),repair_pass=1,
        relabel_fraction=relabel_count/count,
        selection='Separate fixed source quotas; phase/action-family round robin, lowest semantic hashes, 20% end-turn cap in each source. Teacher sources merged first; no outcome filtering.',
        selected=selected,records_sha256=hashlib.sha256((root/'records.jsonl').read_bytes()).hexdigest())
    (root/'sampling.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('selected','source_files','source_provenance')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--teacher',nargs='+',required=True)
    p.add_argument('--relabel',required=True);p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,choices=[10000,50000],required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();sample(a.teacher,a.relabel,a.output,a.count,a.workers)
