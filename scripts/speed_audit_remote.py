"""Replay speed-v2 games on CPU and verify adapter integrity without downloading traces."""
import json,sys,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from audit_incomplete_sft_games import replay

def worker_init():
    import os
    if hasattr(os,'sched_getaffinity'):
        cpus=sorted(os.sched_getaffinity(0));index=multiprocessing.current_process()._identity[0]-1
        os.sched_setaffinity(0,{cpus[(index*2+i)%len(cpus)] for i in range(min(2,len(cpus)))})


def main(root):
    root=Path(root);results={}
    protocol=json.loads((root/'protocol.json').read_text());jobs={j['id']:j for j in protocol['jobs']}
    for kind in ['base','trained']:
        tasks=[]
        for file in (root/(kind+'-games')).glob('worker-*/games/*.result.json'):
            row=json.loads(file.read_text())
            if row['id'] not in jobs or any(row[k]!=v for k,v in jobs[row['id']].items()):raise ValueError('game differs from allocation')
            tasks.append((str(file.parent/(row['id']+'.jsonl.gz')),row))
        if len(tasks)!=50 or len({r['id'] for _,r in tasks})!=50:raise ValueError('full 50-game allocation missing')
        with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init) as pool:
            checks=list(pool.map(replay,tasks))
        results[kind]=dict(games=len(tasks),passed=all(r['passed'] for r in checks),
            decisions=sum(r['decisions'] for r in checks),model_decisions=sum(r['model_decisions'] for r in checks),checks=checks,
            statuses={s:sum(r['status']==s for _,r in tasks) for s in {r['status'] for _,r in tasks}})
    adapter=root/'smoke/model/adapter-final'
    from safetensors import safe_open
    import torch
    count=0;nonzero_b=0
    with safe_open(adapter/'adapter_model.safetensors',framework='pt',device='cpu') as f:
        for name in f.keys():
            tensor=f.get_tensor(name)
            if 'lora_' not in name or not torch.isfinite(tensor).all():raise ValueError('adapter integrity failed')
            count+=1
            if 'lora_B' in name and bool(tensor.any()):nonzero_b+=1
    results['adapter']=dict(tensors=count,nonzero_B=nonzero_b,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter.glob('adapter*')})
    expected=json.loads((root/'smoke/model/run.json').read_text())['adapter_files']
    if results['adapter']['files']!=expected or not nonzero_b:raise ValueError('adapter identity failed')
    if not all(results[k]['passed'] for k in ['base','trained']):raise ValueError('replay failed')
    (root/'audit.json').write_text(json.dumps(results,indent=2));return results

if __name__=='__main__':main(sys.argv[1])
