"""Freeze one extra SFT epoch and a common baseline/checkpoint evaluation set."""
import gzip,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.interface import canonical


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(run_id):
    root=ROOT/'runs'/run_id;root.mkdir(parents=True,exist_ok=False)
    old=ROOT/'runs/speed-v2/end-to-end-v1';diagnostic=ROOT/'runs/movement-diagnostic-v1'
    prior=json.loads((old/'protocol.json').read_text());run=json.loads((old/'smoke-execution.json').read_text())['run']
    d=json.loads((diagnostic/'protocol.json').read_text())
    if d['adapter_files']!=run['adapter_files']:raise ValueError('diagnostic checkpoint differs')
    for name,key in [('training-positions.jsonl.gz','positions_sha256'),('scoring.jsonl.gz','scoring_sha256')]:
        if sha(diagnostic/name)!=d[key]:raise ValueError('diagnostic input changed')
    rows=[json.loads(x) for x in gzip.open(diagnostic/'training-positions.jsonl.gz','rt')]
    metadata=[json.loads(x) for x in gzip.open(diagnostic/'scoring.jsonl.gz','rt')]
    movement_ids={m['position_id'] for m in metadata}
    controls={kind:[] for kind in range(2,11)}
    for filename in prior['positions']['shard_files']:
        if sha(ROOT/filename)!=prior['positions']['uploads'][filename]:raise ValueError('held-out inputs changed')
        for line in gzip.open(ROOT/filename,'rt'):
            position=json.loads(line)
            if position['teacher_command']['kind']==1:rows.append(position)
            else:controls[position['teacher_command']['kind']].append(position)
    control_ids=[]
    for kind,group in sorted(controls.items()):
        selected=sorted(group,key=lambda p:hashlib.sha256(('movement-continuation-control-v1:'+p['position_id']).encode()).hexdigest())[:32]
        if len(selected)!=32:raise ValueError('control coverage missing')
        rows.extend(selected);control_ids.extend(p['position_id'] for p in selected)
    if len(rows)!=2161 or len({r['position_id'] for r in rows})!=2161:raise ValueError('evaluation coverage differs')
    if set(p['position_id'] for p in rows)!=movement_ids|set(control_ids):raise ValueError('movement identity mismatch')
    rows.sort(key=lambda p:hashlib.sha256(('movement-continuation-eval-order-v1:'+p['position_id']).encode()).hexdigest())
    with gzip.open(root/'positions.jsonl.gz','xt') as f:
        for row in rows:f.write(canonical(row)+'\n')
    # Store only scoring metadata; every checkpoint supplies its own predictions.
    with gzip.open(root/'scoring.jsonl.gz','xt') as f:
        for row in metadata:f.write(canonical({k:v for k,v in row.items() if k!='prediction'})+'\n')
    train={**prior['smoke'],'run_id':run_id+'/training','seconds_limit':1800}
    train['continuation']=dict(parent_run='/workspace/runs/speed-v2/end-to-end-v1/smoke/model/run.json',
        adapter=d['adapter'],adapter_files=d['adapter_files'],epoch_offset=1,optimizer_state='fresh',checkpoint_steps=[40,79,118,157])
    protocol=dict(schema='movement-one-more-epoch-v1',run_id=run_id,training=train,config=d['config'],prompt_manifest=d['prompt_manifest'],
        positions_sha256=sha(root/'positions.jsonl.gz'),scoring_sha256=sha(root/'scoring.jsonl.gz'),
        checkpoint_steps=[0,40,79,118,157],evaluation_count=len(rows),train_movement_count=1298,held_out_movement_count=575,
        control_ids=control_ids,control_sampling='32 per non-movement teacher action family, fixed SHA256 order, no prediction-based selection',
        batch_size=64,evaluation_seconds_limit=1200,
        primary_metric='Held-out tie-aware agreement when all globally top teacher actions are movement; show full tie-aware and exact agreement on both splits alongside it.',
        baseline='Fresh evaluation of starting adapter using identical position order and batch size; previous diagnostic retained as historical comparison.',
        learning_schedule=dict(optimizer='fresh AdamW, no saved optimizer state in parent',peak_learning_rate=train['config']['learning_rate'],
            warmup_steps=5,cosine_steps=157,epoch_offset=1,shuffle_seed=18,global_batch_size=64,additional_examples=10000),
        scope='Exactly one extra epoch; evaluate baseline and four fixed checkpoints. No RL, data or prompt changes, full-game evaluation, final-test access, or automatic policy promotion.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print(json.dumps(dict(run_id=run_id,positions=len(rows),checkpoints=protocol['checkpoint_steps'])))


if __name__=='__main__':main(sys.argv[1])
