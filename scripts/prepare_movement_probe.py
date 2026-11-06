"""Freeze a 128-position movement overfit probe without changing prompts or labels."""
import gzip, hashlib, json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.data import CorpusAudit
from polytopia_bot.learning.interface import canonical,digest
from polytopia_bot.learning.movement_diagnostic import score_choice
from polytopia_bot.learning.movement_probe import select_positions,stratum
from polytopia_bot.learning.training import file_hash
from polytopia_bot.learning.freeze import verify_prompt


def main(run_id):
    root=ROOT/'runs'/run_id;root.mkdir(exist_ok=False)
    previous=ROOT/'runs/movement-epoch2-v1';old=json.loads((previous/'protocol.json').read_text())
    parent=json.loads((previous/'training-run.json').read_text())
    if parent['status']!='completed':raise ValueError('completed parent required')
    for name,key in [('positions.jsonl.gz','positions_sha256'),('scoring.jsonl.gz','scoring_sha256')]:
        if file_hash(previous/name)!=old[key]:raise ValueError('diagnostic inputs changed')
    metadata=list(map(json.loads,gzip.open(previous/'scoring.jsonl.gz','rt')))
    strict=[r for r in metadata if score_choice(r['scores'],r['teacher_command'],None)['movement_required_for_top_score']]
    eligible=[r for r in strict if r['split']=='train' and r['source']=='teacher_trajectory']
    selected=select_positions(eligible,128,tag='movement-overfit-training-v1')
    probe_maps={r['board_hash'] for r in selected}
    comparison=select_positions(eligible,128,tag='movement-overfit-parent-training-control-v1',excluded_maps=probe_maps)
    held=[r for r in strict if r['split']=='held_out']
    config={**parent['config'],'epochs':20,'effective_batch_size':8}
    strategy=verify_prompt(parent['data_manifest']['prompt_manifest'],config)
    prepared=root/'prepared';prepared.mkdir()
    parent_prepared=ROOT/'runs/sft0/prepared-efficient-smoke-v1'
    manifest=json.loads((parent_prepared/'manifest.json').read_text())
    if manifest!=parent['data_manifest']:raise ValueError('parent dataset differs')
    raw=ROOT/'runs/sft0/repaired-smoke-sample-v2/records.jsonl'
    if file_hash(raw)!=manifest['source_sha256'] or file_hash(parent_prepared/'tokens.jsonl')!=manifest['tokens_sha256']:
        raise ValueError('parent raw/tokens changed')
    selected_by_index={int(r['position_id'].split('-')[1]):r for r in selected}
    audit=CorpusAudit(split_manifest=manifest['split_manifest'],strategy=strategy)
    lengths=[];supervised=0
    with raw.open() as raws,(parent_prepared/'examples.jsonl').open() as examples,(parent_prepared/'tokens.jsonl').open() as tokens, (prepared/'records.jsonl').open('x') as out_raw,(prepared/'examples.jsonl').open('x') as out_examples,(prepared/'tokens.jsonl').open('x') as out_tokens:
        for index,(raw_line,example_line,token_line) in enumerate(zip(raws,examples,tokens,strict=True)):
            if index not in selected_by_index:continue
            row=json.loads(raw_line);example=json.loads(example_line);encoded=json.loads(token_line);meta=selected_by_index[index]
            accepted=audit.accept(row)
            if accepted is None:raise ValueError('duplicate probe state')
            _,messages,response=accepted
            if row['selected_command']!=meta['teacher_command'] or row['map_hash']!=meta['board_hash'] or row['episode_id']!=meta['episode_id']:
                raise ValueError('selected raw record differs')
            if messages!=example['messages'] or response!=example['response'] or digest(messages)!=meta['prompt_digest']:
                raise ValueError('prompt or target changed')
            out_raw.write(raw_line);out_examples.write(example_line);out_tokens.write(token_line)
            lengths.append(len(encoded['input_ids']));supervised+=sum(x!=-100 for x in encoded['labels'])
    if len(lengths)!=128 or audit.report()['unique_records']!=128:raise ValueError('incomplete subset')
    subset={**manifest,'config':config,'config_hash':digest(config),'audit':audit.report(),
        'source_sha256':file_hash(prepared/'records.jsonl'),'tokens_sha256':file_hash(prepared/'tokens.jsonl'),
        'max_tokens':max(lengths),'processed_tokens':sum(lengths),'supervised_tokens':supervised,
        'subset_provenance':dict(parent_prepared='/workspace/runs/sft0/prepared-efficient-smoke-v1',parent_tokens_sha256=manifest['tokens_sha256'],
                                 parent_indices=sorted(selected_by_index),parent_manifest_sha256=file_hash(parent_prepared/'manifest.json'))}
    (prepared/'manifest.json').write_text(json.dumps(subset,indent=2)+'\n')
    positions={r['position_id']:r for r in map(json.loads,gzip.open(previous/'positions.jsonl.gz','rt'))}
    controls=[]
    for kind in range(2,11):
        pool=[positions[k] for k in old['control_ids'] if positions[k]['teacher_command']['kind']==kind]
        controls.extend(sorted(pool,key=lambda r:hashlib.sha256(('movement-probe-control-v1:'+r['position_id']).encode()).hexdigest())[:16])
    scoring=[{**r,'probe_group':group} for group,rows in [('probe_train',selected),('parent_train_unfocused',comparison),('held_out',held)] for r in rows]
    rows=[positions[r['position_id']] for r in scoring]+controls
    rows.sort(key=lambda r:hashlib.sha256(('movement-probe-eval-order-v1:'+r['position_id']).encode()).hexdigest())
    if len(rows)!=858 or len({r['position_id'] for r in rows})!=858:raise ValueError('evaluation coverage differs')
    if probe_maps & {r['board_hash'] for r in comparison+held}:raise ValueError('probe map overlap')
    for name,content in [('positions.jsonl.gz',rows),('scoring.jsonl.gz',scoring)]:
        with gzip.open(root/name,'xt') as f:
            for row in content:f.write(canonical(row)+'\n')
    training={k:v for k,v in old['training'].items() if k!='continuation'}
    training.update(run_id=run_id+'/training',prepared='/workspace/runs/'+run_id+'/prepared',config=config,stage='fit',seconds_limit=1800,
        probe=dict(parent_run='/workspace/runs/movement-epoch2-v1/training/model/run.json',adapter='/workspace/runs/movement-epoch2-v1/training/model/adapter-final',
                   adapter_files=parent['adapter_files'],optimizer_state='fresh',checkpoint_steps=[80,160,320]))
    upload_files=['positions.jsonl.gz','scoring.jsonl.gz',*[str(p.relative_to(root)) for p in prepared.iterdir()]]
    protocol=dict(schema='movement-only-overfit-probe-v1',run_id=run_id,training=training,config=config,prompt_manifest=old['prompt_manifest'],
        positions_sha256=file_hash(root/'positions.jsonl.gz'),scoring_sha256=file_hash(root/'scoring.jsonl.gz'),
        upload_files=['protocol.json',*upload_files],uploads={name:file_hash(root/name) for name in upload_files},
        checkpoint_steps=[0,80,160,320],checkpoint_epochs={'0':0,'80':5,'160':10,'320':20},evaluation_count=858,
        control_ids=[r['position_id'] for r in controls],batch_size=64,evaluation_seconds_limit=900,
        training_selection='128 ordinary teacher movement-required positions, phase/catalog round-robin, unique maps, SHA256 ordering; no prediction-based selection',
        selection_counts={group:dict(Counter('|'.join(stratum(r)) for r in rows)) for group,rows in [('probe_train',selected),('parent_train_unfocused',comparison),('held_out',held)]},
        fit_target=dict(tie_aware=.95,exact=.90),
        interpretation='Memorization diagnostic: parent already saw the unfocused training controls. Only held_out uses maps never trained on. Balanced probe sampling is not population prevalence.',
        optimizer='Fresh AdamW; 20 focused passes, effective batch 8, LR 1e-4, 10-step warmup then cosine decay over 320 updates.',
        scope='No prompt/representation/label changes, RL, gameplay, final-test access, or automatic promotion. Fixed bounded 20-pass experiment; fit failure is a result, not an excuse to extend it.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(run_id=run_id,train_count=len(selected),evaluation_count=len(rows),selection=protocol['selection_counts'],max_tokens=max(lengths))))


if __name__=='__main__':main(sys.argv[1])
