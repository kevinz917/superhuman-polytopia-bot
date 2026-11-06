"""Prepare and summarize an all-example train/held-out movement diagnostic."""
import argparse,gzip,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.data import observation_from_dict
from polytopia_bot.learning.corpus import phase
from polytopia_bot.learning.interface import digest,canonical
from polytopia_bot.learning.prompt_search import candidate_messages
from polytopia_bot.learning.teacher import ScriptedTeacher
from polytopia_bot.learning.movement_diagnostic import score_choice,summarize


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    prepared=ROOT/'runs/sft0/prepared-efficient-smoke-v1'
    records=ROOT/'runs/sft0/repaired-smoke-sample-v2/records.jsonl'
    previous=ROOT/'runs/speed-v2/end-to-end-v1'
    manifest=read(prepared/'manifest.json');run=read(previous/'smoke-execution.json')['run']
    if manifest!=run['data_manifest'] or sha(records)!=manifest['source_sha256']:
        raise ValueError('training data does not match the checkpoint')
    if sha(prepared/'tokens.jsonl')!=manifest['tokens_sha256']:
        raise ValueError('training tokens changed')
    teacher=ScriptedTeacher();qualified=read(ROOT/'runs/sft0/qualified-teacher-v1.json')
    if teacher.teacher_hash!=qualified['teacher_hash']:raise ValueError('teacher changed')
    allocation=read(ROOT/'runs/sft0/efficient-positions-allocation-v1.json')
    held_report=read(previous/'trained-positions-report.json')
    held_predictions={row['position_id']:row for row in held_report['results']}
    if not held_report['complete'] or len(held_predictions)!=5000:raise ValueError('held-out predictions incomplete')
    sampling=read(ROOT/'runs/sft0/repaired-smoke-sample-v2/sampling.json')
    provenance={(r['episode_id'],r['decision_index']):r['collection_source'] for r in sampling['selected']}
    training_maps=set(manifest['split_manifest']['train']);validation_maps=set(manifest['split_manifest']['validation'])
    if training_maps & validation_maps:raise ValueError('split overlap')
    metadata=[];train_positions=[]
    def add(position,split,source,prediction=None):
        obs=observation_from_dict(position['observation']);choice=teacher.select(obs)
        target=position['teacher_command']
        if target!=dict(choice.action.command):raise ValueError('teacher target no longer reproduces')
        if digest(candidate_messages(obs,''))!=position['prompt_digest']:raise ValueError('prompt mismatch')
        scores=[dict(command=s['command'],score=s['score']) for s in choice.scores]
        meta={k:position[k] for k in ['position_id','board_hash','episode_id','phase','teacher_command','prompt_digest']}
        meta.update(split=split,source=source,scores=scores)
        if prediction is not None:
            for key in ['board_hash','episode_id','teacher_command']:
                if prediction[key]!=position[key]:raise ValueError('saved prediction input mismatch')
            if prediction['input_tokens']!=position['prompt_tokens']:raise ValueError('held-out token count mismatch')
            meta['prediction']=prediction
        metadata.append(meta)
        if len(metadata)%200==0:print(json.dumps(dict(scored=len(metadata))),flush=True)
    with records.open() as raw,(prepared/'examples.jsonl').open() as examples:
        for index,(line,example_line) in enumerate(zip(raw,examples,strict=True)):
            row=json.loads(line);example=json.loads(example_line)
            for key in ['episode_id','decision_index','map_hash','selected_command']:
                if row[key]!=example[key]:raise ValueError('prepared examples differ from raw training data')
            if row['selected_command']['kind']!=1:continue
            if row['map_hash'] not in training_maps:raise ValueError('training map not in train split')
            p=dict(position_id=f'train-{index:05d}',board_hash=row['map_hash'],episode_id=row['episode_id'],
                   phase=phase(row),observation=row['observation'],teacher_command=row['selected_command'],
                   prompt_digest=digest(example['messages']))
            add(p,'train',provenance[(row['episode_id'],row['decision_index'])]);train_positions.append(p)
    for name in allocation['shard_files']:
        file=ROOT/name
        if sha(file)!=allocation['uploads'][name]:raise ValueError('held-out input hash mismatch')
        for line in gzip.open(file,'rt'):
            row=json.loads(line)
            if row['teacher_command']['kind']!=1:continue
            if row['board_hash'] not in validation_maps:raise ValueError('held-out map not in validation split')
            add(row,'held_out','teacher_trajectory',held_predictions[row['position_id']])
    if len(train_positions)!=1298 or sum(r['split']=='held_out' for r in metadata)!=575:
        raise ValueError('movement census differs from inspected dataset')
    if len({r['position_id'] for r in metadata})!=len(metadata):raise ValueError('duplicate position IDs')
    with gzip.open(output/'training-positions.jsonl.gz','xt') as stream:
        for row in train_positions:stream.write(canonical(row)+'\n')
    with gzip.open(output/'scoring.jsonl.gz','xt') as stream:
        for row in metadata:stream.write(canonical(row)+'\n')
    protocol=dict(schema='movement-train-held-out-diagnostic-v1',run_id=str(output.relative_to(ROOT/'runs')),
        config=manifest['config'],prompt_manifest=manifest['prompt_manifest'],teacher_hash=teacher.teacher_hash,
        adapter='/workspace/runs/speed-v2/end-to-end-v1/smoke/model/adapter-final',adapter_files=run['adapter_files'],
        positions_sha256=sha(output/'training-positions.jsonl.gz'),scoring_sha256=sha(output/'scoring.jsonl.gz'),
        train_count=1298,held_out_count=575,batch_size=64,seconds_limit=900,
        selection='Census of all movement teacher targets in the actual 10k training dataset and frozen 5k validation positions; no output-based selection.',
        sources={str(f.relative_to(ROOT)):sha(f) for f in [records,prepared/'manifest.json',prepared/'examples.jsonl',previous/'trained-positions-report.json',ROOT/'runs/sft0/repaired-smoke-sample-v2/sampling.json']},
        analysis='Exact and global-top-score agreement; teacher-trajectory and model-collected training sources separately; phase-standardized comparison and map-cluster bootstrap. Lower training accuracy does not alone distinguish undertraining from representation or label issues.',
        unchanged='No weight update, prompt change, relabeling, gameplay rollout, or final-test access.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(training=len(train_positions),held_out=575,training_sources=dict(Counter(r['source'] for r in metadata if r['split']=='train')))))


def bootstrap_difference(train,held,iterations=5000):
    import numpy as np
    rng=np.random.default_rng(17)
    def samples(rows):
        groups=defaultdict(list)
        for r in rows:groups[r['board_hash']].append(int(r['tie_aware']))
        totals=np.array([(sum(v),len(v)) for v in groups.values()])
        indices=rng.integers(0,len(totals),size=(iterations,len(totals)))
        sampled=totals[indices].sum(axis=1)
        return sampled[:,0]/sampled[:,1]
    delta=samples(train)-samples(held)
    return dict(method='independent board-cluster bootstrap; train minus held-out',iterations=iterations,seed=17,
                interval_95=np.quantile(delta,[.025,.975]).tolist())


def finish(output):
    output=Path(output);protocol=read(output/'protocol.json')
    if sha(output/'scoring.jsonl.gz')!=protocol['scoring_sha256']:raise ValueError('scoring inputs changed')
    execution=read(output/'execution.json');report=read(output/'training-report.json')
    if execution['exit_code'] or not report['complete'] or report['decisions']!=protocol['train_count']:
        raise ValueError('training predictions incomplete')
    predictions={row['position_id']:row for row in report['results']}
    if len(predictions)!=protocol['train_count']:raise ValueError('duplicate training predictions')
    rows=[]
    for line in gzip.open(output/'scoring.jsonl.gz','rt'):
        meta=json.loads(line);pred=predictions[meta['position_id']] if meta['split']=='train' else meta['prediction']
        for key in ['board_hash','episode_id','teacher_command']:
            if pred[key]!=meta[key]:raise ValueError('prediction does not match scoring input')
        result={k:v for k,v in meta.items() if k not in ('scores','prediction')}
        result.update(score_choice(meta['scores'],meta['teacher_command'],pred['selected_command']))
        result.update(first_attempt_valid=pred['first_attempt_valid'],input_tokens=pred['input_tokens'],selected_command=pred['selected_command'])
        if result['exact']!=pred['teacher_agreement']:raise ValueError('exact agreement mismatch')
        rows.append(result)
    groups={split:[r for r in rows if r['split']==split] for split in ['train','held_out']}
    teacher_train=[r for r in groups['train'] if r['source']=='teacher_trajectory']
    strict_groups={key:[r for r in group if r['movement_required_for_top_score']] for key,group in {**groups,'train_teacher_only':teacher_train}.items()}
    summaries={key:summarize(group) for key,group in groups.items()}
    by_phase={key:{phase:summarize([r for r in group if r['phase']==phase]) for phase in sorted({r['phase'] for r in rows})} for key,group in {**groups,'train_teacher_only':teacher_train}.items()}
    weights={phase:values['count']/len(groups['held_out']) for phase,values in by_phase['held_out'].items()}
    standardized={}
    for key,phases in by_phase.items():
        if all(phases[p]['count'] for p,w in weights.items() if w):
            standardized[key]=sum(w*phases[p]['tie_aware'] for p,w in weights.items() if w)
    result=dict(schema='movement-train-held-out-result-v1',counts={k:len(v) for k,v in groups.items()},
        summaries=summaries,training_by_source={source:summarize([r for r in groups['train'] if r['source']==source]) for source in sorted({r['source'] for r in groups['train']})},
        by_phase=by_phase,held_out_phase_weights=weights,phase_standardized_tie_aware=standardized,
        by_movement_requirement={split:{str(required):summarize([r for r in group if r['movement_required_for_top_score']==required]) for required in [True,False]} for split,group in {**groups,'train_teacher_only':teacher_train}.items()},
        movement_required_by_phase={key:{phase:summarize([r for r in group if r['phase']==phase]) for phase in sorted({r['phase'] for r in rows})} for key,group in strict_groups.items()},
        by_legal_catalog_size={split:{label:summarize([r for r in group if lower<=r['legal_count']<upper]) for label,lower,upper in [('under_64',0,64),('64_to_127',64,128),('128_plus',128,float('inf'))]} for split,group in groups.items()},
        by_target_ambiguity={split:{label:summarize([r for r in group if (r['top_count']==1)==single]) for label,single in [('single_best_action',True),('multiple_best_actions',False)]} for split,group in groups.items()},
        train_minus_held_out=summaries['train']['tie_aware']-summaries['held_out']['tie_aware'],
        gap_uncertainty=bootstrap_difference(groups['train'],groups['held_out']),
        teacher_only_gap_uncertainty=bootstrap_difference(teacher_train,groups['held_out']),
        movement_required_gap_uncertainty=bootstrap_difference(strict_groups['train'],strict_groups['held_out']),
        movement_required_teacher_only_gap_uncertainty=bootstrap_difference(strict_groups['train_teacher_only'],strict_groups['held_out']),
        inference_occupied_seconds=execution['occupied_seconds'],
        limitations=['Tie-aware means equal maximum heuristic score, not strategically optimal play.',
                    'Train and held-out positions are not matched states; source, phase and action complexity can differ.',
                    'Held-out predictions are reused from the full same-adapter vLLM run; batching-dependent numerical variation is not excluded.',
                    'This descriptive diagnostic cannot causally distinguish additional-training benefits from representation/target changes.'],rows=rows)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','summarize']);parser.add_argument('output',type=Path)
    args=parser.parse_args();(prepare if args.mode=='prepare' else finish)(args.output.resolve())
