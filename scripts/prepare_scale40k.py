"""Build an audited 40k corpus preserving original action/source proportions."""
import gzip,json,hashlib,multiprocessing,sys,shutil
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.training import file_hash,load_tokenizer
from polytopia_bot.learning.interface import canonical,digest
from polytopia_bot.learning.scale_training import allocate_sources,BOUNDARIES,CHECKPOINT_STEPS
from polytopia_bot.learning.data import CorpusAudit,observation_from_dict,tokenize_example
from polytopia_bot.learning.teacher import ScriptedTeacher
from polytopia_bot.learning.freeze import verify_prompt


def bounded_phase_counts(pool,original,count):
    capacities=Counter(r['phase'] for r in pool);prior=Counter(r['phase'] for r in original)
    assigned={phase:prior.get(phase,0) for phase in capacities};target={phase:count*prior.get(phase,0)/len(original) for phase in capacities}
    while sum(assigned.values())<count:
        options=[p for p in assigned if assigned[p]<capacities[p]]
        if not options:raise ValueError('phase pool exhausted')
        p=min(options,key=lambda p:(((assigned[p]+1-target[p])**2-(assigned[p]-target[p])**2)/max(target[p],1),p));assigned[p]+=1
    return assigned


def prepare_shard(args):
    root,index,parent=args;root=Path(root);config=parent['config'];m=parent['data_manifest'];strategy=verify_prompt(m['prompt_manifest'],config)
    tokenizer=load_tokenizer(config)
    if digest(tokenizer.backend_tokenizer.to_str())!=m['tokenizer_hash'] or digest(tokenizer.chat_template)!=m['chat_template_hash']:raise ValueError('tokenizer changed')
    teacher=ScriptedTeacher();audit=CorpusAudit(split_manifest=m['split_manifest'],strategy=strategy);lengths=[];supervised=0;keys=[];families=Counter();strict=0
    with gzip.open(root/f'raw-{index}.jsonl.gz','rt') as inp,gzip.open(root/f'tokens-{index}.jsonl.gz','wt',compresslevel=1) as tokens,gzip.open(root/f'examples-{index}.jsonl.gz','wt',compresslevel=1) as examples:
        for line in inp:
            row=json.loads(line);accepted=audit.accept(row)
            if accepted is None:raise ValueError('duplicate shard input')
            _,messages,response=accepted;choice=teacher.select(observation_from_dict(row['observation']))
            if row['selected_command']!=dict(choice.action.command) or row['teacher_hash']!=choice.teacher_hash or row['teacher_scores']!=choice.scores:raise ValueError('teacher no longer reproduces')
            encoded=tokenize_example(tokenizer,messages,response,max_length=config['max_length'],max_new_tokens=config['max_new_tokens'])
            tokens.write(canonical(encoded)+'\n');examples.write(canonical(dict(episode_id=row['episode_id'],decision_index=row['decision_index'],map_hash=row['map_hash'],messages=messages,response=response,selected_command=row['selected_command']))+'\n')
            # CorpusAudit stores semantic keys and target identities, independent of episode IDs.
            lengths.append(len(encoded['input_ids']));supervised+=sum(x!=-100 for x in encoded['labels']);families[str(row['selected_command']['kind'])]+=1
            if row['selected_command']['kind']==1:
                best=max(s['score'] for s in choice.scores);strict+=all(s['command']['kind']==1 for s in choice.scores if s['score']==best)
    return dict(index=index,audit=audit.report(),keys=audit.seen,episodes=audit.episodes,maps=sorted(audit.maps),max_tokens=max(lengths),processed_tokens=sum(lengths),supervised_tokens=supervised,families=dict(families),strict_movement=strict)


def main():
    root=ROOT/'runs/movement-scale40k-v1';parent=json.loads((ROOT/'runs/movement-epoch2-v1/training-run.json').read_text());m=parent['data_manifest']
    old=json.loads((ROOT/'runs/movement-mixed-v1/protocol.json').read_text());original=json.loads((ROOT/'runs/sft0/repaired-smoke-sample-v2/sampling.json').read_text());candidate=json.loads((ROOT/'runs/sft0/main-candidate-sample-v2/sampling.json').read_text())
    for folder,meta in [('repaired-smoke-sample-v2',original),('main-candidate-sample-v2',candidate)]:
        if file_hash(ROOT/'runs/sft0'/folder/'records.jsonl')!=meta['records_sha256']:raise ValueError('raw source changed')
    for name,proof in original['source_provenance'].items():
        path=ROOT/name/'report.json'
        if file_hash(path)!=proof['report_sha256'] or not json.loads(path.read_text())['complete'] or path.parent.joinpath('invalidation.json').exists():raise ValueError('source invalid')
    if any(original['source_files'].get(k)!=v for k,v in candidate['source_files'].items()):raise ValueError('candidate provenance differs')
    teachers=[dict(**r,collection_source='teacher_trajectory') for r in candidate['selected']]
    teacherkeys={r['input_hash'] for r in teachers};corr=[];seen=set(teacherkeys)
    for r in json.loads((root/'corrective-index.json').read_text()):
        if r['input_hash'] in seen:continue
        corr.append({k:v for k,v in r.items() if k!='target'});seen.add(r['input_hash'])
    pool=teachers+corr;bykey={r['input_hash']:r for r in pool};original_bykey={r['input_hash']:r for r in original['selected']}
    if any(k not in bykey or bykey[k]['collection_source']!=r['collection_source'] for k,r in original_bykey.items()):raise ValueError('original corpus missing from pool')
    quotas=allocate_sources(original['selected'],Counter(r['family'] for r in teachers),Counter(r['family'] for r in corr));selected=[]
    for (source,kind),count in sorted(quotas.items()):
        rows=[r for r in pool if r['collection_source']==source and r['family']==kind];prior=[r for r in original['selected'] if r['collection_source']==source and r['family']==kind]
        phases=bounded_phase_counts(rows,prior,count)
        for phase,need in sorted(phases.items()):
            oldrows=[r for r in prior if r['phase']==phase];newrows=sorted([r for r in rows if r['phase']==phase and r['input_hash'] not in original_bykey],key=lambda r:digest(['scale40k-v1',r['input_hash']]))
            selected+=oldrows+newrows[:need-len(oldrows)]
    if len(selected)!=40000 or len({r['input_hash'] for r in selected})!=40000:raise ValueError('selection coverage differs')
    wanted={(r['episode_id'],r['decision_index']):r for r in selected};written=[];streams=[gzip.open(root/f'raw-{i}.jsonl.gz','wt',compresslevel=1) for i in range(8)]
    try:
        for line in (ROOT/'runs/sft0/main-candidate-sample-v2/records.jsonl').open():
            row=json.loads(line);key=(row['episode_id'],row['decision_index'])
            if key in wanted:
                meta=wanted.pop(key);streams[len(written)//5000].write(canonical(row)+'\n');written.append(meta)
        for name in sorted(k for k in original['source_files'] if '/raw-relabels-v1/' in k):
            if file_hash(ROOT/name)!=original['source_files'][name]:raise ValueError('corrective source changed')
            for line in gzip.open(ROOT/name,'rt'):
                row=json.loads(line)['record']
                if row is None:continue
                key=(row['episode_id'],row['decision_index'])
                if key in wanted:
                    meta=wanted.pop(key);streams[len(written)//5000].write(canonical(row)+'\n');written.append(meta)
    finally:
        for stream in streams:stream.close()
    if wanted:raise ValueError('selected rows missing')
    print(json.dumps({'selected':len(written),'families':dict(Counter(r['family'] for r in written)),'sources':dict(Counter(r['collection_source'] for r in written)),'joint':{str(k):v for k,v in quotas.items()}}),flush=True)
    (root/'allocation.json').write_text(json.dumps(dict(selected=written,original_selected=original['selected'],quotas={str(k):v for k,v in quotas.items()}),indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as executor:
        futures=[executor.submit(prepare_shard,(str(root),i,parent)) for i in range(8)]
        for f in as_completed(futures):
            result=f.result();results.append(result);print(json.dumps({'shard_complete':result['index'],'labels_verified':len(results)*5000,'max_tokens':result['max_tokens']}),flush=True)
    results.sort(key=lambda r:r['index']);allkeys={};episodes={};maps=set()
    for result in results:
        if allkeys.keys()&result['keys'].keys():raise ValueError('cross-shard duplicate')
        allkeys.update(result['keys']);episodes.update(result['episodes']);maps.update(result['maps'])
    if allkeys.keys()!={r['input_hash'] for r in written}:raise ValueError('selected semantic inputs differ')
    prepared=root/'prepared';prepared.mkdir()
    for prefix,destination in [('raw',root/'records.jsonl.gz'),('tokens',prepared/'tokens.jsonl.gz'),('examples',prepared/'examples.jsonl.gz')]:
        with destination.open('wb') as out:
            for i in range(8):
                with (root/f'{prefix}-{i}.jsonl.gz').open('rb') as inp:shutil.copyfileobj(inp,out)
    with gzip.open(prepared/'tokens.jsonl.gz','rb') as stream:token_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    audit=dict(input_records=40000,unique_records=40000,duplicates=0,episodes=len(episodes),maps=len(maps),sources=['teacher'],teacher_hashes=m['audit']['teacher_hashes'],observation_schema=m['audit']['observation_schema'])
    manifest=dict(schema='polytopia-prepared-v1',config=parent['config'],config_hash=digest(parent['config']),source_sha256=file_hash(root/'records.jsonl.gz'),tokens_sha256=token_sha,
        chat_template_hash=m['chat_template_hash'],tokenizer_hash=m['tokenizer_hash'],audit=audit,split_manifest=m['split_manifest'],prompt_manifest=m['prompt_manifest'],max_tokens=max(r['max_tokens'] for r in results),processed_tokens=sum(r['processed_tokens'] for r in results),supervised_tokens=sum(r['supervised_tokens'] for r in results))
    (prepared/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    selection=dict(schema='original-proportions-scale40k-v1',count=40000,records_sha256=manifest['source_sha256'],labels_reproduced=40000,selected=written,original_selected=original['selected'],
        families=dict(Counter(str(r['family']) for r in written)),collection_sources=dict(Counter(r['collection_source'] for r in written)),phases=dict(Counter(r['phase'] for r in written)),
        original_source_family=dict(Counter(str((r['collection_source'],r['family'])) for r in original['selected'])),source_family={str(k):v for k,v in quotas.items()},
        source_sampling_sha256={name:file_hash(ROOT/name) for name in ['runs/sft0/repaired-smoke-sample-v2/sampling.json','runs/sft0/main-candidate-sample-v2/sampling.json']},
        movement_count=5192,nonmovement_count=34808,new_examples=30000,maps=len(maps),strict_movement=sum(r['strict_movement'] for r in results),
        limitation='Exact action-family and source marginals are quadrupled. Sparse corrective source/action cells cannot be quadrupled; joint quotas are capacity-constrained. Phase matching is approximate within those cells.',
        selection='Includes all original 10000 examples plus 30000 distinct inputs. Hash-based additional selection, closest feasible phase proportions within source/action cells. No prediction or outcome filtering.')
    (root/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    for name,key in [('positions.jsonl.gz','positions_sha256'),('scoring.jsonl.gz','scoring_sha256')]:
        src=ROOT/'runs/movement-mixed-v1'/name
        if file_hash(src)!=old[key]:raise ValueError('held-out inputs changed')
        shutil.copyfile(src,root/name)
    training={k:v for k,v in old['training'].items() if k!='mixed'}
    training.update(run_id='movement-scale40k-v1/training',prepared='/workspace/runs/movement-scale40k-v1/prepared',stage='scale40k',seconds_limit=5400,
        scale=dict(parent_run='/workspace/runs/movement-epoch2-v1/training/model/run.json',adapter='/workspace/runs/movement-epoch2-v1/training/model/adapter-final',adapter_files=parent['adapter_files'],optimizer_state='fresh',checkpoint_steps=CHECKPOINT_STEPS,checkpoint_examples=BOUNDARIES,selection_sha256=file_hash(root/'selection.json')))
    uploads=['selection.json','positions.jsonl.gz','scoring.jsonl.gz','prepared/manifest.json','prepared/tokens.jsonl.gz']
    protocol={**{k:v for k,v in old.items() if k not in ['training','training_selection','optimizer','scope','uploads','upload_files','compressed_inputs','checkpoint_steps']},
        'schema':'original-proportions-scale40k-v1','run_id':'movement-scale40k-v1','training':training,'checkpoint_steps':[0,*CHECKPOINT_STEPS],'checkpoint_examples':{'0':0,**{str(k):v for k,v in zip(CHECKPOINT_STEPS,BOUNDARIES)}},
        'upload_files':['protocol.json',*uploads],'uploads':{n:file_hash(root/n) for n in uploads},'compressed_inputs':{'prepared/tokens.jsonl.gz':'prepared/tokens.jsonl'},
        'training_selection':{k:v for k,v in selection.items() if k not in ['selected','original_selected']},
        'optimizer':'Fresh AdamW, same LR1e-4 and effective batch64. Short batch16 at each exact 10000-example boundary: 628 updates total; 19-step warmup and cosine decay across the full pass.',
        'scope':'One pass over 40000 distinct examples from preserved main. Same frozen prompt and model. Baseline and exact 10k/20k/30k/40k held-out evaluations. No RL, full games, final-test access, automatic promotion, or compute-matched extra run.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (root/'preparation-audit.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in ['keys','episodes','maps']} for r in results],indent=2)+'\n')
    print(json.dumps({'prepared':True,'count':40000,'max_tokens':manifest['max_tokens'],'processed_tokens':manifest['processed_tokens'],'source_family':selection['source_family'],'phases':selection['phases']}),flush=True)

if __name__=='__main__':main()
