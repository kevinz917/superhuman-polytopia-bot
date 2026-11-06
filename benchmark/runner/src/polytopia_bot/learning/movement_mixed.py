"""Deterministic mixed-data selection and fail-closed warm-start verification."""
from collections import defaultdict, Counter
import hashlib, json
from pathlib import Path
from .training import file_hash


def select_balanced(rows, count, *, tag):
    """Round-robin phase/family strata; prefer less represented maps within each."""
    pools=defaultdict(list)
    for row in rows:pools[(row['phase'],row['family'])].append(row)
    for pool in pools.values():
        pool.sort(key=lambda r:hashlib.sha256((tag+':'+r['input_hash']).encode()).hexdigest())
    selected=[];maps=Counter();seen=set()
    while len(selected)<count:
        progress=False
        for key in sorted(pools):
            pool=pools[key]
            if not pool:continue
            i=min(range(len(pool)),key=lambda i:maps[pool[i]['map_hash']])
            row=pool.pop(i)
            if row['input_hash'] in seen:continue
            seen.add(row['input_hash']);selected.append(row);maps[row['map_hash']]+=1;progress=True
            if len(selected)==count:break
        if not progress:raise ValueError('insufficient distinct examples')
    return selected


def verify_mixed(prepared,manifest,mixed,config,stage):
    parent=json.loads(Path(mixed['parent_run']).read_text())
    if parent['status']!='completed' or parent['config']!=config:
        raise ValueError('mixed training requires unchanged completed-parent configuration')
    if stage!='smoke' or config['epochs']!=1 or config['effective_batch_size']!=64 or mixed['checkpoint_steps']!=[40,79,118,157]:
        raise ValueError('mixed training requires bounded one-pass schedule')
    if mixed['adapter_files']!=parent['adapter_files'] or mixed['optimizer_state']!='fresh':
        raise ValueError('mixed parent adapter/optimizer differs')
    for key in ['prompt_manifest','split_manifest','tokenizer_hash','chat_template_hash']:
        if manifest[key]!=parent['data_manifest'][key]:raise ValueError('mixed data changed frozen interface/splits')
    root=Path(prepared).parent
    audit=json.loads((root/'selection.json').read_text())
    if file_hash(root/'selection.json')!=mixed['selection_sha256'] or manifest['source_sha256']!=audit['records_sha256']:
        raise ValueError('mixed selection/source changed')
    if manifest['audit']['unique_records']!=10000 or audit['movement_count']!=4000 or audit['nonmovement_count']!=6000 or audit['labels_reproduced']!=10000:
        raise ValueError('mixed coverage/teacher audit differs')
    return parent


def score_nonmovement(scores,target,selected):
    if target['kind']==1 or not scores:raise ValueError('nonmovement target required')
    best=max(r['score'] for r in scores);top=[r['command'] for r in scores if r['score']==best]
    if target not in top:raise ValueError('target not globally top scoring')
    if selected is not None and sum(r['command']==selected for r in scores)!=1:raise ValueError('selected command not uniquely legal')
    return dict(exact=target==selected,tie_aware=selected in top,selected_movement=selected is not None and selected['kind']==1)
