"""Bounded proportional-data scaling and exact example-count checkpoints."""
import json,math
from pathlib import Path
from collections import Counter
from .training import file_hash

BOUNDARIES=[10000,20000,30000,40000]
CHECKPOINT_STEPS=[157,314,471,628]


def batch_ranges(count,batch_size,boundaries):
    if boundaries!=sorted(set(boundaries)) or not boundaries or boundaries[-1]!=count or boundaries[0]<=0:
        raise ValueError('boundaries must increase through the complete dataset')
    start=0;result=[]
    for boundary in boundaries:
        while start<boundary:
            end=min(start+batch_size,boundary);result.append((start,end));start=end
    return result


def allocate_sources(original,teacher_capacity,corrective_capacity):
    """Keep exact action/source marginals; respect scarce joint-cell capacities."""
    teacher='teacher_trajectory';corrective='smoke_on_policy_relabel'
    original_counts=Counter((r['collection_source'],r['family']) for r in original)
    targets=Counter(r['family'] for r in original)
    targets={k:v*4 for k,v in targets.items()}
    desired={k:4*original_counts[corrective,k] for k in targets}
    low={k:max(original_counts[corrective,k],targets[k]-teacher_capacity.get(k,0)) for k in targets}
    high={k:min(corrective_capacity.get(k,0),targets[k]-original_counts[teacher,k]) for k in targets}
    if any(low[k]>high[k] for k in targets):raise ValueError('action/source constraints infeasible')
    assigned={k:max(low[k],min(high[k],desired[k])) for k in targets}
    while sum(assigned.values())!=8000:
        direction=1 if sum(assigned.values())<8000 else -1
        choices=[k for k in targets if low[k]<=assigned[k]+direction<=high[k]]
        if not choices:raise ValueError('8000 unique corrections unavailable under action quotas')
        def penalty(k):
            return ((assigned[k]+direction-desired[k])**2-(assigned[k]-desired[k])**2)/max(desired[k],1)
        chosen=min(choices,key=lambda k:(penalty(k),k));assigned[chosen]+=direction
    return {(source,k):(assigned[k] if source==corrective else targets[k]-assigned[k]) for source in [teacher,corrective] for k in targets}


def verify_scale(prepared,manifest,scale,config,stage):
    parent=json.loads(Path(scale['parent_run']).read_text())
    if parent['status']!='completed' or parent['config']!=config:raise ValueError('same completed main configuration required')
    if stage!='scale40k' or config['epochs']!=1 or config['effective_batch_size']!=64:raise ValueError('bounded 40k one-pass stage required')
    if scale['checkpoint_steps']!=CHECKPOINT_STEPS or scale['checkpoint_examples']!=BOUNDARIES:raise ValueError('exact 10k checkpoint boundaries required')
    if scale['adapter_files']!=parent['adapter_files'] or scale['optimizer_state']!='fresh':raise ValueError('initial adapter/optimizer differs')
    for key in ['prompt_manifest','split_manifest','tokenizer_hash','chat_template_hash']:
        if manifest[key]!=parent['data_manifest'][key]:raise ValueError('frozen interface/splits changed')
    path=Path(prepared).parent/'selection.json';selection=json.loads(path.read_text())
    if file_hash(path)!=scale['selection_sha256'] or manifest['source_sha256']!=selection['records_sha256']:raise ValueError('selection/source changed')
    rows=selection['selected'];original=selection['original_selected']
    if len(rows)!=40000 or len(original)!=10000 or len({r['input_hash'] for r in rows})!=40000:raise ValueError('distinct record counts differ')
    if not {r['input_hash'] for r in original}<={r['input_hash'] for r in rows}:raise ValueError('original replay missing')
    for key in ['family','collection_source']:
        actual=Counter(r[key] for r in rows);expected=Counter({k:4*v for k,v in Counter(r[key] for r in original).items()})
        if actual!=expected:raise ValueError('original proportions changed')
    if manifest['audit']['unique_records']!=40000 or selection['labels_reproduced']!=40000:raise ValueError('incomplete corpus audit')
    return parent
