from collections import Counter
import pytest
from polytopia_bot.learning.scale_training import batch_ranges,allocate_sources,BOUNDARIES,CHECKPOINT_STEPS


def test_exact_boundaries_cover_each_example_once_with_only_short_boundary_batches():
 ranges=batch_ranges(40000,64,BOUNDARIES)
 assert len(ranges)==628
 assert [ranges[i-1][1] for i in CHECKPOINT_STEPS]==BOUNDARIES
 assert [end-start for start,end in ranges].count(16)==4
 assert all(end-start in [16,64] for start,end in ranges)
 assert [i for start,end in ranges for i in range(start,end)]==list(range(40000))
 assert len(batch_ranges(10000,64,[10000]))==157
 for boundaries in [[],[20000,10000,40000],[10000,20000],[0,40000]]:
  with pytest.raises(ValueError):batch_ranges(40000,64,boundaries)


def test_source_allocator_preserves_both_marginals_when_joint_cells_are_sparse():
 rows=[dict(collection_source='teacher_trajectory',family=k) for k,n in [(1,1000),(2,7000)] for _ in range(n)]
 rows += [dict(collection_source='smoke_on_policy_relabel',family=k) for k,n in [(1,1000),(2,1000)] for _ in range(n)]
 q=allocate_sources(rows,{1:8000,2:30000},{1:1500,2:7000})
 assert q['smoke_on_policy_relabel',1]==1500
 assert q['smoke_on_policy_relabel',2]==6500
 assert sum(v for (s,k),v in q.items() if s=='teacher_trajectory')==32000
 assert sum(v for (s,k),v in q.items() if s=='smoke_on_policy_relabel')==8000
 assert {k:sum(v for (s,f),v in q.items() if f==k) for k in [1,2]}=={1:8000,2:32000}
 with pytest.raises(ValueError,match='unavailable'):
  allocate_sources(rows,{1:8000,2:30000},{1:1500,2:6000})


def test_scale_verifier_rejects_proportion_drift_and_changed_checkpoints(tmp_path):
 import json
 from polytopia_bot.learning.scale_training import verify_scale
 from polytopia_bot.learning.training import file_hash
 config=dict(epochs=1,effective_batch_size=64,learning_rate=.0001)
 rows=[dict(input_hash=str(i),family=1 if i%10==0 else 2,collection_source='teacher_trajectory' if i%10<8 else 'smoke_on_policy_relabel') for i in range(40000)]
 selected=dict(selected=rows,original_selected=rows[:10000],records_sha256='raw',labels_reproduced=40000)
 path=tmp_path/'selection.json';path.write_text(json.dumps(selected))
 manifest=dict(source_sha256='raw',audit={'unique_records':40000},prompt_manifest={'fixed':True},split_manifest={'train':['a']},tokenizer_hash='t',chat_template_hash='c')
 parent=dict(status='completed',config=config,adapter_files={'adapter':'hash'},data_manifest=manifest)
 (tmp_path/'parent.json').write_text(json.dumps(parent))
 scale=dict(parent_run=str(tmp_path/'parent.json'),adapter_files=parent['adapter_files'],optimizer_state='fresh',checkpoint_steps=CHECKPOINT_STEPS,checkpoint_examples=BOUNDARIES,selection_sha256=file_hash(path))
 assert verify_scale(tmp_path/'prepared',manifest,scale,config,'scale40k')==parent
 with pytest.raises(ValueError,match='boundaries'):
  verify_scale(tmp_path/'prepared',manifest,{**scale,'checkpoint_examples':[10048,20032,30016,40000]},config,'scale40k')
 rows[-1]['family']=1;path.write_text(json.dumps(selected));scale['selection_sha256']=file_hash(path)
 with pytest.raises(ValueError,match='proportions'):
  verify_scale(tmp_path/'prepared',manifest,scale,config,'scale40k')
