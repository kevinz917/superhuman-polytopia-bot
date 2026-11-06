import json
import pytest
from polytopia_bot.learning.movement_mixed import select_balanced,verify_mixed,score_nonmovement
from polytopia_bot.learning.training import file_hash


def test_balanced_selection_covers_maps_without_repetition_or_order_dependence():
 rows=[dict(input_hash=str(i),phase='opening' if i%2 else 'late',family=1,map_hash=str(i//20)) for i in range(200)]
 selected=select_balanced(rows,40,tag='test')
 assert selected==select_balanced(list(reversed(rows)),40,tag='test')
 assert len({r['input_hash'] for r in selected})==40
 assert len({r['map_hash'] for r in selected})==10
 with pytest.raises(ValueError,match='insufficient'):select_balanced(rows,201,tag='test')


def test_nonmovement_ties_count_other_optimal_commands_but_reject_illegal():
 target={'kind':2};tie={'kind':3};move={'kind':1}
 scores=[dict(command=c,score=s) for c,s in [(target,3),(tie,3),(move,1)]]
 assert score_nonmovement(scores,target,tie)==dict(exact=False,tie_aware=True,selected_movement=False)
 assert not score_nonmovement(scores,target,move)['tie_aware']
 assert not score_nonmovement(scores,target,None)['tie_aware']
 with pytest.raises(ValueError,match='uniquely legal'):score_nonmovement(scores,target,{'kind':99})


def test_mixed_verification_rejects_changed_parent_prompt_or_selection(tmp_path):
 config=dict(epochs=1,effective_batch_size=64)
 manifest=dict(prompt_manifest={'a':1},split_manifest={'train':['a']},tokenizer_hash='t',chat_template_hash='c',source_sha256='raw',audit={'unique_records':10000})
 parent=dict(status='completed',config=config,adapter_files={'adapter':'hash'},data_manifest=manifest)
 (tmp_path/'run.json').write_text(json.dumps(parent))
 audit=dict(records_sha256='raw',movement_count=4000,nonmovement_count=6000,labels_reproduced=10000)
 (tmp_path/'selection.json').write_text(json.dumps(audit))
 mixed=dict(parent_run=str(tmp_path/'run.json'),adapter_files=parent['adapter_files'],optimizer_state='fresh',checkpoint_steps=[40,79,118,157],selection_sha256=file_hash(tmp_path/'selection.json'))
 assert verify_mixed(tmp_path/'prepared',manifest,mixed,config,'smoke')==parent
 with pytest.raises(ValueError,match='frozen'):verify_mixed(tmp_path/'prepared',{**manifest,'prompt_manifest':{}},mixed,config,'smoke')
 with pytest.raises(ValueError,match='adapter'):verify_mixed(tmp_path/'prepared',manifest,{**mixed,'adapter_files':{}},config,'smoke')
 (tmp_path/'selection.json').write_text('{}')
 with pytest.raises(ValueError,match='selection'):verify_mixed(tmp_path/'prepared',manifest,mixed,config,'smoke')
