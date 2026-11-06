import json
import pytest
from polytopia_bot.learning.movement_probe import select_positions,verify_subset
from polytopia_bot.learning.training import file_hash


def test_sampling_is_order_independent_and_map_disjoint():
    rows=[dict(position_id=str(i),board_hash=str(i//2),phase=['opening','exploration','late'][i%3],scores=[{}]*(20 if i%2 else 150)) for i in range(90)]
    selected=select_positions(rows,12,tag='test')
    assert selected==select_positions(list(reversed(rows)),12,tag='test')
    assert len({r['board_hash'] for r in selected})==12
    assert {r['phase'] for r in selected}=={'opening','exploration','late'}
    rest=select_positions(rows,12,tag='rest',excluded_maps=[r['board_hash'] for r in selected])
    assert not {r['board_hash'] for r in selected}&{r['board_hash'] for r in rest}
    with pytest.raises(ValueError,match='insufficient'):
        select_positions(rows,46,tag='too many')


def test_subset_validation_rejects_relabeling_and_prompt_config_changes(tmp_path):
    parent=tmp_path/'parent';subset=tmp_path/'subset';parent.mkdir();subset.mkdir()
    rows=[{'input_ids':[i,1], 'labels':[-100,1], 'attention_mask':[1,1]} for i in range(256)]
    indices=list(range(0,256,2))
    (parent/'tokens.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (subset/'tokens.jsonl').write_text(''.join(json.dumps(rows[i])+'\n' for i in indices))
    config=dict(epochs=1,effective_batch_size=64,learning_rate=.0001,max_length=32768)
    run=dict(status='completed',config=config,data_manifest=dict(tokens_sha256=file_hash(parent/'tokens.jsonl')))
    (parent/'run.json').write_text(json.dumps(run))
    manifest=dict(subset_provenance=dict(parent_prepared=str(parent),parent_tokens_sha256=run['data_manifest']['tokens_sha256'],parent_indices=indices))
    probe_config={**config,'epochs':20,'effective_batch_size':8}
    assert verify_subset(subset,manifest,parent/'run.json',probe_config)==run
    with pytest.raises(ValueError,match='settings'):
        verify_subset(subset,manifest,parent/'run.json',{**probe_config,'max_length':8192})
    changed=[rows[i].copy() for i in indices];changed[0]['labels']=[-100,2]
    (subset/'tokens.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in changed))
    with pytest.raises(ValueError,match='tokens or labels changed'):
        verify_subset(subset,manifest,parent/'run.json',probe_config)
