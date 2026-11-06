import gzip,json
from dataclasses import replace
import pytest
from polytopia_bot.learning.corpus import sample
from polytopia_bot.learning.data import demonstrate,fixture_observation
from polytopia_bot.learning.teacher import ScriptedTeacher


def make_source(path, *, conflict=False, fixture=False):
    path.mkdir()
    (path/'report.json').write_text('{"complete":true}')
    obs=fixture_observation();rows=[]
    for i in range(10):
        observation=replace(obs,visible_state={**obs.visible_state,'stars':i+1})
        # Deliberately synthetic test-only rows exercise sampling controls.
        row=demonstrate(observation,ScriptedTeacher(),episode_id='e',map_hash='m',decision_index=i,
                        source='fixture' if fixture else 'teacher')
        if i<5:row['selected_command']=dict(obs.legal_actions[1].command)
        rows.append(row)
    duplicate=json.loads(json.dumps(rows[0]))
    if conflict:duplicate['selected_command']=dict(obs.legal_actions[0].command)
    rows.append(duplicate)
    with gzip.open(path/'episode.jsonl.gz','wt') as stream:
        for r in rows:stream.write(json.dumps(dict(record=r))+'\n')


def test_sampling_deduplicates_caps_endturn_and_is_reproducible(tmp_path):
    source=tmp_path/'source';make_source(source)
    a=sample([source],tmp_path/'a',count=5)
    b=sample([source],tmp_path/'b',count=5)
    assert a==b and a['duplicates']==1 and a['unique_available']==10
    assert a['families']['10']<=1
    assert (tmp_path/'a/records.jsonl').read_bytes()==(tmp_path/'b/records.jsonl').read_bytes()


def test_sampling_rejects_conflicts_fixtures_and_insufficient_nontrivial_rows(tmp_path):
    source=tmp_path/'conflict';make_source(source,conflict=True)
    with pytest.raises(ValueError,match='conflicting'):
        sample([source],tmp_path/'a',count=5)
    source=tmp_path/'fixture';make_source(source,fixture=True)
    with pytest.raises(ValueError,match='fixtures'):
        sample([source],tmp_path/'b',count=5)
    source=tmp_path/'small';make_source(source)
    with pytest.raises(ValueError,match='cap'):
        sample([source],tmp_path/'c',count=10)
