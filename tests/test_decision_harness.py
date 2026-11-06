from copy import deepcopy
from dataclasses import asdict, replace
import json
import pytest

from polytopia_bot.learning.data import fixture_observation
from polytopia_bot.learning.decision_harness import VERSION, render_decision, choose_action_once
from polytopia_bot.learning.harness_comparison import evaluate_pairs
from polytopia_bot.learning.interface import Generation, ResponseError, digest, render
from polytopia_bot.learning.teacher import ScriptedTeacher


def test_original_identity_complete_catalog_and_no_teacher_fields():
    obs = fixture_observation()
    original = render(obs)
    assert render_decision(obs, 'original') == original
    richer = render_decision(obs)
    assert richer.catalog == original.catalog and richer.public == original.public
    old_input = json.loads(original.messages[1]['content'])
    new_input = json.loads(richer.messages[1]['content'])
    facts = new_input.pop('decision_facts')
    assert new_input == old_input
    assert 'score' not in json.dumps(facts) and 'teacher' not in json.dumps(facts)
    assert sum(g['count'] for g in facts['action_groups']) == len(original.catalog)


def test_geometry_is_public_and_missing_objective_is_not_zero_distance():
    obs = fixture_observation()
    v = deepcopy(obs.visible_state)
    v['cities'] = [c for c in v['cities'] if c['owner'] == obs.player]
    for t in v['tiles']:
        if t['known'] and t.get('owner') != obs.player:
            t['structure'] = 0
    facts = json.loads(render_decision(replace(obs, visible_state=v)).messages[1]['content'])['decision_facts']
    assert facts['visible_settlements'] == []
    assert all(r[3] is None for r in facts['location_facts']['rows'])
    assert all(r[1] is None for r in facts['movement_facts']['rows'])
    size = v['board_size']
    assert all(r[0] == r[2]*size+r[1] for r in facts['location_facts']['rows'])
    for row in facts['location_facts']['rows']:
        expected = sum(not t['known'] and max(abs(t['tile']%size-row[1]),abs(t['tile']//size-row[2])) == 1 for t in v['tiles'])
        assert row[4] == expected


def test_hidden_fields_rejected_and_single_call_has_no_repairs():
    obs = fixture_observation()
    with pytest.raises(ValueError):
        render_decision(replace(obs, visible_state={**obs.visible_state, 'hidden_enemies': []}))
    calls = []
    def bad(messages):
        calls.append(messages)
        return Generation('bad', [1], [2])
    with pytest.raises(ResponseError):
        choose_action_once(obs, bad)
    assert len(calls) == 1


def test_evaluation_is_paired_and_invalid_is_not_repaired(tmp_path):
    obs = fixture_observation()
    decision = ScriptedTeacher().select(obs)
    versions = ['original', VERSION]
    row = dict(position_id='one', board_hash='map', phase='opening', observation=asdict(obs),
        teacher_command=dict(decision.action.command), scores=decision.scores,
        harness_digests={v: digest(render_decision(obs,v).messages) for v in versions})
    class Generator:
        config = dict(max_new_tokens=32)
        calls = 0
        def batch(self, messages):
            self.calls += 1
            assert len(messages) == 1
            return [Generation(decision.response if self.calls == 1 else 'bad',[1],[2])]
    g = Generator()
    r = evaluate_pairs(tmp_path/'run', [row], g, versions)
    assert r['complete'] and g.calls == 2
    assert r['results'][0]['tie_aware']
    assert not r['results'][1]['valid'] and not r['results'][1]['tie_aware']
