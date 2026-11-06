"""CPU contract tests; no GPU runtime, oracle, model downloads, or network."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from polytopia_bot.environment import LegalAction, Transition
from polytopia_bot.learning.collection import collect_episode
from polytopia_bot.learning.data import audit_records, demonstrate, example, fixture_observation, observation_from_dict
from polytopia_bot.learning.interface import Generation, PolicyForfeit, ResponseError, choose_action, parse_response, render
from polytopia_bot.learning.teacher import ScriptedTeacher, WeakBaseline


def action(kind, source=-1, target=-1, parameter=0, action_id="x"):
    return LegalAction(action_id, dict(kind=kind, source=source, target=target, parameter=parameter))


def test_catalog_uses_semantics_not_upstream_ids_or_order():
    obs = fixture_observation()
    changed = replace(obs, legal_actions=[replace(a, action_id=f"private-{i}") for i, a in enumerate(reversed(obs.legal_actions))])
    assert render(obs).messages == render(changed).messages
    teacher = ScriptedTeacher()
    assert teacher.select(obs).response == teacher.select(changed).response
    assert teacher.select(obs).action.command == teacher.select(changed).action.command


@pytest.mark.parametrize("field", ["seed", "enemy_stars", "private_snapshot", "hidden_units"])
def test_private_fields_are_rejected(field):
    obs = fixture_observation()
    with pytest.raises(ValueError, match="schema"):
        render(replace(obs, visible_state={**obs.visible_state, field: 123}))


def test_unknown_cells_cannot_carry_terrain_or_entities():
    obs = fixture_observation()
    visible = deepcopy(obs.visible_state)
    visible["tiles"][8]["terrain"] = 1
    with pytest.raises(ValueError, match="hidden"):
        render(replace(obs, visible_state=visible))
    visible = deepcopy(obs.visible_state)
    visible["units"].append(dict(tile=8, owner=2, kind=2, health=10))
    with pytest.raises(ValueError, match="unknown tile"):
        render(replace(obs, visible_state=visible))


def test_memory_requires_valid_observation_timestamp():
    obs = fixture_observation()
    with pytest.raises(ValueError):
        render(replace(obs, remembered_state={"tiles": [{"tile": 8, "terrain": 1}]}))
    with pytest.raises(ValueError):
        render(replace(obs, remembered_state={"tiles": [{"tile": 8, "last_seen_turn": 99}]}))


def test_duplicate_or_partial_catalog_is_not_silently_repaired():
    obs = fixture_observation()
    with pytest.raises(ValueError, match="duplicate semantic"):
        render(replace(obs, legal_actions=[obs.legal_actions[0], replace(obs.legal_actions[0], action_id="other")]))
    with pytest.raises(ValueError, match="at least one"):
        render(replace(obs, legal_actions=[]))


@pytest.mark.parametrize("response", ['{"action_id":"a0"} extra', '```json\n{"action_id":"a0"}\n```',
    '{"action_id":"a0","action_id":"a1"}', '{"action_id":"a0","reason":"x"}',
    '{"action_id":0}', '[]', '{"action_id":"a999"}'])
def test_bad_responses_fail(response):
    with pytest.raises(ResponseError):
        parse_response(response, render(fixture_observation()), current_version=0)


def test_stale_action_and_caller_mutation_cannot_change_saved_prompt():
    obs = fixture_observation()
    prompt = render(obs)
    with pytest.raises(ResponseError, match="stale"):
        parse_response('{"action_id":"a0"}', prompt, current_version=1)
    selected = parse_response('{"action_id":"a0"}', prompt, current_version=0)
    original = dict(selected.command)
    for a in obs.legal_actions:
        a.command["parameter"] = 999
    assert selected.command == original


def test_retry_records_failures_and_never_falls_back():
    outputs = iter(["bad", '{"action_id":"a0"}'])
    action_result, attempts = choose_action(fixture_observation(), lambda messages: Generation(next(outputs), [1], [2]))
    assert len(attempts) == 2 and attempts[0]["error"] == "invalid_json"
    assert attempts[1]["messages"][-1]["role"] == "user"
    assert action_result.action_id.startswith("engine-")
    with pytest.raises(PolicyForfeit) as exc:
        choose_action(fixture_observation(), lambda messages: Generation("bad", [1], [2]))
    assert len(exc.value.attempts) == 3


def test_teacher_capture_beats_research_and_resignation():
    obs = replace(fixture_observation(), legal_actions=[action(8, source=4, action_id="capture"),
        action(7, parameter=6, action_id="tech"), action(11, action_id="resign")])
    decision = ScriptedTeacher().select(obs)
    assert decision.action.action_id == "capture"
    assert all(s["reason"] for s in decision.scores)


def test_teacher_explores_and_avoids_unneeded_recruitment():
    teacher = ScriptedTeacher()
    assert teacher.select(fixture_observation()).action.action_id == "engine-move"
    obs = fixture_observation()
    visible = deepcopy(obs.visible_state)
    visible["units"] += [dict(tile=i, owner=1, kind=2, health=10) for i in (0, 1)]
    obs = replace(obs, visible_state=visible, legal_actions=[action(4, source=4, parameter=2, action_id="recruit"), action(10, action_id="end")])
    assert teacher.select(obs).action.action_id == "end"


def test_teacher_prefers_superunit_over_park_in_elimination():
    obs = replace(fixture_observation(), legal_actions=[action(9, target=4, parameter=6, action_id="giant"), action(9, target=4, parameter=2, action_id="park")])
    assert ScriptedTeacher().select(obs).action.action_id == "giant"


def test_weak_baseline_is_legal_reproducible_and_does_not_resign():
    obs = fixture_observation()
    obs = replace(obs, legal_actions=[*obs.legal_actions, action(11, action_id="resign")])
    policy = WeakBaseline()
    assert policy.select(obs) == policy.select(obs)
    assert policy.select(obs).action.command["kind"] != 11
    assert policy.select(obs).action in obs.legal_actions


def record(obs=None):
    return demonstrate(obs or fixture_observation(), ScriptedTeacher(), episode_id="episode", map_hash="board", decision_index=0, source="fixture")


def test_record_roundtrip_excludes_metadata_from_prompt():
    row = record()
    assert observation_from_dict(row["observation"]) == fixture_observation()
    messages, _, _ = example(row)
    assert "teacher_scores" not in json.dumps(messages)
    assert "map_hash" not in json.dumps(messages)


def test_fixture_and_validation_records_cannot_enter_real_training():
    with pytest.raises(ValueError, match="synthetic"):
        audit_records([record()])
    row = record()
    row["split"] = "validation"
    with pytest.raises(ValueError, match="training-split"):
        audit_records([row], allow_fixtures=True)


def test_map_split_overlap_and_nontraining_map_fail():
    with pytest.raises(ValueError, match="overlap"):
        audit_records([record()], allow_fixtures=True, split_manifest={"train": ["board"], "validation": ["board"], "test": []})
    with pytest.raises(ValueError, match="absent"):
        audit_records([record()], allow_fixtures=True, split_manifest={"train": ["other"], "validation": [], "test": []})


def test_semantic_dedup_and_conflicting_labels():
    r = record()
    copy = deepcopy(r)
    copy["observation"]["state_version"] = 123
    unique, audit = audit_records([r, copy], allow_fixtures=True)
    assert len(unique) == 1 and audit["duplicates"] == 1
    copy["selected_command"] = dict(fixture_observation().legal_actions[1].command)
    with pytest.raises(ValueError, match="conflicting"):
        audit_records([r, copy], allow_fixtures=True)


class FixtureEnvironment:
    def __init__(self, terminal=True):
        self.version = 0
        self.terminal = terminal
        self.closed = False
    def reset(self, **kwargs):
        return fixture_observation()
    def observe(self, player):
        obs = fixture_observation()
        if player == 2:
            v = deepcopy(obs.visible_state)
            v["cities"] = []
            obs = replace(obs, visible_state=v)
        return replace(obs, player=player, state_version=self.version)
    def step(self, *, action_id, expected_state_version):
        assert expected_state_version == self.version
        self.version += 1
        done = self.terminal and self.version == 2
        return Transition(self.observe(1), None if done else 2, {1: -1., 2: 1.} if done else {1: 0., 2: 0.}, done, False, "loss" if done else None)
    def close(self):
        self.closed = True


def test_collector_preserves_losing_demonstrations_and_seat_boundary():
    env = FixtureEnvironment()
    rows, receipt = collect_episode(env, settings={}, seed=0, episode_id="e", map_hash="m", teacher=ScriptedTeacher(),
                                    opponent=ScriptedTeacher("pressure"), fixture=True)
    assert len(rows) == 1 and rows[0]["observation"]["player"] == 1
    assert receipt["terminated"] and not receipt["truncated"] and receipt["rewards"][1] == -1
    assert env.closed


def test_collector_action_limit_is_truncation_and_readiness_is_required():
    kwargs = dict(settings={}, seed=0, episode_id="e", map_hash="m", teacher=ScriptedTeacher(), opponent=ScriptedTeacher())
    with pytest.raises(ValueError, match="readiness"):
        collect_episode(FixtureEnvironment(), **kwargs)
    _, receipt = collect_episode(FixtureEnvironment(False), **kwargs, fixture=True, max_decisions=1)
    assert receipt["truncated"] and not receipt["terminated"]
    assert receipt["reason"] == "collector_action_limit"
