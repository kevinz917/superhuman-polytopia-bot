from polytopia_bot.learning.data import demonstrate,example,fixture_observation
from polytopia_bot.learning.teacher import ScriptedTeacher
from polytopia_bot.learning.freeze import verify_prompt
from polytopia_bot.learning.interface import digest
import pytest


def test_selected_guidance_changes_only_instructions_not_labels_or_dedup_key():
    row=demonstrate(fixture_observation(),ScriptedTeacher(),episode_id='e',map_hash='m',decision_index=0,source='fixture')
    a,target,key=example(row)
    b,other,other_key=example(row,strategy='Protect your cities.')
    assert b[0]['content']==a[0]['content']+'\nStrategic guidance:\nProtect your cities.'
    assert b[1:]==a[1:] and other==target and other_key==key


def test_missing_or_tampered_freeze_cannot_authorize_real_training():
    with pytest.raises(ValueError,match='frozen prompt'):
        verify_prompt(None,{})
    with pytest.raises(ValueError,match='digest'):
        verify_prompt(dict(schema='frozen-sft-prompt-v1',manifest_hash='wrong',config_hash=digest({})),{})
