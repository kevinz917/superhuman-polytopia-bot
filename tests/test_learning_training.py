"""Run in training/.venv; the engine environment can skip optional ML packages."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from polytopia_bot.learning.data import ResponseCollator, JsonlDataset, tokenize_example, write_fixtures
from polytopia_bot.learning.smoke import tiny_smoke, tiny_tokenizer
from polytopia_bot.learning.training import prepare, read_config, verified_data
from polytopia_bot.learning.interface import digest


@pytest.fixture(scope="module")
def tokenizer():
    return tiny_tokenizer()


def test_loss_is_only_json_and_eos_even_when_padding_equals_eos(tokenizer):
    response = '{"action_id":"a0"}'
    rows = [tokenize_example(tokenizer, [{"role": "user", "content": text}], response,
                            max_length=8192, max_new_tokens=32) for text in ("short", "much longer input with extra words")]
    batch = ResponseCollator(tokenizer.eos_token_id)(rows)
    for i, row in enumerate(rows):
        labels = row["labels"]
        supervised = [x for x in labels if x != -100]
        assert tokenizer.decode(supervised) == response + "<|im_end|>"
        assert labels[-1] == tokenizer.eos_token_id
        assert all(x == -100 for x in batch["labels"][i, len(labels):])
        assert all(x == 0 for x in batch["attention_mask"][i, len(labels):])


def test_no_silent_catalog_or_response_truncation(tokenizer):
    for length, response_length in ((8, 32), (8192, 1)):
        with pytest.raises(ValueError, match="no truncation"):
            tokenize_example(tokenizer, [{"role": "user", "content": "choose an action"}], '{"action_id":"a0"}',
                             max_length=length, max_new_tokens=response_length)


def test_chat_template_prefix_mismatch_fails(tokenizer):
    broken = deepcopy(tokenizer)
    broken.chat_template += "{% if add_generation_prompt %}BAD{% endif %}"
    with pytest.raises(ValueError, match="generation prefix"):
        tokenize_example(broken, [{"role": "user", "content": "choose"}], '{"action_id":"a0"}', max_length=8192, max_new_tokens=32)


def test_decoding_does_not_hide_invalid_special_tokens(tokenizer):
    from polytopia_bot.learning.inference import decode_completion
    response = '{"action_id":"a0"}'
    ids = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    assert decode_completion(tokenizer, ids) == response
    assert decode_completion(tokenizer, [tokenizer.convert_tokens_to_ids("<|im_start|>")] + ids).startswith("<|im_start|>")


def test_preparation_manifest_hashes_and_production_gate(tmp_path, monkeypatch, tokenizer):
    from polytopia_bot.learning import training
    config = read_config(Path(__file__).parents[1] / "configs" / "sft.json")
    monkeypatch.setattr(training, "load_tokenizer", lambda _: tokenizer)
    source = tmp_path / "fixtures.jsonl"
    write_fixtures(source)
    prepared = tmp_path / "prepared"
    manifest = prepare(source, prepared, config, allow_fixtures=True)
    _, rows = verified_data(prepared, config, stage="fixture")
    assert isinstance(rows, JsonlDataset) and len(rows) == 8
    assert manifest["audit"]["sources"] == ["fixture"]
    with pytest.raises(ValueError, match="teacher-only"):
        verified_data(prepared, config, stage="main")
    with pytest.raises(FileExistsError):
        prepare(source, prepared, config, allow_fixtures=True)
    with (prepared / "tokens.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="changed"):
        verified_data(prepared, config, stage="fixture")


def test_actual_lora_training_freezes_base_and_roundtrips(tmp_path):
    receipt = tiny_smoke(tmp_path / "tiny", steps=80)
    assert receipt["passed"]
    assert receipt["base_weights_unchanged"]
    assert receipt["reload_max_logit_error"] == 0
    assert receipt["checkpoint_steps"] == [20, 40, 80]


def test_real_training_requires_matching_qualification_evidence(tmp_path, monkeypatch, tokenizer):
    from polytopia_bot.learning import training
    config = read_config(Path(__file__).parents[1] / "configs" / "sft.json")
    monkeypatch.setattr(training, "load_tokenizer", lambda _: tokenizer)
    source = tmp_path / "source.jsonl"
    write_fixtures(source)
    # Relabel only inside this negative test to exercise deeper qualification guards.
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows:
        row["source"] = "teacher"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    splits = {"train": [row["map_hash"] for row in rows], "validation": [], "test": []}
    dest = tmp_path / "prepared"
    manifest = prepare(source, dest, config, split_manifest=splits)
    with pytest.raises(ValueError, match="qualification receipts"):
        verified_data(dest, config, stage="smoke")
    ready = {"full_game_ready": True, "teacher_qualified": True, "teacher_hashes": ["wrong"]}
    with pytest.raises(ValueError, match="teacher does not match"):
        verified_data(dest, config, stage="smoke", readiness=ready)
    ready["teacher_hashes"] = manifest["audit"]["teacher_hashes"]
    ready["split_manifest_hash"] = digest(splits)
    with pytest.raises(ValueError, match="qualification evidence"):
        verified_data(dest, config, stage="smoke", readiness=ready)


def test_pinned_qwen_tokenizer_if_cached():
    from transformers import AutoTokenizer
    from polytopia_bot.learning.data import example, demonstrate, fixture_observation
    from polytopia_bot.learning.teacher import ScriptedTeacher
    config = read_config(Path(__file__).parents[1] / "configs" / "sft.json")
    try:
        tok = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"], local_files_only=True)
    except OSError:
        pytest.skip("pinned Qwen tokenizer not cached; prepare downloads tokenizer only")
    row = demonstrate(fixture_observation(), ScriptedTeacher(), episode_id="e", map_hash="m", decision_index=0, source="fixture")
    messages, response, _ = example(row)
    encoded = tokenize_example(tok, messages, response, max_length=8192, max_new_tokens=32)
    assert tok.decode([x for x in encoded["labels"] if x != -100]) == response + "<|im_end|>"


def test_fit_weight_hash_excludes_adapters_but_detects_base_changes():
    from polytopia_bot.learning.training import frozen_weight_hash
    model=torch.nn.Linear(3,2)
    model.weight.requires_grad_(False)
    before=frozen_weight_hash(model)
    with torch.no_grad():model.bias.add_(1)
    assert frozen_weight_hash(model)==before
    with torch.no_grad():model.weight[0,0].add_(1)
    assert frozen_weight_hash(model)!=before


def test_fit_accuracy_compares_complete_json_and_restores_training_mode(tokenizer):
    from polytopia_bot.learning.training import training_response_accuracy
    response='{"action_id":"a0"}'
    rows=[tokenize_example(tokenizer,[dict(role='user',content=text)],response,max_length=512,max_new_tokens=32)
          for text in ('short','a longer observation')]
    class Model(torch.nn.Module):
        device='cpu'
        def generate(self, input_ids, **kwargs):
            tail=torch.tensor([tokenizer.encode(response,add_special_tokens=False)+[tokenizer.eos_token_id]]*len(input_ids))
            return torch.cat([input_ids,tail],dim=1)
    model=Model();model.train()
    result=training_response_accuracy(model,rows,tokenizer,max_new_tokens=32)
    assert result['accuracy']==1 and result['examples']==2 and model.training
