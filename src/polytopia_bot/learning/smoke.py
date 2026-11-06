"""Local real-gradient diagnostic using a tiny random Qwen3, never a 4B result."""
import json
from pathlib import Path

from .data import ResponseCollator, tokenize_example
from .training import checkpoint_callback, file_hash, lora_config


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["choose one action {\"action_id\":\"a0\"}"], trainers.BpeTrainer(vocab_size=280,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<|pad|>", "<|im_start|>", "<|im_end|>"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|pad|>")
    tokenizer.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    return tokenizer


def tiny_smoke(output, steps=80):
    import torch
    from peft import PeftModel, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments, set_seed
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    set_seed(17)
    torch.set_num_threads(2)
    tokenizer = tiny_tokenizer()
    rows = [tokenize_example(tokenizer, [{"role": "user", "content": f"choose one action {i}"}], '{"action_id":"a0"}',
                             max_length=256, max_new_tokens=32) for i in range(4)]
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
               num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
               max_position_embeddings=256, tie_word_embeddings=True))
    model.save_pretrained(output / "tiny-base")
    model.name_or_path = str(output / "tiny-base")
    model = get_peft_model(model, lora_config({"lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0,
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}))
    base_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    adapter_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    collator = ResponseCollator(tokenizer.pad_token_id)
    batch = collator(rows)
    model.eval()
    with torch.no_grad():
        initial_result = model(**batch)
        initial = float(initial_result.loss)
        active = batch["labels"][:, 1:] != -100
        initial_accuracy = float((initial_result.logits[:, :-1].argmax(-1)[active] == batch["labels"][:, 1:][active]).float().mean())
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(output / "trainer"), use_cpu=True,
               per_device_train_batch_size=2, gradient_accumulation_steps=2, max_steps=steps, learning_rate=0.02,
               lr_scheduler_type="constant", save_strategy="no", logging_steps=max(1, steps // 4),
               report_to="none", disable_tqdm=True, dataloader_pin_memory=False, seed=17),
               train_dataset=rows, data_collator=collator, callbacks=[checkpoint_callback()])
    trainer.train()
    model.eval()
    with torch.no_grad():
        result = model(**batch)
        final = float(result.loss)
        final_accuracy = float((result.logits[:, :-1].argmax(-1)[active] == batch["labels"][:, 1:][active]).float().mean())
        expected_logits = result.logits
    frozen = all(torch.equal(p, base_before[n]) for n, p in model.named_parameters() if n in base_before)
    changed = any(not torch.equal(p, adapter_before[n]) for n, p in model.named_parameters() if n in adapter_before)
    model.save_pretrained(output / "adapter")
    restored = PeftModel.from_pretrained(Qwen3ForCausalLM.from_pretrained(output / "tiny-base"), output / "adapter").eval()
    with torch.no_grad():
        error = float((restored(**batch).logits - expected_logits).abs().max())
    receipt = {"kind": "tiny_random_qwen3_plumbing_only", "steps": steps, "initial_loss": initial,
               "final_loss": final, "initial_token_accuracy": initial_accuracy, "final_token_accuracy": final_accuracy,
               "checkpoint_steps": sorted(int(p.name.split("-")[-1]) for p in (output / "trainer").glob("checkpoint-*")),
               "base_weights_unchanged": frozen, "adapter_weights_changed": changed,
               "reload_max_logit_error": error,
               "passed": frozen and changed and final < initial * 0.95 and final_accuracy > initial_accuracy and error < 1e-5,
               "adapter_sha256": file_hash(output / "adapter" / "adapter_model.safetensors")}
    (output / "smoke.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["passed"]:
        raise RuntimeError(f"tiny smoke failed: {receipt}")
    return receipt
