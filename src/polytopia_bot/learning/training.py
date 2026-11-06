"""Explicitly tokenized response-only LoRA using Transformers Trainer + PEFT.

The trainer receives audited labels directly: it cannot re-template, truncate,
pack, or accidentally supervise the game observation.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from .data import ResponseCollator, CorpusAudit, JsonlDataset, iter_records, tokenize_example
from .interface import canonical, digest


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_config(path):
    c = json.loads(Path(path).read_text())
    if len(c["revision"]) != 40 or any(x not in "0123456789abcdef" for x in c["revision"]):
        raise ValueError("model revision must be an immutable commit")
    if c["max_length"] <= c["max_new_tokens"] or c["microbatch_size"] < 1:
        raise ValueError("invalid token or batch limits")
    if c["effective_batch_size"] < 1 or c["effective_batch_size"] % c["microbatch_size"]:
        raise ValueError("effective batch must divide into microbatches")
    return c


def load_tokenizer(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"], trust_remote_code=False)
    if tokenizer.eos_token != "<|im_end|>":
        raise ValueError("expected Qwen end-of-message token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def prepare(data_path, output, config, *, allow_fixtures=False, split_manifest=None, prompt_manifest=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source_hash = file_hash(data_path)
    from .freeze import verify_prompt
    strategy = verify_prompt(prompt_manifest, config) if prompt_manifest else ""
    audit = CorpusAudit(allow_fixtures=allow_fixtures, split_manifest=split_manifest, strategy=strategy)
    tokenizer = load_tokenizer(config)
    if prompt_manifest and (digest(tokenizer.backend_tokenizer.to_str()) != prompt_manifest['tokenizer_hash'] or digest(tokenizer.chat_template) != prompt_manifest['chat_template_hash']):
        raise ValueError('tokenizer differs from frozen prompt')
    output.mkdir(parents=True, exist_ok=False)
    token_path = output / "tokens.jsonl"
    longest, processed, supervised = 0, 0, 0
    with token_path.open("w") as tokens, (output / "examples.jsonl").open("w") as examples:
        for row in iter_records(data_path):
            accepted = audit.accept(row)
            if accepted is None:
                continue
            _, messages, response = accepted
            encoded = tokenize_example(tokenizer, messages, response, max_length=config["max_length"], max_new_tokens=config["max_new_tokens"])
            tokens.write(canonical(encoded) + "\n")
            examples.write(canonical({"episode_id": row["episode_id"], "decision_index": row["decision_index"], "map_hash": row["map_hash"],
                                      "messages": messages, "response": response, "selected_command": row["selected_command"]}) + "\n")
            longest = max(longest, len(encoded["input_ids"]))
            processed += len(encoded["input_ids"])
            supervised += sum(x != -100 for x in encoded["labels"])
    if file_hash(data_path) != source_hash:
        raise ValueError("source data changed during preparation")
    tokenizer.save_pretrained(output / "tokenizer")
    manifest = {"schema": "polytopia-prepared-v1", "config": config, "config_hash": digest(config),
                "source_sha256": source_hash, "tokens_sha256": file_hash(token_path),
                "chat_template_hash": digest(tokenizer.chat_template), "tokenizer_hash": digest(tokenizer.backend_tokenizer.to_str()),
                "audit": audit.report(), "split_manifest": split_manifest, "prompt_manifest": prompt_manifest,
                "max_tokens": longest, "processed_tokens": processed, "supervised_tokens": supervised}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verified_data(path, config, *, stage, readiness=None):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["schema"] != "polytopia-prepared-v1" or manifest["config_hash"] != digest(config):
        raise ValueError("prepared data/config mismatch")
    if file_hash(path / "tokens.jsonl") != manifest["tokens_sha256"]:
        raise ValueError("prepared tokens changed after audit")
    if stage not in ("fixture", "fit", "smoke", "main", "scale40k"):
        raise ValueError("unknown training stage")
    if stage != "fixture":
        if manifest["audit"]["sources"] != ["teacher"] or not manifest["split_manifest"]:
            raise ValueError("real training requires teacher-only data and a frozen split manifest")
        if not readiness or readiness.get("full_game_ready") is not True or readiness.get("teacher_qualified") is not True:
            raise ValueError("real training requires environment and teacher qualification receipts")
        if sorted(readiness.get("teacher_hashes", [])) != manifest["audit"]["teacher_hashes"]:
            raise ValueError("qualified teacher does not match the corpus")
        if readiness.get("split_manifest_hash") != digest(manifest["split_manifest"]):
            raise ValueError("qualification and prepared split manifests differ")
        for key in ("environment_receipt", "teacher_receipt"):
            receipt = readiness.get(key, {})
            if not receipt.get("path") or file_hash(receipt["path"]) != receipt.get("sha256"):
                raise ValueError("missing or changed qualification evidence")
        from .freeze import verify_prompt
        verify_prompt(manifest.get('prompt_manifest'), config)
        expected = (128, 128) if stage == "fit" else (5000, 10000) if stage == "smoke" else (40000, 40000) if stage == "scale40k" else (50000, 50000)
        if not expected[0] <= manifest["audit"]["unique_records"] <= expected[1]:
            raise ValueError(f"{stage} requires {expected[0]}–{expected[1]} unique records")
    rows = JsonlDataset(path / "tokens.jsonl")
    if len(rows) != manifest["audit"]["unique_records"]:
        raise ValueError("prepared record count mismatch")
    for row in rows:
        ids, labels, mask = (row[k] for k in ("input_ids", "labels", "attention_mask"))
        if not ids or len(ids) != len(labels) or len(ids) != len(mask) or len(ids) > config["max_length"]:
            raise ValueError("invalid prepared lengths")
        supervised = [i for i, x in enumerate(labels) if x != -100]
        if not supervised or supervised[0] == 0 or supervised != list(range(supervised[0], len(ids))):
            raise ValueError("labels must be one response-only suffix")
        if len(supervised) > config["max_new_tokens"] or any(labels[i] != ids[i] for i in supervised) or any(x != 1 for x in mask):
            raise ValueError("invalid prepared mask")
    return manifest, rows


def lora_config(config):
    from peft import LoraConfig
    return LoraConfig(task_type="CAUSAL_LM", r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                      lora_dropout=config["lora_dropout"], target_modules=config["target_modules"], bias="none", revision=config.get("revision"))


def checkpoint_callback():
    import math
    from transformers import TrainerCallback

    class SaveFractions(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            control.should_save = state.global_step in {math.ceil(state.max_steps * fraction) for fraction in (0.25, 0.5, 1.0)}
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "grad_norm"):
                if key in (logs or {}) and not math.isfinite(logs[key]):
                    raise FloatingPointError(f"non-finite {key}")
    return SaveFractions()


def parameter_hash(model, *, trainable):
    """Hash actual tensors for the bounded fit's base/adapter update checks."""
    import torch
    h=hashlib.sha256()
    for name,parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            h.update(name.encode())
            raw=parameter.detach().cpu().contiguous().view(torch.uint8).numpy()
            h.update(memoryview(raw).cast('B'))
    return h.hexdigest()


def frozen_weight_hash(model):
    return parameter_hash(model, trainable=False)


def training_response_accuracy(model, rows, tokenizer, *, max_new_tokens, batch_size=4):
    """Greedy target-response accuracy, not inflated by easy JSON token matches."""
    import torch
    from .inference import decode_completion
    was_training=model.training
    model.eval();results=[]
    try:
        for start in range(0,len(rows),batch_size):
            chunk=[rows[i] for i in range(start,min(start+batch_size,len(rows)))]
            prefixes=[];targets=[]
            for row in chunk:
                boundary=next(i for i,label in enumerate(row['labels']) if label!=-100)
                prefixes.append(row['input_ids'][:boundary]);targets.append(row['input_ids'][boundary:])
            width=max(map(len,prefixes))
            ids=torch.tensor([[tokenizer.pad_token_id]*(width-len(p))+p for p in prefixes],device=model.device)
            mask=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prefixes],device=model.device)
            with torch.inference_mode():
                output=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,
                    max_new_tokens=max_new_tokens,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
                    use_cache=True)
            for i,(generated,target) in enumerate(zip(output,targets)):
                tokens=generated[width:].tolist()
                if tokenizer.eos_token_id in tokens:tokens=tokens[:tokens.index(tokenizer.eos_token_id)+1]
                text=decode_completion(tokenizer,tokens);expected=decode_completion(tokenizer,target)
                try:correct=json.loads(text)==json.loads(expected)
                except (ValueError,TypeError):correct=False
                results.append(dict(index=start+i,response=text,target=expected,correct=correct,response_token_ids=tokens))
    finally:
        model.train(was_training)
    return dict(examples=len(results),accuracy=sum(r['correct'] for r in results)/len(results),results=results)


def train(data_path, output, config, *, seed, stage, readiness=None):
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    manifest, rows = verified_data(data_path, config, stage=stage, readiness=readiness)
    if stage == "main" and seed not in config["seeds"]:
        raise ValueError("main training seed must be in the frozen configuration")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("4B BF16 training requires a CUDA BF16 GPU; use tiny-smoke for a local plumbing test")
    # This first runner is single-GPU; reject accidental distributed launches.
    import os
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("single-GPU runner; distributed effective-batch scaling is not implemented")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    started = time.time()
    receipt = {"status": "initializing", "stage": stage, "seed": seed, "config": config,
               "data_manifest": manifest, "readiness": readiness,
               "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate")},
               "source_hashes": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
               "hardware": torch.cuda.get_device_name(), "cuda": torch.version.cuda}
    def save_receipt():
        (output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    save_receipt()
    try:
        tokenizer = load_tokenizer(config)
        if digest(tokenizer.chat_template) != manifest["chat_template_hash"] or digest(tokenizer.backend_tokenizer.to_str()) != manifest["tokenizer_hash"]:
            raise ValueError("tokenizer differs from prepared data")
        model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["revision"],
                torch_dtype=torch.bfloat16, trust_remote_code=False, attn_implementation="sdpa")
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(config["model_id"], revision=config["revision"], local_files_only=True))
        receipt["base_files"] = {str(p.relative_to(snapshot)): file_hash(p) for p in snapshot.rglob("*")
                                  if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".txt", ".jinja")}
        model.config.use_cache = False
        model = get_peft_model(model, lora_config(config))
        if any(p.requires_grad and "lora_" not in n for n, p in model.named_parameters()):
            raise ValueError("non-LoRA parameters unexpectedly trainable")
        receipt["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        args = TrainingArguments(output_dir=str(output), per_device_train_batch_size=config["microbatch_size"],
                gradient_accumulation_steps=config["effective_batch_size"] // config["microbatch_size"],
                learning_rate=config["learning_rate"], num_train_epochs=config["epochs"],
                warmup_ratio=config["warmup_ratio"], lr_scheduler_type="cosine", max_grad_norm=config["max_grad_norm"],
                weight_decay=0, optim="adamw_torch", bf16=True, gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False}, save_strategy="no",
                logging_steps=1, logging_nan_inf_filter=False, report_to="none", seed=seed, data_seed=seed, remove_unused_columns=False,
                dataloader_num_workers=0)
        trainer = Trainer(model=model, args=args, train_dataset=rows, data_collator=ResponseCollator(tokenizer.pad_token_id),
                          callbacks=[checkpoint_callback()])
        if stage == 'fit':
            receipt['frozen_weights_before']=frozen_weight_hash(model)
            receipt['adapter_weights_before']=parameter_hash(model,trainable=True)
            receipt['fit_before']=training_response_accuracy(model,rows,tokenizer,max_new_tokens=config['max_new_tokens'])
            save_receipt()
        result = trainer.train()
        trainer.save_model(str(output / "adapter-final"))
        tokenizer.save_pretrained(output / "adapter-final")
        trainer.save_state()
        if stage == 'fit':
            receipt['fit_after']=training_response_accuracy(model,rows,tokenizer,max_new_tokens=config['max_new_tokens'])
            receipt['frozen_weights_after']=frozen_weight_hash(model)
            receipt['adapter_weights_after']=parameter_hash(model,trainable=True)
            if receipt['frozen_weights_before']!=receipt['frozen_weights_after']:
                raise ValueError('frozen base weights changed during the fit check')
            if receipt['adapter_weights_before']==receipt['adapter_weights_after']:
                raise ValueError('adapter weights did not change during the fit check')
        receipt.update(status="completed", metrics=result.metrics, elapsed_seconds=time.time() - started,
                       max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                       adapter_files={p.name: file_hash(p) for p in (output / "adapter-final").glob("adapter*") if p.is_file()})
    except Exception as e:
        receipt.update(status="failed", error_type=type(e).__name__, error=str(e), elapsed_seconds=time.time() - started)
        raise
    finally:
        save_receipt()
    return receipt
