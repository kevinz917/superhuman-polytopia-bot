"""One-decision Transformers backend; GPU game coordination is separate."""
from .interface import Generation
from .training import load_tokenizer


def decode_completion(tokenizer, token_ids):
    # Only the terminal EOS is structural. Preserve other special tokens so
    # malformed generation cannot become valid JSON merely through decoding.
    content = token_ids[:-1] if token_ids and token_ids[-1] == tokenizer.eos_token_id else token_ids
    return tokenizer.decode(content, skip_special_tokens=False)


class LocalGenerator:
    def __init__(self, config, *, adapter=None, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM
        self.config = config
        self.tokenizer = load_tokenizer(config)
        self.model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["revision"],
                torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32, trust_remote_code=False)
        if adapter:
            from peft import PeftConfig, PeftModel
            saved = PeftConfig.from_pretrained(adapter)
            if saved.base_model_name_or_path != config["model_id"] or saved.revision != config["revision"]:
                raise ValueError("adapter targets a different base model/revision")
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.to(device).eval()

    def __call__(self, messages):
        import torch
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(ids) + self.config["max_new_tokens"] > self.config["max_length"]:
            raise ValueError("prompt exceeds context budget; no catalog truncation")
        tensor = torch.tensor([ids], device=self.model.device)
        with torch.inference_mode():
            result = self.model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), do_sample=False,
                    max_new_tokens=self.config["max_new_tokens"], pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id)
        response = result[0, len(ids):].tolist()
        return Generation(decode_completion(self.tokenizer, response), ids, response)
