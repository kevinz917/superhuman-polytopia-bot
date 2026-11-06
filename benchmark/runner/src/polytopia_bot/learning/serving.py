"""Batched unchanged/base-adapter generation with independently logged inputs."""
from .inference import LocalGenerator, decode_completion
from .interface import Generation


class BatchGenerator(LocalGenerator):
    def batch(self, messages):
        import torch
        if not messages:
            return []
        rows = [self.tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True) for m in messages]
        if any(len(row) + self.config['max_new_tokens'] > self.config['max_length'] for row in rows):
            raise ValueError('prompt exceeds context budget; no catalog truncation')
        longest = max(map(len, rows))
        # Left padding leaves every generation position at the same sequence end.
        ids = [[self.tokenizer.pad_token_id] * (longest-len(row)) + row for row in rows]
        masks = [[0] * (longest-len(row)) + [1] * len(row) for row in rows]
        tensor = torch.tensor(ids, device=self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(input_ids=tensor,
                attention_mask=torch.tensor(masks, device=self.model.device), do_sample=False,
                max_new_tokens=self.config['max_new_tokens'], pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        result=[]
        for prompt, generated_row in zip(rows, generated):
            tokens=generated_row[longest:].tolist()
            if self.tokenizer.eos_token_id in tokens:
                tokens=tokens[:tokens.index(self.tokenizer.eos_token_id)+1]
            result.append(Generation(decode_completion(self.tokenizer,tokens),prompt,tokens))
        return result
