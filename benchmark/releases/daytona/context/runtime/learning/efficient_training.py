"""Versioned single-GPU SFT optimizations; the original runner stays frozen.

Compute the unchanged causal response-token loss without projecting ignored
prompt positions through the vocabulary head. Preserve effective-batch groups
when sorting examples by length to reduce padding.
"""


def response_only_loss(model, inputs, num_items_in_batch=None):
    import torch
    from torch.nn import functional as F

    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if base.config.model_type != 'qwen3':
        raise ValueError('response-only backbone path is validated for Qwen3 only')
    if getattr(base.config, 'attention_dropout', 0) != 0:
        raise ValueError('dropout would change length-reordering equivalence')
    labels = inputs['labels'][:, 1:]
    selected = labels != -100
    if not bool(selected.any()):
        raise ValueError('at least one supervised causal token required')
    if bool((inputs['labels'][:, 0] != -100).any()):
        raise ValueError('the first token cannot be a causal target')
    mask = inputs['attention_mask']
    if mask.shape != inputs['input_ids'].shape or mask.shape != inputs['labels'].shape:
        raise ValueError('attention mask, tokens, and labels must have identical shapes')
    if bool(((mask != 0) & (mask != 1)).any()) or bool((mask[:, 0] != 1).any()) or bool((mask[:, 1:] > mask[:, :-1]).any()):
        raise ValueError('strict nonempty right padding is required')
    if bool(((mask == 0) & (inputs['labels'] != -100)).any()):
        raise ValueError('padded positions cannot be supervised')
    # Causality prevents any real token attending the padded tail. No loss is
    # taken on that tail, so omitting this redundant padding mask preserves
    # real-token outputs/gradients and permits fused causal SDPA kernels.
    # An explicit all-valid 2D mask also bypasses Transformers4.57 packed-
    # sequence detection, which otherwise materializes a redundant 4D mask.
    hidden = base.model(input_ids=inputs['input_ids'], attention_mask=torch.ones_like(mask),
                        use_cache=False, return_dict=True).last_hidden_state
    logits = base.lm_head(hidden[:, :-1][selected]).float()
    denominator = selected.sum() if num_items_in_batch is None else num_items_in_batch
    denominator = torch.as_tensor(denominator, device=logits.device)
    if bool(denominator <= 0):
        raise ValueError('positive response-token denominator required')
    return F.cross_entropy(logits, labels[selected], reduction='sum') / denominator


class EffectiveBatchLengthSampler:
    """Shuffle once; sort only inside each unchanged effective-batch window."""
    def __init__(self, lengths, *, seed, effective_batch_size=64, sort_lengths=True):
        if not lengths or min(lengths) < 1 or effective_batch_size < 1:
            raise ValueError('nonempty positive lengths and batch size required')
        self.lengths = list(lengths)
        self.seed = seed
        self.window = effective_batch_size
        self.sort_lengths = sort_lengths
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        import torch
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.lengths), generator=generator).tolist()
        for start in range(0, len(order), self.window):
            group = order[start:start + self.window]
            if self.sort_lengths:
                group.sort(key=lambda i: (-self.lengths[i], i))
            yield from group


def trainer_class():
    from transformers import Trainer

    class ResponseOnlyTrainer(Trainer):
        def __init__(self, *args, length_order=None, effective_batch_size=64, **kwargs):
            super().__init__(*args, **kwargs)
            if self.accelerator.num_processes != 1:
                raise ValueError('only single-GPU loss normalization is validated')
            self.model_accepts_loss_kwargs = True
            self.length_order = length_order
            self.effective_batch_size = effective_batch_size

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Accelerator wraps the outer model.forward with CUDA autocast.
            # Our direct backbone call bypasses that wrapper, so enter the
            # identical precision context explicitly (including LoRA layers).
            import torch
            from contextlib import nullcontext
            context = (torch.autocast('cuda', dtype=torch.bfloat16)
                       if inputs['input_ids'].device.type == 'cuda' and self.args.bf16
                       else nullcontext())
            with context:
                loss = response_only_loss(model, inputs, num_items_in_batch)
            return (loss, {'loss': loss}) if return_outputs else loss

        def _get_train_sampler(self, train_dataset=None):
            if self.length_order is None:
                return super()._get_train_sampler(train_dataset)
            return EffectiveBatchLengthSampler(self.length_order, seed=self.args.data_seed,
                effective_batch_size=self.effective_batch_size)

    return ResponseOnlyTrainer
