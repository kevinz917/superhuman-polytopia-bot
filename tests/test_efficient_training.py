"""Mathematical equivalence and batch boundaries for the optimized SFT path."""
from copy import deepcopy
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('peft')
from polytopia_bot.learning.data import ResponseCollator
from polytopia_bot.learning.efficient_training import response_only_loss, EffectiveBatchLengthSampler


def model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(7)
    c = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, attention_dropout=0,
        tie_word_embeddings=False, pad_token_id=0)
    return get_peft_model(Qwen3ForCausalLM(c), LoraConfig(r=2, lora_alpha=4,
        lora_dropout=0, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))


def rows():
    return [dict(input_ids=list(range(1,n+1)), attention_mask=[1]*n,
                 labels=[-100]*(n-k)+list(range(n-k+1,n+1)))
            for n,k in [(9,2),(15,3),(7,1),(12,4)]]


def test_sparse_response_head_matches_dense_loss_and_all_lora_gradients():
    dense=model(); sparse=deepcopy(dense);batch=ResponseCollator(0)(rows())
    loss=dense(**batch).loss; other=response_only_loss(sparse,batch)
    torch.testing.assert_close(loss,other,rtol=1e-6,atol=1e-6)
    loss.backward();other.backward()
    for (n,p),(m,q) in zip(dense.named_parameters(),sparse.named_parameters()):
        assert n==m
        if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-6)
        else:assert p.grad is None and q.grad is None


@pytest.mark.parametrize('microbatch',[1,2,4])
def test_response_token_normalization_preserves_accumulated_gradient(microbatch):
    reference=model(); candidate=deepcopy(reference);data=rows();collate=ResponseCollator(0)
    response_only_loss(reference,collate(data)).backward()
    count=sum(sum(t!=-100 for t in r['labels'][1:]) for r in data)
    for start in range(0,len(data),microbatch):
        response_only_loss(candidate,collate(data[start:start+microbatch]),count).backward()
    for (_,p),(_,q) in zip(reference.named_parameters(),candidate.named_parameters()):
        if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-6)


def test_length_order_preserves_every_effective_batch_and_partial_tail():
    lengths=[(i*17)%53+1 for i in range(139)]
    plain=EffectiveBatchLengthSampler(lengths,seed=17,sort_lengths=False)
    grouped=EffectiveBatchLengthSampler(lengths,seed=17)
    for epoch in [0,1]:
        plain.set_epoch(epoch);grouped.set_epoch(epoch);a=list(plain);b=list(grouped)
        assert sorted(b)==list(range(139))
        for start in range(0,len(a),64):
            assert set(a[start:start+64])==set(b[start:start+64])
            assert [lengths[i] for i in b[start:start+64]]==sorted([lengths[i] for i in a[start:start+64]],reverse=True)


def test_invalid_mask_is_not_silently_dropped():
    m=model();batch=ResponseCollator(0)(rows());batch['labels'][:,0]=1
    with pytest.raises(ValueError,match='first token'):response_only_loss(m,batch)


def test_trainer_accumulation_does_not_divide_response_loss_twice(tmp_path):
    from transformers import TrainingArguments
    from polytopia_bot.learning.efficient_training import trainer_class
    outcomes=[]
    for microbatch in [1,4]:
        m=model();optim=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=.01)
        trainer=trainer_class()(model=m,train_dataset=rows(),data_collator=ResponseCollator(0),
            length_order=[len(r['input_ids']) for r in rows()],effective_batch_size=4,
            optimizers=(optim,torch.optim.lr_scheduler.LambdaLR(optim,lambda _:1)),
            args=TrainingArguments(output_dir=str(tmp_path/str(microbatch)),use_cpu=True,
                per_device_train_batch_size=microbatch,gradient_accumulation_steps=4//microbatch,
                max_steps=1,save_strategy='no',logging_strategy='no',report_to='none',
                disable_tqdm=True,seed=17,data_seed=17,remove_unused_columns=False))
        trainer.train()
        outcomes.append({n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad})
    for n in outcomes[0]:torch.testing.assert_close(outcomes[0][n],outcomes[1][n],rtol=1e-5,atol=1e-7)


@pytest.mark.parametrize('kind', ['left', 'interior', 'nonbinary', 'supervised'])
def test_omitted_padding_mask_rejects_unsafe_inputs(kind):
    m=model();batch=ResponseCollator(0)(rows())
    if kind == 'left': batch['attention_mask'][0,0]=0
    elif kind == 'interior': batch['attention_mask'][0,3]=0
    elif kind == 'nonbinary': batch['attention_mask'][0,3]=2
    else: batch['labels'][0,-1]=1
    with pytest.raises(ValueError,match='right padding|padded positions'):
        response_only_loss(m,batch)


def test_right_padding_uses_mask_free_causal_sdpa(monkeypatch):
    from torch.nn import functional as F
    original=F.scaled_dot_product_attention
    calls=[]
    def inspect_attention(query,key,value,*args,**kwargs):
        mask=kwargs.get('attn_mask',args[0] if args else None)
        calls.append((mask,kwargs.get('is_causal',False)))
        return original(query,key,value,*args,**kwargs)
    monkeypatch.setattr(F,'scaled_dot_product_attention',inspect_attention)
    m=model();m.get_base_model().set_attn_implementation('sdpa')
    response_only_loss(m,ResponseCollator(0)(rows())).backward()
    assert calls and all(mask is None and causal for mask,causal in calls)
