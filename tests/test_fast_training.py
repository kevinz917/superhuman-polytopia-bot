import pytest
torch=pytest.importorskip('torch')
from polytopia_bot.learning.fast_training import pack,token_batches


def row(n,k=2):
    return dict(input_ids=list(range(n)),labels=[-100]*(n-k)+list(range(n-k,n)),attention_mask=[1]*n)


def test_packing_boundaries_and_causal_labels():
    rows=[row(7),row(11),row(5)]
    p=pack(rows)
    assert p['input_ids'].shape==(1,23)
    assert p['cu_seq_lens_q'].tolist()==[0,7,18,23]
    assert p['position_ids'].tolist()==[list(range(7))+list(range(11))+list(range(5))]
    assert p['max_length_q']==11
    assert all(p['labels'][0,i]==-100 for i in (0,7,18))
    assert (p['labels'][:,1:]!=-100).sum()==6


def test_token_budget_preserves_records_and_oversized_examples():
    rows=[row(n) for n in (7,20,9,5,8)]
    batches=list(token_batches(rows,16))
    assert sorted(id(r) for b in batches for r in b)==sorted(map(id,rows))
    assert all(sum(len(r['input_ids']) for r in b)<=16 or len(b)==1 for b in batches)
    assert len(batches[0])==1 and batches[0][0] is rows[1]


@pytest.mark.parametrize('bad', ['first_target','padding','length','empty_supervision'])
def test_pack_rejects_unsafe_records(bad):
    r=row(7)
    if bad=='first_target':r['labels'][0]=0
    if bad=='padding':r['attention_mask'][-1]=0
    if bad=='length':r['labels'].pop()
    if bad=='empty_supervision':r['labels']=[-100]*7
    with pytest.raises(ValueError):pack([r])


def test_segmented_sdpa_matches_independent_examples_and_gradients():
    from copy import deepcopy
    from test_efficient_training import model
    from polytopia_bot.learning.fast_training import set_backend,packed_response_loss
    from polytopia_bot.learning.efficient_training import response_only_loss
    from polytopia_bot.learning.data import ResponseCollator
    reference=model();candidate=deepcopy(reference);rows=[row(7),row(11),row(5)]
    denominator=sum(sum(x!=-100 for x in r['labels'][1:]) for r in rows)
    loss=0
    for r in rows:
        l=response_only_loss(reference,ResponseCollator(0)([r]),denominator);loss+=float(l.detach());l.backward()
    set_backend(candidate,'segmented_sdpa')
    other=packed_response_loss(candidate,pack(rows),denominator);other.backward()
    assert float(other.detach())==pytest.approx(loss,rel=1e-6)
    for (_,a),(_,b) in zip(reference.named_parameters(),candidate.named_parameters()):
        if a.requires_grad:torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=2e-6)


def test_distributed_partition_preserves_global_batch_and_sum_normalization():
    from copy import deepcopy
    from test_efficient_training import model
    from polytopia_bot.learning.distributed_training import partition_effective_batch
    from polytopia_bot.learning.efficient_training import response_only_loss
    from polytopia_bot.learning.data import ResponseCollator
    rows=[row(n) for n in [5,7,8,11,13,15,18]]
    parts=partition_effective_batch(rows,4)
    assert sorted(id(r) for b in parts for r in b)==sorted(map(id,rows))
    original=model();replicas=[deepcopy(original) for _ in parts];denom=2*len(rows)
    for r in rows:response_only_loss(original,ResponseCollator(0)([r]),denom).backward()
    for m,part in zip(replicas,parts):
        for r in part:response_only_loss(m,ResponseCollator(0)([r]),denom).backward()
    for name,p in original.named_parameters():
        if p.requires_grad:
            combined=sum(dict(m.named_parameters())[name].grad for m in replicas)
            torch.testing.assert_close(p.grad,combined,rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('bucket', [8, 256])
def test_segmented_cudnn_bucket_padding_is_exact(monkeypatch, bucket):
    # Padded (7->8, 11->16, 5->8) and unpadded (8) segments; GQA (4 query heads, 2 kv heads).
    from copy import deepcopy
    from test_efficient_training import model
    from polytopia_bot.learning import fast_training
    from polytopia_bot.learning.fast_training import set_backend,packed_response_loss
    monkeypatch.setattr(fast_training,'CUDNN_BUCKET',bucket)
    reference=model();candidate=deepcopy(reference);rows=[row(7),row(11),row(8),row(5)]
    set_backend(reference,'segmented_sdpa');set_backend(candidate,'segmented_cudnn')
    a=packed_response_loss(reference,pack(rows),9);a.backward()
    b=packed_response_loss(candidate,pack(rows),9);b.backward()
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    for (_,x),(_,y) in zip(reference.named_parameters(),candidate.named_parameters()):
        if x.requires_grad:torch.testing.assert_close(x.grad,y.grad,rtol=2e-5,atol=2e-6)


def test_adaptive_checkpointing_threshold():
    from polytopia_bot.learning.fast_training import chunk_checkpointing
    v=dict(checkpointing='adaptive',adaptive_tokens=10240)
    assert chunk_checkpointing(v,10240) is False and chunk_checkpointing(v,10241) is True
    assert chunk_checkpointing(dict(checkpointing='alternating'),99999)=='alternating'
    assert chunk_checkpointing(dict(checkpointing=False),99999) is False


def gpu_model(head_dim=64):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(7)
    c=Qwen3Config(vocab_size=64,hidden_size=128,intermediate_size=192,num_hidden_layers=2,num_attention_heads=4,
                  num_key_value_heads=2,head_dim=head_dim,max_position_embeddings=1024,attention_dropout=0,
                  tie_word_embeddings=False,pad_token_id=0)
    m=get_peft_model(Qwen3ForCausalLM(c).to(torch.bfloat16).cuda(),LoraConfig(r=2,lora_alpha=4,lora_dropout=0,
                     target_modules=['q_proj','v_proj','gate_proj'],task_type='CAUSAL_LM'))
    m.train();return m


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@cuda
def test_chunk_step_cudnn_matches_segmented_sdpa_and_adapts_checkpointing():
    from copy import deepcopy
    from polytopia_bot.learning.fast_training import set_backend,prewarm_cudnn
    from polytopia_bot.learning.expert_trainer import ChunkStep
    reference=gpu_model();candidate=deepcopy(reference)
    set_backend(reference,'segmented_sdpa');set_backend(candidate,'segmented_cudnn')
    assert prewarm_cudnn(candidate,600)==3
    def vocab_row(n):
        ids=[i%60+1 for i in range(n)];return dict(input_ids=ids,labels=[-100]*(n-2)+ids[-2:],attention_mask=[1]*n)
    rows=[vocab_row(n) for n in (300,257,41)]
    v=dict(backend='segmented_sdpa',checkpointing=False)
    calls=[]
    import polytopia_bot.learning.expert_trainer as et
    real=et.configure_checkpointing
    def spy(m,mode):calls.append(mode);real(m,mode)
    et.configure_checkpointing=spy
    try:
        step=ChunkStep(candidate,dict(backend='segmented_cudnn',checkpointing='adaptive',adaptive_tokens=400))
        b=step(rows[:2],6)+step(rows[2:],6)
        a=ChunkStep(reference,v)(rows[:2],6)+ChunkStep(reference,v)(rows[2:],6)
    finally:
        et.configure_checkpointing=real
    assert calls==[True,False]
    torch.testing.assert_close(a,b,rtol=2e-2,atol=2e-2)
    ga=torch.cat([p.grad.float().flatten() for p in reference.parameters() if p.requires_grad])
    gb=torch.cat([p.grad.float().flatten() for p in candidate.parameters() if p.requires_grad])
    assert float(torch.nn.functional.cosine_similarity(ga.double(),gb.double(),dim=0))>.99


@cuda
def test_liger_swiglu_patches_this_instance_only():
    pytest.importorskip('liger_kernel')
    import transformers.models.qwen3.modeling_qwen3 as mq
    from copy import deepcopy
    from polytopia_bot.learning.fast_training import apply_liger_swiglu
    original=mq.Qwen3MLP;reference=gpu_model();candidate=deepcopy(reference)
    assert apply_liger_swiglu(candidate)=='0.6.3'
    assert mq.Qwen3MLP is original
    mlp=candidate.get_base_model().model.layers[0].mlp
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    assert getattr(mlp.forward,'__func__',None) is LigerSwiGLUMLP.forward
    assert getattr(reference.get_base_model().model.layers[0].mlp.forward,'__func__',None) is not LigerSwiGLUMLP.forward
    x=torch.randint(1,60,(1,40),device='cuda')
    with torch.no_grad():
        torch.testing.assert_close(candidate(input_ids=x).logits,reference(input_ids=x).logits,rtol=3e-2,atol=3e-2)
