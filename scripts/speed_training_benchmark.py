"""Remote fixed-work benchmark with GPU gradient parity and kernel profile."""
import gc, hashlib, json, math, os, time
from pathlib import Path
import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM, set_seed
from polytopia_bot.learning.training import load_tokenizer, lora_config
from polytopia_bot.learning.data import JsonlDataset, ResponseCollator
from polytopia_bot.learning.efficient_training import response_only_loss
from polytopia_bot.learning.fast_training import pack, token_batches, packed_response_loss, configure_checkpointing, set_backend, set_lora_input_cast


def main(protocol, root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    c=protocol['config'];source=Path(protocol['prepared'])/'tokens.jsonl'
    assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==protocol['tokens_sha256']
    data=JsonlDataset(source);lengths=[len(data[i]['input_ids']) for i in range(len(data))]
    order=sorted(range(len(data)),key=lambda i:(lengths[i],i))
    chosen=[order[min(len(order)-1,int((j+.5)*len(order)/128))] for j in range(128)]
    gen=torch.Generator().manual_seed(17);chosen=[chosen[i] for i in torch.randperm(128,generator=gen).tolist()]
    rows=[data[i] for i in chosen];stress=[data[i] for i in order[-8:]]
    set_seed(17);tok=load_tokenizer(c);collate=ResponseCollator(tok.pad_token_id)
    model=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['revision'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    model.config.use_cache=False;model=get_peft_model(model,lora_config(c));model.train()
    params=[v for v in model.parameters() if v.requires_grad]
    initial={n:v.detach().cpu().clone() for n,v in model.named_parameters() if v.requires_grad}
    def reset():
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n,v in model.named_parameters():
                if v.requires_grad:v.copy_(initial[n])
        gc.collect();torch.cuda.empty_cache()
    def batch(r):return {k:v.cuda() for k,v in collate(r).items()}
    def loss(r,variant,denom):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return packed_response_loss(model,pack(r,'cuda'),denom) if variant['packed'] else response_only_loss(model,batch(r),denom)
    def chunks(group,v):
        if v['packed']:return token_batches(group,v['token_budget'])
        g=sorted(group,key=lambda r:len(r['input_ids']),reverse=True)
        if 'token_budget' in v:return token_batches(g,v['token_budget'])
        return (g[i:i+4] for i in range(0,len(g),4))
    result=dict(schema='speed-training-benchmark-v2',training_only=True,config=c,
                hardware=torch.cuda.get_device_name(),gpu_memory=torch.cuda.get_device_properties(0).total_memory,
                selected_indices=chosen,variants=[],parity={},profiles={})
    def save(): (root/'benchmark.json').write_text(json.dumps(result,indent=2))
    probe=[data[order[j]] for j in (0,250,500,750)]
    denom=sum(sum(t!=-100 for t in r['labels'][1:]) for r in probe)
    configure_checkpointing(model,True)
    # Original serial reference, same denominator across unpadded examples.
    ref_loss=0.
    for row in probe:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            l=response_only_loss(model,batch([row]),denom)
        l.backward();ref_loss+=float(l.detach())
    ref=torch.cat([v.grad.detach().float().flatten() for v in params]);reset()
    model.get_base_model().set_attn_implementation('flash_attention_2')
    l=loss(probe,{'packed':True},denom);l.backward()
    grad=torch.cat([v.grad.detach().float().flatten() for v in params])
    result['parity']=dict(reference_loss=ref_loss,packed_loss=float(l.detach()),
        relative_gradient_error=float(torch.linalg.vector_norm(ref-grad)/torch.linalg.vector_norm(ref)),
        cosine=float(torch.nn.functional.cosine_similarity(ref,grad,dim=0)))
    result['parity']['passed']=abs(ref_loss-float(l.detach()))<.01 and result['parity']['relative_gradient_error']<.03 and result['parity']['cosine']>.999
    save();print('PARITY',json.dumps(result['parity']),flush=True)
    del l,grad;reset()
    variants=[dict(name='current_sdpa_micro4',packed=False,checkpointing=True)]
    variants += [dict(name=f'segmented_{budget}_checkpoint_{cp}',packed=True,backend='segmented_sdpa',token_budget=budget,checkpointing=cp)
                 for cp,budget in [(True,16384),('alternating',16384),(False,8192)]]
    variants += [dict(name=f'sdpa_budget_{budget}_checkpoint_{cp}',packed=False,token_budget=budget,checkpointing=cp)
                 for cp,budget in [(False,8192),('alternating',16384)]]
    variants += [dict(name=f'castfree_segmented_{budget}_checkpoint_{cp}',packed=True,backend='segmented_sdpa',token_budget=budget,checkpointing=cp,skip_lora_cast=True)
                 for cp,budget in [('alternating',16384),(False,8192)]]
    for v in variants:
        set_lora_input_cast(model,not v.get('skip_lora_cast',False))
        reset();configure_checkpointing(model,v['checkpointing'])
        set_backend(model,v.get('backend','flash_attention_2' if v['packed'] else 'sdpa'))
        entry={**v,'status':'running'};result['variants'].append(entry);save()
        opt=torch.optim.AdamW(params,lr=c['learning_rate'],weight_decay=0)
        try:
            pv=0.
            for chunk in chunks(probe,v):
                pl=loss(chunk,v,denom);pl.backward();pv+=float(pl.detach());del pl
            pg=torch.cat([p.grad.detach().float().flatten() for p in params])
            parity=dict(loss=pv,relative_gradient_error=float(torch.linalg.vector_norm(ref-pg)/torch.linalg.vector_norm(ref)),cosine=float(torch.nn.functional.cosine_similarity(ref,pg,dim=0)))
            parity['passed']=abs(pv-ref_loss)<.01 and parity['relative_gradient_error']<.03 and parity['cosine']>.999
            entry['parity']=parity;del pg;model.zero_grad(set_to_none=True)
            if not parity['passed']:
                entry['status']='parity_failed'
                continue
            warm=next(iter(chunks(rows[:8],v)));loss(warm,v,denom).backward();model.zero_grad(set_to_none=True)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic();fw=bw=0.;losses=[]
            for begin in (0,64):
                group=rows[begin:begin+64];normalizer=sum(sum(x!=-100 for x in r['labels'][1:]) for r in group);total=0.
                for chunk in chunks(group,v):
                    events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
                    events[0].record();l=loss(chunk,v,normalizer);events[1].record();l.backward();events[2].record()
                    torch.cuda.synchronize();fw+=events[0].elapsed_time(events[1])/1000;bw+=events[1].elapsed_time(events[2])/1000
                    total+=float(l.detach());del l
                norm=torch.nn.utils.clip_grad_norm_(params,c['max_grad_norm'])
                if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
                opt.step();opt.zero_grad(set_to_none=True);losses.append(total)
            torch.cuda.synchronize();elapsed=time.monotonic()-started;peak=torch.cuda.max_memory_allocated()
            for chunk in chunks(stress,v):loss(chunk,v,None).backward();model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            entry.update(status='passed',seconds=elapsed,forward_seconds=fw,backward_seconds=bw,
                         peak_bytes=peak,stress_peak_bytes=torch.cuda.max_memory_allocated(),stress_max_length=max(lengths),
                         examples=128,tokens=sum(len(r['input_ids']) for r in rows),losses=losses)
        except torch.OutOfMemoryError:
            entry.update(status='oom');l=None
        finally:
            model.zero_grad(set_to_none=True);del opt;gc.collect();torch.cuda.empty_cache();save();print('VARIANT',json.dumps(entry),flush=True)
    valid=[r for r in result['variants'] if r['status']=='passed'];baseline=next(r for r in valid if r['name']=='current_sdpa_micro4')
    best=min(valid,key=lambda v:v['seconds']);result['selected']=best;result['speedup']=baseline['seconds']/best['seconds'];result['parity']=best['parity']
    # Separate profiling run, outside measured timings, for baseline and winner.
    for v in [baseline,best]:
        set_lora_input_cast(model,not v.get('skip_lora_cast',False));reset();configure_checkpointing(model,v['checkpointing']);set_backend(model,v.get('backend','flash_attention_2' if v['packed'] else 'sdpa'))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            loss(next(iter(chunks(rows[:8],v))),v,None).backward();torch.cuda.synchronize()
        result['profiles'][v['name']]=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=20)
    save();return result

if __name__=='__main__':
    import sys
    p=json.loads(Path(sys.argv[1]).read_text());main(p,sys.argv[2])
