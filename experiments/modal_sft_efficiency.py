"""Bounded B200 throughput benchmark on audited training records only."""
import hashlib,json,sys
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('uv==0.9.25')
    .add_local_file(ROOT/'training/pyproject.toml','/workspace/training/pyproject.toml',copy=True)
    .add_local_file(ROOT/'training/uv.lock','/workspace/training/uv.lock',copy=True)
    .run_commands('UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple uv sync --project /workspace/training --locked --no-dev')
    .env({'PYTHONPATH':'/workspace/src','TOKENIZERS_PARALLELISM':'false'})
    .add_local_dir(ROOT/'src','/workspace/src',ignore=['**/__pycache__/**']))
volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
app=modal.App('polytopia-sft-efficiency',image=image)

@app.function(gpu='B200',cpu=4,memory=16384,timeout=1800,max_containers=1,retries=0,
              scaledown_window=2,volumes={'/workspace/runs':volume})
def benchmark(protocol):
    import subprocess,time
    started=time.monotonic();root=Path('/workspace/runs')/protocol['run_id'];root.mkdir(parents=True,exist_ok=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    code=r'''
import gc,hashlib,json,math,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,set_seed
from peft import get_peft_model
from polytopia_bot.learning.training import load_tokenizer,lora_config
from polytopia_bot.learning.data import JsonlDataset,ResponseCollator
from polytopia_bot.learning.efficient_training import response_only_loss
os.chdir('/workspace');root=Path(os.environ['EFFICIENCY_OUTPUT']);p=json.loads((root/'protocol.json').read_text());c=p['config']
path=Path(p['prepared'])/'tokens.jsonl'
with path.open('rb') as f:
 if hashlib.file_digest(f,'sha256').hexdigest()!=p['tokens_sha256']:raise ValueError('audited training tokens changed')
data=JsonlDataset(path);lengths=[len(data[i]['input_ids']) for i in range(len(data))]
ordered=sorted(range(len(data)),key=lambda i:(lengths[i],i));chosen=[ordered[min(len(ordered)-1,int((j+.5)*len(ordered)/128))] for j in range(128)]
# Fixed randomized effective groups; all variants see identical examples per update.
gen=torch.Generator().manual_seed(17);chosen=[chosen[i] for i in torch.randperm(128,generator=gen).tolist()]
records=[data[i] for i in chosen];stress=[data[i] for i in ordered[-16:]]
set_seed(17);tok=load_tokenizer(c);collator=ResponseCollator(tok.pad_token_id)
m=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['revision'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False).to('cuda')
m.config.use_cache=False;m=get_peft_model(m,lora_config(c));m.train();m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
initial={n:v.detach().cpu().clone() for n,v in m.named_parameters() if v.requires_grad}
params=[v for v in m.parameters() if v.requires_grad]
def reset():
 m.zero_grad(set_to_none=True)
 with torch.no_grad():
  for n,v in m.named_parameters():
   if v.requires_grad:v.copy_(initial[n])
 gc.collect();torch.cuda.empty_cache()
def batch(rows):return {k:v.to('cuda') for k,v in collator(rows).items()}
@torch.autocast('cuda',dtype=torch.bfloat16)
def forward(inp,sparse,denom=None):
 return response_only_loss(m,inp,denom) if sparse else m(**inp,**({} if denom is None else {'num_items_in_batch':torch.as_tensor(denom,device='cuda')})).loss
# Verify the actual pinned model before benchmarking candidate arithmetic.
probe=batch([data[ordered[j]] for j in (0,250,500,750)])
normalizer=(probe['labels'][:,1:]!=-100).sum()
dense=forward(probe,False,normalizer);dense.backward();padded_gradient=torch.cat([v.grad.detach().float().flatten() for v in params]);padded_value=float(dense.detach());del dense
m.zero_grad(set_to_none=True);dense_value=0.
# The production baseline is microbatch1, with no padding, accumulated under
# one shared response-token denominator. Padded-batch kernels are diagnostic.
for index in (0,250,500,750):
 single=batch([data[ordered[index]]]);dense=forward(single,False,normalizer);dense.backward();dense_value+=float(dense.detach());del dense,single
a=torch.cat([v.grad.detach().float().flatten() for v in params])
m.zero_grad(set_to_none=True)
original_sdpa=torch.nn.functional.scaled_dot_product_attention
attention_calls=[]
def checked_sdpa(query,key,value,*args,**kwargs):
 mask=kwargs.get('attn_mask',args[0] if args else None)
 attention_calls.append(dict(mask_is_none=mask is None,is_causal=kwargs.get('is_causal',False)))
 return original_sdpa(query,key,value,*args,**kwargs)
torch.nn.functional.scaled_dot_product_attention=checked_sdpa
sparse=forward(probe,True,normalizer);sparse.backward();b=torch.cat([v.grad.detach().float().flatten() for v in params]);sparse_value=float(sparse.detach())
torch.nn.functional.scaled_dot_product_attention=original_sdpa
if not attention_calls or not all(x['mask_is_none'] and x['is_causal'] for x in attention_calls):raise ValueError('optimized path did not reach mask-free causal SDPA')
parity=dict(reference='production-unpadded-microbatch1',padded_dense_loss=padded_value,padded_vs_serial_relative_gradient_error=float(torch.linalg.vector_norm(padded_gradient-a)/torch.linalg.vector_norm(a)),dense_loss=dense_value,response_loss=sparse_value,relative_gradient_error=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a)),gradient_cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)))
parity['passed']=abs(dense_value-sparse_value)<.01 and parity['relative_gradient_error']<.03 and parity['gradient_cosine']>.999
(root/'parity.json').write_text(json.dumps(parity,indent=2))
if not parity['passed']:raise ValueError('pinned-model loss/gradient equivalence failed: '+str(parity))
del a,b,padded_gradient,probe,sparse;reset()
result=dict(schema='sft-throughput-benchmark-v1',hardware=torch.cuda.get_device_name(),total_gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,parity=parity,selected_record_indices=chosen,training_only=True,mask_free_causal_sdpa_verified=True,cuda_autocast='bfloat16',effective_batch=64,variants=[])
def save(): (root/'benchmark.json').write_text(json.dumps(result,indent=2))
save()
for sparse,micro in [(False,1),(True,1),(True,4),(True,8),(True,16)]:
 reset();opt=torch.optim.AdamW(params,lr=c['learning_rate'],weight_decay=0);entry=dict(response_only=sparse,microbatch=micro,effective_batch=64,length_sort_within_effective_batch=sparse,status='running');result['variants'].append(entry);save()
 try:
  # Unmeasured warmup has the same operation shape as a representative microbatch.
  warm=batch(records[:micro]);loss=forward(warm,sparse);loss.backward();m.zero_grad(set_to_none=True);del warm,loss
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic();token_count=0;padded_count=0;losses=[]
  for begin in (0,64):
   group=records[begin:begin+64]
   if sparse:group=sorted(group,key=lambda r:len(r['input_ids']),reverse=True)
   denom=sum(sum(t!=-100 for t in r['labels'][1:]) for r in group)
   total_loss=0.
   for offset in range(0,64,micro):
    chunk=group[offset:offset+micro];inp=batch(chunk);loss=forward(inp,sparse,denom)
    loss.backward();total_loss+=float(loss.detach());token_count+=sum(len(r['input_ids']) for r in chunk);padded_count+=inp['input_ids'].numel();del inp,loss
   torch.nn.utils.clip_grad_norm_(params,1.0);opt.step();opt.zero_grad(set_to_none=True);losses.append(total_loss)
  torch.cuda.synchronize();seconds=time.monotonic()-start;peak=torch.cuda.max_memory_allocated()
  # Consecutive longest actual records: no truncation or filtering of hard inputs.
  stress_input=batch(stress[-micro:]);stress_loss=forward(stress_input,sparse);stress_loss.backward();torch.cuda.synchronize()
  stress_peak=torch.cuda.max_memory_allocated();del stress_input,stress_loss
  entry.update(status='passed',seconds=seconds,examples=128,examples_per_second=128/seconds,tokens=token_count,tokens_per_second=token_count/seconds,padded_tokens=padded_count,peak_gpu_bytes=peak,stress_peak_gpu_bytes=stress_peak,stress_max_length=max(lengths),losses=losses)
 except torch.OutOfMemoryError:
  entry.update(status='oom');m.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
 finally:
  warm=loss=inp=stress_input=stress_loss=None
  m.zero_grad(set_to_none=True);del opt;gc.collect();torch.cuda.empty_cache();save();print(json.dumps(entry),flush=True)
valid=[r for r in result['variants'] if r['status']=='passed'];baseline=next(r for r in valid if not r['response_only']);best=min((r for r in valid if r['response_only']),key=lambda r:r['seconds']);result['selected']=best;result['measured_speedup']=baseline['seconds']/best['seconds'];result['note']='Two measured effective batches, plus warmup and longest-record stress. Engineering pilot, not gameplay evidence or a guarantee of full-run speed.';save()
'''
    import os
    try:
        proc=subprocess.run(['/workspace/training/.venv/bin/python','-c',code],env={**os.environ,'EFFICIENCY_OUTPUT':str(root)},timeout=1650,check=False)
        result={'exit_code':proc.returncode}
    except subprocess.TimeoutExpired:result={'error':'bounded benchmark timeout'}
    result['elapsed_seconds']=time.monotonic()-started
    result['benchmark']=json.loads((root/'benchmark.json').read_text()) if (root/'benchmark.json').exists() else None
    (root/'execution.json').write_text(json.dumps(result,indent=2));volume.commit()
    result['occupied_seconds_with_commit']=time.monotonic()-started
    return result

@app.local_entrypoint()
def main():
    from modal.config import config as modal_config
    if modal_config.get('environment')!='staging':raise ValueError('staging required')
    path=ROOT/'runs/sft0/prepared-repaired-smoke-v2/manifest.json';manifest=json.loads(path.read_text())
    protocol=dict(run_id='sft-efficiency-v6',prepared='/workspace/runs/sft0/prepared-repaired-smoke-v2',config=manifest['config'],tokens_sha256=manifest['tokens_sha256'],prepared_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds_limit=1800,compute_planning_allowance_usd=3.29,sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/polytopia_bot/learning/efficient_training.py']})
    out=ROOT/'runs/sft0/efficiency-training-benchmark-v6';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));call=benchmark.spawn(protocol);(out/'call.json').write_text(json.dumps({'call_id':call.object_id}));result=call.get();(out/'execution.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
