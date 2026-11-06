"""Four-GPU throughput and sum-normalization check on the same 128 records."""
import json,os,time,gc,sys,hashlib
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM,set_seed
from peft import get_peft_model
from polytopia_bot.learning.training import lora_config
from polytopia_bot.learning.data import JsonlDataset,ResponseCollator
from polytopia_bot.learning.efficient_training import response_only_loss
from polytopia_bot.learning.fast_training import pack,token_batches,packed_response_loss,configure_checkpointing,set_backend,set_lora_input_cast
from polytopia_bot.learning.distributed_training import partition_effective_batch,sum_gradients


def main(p,output):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
 c=p['config'];v=p['benchmark']['selected'];source=Path(p['prepared'])/'tokens.jsonl'
 assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==p['tokens_sha256']
 data=JsonlDataset(source);rows=[data[i] for i in p['benchmark']['selected_indices']]
 lengths=[len(data[i]['input_ids']) for i in range(len(data))];order=sorted(range(len(data)),key=lambda i:(lengths[i],i))
 probe=[data[order[j]] for j in (0,250,500,750)];stress=[data[i] for i in order[-8:]]
 set_seed(17);model=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['revision'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
 model.config.use_cache=False;model=get_peft_model(model,lora_config(c));model.train();params=[x for x in model.parameters() if x.requires_grad]
 initial=[x.detach().clone() for x in params];collate=ResponseCollator(151643)
 configure_checkpointing(model,True)
 denom=sum(sum(x!=-100 for x in r['labels'][1:]) for r in probe)
 if rank==0:
  reference_loss=0.
  for row in probe:
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=response_only_loss(model,{k:t.cuda() for k,t in collate([row]).items()},denom)
   loss.backward();reference_loss+=float(loss.detach())
  reference=torch.cat([x.grad.detach().float().flatten() for x in params]);del loss
 model.zero_grad(set_to_none=True);dist.barrier()
 set_backend(model,v.get('backend','flash_attention_2' if v['packed'] else 'sdpa'));configure_checkpointing(model,v['checkpointing']);set_lora_input_cast(model,not v.get('skip_lora_cast',False))
 def parts(group):
  local=partition_effective_batch(group,world)[rank]
  return token_batches(local,v['token_budget']) if 'token_budget' in v else (local[i:i+4] for i in range(0,len(local),4))
 def backward(group):
  normalizer=sum(sum(x!=-100 for x in r['labels'][1:]) for r in group);total=torch.zeros((),device='cuda')
  for chunk in parts(group):
   with torch.autocast('cuda',dtype=torch.bfloat16):
    loss=packed_response_loss(model,pack(chunk,'cuda'),normalizer) if v['packed'] else response_only_loss(model,{k:t.cuda() for k,t in collate(chunk).items()},normalizer)
   loss.backward();total+=loss.detach()
  sum_gradients(params);dist.all_reduce(total,op=dist.ReduceOp.SUM);return total
 value=backward(probe)
 if rank==0:
  gradient=torch.cat([x.grad.detach().float().flatten() for x in params])
  parity=dict(reference_loss=reference_loss,distributed_loss=float(value),relative_gradient_error=float(torch.linalg.vector_norm(reference-gradient)/torch.linalg.vector_norm(reference)),cosine=float(torch.nn.functional.cosine_similarity(reference,gradient,dim=0)))
  parity['passed']=abs(reference_loss-float(value))<.01 and parity['relative_gradient_error']<.03 and parity['cosine']>.999
  passed=torch.tensor(int(parity['passed']),device='cuda');del reference,gradient
 else:passed=torch.tensor(0,device='cuda')
 dist.broadcast(passed,0)
 if not passed.item():raise ValueError('distributed gradient parity failed')
 model.zero_grad(set_to_none=True);backward(rows[:64]);model.zero_grad(set_to_none=True)
 with torch.no_grad():
  for x,original in zip(params,initial):x.copy_(original)
 opt=torch.optim.AdamW(params,lr=c['learning_rate'],weight_decay=0);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();dist.barrier();start=time.monotonic();losses=[]
 for begin in (0,64):
  value=backward(rows[begin:begin+64]);torch.nn.utils.clip_grad_norm_(params,c['max_grad_norm']);opt.step();opt.zero_grad(set_to_none=True);losses.append(float(value))
 torch.cuda.synchronize();seconds=torch.tensor(time.monotonic()-start,device='cuda');dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
 backward(stress);model.zero_grad(set_to_none=True);peak=torch.tensor(torch.cuda.max_memory_allocated(),device='cuda');dist.all_reduce(peak,op=dist.ReduceOp.MAX)
 if rank==0:
  baseline=next(x['seconds'] for x in p['benchmark']['variants'] if x['name']=='current_sdpa_micro4')
  r=dict(schema='distributed-speed-benchmark-v2',world_size=world,hardware=torch.cuda.get_device_name(),parity=parity,
   seconds=float(seconds),single_current_seconds=baseline,single_selected_seconds=v['seconds'],
   wall_speedup_vs_current=baseline/float(seconds),gpu_seconds_ratio_vs_current=world*float(seconds)/baseline,
   selected=v,peak_bytes=int(peak),examples=128,tokens=sum(len(r['input_ids']) for r in rows),losses=losses)
  Path(output).write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
 dist.destroy_process_group()

if __name__=='__main__':main(json.loads(Path(sys.argv[1]).read_text()),sys.argv[2])
