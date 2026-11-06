"""Continuous request scheduler for the unchanged public prompt and output ABI."""
from .interface import Generation
from .inference import decode_completion
from .training import load_tokenizer


class ContinuousGenerator:
    def __init__(self, config, *, adapter=None, merged_model=None, concurrency=32, load_strategy='lazy'):
        from vllm import LLMEngine, EngineArgs, SamplingParams
        from vllm.lora.request import LoRARequest
        self.config=config
        if adapter:
            from peft import PeftConfig
            saved=PeftConfig.from_pretrained(adapter)
            if saved.base_model_name_or_path!=config['model_id'] or saved.revision!=config['revision']:
                raise ValueError('adapter model/revision differs')
        self.tokenizer=load_tokenizer(config)
        self.engine=LLMEngine.from_engine_args(EngineArgs(
            model=merged_model or config['model_id'],revision=None if merged_model else config['revision'],
            safetensors_load_strategy=load_strategy,
            tokenizer=config['model_id'],tokenizer_revision=config['revision'],
            dtype='bfloat16',max_model_len=config['max_length'],gpu_memory_utilization=.85,
            max_num_seqs=concurrency,max_num_batched_tokens=8192,
            enable_prefix_caching=True,enable_chunked_prefill=True,
            enable_lora=bool(adapter and not merged_model),max_lora_rank=32,
            disable_log_stats=True,seed=17))
        self.sampling=SamplingParams(temperature=0,max_tokens=config['max_new_tokens'],
            stop_token_ids=[self.tokenizer.eos_token_id],detokenize=False)
        self.adapter=LoRARequest('policy',1,adapter) if adapter and not merged_model else None
        self.prompts={};self.sequence=0

    def submit(self, messages):
        ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        if len(ids)+self.config['max_new_tokens']>self.config['max_length']:
            raise ValueError('prompt exceeds frozen context budget')
        request=str(self.sequence);self.sequence+=1;self.prompts[request]=ids
        self.engine.add_request(request,{'prompt_token_ids':ids},self.sampling,lora_request=self.adapter)
        return request

    def step(self):
        finished=[]
        for output in self.engine.step():
            if output.finished:
                tokens=list(output.outputs[0].token_ids)
                finished.append((output.request_id,Generation(decode_completion(self.tokenizer,tokens),
                    self.prompts.pop(output.request_id),tokens)))
        return finished

    def batch(self,messages):
        requests=[self.submit(m) for m in messages];results={}
        while len(results)<len(requests):results.update(self.step())
        return [results[r] for r in requests]
