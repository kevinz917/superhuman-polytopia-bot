"""Cloud-only fixed-output serving probe and natural-completion prompt probe."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time

from benchmark_thinking_repaired import ObservableGenerator
from polytopia_bot.learning.async_gameplay import atomic_json
from polytopia_bot.learning.thinking_readable import readable_messages, restore_sparse_body


async def main(root, phase):
    if os.environ.get('MODAL_TASK_ID') is None:
        raise RuntimeError('diagnostic inference must run on Modal')
    root = Path(root)
    p = json.loads((root/'protocol.json').read_text())
    samples = json.loads((root/'inputs.json').read_text())
    p['serving'] = dict(enforce_eager=phase=='speed-eager')
    p['concurrency'] = len(samples)
    g = ObservableGenerator(p, True, root/phase, probe=True)
    # Fixed generated length isolates engine throughput from EOS/prompt behavior.
    # Never used as a gameplay limit, and no actions from this speed probe execute.
    if phase.startswith('speed-'):
        from vllm import SamplingParams
        async def fixed(sample, number):
            ids=g.tokenizer.apply_chat_template(sample['messages'], tokenize=True,
                add_generation_prompt=True,enable_thinking=True)
            started=time.monotonic(); tokens=0
            async for out in g.engine.generate(dict(prompt_token_ids=ids),
                    SamplingParams(temperature=0, max_tokens=512, min_tokens=512,
                        ignore_eos=True, detokenize=False), str(number)):
                tokens=len(out.outputs[0].token_ids)
            return dict(tokens=tokens, seconds=time.monotonic()-started)
        # One warm-up batch, then a measured batch; both configurations identical.
        await asyncio.gather(*(fixed(s,'warm-'+str(i)) for i,s in enumerate(samples)))
        start=time.monotonic()
        rows=await asyncio.gather(*(fixed(s,i) for i,s in enumerate(samples)))
        elapsed=time.monotonic()-start
        atomic_json(root/(phase+'.json'),dict(rows=rows,seconds=elapsed,
            output_tokens=sum(r['tokens'] for r in rows),tokens_per_second=sum(r['tokens'] for r in rows)/elapsed))
    else:
        rows=[]
        if (root/'off-inputs.json').exists():
            g.thinking=False
            for sample in json.loads((root/'off-inputs.json').read_text()):
                messages=readable_messages(sample['messages'])
                output,timing=await g.generate(messages,job=dict(sample['job'],id='off-'+sample['job']['id']),decision=sample['decision'])
                valid=False
                try:
                    action=json.loads(output.text)
                    valid=set(action)=={'action_id'} and action['action_id'] in {r[0] for r in json.loads(sample['messages'][1]['content'])['legal_actions']['rows']}
                except (ValueError,TypeError):pass
                rows.append(dict(profile='off_named_constrained',valid=valid,**timing))
                atomic_json(root/'prompt-results.json',rows)
            g.thinking=True
        for profile in ['sparse_brief','named']:
            async def one(sample):
                messages=sample['messages']
                if profile=='named':
                    messages=readable_messages(messages)
                    assert restore_sparse_body(json.loads(messages[1]['content']))==json.loads(sample['messages'][1]['content'])
                job=dict(sample['job'],id=profile+'-'+sample['job']['id'])
                output,timing=await g.generate(messages,job=job,decision=sample['decision'])
                body=json.loads(sample['messages'][1]['content'])
                valid=False
                if timing['finish_reason']=='stop' and output.text.count('</think>')==1:
                    try:
                        action=json.loads(output.text.split('</think>',1)[1])
                        valid=set(action)=={'action_id'} and action['action_id'] in {r[0] for r in body['legal_actions']['rows']}
                    except (ValueError,TypeError):pass
                rows.append(dict(profile=profile,valid=valid,**timing))
                atomic_json(root/'prompt-results.json', rows)
            await asyncio.gather(*(one(s) for s in samples))
    g.log.close();g.engine.shutdown()


if __name__=='__main__':asyncio.run(main(*sys.argv[1:]))
