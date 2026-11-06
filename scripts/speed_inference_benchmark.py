"""Same fixed training prompts, separately loaded inference backends."""
import hashlib,json,sys,time
from pathlib import Path
from dataclasses import asdict

def main(p,output,mode):
    from polytopia_bot.learning.serving import BatchGenerator
    from polytopia_bot.learning.fast_serving import ContinuousGenerator
    path=Path(p['prepared'])/'examples.jsonl'
    assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==p['examples_sha256']
    rows=[json.loads(line) for i,line in enumerate(path.open()) if i%78==0][:128]
    started=time.monotonic()
    if mode in ('hf','hf_merged'):
        g=BatchGenerator(p['config'],adapter=p['adapter'])
        if mode=='hf_merged':g.model=g.model.merge_and_unload()
        batch_size=8
    else:
        g=ContinuousGenerator(p['config'],adapter=p['adapter'],concurrency=32,load_strategy=p.get('load_strategy','lazy'))
        batch_size=32
    load=time.monotonic()-started
    g.batch([r['messages'] for r in rows[:8]])
    started=time.monotonic();outputs=[]
    for i in range(0,len(rows),batch_size):outputs.extend(g.batch([r['messages'] for r in rows[i:i+batch_size]]))
    seconds=time.monotonic()-started
    report=dict(mode=mode,load_seconds=load,seconds=seconds,requests=len(rows),batch_size=batch_size,
        prompt_tokens=sum(len(x.prompt_token_ids) for x in outputs),output_tokens=sum(len(x.response_token_ids) for x in outputs),
        generations=[asdict(x) for x in outputs])
    Path(output).write_text(json.dumps(report));print(json.dumps({k:v for k,v in report.items() if k!='generations'}),flush=True)

if __name__=='__main__':main(json.loads(Path(sys.argv[1]).read_text()),sys.argv[2],sys.argv[3])
