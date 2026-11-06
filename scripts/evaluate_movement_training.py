"""Inference only on the frozen checkpoint's actual movement training inputs."""
import gzip,hashlib,json,sys,time
from pathlib import Path
from polytopia_bot.learning.fast_serving import ContinuousGenerator
from polytopia_bot.learning.interface import digest
from polytopia_bot.learning.offline import evaluate


def main(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text())
    path=root/'training-positions.jsonl.gz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['positions_sha256']:raise ValueError('position hash mismatch')
    rows=[json.loads(line) for line in gzip.open(path,'rt')]
    if len(rows)!=protocol['train_count']:raise ValueError('position count mismatch')
    for name,sha in protocol['adapter_files'].items():
        if hashlib.sha256((Path(protocol['adapter'])/name).read_bytes()).hexdigest()!=sha:raise ValueError('adapter hash mismatch')
    started=time.monotonic();generator=ContinuousGenerator(protocol['config'],adapter=protocol['adapter'])
    for key,value in [('tokenizer_hash',digest(generator.tokenizer.backend_tokenizer.to_str())),('chat_template_hash',digest(generator.tokenizer.chat_template))]:
        if value!=protocol['prompt_manifest'][key]:raise ValueError('tokenizer mismatch')
    load=time.monotonic()-started
    report=evaluate(root/'training-results',rows,generator,batch_size=protocol['batch_size'],seconds_limit=protocol['seconds_limit'])
    summary={k:v for k,v in report.items() if k!='results'};summary['load_seconds']=load
    (root/'inference-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not report['complete']:raise ValueError('incomplete diagnostic inference')


if __name__=='__main__':main(sys.argv[1])
