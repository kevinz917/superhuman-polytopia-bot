"""Common movement/control evaluation for the baseline and continuation adapters."""
import gzip,hashlib,json,sys,time
from pathlib import Path
from polytopia_bot.learning.fast_serving import ContinuousGenerator
from polytopia_bot.learning.interface import digest
from polytopia_bot.learning.offline import evaluate


def main(input_path):
    p=json.loads(Path(input_path).read_text());root=Path(p['root']);protocol=json.loads((root/'protocol.json').read_text())
    if hashlib.sha256((root/'positions.jsonl.gz').read_bytes()).hexdigest()!=protocol['positions_sha256']:raise ValueError('positions changed')
    rows=[json.loads(line) for line in gzip.open(root/'positions.jsonl.gz','rt')]
    if len(rows)!=protocol['evaluation_count']:raise ValueError('evaluation count changed')
    for name,sha in p['adapter_files'].items():
        if hashlib.sha256((Path(p['adapter'])/name).read_bytes()).hexdigest()!=sha:raise ValueError('adapter changed')
    started=time.monotonic();g=ContinuousGenerator(protocol['config'],adapter=p['adapter'])
    for key,value in [('tokenizer_hash',digest(g.tokenizer.backend_tokenizer.to_str())),('chat_template_hash',digest(g.tokenizer.chat_template))]:
        if value!=protocol['prompt_manifest'][key]:raise ValueError('tokenizer changed')
    load=time.monotonic()-started
    report=evaluate(p['output'],rows,g,batch_size=protocol['batch_size'],seconds_limit=protocol['evaluation_seconds_limit'])
    summary={k:v for k,v in report.items() if k!='results'};summary['load_seconds']=load
    Path(p['output']).parent.joinpath('summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not report['complete']:raise ValueError('checkpoint evaluation incomplete')


if __name__=='__main__':main(sys.argv[1])
