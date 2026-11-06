"""Same frozen positions, evaluated with a versioned faster backend."""
import gzip,hashlib,json,sys,time
from collections import defaultdict
from pathlib import Path
from polytopia_bot.learning.fast_serving import ContinuousGenerator
from polytopia_bot.learning.offline import evaluate
from polytopia_bot.learning.interface import digest


def diagnostic_positions(rows,count):
    groups=defaultdict(list)
    for r in rows:groups[(r.get('phase'),r['teacher_command']['kind'])].append(r)
    for group in groups.values():group.sort(key=lambda r:hashlib.sha256(r['position_id'].encode()).hexdigest())
    selected=[]
    while len(selected)<count:
        for key in sorted(groups,key=str):
            if groups[key] and len(selected)<count:selected.append(groups[key].pop())
        if not any(groups.values()):break
    return selected


def main(p):
    rows=[]
    for file in p['shard_files']:
        path=Path('/evaluation')/file
        assert hashlib.sha256(path.read_bytes()).hexdigest()==p['uploads'][file]
        rows.extend(json.loads(x) for x in gzip.open(path,'rt'))
    if p.get('count'):rows=diagnostic_positions(rows,p['count'])
    if p.get('adapter_files'):
        for name,sha in p['adapter_files'].items():
            assert hashlib.sha256((Path(p['adapter'])/name).read_bytes()).hexdigest()==sha
    before=time.monotonic();g=ContinuousGenerator(p['config'],adapter=p.get('adapter'))
    for key,value in [('tokenizer_hash',digest(g.tokenizer.backend_tokenizer.to_str())),('chat_template_hash',digest(g.tokenizer.chat_template))]:
        if value!=p['prompt_manifest'][key]:raise ValueError('frozen tokenizer mismatch')
    load=time.monotonic()-before
    report=evaluate(p['output'],rows,g,batch_size=64,seconds_limit=p['seconds_limit'])
    summary={k:v for k,v in report.items() if k!='results'}
    summary.update(load_seconds=load,agreement=report['teacher_agreement']/max(1,report['decisions']),
                   validity=report['first_attempt_valid']/max(1,report['decisions']))
    summary['by_action_family']={}
    for row in report['results']:
        family=summary['by_action_family'].setdefault(str(row['family']),dict(decisions=0,teacher_agreement=0,first_attempt_valid=0))
        for key in family:family[key]+=1 if key=='decisions' else int(row[key])
    Path(p['output']).parent.joinpath('summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main(json.loads(Path(sys.argv[1]).read_text()))
