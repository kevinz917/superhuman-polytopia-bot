"""Fetch small diagnostic receipts and logs; never download model checkpoints."""
import json
from pathlib import Path
import sys
import modal


def main(root):
    root=Path(root)
    volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
    for name in ['speed-eager.json','speed-graphs.json','prompt-results.json',
                 'speed-eager.log','speed-graphs.log','prompts-graphs.log']:
        try:data=b''.join(volume.read_file(root.name+'/'+name))
        except FileNotFoundError:continue
        (root/name).write_bytes(data)
        if name.endswith('.log'):
            print(name,data.decode()[-900:])
        elif name=='prompt-results.json':
            rows=json.loads(data)
            for profile in sorted({r['profile'] for r in rows}):
                selected=[r for r in rows if r['profile']==profile]
                print(profile,json.dumps(dict(calls=len(selected),valid=sum(r['valid'] for r in selected),
                    mean_tokens=sum(r['output_tokens'] for r in selected)/len(selected),
                    max_tokens=max(r['output_tokens'] for r in selected),
                    mean_seconds=sum(r['request_seconds'] for r in selected)/len(selected))))
        else:print(name,data.decode())
    receipt=json.loads((root/'dispatch.json').read_text())
    try:print('call',modal.FunctionCall.from_id(receipt['call_id']).get(timeout=0))
    except TimeoutError:print('call still running')


if __name__=='__main__':main(sys.argv[1])
