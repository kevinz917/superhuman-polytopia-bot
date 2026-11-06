"""Fetch only small persisted diagnostic reports and logs from Modal."""
import json
from pathlib import Path
import sys
import modal


def main(root):
    root=Path(root)
    p=json.loads((root/'protocol.json').read_text())
    volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
    names=['status.json','natural-results.json']
    for phase in p.get('phases',['batch-100','batch-40']):
        names.extend([phase+'-throughput.json',phase+'.log',phase+'-progress.json'])
    for name in names:
        try:
            data=b''.join(volume.read_file(root.name+'/'+name))
        except FileNotFoundError:
            continue
        (root/name).write_bytes(data)
        if name.endswith('.log'):
            print(name, data.decode()[-1800:])
        else:
            result=json.loads(data)
            if name.endswith('-progress.json'):
                active=list(result['pending'].values())
                result=dict(active=len(active),output_tokens=sum(x['output_tokens'] for x in active),
                    max_output_tokens=max((x['output_tokens'] for x in active),default=0))
            if isinstance(result,dict):
                result.pop('rows',None)
            print(name,json.dumps(result))


if __name__=='__main__':
    main(sys.argv[1])
