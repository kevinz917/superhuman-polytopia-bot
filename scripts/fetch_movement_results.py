"""Recover immutable reports and current receipts without launching GPU work."""
import json
import sys
from pathlib import Path
import modal


def main(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text())
    volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
    def fetch(remote,local):
        try:data=b''.join(volume.read_file(protocol['run_id']+'/'+remote))
        except FileNotFoundError:return None
        (root/local).write_bytes(data)
        return json.loads(data)
    state=fetch('status.json','cloud-status.json')
    completed=[]
    for step in protocol['checkpoint_steps']:
        path=root/f'report-{step}.json'
        if not path.exists():fetch(f'evaluation-{step}/results/report.json',path.name)
        if path.exists() and json.loads(path.read_text())['complete']:completed.append(step)
    if state and 'training' in state:
        (root/'training-execution.json').write_text(json.dumps(state['training'],indent=2)+'\n')
        if 'run' in state['training']:
            (root/'training-run.json').write_text(json.dumps(state['training']['run'],indent=2)+'\n')
    print(json.dumps(dict(status=None if state is None else state['status'],complete_checkpoints=completed,
                         training_status=None if not state else state.get('training',{}).get('run',{}).get('status'))))


if __name__=='__main__':main(sys.argv[1])
