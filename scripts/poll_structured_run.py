"""Read the durable function-call handle; never resubmit on a polling timeout."""
import argparse
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from polytopia_bot.rl.budget import record


def poll(run_id):
    folder = ROOT/'runs/structured-ppo-v3'
    ledger = json.loads((folder/'budget.json').read_text())
    row = next(r for r in ledger['runs'] if r['run_id'] == run_id)
    call = modal.FunctionCall.from_id(row['function_call_id'])
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        return dict(run_id=run_id, state='pending', function_call_id=row['function_call_id'])
    (folder/(run_id+'-summary.json')).write_text(json.dumps(result,indent=2)+'\n')
    record(folder, run_id, status='complete')
    return dict(run_id=run_id, state='complete', evaluations={
        mode:{k:v for k,v in report.items() if k not in ('details','action_counts')}
        for mode,report in result.get('evaluations',{}).items()})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_id')
    print(json.dumps(poll(parser.parse_args().run_id),indent=2))
