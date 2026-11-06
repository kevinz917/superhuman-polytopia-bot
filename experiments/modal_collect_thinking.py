"""Durable CPU-only collection of already-dispatched gameplay jobs."""
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.12')
       .add_local_file(ROOT/'scripts/summarize_unrestricted_games.py','/workspace/scripts/summarize_unrestricted_games.py'))
volume=modal.Volume.from_name('polytopia-sft0-training',environment_name='staging')
app=modal.App('polytopia-thinking-result-collector')


@app.function(image=image,cpu=.125,memory=512,timeout=86400,max_containers=1,
              retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def collect(run_id,calls,protocol_sha256):
    import hashlib,json,subprocess,time,traceback
    root=Path('/workspace/runs')/run_id
    volume.reload()
    if hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()!=protocol_sha256:
        raise ValueError('protocol changed')
    status=dict(status='waiting',calls=calls,started_unix=time.time(),gpu_jobs_launched=0)
    (root/'collection.json').write_text(json.dumps(status,indent=2)+'\n');volume.commit()
    try:
        results={}
        for mode,call_id in calls.items():
            results[mode]=modal.FunctionCall.from_id(call_id).get()
        volume.reload()
        subprocess.run(['python','/workspace/scripts/summarize_unrestricted_games.py',str(root)],check=True)
        summary=json.loads((root/'summary.json').read_text())
        lines=['# Unrestricted Qwen3.5-4B gameplay comparison','',
               'Same unadapted model and current harness; 40 matched games per mode. No separate output or thinking token budget.','',
               '| Result | Thinking off | Thinking on |','|---|---:|---:|']
        a,b=summary['modes']['off'],summary['modes']['on']
        for title,key in [('Native wins','native_wins'),('Native losses','native_losses'),
            ('Invalid-action forfeits','forfeits'),('Valid actions','valid_actions'),
            ('Completed model calls','calls'),('Total output tokens','output_tokens'),
            ('Longest completed response','max_output_tokens')]:
            lines.append(f'| {title} | {a[key]} | {b[key]} |')
        lines+=['',f"Thinking off statuses: `{a['status_counts']}`.",
                f"Thinking on statuses: `{b['status_counts']}`.",'',
                'Truncations, context exhaustion and infrastructure errors are not wins, losses or draws. '
                'Forfeits are operational failures, distinct from native game losses.','',
                f"Paired comparison: `{summary['paired_win_difference']}`.",'',
                'See protocol.json and summary.json for provenance, full per-game outcomes, timing and limitations.','']
        (root/'result.md').write_text('\n'.join(lines))
        status.update(status='complete',results=results,finished_unix=time.time())
    except Exception:
        status.update(status='failed',error=traceback.format_exc(),finished_unix=time.time())
        raise
    finally:
        (root/'collection.json').write_text(json.dumps(status,indent=2)+'\n')
        volume.commit()
    return status


@app.local_entrypoint()
def main(run_id: str='qwen35-thinking-games-unrestricted-v1'):
    import hashlib,json
    from modal.config import config
    if config.get('environment')!='staging':raise ValueError('staging required')
    root=ROOT/'runs'/run_id
    receipt=root/'collector-call.json'
    if receipt.exists():raise ValueError('collector already dispatched')
    calls=json.loads((root/'calls.json').read_text())
    sha=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
    call=collect.spawn(run_id,calls,sha)
    receipt.write_text(json.dumps(dict(call_id=call.object_id,protocol_sha256=sha),indent=2)+'\n')
    print(receipt.read_text(),flush=True)
    print(call.get(),flush=True)
