"""Independent H100 diagnostic; no gameplay or model work on the client."""
import json
from pathlib import Path
import modal
from experiments.modal_thinking_diagnostic import image, volume, ROOT

app = modal.App('polytopia-thinking-independent-diagnostic')


@app.function(image=image,gpu='H100',cpu=8,memory=49152,timeout=3600,
              retries=0,scaledown_window=2,volumes={'/workspace/runs':volume})
def run(run_id):
    import subprocess
    import time
    root=Path('/workspace/runs')/run_id
    volume.reload()
    try:
        protocol=json.loads((root/'protocol.json').read_text())
        for phase in protocol.get('phases',['batch-100','batch-40']):
            with (root/(phase+'.log')).open('w') as log:
                proc=subprocess.Popen(['python','/workspace/scripts/diagnose_thinking_parallelism.py',str(root),phase],
                    cwd='/workspace',stdout=log,stderr=subprocess.STDOUT)
                try:
                    while proc.poll() is None:
                        volume.commit();time.sleep(10)
                    if proc.returncode:
                        raise RuntimeError(f'{phase}: {proc.returncode}')
                finally:
                    if proc.poll() is None:
                        proc.terminate();proc.wait(timeout=15)
                    volume.commit()
        result=dict(status='complete')
    except Exception as exc:
        result=dict(status='failed',error=repr(exc))
        raise
    finally:
        (root/'status.json').write_text(json.dumps(result)+'\n')
        volume.commit()
    return result


@app.local_entrypoint()
def main(run_id: str):
    root=ROOT/'runs'/run_id
    if (root/'dispatch.json').exists():
        raise ValueError('already dispatched')
    with volume.batch_upload() as upload:
        for name in ['protocol.json','inputs.json']:
            upload.put_file(root/name,run_id+'/'+name)
    call=run.spawn(run_id)
    receipt=dict(app_id=app.app_id,call_id=call.object_id)
    (root/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
