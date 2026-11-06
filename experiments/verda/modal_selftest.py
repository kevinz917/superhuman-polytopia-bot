"""control.py tests on Modal (no secrets, no Verda calls): modal run --env staging experiments/verda/modal_selftest.py"""
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version='3.12').pip_install('pytest>=9,<10')
         .add_local_file(HERE / 'control.py', '/ctl/control.py')
         .add_local_file(HERE / 'test_control.py', '/ctl/test_control.py'))
app = modal.App('verda-controller-selftest')


@app.function(image=image, timeout=300)
def run():
    out = subprocess.run(['python', '-m', 'pytest', '-q', 'test_control.py'], cwd='/ctl', capture_output=True,
                         text=True)
    return out.returncode, out.stdout[-4000:] + out.stderr[-2000:]


@app.local_entrypoint()
def main():
    code, text = run.remote()
    print(text)
    if code:
        raise SystemExit(code)
