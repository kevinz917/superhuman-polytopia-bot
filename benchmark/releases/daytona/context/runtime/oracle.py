"""Validation-only client. GPU runtime modules never import this module."""
import json
from pathlib import Path
import subprocess


class Oracle:
    def __init__(self, binary=None):
        root = Path(__file__).resolve().parents[2]
        binary = Path(binary or root / "vendor/Polyfish/polyfish-rs/target/debug/polyfish-oracle")
        if not binary.is_file():
            raise FileNotFoundError("Build the validation oracle with scripts/bootstrap_oracle.py")
        self.process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def request(self, **request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"oracle exited ({self.process.poll()})")
        response = json.loads(line)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
