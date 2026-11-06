"""Mac-side driver for Daytona sandboxes (no SDK: the REST API with the `daytona login` token, plus the CLI).

    python experiments/daytona/launch.py snapshot                      # build the dependency image as a snapshot
    python experiments/daytona/launch.py up NAME [--gpu N --gpu-type H100,RTX_PRO_6000 --spot --cpu C --memory G]
    python experiments/daytona/launch.py push NAME                     # code (git HEAD) + engine-rs wheel build
    python experiments/daytona/launch.py run NAME [--tag T] CMD ARGS... # node.py CMD, detached; /tmp/polytopia/
    python experiments/daytona/launch.py results NAME                  # cat /tmp/polytopia/results.jsonl
    python experiments/daytona/launch.py down NAME|--all               # delete (every polytopia=bench sandbox)
    python experiments/daytona/launch.py quote                         # hourly prices from PRICES below

Every sandbox gets label polytopia=bench, an auto-stop of 30 min (GPU sandboxes are deleted when stopped) and a
wall-clock TTL, so a forgotten sandbox stops billing by itself. Uses the CLI's active organization.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path.home() / 'Library/Application Support/daytona/config.json'
# Daytona list prices (daytona.io/pricing, read 2026-10-26), USD per hour, billed per second.
PRICES = dict(vcpu=0.0504, mem_gib=0.0162, disk_gib=0.000108,
              spot={'B300': 4.08, 'B200': 3.59, 'H200': 2.61, 'H100': 2.27, 'RTX_PRO_6000': 1.74,
                    'RTX_5090': 0.74, 'RTX_4090': 0.57, 'MI355X': 3.44},
              on_demand={'H100': 3.95})   # only the H100 on-demand rate is confirmed (third-party trackers)


def _profile():
    subprocess.run(['daytona', 'organization', 'list'], capture_output=True)   # refreshes an expired token
    c = json.loads(CONFIG.read_text())
    return next(p for p in c['profiles'] if p['id'] == c['activeProfile'])


def api(method, path, body=None, retries=5):
    p = _profile()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(p['api']['url'] + path, data=data, method=method, headers={
        'Authorization': 'Bearer ' + p['api']['token']['accessToken'],
        'X-Daytona-Organization-ID': p['activeOrganizationId'], 'Content-Type': 'application/json'})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            raise SystemExit(f'{method} {path}: HTTP {e.code} {e.read()[:500]!r}')
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:   # the Mac's network drops TLS handshakes
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def sandbox(name):
    rows = api('GET', '/sandbox?labels=' + urllib.request.quote(json.dumps({'polytopia': 'bench'})))
    rows = rows['items'] if isinstance(rows, dict) and 'items' in rows else rows
    match = [s for s in rows if s.get('name') == name or s.get('id') == name]
    if not match:
        raise SystemExit(f'no sandbox {name}')
    return match[0]


SNAPSHOT = 'polytopia-deps-v1'   # bump when the Dockerfile changes


def cmd_snapshot(a):
    name = a.name or SNAPSHOT
    args = ['daytona', 'snapshot', 'create', name, '-f', 'experiments/daytona/Dockerfile', '--cpu', '4', '--memory', '8',
            '--disk', '10']
    if a.region:
        args += ['--region', a.region]
    print(' '.join(args), flush=True)
    raise SystemExit(subprocess.run(args, cwd=ROOT).returncode)


def hourly(cpu, memory, disk, gpu=0, gpu_type=None, spot=True):
    rate = (PRICES['spot'] if spot else PRICES['on_demand']).get(gpu_type, float('nan')) if gpu else 0.
    return cpu * PRICES['vcpu'] + memory * PRICES['mem_gib'] + max(0, disk - 5) * PRICES['disk_gib'] + gpu * rate


def cmd_up(a):
    # A snapshot fixes the resources (the API refuses cpu/memory/gpu next to one), so sandboxes are created from the
    # Dockerfile itself; Daytona reuses the image it built for identical content (the `snapshot` command's build).
    image = dict(buildInfo=dict(dockerfileContent=(Path(__file__).parent / 'Dockerfile').read_text(), contextHashes=[]))
    body = dict(name=a.name, **({'snapshot': a.snapshot} if a.snapshot else image), cpu=a.cpu, memory=a.memory, disk=a.disk,
                labels={'polytopia': 'bench'}, autoStopInterval=30, ttlMinutes=a.ttl, autoArchiveInterval=0)
    gpu_types = a.gpu_type.split(',') if a.gpu_type else []
    if a.gpu:
        body.update(gpu=a.gpu, autoDeleteInterval=0)
        if gpu_types:
            body['gpuType'] = gpu_types
        if a.spot:
            body['spot'] = True
    else:
        body['autoDeleteInterval'] = 60   # CPU sandboxes: delete an hour after they stop
    quote = hourly(a.cpu, a.memory, a.disk, a.gpu, gpu_types[0] if gpu_types else None, a.spot)
    print(json.dumps(dict(request=body, quoted_usd_per_hour=round(quote, 3))), flush=True)
    s = api('POST', '/sandbox', body)
    deadline = time.time() + a.wait
    while s.get('state') not in ('started', 'error', 'build_failed') and time.time() < deadline:
        time.sleep(5)
        s = api('GET', f"/sandbox/{s['id']}")
    keep = ('id', 'name', 'state', 'cpu', 'memory', 'disk', 'gpu', 'gpuType', 'spot', 'target', 'errorReason', 'class')
    print(json.dumps({k: s.get(k) for k in keep}), flush=True)


def exec_(name, command, timeout=600):
    env = 'export PATH=/opt/venv/bin:/opt/cargo/bin:$PATH RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PYTHONPATH=/workspace/src; '
    return subprocess.run(['daytona', 'exec', name, '--timeout', str(timeout), '--', 'bash', '-c', env + command],
                          capture_output=True, text=True)


def push_node(name):
    """Write this checkout's node.py to /workspace/daytona/node.py in the sandbox."""
    import base64
    blob = base64.b64encode((Path(__file__).parent / 'node.py').read_bytes()).decode()
    must(name, f'mkdir -p /workspace/daytona /tmp/polytopia && echo {blob} | base64 -d > /workspace/daytona/node.py')


def cmd_push(a):
    """git archive of HEAD's src/, engine-rs/, scripts/ in ~100 kB exec chunks (exec retries ride out the Mac's
    TLS drops), then the engine-rs wheel built in the sandbox. Uncommitted changes are not sent."""
    import base64
    sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    blob = base64.b64encode(subprocess.run(['git', 'archive', '--format=tar.gz', 'HEAD', 'src', 'engine-rs', 'scripts'],
                                           capture_output=True, check=True, cwd=ROOT).stdout).decode()
    step = 100_000
    must(a.name, 'rm -f /tmp/code.b64')
    for i in range(0, len(blob), step):
        must(a.name, f'echo -n {blob[i:i + step]} >> /tmp/code.b64')
    must(a.name, 'cd /workspace && rm -rf src engine-rs scripts && base64 -d /tmp/code.b64 | tar xz && '
                 f'echo {sha} > /workspace/CODE_SHA && cd engine-rs && maturin build --release -q -o /workspace/wheels '
                 '&& pip install -q --force-reinstall /workspace/wheels/*.whl && rm -rf target', timeout=1800)
    push_node(a.name)
    print(json.dumps(dict(pushed=sha, bytes=len(blob) * 3 // 4)))


def must(name, command, timeout=600, retries=4):
    for attempt in range(retries):
        out = exec_(name, command, timeout)
        if out.returncode == 0:
            return out
        if not any(x in out.stderr for x in ('handshake', 'EOF', 'timeout', 'connection reset', 'failed to execute request')):
            break
        time.sleep(5 * (attempt + 1))
    raise SystemExit(f'exec failed ({out.returncode}): {command[:80]}...\n{out.stdout[-2000:]}{out.stderr[-2000:]}')


def cmd_run(a):
    tag = a.tag or a.args[0]
    push_node(a.name)
    cmd = ' '.join(shlex.quote(x) for x in a.args)
    out = must(a.name, f'cd /workspace && nohup python /workspace/daytona/node.py {cmd} '
                        f'> /tmp/polytopia/{tag}.log 2>&1 & echo started')
    print(out.stdout + out.stderr)


def cmd_results(a):
    out = must(a.name, 'cat /tmp/polytopia/results.jsonl 2>/dev/null; echo ---; tail -c 1500 /tmp/polytopia/*.log 2>/dev/null; true')
    print(out.stdout + out.stderr)


def cmd_down(a):
    rows = api('GET', '/sandbox?labels=' + urllib.request.quote(json.dumps({'polytopia': 'bench'})))
    rows = rows['items'] if isinstance(rows, dict) and 'items' in rows else rows
    for s in rows:
        if a.all or s.get('name') == a.name or s.get('id') == a.name:
            api('DELETE', f"/sandbox/{s['id']}")
            print('deleted', s.get('name'), s['id'])


def cmd_quote(a):
    for g, r in PRICES['spot'].items():
        for cpu, mem in ((16, 64), (16, 32), (8, 32)):
            print(f'{g:13s} spot  {cpu:2d} vCPU {mem:3d} GiB  ${hourly(cpu, mem, 50, 1, g):.2f}/h')
    print(f"CPU sandbox 4 vCPU 8 GiB: ${hourly(4, 8, 10):.3f}/h")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('snapshot'); p.add_argument('--name'); p.add_argument('--region')
    p = sub.add_parser('up'); p.add_argument('name'); p.add_argument('--snapshot'); p.add_argument('--cpu', type=int, default=4)
    p.add_argument('--memory', type=int, default=8); p.add_argument('--disk', type=int, default=10)
    p.add_argument('--gpu', type=int, default=0); p.add_argument('--gpu-type'); p.add_argument('--spot', action='store_true')
    p.add_argument('--ttl', type=int, default=90); p.add_argument('--wait', type=int, default=900)
    p = sub.add_parser('push'); p.add_argument('name')
    p = sub.add_parser('run'); p.add_argument('name'); p.add_argument('--tag')
    p.add_argument('args', nargs=argparse.REMAINDER)
    p = sub.add_parser('results'); p.add_argument('name')
    p = sub.add_parser('down'); p.add_argument('name', nargs='?'); p.add_argument('--all', action='store_true')
    sub.add_parser('quote')
    a = ap.parse_args()
    globals()['cmd_' + a.cmd](a)
