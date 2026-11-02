"""Read Daytona quotas and optionally run one disposable platform setup probe.

No model calls, benchmark authoring, paid API fallback, or account changes.
The probe is a generic platform check, not benchmark end-to-end validation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import urllib.request
import uuid


def credentials():
    """Reuse existing host credentials; never persist or copy them to a sandbox."""
    base = os.environ.get('DAYTONA_API_URL', 'https://app.daytona.io/api')
    key = os.environ.get('DAYTONA_API_KEY')
    jwt = os.environ.get('DAYTONA_JWT_TOKEN')
    org = os.environ.get('DAYTONA_ORGANIZATION_ID')
    if key or jwt:
        if not org:
            raise ValueError('Set DAYTONA_ORGANIZATION_ID with explicit Daytona credentials for organization checks.')
        return dict(api_url=base, api_key=key, jwt_token=jwt, organization_id=org)
    config = Path.home()/'Library/Application Support/daytona/config.json'
    data = json.loads(config.read_text())
    profile = next(p for p in data['profiles'] if p['id'] == data['activeProfile'])
    api = profile['api']
    return dict(api_url=api['url'], api_key=api.get('key'),
                jwt_token=(api.get('token') or {}).get('accessToken'),
                organization_id=profile['activeOrganizationId'])


def get_json(auth, path):
    headers = {'Authorization': 'Bearer '+(auth['api_key'] or auth['jwt_token'])}
    if auth.get('organization_id'):
        headers['X-Daytona-Organization-ID'] = auth['organization_id']
    request = urllib.request.Request(auth['api_url'].rstrip('/')+path, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', action='store_true', help='Create and delete one tiny sandbox; uses a small amount of credit.')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    auth = credentials()
    organizations = get_json(auth, '/organizations')
    org = next(o for o in organizations if o['id'] == auth['organization_id'])
    usage = get_json(auth, '/organizations/'+org['id']+'/usage')
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        organization=org['name'], organization_id=org['id'],
        api_url=auth['api_url'], default_region=org['defaultRegionId'],
        suspended=org['suspended'],
        per_sandbox_limits=dict(cpu=org['maxCpuPerSandbox'], memory_gib=org['maxMemoryPerSandbox'],
                                disk_gib=org['maxDiskPerSandbox']),
        resource_usage=usage, wallet_balance=None,
        wallet_note='Not exposed by the installed CLI/SDK; no credit balance was inferred.',
        actual_model_calls=0, benchmark_end_to_end_verified=False,
        integration_remaining=['Replace Docker-only agent attachment with Daytona transport.',
            'Build separate x86_64 coder/verifier snapshots with the pinned benchmark toolchain.',
            'Verify the trusted cutoff, independent verifier, artifact transfer and deadline behavior.',
            'Load-test proxy concurrency before the full campaign.'])
    sandbox = None
    if args.probe:
        from daytona import Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams
        daytona = Daytona(DaytonaConfig(**auth, otel_enabled=False))
        name = 'polytopia-skirmish-setup-'+uuid.uuid4().hex[:10]
        started = time.monotonic()
        try:
            sandbox = daytona.create(CreateSandboxFromSnapshotParams(
                name=name, snapshot='daytonaio/sandbox:0.11.0', os_user='root',
                target=org['defaultRegionId'], public=False, network_block_all=True,
                auto_stop_interval=5, auto_delete_interval=0, ttl_minutes=15,
                labels={'purpose': 'polytopia-skirmish-platform-probe'}), timeout=120)
            payload = b'polytopia-skirmish-daytona-filesystem-roundtrip\n'
            remote = '/tmp/polytopia-skirmish-probe.txt'
            sandbox.fs.upload_file(payload, remote)
            downloaded = sandbox.fs.download_file(remote)
            assert downloaded == payload, 'remote file transfer mismatch'
            code = '''import json,os,platform,socket
from pathlib import Path
s=socket.socket();s.settimeout(2)
d=dict(uid=os.getuid(),architecture=platform.machine(),python=platform.python_version(),
 memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
 cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
 credential_env_names=[k for k in os.environ if any(w in k.upper() for w in ['API_KEY','TOKEN','SECRET'])],
 blocked_network_connect_result=s.connect_ex(('1.1.1.1',443)))
print(json.dumps(d))'''
            command = 'python3 -c '+shlex.quote(code)
            response = sandbox.process.exec(command, timeout=30)
            assert response.exit_code == 0, 'remote execution failed'
            details = json.loads(response.result)
            default_uid = details['uid']
            if default_uid != 0:
                # The stock snapshot's toolbox runs as daytona even when create
                # requests root. Explicit privilege selection must be tested.
                response = sandbox.process.exec('sudo -n '+command, timeout=30)
                assert response.exit_code == 0, 'trusted root execution failed'
                details = json.loads(response.result)
            assert details['uid'] == 0, 'trusted controller requires root access'
            assert not details['credential_env_names'], 'unexpected credential environment names'
            assert details['blocked_network_connect_result'] != 0, 'unexpected outbound connectivity'
            report['platform_probe'] = dict(passed=True, sandbox_id=sandbox.id,
                name=name, snapshot='daytonaio/sandbox:0.11.0', runtime=details,
                default_exec_uid=default_uid, root_exec_prefix='sudo -n' if default_uid else '',
                network_block_all=True, private=True,
                filesystem_roundtrip_sha256=hashlib.sha256(payload).hexdigest())
        finally:
            if sandbox is not None:
                daytona.delete(sandbox)
                report.setdefault('platform_probe', {}).update(deleted=True,
                    elapsed_seconds=time.monotonic()-started)
            report['resource_usage_after_cleanup'] = get_json(auth, '/organizations/'+org['id']+'/usage')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
