"""Build private x86_64 Daytona snapshots from the shared Harbor build context."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.build_skirmish_harbor import prepare_context
from scripts.check_skirmish_daytona import credentials
from polytopia_bot.skirmish.common import canonical, sha, write_json
from polytopia_bot.skirmish.continuous import source_files
from polytopia_bot.skirmish.local import checked_references


def bundle(path, root, members):
    """Stable bytes permit remote build-cache reuse across package revisions."""
    def normalize(member):
        member.mtime = 0
        member.uid = member.gid = 0
        member.uname = member.gname = ''
        member.pax_headers = {}
        return member
    with path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0,compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed,mode='w') as archive:
                for member in members:
                    archive.add(root/member,arcname=member,filter=normalize)


def snapshot_ready(daytona, params):
    from daytona.common.errors import DaytonaNotFoundError
    try:
        snapshot = daytona.snapshot.get(params.name)
    except DaytonaNotFoundError:
        # Build logs are an optional separate endpoint. Poll metadata instead:
        # account JWTs can build successfully even if that log endpoint denies access.
        snapshot = daytona.snapshot.create(params, timeout=1800)
    deadline = time.monotonic()+1800
    while snapshot.state.value not in ('active','error','build_failed'):
        if time.monotonic()>deadline:
            raise TimeoutError('Daytona snapshot build deadline')
        time.sleep(3)
        snapshot = daytona.snapshot.get(snapshot.id)
    if snapshot.state.value != 'active':
        raise RuntimeError('Daytona snapshot build failed: '+str(snapshot.error_reason))
    print(json.dumps(dict(snapshot=snapshot.name, state=snapshot.state.value)), flush=True)
    return snapshot


def build(package, shared_package, references):
    from daytona import Daytona, DaytonaConfig, Image, Resources, CreateSnapshotParams
    package = package.resolve()
    package.mkdir(parents=True, exist_ok=False)
    sources = source_files()
    shared_package = shared_package.resolve()
    manifest = json.loads((shared_package/'manifest.json').read_text())
    maps = json.loads((shared_package/'maps.json').read_text())
    checked_references(manifest, references)
    write_json(package/'manifest.json', manifest)
    write_json(package/'maps.json', maps)
    context = package/'context'
    context.mkdir()
    prepare_context(context, manifest, maps, references)
    daytona = Daytona(DaytonaConfig(**credentials(), otel_enabled=False))
    public = ['engine-rs','bot','mini','sdk','rules.json']
    private = ['runtime','refs','tests','manifest.json','maps.json']
    # Single archive per boundary avoids directory-dropping context upload bugs.
    # Crucially, the coder build NEVER receives the private archive in its context.
    for name, members in [('public',public),('private',private)]:
        bundle(context/f'{name}.tar.gz',context,members)
    original = (context/'Dockerfile').read_text().split('FROM base AS referee')[0]
    lines = []
    inserted_context = False
    for line in original.splitlines():
        if line.startswith('COPY '):
            if not inserted_context:
                lines += ['COPY public.tar.gz /tmp/public.tar.gz',
                    'RUN mkdir /build-context && tar -xzf /tmp/public.tar.gz -C /build-context && rm /tmp/public.tar.gz']
                inserted_context = True
            _, source, destination = line.split()
            if (context/source).is_dir():
                line = f'RUN mkdir -p {destination} && cp -a /build-context/{source}/. {destination}/'
            else:
                line = f'RUN mkdir -p $(dirname {destination}) && cp -a /build-context/{source} {destination}'
        lines.append(line)
    lines += ['RUN rm -rf /build-context', 'USER root']
    coder_text = '\n'.join(lines)+'\n'
    coder_dir = package/'coder-build'
    coder_dir.mkdir()
    shutil.copyfile(context/'public.tar.gz', coder_dir/'public.tar.gz')
    (coder_dir/'Dockerfile').write_text(coder_text)
    coder_stamp = sha(coder_text.encode()+(context/'public.tar.gz').read_bytes())[:12]
    coder = snapshot_ready(daytona, CreateSnapshotParams(
        name=f'polytopia-skirmish-coder-{coder_stamp}',
        image=Image.from_dockerfile(coder_dir/'Dockerfile',strict_context=True),
        resources=Resources(cpu=2,memory=2,disk=10), region_id='us',entrypoint=['sleep','infinity']))
    verifier_dir = package/'verifier-build'
    verifier_dir.mkdir()
    shutil.copyfile(context/'public.tar.gz', verifier_dir/'public.tar.gz')
    shutil.copyfile(context/'private.tar.gz', verifier_dir/'private.tar.gz')
    verifier_text = coder_text+'''FROM coder AS verifier
COPY private.tar.gz /tmp/private.tar.gz
RUN mkdir /private-context && tar -xzf /tmp/private.tar.gz -C /private-context && rm /tmp/private.tar.gz && mkdir -p /workspace/src /suite && cp -a /private-context/runtime /workspace/src/polytopia_bot && cp -a /private-context/refs /refs && cp -a /private-context/tests /tests && cp /private-context/manifest.json /private-context/maps.json /suite/ && rm -rf /private-context
ENV PYTHONPATH=/workspace/src
WORKDIR /workspace
USER root
'''
    (verifier_dir/'Dockerfile').write_text(verifier_text)
    verifier_stamp = sha(verifier_text.encode()+(context/'public.tar.gz').read_bytes()+
                         (context/'private.tar.gz').read_bytes())[:12]
    verifier = snapshot_ready(daytona, CreateSnapshotParams(
        name=f'polytopia-skirmish-verifier-{verifier_stamp}',
        image=Image.from_dockerfile(verifier_dir/'Dockerfile',strict_context=True),
        resources=Resources(cpu=2,memory=4,disk=10),region_id='us',entrypoint=['sleep','infinity']))
    import toml
    task = package/'task'
    shutil.copytree(ROOT/'harbor/tasks/polytopia-skirmish', task)
    cfg = toml.loads((task/'task.toml').read_text())
    cfg['environment']['docker_image'] = 'daytona-snapshot:'+coder.name
    cfg['verifier']['environment']['docker_image'] = 'daytona-snapshot:'+verifier.name
    (task/'task.toml').write_text(toml.dumps(cfg))
    if source_files() != sources:
        raise ValueError('source changed during build; allocate a fresh package')
    write_json(package/'build.json',dict(task=str(task), backend='daytona',
        source_files=sources, coder_snapshot=coder.model_dump(mode='json'),
        verifier_snapshot=verifier.model_dump(mode='json'),
        manifest_sha256=sha(canonical(manifest).encode()),maps_sha256=sha(canonical(maps).encode())))
    with tarfile.open(package/'source.tar.gz','w:gz') as archive:
        for name in sources:
            archive.add(ROOT/name,arcname=name)
    print(json.dumps(dict(package=str(package), status='ready')),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--shared-package', type=Path, required=True)
    p.add_argument('--references', type=Path, default=ROOT/'artifacts/skirmish-references')
    a = p.parse_args()
    build(a.package,a.shared_package,a.references)
