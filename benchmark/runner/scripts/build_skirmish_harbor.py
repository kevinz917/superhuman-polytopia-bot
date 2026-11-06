"""Build the public Harbor task and private verifier, then pin a local package."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.build_skirmish_local import DOCKERFILE, SDK
from polytopia_bot.skirmish.common import canonical, sha, write_json
from polytopia_bot.skirmish.docker_sandbox import docker
from polytopia_bot.skirmish.local import checked_references, image_id
from polytopia_bot.skirmish.v6 import make_maps

CODER = 'polytopia-skirmish-coder:continuous-v1'
VERIFIER = 'polytopia-skirmish-verifier:continuous-v1'


def prepare_context(context, manifest, maps, references):
    shutil.copytree(ROOT/'engine-rs', context/'engine-rs', ignore=shutil.ignore_patterns('target'))
    shutil.copytree(ROOT/'skirmish/bot-template', context/'bot', ignore=shutil.ignore_patterns('target'))
    shutil.copytree(ROOT/'skirmish/mini', context/'mini')
    protocol = (ROOT/'skirmish/continuous/PROTOCOL.md').read_text()
    (context/'mini/PROTOCOL.md').write_text(protocol)
    rules = (context/'mini/RULES.md').read_text()
    start = rules.index('## skirmish/mini/PROTOCOL.md')
    end = rules.index('\n## docs/rules-alignment-', start)
    (context/'mini/RULES.md').write_text('Active protocol: /reference/PROTOCOL.md supersedes embedded historical defaults.\n\n'+rules[:start]+protocol+rules[end:])
    tools = '''# Authoring tools\n\nUse `skirmish --help`. `maps`, `inspect`, `replay`, `compile`, `check` and `example` retain the real-engine SDK behavior.\n\n`skirmish practice --map drylands-0 --seat 1` compiles current source and plays public-greedy-v1. Both seats and all nine named public maps are supported. The returned report names `/bot/practice/<id>/game.jsonl`. `skirmish trace --path PATH --offset 0 --limit 1` prints your recorded observation/action entries. You may also use Python to summarize those files. Compilation, practice play, replay checks and inspection consume coding time. Practice uses 200 rounds, 40,000 actions and four trillion fuel, with the same Wasm limits as final evaluation. No hidden-map/champion query exists.\n'''
    (context/'mini/TOOLS.md').write_text(tools)
    (context/'mini/QUICKSTART.md').write_text((ROOT/'harbor/tasks/polytopia-skirmish/instruction.md').read_text())
    shutil.copyfile(ROOT/'src/polytopia_bot/gpu/rules.json', context/'rules.json')
    extras = ['maps/stream.py','skirmish/protocol.py','skirmish/game.py','skirmish/local_referee.py',
              'skirmish/continuous.py','skirmish/practice.py']
    paths = set(SDK+extras+[str(p.relative_to(ROOT/'src/polytopia_bot')) for p in (ROOT/'src/polytopia_bot/gpu').glob('*.py')]+['gpu/rules.json'])
    for name in paths:
        destination = context/'sdk'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/'src/polytopia_bot'/name, destination)
    shutil.copytree(ROOT/'src/polytopia_bot', context/'runtime', ignore=shutil.ignore_patterns('__pycache__','*.pyc','static'))
    for ref in manifest['references']:
        if 'file' in ref:
            source = references/ref['file']
            destination = context/'refs'/source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
            if ref['kind'] in ('expert','hybrid'):
                shutil.copyfile(source.with_suffix('.json'), destination.with_suffix('.json'))
    shutil.copytree(ROOT/'harbor/tasks/polytopia-skirmish/tests', context/'tests')
    write_json(context/'manifest.json', manifest)
    write_json(context/'maps.json', maps)
    base = DOCKERFILE.replace('FROM base AS coder\n','FROM base AS coder\nLABEL polytopia.skirmish.role=coder\n')
    base = base.replace('WORKDIR /bot\n', 'RUN mkdir -p /bot/practice && chown coder:coder /bot/practice\nENV POLYTOPIA_SKIRMISH_CONTINUOUS=1 POLYTOPIA_EXECUTION_BACKEND=local-docker POLYTOPIA_SKIRMISH_CHECK_FUEL=4000000000000\nWORKDIR /bot\n')
    base += '''\nFROM coder AS verifier\nCOPY runtime /workspace/src/polytopia_bot\nCOPY refs /refs\nCOPY tests /tests\nCOPY manifest.json /suite/manifest.json\nCOPY maps.json /suite/maps.json\nENV PYTHONPATH=/workspace/src\nWORKDIR /workspace\n'''
    (context/'Dockerfile').write_text(base)


def build(package, references, shared_suite=None):
    from polytopia_bot.skirmish.continuous import source_files
    sources = source_files()
    package = package.resolve()
    package.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/'configs/skirmish/continuous-v1.json').read_text())
    checked_references(manifest, references)
    if shared_suite:
        suite_manifest = json.loads((shared_suite/'manifest.json').read_text())
        if suite_manifest != manifest:
            raise ValueError('shared suite belongs to a different benchmark manifest')
        maps = json.loads((shared_suite/'maps.json').read_text())
    else:
        maps = make_maps(manifest['maps_per_family'])
    write_json(package/'maps.json', maps)
    write_json(package/'manifest.json', manifest)
    with tempfile.TemporaryDirectory(prefix='skirmish-harbor-build-') as temp:
        context = Path(temp)
        prepare_context(context, manifest, maps, references)
        for stage, tag in [('coder',CODER),('verifier',VERIFIER)]:
            subprocess.run(docker('build','--target',stage,'-t',tag,context), check=True)
            digest = image_id(tag)
            retained_tag = tag.split(':')[0]+':sha-'+digest.removeprefix('sha256:')
            subprocess.run(docker('tag',digest,retained_tag),check=True)
    import toml
    task = package/'task'
    shutil.copytree(ROOT/'harbor/tasks/polytopia-skirmish', task)
    cfg = toml.loads((task/'task.toml').read_text())
    cfg['environment']['docker_image'] = image_id(CODER)
    cfg['verifier']['environment']['docker_image'] = image_id(VERIFIER)
    (task/'task.toml').write_text(toml.dumps(cfg))
    if source_files() != sources:
        raise ValueError('source changed during build; allocate a fresh package')
    write_json(package/'build.json', dict(task=str(task), coder_image=cfg['environment']['docker_image'],
        verifier_image=cfg['verifier']['environment']['docker_image'], source_files=sources,
        manifest_sha256=sha(canonical(manifest).encode()), maps_sha256=sha(canonical(maps).encode())))
    import tarfile
    with tarfile.open(package/'source.tar.gz','w:gz') as archive:
        for name in sources:
            archive.add(ROOT/name, arcname=name)
    print(json.dumps(dict(package=str(package), task=str(task))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, default=ROOT/'artifacts/harbor-packages'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    p.add_argument('--references', type=Path, default=ROOT/'artifacts/skirmish-references')
    p.add_argument('--shared-suite', type=Path, help='Explicitly reuse a prepared hidden suite across infrastructure builds.')
    args = p.parse_args()
    build(args.package, args.references,args.shared_suite)
