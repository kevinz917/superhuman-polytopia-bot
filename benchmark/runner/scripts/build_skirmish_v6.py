"""Build v6 images alongside existing v1 images; never overwrite archived protocols."""
import json,shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.build_skirmish_local import SDK,DOCKERFILE
from polytopia_bot.skirmish.docker_sandbox import docker

def build():
    with tempfile.TemporaryDirectory(prefix='skirmish-v6-build-') as directory:
        c=Path(directory)
        shutil.copytree(ROOT/'engine-rs',c/'engine-rs',ignore=shutil.ignore_patterns('target','__pycache__'))
        shutil.copytree(ROOT/'skirmish/bot-template',c/'bot',ignore=shutil.ignore_patterns('target'))
        shutil.copytree(ROOT/'skirmish/mini',c/'mini')
        protocol=(ROOT/'skirmish/v6/PROTOCOL.md').read_text()
        (c/'mini/PROTOCOL.md').write_text(protocol)
        rules=(c/'mini/RULES.md').read_text()
        # Do not feed contradictory historical limits to new authors.
        start=rules.index('## skirmish/mini/PROTOCOL.md')
        end=rules.index('\n## docs/rules-alignment-',start)
        rules=rules[:start]+'## Local v6 protocol\n\n````markdown\n'+protocol+'\n````\n'+rules[end:]
        (c/'mini/RULES.md').write_text('V6 OVERRIDE: /reference/PROTOCOL.md and the active task receipt define timing/runtime. The engine code below retains historical defaults; the trusted v6 referee supplies exact-round overrides.\n\n'+rules)
        (c/'mini/QUICKSTART.md').write_text((ROOT/'skirmish/mini/QUICKSTART.md').read_text().replace('six expert games','six mixed-reference games').replace('No opponent or opponent code is supplied.','No opponent implementation or code is supplied.'))
        (c/'mini/TOOLS.md').write_text((c/'mini/TOOLS.md').read_text().replace('one trillion fuel','four trillion fuel')+'\nV6: the task receipt and PROTOCOL.md supersede historical track names and timing.\n')
        shutil.copyfile(ROOT/'src/polytopia_bot/gpu/rules.json',c/'rules.json')
        files=SDK+[str(p.relative_to(ROOT/'src/polytopia_bot')) for p in (ROOT/'src/polytopia_bot/gpu').glob('*.py')]+['gpu/rules.json']
        for name in files:
            dst=c/'sdk'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'src/polytopia_bot'/name,dst)
        shutil.copytree(ROOT/'src/polytopia_bot',c/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc','static'))
        (c/'Dockerfile').write_text(DOCKERFILE.replace('FROM base AS coder\n',
            'FROM base AS coder\nENV POLYTOPIA_SKIRMISH_CHECK_FUEL=4000000000000\n'))
        for stage in ['coder','referee']:
            subprocess.run(docker('build','--target',stage,'-t',f'polytopia-skirmish-{stage}:local-v6',c),check=True)
    print(json.dumps(dict(coder='polytopia-skirmish-coder:local-v6',referee='polytopia-skirmish-referee:local-v6')))
if __name__=='__main__':build()
