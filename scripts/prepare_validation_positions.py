#!/usr/bin/env python3
"""Freeze 5,000 validation positions without consulting a trained model."""
import argparse,gzip,hashlib,json,sys
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.interface import digest
from polytopia_bot.learning.corpus import phase


def prepare(output):
    source=ROOT/'runs/sft0/teacher-qualification-v3'
    qualification=json.loads((source/'report.json').read_text())
    if not qualification['teacher_qualified'] or (source/'invalidation.json').exists():raise ValueError('valid qualification traces required')
    teacher=json.loads((ROOT/'runs/sft0/qualified-teacher-v1.json').read_text())['teacher_hash']
    manifest=json.loads((ROOT/'runs/sft0/splits-v1/maps.json').read_text())
    maps={r['seed']:r['board_hash'] for r in manifest['maps']['validation']}
    games={(g['seed'],g['seat']):g for g in json.loads((source/'games.json').read_text())}
    groups=defaultdict(list);seen=set();files={}
    root=ROOT/output;root.mkdir(parents=True,exist_ok=False)
    for path in sorted(source.glob('worker-*/*.jsonl.gz')):
        seed=int(path.name.split('-')[1]);seat=int(path.name.split('-')[3].split('.')[0])
        if games[(seed,seat)]['board_hash']!=maps[seed]:raise ValueError('source outside reserved validation maps')
        files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for index,line in enumerate(gzip.open(path,'rt')):
            row=json.loads(line)
            if row['selection']['teacher_hash']!=teacher:continue
            key=row['input_digest']
            if key in seen:continue
            seen.add(key);command=row['selection']['action']['command']
            stage=phase(dict(observation=row['observation'],selected_command=command))
            groups[(stage,command['kind'])].append(dict(file=str(path.relative_to(ROOT)),line=index,
                position_id=f'{seed}-{seat}-{index}',input_digest=key,phase=stage,family=command['kind'],
                board_hash=maps[seed],episode_id=f'{seed}-{seat}'))
    for rows in groups.values():rows.sort(key=lambda r:digest(['validation-sample-v1',r['input_digest']]),reverse=True)
    selected=[];ends=0
    while len(selected)<5000:
        changed=False
        for group in sorted(groups):
            if not groups[group] or (group[1]==10 and ends>=1000):continue
            selected.append(groups[group].pop());ends+=group[1]==10;changed=True
            if len(selected)==5000:break
        if not changed:raise ValueError('insufficient validation positions under fixed cap')
    targets={(r['file'],r['line']):r for r in selected}
    with gzip.open(root/'positions.jsonl.gz','xt',compresslevel=1) as out:
        for path in files:
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=files[path]:raise ValueError('source changed during sampling')
            for index,line in enumerate(gzip.open(ROOT/path,'rt')):
                meta=targets.get((path,index))
                if meta is not None:
                    row=json.loads(line)
                    out.write(json.dumps(dict(**meta,observation=row['observation'],teacher_command=row['selection']['action']['command']),separators=(',',':'))+'\n')
    report=dict(schema='validation-positions-v1',count=5000,teacher_hash=teacher,
        split_manifest_hash=manifest['manifest_hash'],qualification_sha256=hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),
        source_files=files,positions_sha256=hashlib.sha256((root/'positions.jsonl.gz').read_bytes()).hexdigest(),
        maps=len({r['board_hash'] for r in selected}),episodes=len({r['episode_id'] for r in selected}),
        phases=dict(Counter(r['phase'] for r in selected)),families=dict(Counter(str(r['family']) for r in selected)),
        selection='Phase/action-family round robin over independent SHA256 order; end-turn capped at 20%; no model results used',
        role='Saved-position agreement/validity only; never SFT inputs or final-test evidence')
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.output)
