#!/usr/bin/env python3
"""Allocate the single planned distribution-repair pass on training maps only."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.interface import digest


def allocate(output,run_id,diagnostic_path,seconds_limit=3600):
    if Path(run_id).name!=run_id or run_id in ('.','..') or not 600<=seconds_limit<=7200:
        raise ValueError('fresh run ID and bounded time allocation required')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    design_path=ROOT/'runs/sft0/repair-design-v1.json'
    design=json.loads(design_path.read_text())
    if design['repair_pass']!=1 or design['collection']['seconds_per_shard']!=seconds_limit:
        raise ValueError('allocation differs from the recorded single-pass design')
    diagnostic=json.loads((ROOT/diagnostic_path).read_text())
    if not diagnostic.get('live_game_replay_passed') or not diagnostic['positions']['format_passed'] or not diagnostic['positions']['learning_passed']:
        raise ValueError('complete audited smoke diagnostic with actual action learning required')
    if not diagnostic['games']['execution_passed']:
        raise ValueError('resolve infrastructure/integrity failures before distribution repair')
    parent=ROOT/'runs/sft0/smoke-games-allocation-v1.json'
    protocol=copy.deepcopy(json.loads(parent.read_text()))
    if protocol['kind']!='smoke' or protocol['final_test'] or not protocol.get('adapter'):
        raise ValueError('registered smoke adapter allocation required')
    for item in diagnostic['evidence']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('diagnostic evidence changed')
    for name,expected in protocol['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('frozen policy or simulation source changed')
    for item in protocol['prerequisites']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('smoke prerequisite changed')
    manifest=json.loads((ROOT/'runs/sft0/splits-v1/maps.json').read_text())
    if digest({k:v for k,v in manifest.items() if k!='manifest_hash'})!=protocol['map_manifest_hash']:
        raise ValueError('split manifest changed')
    maps=manifest['maps']['train'][:25]
    if len(maps)!=25 or manifest['size']!=14:raise ValueError('25 training maps required')
    held_out={m['board_hash'] for split in ('validation','test') for m in manifest['maps'][split]}
    if any(m['board_hash'] in held_out for m in maps):raise ValueError('held-out map cannot enter relabel collection')
    template=protocol['jobs'][0]
    protocol.update(stage='training-distribution-repair',kind='smoke-relabel',run_id=run_id,
        jobs=[dict(id=f'relabel-{m["seed"]}-{seat}',seed=m['seed'],board_hash=m['board_hash'],seat=seat,
                   candidate=template['candidate'],strategy=template['strategy'],opponent='weak') for m in maps for seat in (1,2)],
        seconds_limit=seconds_limit,occupied_gpu_hours_ceiling=4*(seconds_limit+120)/3600,
        planning_compute_allowance_usd=round(4*(seconds_limit+120)*.00078192,2),
        comparison='Training-only data collection from the saved smoke policy; not validation or final-test evidence.',
        repair_pass=1,max_relabel_states=10000,max_main_relabel_fraction=.20,
        provenance='Frozen teacher labels public observations reached after the model deviates from the teacher. Preserve actual model transitions and source tags in sidecars. No outcome filtering or reward gradient.',
        training_only=True,final_test=False,apex=False)
    protocol.pop('gates',None)
    for p in (parent,ROOT/diagnostic_path,design_path):
        protocol['prerequisites'].append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    protocol['sources'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    target=ROOT/output
    with target.open('x') as stream:json.dump(protocol,stream,indent=2)
    archive=target.with_suffix('');archive.mkdir(exist_ok=False)
    for name in protocol['sources']:
        path=archive/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    print(json.dumps({k:v for k,v in protocol.items() if k not in ('jobs','config','sources','prerequisites','adapter_files')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--seconds-limit',type=int,default=3600)
    a=p.parse_args();allocate(a.output,a.run_id,a.diagnostic,a.seconds_limit)
