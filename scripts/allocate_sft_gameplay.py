#!/usr/bin/env python3
"""Freeze the planned 25-map, both-seat base/smoke gameplay comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from polytopia_bot.learning.freeze import verify_prompt, inference_config
from polytopia_bot.learning.interface import digest


def allocate(output, run_id, *, adapter_result=None, seconds_per_shard=3600):
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run ID must be a simple new name')
    if not 600 <= seconds_per_shard <= 28800:
        raise ValueError('invalid occupied-time ceiling')
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    config = json.loads((ROOT/'configs/sft.json').read_text())
    prompt_path = ROOT/'runs/sft0/frozen-prompt-v1.json'
    prompt = json.loads(prompt_path.read_text())
    verify_prompt(prompt, config)
    qualified_path = ROOT/'runs/sft0/qualified-teacher-v1.json'
    qualified = json.loads(qualified_path.read_text())
    if not qualified['qualification']['teacher_qualified']:
        raise ValueError('qualified teacher required')
    for name, expected in qualified['interface_sources'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('qualified public interface or teacher changed')
    manifest_path = ROOT/'runs/sft0/splits-v1/maps.json'
    manifest = json.loads(manifest_path.read_text())
    if digest({k:v for k,v in manifest.items() if k != 'manifest_hash'}) != manifest['manifest_hash']:
        raise ValueError('map manifest digest mismatch')
    maps = manifest['maps']['validation'][:25]
    if len(maps) != 25 or manifest['size'] != 14:
        raise ValueError('exact 25-map validation pilot required')
    adapter = None
    adapter_files = None
    kind = 'base'
    prerequisites = [prompt_path, qualified_path, manifest_path,
                     ROOT/qualified['qualification_path']]
    if sha(prerequisites[-1]) != qualified['qualification_sha256']:
        raise ValueError('qualification receipt changed')
    if adapter_result:
        folder = ROOT/adapter_result
        execution = json.loads((folder/'execution.json').read_text())
        receipt = execution.get('run') or {}
        allocation = json.loads((folder/'allocation.json').read_text())
        if execution.get('exit_code') != 0 or receipt.get('status') != 'completed' or receipt.get('stage') != 'smoke':
            raise ValueError('completed smoke adapter required')
        if inference_config(allocation['config']) != inference_config(config):
            raise ValueError('smoke inference configuration differs')
        adapter_root = folder/'training-run/model/adapter-final'
        adapter_files = {str(p.relative_to(adapter_root)): sha(p) for p in adapter_root.rglob('*') if p.is_file()}
        if 'adapter_model.safetensors' not in adapter_files or 'adapter_config.json' not in adapter_files:
            raise ValueError('downloaded adapter files required')
        if any(adapter_files.get(name)!=expected for name,expected in receipt['adapter_files'].items()):
            raise ValueError('downloaded adapter differs from training receipt')
        adapter = '/workspace/runs/'+allocation['run_id']+'/model/adapter-final'
        prerequisites += [folder/'execution.json', folder/'allocation.json']
        kind = 'smoke'
    jobs = [dict(id=f'{kind}-{m["seed"]}-{seat}', seed=m['seed'], board_hash=m['board_hash'],
                 seat=seat, candidate=prompt['candidate'], strategy=prompt['strategy'], opponent='weak')
            for m in maps for seat in (1,2)]
    files = [*sorted((ROOT/'src/polytopia_bot/learning').glob('*.py')),
             ROOT/'training/pyproject.toml', ROOT/'training/uv.lock', ROOT/'configs/ruleset-v1.json',
             ROOT/'experiments/modal_prompt_games.py', Path(__file__)]
    rules = json.loads((ROOT/'configs/ruleset-v1.json').read_text())
    for name, expected in rules['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('frozen simulator changed')
        files.append(ROOT/name)
    protocol = dict(schema='sft-gameplay-allocation-v1', stage='smoke-gate', kind=kind,
        run_id=run_id, jobs=jobs, config=config, adapter=adapter, adapter_files=adapter_files,
        prompt_manifest_hash=prompt['manifest_hash'], tokenizer_hash=prompt['tokenizer_hash'],
        chat_template_hash=prompt['chat_template_hash'], map_manifest_hash=manifest['manifest_hash'],
        weak_opponent_hash=qualified['weak_opponent_hash'],
        prerequisites=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in prerequisites],
        sources={str(p.relative_to(ROOT)):sha(p) for p in files},
        shards=4, seconds_limit=seconds_per_shard,
        occupied_gpu_hours_ceiling=4*seconds_per_shard/3600,
        planning_compute_allowance_usd=round(4*(seconds_per_shard+120)*.00078192,2),
        comparison='First 25 reserved validation maps, both seats; identical base/smoke prompt, decoding and weak opponent.',
        gates=dict(saved_position_first_valid_min=.95, saved_position_agreement_gain_min=.05,
                   smoke_native_completion_min=.80, all_allocated_games_required=True,
                   no_observation_leak_or_rule_failure=True),
        budget_note='Four A100 workers with hard time ceilings. Budget-incomplete games are not draws or native endings. Estimates exclude builds/storage and are not invoice caps.',
        final_test=False, apex=False)
    path = ROOT/output
    with path.open('x') as stream:
        json.dump(protocol, stream, indent=2)
    archive = path.with_suffix('')
    archive.mkdir(exist_ok=False)
    for name in protocol['sources']:
        target = archive/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    print(json.dumps({k:v for k,v in protocol.items() if k not in ('sources','jobs','config','prerequisites','adapter_files')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--adapter-result', type=Path)
    p.add_argument('--seconds-per-shard', type=int, default=3600)
    a = p.parse_args()
    allocate(a.output, a.run_id, adapter_result=a.adapter_result, seconds_per_shard=a.seconds_per_shard)
