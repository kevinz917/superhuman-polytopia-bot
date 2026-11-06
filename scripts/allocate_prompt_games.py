#!/usr/bin/env python3
"""Freeze a bounded gameplay allocation after the completed prompt screen."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.prompt_search import STRATEGIES
from polytopia_bot.learning.interface import digest


def allocate(output, *, stage, selection=None):
    qualification=ROOT/'runs/sft0/teacher-qualification-v3/report.json'
    screen=ROOT/'runs/sft0/prompt-screen-v2/report.json'
    q=json.loads(qualification.read_text())
    s=json.loads(screen.read_text())
    if q.get('teacher_qualified') is not True or s.get('proceed_to_gameplay') is not True:
        raise ValueError('qualification and bounded screen must pass')
    prerequisites=[qualification,screen,screen.parent/'protocol.json']
    with gzip.open(screen.parent/'results.json.gz','rt') as stream:result=json.load(stream)
    generations=[a for c in result['results'] for r in c['rows'] for a in r['attempts']]
    gpu_seconds=sum(a['batch_seconds']/a['batch_size'] for a in generations)
    measured_calls_per_second=len(generations)/gpu_seconds
    manifest=json.loads((ROOT/'runs/sft0/splits-v1/maps.json').read_text())
    if stage=='search':
        names=['original',*s['selected_for_gameplay']]
        if len(names)!=3 or len(set(names))!=3:raise ValueError('exactly two revised survivors required')
        maps=manifest['maps']['prompt_search'];opponent='weak';ceiling=14400
    else:
        if not selection:raise ValueError('confirmation requires complete search selection receipt')
        choice=json.loads((ROOT/selection).read_text())['selected']
        if choice=='original':raise ValueError('no revision selected; retain original without redundant confirmation')
        names=['original',choice];maps=manifest['maps']['prompt_confirmation'];opponent='pressure';ceiling=10800
        prerequisites.append(ROOT/selection)
    if len(maps)!=10:raise ValueError('exact registered ten-map stage required')
    jobs=[dict(id=f'{stage}-{m["seed"]}-{seat}-{name}',seed=m['seed'],board_hash=m['board_hash'],seat=seat,
               candidate=name,strategy=STRATEGIES[name],opponent=opponent)
          for m in maps for seat in (1,2) for name in names]
    # Interleave candidates within map/seat blocks so a budget stop is explicit,
    # never silently presented as a complete candidate comparison.
    files=[*sorted((ROOT/'src/polytopia_bot/learning').glob('*.py')),
           ROOT/'training/pyproject.toml',ROOT/'training/uv.lock',ROOT/'configs/sft.json',
           ROOT/'configs/ruleset-v1.json',ROOT/'experiments/modal_prompt_games.py']
    rule_sources=json.loads((ROOT/'configs/ruleset-v1.json').read_text())['source_sha256']
    files.extend(ROOT/name for name in rule_sources)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    for path in prerequisites:
        if path.parent.joinpath('invalidation.json').exists():raise ValueError('invalidated prerequisite')
    rate=.000694+4*.0000131+16*.00000222
    protocol=dict(schema='prompt-gameplay-allocation-v1',stage=stage,run_id=f'sft0-prompt-{stage}-v1',
        jobs=jobs,config=result['config'],map_manifest_hash=manifest['manifest_hash'],
        prerequisites=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in prerequisites],
        sources={str(p.relative_to(ROOT)):sha(p) for p in files},
        shards=4,seconds_limit=ceiling//4,occupied_gpu_hours_ceiling=ceiling/3600,
        planning_compute_allowance_usd=round((ceiling+4*300)*rate,2),
        measured_screen_calls_per_second=measured_calls_per_second,
        illustrative_seconds_at_600_calls_per_game=len(jobs)*600/measured_calls_per_second,
        cost_limitation='Screen throughput excludes sequential game coordination and longer on-policy states; hard time ceiling applies even if allocation is incomplete. Dollar value uses published function rates, not an invoice cap.',
        selection=dict(method='Map-paired bootstrap with 10000 resamples, seed 17029; original retained unless qualified revision improves clearly',
            native_min=.95,truncation_strict_max=.05,first_attempt_valid_min=.95,
            score_gain_min=.10,paired_lower95_strict_min=0,tie_break='candidate insertion order'),
        search_total_game_cap=100,apex=False)
    path=ROOT/output
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(protocol,stream,indent=2)
    print(json.dumps({k:v for k,v in protocol.items() if k not in ('jobs','sources','config','prerequisites')},indent=2))
    return protocol


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--stage',choices=['search','confirmation'],required=True)
    p.add_argument('--selection')
    a=p.parse_args();allocate(a.output,stage=a.stage,selection=a.selection)
