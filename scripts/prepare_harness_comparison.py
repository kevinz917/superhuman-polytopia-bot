"""Freeze paired prompts and disjoint scoring metadata before inference."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from polytopia_bot.learning.data import observation_from_dict
from polytopia_bot.learning.decision_harness import VERSION, render_decision
from polytopia_bot.learning.harness_comparison import score_prediction
from polytopia_bot.learning.interface import canonical, digest
from polytopia_bot.learning.teacher import ScriptedTeacher


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(run_id='decision-harness-v1'):
    root = ROOT/'runs'/run_id
    root.mkdir(exist_ok=False)
    source = ROOT/'runs/movement-scale40k-v1'
    old = json.loads((source/'protocol.json').read_text())
    assert sha(source/'positions.jsonl.gz') == old['positions_sha256']
    inputs = list(map(json.loads, gzip.open(source/'positions.jsonl.gz', 'rt')))
    teacher = ScriptedTeacher()
    rows = []
    for p in inputs:
        obs = observation_from_dict(p['observation'])
        scores = [dict(command=s['command'], score=s['score']) for s in teacher.select(obs).scores]
        row = {k: p[k] for k in ['position_id','board_hash','episode_id','phase','observation','teacher_command']}
        row['scores'] = scores
        row['harness_digests'] = {v: digest(render_decision(obs, v).messages) for v in ['original', VERSION]}
        if row['harness_digests']['original'] != p['prompt_digest']:
            raise ValueError('original prompt changed')
        row['movement_required'] = score_prediction(row, None)['movement_required']
        rows.append(row)
    # Fixed hash sample without inspecting model predictions; family-balanced controls.
    ordered = sorted(rows, key=lambda r: digest(['harness-screen-17', r['position_id']]))
    screen = [r['position_id'] for r in ordered if r['movement_required']][:180]
    for k in range(2, 11):
        chosen = [r['position_id'] for r in ordered if r['teacher_command']['kind'] == k][:20]
        if len(chosen) != 20:
            raise ValueError('insufficient family controls')
        screen.extend(chosen)
    assert len(screen) == len(set(screen)) == 360
    # Identical order for every model; independent of family and teacher scores.
    rows.sort(key=lambda r: digest(['harness-order-17', r['position_id']]))
    with gzip.open(root/'positions.jsonl.gz', 'wt') as f:
        for row in rows:
            f.write(canonical(row)+'\n')
    base = {k: old['config'][k] for k in ['model_id','revision']}
    main = old['training']['scale']
    models = dict(main={**base, 'adapter': main['adapter'], 'adapter_files': main['adapter_files'], 'cohort': 'full'})
    calls = json.loads((source/'evaluation-calls.json').read_text())
    for key, step in [('10k', '157'), ('20k', '314')]:
        cp = calls[step]['checkpoint']
        models[key] = {**base, 'adapter': cp['path'], 'adapter_files': cp['adapter_files'], 'cohort': 'full'}
    for key, model in [('qwen35-4b','Qwen/Qwen3.5-4B'), ('qwen35-9b','Qwen/Qwen3.5-9B')]:
        with urllib.request.urlopen('https://huggingface.co/api/models/'+model) as f:
            revision = json.load(f)['sha']
        models[key] = dict(model_id=model, revision=revision, cohort='screen', multimodal=True)
    p = dict(schema='single-call-harness-comparison-v1', run_id=run_id,
        versions=['original',VERSION], models=models, evaluation_count=len(rows), screen_ids=screen,
        positions_sha256=sha(root/'positions.jsonl.gz'), source_positions_sha256=old['positions_sha256'],
        teacher_hash=teacher.teacher_hash,
        inference=dict(max_length=32768, max_new_tokens=32, batch_size=32, seconds_limit=2100),
        scope='Inference only. No training, RL, checkpoint promotion, final-test access or game rollout.',
        limitations=['Reused development evaluation positions, not a fresh confirmation set.',
                     'Fresh Qwen3.5 weights versus domain-finetuned Qwen3 checkpoints compare practical starting points, not isolated capacity.',
                     'Teacher agreement is heuristic imitation, not game strength.',
                     'All arms are one call, greedy direct action; native Qwen3.5 thinking is explicitly disabled.',
                     'Feature augmentation changes both representation and input token count.'])
    (root/'protocol.json').write_text(json.dumps(p, indent=2)+'\n')
    hashes = {str(f.relative_to(ROOT)): sha(f) for folder in ['src','scripts','experiments'] for f in (ROOT/folder).rglob('*.py')}
    (root/'source-hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps(dict(run_id=run_id, positions=len(rows), movement_required=sum(r['movement_required'] for r in rows),
                          screen=len(screen), protocol_sha256=sha(root/'protocol.json'))))


if __name__ == '__main__':
    prepare(*sys.argv[1:])
