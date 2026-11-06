"""Compact full-state prompt (polytopia-compact-v2): measure, regenerate expert-SFT data, train, evaluate. Modal only.

  modal run --env staging -m experiments.modal_compact_prompt --action scan        # glossary/shape census
  modal run --env staging -m experiments.modal_compact_prompt --action test        # pytest incl. real decisions
  modal run --env staging -m experiments.modal_compact_prompt --action measure     # tokens + shared prefix, all 1,000 games
  modal run --env staging -m experiments.modal_compact_prompt --action smoke       # data round trip + 2-step train + LoRA preflight
  modal run --detach --env staging -m experiments.modal_compact_prompt --action launch --run-id compact-prompt-v2
  modal run --env staging -m experiments.modal_compact_prompt --action compare --run-id compact-prompt-v2

Ablation = expert-sft-v1 with only the prompt changed: the same 39,626 (game, decision, label)
examples from expert-sft-v1/prepared/index.jsonl, the identical training config/seed/runtime on
4 x B200, and the same 30 development jobs, sampling and seeds on one H100 per arm.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import modal
from experiments.modal_thinking_repaired import image, volume, ROOT

app = modal.App('polytopia-compact-prompt')
RUNS = Path('/workspace/runs')
SOURCE = 'expert-sft-v1'
ENV = {'PYTHONPATH': '/workspace/src', 'HF_HOME': '/workspace/runs/hf-cache', 'TOKENIZERS_PARALLELISM': 'false',
       'JAX_PLATFORMS': 'cpu'}
test_image = (modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
    .run_commands('uv pip install --system vllm==0.19.0 transformers==4.57.6 peft==0.17.1 accelerate==1.10.1 jax==0.11.2')
    .pip_install('pytest==9.1.1')
    .env(ENV)
    .add_local_dir(ROOT/'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'scripts', '/workspace/scripts', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'configs', '/workspace/configs')
    .add_local_dir(ROOT/'tests', '/workspace/tests', ignore=['**/__pycache__/**']))
# Same training image as experiments/modal_expert_sft.py (copied, not imported, to keep that app separate).
wheel = 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
training_image = (modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
    .add_local_file(ROOT/'training/pyproject.toml', '/workspace/training/pyproject.toml', copy=True)
    .add_local_file(ROOT/'training/uv.lock', '/workspace/training/uv.lock', copy=True)
    .run_commands('UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple uv sync --project /workspace/training --locked --no-dev --extra simulation')
    .run_commands(f'uv pip install --python /workspace/training/.venv/bin/python --no-deps {wheel}',
                  'uv pip install --python /workspace/training/.venv/bin/python einops==0.8.1')
    .env({'PYTHONPATH': '/workspace/src', 'TOKENIZERS_PARALLELISM': 'false', 'HF_HOME': '/workspace/runs/hf-cache', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT/'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'scripts', '/workspace/scripts', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'configs', '/workspace/configs'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_protocol(root, protocol_sha):
    if sha(root/'protocol.json') != protocol_sha:
        raise ValueError('protocol changed')
    return json.loads((root/'protocol.json').read_text())


def scripts_path():
    import sys
    if '/workspace/scripts' not in sys.path:
        sys.path.insert(0, '/workspace/scripts')


@app.function(image=image, cpu=1, memory=4096, timeout=1800, max_containers=50, volumes={'/workspace/runs': volume})
def scan_files(files):
    """Census of what the readable expert prompts actually contain (glossary IDs, record shapes)."""
    from polytopia_bot.learning.expert_data import read_game
    ids = {k: set() for k in ('units', 'technologies', 'structures')}
    stats = Counter()
    for f in files:
        header, examples, _ = read_game(RUNS/SOURCE/'expert-games'/f)
        for ex in examples:
            u = json.loads(ex['user'])
            for k in ids:
                ids[k] |= {r['id'] for r in u['rules'][k]['records']}
            v = u['visible']
            stats['examples'] += 1
            stats['absent_nonempty'] += bool(v['tiles']['absent'])
            stats['memory_tiles'] += bool(u['remembered']['tiles'])
            stats['historical'] += bool(u['remembered']['historical'])
            stats['relations'] += 'relations' in v
            stats['achievements'] += 'achievements' in v
            stats['sparse:'+','.join(sorted(v['tiles']['sparse']))] += 1
            for r in v['units']['records']:
                stats['unit_health_'+type(r['health']).__name__] += 1
                stats['unit_health_fractional'] += r['health'] != int(r['health'])
                stats['unit_keys_'+str(sum(x is not None for x in r.values()))] += 1
                stats['unit_passenger_nonzero'] += bool(r.get('passenger'))
            for r in u['legal_actions']['records']:
                if r['kind'] == 3:
                    stats[f"ability_{r['parameter']}_{r['source'] == -1}_{r['target'] == -1}"] += 1
                elif r['kind'] in (1, 2) and r['parameter']:
                    stats['move_attack_param_nonzero'] += 1
    return {k: sorted(v) for k, v in ids.items()}, dict(stats)


@app.function(image=test_image, cpu=4, memory=16384, timeout=3600, volumes={'/workspace/runs': volume})
def run_tests(game_limit=40):
    import subprocess
    env = dict(os.environ, POLYTOPIA_EXPERT_GAMES=str(RUNS/SOURCE/'expert-games'), POLYTOPIA_EXPERT_GAME_LIMIT=str(game_limit))
    out = subprocess.run(['python', '-m', 'pytest', '-q', '-rs', 'tests/test_compact_prompt.py'],
                         cwd='/workspace', env=env, capture_output=True, text=True)
    return dict(returncode=out.returncode, output=out.stdout[-6000:] + out.stderr[-3000:])


SECTIONS = [('header', None), ('map', 'terrain:'), ('features', None), ('cities', 'cities:'), ('technology', 'technology:'),
            ('enemy_units', 'enemy units:'), ('your_units', 'your units:'), ('recent', 'recent actions:'), ('status', 'turn '),
            ('legal', 'legal actions:')]


def split_sections(user):
    lines = user.split('\n')
    out, current = Counter(), 'header'
    for i, line in enumerate(lines):
        if i == 0:
            current = 'header'
        elif line == 'terrain:':
            current = 'map'
        elif current == 'map' and not line.startswith('y') and line != 'territory:':
            current = 'features'
        for name, head in SECTIONS[3:]:
            if head and line.startswith(head):
                current = name
        out[current] += 1
        yield current, line


@app.function(image=image, cpu=2, memory=8192, timeout=3600, max_containers=100, volumes={'/workspace/runs': volume})
def measure_files(files, config):
    """Exact equivalence on every decision + token counts and consecutive-decision shared prefixes."""
    import os.path
    scripts_path()
    from prepare_compact_sft import convert
    from polytopia_bot.learning.expert_data import read_game
    from polytopia_bot.learning.training import load_tokenizer
    tok = load_tokenizer(config)
    res = dict(old=[], new=[], old_lcp=[], new_lcp=[], same_turn=[], old_chars=0, new_chars=0, failures=[],
               sections=Counter(), section_examples=0, new_system=None, old_system=None)
    for f in files:
        header, examples, _ = read_game(RUNS/SOURCE/'expert-games'/f)
        prev = None
        for i, ex in enumerate(examples):
            try:
                messages, _ = convert(header, ex)
            except Exception as exc:  # recorded, never silently dropped
                res['failures'].append(dict(file=f, decision=ex['decision'], error=repr(exc)[:300]))
                prev = None
                continue
            old_ids = tok.apply_chat_template([dict(role='system', content=header['system']), dict(role='user', content=ex['user'])],
                                              tokenize=True, add_generation_prompt=True)
            new_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            res['old'].append(len(old_ids)); res['new'].append(len(new_ids))
            res['old_chars'] += len(ex['user']); res['new_chars'] += len(messages[1]['content'])
            if prev is not None:
                res['old_lcp'].append(len(os.path.commonprefix([prev[0], old_ids])))
                res['new_lcp'].append(len(os.path.commonprefix([prev[1], new_ids])))
                res['same_turn'].append(prev[2] == ex['turn'])
            else:
                res['old_lcp'].append(None); res['new_lcp'].append(None); res['same_turn'].append(None)
            prev = (old_ids, new_ids, ex['turn'])
            if res['new_system'] is None:
                res['new_system'] = len(tok.apply_chat_template(messages[:1], tokenize=True))
                res['old_system'] = len(tok.apply_chat_template([dict(role='system', content=header['system'])], tokenize=True))
            if i % 10 == 0:
                parts = {}
                for name, line in split_sections(messages[1]['content']):
                    parts.setdefault(name, []).append(line)
                for name, lines in parts.items():
                    res['sections'][name] += len(tok('\n'.join(lines) + '\n', add_special_tokens=False)['input_ids'])
                res['section_examples'] += 1
    return res


@app.function(image=image, cpu=4, memory=16384, timeout=3600, max_containers=40, retries=1, volumes={'/workspace/runs': volume})
def prepare_shard(run_id, protocol_sha, index, files, selected):
    scripts_path()
    from prepare_compact_sft import shard
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    volume.reload()
    (root/'prepared-shards').mkdir(exist_ok=True)
    out = root/'prepared-shards'/f'shard-{index:03d}.jsonl'
    result = shard([RUNS/SOURCE/'expert-games'/f for f in files], selected, p['training']['config'], out)
    volume.commit()
    return result


@app.function(image=image, cpu=4, memory=32768, timeout=7200, retries=0, volumes={'/workspace/runs': volume})
def combine_shards(run_id, protocol_sha, shards):
    scripts_path()
    from prepare_expert_sft import combine
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    volume.reload()
    manifest = combine(shards, root/'prepared', p['training']['config'], expert=p['expert'], data_protocol=p['data'])
    # The ablation must use exactly the v1 examples, in the v1 order, with the v1 labels.
    v1 = [json.loads(l) for l in open(RUNS/SOURCE/'prepared'/'index.jsonl')]
    mine = [json.loads(l) for l in open(root/'prepared'/'index.jsonl')]
    if [(r['id'], r['label']) for r in v1] != [(r['id'], r['label']) for r in mine] and not p['data'].get('smoke'):
        raise ValueError('regenerated corpus differs from expert-sft-v1 examples')
    volume.commit()
    return manifest


@app.function(image=training_image, gpu='B200:4', cpu=16, memory=131072, timeout=6*3600, max_containers=1,
              retries=0, scaledown_window=2, volumes={'/workspace/runs': volume})
def train(run_id, protocol_sha):
    import subprocess
    root = RUNS/run_id
    volume.reload()
    p = check_protocol(root, protocol_sha)
    payload = dict(prepared=str(root/'prepared'), output=str(root/'training'/'model'), config=p['training']['config'],
                   seed=p['training']['seed'], benchmark=p['training']['benchmark'], checkpoint_every=p['training']['checkpoint_every'])
    (root/'training').mkdir(exist_ok=False)
    (root/'training'/'input.json').write_text(json.dumps(payload, indent=2)); volume.commit()
    started = time.monotonic()
    proc = subprocess.Popen(['/workspace/training/.venv/bin/python', '-m', 'torch.distributed.run', '--standalone',
                             '--nproc_per_node=4', '/workspace/scripts/expert_train_distributed.py',
                             str(root/'training'/'input.json')], cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit(); time.sleep(15)
        result = dict(exit_code=proc.returncode, occupied_seconds=time.monotonic()-started, finished_unix=time.time())
        if (root/'training'/'model'/'run.json').exists():
            run = json.loads((root/'training'/'model'/'run.json').read_text())
            result.update(status=run['status'], train_seconds=run.get('train_seconds'), adapter_files=run.get('adapter_files'))
        (root/'training'/'execution.json').write_text(json.dumps(result, indent=2)); volume.commit()
        if proc.returncode:
            raise RuntimeError(result)
        return result
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=30)
        volume.commit()


@app.function(image=image, gpu='H100', cpu=8, memory=49152, timeout=6*3600, max_containers=3, retries=0,
              scaledown_window=2, volumes={'/workspace/runs': volume})
def evaluate(run_id, protocol_sha, arm, preflight_only=False):
    import subprocess
    root = RUNS/run_id
    volume.reload()
    check_protocol(root, protocol_sha)
    started = time.monotonic()
    cmd = ['python', '/workspace/scripts/benchmark_compact_sft.py', str(root), arm] + (['--preflight-only'] if preflight_only else [])
    proc = subprocess.Popen(cmd, cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit(); time.sleep(10)
        result = dict(arm=arm, exit_code=proc.returncode, worker_seconds=time.monotonic()-started, finished_unix=time.time(),
                      preflight_only=preflight_only)
        (root/f'{arm}-execution{"-preflight" if preflight_only else ""}.json').write_text(json.dumps(result, indent=2)); volume.commit()
        if proc.returncode:
            raise RuntimeError(result)
        return result
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=15)
        volume.commit()


def timing(path):
    rows = [json.loads(l) for l in open(path)] if Path(path).exists() else []
    if not rows:
        return None
    n = len(rows)
    return dict(calls=n, mean_input_tokens=sum(r['input_tokens'] for r in rows)/n, total_input_tokens=sum(r['input_tokens'] for r in rows),
                max_input_tokens=max(r['input_tokens'] for r in rows), mean_output_tokens=sum(r['output_tokens'] for r in rows)/n,
                mean_request_seconds=sum(r['request_seconds'] for r in rows)/n,
                finish_reasons=dict(Counter(r['finish_reason'] for r in rows)))


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={'/workspace/runs': volume})
def compare(run_id, protocol_sha):
    """Compact arms vs the expert-sft-v1 adapter on the same 30 jobs."""
    import random
    import statistics
    scripts_path()
    from summarize_expert_sft import aggregate, llm_arm, paired, win
    root = RUNS/run_id
    volume.reload()
    p = check_protocol(root, protocol_sha)
    jobs, v1 = p['jobs'], RUNS/SOURCE
    games = {'expert_sft_v1': llm_arm(v1, 'expert_sft', jobs), 'base_v1': llm_arm(v1, 'base', jobs)}
    arms = [a for a in p['arms'] if (root/a/'games'/'games.json').exists()]
    for a in arms:
        games[a] = llm_arm(root, a, jobs)
    result = dict(schema='compact-prompt-ablation-v1', arms={a: aggregate(g) for a, g in games.items()},
                  paired={f'{a}_vs_expert_sft_v1': paired(games[a], games['expert_sft_v1'], jobs) for a in arms})
    result['paired'].update({f'{a}_vs_base_v1': paired(games[a], games['base_v1'], jobs) for a in arms})
    if len(arms) > 1:
        # Pool the compact arms per map (4 games) against v1 (2 games), map-cluster bootstrap.
        maps = sorted({j['seed'] for j in jobs})
        diffs = [statistics.mean(win(games[a][j['id']]) for a in arms for j in jobs if j['seed'] == m)
                 - statistics.mean(win(games['expert_sft_v1'][j['id']]) for j in jobs if j['seed'] == m) for m in maps]
        rng = random.Random(1701)
        boot = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(20000))
        result['paired']['compact_pooled_vs_expert_sft_v1'] = dict(a_minus_b=statistics.mean(diffs),
            map_cluster_bootstrap_95=[boot[499], boot[19499]], maps=len(maps), maps_a_better=sum(d > 0 for d in diffs),
            maps_b_better=sum(d < 0 for d in diffs))
    ev = {'expert_sft_v1': dict(timing=timing(v1/'expert_sft'/'generation-timing.jsonl'),
                                execution=json.loads((v1/'expert_sft-execution.json').read_text()),
                                report_elapsed=json.loads((v1/'expert_sft'/'report.json').read_text())['elapsed_seconds'])}
    for a in arms:
        ev[a] = dict(timing=timing(root/a/'generation-timing.jsonl'),
                     execution=json.loads((root/f'{a}-execution.json').read_text()) if (root/f'{a}-execution.json').exists() else None,
                     report_elapsed=json.loads((root/a/'report.json').read_text())['elapsed_seconds'] if (root/a/'report.json').exists() else None,
                     prefix_caching=p['arms'][a].get('prefix_caching', False))
    result['evaluation'] = ev
    tr = {}
    for name, base in (('expert_sft_v1', v1), ('compact', root)):
        run = json.loads((base/'training'/'model'/'run.json').read_text())
        man = json.loads((base/'prepared'/'manifest.json').read_text())
        tr[name] = dict(status=run['status'], train_seconds=run.get('train_seconds'), elapsed_seconds=run.get('elapsed_seconds'),
                        steps=run.get('global_step'), stress=run.get('corpus_stress'),
                        loss_first=run['logs'][0]['loss'] if run['logs'] else None,
                        loss_last_20=statistics.mean(l['loss'] for l in run['logs'][-20:]) if run['logs'] else None,
                        records=man['records'], mean_tokens=man['mean_tokens'], max_tokens=man['max_tokens'], total_tokens=man['total_tokens'])
    result['training'] = tr
    result['games'] = games
    (root/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    small = {k: v for k, v in result.items() if k != 'games'}
    (root/'comparison-small.json').write_text(json.dumps(small, indent=2)+'\n')
    volume.commit()
    return small


retained = modal.Volume.from_name('polytopia-structured-rl-retained', environment_name='staging')
IMITATION = 'compact-prompt-v2-imitation'
_cache = {}


@app.function(image=image, cpu=2, memory=8192, timeout=3600, max_containers=40, retries=1,
              volumes={'/workspace/runs': volume, '/retained': retained.read_only()})
def heldout_game(job, expert):
    """One fresh expert game on a validation map (never used for training), stored like the v1 corpus."""
    from polytopia_bot.learning.expert_data import play_expert_game, read_game
    out = RUNS/IMITATION/'games'
    if (out/(job['id']+'.jsonl.gz')).exists():
        return read_game(out/(job['id']+'.jsonl.gz'))[2]
    if 'expert' not in _cache:
        if sha('/retained/'+expert['checkpoint']) != expert['sha256']:
            raise ValueError('expert checkpoint changed')
        from polytopia_bot.learning.adapter import GPUEnvironment
        from polytopia_bot.rl.policy import StructuredPolicy
        _cache['expert'] = StructuredPolicy('/retained/'+expert['checkpoint'], deterministic=True)
        _cache['env'] = GPUEnvironment(14)
    env = _cache['env']
    class Keep:
        def __getattr__(self, name):
            return getattr(env, name)
        def close(self):
            pass
    result = play_expert_game(job, '/retained/'+expert['checkpoint'], out, environment_factory=lambda: Keep(),
                              policy_factory=lambda _: _cache['expert'])
    volume.commit()
    return result


@app.function(image=image, gpu='H100', cpu=8, memory=65536, timeout=4*3600, retries=0, volumes={'/workspace/runs': volume})
def imitation(per_game, arms):
    """Greedy legal-grammar choice vs the expert label on held-out decisions; each adapter in its own prompt format."""
    import random
    import statistics
    scripts_path()
    from prepare_compact_sft import convert
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams
    from polytopia_bot.learning.compact_prompt import action_schema
    from polytopia_bot.learning.expert_data import read_game
    volume.reload()
    v1 = json.loads((RUNS/SOURCE/'protocol.json').read_text())
    root = RUNS/IMITATION
    rows = []
    for path in sorted((root/'games').glob('*.jsonl.gz')):
        header, examples, _ = read_game(path)
        rng = random.Random(f"17:{header['job']['id']}")
        for ex in sorted(rng.sample(examples, min(per_game, len(examples))), key=lambda e: e['decision']):
            compact, _ = convert(header, ex)
            readable = [dict(role='system', content=header['system']), dict(role='user', content=ex['user'])]
            rows.append(dict(game=header['job']['id'], decision=ex['decision'], family=ex['family'], label=ex['label'],
                             legal=ex['legal'], messages=dict(readable=readable, compact=compact)))
    llm = LLM(model=v1['model_id'], revision=v1['revision'], dtype='bfloat16', max_model_len=40960, gpu_memory_utilization=.85,
              enable_lora=True, max_lora_rank=32, max_loras=1, enable_prefix_caching=True, seed=17,
              structured_outputs_config=dict(backend='xgrammar'))
    tok = llm.get_tokenizer()
    result = dict(schema='compact-imitation-v1', decisions=len(rows), games=len({r['game'] for r in rows}), arms={})
    for i, (name, spec) in enumerate(arms.items()):
        prompts = [dict(prompt_token_ids=tok.apply_chat_template(r['messages'][spec['prompt']], tokenize=True, add_generation_prompt=True))
                   for r in rows]
        params = [SamplingParams(temperature=0, max_tokens=24, structured_outputs=StructuredOutputsParams(json=action_schema(r['messages'][spec['prompt']])))
                  for r in rows]
        started = time.monotonic()
        outs = llm.generate(prompts, params, lora_request=LoRARequest(name, i + 1, str(RUNS/spec['adapter'])), use_tqdm=False)
        seconds = time.monotonic() - started
        for r, o in zip(rows, outs):
            r[name] = json.loads(o.outputs[0].text)['action_id'] == r['label']
        result['arms'][name] = dict(prompt=spec['prompt'], adapter=spec['adapter'], accuracy=statistics.mean(r[name] for r in rows),
            seconds=seconds, prompt_tokens=sum(len(p['prompt_token_ids']) for p in prompts),
            by_family={f: statistics.mean(r[name] for r in rows if r['family'] == f) for f in sorted({r['family'] for r in rows})})
    names = list(arms)
    if len(names) == 2:
        a, b = names
        games = sorted({r['game'] for r in rows})
        per = {g: [r[a] - r[b] for r in rows if r['game'] == g] for g in games}
        rng = random.Random(1701)
        boot = []
        for _ in range(20000):
            pick = [per[g] for g in rng.choices(games, k=len(games))]
            boot.append(sum(map(sum, pick)) / sum(map(len, pick)))
        boot.sort()
        result['paired'] = dict(a=a, b=b, a_minus_b=statistics.mean(r[a] - r[b] for r in rows),
            game_cluster_bootstrap_95=[boot[499], boot[19499]], a_only=sum(r[a] and not r[b] for r in rows),
            b_only=sum(r[b] and not r[a] for r in rows), both=sum(r[a] and r[b] for r in rows))
    (root/'imitation.json').write_text(json.dumps(result, indent=2)+'\n')
    with open(root/'imitation-rows.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != 'messages'})+'\n')
    volume.commit()
    return result


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={'/workspace/runs': volume})
def ability_census(run_id):
    """Which abilities each arm actually used, and how the expert uses them in the training corpus."""
    scripts_path()
    from summarize_expert_sft import rows
    from polytopia_bot.learning.compact_prompt import ABILITIES
    out = {}
    for name, base in (('expert_sft_v1', RUNS/SOURCE/'expert_sft'), *((a, RUNS/run_id/a) for a in ('compact_sft', 'compact_sft_prefix_cache'))):
        jobs = json.loads((base/'games'/'jobs.json').read_text())
        c, offered, games_with = Counter(), Counter(), Counter()
        for j in jobs:
            used = set()
            for r in rows(base/'games'/(j['id']+'.jsonl.gz')):
                if r.get('policy') != 'model' or 'selected_command' not in r:
                    continue
                cmd = r['selected_command']
                offered['decisions'] += 1
                if any(a['command']['kind'] == 3 for a in r['observation']['legal_actions']):
                    offered['ability_offered'] += 1
                if cmd['kind'] == 3:
                    c[ABILITIES.get(cmd['parameter'], cmd['parameter'])] += 1
                    used.add(ABILITIES.get(cmd['parameter'], cmd['parameter']))
            games_with.update(used)
        out[name] = dict(abilities=dict(c.most_common()), games_using=dict(games_with), **offered)
    corpus = Counter()
    shard_dir = RUNS/run_id/'prepared-shards'
    for f in sorted(shard_dir.glob('shard-*.jsonl')):
        for line in open(f):
            r = json.loads(line)
            if r['command']['kind'] == 3:
                corpus[ABILITIES.get(r['command']['parameter'], r['command']['parameter'])] += 1
    out['training_corpus_labels'] = dict(corpus.most_common())
    return out


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes={'/workspace/runs': volume})
def destroy_trace(run_id, arm):
    """Per-game destroy counts, and the model's command sequence around the first destroys of the worst game."""
    scripts_path()
    from summarize_expert_sft import rows
    from polytopia_bot.learning.compact_prompt import _Tiles, command_text
    base = RUNS/run_id/arm/'games'
    jobs = json.loads((base/'jobs.json').read_text())
    per, seqs = {}, {}
    for j in jobs:
        seq = []
        for r in rows(base/(j['id']+'.jsonl.gz')):
            if r.get('policy') == 'model' and 'selected_command' in r:
                obs = r['observation']
                t = _Tiles(obs['visible_state']['board_size'])
                tiles = {x['tile']: x for x in obs['visible_state']['tiles']}
                c = r['selected_command']
                extra = ''
                if c['kind'] == 3 and c['parameter'] == 4:
                    tile = tiles[c['target']]
                    extra = f" [structure {tile.get('structure')} owner {tile.get('owner')} stars {obs['visible_state']['stars']}]"
                seq.append(f"t{obs['visible_state']['turn']} {command_text(t, c)}{extra}")
        per[j['id']] = sum(' demolish_own>' in s for s in seq)
        seqs[j['id']] = seq
    worst = max(per, key=per.get)
    first = next(i for i, s in enumerate(seqs[worst]) if ' demolish_own>' in s)
    return dict(per_game=per, worst=worst, excerpt=seqs[worst][max(0, first-15):first+40])


def selection():
    by = {}
    for line in open(RUNS/SOURCE/'prepared'/'index.jsonl'):
        r = json.loads(line)
        job, decision = r['id'].rsplit(':', 1)
        by.setdefault(job, {})[decision] = r['label']
    return by


@app.function(image=image, cpu=1, memory=4096, timeout=24*3600, max_containers=1, retries=0,
              volumes={'/workspace/runs': volume})
def coordinate(run_id, protocol_sha):
    import traceback
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    status_file = root/'collection.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else dict(status='starting', started_unix=time.time(), stages={}, calls={})
    status['restarts'] = status.get('restarts', -1) + 1
    def publish(**kw):
        status.update(kw, observed_unix=time.time())
        status_file.write_text(json.dumps(status, indent=2)+'\n'); volume.commit()
    try:
        publish(status='preparing')
        if not (root/'prepared'/'manifest.json').exists():
            started = time.monotonic()
            sel = selection()
            files = sorted(e.path.split('/')[-1] for e in volume.listdir(f'{SOURCE}/expert-games'))
            n = p['data']['prepare_shards']
            shards = [files[i::n] for i in range(n)]
            args = [(run_id, protocol_sha, i, s, {f[:-len('.jsonl.gz')]: sel.get(f[:-len('.jsonl.gz')], {}) for f in s})
                    for i, s in enumerate(shards)]
            results = list(prepare_shard.starmap(args))
            manifest = combine_shards.remote(run_id, protocol_sha, results)
            status['stages']['prepare'] = dict(seconds=time.monotonic()-started, **{k: manifest[k] for k in
                ('records', 'duplicates_dropped', 'conflicting_duplicate_labels', 'max_tokens', 'mean_tokens', 'total_tokens', 'families')})
        publish(status='training')
        run_file = root/'training'/'model'/'run.json'
        if not (run_file.exists() and json.loads(run_file.read_text())['status'] == 'completed'):
            if (root/'training').exists():
                raise RuntimeError('incomplete training directory exists; inspect before retrying')
            started = time.monotonic()
            status['stages']['train'] = dict(**train.remote(run_id, protocol_sha), seconds=time.monotonic()-started)
        publish(status='evaluating')
        calls = {}
        for arm in p['arms']:
            if (root/arm/'report.json').exists():
                continue
            if arm in status['calls']:  # restarted coordinator: reattach instead of relaunching
                calls[arm] = modal.FunctionCall.from_id(status['calls'][arm])
            else:
                calls[arm] = evaluate.spawn(run_id, protocol_sha, arm)
                status['calls'][arm] = calls[arm].object_id
        publish()
        status['stages']['evaluate'] = {arm: c.get() for arm, c in calls.items()}
        publish(status='comparing')
        small = compare.remote(run_id, protocol_sha)
        publish(status='complete', summary=small, finished_unix=time.time())
    except Exception:
        publish(status='failed', error=traceback.format_exc(), finished_unix=time.time())
        raise
    return status


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={'/workspace/runs': volume})
def smoke_data(run_id, protocol_sha):
    """Two v1 games: label round trip, compact tokens vs v1 tokens for the same selected decisions."""
    scripts_path()
    from prepare_compact_sft import shard
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    sel = selection()
    files = sorted(e.path.split('/')[-1] for e in volume.listdir(f'{SOURCE}/expert-games'))[:2]
    chosen = {f[:-len('.jsonl.gz')]: sel[f[:-len('.jsonl.gz')]] for f in files}
    (root/'prepared-shards').mkdir(exist_ok=True)
    sh = shard([RUNS/SOURCE/'expert-games'/f for f in files], chosen, p['training']['config'], root/'prepared-shards'/'smoke.jsonl')
    rows = [json.loads(l) for l in open(root/'prepared-shards'/'smoke.jsonl')]
    from polytopia_bot.learning.training import load_tokenizer
    tok = load_tokenizer(p['training']['config'])
    labels = [tok.decode([t for t in r['labels'] if t != -100]) for r in rows]
    if any(lab != '{"action_id":"%s"}<|im_end|>' % r['label'] for lab, r in zip(labels, rows)):
        raise ValueError('supervised suffix is not the label')
    volume.commit()
    lengths = [len(r['input_ids']) for r in rows]
    return dict(files=files, shard=sh, records=len(rows), mean_tokens=sum(lengths)/len(lengths), max_tokens=max(lengths),
                sample_label=labels[0])


@app.function(image=image, cpu=2, memory=8192, timeout=7200, retries=0, volumes={'/workspace/runs': volume})
def smoke_train(run_id, protocol_sha):
    """Combine the smoke shard and train it with the real 4-GPU trainer (2 updates), then a LoRA preflight."""
    scripts_path()
    from prepare_expert_sft import combine
    from polytopia_bot.learning.interface import digest
    from polytopia_bot.learning.training import load_tokenizer
    root = RUNS/run_id
    p = check_protocol(root, protocol_sha)
    volume.reload()
    tok = load_tokenizer(p['training']['config'])
    shard = dict(stats={}, tokenizer_hash=digest(tok.backend_tokenizer.to_str()), chat_template_hash=digest(tok.chat_template),
                 path=str(root/'prepared-shards'/'smoke.jsonl'))
    manifest = combine([shard], root/'prepared', p['training']['config'], expert=p['expert'], data_protocol=dict(smoke=True))
    volume.commit()
    result = train.remote(run_id, protocol_sha)
    volume.reload()
    run = json.loads((root/'training'/'model'/'run.json').read_text())
    pre = evaluate.remote(run_id, protocol_sha, 'compact_sft', preflight_only=True)
    volume.reload()
    runtime = json.loads((root/'compact_sft'/'runtime.json').read_text())
    return dict(manifest={k: manifest[k] for k in ('records', 'max_tokens', 'mean_tokens', 'total_tokens')}, train=result,
                stress=run.get('corpus_stress'), steps=run.get('global_step'), losses=[l['loss'] for l in run['logs']],
                preflight=pre, preflight_detail=runtime['preflight'])


def build_protocol(run_id, smoke=False):
    v1 = json.loads((ROOT/'runs'/SOURCE/'protocol.json').read_text())
    used = ['src/polytopia_bot/learning/compact_prompt.py', 'scripts/prepare_compact_sft.py', 'scripts/benchmark_compact_sft.py',
            'scripts/prepare_expert_sft.py', 'scripts/expert_train_distributed.py', 'scripts/benchmark_qwen3_instruct.py',
            'src/polytopia_bot/learning/expert_trainer.py', 'src/polytopia_bot/learning/expert_data.py',
            'src/polytopia_bot/learning/async_gameplay.py', 'src/polytopia_bot/learning/interface.py',
            'src/polytopia_bot/learning/adapter.py', 'src/polytopia_bot/learning/teacher.py',
            'src/polytopia_bot/learning/thinking_gameplay.py', 'src/polytopia_bot/learning/thinking_readable.py',
            'src/polytopia_bot/learning/thinking_debug.py', 'src/polytopia_bot/learning/decision_harness.py',
            'src/polytopia_bot/learning/encoding.py', 'src/polytopia_bot/learning/data.py',
            'src/polytopia_bot/learning/fast_training.py', 'src/polytopia_bot/learning/training.py',
            'scripts/benchmark_thinking_unbounded.py']
    hashes = {rel: sha(ROOT/rel) for rel in used}
    unchanged_from_v1 = {k: v1['source_hashes'][k] == hashes[k] for k in hashes if k in v1['source_hashes']}
    adapter = f'{run_id}/training/model/adapter-final'
    arms = dict(compact_sft=dict(adapter=adapter, adapter_files=None, training_receipt=f'{run_id}/training/model/run.json'))
    if not smoke:
        # Same adapter, prefix caching on: measures what the prefix-friendly order buys at serving time.
        arms['compact_sft_prefix_cache'] = dict(arms['compact_sft'], prefix_caching=True)
    return dict(schema='compact-prompt-ablation-v1', run_id=run_id, created_unix=time.time(), smoke=smoke,
        model_id=v1['model_id'], revision=v1['revision'], expert=v1['expert'],
        data=dict(source=f'{SOURCE}/prepared/index.jsonl (same 39,626 game/decision/label rows, same shard order)',
                  games=f'{SOURCE}/expert-games', prepare_shards=v1['data']['prepare_shards'], smoke=smoke,
                  prompt='render_compact_decision (polytopia-compact-v2.1)',
                  label='same prompt-local aN as v1; catalog equality and state round trip checked per example'),
        training=dict(v1['training'], recipe=v1['training']['recipe'] + ' (identical to expert-sft-v1)'),
        arms=arms, lora=v1['lora'], harness='polytopia-compact-v2.1-legal-json', constrain_actions=True,
        max_length=v1['max_length'], max_new_tokens=None, thinking=False, seed=v1['seed'], sampling=v1['sampling'],
        concurrency=v1['concurrency'], serving=v1['serving'], gpu='H100', jobs=v1['jobs'],
        source_hashes=hashes, unchanged_from_v1=unchanged_from_v1,
        comparison=dict(run_id=SOURCE, arm='expert_sft', protocol_sha256=sha(ROOT/'runs'/SOURCE/'protocol.json')))


def new_run(run_id, smoke=False):
    local = ROOT/'runs'/run_id
    if local.exists():
        raise ValueError(f'{local} exists; choose a new run id')
    local.mkdir(parents=True)
    (local/'protocol.json').write_text(json.dumps(build_protocol(run_id, smoke), indent=2)+'\n')
    with volume.batch_upload() as up:
        up.put_file(local/'protocol.json', f'{run_id}/protocol.json')
    return local, sha(local/'protocol.json')


@app.local_entrypoint()
def main(action: str = 'scan', run_id: str = '', limit: int = 0):
    out_dir = ROOT/'runs'/'compact-prompt-v2-measure'
    if action == 'scan':
        files = sorted(e.path.split('/')[-1] for e in volume.listdir(f'{SOURCE}/expert-games'))
        ids = {k: set() for k in ('units', 'technologies', 'structures')}
        stats = Counter()
        for got, st in scan_files.map([files[i::50] for i in range(50)]):
            for k in ids:
                ids[k] |= set(got[k])
            stats.update(st)
        out = dict(files=len(files), ids={k: sorted(v) for k, v in ids.items()}, stats=dict(stats.most_common()))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir/'scan.json').write_text(json.dumps(out, indent=1)+'\n')
        print(json.dumps(out)[:3000])
        return
    if action == 'test':
        r = run_tests.remote(limit or 40)
        print(r['output'])
        if r['returncode']:
            raise SystemExit(r['returncode'])
        return
    if action == 'measure':
        v1 = json.loads((ROOT/'runs'/SOURCE/'protocol.json').read_text())
        files = sorted(e.path.split('/')[-1] for e in volume.listdir(f'{SOURCE}/expert-games'))
        if limit:
            files = files[:limit]
        chunks = [c for c in (files[i::100] for i in range(100)) if c]
        agg = dict(old=[], new=[], old_lcp=[], new_lcp=[], same_turn=[], old_chars=0, new_chars=0, failures=[],
                   sections=Counter(), section_examples=0)
        for r in measure_files.map(chunks, kwargs=dict(config=v1['training']['config'])):
            for k in ('old', 'new', 'old_lcp', 'new_lcp', 'same_turn', 'failures'):
                agg[k] += r[k]
            for k in ('old_chars', 'new_chars', 'section_examples'):
                agg[k] += r[k]
            agg['sections'].update(r['sections'])
            agg['new_system'], agg['old_system'] = r['new_system'] or agg.get('new_system'), r['old_system'] or agg.get('old_system')
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir/'measure-raw.json').write_text(json.dumps({k: (dict(v) if isinstance(v, Counter) else v) for k, v in agg.items()}))
        print(json.dumps(summarize_measure(agg), indent=1))
        (out_dir/'measure.json').write_text(json.dumps(summarize_measure(agg), indent=2)+'\n')
        return
    if action == 'smoke':
        run_id = run_id or f'compact-prompt-smoke-{int(time.time())}'
        local, protocol_sha = new_run(run_id, smoke=True)
        data = smoke_data.remote(run_id, protocol_sha)
        print('SMOKE DATA', json.dumps(data, indent=1))
        tr = smoke_train.remote(run_id, protocol_sha)
        print('SMOKE TRAIN+PREFLIGHT', json.dumps(tr, indent=1)[:6000])
        (local/'smoke.json').write_text(json.dumps(dict(data=data, train=tr), indent=2)+'\n')
        return
    if action == 'launch':
        if not run_id:
            raise ValueError('run_id required')
        local, protocol_sha = new_run(run_id)
        call = coordinate.spawn(run_id, protocol_sha)
        receipt = dict(app_id=app.app_id, call_id=call.object_id, protocol_sha256=protocol_sha, launcher_sha256=sha(Path(__file__)))
        (local/'dispatch.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt))
        return
    if action == 'imitation':
        # 20 validation maps (100541-100560, never trained on) x 2 seats vs balanced; 50 decisions per game.
        v1 = json.loads((ROOT/'runs'/SOURCE/'protocol.json').read_text())
        maps = json.loads((ROOT/'runs/sft0/splits-v1/maps.json').read_text())['maps']['validation'][:20]
        jobs = [dict(id=f'heldout-{m["seed"]}-seat-{s}', seed=m['seed'], board_hash=m['board_hash'], seat=s, opponent='balanced')
                for m in maps for s in (1, 2)]
        train_seeds = {j['seed'] for j in v1['data']['jobs']}
        if {j['seed'] for j in jobs} & train_seeds:
            raise ValueError('held-out maps overlap training maps')
        games = list(heldout_game.map(jobs, kwargs=dict(expert=v1['expert']), return_exceptions=True))
        print('games', sum(not isinstance(g, Exception) for g in games), [repr(g)[:200] for g in games if isinstance(g, Exception)][:3])
        arms = dict(readable_v1=dict(prompt='readable', adapter=f'{SOURCE}/training/model/adapter-final'),
                    compact_v2=dict(prompt='compact', adapter=f'{run_id or "compact-prompt-v2"}/training/model/adapter-final'))
        res = imitation.remote(limit or 50, arms)
        out = ROOT/'runs'/IMITATION
        out.mkdir(parents=True, exist_ok=True)
        (out/'imitation.json').write_text(json.dumps(dict(jobs=jobs, **res), indent=2)+'\n')
        print(json.dumps(res, indent=1))
        return
    if action == 'abilities':
        res = ability_census.remote(run_id or 'compact-prompt-v2')
        (ROOT/'runs'/(run_id or 'compact-prompt-v2')/'abilities.json').write_text(json.dumps(res, indent=2)+'\n')
        print(json.dumps(res, indent=1))
        return
    if action == 'destroy-trace':
        runs = [(r, a) for r in ('compact-prompt-v2', 'compact-prompt-v2.1') for a in ('compact_sft', 'compact_sft_prefix_cache')]
        res = {f'{r}/{a}': x for (r, a), x in zip(runs, destroy_trace.starmap(runs))}
        (ROOT/'runs'/'compact-prompt-v2.1'/'demolition.json').write_text(json.dumps(res, indent=2)+'\n')
        print(json.dumps({k: v['per_game'] for k, v in res.items()}))
        return
    if action == 'compare':
        local = ROOT/'runs'/run_id
        small = compare.remote(run_id, sha(local/'protocol.json'))
        (local/'comparison-small.json').write_text(json.dumps(small, indent=2)+'\n')
        print(json.dumps(small, indent=1)[:8000])
        return
    raise ValueError(action)


def summarize_measure(agg):
    import statistics
    def stats(xs):
        xs = sorted(xs)
        return dict(n=len(xs), mean=sum(xs)/len(xs), median=xs[len(xs)//2], p90=xs[int(len(xs)*.9)], max=xs[-1])
    pairs = [(o, n, ol, nl, s) for o, n, ol, nl, s in zip(agg['old'], agg['new'], agg['old_lcp'], agg['new_lcp'], agg['same_turn'])
             if ol is not None]
    def prefix(sel):
        rows = [r for r in pairs if sel(r)]
        return dict(pairs=len(rows),
            old_shared_fraction=statistics.mean(r[2]/r[0] for r in rows), new_shared_fraction=statistics.mean(r[3]/r[1] for r in rows),
            old_uncached_tokens=statistics.mean(r[0]-r[2] for r in rows), new_uncached_tokens=statistics.mean(r[1]-r[3] for r in rows))
    sec = agg['sections']; total = sum(sec.values())
    return dict(examples=len(agg['old']), failures=len(agg['failures']), failure_samples=agg['failures'][:5],
        old_tokens=stats(agg['old']), new_tokens=stats(agg['new']),
        token_ratio_new_over_old=sum(agg['new'])/sum(agg['old']),
        old_system_tokens=agg.get('old_system'), new_system_tokens=agg.get('new_system'),
        old_user_chars_mean=agg['old_chars']/len(agg['old']), new_user_chars_mean=agg['new_chars']/len(agg['new']),
        prefix_all=prefix(lambda r: True), prefix_same_turn=prefix(lambda r: r[4]), prefix_next_turn=prefix(lambda r: not r[4]),
        new_user_section_tokens_mean={k: v/agg['section_examples'] for k, v in sorted(sec.items(), key=lambda kv: -kv[1])},
        new_user_section_share={k: v/total for k, v in sorted(sec.items(), key=lambda kv: -kv[1])})
