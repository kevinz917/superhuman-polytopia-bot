"""Configurable expert-SFT pipeline: config schema, prompt-renderer registry, players,
game recording, example selection and cost estimate.

Pure Python (no Modal). Orchestration lives in ``experiments/modal_sft_pipeline.py``;
tokenization in ``scripts/sft_pipeline_prepare.py``; evaluation reuses
``scripts/benchmark_expert_sft.py`` through ``scripts/sft_pipeline_eval.py``.
See ``docs/sft-pipeline-config-2026-10-19.md``.
"""
from collections import Counter
import copy
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import time

from .expert_data import FAMILIES
from .interface import digest

SCHEMA = 'sft-pipeline-config-v1'
SCRIPTED = ('balanced', 'pressure', 'economy')
STAGES = ('generate', 'prepare', 'train', 'evaluate', 'summarize')
FORCED_KINDS = {10, 11}  # end_turn, resign

# Prompt renderers by name. ``render(obs) -> prompt`` with ``.messages`` and ``.catalog``
# (prompt-local label -> LegalAction); ``schema(messages)`` is the evaluation grammar;
# ``response(label)`` is the supervised target string. A future compact prompt registers
# here (or a config may name ``module:function`` directly for ``render``).
RENDERERS = {
    'readable-v1': dict(
        render='polytopia_bot.learning.thinking_readable:render_readable_decision',
        schema='polytopia_bot.learning.thinking_readable:action_schema',
        response='polytopia_bot.learning.expert_data:response_text',
        harness='polytopia-named-fields-v1-legal-json'),
    # Compact prompt v2 (docs/compact-prompt-v2-2026-10-19.md); same {"action_id": "aN"} target.
    'compact-v2': dict(
        render='polytopia_bot.learning.compact_prompt:render_compact_decision',
        schema='polytopia_bot.learning.compact_prompt:action_schema',
        response='polytopia_bot.learning.expert_data:response_text',
        harness='polytopia-compact-v2-legal-json'),
}


def register_renderer(name, *, render, schema, response='polytopia_bot.learning.expert_data:response_text',
                      harness=None):
    RENDERERS[name] = dict(render=render, schema=schema, response=response, harness=harness or name)


def _load(ref):
    module, _, name = ref.partition(':')
    return getattr(importlib.import_module(module), name)


def renderer_spec(name):
    if name in RENDERERS:
        return dict(RENDERERS[name], name=name)
    if ':' in name:
        return dict(render=name, schema=None, response=RENDERERS['readable-v1']['response'], harness=name, name=name)
    raise ValueError(f'unknown prompt renderer {name!r}; registered: {sorted(RENDERERS)}')


def resolve_renderer(spec):
    """``spec``: a renderer_spec() dict (as stored in the protocol). Returns callables."""
    return dict(render=_load(spec['render']), schema=_load(spec['schema']) if spec.get('schema') else None,
                response=_load(spec['response']))


# ---------------------------------------------------------------- config

DEFAULTS = {
    'schema': SCHEMA,
    'description': '',
    'stages': list(STAGES),
    'players': {},
    'prompt': {'renderer': 'readable-v1'},
    'generation': {
        'experts': [],
        'map_manifest': 'runs/sft0/splits-v1/maps.json',
        'map_pool': [{'split': 'train'}],
        'seats': [1, 2],
        'opponents': [{'name': 'balanced'}],
        'games_per_call': 8,
        'max_failed_games': 0.02,
        'max_rounds': 3,
        'record_trajectory': True,
        'reuse_run': None,
    },
    'data': {
        'examples_per_game': 40,
        'sample_seed': 17,
        'skip_forced': True,
        'max_move_fraction': None,
        'upweight': {},
        'outcomes': ['win', 'loss', 'draw', 'truncated'],
        'max_examples': None,
        'min_examples': 64,
        'games_per_shard': 25,
    },
    'training': {
        'base': 'runs/movement-scale40k-v1/protocol.json',
        'gpu_type': 'B200',
        'gpus': 4,
        'seed': 17,
        'checkpoint_every': 157,
        'timeout_hours': 23.5,
        'overrides': {'epochs': 1, 'max_length': 24576},
    },
    'evaluation': {
        'harness_source': 'runs/expert-sft-v1/protocol.json',
        'gpu': 'H100',
        'maps': {'split': 'validation', 'offset': 0, 'count': 15},
        'seats': [1, 2],
        'opponents': ['balanced'],
        'shard_by_opponent': True,
        'arms': {'sft': {'adapter': 'trained'}},
        'rl_reference': [],
        'pairs': [],
        'start_independent_arms_early': True,
        'serving': {},
    },
}

TRAIN_KEYS = {'lora_rank', 'lora_alpha', 'lora_dropout', 'target_modules', 'learning_rate', 'warmup_ratio', 'epochs',
              'max_length', 'max_new_tokens', 'effective_batch_size', 'max_grad_norm', 'microbatch_size'}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ('players', 'upweight', 'arms', 'overrides', 'serving'):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path):
    raw = json.loads(Path(path).read_text())
    unknown = set(raw) - set(DEFAULTS) - {'name'}
    if unknown:
        raise ValueError(f'unknown config sections {sorted(unknown)}')
    for section in ('generation', 'data', 'training', 'evaluation', 'prompt'):
        extra = set(raw.get(section, {})) - set(DEFAULTS[section])
        if extra:
            raise ValueError(f'unknown {section} fields {sorted(extra)}')
    cfg = _merge(DEFAULTS, raw)
    cfg['name'] = raw.get('name') or Path(path).stem
    validate(cfg)
    return cfg


def player_spec(cfg, name):
    if name in cfg['players']:
        return cfg['players'][name]
    if name in SCRIPTED:
        return {'kind': 'scripted', 'style': name}
    raise ValueError(f'unknown player {name!r}: define it under "players" or use a scripted style {SCRIPTED}')


def validate(cfg):
    if cfg['schema'] != SCHEMA:
        raise ValueError(f'config schema must be {SCHEMA}')
    if not set(cfg['stages']) <= set(STAGES) or not cfg['stages']:
        raise ValueError(f'stages must be a subset of {STAGES}')
    for name, spec in cfg['players'].items():
        if spec.get('kind') == 'rl':
            if not spec.get('path', '').startswith('/retained/') or len(spec.get('sha256', '')) != 64:
                raise ValueError(f'RL player {name}: path under /retained/ and sha256 required')
            spec.setdefault('deterministic', True)
        elif spec.get('kind') == 'scripted':
            if spec.get('style') not in SCRIPTED:
                raise ValueError(f'scripted player {name}: style must be one of {SCRIPTED}')
        else:
            raise ValueError(f'player {name}: kind must be rl or scripted')
    g, d, t, e = cfg['generation'], cfg['data'], cfg['training'], cfg['evaluation']
    if 'generate' in cfg['stages']:
        if not g['experts'] and not g['reuse_run']:
            raise ValueError('generation.experts required')
        for name in g['experts']:
            if player_spec(cfg, name)['kind'] != 'rl':
                raise ValueError(f'expert {name} must be an RL checkpoint')
        for o in g['opponents']:
            player_spec(cfg, o['name'])
        if not set(g['seats']) <= {1, 2} or not g['seats']:
            raise ValueError('seats must be a subset of [1, 2]')
    n = d['examples_per_game']
    if not (n == 'all' or (isinstance(n, int) and n > 0)):
        raise ValueError('examples_per_game must be a positive integer or "all"')
    if d['max_move_fraction'] is not None and not 0 < d['max_move_fraction'] <= 1:
        raise ValueError('max_move_fraction must be in (0, 1]')
    for fam, w in d['upweight'].items():
        if fam not in FAMILIES.values() or not 0 <= w <= 10:
            raise ValueError(f'upweight {fam}: family in {sorted(FAMILIES.values())}, weight in [0, 10]')
    if not set(d['outcomes']) <= {'win', 'loss', 'draw', 'truncated'}:
        raise ValueError('outcomes must be a subset of win/loss/draw/truncated')
    if set(t['overrides']) - TRAIN_KEYS:
        raise ValueError(f'unknown training overrides {sorted(set(t["overrides"]) - TRAIN_KEYS)}')
    if t['gpus'] not in (1, 2, 4, 8):
        raise ValueError('training.gpus must be 1, 2, 4 or 8')
    if not 0 < t['timeout_hours'] <= 24:
        raise ValueError('training.timeout_hours must be in (0, 24] (Modal limit)')
    for arm, spec in e['arms'].items():
        if '@' in arm or '/' in arm:
            raise ValueError('arm names may not contain @ or /')
        a = spec.get('adapter')
        if not (a is None or a == 'trained' or (isinstance(a, dict) and 'path' in a and ('files' in a) != ('training_receipt' in a))):
            raise ValueError(f'arm {arm}: adapter must be null, "trained" or {{path, files | training_receipt}}')
        if a == 'trained' and 'train' not in cfg['stages']:
            raise ValueError(f'arm {arm} needs the train stage')
    if set(e['serving']) - {'max_num_batched_tokens', 'enable_prefix_caching', 'gpu_memory_utilization'}:
        raise ValueError('evaluation.serving: only max_num_batched_tokens, enable_prefix_caching, gpu_memory_utilization')
    for o in e['opponents']:
        s = player_spec(cfg, o)
        if s['kind'] == 'rl' and not s['deterministic']:
            raise ValueError('evaluation RL opponents must be deterministic (benchmark harness)')
    for name in e['rl_reference']:
        if player_spec(cfg, name)['kind'] != 'rl':
            raise ValueError(f'rl_reference {name} must be an RL player')
    models = set(e['arms']) | {f'rl:{n}' for n in e['rl_reference']}
    for a, b in e['pairs']:
        if a not in models or b not in models:
            raise ValueError(f'pair {a} vs {b}: both must be arms or rl:<reference>')


# ---------------------------------------------------------------- jobs

def map_pool(cfg, manifest):
    """Ordered generation map pool: manifest splits and/or fresh seeds outside the manifest."""
    pool, by_seed = [], {m['seed']: m for maps in manifest['maps'].values() for m in maps}
    for src in cfg['generation']['map_pool']:
        if 'split' in src:
            maps = sorted(manifest['maps'][src['split']], key=lambda m: m['seed'])
            off = src.get('offset', 0)
            maps = maps[off:off + src['count']] if 'count' in src else maps[off:]
            pool += [dict(seed=m['seed'], board_hash=m['board_hash'], symmetry_hash=m['symmetry_hash'],
                          source=src['split']) for m in maps]
        elif 'fresh_seeds' in src:
            start, count = src['fresh_seeds']['start'], src['fresh_seeds']['count']
            if start <= max(by_seed):
                raise ValueError('fresh seeds must start above every manifest seed')
            pool += [dict(seed=s, board_hash=None, symmetry_hash=None, source='fresh') for s in range(start, start + count)]
        else:
            raise ValueError('map_pool entries need split or fresh_seeds')
    if len({m['seed'] for m in pool}) != len(pool):
        raise ValueError('duplicate seeds in the map pool')
    return pool


def generation_jobs(cfg, pool):
    g, jobs = cfg['generation'], []
    for expert in g['experts']:
        for opp in g['opponents']:
            off = opp.get('map_offset', 0)
            count = opp.get('maps', len(pool) - off)
            maps = pool[off:off + count]
            if len(maps) != count:
                raise ValueError(f'opponent {opp["name"]}: map pool has {len(pool)} maps, needs {off + count}')
            for m in maps:
                for seat in opp.get('seats', g['seats']):
                    jid = f'{expert}-vs-{opp["name"]}-{m["seed"]}-s{seat}'
                    jobs.append(dict(id=jid, seed=m['seed'], board_hash=m['board_hash'], symmetry_hash=m['symmetry_hash'],
                                     seat=seat, expert=expert, opponent=opp['name'],
                                     game_seed=int(hashlib.sha256(jid.encode()).hexdigest()[:8], 16)))
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('duplicate generation job ids')
    return jobs


def evaluation_jobs(cfg, manifest):
    e = cfg['evaluation']
    maps = sorted(manifest['maps'][e['maps']['split']], key=lambda m: m['seed'])
    maps = maps[e['maps'].get('offset', 0):e['maps'].get('offset', 0) + e['maps']['count']]
    jobs = []
    for opp in e['opponents']:
        for m in maps:
            for seat in e['seats']:
                jobs.append(dict(id=f'eval-{m["seed"]}-s{seat}-{opp}', seed=m['seed'], board_hash=m['board_hash'],
                                 symmetry_hash=m['symmetry_hash'], seat=seat, candidate=cfg['name'], strategy='',
                                 opponent=opp, rng_index=len(jobs)))
    return jobs


def eval_arms(cfg, run_id):
    e, arms = cfg['evaluation'], {}
    for name, spec in e['arms'].items():
        a = spec.get('adapter')
        if a is None:
            base = dict(model=name, adapter=None, adapter_files=None)
        elif a == 'trained':
            base = dict(model=name, adapter=f'{run_id}/training/model/adapter-final', adapter_files=None,
                        training_receipt=f'{run_id}/training/model/run.json')
        else:
            base = dict(model=name, adapter=a['path'], adapter_files=a.get('files'))
            if 'training_receipt' in a:
                base['training_receipt'] = a['training_receipt']
        if e['shard_by_opponent']:
            for opp in e['opponents']:
                arms[f'{name}@{opp}'] = dict(base, opponent=opp)
        else:
            arms[name] = dict(base, opponent=None)
    return arms


# ---------------------------------------------------------------- players and games

class MapExcluded(ValueError):
    """The map is a held-out map (symmetry hash): the job is skipped, never retried."""


_players = {}


def make_player(spec, game_seed):
    """A fresh-per-game player. RL checkpoints are loaded once per process and hash-checked."""
    if spec['kind'] == 'scripted':
        from .teacher import ScriptedTeacher
        return ScriptedTeacher(spec['style'])
    key = (spec['path'], spec['deterministic'])
    if key not in _players:
        if hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError(f'RL checkpoint changed: {spec["path"]}')
        from polytopia_bot.rl.policy import StructuredPolicy
        _players[key] = StructuredPolicy(spec['path'], deterministic=spec['deterministic'])
    policy = _players[key]
    if not spec['deterministic']:
        import jax
        policy.key = jax.random.key(game_seed)
    return policy


_boards = {}


def board_identity(seed, size=14):
    """(board_hash, symmetry_hash) of a seed's map, the same identity as the split manifest."""
    if 'layout' not in _boards:
        import jax
        from polytopia_bot.gpu.mapgen import layout
        _boards['layout'] = jax.jit(lambda s: layout(s, size=size))
    import jax
    import jax.numpy as jnp
    from .splits import board_hashes
    return board_hashes(jax.device_get(_boards['layout'](jnp.int32(seed))), size)


def outcome(result):
    if result['status'] != 'native':
        return 'truncated'
    return 'win' if result['reward'] > 0 else 'loss' if result['reward'] < 0 else 'draw'


def play_game(job, expert, opponent, env, output, *, renderer=None, record_trajectory=True, forbidden=frozenset()):
    """Play one expert game; with ``renderer`` record every expert decision as an SFT example.

    Writes ``<output>/<id>.jsonl.gz`` (header, examples, result) atomically under a
    process-unique temporary name, so a duplicated worker can never corrupt a file.
    The format stays readable by ``expert_data.read_game``.
    """
    from .interface import parse_response
    raw, symmetric = board_identity(job['seed'])
    if job.get('board_hash') and raw != job['board_hash']:
        raise ValueError('map changed')
    if symmetric in forbidden:
        raise MapExcluded('map overlaps a held-out split (symmetry hash)')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/(job['id']+'.jsonl.gz')
    tmp = output/(f'.{job["id"]}.{os.getpid()}.{time.time_ns()}.tmp')
    started = time.monotonic()
    counts = dict(decisions=0, expert_decisions=0, forced_decisions=0)
    system, actions = None, []
    with gzip.open(tmp, 'wt', compresslevel=6) as f:
        obs = env.reset(seed=job['seed'], settings={})
        while True:
            if obs.player == job['seat']:
                chosen = expert.select(obs).action
                if renderer is not None:
                    prompt = renderer['render'](obs)
                    labels = [k for k, a in prompt.catalog.items() if a.action_id == chosen.action_id]
                    if len(labels) != 1:
                        raise ValueError('expert action is not exactly one prompt action')
                    label = labels[0]
                    if parse_response(renderer['response'](label), prompt,
                                      current_version=obs.state_version).action_id != chosen.action_id:
                        raise ValueError('label round trip failed')
                    if system is None:
                        system = prompt.messages[0]['content']
                        f.write(json.dumps(dict(header=dict(job=job, system=system, renderer=renderer['name'],
                            system_sha256=hashlib.sha256(system.encode()).hexdigest())))+'\n')
                    elif prompt.messages[0]['content'] != system:
                        raise ValueError('system prompt changed within a game')
                    kinds = Counter(FAMILIES.get(a.command['kind'], str(a.command['kind'])) for a in prompt.catalog.values())
                    forced = all(a.command['kind'] in FORCED_KINDS for a in prompt.catalog.values())
                    counts['forced_decisions'] += forced
                    counts['expert_decisions'] += 1
                    f.write(json.dumps(dict(decision=counts['expert_decisions'], turn=obs.visible_state['turn'],
                        family=FAMILIES.get(chosen.command['kind'], str(chosen.command['kind'])),
                        legal=len(prompt.catalog), legal_families=dict(kinds), forced=forced,
                        user=prompt.messages[1]['content'], label=label))+'\n')
                else:
                    counts['expert_decisions'] += 1
                action = chosen
            else:
                action = opponent.select(obs).action
            if record_trajectory:
                actions.append(int(action.action_id))
            transition = env.step(action_id=action.action_id, expected_state_version=obs.state_version)
            counts['decisions'] += 1
            if transition.terminated or transition.truncated:
                final = transition.actor_observation
                seat_view = env.observe(job['seat']).visible_state
                result = dict(**job, **counts, status='native' if transition.terminated else 'truncated',
                              reason=transition.reason, reward=transition.rewards[job['seat']],
                              round=final.visible_state['turn'], board_hash_checked=raw, symmetry_hash_checked=symmetric,
                              final_cities=sum(c['owner'] == job['seat'] for c in seat_view['cities']),
                              seconds=time.monotonic()-started)
                break
            obs = env.observe(transition.next_player)
        if system is None and renderer is not None:
            f.write(json.dumps(dict(header=dict(job=job, system=None, renderer=renderer['name'])))+'\n')
        f.write(json.dumps(dict(result=dict(result, actions=actions) if record_trajectory else result))+'\n')
    tmp.replace(path)
    return result


# ---------------------------------------------------------------- selection

def select_examples(examples, result, job_id, data):
    """Per-game, shard-independent selection. Returns [(example, copies)], Counter of stats.

    Order: outcome filter -> drop forced (only end_turn/resign legal) -> cap the plain-move
    fraction -> sample examples_per_game (or all) -> family upweighting by duplication
    (weight w: floor(w) copies plus one more with probability frac(w); 0 drops).
    """
    stats = Counter(decisions=len(examples))
    if outcome(result) not in data['outcomes']:
        stats['outcome_filtered'] += len(examples)
        return [], stats
    rng = random.Random(f'{data["sample_seed"]}:{job_id}')
    pool = list(examples)
    if data['skip_forced']:
        if pool and 'forced' not in pool[0]:
            raise ValueError('skip_forced needs games recorded by the sft pipeline (forced flag)')
        kept = [e for e in pool if not e['forced']]
        stats['forced_skipped'] += len(pool) - len(kept)
        pool = kept
    cap = data['max_move_fraction']
    if cap is not None and cap < 1:
        moves = [e for e in pool if e['family'] == 'move']
        other = [e for e in pool if e['family'] != 'move']
        keep = math.floor(cap*len(other)/(1-cap))
        if keep < len(moves):
            picked = [moves[i] for i in sorted(rng.sample(range(len(moves)), keep))]
            stats['move_capped'] += len(moves) - keep
            pool = sorted(other + picked, key=lambda e: e['decision'])
    n = data['examples_per_game']
    if n != 'all' and len(pool) > n:
        pool = [pool[i] for i in sorted(rng.sample(range(len(pool)), n))]
    out = []
    for e in pool:
        w = data['upweight'].get(e['family'], 1.0)
        copies = int(w) + (rng.random() < w - int(w))
        stats['sampled'] += 1
        stats['copies'] += copies
        if copies == 0:
            stats['downweight_dropped'] += 1
            continue
        out.append((e, copies))
    return out, stats


# ---------------------------------------------------------------- estimate

# Measured on expert-sft-v1 (2026-10-19): 1,000 league96-vs-balanced games, 356,727 expert
# decisions in 8.3 min on <=100 x 2-CPU containers; 39,626 examples (mean 9,329 tokens)
# tokenized in 5.5 min on 40 shards (sftp-smoke-3: 18 examples/s per 4-CPU shard container); 620 steps / 369.7M tokens in 6,997 s on 4 x B200.
MEASURED = dict(decisions_per_game=356.7, game_container_seconds=50.0, mean_tokens=9329,
                train_tokens_per_second_4xB200=52_840, tokenize_examples_per_container_second=10.0)
# Modal list prices (USD per second) used for the estimate.
PRICE = dict(cpu_core=0.0000131, mem_gib=0.00000222, B200=0.001736, H100=0.001097, H200=0.001261, A100=0.000694)


def estimate(cfg, n_jobs, *, measured=MEASURED, forced_fraction=0.06, move_fraction=0.47):
    d, t = cfg['data'], cfg['training']
    if not {'generate', 'prepare', 'train'} & set(cfg['stages']):
        return dict(games=0, note='no generation, preparation or training stage')
    per_game = measured['decisions_per_game']*(1-forced_fraction if d['skip_forced'] else 1)
    if d['max_move_fraction'] is not None and d['max_move_fraction'] < move_fraction:
        cap = d['max_move_fraction']
        per_game = per_game*(1-move_fraction) / (1-cap)
    if d['examples_per_game'] != 'all':
        per_game = min(per_game, d['examples_per_game'])
    examples = n_jobs*per_game
    if d['max_examples']:
        examples = min(examples, d['max_examples'])
    examples *= 1 + sum((w-1)*f for fam, w in d['upweight'].items()
                        for f in [dict(attack=.10, capture=.03, recruit=.08, build=.10).get(fam, .02)])
    epochs = t['overrides'].get('epochs', 1)
    tokens = examples*measured['mean_tokens']*epochs
    gen_cs = n_jobs*measured['game_container_seconds']
    gen_cost = gen_cs*(2*PRICE['cpu_core'] + 8*PRICE['mem_gib'])
    tok_cs = examples/measured['tokenize_examples_per_container_second']
    prep_cost = tok_cs*(4*PRICE['cpu_core'] + 16*PRICE['mem_gib'])
    train_s = tokens/(measured['train_tokens_per_second_4xB200']*t['gpus']/4) + 600
    train_cost = train_s*t['gpus']*PRICE.get(t['gpu_type'], PRICE['B200'])
    return dict(games=n_jobs, examples=round(examples), train_tokens=round(tokens), steps=math.ceil(examples/64)*epochs,
                generate=dict(container_hours=round(gen_cs/3600, 1), usd=round(gen_cost, 2),
                              wall_minutes_at_200_containers=round(gen_cs/200/60 + 3, 1)),
                prepare=dict(container_hours=round(tok_cs/3600, 1), usd=round(prep_cost, 2),
                             wall_minutes=round(tok_cs/max(1, math.ceil(n_jobs/d['games_per_shard']))/60 + examples/40000*3 + 3, 1)),
                train=dict(hours=round(train_s/3600, 2), gpu_hours=round(train_s*t['gpus']/3600, 1), usd=round(train_cost, 0),
                           fits_one_call=train_s < t['timeout_hours']*3600),
                assumptions='expert-sft-v1 throughput; readable-v1 prompt (mean 9.3k tokens); Modal list prices',
                total_usd_before_eval=round(gen_cost + prep_cost + train_cost, 0))


def fingerprint(*parts):
    return digest(list(parts))
