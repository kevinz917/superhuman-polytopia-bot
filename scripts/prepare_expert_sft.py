"""Sample and tokenize RL-expert examples into a prepared SFT corpus (run on Modal).

shard(files, config, per_game, seed, out) -> writes tokenized rows for a fixed list of game files
combine(shard_files, ...)                 -> tokens.jsonl + manifest.json with hashes and counts
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polytopia_bot.learning.data import tokenize_example  # noqa: E402
from polytopia_bot.learning.expert_data import read_game, response_text  # noqa: E402
from polytopia_bot.learning.interface import digest  # noqa: E402


def game_sample(examples, job_id, per_game, seed):
    rng = random.Random(f'{seed}:{job_id}')
    if len(examples) <= per_game:
        return list(examples)
    picked = sorted(rng.sample(range(len(examples)), per_game))
    return [examples[i] for i in picked]


def shard(files, config, per_game, seed, out):
    from polytopia_bot.learning.training import load_tokenizer
    tokenizer = load_tokenizer(config)
    stats = Counter(); out = Path(out); tmp = out.with_suffix('.tmp')
    sink = open(tmp, 'w')
    for path in files:
        header, examples, result = read_game(path)
        stats['games'] += 1; stats['expert_decisions'] += len(examples)
        stats['game_wins'] += result['reward'] > 0
        for ex in game_sample(examples, header['job']['id'], per_game, seed):
            messages = [{'role': 'system', 'content': header['system']}, {'role': 'user', 'content': ex['user']}]
            try:
                tok = tokenize_example(tokenizer, messages, response_text(ex['label']),
                                       max_length=config['max_length'], max_new_tokens=config['max_new_tokens'])
            except ValueError as exc:
                # Only the context cap is tolerated (counted, never truncated).
                if 'exceeds context' not in str(exc):
                    raise
                stats['skipped_too_long'] += 1
                continue
            sink.write(json.dumps(dict(**tok, id=f"{header['job']['id']}:{ex['decision']}", family=ex['family'],
                             turn=ex['turn'], key=hashlib.sha256((header['system']+'\0'+ex['user']).encode()).hexdigest(),
                             label=ex['label']))+'\n')
            stats['sampled'] += 1
    sink.close(); tmp.replace(out)
    return dict(stats=dict(stats), tokenizer_hash=digest(tokenizer.backend_tokenizer.to_str()),
                chat_template_hash=digest(tokenizer.chat_template), path=str(out))


def combine(shards, output, config, *, expert, data_protocol):
    """``shards``: shard() results in a fixed order. Rows stream straight to disk."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seen, conflicts, duplicates = {}, 0, 0
    stats, tokenizer_hashes, families, lengths = Counter(), set(), Counter(), []
    sha = hashlib.sha256()
    with open(output/'tokens.jsonl', 'w') as tokens, open(output/'index.jsonl', 'w') as index:
        for sh in shards:
            stats.update(sh['stats']); tokenizer_hashes.add((sh['tokenizer_hash'], sh['chat_template_hash']))
            with open(sh['path']) as f:
                for line in f:
                    r = json.loads(line)
                    if r['key'] in seen:
                        duplicates += 1
                        conflicts += seen[r['key']] != r['label']
                        continue
                    seen[r['key']] = r['label']
                    text = json.dumps({k: r[k] for k in ('input_ids', 'labels', 'attention_mask')})+'\n'
                    tokens.write(text); sha.update(text.encode())
                    index.write(json.dumps({k: r[k] for k in ('id', 'family', 'turn', 'label')})+'\n')
                    families[r['family']] += 1; lengths.append(len(r['input_ids']))
    if len(tokenizer_hashes) != 1:
        raise ValueError('shards used different tokenizers')
    tok_hash, tmpl_hash = tokenizer_hashes.pop()
    if sha.hexdigest() != hashlib.sha256((output/'tokens.jsonl').read_bytes()).hexdigest():
        raise ValueError('token file hash mismatch')
    manifest = dict(schema='polytopia-expert-prepared-v1', config_hash=digest(config),
        tokens_sha256=sha.hexdigest(), tokenizer_hash=tok_hash, chat_template_hash=tmpl_hash,
        records=len(lengths), duplicates_dropped=duplicates, conflicting_duplicate_labels=conflicts,
        max_tokens=max(lengths), mean_tokens=sum(lengths)/len(lengths), total_tokens=sum(lengths),
        families=dict(families.most_common()), source_stats=dict(stats),
        sources=['rl_expert'], expert=expert, data_protocol=data_protocol)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
