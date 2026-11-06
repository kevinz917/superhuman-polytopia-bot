"""Select and tokenize SFT-pipeline games into a prepared corpus (run on Modal).

shard(files, ...)   -> one shard's tokenized rows (+ a .json stats marker), idempotent
combine(shards,...) -> prepared/{tokens,index}.jsonl + manifest.json in the schema the
                       expert trainer verifies (polytopia-expert-prepared-v1)

Selection rules live in ``polytopia_bot.learning.sft_pipeline.select_examples``.
Duplicated states (same system+user text) keep their first label; family upweighting
writes extra copies of a row after deduplication. ``max_examples`` keeps a fixed,
hash-ranked subset of unique states (independent of sharding).
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polytopia_bot.learning.data import tokenize_example  # noqa: E402
from polytopia_bot.learning.expert_data import read_game  # noqa: E402
from polytopia_bot.learning.interface import digest  # noqa: E402
from polytopia_bot.learning.sft_pipeline import resolve_renderer, select_examples  # noqa: E402


def shard(files, config, data, renderer_spec, out, fingerprint):
    from polytopia_bot.learning.training import load_tokenizer
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    marker = out.with_suffix('.json')
    if marker.exists():
        done = json.loads(marker.read_text())
        if done['fingerprint'] == fingerprint and out.exists() and out.stat().st_size == done['bytes']:
            return dict(done, resumed=True)
    started = time.monotonic()
    response = resolve_renderer(renderer_spec)['response']
    tokenizer = load_tokenizer(config)
    stats, families, tokens = Counter(), Counter(), 0
    tmp = out.with_name(f'.{out.name}.{os.getpid()}.tmp')
    with open(tmp, 'w') as sink:
        for path in files:
            header, examples, result = read_game(path)
            stats['games'] += 1
            if header.get('renderer', renderer_spec['name']) != renderer_spec['name']:
                raise ValueError(f'{path}: rendered with {header.get("renderer")}, protocol wants {renderer_spec["name"]}')
            picked, s = select_examples(examples, result, header['job']['id'], data)
            stats.update(s)
            for ex, copies in picked:
                messages = [{'role': 'system', 'content': header['system']}, {'role': 'user', 'content': ex['user']}]
                try:
                    tok = tokenize_example(tokenizer, messages, response(ex['label']),
                                           max_length=config['max_length'], max_new_tokens=config['max_new_tokens'])
                except ValueError as exc:
                    if 'exceeds context' not in str(exc):  # only the context cap is tolerated (counted, never truncated)
                        raise
                    stats['skipped_too_long'] += 1
                    continue
                sink.write(json.dumps(dict(**tok, id=f"{header['job']['id']}:{ex['decision']}", family=ex['family'],
                    turn=ex['turn'], copies=copies, label=ex['label'],
                    key=hashlib.sha256((header['system']+'\0'+ex['user']).encode()).hexdigest()))+'\n')
                stats['tokenized'] += 1; families[ex['family']] += 1; tokens += len(tok['input_ids'])
    tmp.replace(out)
    done = dict(fingerprint=fingerprint, path=str(out), bytes=out.stat().st_size, stats=dict(stats),
                families=dict(families), tokens=tokens, seconds=time.monotonic()-started,
                tokenizer_hash=digest(tokenizer.backend_tokenizer.to_str()), chat_template_hash=digest(tokenizer.chat_template))
    marker.write_text(json.dumps(done, indent=2))
    return done


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def _rank(seed, row_id):
    return hashlib.sha256(f'{seed}:{row_id}'.encode()).hexdigest()


def combine(shards, output, config, *, data, provenance):
    """``shards``: shard() results in a fixed order. Writes ``output`` atomically (tmp dir + rename)."""
    output = Path(output)
    tmp = output.with_name(f'.{output.name}.{os.getpid()}.tmp')
    tmp.mkdir(parents=True)
    started = time.monotonic()
    # Pass 1: unique states (first occurrence wins), optional global cap.
    first, conflicts, duplicates = {}, 0, 0
    for sh in shards:
        with open(sh['path']) as f:
            for line in f:
                r = json.loads(line)
                if r['key'] in first:
                    duplicates += 1
                    conflicts += first[r['key']][1] != r['label']
                    continue
                first[r['key']] = (r['id'], r['label'], r['copies'])
    keep = set(first)
    if data['max_examples']:
        # Budget on final rows (upweight copies included); unique states enter in a fixed hash order.
        keep, rows = set(), 0
        for k in sorted(first, key=lambda k: _rank(data['sample_seed'], first[k][0])):
            if rows + first[k][2] > data['max_examples']:
                continue
            keep.add(k); rows += first[k][2]
    capped = len(first) - len(keep)
    # Pass 2: stream kept rows (with upweight copies) to disk.
    stats, hashes, families, lengths, unique = Counter(), set(), Counter(), [], 0
    sha = hashlib.sha256()
    written = set()
    with open(tmp/'tokens.jsonl', 'w') as tokens, open(tmp/'index.jsonl', 'w') as index:
        for sh in shards:
            stats.update(sh['stats']); hashes.add((sh['tokenizer_hash'], sh['chat_template_hash']))
            with open(sh['path']) as f:
                for line in f:
                    r = json.loads(line)
                    if r['key'] not in keep or r['key'] in written or first[r['key']][0] != r['id']:
                        continue
                    written.add(r['key']); unique += 1
                    text = json.dumps({k: r[k] for k in ('input_ids', 'labels', 'attention_mask')})+'\n'
                    for copy in range(r['copies']):
                        tokens.write(text); sha.update(text.encode())
                        index.write(json.dumps(dict(id=r['id'], copy=copy, family=r['family'], turn=r['turn'],
                                                    label=r['label']))+'\n')
                        families[r['family']] += 1; lengths.append(len(r['input_ids']))
    if len(hashes) != 1:
        raise ValueError('shards used different tokenizers')
    if len(lengths) < data['min_examples']:
        raise ValueError(f'only {len(lengths)} examples (< min_examples {data["min_examples"]})')
    tok_hash, tmpl_hash = hashes.pop()
    if sha.hexdigest() != file_sha256(tmp/'tokens.jsonl'):
        raise ValueError('token file hash mismatch')
    manifest = dict(schema='polytopia-expert-prepared-v1', config_hash=digest(config),
        tokens_sha256=sha.hexdigest(), tokenizer_hash=tok_hash, chat_template_hash=tmpl_hash,
        records=len(lengths), unique_states=unique, duplicates_dropped=duplicates, conflicting_duplicate_labels=conflicts,
        capped_by_max_examples=capped, max_tokens=max(lengths), mean_tokens=sum(lengths)/len(lengths),
        total_tokens=sum(lengths), families=dict(families.most_common()),
        move_fraction=families['move']/len(lengths), source_stats=dict(stats),
        sources=['rl_expert'], combine_seconds=time.monotonic()-started, **provenance)
    (tmp/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    tmp.replace(output)
    return manifest
