"""Continue thinking games that Modal worker timeouts interrupted; all on Modal.

Segments: 0 is the original run (R/shards/S/on); segment 1 is R/continuation; segment
k >= 2 is R/continuation-k. Each segment resumes the previous segment's unfinished
games from their last committed checkpoints.

prepare  R K         reconcile segment K-1's unfinished games into segment K
worker   R SHARD K   resume that shard's segment-K games on the identical serving stack
finalize R           stitch every segment into R/final and summarize

Nothing under R outside the continuation folders and R/final is modified. Games are
continued, never restarted: decision numbers and therefore per-decision sampling
seeds continue from the checkpoint.
"""
import asyncio
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polytopia_bot.learning.async_gameplay import atomic_json  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete_lines(path, gz=False, raw=False):
    """Complete JSON lines; a worker killed mid-write leaves a partial tail."""
    out = []
    if not Path(path).exists():
        return out
    if gz:
        # gzip.open drops its last decoded block when the end marker is missing,
        # which is exactly the state of a trace whose writer was killed. Stream
        # the raw deflate data instead and keep everything that was flushed.
        decoder, chunks, data = zlib.decompressobj(31), [], Path(path).read_bytes()
        try:
            for start in range(0, len(data), 1 << 16):
                chunks.append(decoder.decompress(data[start:start + (1 << 16)]))
        except zlib.error:
            pass
        text = b''.join(chunks).decode('utf-8', errors='replace')
    else:
        text = Path(path).read_text()
    # Split on newline only: model text inside rows may hold other separators.
    pieces = text.split('\n')
    for line in (p+'\n' for p in pieces[:-1]):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            break
        out.append(line if raw else parsed)
    return out


def shard_ids(root):
    p = json.loads((root/'protocol.json').read_text())
    return range(p['replicas_per_mode'])


def segment_dir(root, k):
    return Path(root)/('continuation' if k == 1 else f'continuation-{k}')


def segment_on(root, k, shard):
    """Folder holding segment k's worker output (games/, generation-timing.jsonl, report.json)."""
    if k == 0:
        return Path(root)/'shards'/str(shard)/'on'
    return segment_dir(root, k)/'shards'/str(shard)/'on'


def segment_execution(root, k, shard):
    if k == 0:
        return Path(root)/'shards'/str(shard)/'on-execution.json'
    return segment_dir(root, k)/'shards'/str(shard)/'on-execution.json'


def segment_scope(root, k, shard):
    """Games running in segment k with the counters they started from."""
    if k == 0:
        jobs = json.loads((Path(root)/'shards'/str(shard)/'protocol.json').read_text())['jobs']
        zero = dict(decisions=0, model_decisions=0, attempts=0, first_attempt_valid=0, max_prompt_tokens=0)
        return {j['id']: dict(job=j, counters=dict(zero), snapshot=None) for j in jobs}
    entries = json.loads((segment_dir(root, k)/'shards'/str(shard)/'entries.json').read_text())
    return {e['job']['id']: e for e in entries}


def segments_present(root):
    k = 0
    while segment_dir(root, k+1).exists():
        k += 1
    return k


def prepare(root, terminations, out=None, segment=1):
    from dataclasses import asdict
    from polytopia_bot.learning.adapter import GPUEnvironment
    from polytopia_bot.learning.interface import canonical
    root = Path(root); target = Path(out) if out else segment_dir(root, segment)
    if target.exists():
        raise ValueError(f'segment {segment} already prepared')
    if segment > 1 and not segment_dir(root, segment-1).exists():
        raise ValueError(f'segment {segment-1} missing')
    source_k = segment - 1
    # Build beside the target and rename only after every game reconciles.
    cont = target.with_name(target.name+'.partial')
    if cont.exists():
        shutil.rmtree(cont)
    cont.mkdir(parents=True)
    receipt = dict(schema='thinking-continuation-prepare-v2', segment=segment, source_segment=source_k,
                   protocol_sha256=sha(root/'protocol.json'), original_terminations=terminations, shards={})
    env = GPUEnvironment(14)
    for shard in shard_ids(root):
        child = root/'shards'/str(shard)/'protocol.json'
        source = segment_on(root, source_k, shard)
        scope = segment_scope(root, source_k, shard) if (source_k == 0 or (segment_dir(root, source_k)/'shards'/str(shard)/'entries.json').exists()) else {}
        finished = {g['id']: g for g in (json.loads((source/'games/games.json').read_text())
                                          if (source/'games/games.json').exists() else [])}
        if set(finished) - set(scope):
            raise ValueError(f'shard {shard}: finished game outside segment scope')
        folder = cont/'shards'/str(shard); folder.mkdir(parents=True)
        shutil.copyfile(child, folder/'protocol.json')
        if not scope or ((source/'report.json').exists() and segment_execution(root, source_k, shard).exists()):
            # Segment worker completed normally (or had nothing to run): nothing to continue.
            if set(finished) != set(scope):
                raise ValueError(f'shard {shard} report exists but coverage is incomplete')
            atomic_json(folder/'entries.json', [])
            (folder/'kept-timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in complete_lines(source/'generation-timing.jsonl')))
            (folder/'abandoned-timing.jsonl').write_text('')
            receipt['shards'][shard] = dict(completed_in_source=True, completed_originally=source_k == 0,
                                            finished_in_source=sorted(finished), continued={})
            continue
        timing = complete_lines(source/'generation-timing.jsonl')
        by_game = {}
        for row in timing:
            by_game.setdefault(row['id'], []).append(row)
        if set(by_game) - set(scope):
            raise ValueError(f'shard {shard}: timing for a game outside segment scope')
        entries, kept, abandoned, details = [], [], [], {}
        for jid, start in scope.items():
            job, s0 = start['job'], start['counters']
            rows = by_game.get(jid, [])
            if jid in finished:
                g = finished[jid]
                decisions = sorted(r['decision'] for r in rows)
                if decisions != list(range(s0['model_decisions']+1, g['attempts']+1)):
                    raise ValueError(f"timing mismatch for finished {jid}")
                kept.extend(rows)
                continue
            cp_path = source/'games/private'/(jid+'.checkpoint.json')
            cp = json.loads(cp_path.read_text())
            if cp['job'] != job:
                raise ValueError(f"checkpoint job mismatch {jid}")
            c = cp['counters']
            n = c['decisions'] - s0['decisions']           # trace rows written in the source segment
            trace = complete_lines(source/'games'/(jid+'.jsonl.gz'), gz=True)
            if n < 0 or len(trace) < n or any('selected_command' not in r for r in trace[:n]):
                raise ValueError(f"trace shorter than checkpoint for {jid}: {len(trace)} < {n}")
            model_rows = sum(r.get('policy') == 'model' for r in trace[:n])
            if model_rows != c['model_decisions'] - s0['model_decisions']:
                raise ValueError(f"model decision count mismatch for {jid}")
            obs = env.restore(base64.b64decode(cp['snapshot'], validate=True))
            if n and json.loads(canonical(asdict(obs))) != trace[n-1]['after'][obs.player-1]:
                raise ValueError(f"restored state differs from trace for {jid}")
            if not n and start['snapshot'] is not None and cp['snapshot'] != start['snapshot']:
                raise ValueError(f"no progress but checkpoint changed for {jid}")
            keep = [r for r in rows if r['decision'] <= c['model_decisions']]
            drop = [r for r in rows if r['decision'] > c['model_decisions']]
            if sorted(r['decision'] for r in keep) != list(range(s0['model_decisions']+1, c['model_decisions']+1)):
                raise ValueError(f"timing gap for {jid}")
            kept.extend(keep); abandoned.extend(drop)
            entries.append(dict(job=job, counters=c, snapshot=cp['snapshot'], trace_rows=n,
                                checkpoint_sha256=sha(cp_path)))
            details[jid] = dict(counters=c, trace_rows_total=len(trace), trace_rows_kept=n,
                                round=obs.visible_state['turn'], next_player=obs.player,
                                abandoned_decisions=[r['decision'] for r in drop],
                                abandoned_output_tokens=sum(r['output_tokens'] for r in drop))
        atomic_json(folder/'entries.json', entries)
        (folder/'kept-timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in kept))
        (folder/'abandoned-timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in abandoned))
        receipt['shards'][shard] = dict(completed_in_source=False, completed_originally=False,
            finished_in_source=sorted(finished), finished_originally=sorted(finished) if source_k == 0 else None,
            continued=details, timing_rows_read=len(timing), timing_rows_kept=len(kept),
            timing_rows_abandoned=len(abandoned), entries_sha256=sha(folder/'entries.json'))
    env.close()
    receipt['shards'] = {str(k): v for k, v in receipt['shards'].items()}
    atomic_json(cont/'prepare.json', receipt)
    cont.rename(target)
    return receipt


async def worker(root, shard, segment=1):
    import importlib.metadata
    import torch
    from benchmark_thinking_repaired import ObservableGenerator
    from polytopia_bot.learning.async_resume import resume_async_games
    from polytopia_bot.learning.interface import digest
    root = Path(root); folder = segment_dir(root, segment)/'shards'/str(shard)
    original = root/'shards'/str(shard)
    if sha(folder/'protocol.json') != sha(original/'protocol.json'):
        raise ValueError('shard protocol changed')
    p = json.loads((folder/'protocol.json').read_text())
    rules = json.loads(Path('/workspace/configs/ruleset-v1.json').read_text())
    for name, digest_ in {**rules['source_sha256'], **p['source_hashes']}.items():
        if sha(Path('/workspace')/name) != digest_:
            raise ValueError('source changed: '+name)
    entries = json.loads((folder/'entries.json').read_text())
    if not entries:
        return
    g = ObservableGenerator(p, True, folder/'on')
    receipt = dict(model_id=p['model_id'], revision=p['revision'], adapter=None, thinking=True,
        max_tokens=None, thinking_token_budget=None, max_model_len=g.engine.model_config.max_model_len,
        device=torch.cuda.get_device_name(), tokenizer_hash=digest(g.tokenizer.backend_tokenizer.to_str()),
        chat_template_hash=digest(g.tokenizer.chat_template),
        packages={k: importlib.metadata.version(k) for k in ['vllm', 'torch', 'transformers', 'jax']},
        serving=p.get('serving', dict(enforce_eager=True)),
        max_num_seqs=g.engine.vllm_config.scheduler_config.max_num_seqs,
        actual_cudagraph_capture_size=g.engine.vllm_config.compilation_config.max_cudagraph_capture_size,
        constrain_actions=p.get('constrain_actions', False), harness=p['harness'])
    atomic_json(folder/'on'/'runtime.json', receipt)
    try:
        if receipt != json.loads((original/'on'/'runtime.json').read_text()):
            raise ValueError('continuation runtime differs from the interrupted worker')
        if p.get('named_fields', False):
            from polytopia_bot.learning.thinking_readable import render_readable_decision as renderer
        else:
            from polytopia_bot.learning.thinking_debug import render_sparse_decision as renderer
        result = await resume_async_games(folder/'on'/'games', entries, g, thinking=True, prompt_renderer=renderer)
        atomic_json(folder/'on'/'report.json', result)
    finally:
        g.log.close(); g.engine.shutdown()


def finalize(root, interrupted_worker_seconds):
    """``interrupted_worker_seconds``: {f'{segment}:{shard}': seconds} for timed-out workers."""
    sys.path.insert(0, '/workspace/scripts')
    from merge_thinking_shards import merge
    from summarize_unrestricted_games import summarize
    root = Path(root); target = root/'final'
    if target.exists():
        raise ValueError('already finalized')
    final = root/'final.partial'
    if final.exists():
        shutil.rmtree(final)
    final.mkdir()
    shutil.copyfile(root/'protocol.json', final/'protocol.json')
    last = segments_present(root)
    segments = {}
    for shard in shard_ids(root):
        base = root/'shards'/str(shard); out = final/'shards'/str(shard)
        (out/'on'/'games').mkdir(parents=True)
        shutil.copyfile(base/'protocol.json', out/'protocol.json')
        (out/'off').symlink_to(os.path.relpath(base/'off', out))
        shutil.copyfile(base/'off-execution.json', out/'off-execution.json')
        shutil.copyfile(base/'on'/'runtime.json', out/'on'/'runtime.json')
        # Walk segments 0..last; each game's final segment is where it finished.
        games, timing, elapsed, worker_seconds, exit_code = [], '', 0., 0., 0
        pieces = {}            # job id -> [(segment, rows to keep or None for all)]
        used = []
        for k in range(0, last+1):
            scope = segment_scope(root, k, shard) if k == 0 or (segment_dir(root, k)/'shards'/str(shard)/'entries.json').exists() else {}
            if not scope:
                continue
            used.append(k)
            on = segment_on(root, k, shard)
            nxt = {}
            if k < last:
                nxt = segment_scope(root, k+1, shard)
                timing += (segment_dir(root, k+1)/'shards'/str(shard)/'kept-timing.jsonl').read_text()
            done = json.loads((on/'games/games.json').read_text()) if (on/'games/games.json').exists() else []
            if set(nxt) & {g['id'] for g in done} or set(nxt) - set(scope):
                raise ValueError(f'segment {k+1} scope inconsistent with segment {k}')
            for jid in scope:
                pieces.setdefault(jid, []).append((k, nxt[jid]['trace_rows'] if jid in nxt else None))
            for g in done:
                if k == 0:
                    games.append(g)
                else:
                    games.append(dict(g, continued_from_checkpoint=True, seconds=None, final_segment=k,
                                      final_segment_seconds=g['seconds'], resumed_at=scope[g['id']]['counters']))
            missing = set(scope) - set(nxt) - {g['id'] for g in done}
            if missing:
                raise ValueError(f'segment {k} lost games {sorted(missing)}')
            ex = segment_execution(root, k, shard)
            if (on/'report.json').exists() and ex.exists():
                elapsed += json.loads((on/'report.json').read_text())['elapsed_seconds']
                e = json.loads(ex.read_text()); worker_seconds += e['worker_seconds']; exit_code |= e['exit_code']
            else:
                key = f'{k}:{shard}'
                if key not in interrupted_worker_seconds:
                    raise ValueError(f'no receipt and no assumed seconds for interrupted segment {key}')
                elapsed += json.loads((on/'games'/'progress.json').read_text())['elapsed_seconds']
                worker_seconds += interrupted_worker_seconds[key]
                if k == last:
                    raise ValueError(f'last segment {k} of shard {shard} did not complete')
            if k == last:
                timing += (on/'generation-timing.jsonl').read_text()
        for jid, parts in pieces.items():
            if len(parts) == 1:
                k, _ = parts[0]
                (out/'on'/'games'/(jid+'.jsonl.gz')).symlink_to(
                    os.path.relpath(segment_on(root, k, shard)/'games'/(jid+'.jsonl.gz'), out/'on'/'games'))
                continue
            lines = []
            for k, keep in parts:
                rows = complete_lines(segment_on(root, k, shard)/'games'/(jid+'.jsonl.gz'), gz=True, raw=True)
                if keep is not None:
                    if len(rows) < keep:
                        raise ValueError(f'cannot stitch {jid}: segment {k} has {len(rows)} < {keep} rows')
                    rows = rows[:keep]
                elif not rows or 'game_result' not in json.loads(rows[-1]):
                    raise ValueError(f'cannot stitch {jid}: segment {k} has no game result')
                lines.extend(rows)
            with gzip.open(out/'on'/'games'/(jid+'.jsonl.gz'), 'wt', compresslevel=1) as f:
                f.write(''.join(lines))
        (out/'on'/'generation-timing.jsonl').write_text(timing)
        atomic_json(out/'on'/'report.json', dict(schema='async-gameplay-continued-v2', games=games,
            allocated_games=len(games), finished_games=len(games),
            complete=all(g['status'] in ('native', 'forfeit', 'truncated') for g in games),
            elapsed_seconds=elapsed, segments_used=used))
        atomic_json(out/'on'/'games'/'games.json', games)
        atomic_json(out/'on-execution.json', dict(mode='on', shard=shard, exit_code=exit_code,
            worker_seconds=worker_seconds, segments_used=used,
            interrupted_worker_seconds_assumed={k: v for k, v in interrupted_worker_seconds.items() if k.endswith(f':{shard}')}))
        segments[shard] = dict(segments_used=used, elapsed_seconds=elapsed, worker_seconds=worker_seconds,
                               continued=sorted(j for j, p in pieces.items() if len(p) > 1))
    merge(final, final=True)
    summarize(final)
    atomic_json(final/'continuation-segments.json', segments)
    final.rename(target)
    return segments


if __name__ == '__main__':
    command, root = sys.argv[1], sys.argv[2]
    if command == 'worker':
        asyncio.run(worker(root, int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1))
    else:
        raise SystemExit('prepare/finalize run inside the Modal app functions')
