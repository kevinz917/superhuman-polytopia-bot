"""Inference-only paired evaluation; scores never enter the decision prompt."""
import gzip
import json
import time
from pathlib import Path

from .data import observation_from_dict
from .decision_harness import render_decision
from .interface import canonical, digest, parse_response, ResponseError


def score_prediction(position, selected):
    scores = position['scores']
    best = max(r['score'] for r in scores)
    top = [r['command'] for r in scores if r['score'] == best]
    chosen = [r['score'] for r in scores if r['command'] == selected]
    if position['teacher_command'] not in top:
        raise ValueError('teacher target is not optimal under frozen scores')
    if selected is not None and len(chosen) != 1:
        raise ValueError('selected command is not uniquely scored')
    return dict(exact=selected == position['teacher_command'], tie_aware=selected in top,
                movement_required=all(c['kind'] == 1 for c in top),
                selected_movement=selected is not None and selected['kind'] == 1,
                score_gap=best-chosen[0] if chosen else None,
                top_source=selected is not None and selected['kind'] == 1 and any(
                    c['kind'] == 1 and c['source'] == selected['source'] for c in top),
                top_target=selected is not None and selected['kind'] == 1 and any(
                    c['kind'] == 1 and c['target'] == selected['target'] for c in top))


def evaluate_pairs(root, positions, generator, versions, *, batch_size=32, seconds_limit=2100):
    """Alternate arm order by batch; same positions, no retries or fallbacks."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    if len({p['position_id'] for p in positions}) != len(positions):
        raise ValueError('duplicate position IDs')
    started = time.monotonic()
    rows, batches = [], []
    with gzip.open(root/'predictions.jsonl.gz', 'xt', compresslevel=1) as log:
        for start in range(0, len(positions), batch_size):
            if time.monotonic()-started > seconds_limit:
                break
            batch = positions[start:start+batch_size]
            observations = [observation_from_dict(p['observation']) for p in batch]
            for version in versions if (start//batch_size) % 2 == 0 else list(reversed(versions)):
                prompts = [render_decision(o, version) for o in observations]
                for p, prompt in zip(batch, prompts):
                    if digest(prompt.messages) != p['harness_digests'][version]:
                        raise ValueError('prompt changed after preflight')
                t = time.monotonic()
                outputs = generator.batch([p.messages for p in prompts])
                elapsed = time.monotonic()-t
                batches.append(dict(version=version, start=start, count=len(batch), seconds=elapsed))
                if len(outputs) != len(batch):
                    raise ValueError('generation count mismatch')
                for p, prompt, output in zip(batch, prompts, outputs):
                    selected, error = None, None
                    try:
                        selected = dict(parse_response(output.text, prompt, current_version=prompt.state_version).command)
                    except ResponseError as exc:
                        error = str(exc)
                    row = dict(position_id=p['position_id'], board_hash=p['board_hash'], phase=p['phase'],
                        family=p['teacher_command']['kind'], version=version, selected_command=selected,
                        valid=selected is not None, error=error, text=output.text,
                        prompt_digest=digest(prompt.messages), input_tokens=len(output.prompt_token_ids),
                        output_tokens=len(output.response_token_ids), batch_start=start,
                        batch_seconds=elapsed, batch_size=len(batch),
                        output_at_limit=len(output.response_token_ids) >= generator.config['max_new_tokens'])
                    row.update(score_prediction(p, selected))
                    log.write(canonical(row)+'\n')
                    rows.append(row)
                log.flush()
            print(canonical(dict(positions_complete=min(start+batch_size,len(positions)), allocated=len(positions))), flush=True)
    report = dict(complete=len(rows) == len(positions)*len(versions), allocated=len(positions),
                  versions=versions, elapsed_seconds=time.monotonic()-started, batches=batches, results=rows)
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
