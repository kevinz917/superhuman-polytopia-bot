"""Descriptive movement imitation metrics; never changes labels or acceptance gates."""
from collections import Counter


def score_choice(scores, target, selected):
    """Accept all globally highest teacher scores, not only the chosen tie-break."""
    if target['kind'] != 1 or not scores:
        raise ValueError('movement teacher target and complete scores required')
    best = max(row['score'] for row in scores)
    top = [row['command'] for row in scores if row['score'] == best]
    if target not in top:
        raise ValueError('original target is not teacher-optimal')
    selected_rows = [row for row in scores if row['command'] == selected]
    if selected is not None and len(selected_rows) != 1:
        raise ValueError('selected command not uniquely legal')
    movement = selected is not None and selected['kind'] == 1
    return dict(exact=selected == target, tie_aware=selected in top,
                selected_movement=movement, selected_kind=None if selected is None else selected['kind'],
                selected_score=None if not selected_rows else selected_rows[0]['score'],
                score_gap=None if not selected_rows else best-selected_rows[0]['score'],
                top_score=best, top_count=len(top), legal_count=len(scores),
                movement_required_for_top_score=all(c['kind']==1 for c in top),
                legal_movement_count=sum(row['command']['kind']==1 for row in scores),
                top_movement_count=sum(command['kind']==1 for command in top),
                movement_uses_top_source=movement and any(c['kind']==1 and c['source']==selected['source'] for c in top),
                movement_reaches_top_target=movement and any(c['kind']==1 and c['target']==selected['target'] for c in top))


def summarize(rows):
    count=len(rows)
    if not count:
        return dict(count=0)
    moves=[r for r in rows if r['selected_movement']]
    return dict(count=count, maps=len({r['board_hash'] for r in rows}),
                exact_matches=sum(r['exact'] for r in rows),
                tie_aware_matches=sum(r['tie_aware'] for r in rows),
                exact=sum(r['exact'] for r in rows)/count,
                tie_aware=sum(r['tie_aware'] for r in rows)/count,
                selected_movement=len(moves)/count,
                tie_aware_given_movement=sum(r['tie_aware'] for r in moves)/len(moves) if moves else None,
                top_source_given_movement=sum(r['movement_uses_top_source'] for r in moves)/len(moves) if moves else None,
                top_target_given_movement=sum(r['movement_reaches_top_target'] for r in moves)/len(moves) if moves else None,
                first_attempt_valid=sum(r['first_attempt_valid'] for r in rows)/count,
                tied_positions=sum(r['top_count']>1 for r in rows),
                mean_top_count=sum(r['top_count'] for r in rows)/count,
                movement_required_positions=sum(r['movement_required_for_top_score'] for r in rows),
                mean_legal_count=sum(r['legal_count'] for r in rows)/count,
                mean_input_tokens=sum(r['input_tokens'] for r in rows)/count,
                uniform_legal_top_score_baseline=sum(r['top_count']/r['legal_count'] for r in rows)/count,
                selected_families=dict(Counter(str(r['selected_kind']) for r in rows)),
                error_counts=dict(nonmovement=sum(not r['selected_movement'] and not r['tie_aware'] for r in rows),
                                  suboptimal_movement=sum(r['selected_movement'] and not r['tie_aware'] for r in rows)))
