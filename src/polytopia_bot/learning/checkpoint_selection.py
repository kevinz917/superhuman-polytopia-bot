"""Main-step selection on the complete reserved validation set only."""
from .selection import conservative_score


def select_checkpoint(reports, expected_jobs, expected_steps):
    if set(reports)!=set(expected_steps) or len(expected_steps)!=3:
        raise ValueError('exactly the three preregistered checkpoints are required')
    def identity(game):
        return tuple(game[key] for key in ('seed','board_hash','seat','opponent','strategy'))
    expected={identity(job) for job in expected_jobs}
    if len(expected)!=400 or len(expected_jobs)!=400:
        raise ValueError('200 validation maps with both seats required')
    pairs={}
    for seed,board,seat,opponent,strategy in expected:
        if opponent!='weak' or seat not in (1,2):raise ValueError('validation protocol mismatch')
        pairs.setdefault((seed,board),set()).add(seat)
    if len(pairs)!=200 or any(seats!={1,2} for seats in pairs.values()):
        raise ValueError('validation maps must have both seats')
    summaries={}
    for step in sorted(expected_steps):
        report=reports[step];rows=report['games']
        if not report['complete'] or len(rows)!=400 or report.get('unstarted_jobs'):
            raise ValueError('all 400 checkpoint games must complete')
        if {identity(g) for g in rows}!=expected:
            raise ValueError('checkpoint used a different map/seat/opponent allocation')
        if any(g['status'] not in ('native','truncated','forfeit') for g in rows):
            raise ValueError('incomplete or infrastructure-error games cannot select a checkpoint')
        if any(not 0<=g['first_attempt_valid']<=g['model_decisions'] for g in rows):
            raise ValueError('invalid decision counters')
        summaries[step]=dict(games=400,conservative_score=sum(conservative_score(g) for g in rows)/400,
            wins=sum(g['status']=='native' and g['reward']>0 for g in rows),
            native_completions=sum(g['status']=='native' for g in rows),
            truncations=sum(g['status']=='truncated' for g in rows),forfeits=sum(g['status']=='forfeit' for g in rows),
            first_attempt_errors=sum(g['model_decisions']-g['first_attempt_valid'] for g in rows),
            model_decisions=sum(g['model_decisions'] for g in rows))
    selected=min(expected_steps,key=lambda step:(-summaries[step]['conservative_score'],summaries[step]['first_attempt_errors'],step))
    return dict(selected_step=selected,summaries=summaries,
        rule='Highest conservative win-equivalent score; tie by fewer first-attempt errors, then earlier checkpoint. All 200 validation maps, both seats. No final-test outcomes.',
        limitation='Selection identifies a step; it does not certify final checkpoint acceptance.')
