"""Validate complete native Harbor results and render the single-benchmark scorecard."""
import argparse
from collections import defaultdict
import hashlib
import gzip
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.skirmish.common import canonical
from polytopia_bot.skirmish.continuous import CODING_SECONDS, SCHEMA


def trace_digest(path):
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(root):
    job = json.loads((root/'result.json').read_text())
    paths = sorted(root.glob('*/result.json'))
    if len(paths) != job['n_total_trials'] or job['stats']['n_errored_trials'] or job['stats']['n_running_trials']:
        raise RuntimeError('Harbor job is incomplete or has infrastructure errors')
    trials = []
    schedule = None
    for path in paths:
        result = json.loads(path.read_text())
        if result['exception_info'] is not None:
            raise RuntimeError(f'{path.parent.name}: Harbor trial failed')
        trial = path.parent
        author = json.loads((trial/'agent/authoring.json').read_text())
        protocol = json.loads((trial/'agent/protocol.json').read_text())
        grading = json.loads((trial/'verifier/summary.json').read_text())
        plan = json.loads((trial/'verifier/evaluation-plan.json').read_text())
        rows = json.loads((trial/'verifier/games.json').read_text())
        assert protocol['schema']==SCHEMA and protocol['coding_seconds']==CODING_SECONDS
        assert author['coding_budget_seconds']==CODING_SECONDS
        assert author['timing']['coding_seconds'] <= CODING_SECONDS
        assert len(rows)==len(plan['jobs'])==grading['rewards']['games']
        assert len({r['job_index'] for r in rows})==len(rows)
        for row in rows:
            planned = plan['jobs'][row['job_index']]
            assert row['job_sha256']==planned['job_sha256']
            assert hashlib.sha256(canonical({k:v for k,v in planned.items() if k!='job_sha256'}).encode()).hexdigest()==planned['job_sha256']
            assert row['audit']['passed'] and row['result']!='unresolved'
            if not row['audit'].get('virtual'):
                relative = Path(row['trace_relative']).relative_to('games')
                trace = trial/'verifier/traces'/relative
                assert trace_digest(trace)==row['trace_sha256'],'exported replay trace digest mismatch'
        current = [(j['opponent'],j['map_type'],j['seed'],j['seat'],j['evaluation']) for j in plan['jobs']]
        if schedule is None:schedule=current
        else:assert current==schedule,'models were not evaluated on identical schedules'
        capture = author['capture']
        if not capture.get('error'):
            source = (trial/'artifacts/submission/main.rs').read_bytes()
            assert hashlib.sha256(source).hexdigest()==capture['source_sha256']==grading['submission']['source_sha256']
        wasm = trial/'verifier/bot.wasm'
        if grading['submission']['returncode']==0:
            assert hashlib.sha256(wasm.read_bytes()).hexdigest()==grading['submission']['wasm_sha256']
        hero = [r for r in rows if r['opponent_label']=='hero-1v1']
        practices = list((trial/'artifacts/practice').glob('*/report.json'))
        trials.append(dict(model=result['config']['agent']['model_name'],trial=trial.name,
            score=grading['score'],games=len(rows),profile=grading['profile'],
            quality=grading['submission']['submission_quality'],compile_returncode=grading['submission']['returncode'],
            champion=dict(wins=sum(r['result']=='win' for r in hero),draws=sum(r['result']=='draw' for r in hero),losses=sum(r['result']=='loss' for r in hero)),
            practice_matches=len(practices),coding_budget_seconds=CODING_SECONDS,
            timing=author['timing'],usage=author['usage'],inference=protocol['inference'],
            replay_actions=grading['replay_actions'],rewards=grading['rewards']))
    output = dict(status='completed',schema=SCHEMA,execution='native Harbor; explicit execution/inference routes in each trial receipt',
        trials=trials,total_final_outcomes=sum(t['games'] for t in trials),
        replay_actions=sum(t['replay_actions'] for t in trials),
        caveat='Integration demo when profile=demo. One authored bot per model does not establish a ranking. Subscription cost is unavailable, not zero.')
    (root/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    text = ['# Polytopia Skirmish: continuous Harbor results','',
        'One continuous session per trial. Coding allowance: 40 active minutes, including on-demand practice and trace inspection. Models may submit early. No forced feedback cycles.','',
        output['caveat'],'',
        '| Model | Panel points /100 | Champion W–D–L | Practice matches | Active minutes | Final games | Submission |',
        '|---|---:|---:|---:|---:|---:|---|']
    for t in trials:
        h=t['champion']
        text.append(f"| {t['model']} | {t['score']:.2f} | {h['wins']}–{h['draws']}–{h['losses']} | {t['practice_matches']} | {t['timing']['coding_seconds']/60:.2f} | {t['games']} | {t['quality']} |")
    text += ['',f"All {output['total_final_outcomes']} scheduled final outcomes are present. Replay-verified actions: {output['replay_actions']:,}. All trials use identical held-out maps, opponents, seats and runtime settings. Captured source hashes and independently rebuilt Wasm hashes were verified.",'',
        'Full native Harbor configurations, conversations, usage/timing receipts, practice artifacts, compilation receipts, evaluation plans and replay traces remain in each trial directory. Primary score = 100 × (wins + half of draws) / scheduled matches.']
    (root/'summary.md').write_text('\n'.join(text)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    output=summarize(p.parse_args().root)
    print(json.dumps(output,indent=2))
