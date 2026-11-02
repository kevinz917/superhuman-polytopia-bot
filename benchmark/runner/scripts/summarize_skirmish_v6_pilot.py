"""Compact, source-backed interpretation of a completed local-v6 pilot."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.skirmish.cohort import score_rows
from polytopia_bot.skirmish.v6 import final_stage


def summarize(root):
    report=json.loads((root/'report.json').read_text());plan=json.loads((root/'evaluation-plan.json').read_text())
    allocation=json.loads((root/'allocation.json').read_text())
    model_ids={m['label']:m['model'] for m in allocation['manifest']['models']}
    rows=json.loads((root/'games.json').read_text());origin=json.loads((root/'authoring-origin.json').read_text())
    assert len(rows)==len(plan['jobs']) and len({r['job_index'] for r in rows})==len(rows)
    for r in rows:
        assert r['job_sha256']==plan['jobs'][r['job_index']]['job_sha256']
        assert r['audit']['passed']
    finals=[r for r in rows if r['phase']=='heldout' and r['stage']==final_stage(r['track'])]
    scores=defaultdict(dict);quality={}
    for r in finals:
        quality[r['model_label'],r['track']]=r['submission_quality']
    for track in ['independent','segmented-control','feedback']:
        for model in sorted({r['model_label'] for r in finals}):
            part=[r for r in finals if r['model_label']==model and r['track']==track]
            scores[model][track]=dict(**score_rows(part),quality=quality[model,track],
                runtime_faults=sum(r['status']=='forfeit' for r in part))
    hero=[r for r in finals if r['opponent_label']=='hero-1v1']
    hero_outcomes=dict(Counter(r['result'] for r in hero))
    caps={}
    for family in ['drylands','lakes','continents']:
        part=[r for r in finals if r['map_type']==family]
        caps[family]=dict(games=len(part),cap_draws=sum(r['status']=='draw' for r in part),
            cap_rate=sum(r['status']=='draw' for r in part)/len(part))
    sensitive=json.loads((root/'generated-cap-games.json').read_text())
    groups=defaultdict(list)
    for r in sensitive:groups[r['track'],r['evaluation']['max_rounds'],r['model_label']].append(r)
    cap_scores={f'{track}|{h}|{m}':score_rows(part) for (track,h,m),part in groups.items()}
    stability={}
    for track in ['independent','segmented-control','feedback']:
        table={}
        for horizon in [100,150,200,300]:
            part=[(m,score_rows(rs)['score']) for (t,h,m),rs in groups.items() if t==track and h==horizon]
            table[str(horizon)]=sorted(part,key=lambda x:x[1] if x[1] is not None else -1,reverse=True)
        stability[track]=table
    diagnostic=[]
    diagnostic_plan=json.loads((root/'diagnostic-plan.json').read_text())
    for name in ['cap-calibration-games.json','transfer-games.json','tactical-games.json','generated-cap-games.json']:
        ds=json.loads((root/name).read_text())
        key={'cap-calibration-games.json':'calibration','transfer-games.json':'transfer',
             'tactical-games.json':'tactical','generated-cap-games.json':'sensitivity'}[name]
        jobs=diagnostic_plan[key]
        assert len(ds)==len(jobs) and len({r['job_index'] for r in ds})==len(ds)
        for r in ds:
            assert r['job_sha256']==jobs[r['job_index']]['job_sha256']
            assert r['audit']['passed']
        diagnostic+=ds
    all_rows=rows+diagnostic
    transfer=defaultdict(list);tactics=defaultdict(list);horizon_blocks=defaultdict(dict)
    for r in diagnostic:
        if r['phase']=='transfer':transfer[r['model_label'],r['track']].append(r)
        if r['phase']=='tactical':tactics[r['model_label'],r['track']].append(r)
    for r in sensitive:
        horizon_blocks[r['model_label'],r['track'],r['opponent_label'],r['seed'],r['seat']][r['evaluation']['max_rounds']]=r
    assert all(set(block)=={100,150,200,300} for block in horizon_blocks.values())
    horizon_changes={f'{a}_vs_{b}':dict(blocks=len(horizon_blocks),
        changed_outcomes=sum(rs[a]['result']!=rs[b]['result'] for rs in horizon_blocks.values()),
        identical_complete_action_chains=sum(rs[a]['action_chain_sha256']==rs[b]['action_chain_sha256'] for rs in horizon_blocks.values()),
        native_chain_mismatches=sum(rs[a]['status']=='native' and rs[b]['status']=='native' and rs[a]['action_chain_sha256']!=rs[b]['action_chain_sha256'] for rs in horizon_blocks.values()),
        caps_at_first=sum(rs[a]['status']=='draw' for rs in horizon_blocks.values()),
        caps_at_second=sum(rs[b]['status']=='draw' for rs in horizon_blocks.values()))
        for a,b in [(100,300),(150,300),(200,300)]}
    transfer_scores={'|'.join(k):score_rows(rs) for k,rs in transfer.items()}
    tactical_scores={'|'.join(k):dict(passed=sum(r['tactical_pass'] for r in rs),checks=len(rs)) for k,rs in tactics.items()}
    counts=dict(core_scheduled=len(rows),diagnostic_scheduled=len(diagnostic),
        actual_episodes=sum(not r['audit'].get('virtual',False) for r in all_rows),
        full_games=sum(not r['audit'].get('virtual',False) and r['phase']!='tactical' for r in all_rows),
        tactical_checks=sum(r['phase']=='tactical' for r in all_rows),
        virtual_compile_losses=sum(r['audit'].get('virtual',False) for r in all_rows),
        replay_actions=sum(r['audit']['actions'] for r in all_rows),
        unresolved=sum(r['result']=='unresolved' for r in all_rows),
        independent_rebuilds=len(list((root/'rebuilds').glob('**/submission.json'))))
    out=dict(status=report['status'],pilot=True,evaluation_revision=report['evaluation_revision'],
        authoring_builds=len(origin['runs']),model_ids=model_ids,new_model_calls_during_grading=0,
        original_authoring_caveats=[
            'Original protocol-check helper used legacy one-trillion fuel; actual feedback and final evaluation used four trillion. A scoped future-authoring fix does not change these frozen bots.',
            'Original feedback reports named host archive paths; public traces were available under /feedback/round-N/00.jsonl etc. Future reports explicitly name these sandbox paths.'],
        scores=dict(scores),counts=counts,strongest_reference_outcomes=hero_outcomes,transfer_scores=transfer_scores,tactical_scores=tactical_scores,horizon_changes=horizon_changes,final_core_caps_by_family=caps,cap_scores=cap_scores,cap_rankings_by_track=stability,
        intervals='not estimable with one independently authored build and one hidden layout per family',
        provenance='Original fixed feedback preserved; every frozen submission/stage scored under the corrected evaluation-only panel. No authoring repair or best-of selection.')
    (root/'pilot-summary.json').write_text(json.dumps(out,indent=2)+'\n')
    text=['# Four-model CLI-proxy pilot','','Local VM/Docker execution. One authored build/model/track, up to ten coding minutes. These are panel match points, not expert win rates. No defensible model ranking or confidence interval follows from this sample.','',out['provenance'],'',
        '| Model | Independent | Segmented, no feedback | Feedback |','|---|---:|---:|---:|']
    for model,v in scores.items():text.append('| '+model_ids.get(model,model)+' | '+' | '.join(f"{v[t]['score']:.2f}" if v[t]['score'] is not None else 'unresolved' for t in ['independent','segmented-control','feedback'])+' |')
    text+=['',f"Strongest-reference outcomes across all final builds: {hero_outcomes}. Panel points measure broader opposition; they are not hero win rates.",'','## Integrity','','```json',json.dumps(counts,indent=2),'```','','## Final-core cap draws','','| Family | Cap draws | Games |','|---|---:|---:|']
    for family,v in caps.items():text.append(f"| {family} | {v['cap_draws']} | {v['games']} |")
    text+=['','## Horizon sensitivity','','Separate diagnostic games against siege, naval-50 and hero, on reserved calibration maps. Results do not replace the fixed 200-round primary score.','']
    for track,table in stability.items():
        text+=['### '+track,'']
        for h,ranking in table.items():text.append(f'- {h} rounds: '+', '.join(f'{m} {s:.2f}' if s is not None else f'{m} unresolved' for m,s in ranking))
        text.append('')
    text+=['## Transfer and tactical diagnostics','','Transfer points against scripted siege across 11/14/18 boards and pangea/archipelago/highlands. Tactical probes test only three declared one-decision situations. These are separate from primary scores.','','| Model / track | Transfer points /100 | Tactical passes |','|---|---:|---:|']
    for k,ts in sorted(transfer_scores.items()):
        tc=tactical_scores[k];text.append(f"| {k.replace('|',' / ')} | {ts['score']:.2f} | {tc['passed']}/{tc['checks']} |")
    text+=['','Matched diagnostic outcome changes across horizons (same frozen bot, opponent, map and seat):','']
    for k,c in horizon_changes.items():text.append(f"- {k}: {c['changed_outcomes']}/{c['blocks']} outcomes changed; cap draws {c['caps_at_first']} → {c['caps_at_second']}.")
    text+=['','## Reliability','']
    for model,v in scores.items():
        for track,s in v.items():
            if s['quality']!='custom_submission' or s['runtime_faults']:text.append(f"- {model}/{track}: {s['quality']}; {s['runtime_faults']} runtime forfeits in {s['games']} final-panel outcomes.")
    text+=['','## Original authoring caveats','']+['- '+c for c in out['original_authoring_caveats']]
    text+=['','Raw plans, all outcomes, source rebuilds, authoring origin and the complete report remain beside this summary.']
    (root/'pilot-summary.md').write_text('\n'.join(text)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    r=summarize(a.root);print(json.dumps(dict(status=r['status'],scores=r['scores'],counts=r['counts']),indent=2))
