"""Render a multi-model cohort from compact Modal receipts; no bot sources or raw reasoning."""
import argparse
import json
from pathlib import Path

REFERENCE='hero-1v1'
STARTER='6a0e6e4310e060a2a5953e4b0aee46cf83c9bd32ddfabd6ccc71652cc0cc5c61'


def score(values):
    return f"{values.get('wins',0)}–{values.get('losses',0)}–{values.get('unresolved',0)}"


def versus(ladder, entrant, expert=False):
    counts=dict(wins=0,losses=0,unresolved=0)
    for key,pair in ladder.get('pairs',{}).items():
        a,b=key.split('__')
        if entrant not in (a,b):continue
        opponent=b if entrant==a else a
        if (opponent==REFERENCE)!=expert:continue
        counts['wins']+=pair['wins'] if entrant==a else pair['losses']
        counts['losses']+=pair['losses'] if entrant==a else pair['wins']
        counts['unresolved']+=pair['unresolved']
    return counts


def render(data):
    replacements=data.get('replacements',{})
    runs={name:r for name,r in data['runs'].items() if name not in replacements}
    ladders=data['ladders']
    terminal={'completed','failed_submission','failed_harness','failed'}
    if not runs or any(r['status'] not in terminal for r in runs.values()):
        raise ValueError('cohort is not terminal')
    label=lambda name: 'Expert' if name==REFERENCE else runs[name]['model'].split('/',1)[1]
    lines=['# Polytopia coding benchmark — ten-model cohort, 2026-10-30','',
        f"Cohort: `{data['prefix']}`. One independent authoring attempt per model per track. "
        'Results rank these frozen submissions; they are preliminary evidence about models, not a robust model ordering.', '',
        '## Fixed setup','',
        '- mini-SWE-agent 2.4.6, upstream DefaultAgent and OpenRouterModel; all model APIs through OpenRouter.',
        '- Protocol v4: same real engine CLI, nine public test maps, compiler, static annotated protocol example and persistent WASI checker for every model.',
        '- No supplied public opponent or automatic practice-match runner. Local self-written experiments consume the coding clock.',
        '- High reasoning; 32,768 maximum output tokens per call. Specific provider endpoints pinned; automatic provider fallback disabled.',
        '- One-shot: up to 20 continuous coding minutes, no expert feedback.',
        '- Revision: initial five minutes plus three five-minute revisions; six full expert traces after each of the first three stages. Evaluation is outside coding time.',
        '- Isolated offline sandbox: two CPU cores, 2 GiB; Rust 1.90.0, std + serde_json 1.0.145.',
        '- Bardur 1v1, 14×14; drylands/lakes/continents. Match runtime: 256 MiB, one trillion fuel/game, two seconds per decision plus five-second startup, 180 seconds/game, 100 rounds, 10,000 actions.',
        '- The released hero-1v1 expert is a fixed reference. Each tournament uses fresh shared hidden maps, both seats per map and pairing.', '',
        '## Authoring and independent expert checks','',
        'These isolated expert checks use different seeds for different runs. Use the common-map tournaments below for comparable performance.', '',
        '| Model | Track | Status | Expert W–L–U | Forfeits | Coding minutes | API calls | Saved API USD |',
        '|---|---|---|---|---|---|---|---|']
    for name,r in sorted(runs.items(),key=lambda kv:(kv[1]['track'],kv[1]['model'])):
        final=r.get('evaluations',{}).get('final',{})
        cost=r.get('api_cost_usd')
        calls=r.get('model_calls')
        if cost is None:cost=sum(t.get('instance_cost',0) for t in r.get('trajectories',[]))
        if calls is None:calls=sum(t.get('api_calls',0) for t in r.get('trajectories',[]))
        times=[s.get('authoring_seconds') for s in r['stages']]
        elapsed=f"{sum(times)/60:.2f}" if all(t is not None for t in times) else 'not recorded*'
        lines.append(f"| {label(name)} | {r['track']} | {r['status']} | {score(final) if 'wins' in final else '—'} | "
            f"{final.get('forfeits','—')} | {elapsed} | {calls} | ${cost:.4f} |")
    lines+=['','Costs cover saved completed calls, excluding smoke tests, Modal compute, and potentially billed requests interrupted at deadlines.','']
    if replacements:
        lines+=['## Infrastructure exclusions','',
            'These attempts were invalidated for infrastructure reasons and excluded from rankings. '
            'Interrupted authoring attempts restart with the original full budget. '
            'The source-cap recovery replays the original recorded source writes without another model call.','']
        for original,replacement in replacements.items():
            lines.append(f"- `{original}` → `{replacement['run_id']}`: {replacement['reason']}")
        lines+=['','The zero-cost compatibility setting affects accounting validation, not the prompt, model request parameters, tools, time budget, or bot runtime. '
            '*Exact coding time was not persisted for the source-cap recovery. Its recorded countdown and submission show it stayed within the original 20-minute window.','']
    if any(r.get('isolated_diagnostic_status')=='interrupted_and_excluded' for r in runs.values()):
        lines+=['The restored Sol one-shot submission has no standalone expert-check score: that extra diagnostic batch was interrupted and excluded. '
            'Its ranking uses the complete shared-map tournament, including six expert games under the same schedule as every entrant.','']
    for ident,ladder in sorted(ladders.items()):
        if ladder.get('status')!='completed':
            lines += [f'## {ident}', '', f"Status: {ladder.get('status')}", ''];continue
        lines += [f"## {ladder['track']} common-map ladder",'',
            f"{ladder['games']} games: {ladder['resolved']} resolved, {ladder['unresolved']} unresolved. "
            'Elo is fitted within this track and roster, anchored to the expert at zero. It is not comparable across tracks or older cohorts.', '',
            '| Rank | Entrant | Elo | Peer W–L–U | Expert W–L–U | Forfeits |',
            '|---|---|---|---|---|---|']
        for index,row in enumerate(ladder['ladder'],1):
            entrant=row['entrant'];elo='—' if row['elo'] is None else f"{row['elo']:.0f}"
            peer=score(versus(ladder,entrant)) if entrant!=REFERENCE else '—'
            expert=score(versus(ladder,entrant,True)) if entrant!=REFERENCE else '—'
            lines.append(f"| {index} | {label(entrant)} | {elo} | {peer} | {expert} | {row['forfeits']} |")
        lines+=['', '### Complete pairwise results', '',
                'Each cell is row wins / column wins / unresolved, using the same maps in both seats.', '']
        names=[r['entrant'] for r in ladder['ladder']]
        matrix={}
        for key,p in ladder['pairs'].items():
            a,b=key.split('__');matrix[a,b]=f"{p['wins']}/{p['losses']}/{p['unresolved']}"
            matrix[b,a]=f"{p['losses']}/{p['wins']}/{p['unresolved']}"
        lines += ['| Entrant | '+' | '.join(label(n) for n in names)+' |',
                  '|---|'+'---|'*len(names)]
        for n in names:lines.append('| '+label(n)+' | '+' | '.join('—' if n==m else matrix.get((n,m),'missing') for m in names)+' |')
        audit=ladder.get('audit') or {}
        lines+=['',f"Replay audit: {audit.get('passed','not recorded')}; {audit.get('actions','—')} actions.",'']
    lines+=['## Revisions and submission reliability','',
        '| Model | Stage | Coding seconds | Compile | Feedback W–L–U | Source SHA-256 |',
        '|---|---|---|---|---|---|']
    for name,r in sorted(runs.items()):
        if r['track']!='feedback-3r':continue
        for s in r['stages']:
            feedback=r.get('evaluations',{}).get(f"feedback-{s['stage']}",{})
            lines.append(f"| {label(name)} | {s['stage']} | {s['authoring_seconds']:.1f} | {s['returncode']==0} | "
                f"{score(feedback) if 'wins' in feedback else '—'} | `{s['source_sha256']}` |")
    unchanged=[name for name,r in runs.items() if r.get('final_source_sha256')==STARTER]
    lines+=['', 'Unchanged starters: '+(', '.join(f'`{n}`' for n in unchanged) if unchanged else 'none')+'.', '',
        'Failed submissions remain failures; no old source is substituted. They do not receive an invented Elo. '
        'Guest faults are forfeits. Engine/referee caps remain unresolved and are shown separately; unresolved games are excluded from Elo.', '',
        '## Evidence and limitations', '',
        '- Sources, Wasm, full model trajectories, feedback, private referee chains and source archives stay on Modal staging volume polytopia-skirmish.',
        f"- Run roots: `mini-runs/{data['prefix']}-*`; tournament roots: `mini-tournaments/{data['prefix']}-*`.",
        '- The adjacent JSON contains compact receipts, ratings and diagnostics, not generated source or raw reasoning.',
        '- Ten model compatibility/resume probes passed. The CLI/referee suite passed 38 tests; live offline sandbox preflight passed.',
        '- One authoring attempt and one map per family limit generalization. Three feedback stages use different practice maps; changing practice scores do not establish revision improvement.',
        '- Time limits include API latency and local tests. Different reasoning settings do not imply equal reasoning compute. Neutral prompts do not measure psychological pressure.', '']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
    Path(a.output).write_text(render(json.loads(Path(a.input).read_text())))
