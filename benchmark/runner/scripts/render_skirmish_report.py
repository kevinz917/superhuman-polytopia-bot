"""Render a compact, source-backed pilot report exported from Modal (no bot artifacts)."""
import argparse
import json
from pathlib import Path


def result(v):
    return f"{v['wins']}–{v['losses']}–{v['unresolved']}"


def label(name,runs):
    if name=='hero-1v1':return 'Expert'
    model=runs.get(name,{}).get('model',name)
    if model=='deepseek/deepseek-v4-flash-0731':return 'DeepSeek Flash'
    if model=='openai/gpt-6-luna':return 'GPT-6 Luna high'
    return model


def render(data):
    runs=data['runs']
    if not runs or any(v['status'] not in ('completed','failed_submission','failed_harness','failed') for v in runs.values()):
        raise ValueError('all authoring runs must reach terminal status before publishing results')
    lines=['# mini-SWE-agent Polytopia pilot — 2026-10-30','',
        'One independent authoring attempt per model and track. These are integration-pilot results, '
        'not a statistically reliable ranking of the underlying models. All calls use OpenRouter.', '',
        '- Harness: mini-SWE-agent 2.4.6, upstream DefaultAgent/OpenRouterModel with a Modal bash adapter.',
        '- Models: `deepseek/deepseek-v4-flash-0731` and `openai/gpt-6-luna`; both use high reasoning and 32,768 maximum output tokens/call.',
        '- One-shot: one 1,200-second coding window, no expert-game feedback.',
        '- Revision: initial 300 seconds plus three 300-second coding revisions. After each of the first three stages, '
        'six full public traces are supplied (drylands/lakes/continents, both seats). Evaluation is outside coding time.',
        '- Offline isolated sandbox: two CPU cores, 2 GiB, Rust 1.90, std/serde_json 1.0.145, complete Markdown rule book.',
        '- Fixed expert: released `hero-1v1`, checkpoint SHA `e6f4e1977e7f9cdcfbbe17358d87e49d638d82150496a6d4099e0cd389b0934f`.',
        '- Existing Rust engine: `polytopia-2.17.3`, Bardur 1v1, 14×14. Guest: 256 MiB, one trillion fuel/game, '
        'two-second decisions plus five-second startup. Referee: 180 seconds, 100 rounds, 10,000 actions. '
        'Forfeits and unresolved games are reported separately.', '',
        '## Isolated final checks', '',
        'Each authoring run received fresh hidden maps after its source was frozen. The shared-map round robin below '
        'is the comparable ranking; these six-game checks use independent seeds.', '',
        '| Model | Track | Status | Expert W–L–unresolved | Forfeits | Coding seconds | Saved calls | Saved API USD |',
        '|---|---|---|---|---|---|---|---|']
    for name,v in runs.items():
        short=label(name,runs)
        final=v['evaluations'].get('final')
        score=result(final) if final and 'wins' in final else '—'
        forfeits=str(final['forfeits']) if final and 'forfeits' in final else '—'
        seconds=sum(s['authoring_seconds'] for s in v['stages'])
        calls=v.get('model_calls')
        cost=v.get('api_cost_usd')
        lines.append(f"| {short} | {v['track']} | {v['status']} | {score} | {forfeits} | {seconds:.1f} | {calls if calls is not None else '—'} | {cost:.6f} |" if cost is not None else
                     f"| {short} | {v['track']} | {v['status']} | {score} | {forfeits} | {seconds:.1f} | — | — |")
    lines+=['', 'API amounts cover saved completed calls only; a request killed at a coding deadline can still be billed. '
            'These amounts exclude Modal compute, short smoke tests and the earlier integration cohort.', '']
    unchanged=[name for name,v in runs.items() if v.get('final_source_sha256')==
               '6a0e6e4310e060a2a5953e4b0aee46cf83c9bd32ddfabd6ccc71652cc0cc5c61']
    if unchanged:
        lines += ['**Unchanged starter:** '+', '.join(f'`{name}`' for name in unchanged)+
                  '. These are valid executables but were not custom bots produced in that trial. '
                  'The score is retained; do not describe these as successfully written new bots.', '']
    lines += ['## Shared-map round robins', '']
    lines+=['These wins/losses include every opponent in the roster. Direct model-versus-model results:', '',
            '| Track | DeepSeek wins | Luna wins | Unresolved | Forfeits |', '|---|---|---|---|---|']
    for ladder in data['ladders'].values():
        if not ladder or ladder.get('status')!='completed':continue
        pair=next((v for k,v in ladder['pairs'].items() if k.startswith('mini-') and '__' in k),None)
        if pair:lines.append(f"| {ladder['track']} | {pair['wins']} | {pair['losses']} | {pair['unresolved']} | {pair['forfeits']} |")
    lines.append('')
    for name,ladder in data['ladders'].items():
        if not ladder or ladder.get('status')!='completed':
            lines.append(f'- `{name}`: not completed.');continue
        lines += [f"### {ladder['track']}",'',
                  f"{ladder['games']} games; {ladder['resolved']} resolved, {ladder['unresolved']} unresolved. "
                  'Three new shared map seeds, one per map type, both seats for every pairing. '
                  'Elo is relative to this roster and anchored to the expert at zero.', '',
                  '| Rank | Entrant | Elo | Wins | Losses | Unresolved | Forfeits |','|---|---|---|---|---|---|---|']
        for i,row in enumerate(ladder['ladder'],1):
            display=label(row['entrant'],runs)
            elo='—' if row['elo'] is None else f"{row['elo']:.0f}"
            lines.append(f"| {i} | {display} | {elo} | {row['wins']} | {row['losses']} | {row['unresolved']} | {row['forfeits']} |")
        lines.append('')
        if ladder.get('audit'):
            audit=ladder['audit']
            lines += [f"Replay audit: passed={audit['passed']}; {audit['games']} games, {audit['actions']} actions.",'']
    lines+=['## Revision feedback and frozen sources','','| Model | Stage | Coding seconds | Compile | Source SHA-256 | Expert feedback W–L–unresolved |',
            '|---|---|---|---|---|---|']
    for name,v in runs.items():
        if v['track']!='feedback-3r':continue
        for s in v['stages']:
            feedback=v['evaluations'].get(f"feedback-{s['stage']}")
            score=result(feedback) if feedback and 'wins' in feedback else 'Final stage; no further feedback' if s['stage']==3 else 'No game batch'
            display=label(name,runs)
            lines.append(f"| {display} | {s['stage']} | {s['authoring_seconds']:.1f} | {s['returncode']==0} | `{s['source_sha256']}` | {score} |")
    lines+=['', 'An unchanged starter is a valid executable, but is explicitly recognizable by source SHA '
            '`6a0e6e4310e060a2a5953e4b0aee46cf83c9bd32ddfabd6ccc71652cc0cc5c61`. '
            'A failed current final source is a failed submission; the evaluator never substitutes an older bot.', '',
            'Source-hash differences describe changes between frozen stages, not every intermediate shell edit.', '',
            '## Verification and artifacts', '',
            '- Targeted Modal checks: 33 passed, one skipped (the rule-book generator is operator tooling outside the runtime image). '
            'The generated Markdown was separately checked against its source files locally.',
            '- Live sandbox preflight passed: unprivileged offline compilation, protected trusted-build directory, '
            'background-job cutoff, pristine frozen-source compilation. Both API models also passed short pipeline smoke tests.',
            '- A prior integration cohort exposed a missing tool-result receipt after an early submission. '
            'The affected trial was excluded as an infrastructure failure, not scored as a model loss. The corrected v2 adapter passed '
            'a live OpenRouter early-submit/resume probe. Further calibration exposed a cumulative fuel ceiling and output truncation. '
            'This report uses a fresh v3 cohort with the calibrated runtime and explicit larger response allowance.',
            '- Full sources, Wasm, model trajectories, feedback JSONL, trusted private action chains, source archive, '
            'package versions and receipts remain in Modal staging volume `polytopia-skirmish`.',
            f"- Run prefix: `{data['prefix']}`; roots: `mini-runs/<run-id>` and `mini-tournaments/<tournament-id>`.",
            '- The adjacent JSON is a compact read-only export. It contains no generated bot source, checkpoint or full trace.', '']
    for name,v in runs.items():
        audit=v.get('audit')
        if audit: lines.append(f"- `{name}`: replay audit passed={audit['passed']}, {audit['games']} games, {audit['actions']} actions.")
    lines+=['', 'Limitations: one authoring run per cell and one map per family are too small for robust model rankings '
            'or causal claims about feedback. The two tracks share a maximum 20 coding minutes, but models can submit '
            'early and differ in API latency. High reasoning, document-reading cost and context-management behavior are '
            'part of this fixed harness. These neutral prompts do not test psychological pressure.', '']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input');parser.add_argument('output');args=parser.parse_args()
    Path(args.output).write_text(render(json.loads(Path(args.input).read_text())))
