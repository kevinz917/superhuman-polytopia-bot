"""Package the completed efficiency continuation without promoting failed gates."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main(output,markdown,incomplete=False):
    evidence=[]
    def read(name):
        p=ROOT/name;evidence.append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        return json.loads(p.read_text())
    before=read('runs/sft0/repaired-smoke-result-v2/summary.json')
    after=read('runs/sft0/efficient-smoke-result-v1/summary.json')
    if not before['artifact_integrity_passed'] or not after['artifact_integrity_passed']:
        raise ValueError('both training integrity summaries must pass')
    old=read('runs/sft0/prepared-repaired-smoke-v2/manifest.json')
    new=read('runs/sft0/prepared-efficient-smoke-v1/manifest.json')
    for key in ('source_sha256','tokens_sha256','processed_tokens','supervised_tokens','tokenizer_hash','chat_template_hash'):
        if old[key]!=new[key]:raise ValueError('training data changed: '+key)
    if after['examples']!=before['examples'] or after['examples']!=10000:raise ValueError('same 10k examples required')
    differences={k:[old['config'].get(k),new['config'].get(k)] for k in old['config'] if old['config'].get(k)!=new['config'].get(k)}
    if set(differences)!={'microbatch_size'}:raise ValueError('unexpected training configuration change')
    if incomplete:
        report=read('runs/sft0/efficient-games-v1/combined/games/report.json')
        audit=read('runs/sft0/efficient-game-prefix-audit-v1.json')
        protocol=read('runs/sft0/efficient-games-v1/protocol.json')
        if (report['complete'] or not audit['passed'] or audit['evaluation_complete']
            or audit['report_sha256']!=hashlib.sha256((ROOT/'runs/sft0/efficient-games-v1/combined/games/report.json').read_bytes()).hexdigest()):
            raise ValueError('matching incomplete prefix audit required')
        identity=lambda g:tuple(g[k] for k in ('id','seed','board_hash','seat','candidate','strategy','opponent'))
        if len(report['games'])!=50 or report['unstarted_jobs'] or {identity(g) for g in report['games']}!={identity(g) for g in protocol['jobs']}:
            raise ValueError('all allocated game prefixes required')
        statuses=dict(Counter(g['status'] for g in report['games']))
        upper=(statuses.get('native',0)+statuses.get('budget_incomplete',0))/50
        if upper>=.8:raise ValueError('completion failure cannot yet be established')
        game=dict(games=50,statuses=statuses,native_completion=statuses.get('native',0)/50,
                  native_completion_upper_bound=upper,evaluation_complete=False,
                  wins=sum(g['status']=='native' and g['reward']>0 for g in report['games']),
                  losses=sum(g['status']=='native' and g['reward']<0 for g in report['games']))
        gate=dict(learning_diagnostic_passed=False,positions=read('runs/sft0/efficient-positions-v1/summary.json'),
                  games=dict(models=dict(smoke=game),completion_passed=False,execution_passed=False))
    else:
        gate=read('runs/sft0/efficient-smoke-diagnostic-v1.json')
    runtime=read('runs/sft0/efficient-game-runtime-v1.json')
    rollout=read('runs/sft0/efficiency-rollout-benchmark-v1/execution.json')
    benchmark=read('runs/sft0/efficiency-training-benchmark-v6/execution.json')
    allocation=read('runs/sft0/efficient-smoke-result-v1/allocation.json')
    adapter=ROOT/'runs/sft0/efficient-smoke-result-v1/training-run/model/adapter-final'
    for name,expected in after['adapter_files'].items():
        if hashlib.sha256((adapter/name).read_bytes()).hexdigest()!=expected:raise ValueError('saved adapter changed')
    game=gate['games']['models']['smoke'];positions=dict(gate['positions'])
    position_coordinator=read('runs/sft0/efficient-positions-v1/coordinator.json')
    if 'estimated_compute_usd' in positions:
        positions['estimated_compute_usd']+=position_coordinator['elapsed_seconds']*.00001532
        positions['cost_note']='Includes cloud coordinator; supersedes GPU-only cost in the historical position summary.'
    ratio=before['train_seconds']/after['train_seconds']
    result=dict(schema='efficient-sft-checkpoint-result-v1',checkpoint_trained=True,
        smoke_learning_diagnostic_passed=gate['learning_diagnostic_passed'],accepted_gameplay_checkpoint=False,
        main_training_run=False,final_test_run=False,evaluation_complete=not incomplete,
        full_game_replay_passed=not incomplete,recorded_transition_replay_passed=True,adapter_directory=str(adapter),
        base_model=allocation['config']['model_id'],base_revision=allocation['config']['revision'],
        adapter_files=after['adapter_files'],same_training_examples=True,configuration_difference=differences,
        training=dict(examples=after['examples'],processed_tokens=after['processed_tokens'],microbatch=allocation['config']['microbatch_size'],effective_batch=allocation['config']['effective_batch_size'],before_seconds=before['train_seconds'],after_seconds=after['train_seconds'],measured_full_run_speedup=ratio,benchmark_speedup=benchmark['benchmark']['measured_speedup']),
        rollout=dict(prefix_pilot_speedup=rollout['comparison']['speedup'],exact_prefix_parity=True,full_runtime=runtime,comparison_limit='Full-game policies and batch compositions differ from the old serial run; do not attribute their total-time ratio solely to pipelining.'),
        positions=positions,games=game,completion_gate_passed=gate['games']['completion_passed'],execution_gate_passed=gate['games']['execution_passed'],
        evidence=evidence,limitations=['Smoke checkpoint only; no main or final-test acceptance.','Validation outcomes are diagnostics, not held-out final-test performance.','Published-rate compute estimates exclude startup/idle, builds, storage and egress.'])
    p=ROOT/output
    with p.open('x') as f:json.dump(result,f,indent=2)
    status='passed' if gate['learning_diagnostic_passed'] else 'failed'
    replay_text=('All recorded transitions and public observations passed independent prefix replay. One game hit the fixed runtime ceiling; full-game evaluation and its execution gate remain incomplete.' if incomplete else 'Independent replay and execution checks passed.')
    body=f'''# First SFT checkpoint: efficiency continuation

The same-data 10k smoke checkpoint is trained and its saved adapter passed integrity checks. The smoke learning/gameplay diagnostic **{status}**. This is an experimental smoke adapter, not an accepted main gameplay checkpoint. No main training or final-test evaluation ran.

Adapter directory: `{adapter}`. Load it as a PEFT adapter on `{result['base_model']}`, pinned revision `{result['base_revision']}`. The frozen prompt, tokenizer, context and inference settings are recorded in `runs/sft0/efficient-smoke-result-v1/allocation.json`.

## Runtime changes

- Rollout CPU lanes overlap with one persistent model worker per GPU. The fixed-prefix pilot was **{rollout['comparison']['speedup']:.2f}× faster**, with identical prompts, outputs and transitions. Full-game evaluation uses a fresh common base/SFT pipeline allocation and cloud coordinators.
- Training uses microbatch **{allocation['config']['microbatch_size']}**, effective batch **{allocation['config']['effective_batch_size']}**, response-only vocabulary projection and length sorting within each unchanged effective batch. GPU loss/gradient checks, longest-input stress and fit/reload checks passed.
- Full 10k optimizer time: **{before['train_seconds']/60:.2f} → {after['train_seconds']/60:.2f} minutes**, or **{ratio:.3f}× throughput**. The small benchmark measured {benchmark['benchmark']['measured_speedup']:.3f}×. These are throughput results, not evidence of better sample efficiency.
- In the new game run, inference averaged **{runtime['mean_batch_size']:.2f} requests per batch**. The model-call path, including tokenization and decoding, occupied **{runtime['model_call_path_fraction']:.1%}** of worker time; singleton batches used **{runtime['singleton_model_time_fraction']:.1%}** of model-call time. This is not a GPU hardware-utilization measurement. Long prompts and many decisions per game remain the major runtime costs. Different policies and batch compositions prevent treating the old/new full-game elapsed-time ratio as a pipeline speedup.

## Checkpoint results

- Saved positions: **{positions['smoke_agreement']:.2%}** exact teacher agreement versus **{positions['base_agreement']:.2%}** for the base model; **{positions['smoke_first_attempt_valid']:.2%}** first-attempt validity.
- Live games: **{game['wins']} wins**, **{game['losses']} native losses**, status counts `{game['statuses']}` across all 50 allocated games. Native completion was **{game['native_completion']:.2%}**, against the required 80%.
- {replay_text} Horizon truncations and budget-incomplete games are not native draws.
- The bounded game evaluation cost approximately **${runtime['estimated_compute_usd']:.2f}** at recorded published compute rates. This excludes startup/idle, builds, storage and egress. No research GPU allocations remain active.

The next research issue is closed-loop gameplay quality, especially the large number of episodes reaching the horizon. Any further data or teacher change needs a new versioned research protocol; these validation traces must remain diagnostic. Runtime improvements do not waive the failed completion gate.

Authoritative result: `runs/sft0/efficient-checkpoint-result-v1.json`. Detailed implementation and failed benchmark attempts: `docs/sft-efficiency-implementation.md`. Cost evidence: `runs/sft0/cost-ledger.json`.
'''
    with (ROOT/markdown).open('x') as f:f.write(body)
    print(json.dumps({k:result[k] for k in ['checkpoint_trained','smoke_learning_diagnostic_passed','accepted_gameplay_checkpoint','adapter_directory','training']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='runs/sft0/efficient-checkpoint-result-v1.json');p.add_argument('--markdown',default='docs/sft-checkpoint-result.md')
    p.add_argument('--incomplete',action='store_true',help='Package a proven negative result from audited incomplete prefixes; no acceptance.')
    a=p.parse_args();main(a.output,a.markdown,a.incomplete)
