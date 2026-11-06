#!/usr/bin/env python3
"""Verify downloaded training artifacts and summarize the recorded run."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summarize(source, output):
    root = ROOT / source
    execution = json.loads((root/'execution.json').read_text())
    allocation = json.loads((root/'allocation.json').read_text())
    run = execution.get('run') or {}
    model = root/'training-run/model'
    state = json.loads((model/'trainer_state.json').read_text())
    prepared = json.loads((ROOT/allocation['prepared']/'manifest.json').read_text())
    expected_steps = math.ceil(prepared['audit']['unique_records']/allocation['config']['effective_batch_size'])*allocation['config']['epochs']
    checks = dict(
        completed=execution.get('exit_code') == 0 and run.get('status') == 'completed',
        registered_stage_and_seed=run.get('stage') == allocation['stage'] and run.get('seed') == allocation['seed'],
        same_config=run.get('config') == allocation['config'] == prepared['config'],
        same_prepared_manifest=run.get('data_manifest') == prepared,
        all_planned_steps=state['global_step'] == expected_steps,
        all_planned_epochs=math.isclose(state['epoch'], allocation['config']['epochs'], abs_tol=1e-9),
        finite_recorded_metrics=all(math.isfinite(v) for row in [run['metrics'], *state['log_history']]
            for v in row.values() if isinstance(v, (int, float))),
        downloaded_adapter_hashes=bool(run.get('adapter_files')) and all(
            sha(model/'adapter-final'/name) == expected for name, expected in run['adapter_files'].items()),
        downloaded_archive_hash=sha(root/execution['archive']) == execution['archive_sha256'])
    # The frozen trainer records before/after parameter fingerprints only for
    # the fit diagnostic. Do not invent those measurements for smoke/main.
    from safetensors import safe_open
    import torch
    torch.set_num_threads(2)
    tensor_count = 0
    finite = True
    only_lora = True
    nonzero_b = False
    with safe_open(model/'adapter-final/adapter_model.safetensors', framework='pt', device='cpu') as tensors:
        for name in tensors.keys():
            tensor = tensors.get_tensor(name)
            tensor_count += 1
            only_lora &= '.lora_' in name
            finite &= bool(torch.isfinite(tensor).all())
            if '.lora_B.' in name:
                nonzero_b |= bool(torch.count_nonzero(tensor))
    checks.update(saved_tensors_only_lora=only_lora and tensor_count > 0,
                  saved_adapter_tensors_finite=finite,
                  nonzero_saved_lora_B=nonzero_b,
                  gate_receipts_unchanged=all(sha(ROOT/name) == expected
                      for name, expected in allocation.get('gate_receipts', {}).items()))
    if allocation['stage'] == 'fit':
        reload = json.loads((model/'fit-reload.json').read_text())
        checks['reload_identical'] = execution.get('reload_exit_code') == 0 and reload.get('responses_identical_after_reload') is True
        checks['base_unchanged'] = run['frozen_weights_before'] == run['frozen_weights_after']
        checks['adapter_changed'] = run['adapter_weights_before'] != run['adapter_weights_after']
    if not all(checks.values()):
        raise ValueError('Training artifact verification failed: '+str(checks))
    rates = {'A100-80GB': .00078192, 'H100': .00118492, 'B200': .00182392}
    seconds = execution.get('occupied_seconds_with_archive', execution['elapsed_seconds'])
    result = dict(schema='verified-sft-training-summary-v1', artifact_integrity_passed=True, checks=checks,
        stage=run['stage'], seed=run['seed'], hardware=run['hardware'], global_step=state['global_step'],
        examples=prepared['audit']['unique_records'], processed_tokens=allocation['planned_processed_tokens'],
        trainable_parameters=run['trainable_parameters'], train_loss=run['metrics']['train_loss'],
        train_seconds=run['metrics']['train_runtime'], function_seconds=seconds,
        timing_includes_archive='occupied_seconds_with_archive' in execution,
        processed_tokens_per_train_second=allocation['planned_processed_tokens']/run['metrics']['train_runtime'],
        max_gpu_memory_bytes=run['max_gpu_memory_bytes'], adapter_files=run['adapter_files'],
        saved_adapter_tensors=tensor_count,
        parameter_update_evidence='Before/after parameter fingerprints are recorded only for fit. Smoke/main use the frozen trainer LoRA-only trainability guard, completed optimization, and verified finite saved LoRA tensors with nonzero B weights.',
        estimated_compute_usd=seconds*rates[allocation['gpu']],
        evidence=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in
            (root/'execution.json', root/'allocation.json', model/'trainer_state.json', ROOT/allocation['prepared']/'manifest.json')],
        summary_script_sha256=sha(Path(__file__)),
        limitation='Artifact/training integrity only, not policy acceptance. Finite checks cover recorded metrics. Published-rate compute estimate excludes startup/idle, builds, storage and egress; it is not an invoice.')
    with (ROOT/output).open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence','adapter_files')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    summarize(a.source, a.output)
