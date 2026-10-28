"""Print the latest learner status rows of distributed runs (reads the Modal registry Dict).

Run with Modal's Python: `$(head -1 $(which modal) | cut -c3-) scripts/distributed_status.py RUN_ID...`
"""
import json
import sys
import modal

registry = modal.Dict.from_name('polytopia-distributed-rl-registry', environment_name='staging')
for run_id in [a for a in sys.argv[1:] if not a.startswith('--')]:
    entry, row = registry.get(run_id), registry.get(f'{run_id}/status')
    print(json.dumps(dict(run_id=run_id, entry=entry)))
    total = 0.
    for block in range(0, 1000, 100):
        for index in range(block, block + 64):
            a = registry.get(f'{run_id}/actor/{index}')
            if a is None:
                break
            errors = a.pop('errors', [])
            compile_s = a.get('first_compile_seconds')   # the Rust engine's compiled event carries the engine name
            rate = a['steps'] / max(a['elapsed'] - (compile_s if isinstance(compile_s, (int, float)) else 0), 1)
            if a.get('alive'):
                total += rate
            print('actor', index, f"alive={a.get('alive')} steps/s~{rate:.0f}", json.dumps(
                {k: a[k] for k in ('connected', 'compiled', 'sent_samples', 'steps', 'elapsed') if k in a}))
            for e in errors[-1:]:
                print(e)
    print('approx live steps/s', round(total))
    if row:
        actors = row.pop('actors', {})
        keep = ('update', 'elapsed', 'seconds', 'assemble_seconds', 'train_seconds', 'received_samples_per_second',
                'connections', 'inbox', 'stale_dropped', 'mean_lag', 'approx_kl', 'clip_fraction', 'entropy',
                'value_loss', 'unit_kl', 'unit_entropy', 'epochs_run', 'ready_wait_seconds', 'device_batch_seconds',
                'messages', 'asm_total_seconds', 'asm_wait_seconds', 'asm_stream_seconds', 'asm_views_seconds',
                'asm_value_seconds', 'asm_gae_seconds', 'receiver_decompress_seconds', 'receiver_unpickle_seconds')
        print(json.dumps({k: row[k] for k in keep if k in row}))
        print(json.dumps(actors))

if '--ladder' in sys.argv or len(sys.argv) == 1:
    for row in registry.get('ladder/history', []):
        print(json.dumps(row))
