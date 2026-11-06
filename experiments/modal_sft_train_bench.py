"""Expert-SFT trainer benchmark (Modal only; reads expert-sft-v1/prepared, writes sft-train-bench/).

  modal run --env staging -m experiments.modal_sft_train_bench --action lengths
  modal run --env staging -m experiments.modal_sft_train_bench --action explore --gpu B200 --preset explore_b200
  modal run --env staging -m experiments.modal_sft_train_bench --action dist --gpu B200:4 --preset dist_b200 --steps 30

Results land in sft-train-bench/<tag>/result.json on the volume and in runs/sft-train-bench-2026-10-19/.
Never trains a deliverable adapter and never writes checkpoints.
"""
import json
import time
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name('polytopia-sft0-training', environment_name='staging')
app = modal.App('polytopia-sft-train-bench')
wheel = 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
# The expert trainer's image plus liger-kernel (fused RMSNorm/RoPE/SwiGLU), contemporaneous with transformers 4.57.1.
image = (modal.Image.debian_slim(python_version='3.12').pip_install('uv==0.9.25')
    .add_local_file(ROOT/'training/pyproject.toml', '/workspace/training/pyproject.toml', copy=True)
    .add_local_file(ROOT/'training/uv.lock', '/workspace/training/uv.lock', copy=True)
    .run_commands('UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple uv sync --project /workspace/training --locked --no-dev --extra simulation')
    .run_commands(f'uv pip install --python /workspace/training/.venv/bin/python --no-deps {wheel}',
                  'uv pip install --python /workspace/training/.venv/bin/python einops==0.8.1')
    .run_commands('uv pip install --python /workspace/training/.venv/bin/python --no-deps liger-kernel==0.6.3')
    .env({'PYTHONPATH': '/workspace/src', 'TOKENIZERS_PARALLELISM': 'false', 'HF_HOME': '/workspace/runs/hf-cache', 'JAX_PLATFORMS': 'cpu'})
    .add_local_dir(ROOT/'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_dir(ROOT/'scripts', '/workspace/scripts', ignore=['**/__pycache__/**']))
RUNS = Path('/workspace/runs')
PREPARED = 'expert-sft-v1/prepared'
OUT = 'sft-train-bench'
LOCAL = ROOT/'runs'/'sft-train-bench-2026-10-19'


def V(name, budget=8192, ckpt=False, attention='default', liger=False, **kw):
    return dict(name=name, token_budget=budget, checkpointing=ckpt, attention=attention, liger=liger, **kw)


PRESETS = {
    # First entry is the parity reference; liger entries last (the patch is not reversible in-process).
    'explore_b200': [V('qualified_8192'), V('b16384', 16384), V('b32768', 32768), V('b32768_alternating', 32768, 'alternating'),
                     V('cudnn_8192', attention='cudnn'), V('cudnn_16384', 16384, attention='cudnn'),
                     V('liger_8192', liger=True), V('liger_cudnn_8192', attention='cudnn', liger=True),
                     V('liger_cudnn_16384', 16384, attention='cudnn', liger=True),
                     V('liger_cudnn_32768', 32768, attention='cudnn', liger=True)],
    # Round 2: why cuDNN and liger failed parity (GQA vs layout; which liger kernel) and steady-state speed.
    'explore_b200_r2': [V('qualified_8192'), V('cudnn_rkv_8192', attention='cudnn', repeat_kv=True),
                        V('cudnn_gqa_contig_8192', attention='cudnn', contiguous=True),
                        V('cudnn_rkv_bucket256_8192', attention='cudnn', repeat_kv=True, bucket=256),
                        V('cudnn_rkv_bucket256_16384', 16384, attention='cudnn', repeat_kv=True, bucket=256),
                        V('liger_rope_8192', liger=dict(rope=True, rms_norm=False, swiglu=False)),
                        V('liger_rms_8192', liger=dict(rope=False, rms_norm=True, swiglu=False)),
                        V('liger_swiglu_8192', liger=dict(rope=False, rms_norm=False, swiglu=True))],
    # Ground truth on short examples: fp32 weights, no autocast, math attention is the reference (first entry).
    'truth_b200': [V('gold_fp32_math', ckpt=True, attention='math', precision='fp32'),
                   V('fp32_default', ckpt=True, precision='fp32'),
                   V('qualified_8192'), V('math_bf16_8192', ckpt=True, attention='math'),
                   V('cudnn_8192', attention='cudnn'), V('cudnn_rkv_8192', attention='cudnn', repeat_kv=True),
                   V('liger_rope_8192', liger=dict(rope=True, rms_norm=False, swiglu=False)),
                   V('liger_rms_8192', liger=dict(rope=False, rms_norm=True, swiglu=False)),
                   V('liger_swiglu_8192', liger=dict(rope=False, rms_norm=False, swiglu=True))],
    'dist_b200': [V('qualified_8192'), V('b16384', 16384),
                  V('liger_swiglu_16384', 16384, liger=dict(rope=False, rms_norm=False, swiglu=True)),
                  V('cudnn_b256_swiglu_16384', 16384, attention='cudnn', bucket=256, contiguous=True, prewarm=24576,
                    liger=dict(rope=False, rms_norm=False, swiglu=True))],
    'dist_h100': [V('adaptive10k_8192', ckpt='adaptive', adaptive_tokens=10240),
                  V('liger_swiglu_adaptive12k_8192', ckpt='adaptive', adaptive_tokens=12288,
                    liger=dict(rope=False, rms_norm=False, swiglu=True))],
    # Steady-state cuDNN (GQA, 256-token length buckets, plans prebuilt) + liger SwiGLU vs the qualified path on the
    # same batches from the same init: the loss trajectory against qualified 4xB200 vs 8xB200 is the noise floor.
    'dist_b200x8': [V('qualified_8192'),
                    V('cudnn_b256_swiglu_16384', 16384, attention='cudnn', bucket=256, contiguous=True, prewarm=24576,
                      liger=dict(rope=False, rms_norm=False, swiglu=True))],
    'dist_h100x4': [V('liger_swiglu_adaptive12k_8192', ckpt='adaptive', adaptive_tokens=12288,
                      liger=dict(rope=False, rms_norm=False, swiglu=True))],
    # Short-prompt proxy (rows of 2,000-4,600 tokens; compact prompt v2 mean 3,150, p90 4,598).
    'short_b200': [V('qualified_8192'), V('b16384', 16384), V('b32768_alternating', 32768, 'alternating'),
                   V('liger_swiglu_16384', 16384, liger=dict(rope=False, rms_norm=False, swiglu=True)),
                   V('cudnn_b256_swiglu_16384', 16384, attention='cudnn', bucket=256, contiguous=True, prewarm=4608,
                     liger=dict(rope=False, rms_norm=False, swiglu=True))],
    'short_h100': [V('qualified_8192'), V('b16384', 16384),
                   V('liger_swiglu_16384', 16384, liger=dict(rope=False, rms_norm=False, swiglu=True))],
    'explore_h100': [V('ckpt_8192', ckpt=True), V('qualified_8192'), V('alternating_8192', ckpt='alternating'),
                     V('adaptive10k_8192', ckpt='adaptive', adaptive_tokens=10240),
                     V('cudnn_adaptive10k_8192', ckpt='adaptive', adaptive_tokens=10240, attention='cudnn'),
                     V('cudnn_ckpt_16384', 16384, True, attention='cudnn'),
                     V('liger_cudnn_adaptive10k_8192', ckpt='adaptive', adaptive_tokens=10240, attention='cudnn', liger=True),
                     V('liger_cudnn_adaptive12k_8192', ckpt='adaptive', adaptive_tokens=12288, attention='cudnn', liger=True),
                     V('liger_cudnn_ckpt_16384', 16384, True, attention='cudnn', liger=True)],
}


@app.function(image=image, cpu=4, memory=32768, timeout=3600, retries=0, volumes={'/workspace/runs': volume})
def lengths():
    """Token lengths of every prepared record, the 8 longest, and prompt-prefix sharing statistics."""
    import random
    volume.reload()
    path = RUNS/PREPARED/'tokens.jsonl'
    rng = random.Random(17); keep = set(rng.sample(range(39626), 400))
    out, kept, supervised, first, common = [], {}, [], None, None
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line); ids = r['input_ids']; out.append(len(ids))
            supervised.append(sum(t != -100 for t in r['labels']))
            if first is None:
                first = ids; common = len(ids)
            else:
                n = min(common, len(ids)); j = 0
                while j < n and ids[j] == first[j]:
                    j += 1
                common = j
            if i in keep:
                kept[i] = ids
    def lcp(a, b):
        n = min(len(a), len(b)); j = 0
        while j < n and a[j] == b[j]:
            j += 1
        return j
    ks = sorted(kept); pairs = [rng.sample(ks, 2) for _ in range(2000)]
    pair_lcp = sorted(lcp(kept[a], kept[b]) for a, b in pairs)
    s = sorted(out)
    meta = dict(records=len(out), lengths=out, longest=sorted(range(len(out)), key=lambda i: -out[i])[:16],
                total_tokens=sum(out), mean=sum(out)/len(out), quantiles={q: s[min(len(s)-1, int(q*len(s)))] for q in
                (.01, .1, .25, .5, .75, .9, .95, .99, 1.0)}, supervised_mean=sum(supervised)/len(supervised),
                global_common_prefix=common, pair_lcp=dict(mean=sum(pair_lcp)/len(pair_lcp), median=pair_lcp[len(pair_lcp)//2],
                p90=pair_lcp[int(.9*len(pair_lcp))], max=pair_lcp[-1]),
                over={t: sum(x > t for x in out)/len(out) for t in (8192, 10240, 12288, 16384)})
    (RUNS/OUT).mkdir(exist_ok=True)
    (RUNS/OUT/'lengths.json').write_text(json.dumps(meta)); volume.commit()
    return {k: v for k, v in meta.items() if k != 'lengths'}


@app.function(image=image, gpu='B200', cpu=8, memory=65536, timeout=3*3600, retries=0, scaledown_window=2,
              volumes={'/workspace/runs': volume})
def bench(tag, mode, nproc, config, variants, steps, warmup, parity_only=False):
    import subprocess
    volume.reload()
    meta = json.loads((RUNS/OUT/'lengths.json').read_text())
    root = RUNS/OUT/tag; root.mkdir(parents=True, exist_ok=False)
    payload = dict(mode=mode, prepared=str(RUNS/PREPARED), config=config, seed=17, effective=config['effective_batch_size'],
                   stress_indices=meta['longest'][:8], variants=variants, steps=steps, warmup_steps=warmup,
                   parity_only=parity_only, **({'probe_quantiles': [.05, .2, .35, .5]} if tag.startswith('truth') else {}),
                   **({'length_range': [2000, 4600]} if tag.startswith('short') else {}))
    if mode == 'dist':
        payload['lengths'] = meta['lengths']
    (root/'input.json').write_text(json.dumps(payload))
    py = '/workspace/training/.venv/bin/python'
    cmd = ([py, '/workspace/scripts/sft_train_bench.py'] if mode == 'explore' else
           [py, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={nproc}', '/workspace/scripts/sft_train_bench.py'])
    started = time.monotonic()
    proc = subprocess.Popen(cmd + [str(root/'input.json'), str(root/'result.json')], cwd='/workspace')
    try:
        while proc.poll() is None:
            volume.commit(); time.sleep(20)
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=30)
        volume.commit()
    result = json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {}
    result.update(exit_code=proc.returncode, container_seconds=time.monotonic()-started, tag=tag)
    (root/'result.json').write_text(json.dumps(result, indent=2)); volume.commit()
    return result


@app.local_entrypoint()
def main(action: str = 'lengths', gpu: str = 'B200', preset: str = '', steps: int = 30, warmup: int = 2, tag: str = '',
         parity_only: bool = False):
    LOCAL.mkdir(parents=True, exist_ok=True)
    if action == 'lengths':
        meta = lengths.remote()
        (LOCAL/'lengths-summary.json').write_text(json.dumps(meta, indent=2)+'\n'); print(json.dumps(meta, indent=1))
        return
    config = json.loads((ROOT/'runs/expert-sft-v1/protocol.json').read_text())['training']['config']
    variants = PRESETS[preset]
    kind, _, count = gpu.partition(':'); nproc = int(count or 1)
    tag = tag or f'{action}-{kind.lower()}x{nproc}-{preset}-{int(time.time())}'
    fn = bench.with_options(gpu=gpu, cpu=8*nproc, memory=min(32768*nproc+32768, 344064))
    result = fn.remote(tag, action if action == 'explore' else 'dist', nproc, config, variants, steps, warmup, parity_only)
    (LOCAL/f'{tag}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('variants', 'configs')}))
    for e in result.get('variants', result.get('configs', [])):
        keep = ('name', 'status', 'stress_status', 'tokens_per_second', 'seconds', 'forward_seconds', 'backward_seconds',
                'seconds_per_step', 'peak_bytes', 'stress_peak_bytes', 'batch_peak_bytes', 'error')
        print({k: e[k] for k in keep if k in e}, e.get('parity_vs_qualified'), e.get('phases_mean_rank'))
