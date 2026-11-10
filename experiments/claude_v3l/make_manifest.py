"""Build v3l manifests (qualification Q0-Q3 and the flagship stages) from one place.

    make_manifest.py q0 RUN_ID --faults JSON [--cap USD] [--seconds S] [--checkpoint-every N]   > manifest.json
    make_manifest.py q1 RUN_ID --arch ARCH --width W --blocks B --config CFG [--extra '...']   (8xH100 throughput)
    make_manifest.py flagship RUN_ID --stage N --spent-before USD --arch ... --config ... --campaign-seconds S

Recipe defaults follow session E's V3 runs (experiments/claude_v3/modal_v3.py: Tokamax kernels, HIGHEST matmul,
jax-mem-fraction .45, 4096 games x 4 sets per GPU) and session A's 8-GPU recipe (grad-accum 2 at 8 devices keeps the
global 262144-row batch and 16384-row minibatch; per-device minibatch 2048).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launcher as L  # noqa: E402

KERNEL_ENV = {'POLYTOPIA_TRANSFORMER_ATTENTION': 'tokamax-triton-bfloat16', 'POLYTOPIA_LAYER_NORM': 'tokamax-affine',
              'POLYTOPIA_TRANSFORMER_REMAT': '0', 'PC_CUDA_WARP_TICK': '1'}
MAP_TABLE = '/flagship/maps/table-classic-14-2p-200000-300000.npz'


def trainer_args(run_id, *, arch, width, blocks, config, seconds, devices, grad_accum, checkpoint_every,
                 snapshot_every=250, extra='', games=4096):
    a = ['--config', config, '--parent', '', '--architecture', arch, '--width', str(width), '--blocks', str(blocks),
         '--run-id', run_id, '--run-root', '/flagship/runs', '--seconds', str(int(seconds)), '--games', str(games),
         '--sets', '4', '--grad-accum', str(grad_accum), '--matmul-precision', 'HIGHEST',
         '--checkpoint-every', str(checkpoint_every), '--snapshot-every', str(snapshot_every),
         '--jax-mem-fraction', '.45', '--map-table', MAP_TABLE]
    if devices not in (1, 'per-process'):
        a += ['--devices', str(devices)]
    return a + extra.split()


def base(mode, run_id, worker, stage, cap, args, **kw):
    gpus = L.WORKERS[worker][1]
    m = dict(schema='v3l-manifest-1', mode=mode, run_id=run_id, stage=stage, stage_cap_usd=cap,
             campaign_spent_before_usd=0., usd_per_hour=round(L.RATES[gpus], 4), stop_reserve_usd=0.5 if mode != 'flagship' else 3.,
             startup_allowance_s=180, max_resumes_per_stage=3, nan_window_s=3600, nan_pause_count=3,
             illegal_pause_fraction=1e-4, illegal_min_decisions=10_000_000, illegal_abs_pause=1000,
             worker=worker, timeout_s=7200 if mode != 'flagship' else 86400, trainer_args=args, env=dict(KERNEL_ENV),
             faults=[], leak_check=True, ckpt_keep=3, poll_s=30, heartbeat_s=60, source_sha=L.git_head())
    m.update(kw)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=('q0', 'q1', 'flagship', 'control'))
    ap.add_argument('run_id')
    ap.add_argument('--faults', default='[]')
    ap.add_argument('--cap', type=float, default=3.)
    ap.add_argument('--seconds', type=float, default=3600)
    ap.add_argument('--checkpoint-every', type=int, default=5)
    ap.add_argument('--arch', default='transformer-v3-p2')
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--config', default='experiments/claude_v3/hero-1v1-classic-nopeace-warm150.json')
    ap.add_argument('--extra', default='')
    ap.add_argument('--stage', type=int, default=1)
    ap.add_argument('--spent-before', type=float, default=0.)
    ap.add_argument('--campaign-seconds', type=float, default=0.)
    ap.add_argument('--set', default='{}', help='JSON of manifest fields to override')
    ap.add_argument('--processes', type=int, default=8, help='8-GPU kinds: 8 = one trainer process per GPU (session A, '
                    'GIL-free, 3.3x the threaded layout), 1 = one process with 8 device threads')
    ap.add_argument('--games', type=int, default=4096, help='games per device')
    ap.add_argument('--gpus', type=int, default=8, choices=(8, 16), help='16 = 2 clustered 8xH100 nodes, one process per GPU')
    ap.add_argument('--lr-restart', default='', help='JSON for the stage-3 LR warm restart (flagship_harness.lr_restart_schedule)')
    ap.add_argument('--snapshot-every', type=int, default=250)
    ap.add_argument('--snapshot-at-usd', default='', help='JSON list of cumulative $ marks')
    a = ap.parse_args()
    if a.kind == 'q0':
        args = trainer_args(a.run_id, arch=a.arch, width=a.width, blocks=a.blocks, config=a.config, seconds=a.seconds,
                            devices=1, grad_accum=16, checkpoint_every=a.checkpoint_every, snapshot_every=1000000,
                            extra=a.extra)
        m = base('qualification', a.run_id, 'qual_worker_1', 1, a.cap, args, faults=json.loads(a.faults),
                 timeout_s=2700)
    elif a.kind == 'q1':
        args = trainer_args(a.run_id, arch=a.arch, width=a.width, blocks=a.blocks, config=a.config, seconds=a.seconds,
                            devices=('per-process' if a.processes == 8 or a.gpus == 16 else 8),
                            grad_accum=(1 if a.gpus == 16 else 2), games=a.games, checkpoint_every=a.checkpoint_every,
                            snapshot_every=a.snapshot_every, extra=a.extra)
        m = base('qualification', a.run_id, 'qual_worker_16' if a.gpus == 16 else 'qual_worker_8', 1, a.cap, args,
                 faults=json.loads(a.faults), **({'timeout_s': 5400} if a.gpus == 16 else {}))
    elif a.kind == 'control':
        # Jack ~08:55 PT: hero CNN 4x128 from scratch, ~$150, concurrent with flagship stages 1-2, compared at matched
        # cumulative dollars. Same LR schedule length as the flagship (--campaign-seconds) so both are at the constant
        # base LR at every compared dollar mark [L decision].
        args = trainer_args(a.run_id, arch='normalized-v2', width=128, blocks=4, config=a.config,
                            seconds=a.campaign_seconds, devices=('per-process' if a.processes == 8 else 8), grad_accum=2, games=a.games, checkpoint_every=a.checkpoint_every,
                            extra=a.extra)
        m = base('control', a.run_id, 'stage_worker', 1, 100., args, leak_check=False,
                 snapshot_at_usd=[25, 50, 100])
    else:
        bounds = {1: 100., 2: 300., 3: 1000.}
        args = trainer_args(a.run_id, arch=a.arch, width=a.width, blocks=a.blocks, config=a.config,
                            seconds=a.campaign_seconds, devices=('per-process' if a.processes == 8 or a.gpus == 16 else 8),
                            grad_accum=(1 if a.gpus == 16 else 2), games=a.games, checkpoint_every=a.checkpoint_every,
                            snapshot_every=a.snapshot_every, extra=a.extra)
        marks = json.loads(a.snapshot_at_usd) if a.snapshot_at_usd else [25, 50, 75, 100, 150, 200, 250, 300]
        m = base('flagship', a.run_id, 'stage_worker_16' if a.gpus == 16 else 'stage_worker', a.stage, bounds[a.stage],
                 args, campaign_spent_before_usd=a.spent_before, snapshot_at_usd=marks)
        if a.lr_restart:
            m['lr_restart'] = json.loads(a.lr_restart)
    if a.kind != 'q0' and a.gpus == 16:
        m['processes'], m['nodes'] = 16, 2
    elif a.kind != 'q0' and a.processes == 8:
        m['processes'] = 8
    m.update(json.loads(a.set))
    print(json.dumps(m, indent=1))


if __name__ == '__main__':
    main()
