"""Live throughput trials on the Curio head (run from the Mac): a learner plus any number of GPU-served actor nodes,
with explicit GPU/core pinning, for a few minutes; reports steady-state averages, then tears everything down.

    python experiments/curio/trial.py NAME --minutes 8 \
        --learner-gpus 4,5,6,7 --learner-cores 60-119,180-239 \
        --actor 0:0-29,120-149:96 --actor 1:...:96 ...   (gpu:cores:processes[:games]; gpu 'cpu' = CPU inference)
        [--set coded_views=true ...] [--env K=V ...] [--unset-env NCCL_P2P_DISABLE ...] [--parent P.npz]

Each update row of the learner log after --warmup updates is averaged: seconds per update, train_seconds,
received samples/s, assembler phases, stale drops, KL. The result JSON goes to stdout (RESULT line) and to
<root>/trials/results/NAME.json on the head. A hang (no update for 3 min after the first) ends the trial early with
a thread dump in the learner log.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hosts_from, load_plan, paths, LEARNER_ENV, ACTOR_ENV  # noqa: E402

KEYS = ('seconds', 'train_seconds', 'assemble_seconds', 'received_samples_per_second', 'asm_total_seconds',
        'asm_concat_seconds', 'asm_views_seconds', 'asm_value_seconds', 'receiver_decompress_seconds',
        'stale_dropped', 'mean_lag', 'approx_kl', 'ready_wait_seconds')


def parse_value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--minutes', type=float, default=8)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--learner-gpus', default='4,5,6,7')
    ap.add_argument('--learner-cores', default='60-119,180-239')
    ap.add_argument('--actor', action='append', default=[], help='gpu:cores:processes[:games]')
    ap.add_argument('--worker-actor', action='append', default=[], help='worker:cores:processes[:games] (CPU inference)')
    ap.add_argument('--serve-options-json', default='{"trunk_dtype": "bfloat16", "server_cores": 2}')
    ap.add_argument('--config', default='hero-1v1-scratch.json')
    ap.add_argument('--parent', default='/data/polytopia/ckpt/scratch-init/hero-1v1-init-4x128-obs2-s1.npz')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--unset-env', action='append', default=[])
    args = ap.parse_args()
    plan = load_plan()
    hosts = hosts_from(plan)
    head = hosts['head']
    p = paths(head.root)
    run = f'trial-{args.name}'
    cfg = json.loads((Path(__file__).resolve().parent / 'configs' / args.config).read_text())
    for kv in args.set:
        k, _, v = kv.partition('=')
        cfg[k] = parse_value(v)
    cfg['_trial'] = args.name
    cfg_path = f'{p["root"]}/trials/{args.name}.json'
    head.write(cfg_path, json.dumps(cfg, indent=1))
    env = dict(LEARNER_ENV, CUDA_VISIBLE_DEVICES=args.learner_gpus, PYTHONPATH=p['src'], JAX_COMPILATION_CACHE_DIR=p['cache'])
    for k in args.unset_env:
        env.pop(k, None)
    for kv in args.env:
        k, _, v = kv.partition('=')
        env[k] = v
    head.run(f'rm -rf {p["runs"]}/{run} {p["logs"]}/learner-{run}.log {p["logs"]}/trial-actor-*', check=False)
    lcmd = (f'{p["python"]} -X faulthandler experiments/curio/node_learner.py run --root {p["root"]} --run {run} '
            f'--config {cfg_path} --parent {args.parent} --gpus {args.learner_gpus} --cores {args.learner_cores} --hours 3')
    print('learner:', head.start('trial-learner', lcmd, env, p['code'], f'{p["logs"]}/learner-{run}.log',
                                 f'{p["pids"]}/trial-learner.pid'), flush=True)
    for _ in range(60):
        if head.exists(f'{p["runs"]}/{run}/protocol.json'):
            break
        time.sleep(5)
    serve = json.loads(args.serve_options_json) if args.serve_options_json else None
    for i, spec in enumerate(args.actor):
        parts = spec.split(':')
        gpu, cores, procs = parts[0], parts[1], int(parts[2])
        games = int(parts[3]) if len(parts) > 3 else 24
        inference = 'cpu' if gpu == 'cpu' else 'cuda'
        acmd = (f'{p["python"]} experiments/curio/node_actor.py --host {plan["head"]["data_ip"]} --port {plan["port"]} '
                f'--index {900 + i} --processes {procs} --games {games} --cores {cores} --inference {inference} '
                f'--report {p["logs"]}/trial-actor-{i}.json --until-unix {int(time.time() + args.minutes * 60 + 600)}'
                + (f" --serve-options-json '{json.dumps(serve)}'" if serve and inference == 'cuda' else ''))
        aenv = dict(ACTOR_ENV, PYTHONPATH=p['src'], JAX_COMPILATION_CACHE_DIR=p['cache'])
        if inference == 'cuda':
            aenv['CUDA_VISIBLE_DEVICES'] = gpu
        print(f'actor {i}:', head.start(f'trial-actor-{i}', acmd, aenv, p['code'], f'{p["logs"]}/trial-actor-{i}.log',
                                         f'{p["pids"]}/trial-actor-{i}.pid'), flush=True)
    for j, spec in enumerate(args.worker_actor):
        name, cores, procs, *rest = spec.split(':')
        games = int(rest[0]) if rest else 48
        w = hosts[name]
        wp = paths(w.root)
        w.run(f'rm -f {wp["logs"]}/trial-wactor-*', check=False)
        acmd = (f'{wp["python"]} experiments/curio/node_actor.py --host {plan["head"]["data_ip"]} --port {plan["port"]} '
                f'--index {950 + j} --processes {procs} --games {games} --cores {cores} --inference cpu '
                f'--report {wp["logs"]}/trial-wactor-{j}.json --until-unix {int(time.time() + args.minutes * 60 + 600)}')
        aenv = dict(ACTOR_ENV, PYTHONPATH=wp['src'], JAX_COMPILATION_CACHE_DIR=wp['cache'])
        print(f'worker actor {name}:', w.start(f'trial-wactor-{j}', acmd, aenv, wp['code'], f'{wp["logs"]}/trial-wactor-{j}.log',
                                               f'{wp["pids"]}/trial-wactor-{j}.pid'), flush=True)
    started = time.time()
    rows, last_seen, last_count, err = [], None, 0, '0'
    while time.time() - started < args.minutes * 60:
        time.sleep(30)
        text = head.out(f'grep -a \'^{{"stage": "update"\' {p["logs"]}/learner-{run}.log', default='') or ''
        rows = []
        for line in text.splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        err = head.out(f'grep -a -c learner_error {p["logs"]}/learner-{run}.log; true', default='0')
        if len(rows) != last_count:
            last_count, last_seen = len(rows), time.time()
        util = head.out('nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr "\\n" " "')
        r = rows[-1] if rows else {}
        print(time.strftime('%H:%M:%S'), f'updates={len(rows)} sps={round(r.get("received_samples_per_second", 0))} '
              f'step={round(r.get("seconds", 0), 2)} train={round(r.get("train_seconds", 0), 2)} '
              f'asm={round(r.get("asm_total_seconds", 0), 2)} err={err} util={util}', flush=True)
        if err not in ('0', None):
            break
        if rows and last_seen and time.time() - last_seen > 180:
            print('HANG: no update for 3 min', flush=True)
            break
    steady = rows[args.warmup:]
    mean = {k: round(sum(float(r.get(k, 0) or 0) for r in steady) / len(steady), 4) for k in KEYS} if steady else {}
    if len(steady) > 1:
        span = steady[-1]['elapsed'] - steady[0]['elapsed']
        mean['trained_samples_per_second'] = round((len(steady) - 1) * cfg['batch'] / span) if span else 0
    actors = []
    for i in range(len(args.actor)):
        try:
            a = json.loads(head.read(f'{p["logs"]}/trial-actor-{i}.json') or '{}')
            actors.append({k: a.get(k) for k in ('connected', 'compiled', 'sent_samples', 'restarts', 'errors')})
        except ValueError:
            actors.append(None)
    hung = bool(rows) and last_seen is not None and time.time() - last_seen > 180
    pid = head.read(f'{p["pids"]}/trial-learner.pid')
    if (hung or err not in ('0', None)) and pid:
        head.run(f'kill -USR1 {pid}; sleep 3; true', check=False)
    head.stop(f'{p["pids"]}/trial-learner.pid', grace=3)
    for i in range(len(args.actor)):
        head.stop(f'{p["pids"]}/trial-actor-{i}.pid', grace=3)
    head.run(f'pkill -9 -f "^{p["python"]} "; true', check=False)
    for j, spec in enumerate(args.worker_actor):
        w = hosts[spec.split(':')[0]]
        wp = paths(w.root)
        try:
            a = json.loads(w.read(f'{wp["logs"]}/trial-wactor-{j}.json') or '{}')
            actors.append({k: a.get(k) for k in ('connected', 'compiled', 'sent_samples', 'restarts', 'errors')})
        except ValueError:
            actors.append(None)
        w.stop(f'{wp["pids"]}/trial-wactor-{j}.pid', grace=3)
        w.run(f'pkill -9 -f "^{wp["python"]} "; true', check=False)
    head.run(f'rm -rf {p["runs"]}/{run}', check=False)
    result = dict(trial=args.name, updates=len(rows), steady_updates=len(steady), hung=hung, errors=err, mean=mean,
                  actors=actors, set=args.set, env=args.env, unset_env=args.unset_env, learner_gpus=args.learner_gpus,
                  learner_cores=args.learner_cores, actor_specs=args.actor,
                  worker_actor_specs=args.worker_actor)
    head.write(f'{p["root"]}/trials/results/{args.name}.json', json.dumps(result, indent=1))
    print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
