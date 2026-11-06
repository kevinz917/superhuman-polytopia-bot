"""Actor node on a Curio box: distributed.run_actor_container against the learner's address, no Modal.

    python node_actor.py --host 10.0.1.4 --port 7070 --index 3 --processes 96 --games 24 --cores 32-79
                         --inference cuda|cpu [--serve-options-json '{"trunk_dtype":"bfloat16","server_cores":2}']
                         [--until-unix T] [--stop-file PATH] [--report PATH]

One node = one "container": `processes` pinned engine processes over the given cores (round-robin when there are more
processes than cores, as with a GPU inference server), one WeightRelay connection to the learner, and with
--inference cuda an in-node GPU inference server on the GPU named by CUDA_VISIBLE_DEVICES (set by the launcher).
The node loops: when the container returns (learner gone, all processes exited) it waits 30 s and starts again, so
a learner restart is picked up without the keeper touching the actors. It exits when --until-unix passes, the
--stop-file appears, or SIGTERM arrives.

The parent process must not import JAX: actor processes are spawned and re-import this file.
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True)
    ap.add_argument('--port', type=int, default=7070)
    ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--processes', type=int, required=True)
    ap.add_argument('--games', type=int, default=24)
    ap.add_argument('--cores', default='')
    ap.add_argument('--inference', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--serve-options-json', default='')
    ap.add_argument('--until-unix', type=float, default=0.)
    ap.add_argument('--stop-file', default='')
    ap.add_argument('--report', default='')
    return ap.parse_args()


def main():
    args = parse()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    if args.cores:
        from common import parse_cores
        os.sched_setaffinity(0, parse_cores(args.cores))   # run_actor_container pins its processes within this set
    from polytopia_bot.rl.distributed import run_actor_container

    stop = {'flag': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__('flag', True))

    def stopping():
        return (stop['flag'] or (args.until_unix and time.time() > args.until_unix)
                or (args.stop_file and Path(args.stop_file).exists()))

    def report(summary):
        row = dict(summary, at=time.time(), index=args.index, host=os.uname().nodename, inference=args.inference,
                   gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
        print(json.dumps(row), flush=True)
        if args.report:
            tmp = Path(args.report).with_suffix('.tmp')
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(row))
            tmp.replace(args.report)

    address = (args.host, args.port)
    serve_options = json.loads(args.serve_options_json) if args.serve_options_json else None
    print(json.dumps(dict(stage='node_start', index=args.index, address=address, processes=args.processes,
                          games=args.games, cores=args.cores, inference=args.inference, serve_options=serve_options,
                          until_unix=args.until_unix, pid=os.getpid())), flush=True)
    while not stopping():
        started = time.monotonic()
        try:
            summary = run_actor_container(address, args.processes, args.games, first_id=args.index * 1000,
                                          report=report, should_stop=stopping,
                                          resolve_address=lambda: None if stopping() else address,
                                          relay=True, inference='cuda' if args.inference == 'cuda' else None,
                                          serve_options=dict(serve_options) if serve_options else None)
            print(json.dumps(dict(stage='container_exit', seconds=round(time.monotonic() - started, 1),
                                  **{k: summary.get(k) for k in ('samples', 'games', 'errors') if k in summary})),
                  flush=True)
        except Exception as error:   # noqa: BLE001 - log and retry; the keeper restarts the node if it keeps failing
            print(json.dumps(dict(stage='container_error', error=repr(error)[:2000])), flush=True)
        if not stopping():
            time.sleep(30)
    print(json.dumps(dict(stage='node_exit')), flush=True)
    os._exit(0)


if __name__ == '__main__':   # spawned actor processes re-import this file; keep imports in main()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
