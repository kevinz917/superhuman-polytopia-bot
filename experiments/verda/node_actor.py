"""Actor node on a Verda CPU instance: distributed.run_actor_container against the learner's IP, no Modal.

    python node_actor.py LEARNER_IP INDEX [games] [processes]
processes 0 (default): one pinned process per CPU. One WeightRelay connection per node, as on Modal. Under systemd
(polytopia-actor.service, Restart=always) the node starts again whenever it exits, e.g. after a learner restart.
"""
import json
import sys
import time

from polytopia_bot.rl.distributed import run_actor_container


def report(summary):
    print(json.dumps(dict(summary, at=time.time())), flush=True)


if __name__ == '__main__':   # actor processes are spawned and re-import this file
    host, index = sys.argv[1], int(sys.argv[2])
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    processes = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    summary = run_actor_container((host, 7070), processes, games, first_id=index * 1000, report=report, relay=True)
    print(json.dumps(dict(stage='actor_done', **summary)), flush=True)
