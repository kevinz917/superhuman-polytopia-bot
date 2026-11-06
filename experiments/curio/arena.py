"""Matches on a Curio box, no Modal: greedy ladders (rl.cpu_arena.match) and search arenas (rl.search_arena).

    python arena.py match --candidate C.npz --opponent O.npz --seed-base 560000 --maps 32 [--games 4] [--processes 128]
                          [--map-mix proposed-v1] [--size 14] --out OUT.json [--name NAME]
    python arena.py search --cfg CFG.json --out OUT.json [--processes 128]

`match` plays every map seed from both seats with both nets greedy (the ladder convention: the outcome is a pure
function of the map), and writes the cpu_arena summary plus the fields experiments/modal_distributed_rl.py's
arena_match records, so scripts/ladder_elo.py and the status tables read it unchanged.
`search` plays cfg['maps'] seeds from cfg['seed_base'] from both seats through rl.search_arena.run; cfg carries
candidate/opponent/value/opponent_search/lanes/depth/replies/determinize/map_mix (see configs/cluster.json 'search').
Run under `taskset -c` to keep the games off the actor cores; the pools pin one process per allowed core.
"""
import argparse
import json
import os
import time
from pathlib import Path


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def match(args):
    from polytopia_bot.rl import cpu_arena
    processes = args.processes or min(2 * args.maps, len(os.sched_getaffinity(0)))
    started = time.monotonic()
    out = cpu_arena.match(args.candidate, args.opponent, range(args.seed_base, args.seed_base + args.maps),
                          games=args.games, deterministic=True, processes=processes, maps=args.map_mix or None,
                          size=args.size)
    out.update(name=args.name or Path(args.out).stem, size=args.size, candidate=args.candidate,
               opponent=args.opponent, seed_base=args.seed_base, maps=args.maps, deterministic=True, engine='rust',
               map_mix=args.map_mix or None, processes=processes, wall_seconds=round(time.monotonic() - started, 1),
               host=os.uname().nodename)
    write(args.out, out)
    print(json.dumps({k: out.get(k) for k in ('name', 'wins', 'losses', 'draws_or_truncations', 'win_rate', 'ci95',
                                               'wall_seconds')}), flush=True)


def search(args):
    from polytopia_bot.rl import search_arena
    cfg = json.loads(Path(args.cfg).read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
    maps, seed_base = int(cfg.pop('maps')), int(cfg.pop('seed_base'))
    name = cfg.pop('name', Path(args.out).stem)
    items = [(seed, seat) for seed in range(seed_base, seed_base + maps) for seat in (1, 2)]
    started = time.monotonic()
    rows = search_arena.run(cfg, items, processes=args.processes or None)
    out = dict(search_arena.summarize(rows), name=name, cfg=cfg, maps=maps, seed_base=seed_base, rows=rows,
               wall_seconds=round(time.monotonic() - started, 1), host=os.uname().nodename)
    write(args.out, out)
    print(json.dumps({k: out.get(k) for k in ('name', 'wins', 'losses', 'truncations', 'win_ci95', 'elo', 'elo_ci95',
                                               'wall_seconds')}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('match')
    m.add_argument('--candidate', required=True)
    m.add_argument('--opponent', required=True)
    m.add_argument('--seed-base', type=int, required=True)
    m.add_argument('--maps', type=int, required=True)
    m.add_argument('--games', type=int, default=4)
    m.add_argument('--processes', type=int, default=0)
    m.add_argument('--map-mix', default='proposed-v1')
    m.add_argument('--size', type=int, default=14)
    m.add_argument('--out', required=True)
    m.add_argument('--name', default='')
    m.set_defaults(fn=match)
    s = sub.add_parser('search')
    s.add_argument('--cfg', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--processes', type=int, default=0)
    s.set_defaults(fn=search)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':   # the arena pools spawn workers that re-import this file
    main()
