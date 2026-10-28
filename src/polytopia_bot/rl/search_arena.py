"""Search-bot vs raw-bot matches on engine-rs (the real game's rules), one game per pinned CPU process.

Each (map seed, seat) item is one game: the candidate plays its turns through
search.TurnSearcher (lanes = K), the opponent plays its raw greedy policy
(a TurnSearcher with one lane, which is exactly network.sample(deterministic=True)),
or searches too when cfg['opponent_search'] gives its settings.
cfg['value'] names a checkpoint whose value head scores the lanes instead of the
candidate's own (TurnSearcher(value_params=...)).
Every replayed sub-action goes through engine-rs and must be legal.
Per game we record outcome, search compute time, how often search overrode the
greedy turn, and the critic's score of the greedy turn (for calibration).
"""
import json
import math
import os
import time
from pathlib import Path

import numpy as np

SIZE = 14  # default board; cfg['size'] overrides
SUMMON_FAMILY = 2
UNIT_NAMES = {int(u['id']): u['name'] for u in json.loads(
    (Path(__file__).parents[1] / 'gpu/rules.json').read_text())['units']}
_W = {}


def _init(cfg, cores):
    # An initializer that raises makes Pool respawn the worker, which then blocks forever on the empty core
    # queue (the c3-v1 sweep hung that way). Keep the error and raise it from the first game instead.
    try:
        _setup(cfg, cores.get(timeout=60))
    except BaseException as e:
        _W['error'] = repr(e)


def _setup(cfg, core):
    if core is not None and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {core})
    import jax
    from .distributed import checkpoint_template
    from .runner import load_checkpoint
    from .rust_engine import RustGames
    from .search import TurnSearcher, first
    SIZE = cfg.get('size', 14)
    cand = load_checkpoint(cfg['candidate'], checkpoint_template(cfg['candidate']))
    opp = load_checkpoint(cfg['opponent'], checkpoint_template(cfg['opponent']))
    # cfg['value']: a second checkpoint whose (fog-only) value head scores the candidate's lanes
    value = load_checkpoint(cfg['value'], checkpoint_template(cfg['value'])) if cfg.get('value') else None
    if cfg.get('map_mix'):   # map variants: the board for each seed comes from maps.stream (classic = identical)
        from polytopia_bot.maps.stream import MapSource
        source = MapSource(cfg['map_mix'], SIZE)
        games = RustGames(SIZE, 1, map_batch=True)

        def reset(seed):
            games.reset(source.batch([int(seed)])[0])
            return first(games.states())
        reset.map_type = source.type_for
    else:
        games = RustGames(SIZE, 1)

        def reset(seed):
            games.reset(np.array([int(seed)], np.int32))
            return first(games.states())
    state = reset(0)
    started = time.monotonic()
    searcher = TurnSearcher(cand, size=SIZE, lanes=cfg['lanes'], depth=cfg.get('depth', 0),
                            temperature=cfg.get('temperature', 1.), rollout_temperature=cfg.get('rollout_temperature', 0.),
                            max_turn_actions=cfg.get('max_turn_actions', 96), root=cfg.get('root', 'sequence'),
                            margin=cfg.get('margin', 0.), prior_coef=cfg.get('prior_coef', 0.),
                            after_end=cfg.get('after_end', False), replies=cfg.get('replies', 0),
                            determinize=cfg.get('determinize', False), parallel=cfg.get('parallel', False),
                            workers=cfg.get('workers'), value_params=value)
    searcher.compile(state, jax.random.key(0))
    # cfg['opponent_search']: TurnSearcher settings for a searching opponent (search vs search); its lanes are scored
    # by cfg['opponent_value']'s value head if given, else its own. Default: the raw greedy opponent.
    opp_search = cfg.get('opponent_search')
    if opp_search:
        opp_value = (load_checkpoint(cfg['opponent_value'], checkpoint_template(cfg['opponent_value']))
                     if cfg.get('opponent_value') else None)
        raw = TurnSearcher(opp, size=SIZE, max_turn_actions=cfg.get('max_turn_actions', 96), value_params=opp_value,
                           **opp_search)
    else:
        raw = TurnSearcher(opp, size=SIZE, lanes=1, max_turn_actions=cfg.get('max_turn_actions', 96))
    raw.compile(state, jax.random.key(0))
    _W.update(cfg=cfg, searcher=searcher, raw=raw, reset=reset, compile_seconds=time.monotonic() - started)


def play_game(searcher, raw, reset, seed, seat, salt=0):
    import jax
    key = jax.random.fold_in(jax.random.key(salt), seed * 2 + seat)
    s = reset(seed)
    row = dict(seed=int(seed), seat=int(seat), search_turns=0, search_seconds=0., sub_actions=0, plans=0,
               overrides=0, units={}, trace=[], lane_steps=0, incomplete_lanes=0)
    started, fallbacks = time.monotonic(), getattr(searcher, 'fallbacks', 0)
    while not bool(s.terminated | s.truncated):
        key, draw = jax.random.split(key)
        if int(s.current_player) == seat:
            turn = int(s.turn)
            t0 = time.perf_counter()
            s, plans = searcher.play_turn(s, draw)
            row['search_seconds'] += time.perf_counter() - t0
            row['search_turns'] += 1
            for i, p in enumerate(plans):
                row['plans'] += 1
                row['sub_actions'] += int(p.count)
                row['overrides'] += int(p.best != 0)
                row['lane_steps'] += int(p.steps.sum())
                row['incomplete_lanes'] += int((~p.complete).sum())
                acts = p.actions[:int(p.count)]
                for unit in acts[acts[:, 0] == SUMMON_FAMILY, 2]:
                    row['units'][int(unit)] = row['units'].get(int(unit), 0) + 1
                if i == 0:
                    # (turn, greedy-lane score, chosen score, mean lane score)
                    row['trace'].append((turn, round(float(p.scores[0]), 4), round(float(p.scores[p.best]), 4),
                                         round(float(p.scores.mean()), 4)))
        else:
            s, _ = raw.play_turn(s, draw)
    row.update(reward=float(searcher.rewards(s)[seat - 1]), truncated=bool(s.truncated),
               map_type=getattr(reset, 'map_type', lambda _: 'classic')(seed),
               turns=int(s.turn), seconds=time.monotonic() - started,
               fallbacks=getattr(searcher, 'fallbacks', 0) - fallbacks)
    return row


def _play(item):
    seed, seat = item
    w = _W
    if 'error' in w:
        raise RuntimeError(f"search_arena worker setup failed: {w['error']}")
    row = play_game(w['searcher'], w['raw'], w['reset'], seed, seat, salt=w['cfg'].get('salt', 0))
    row['compile_seconds'] = w['compile_seconds']
    return row


def run(cfg, items, processes=None):
    """Plays items [(seed, seat)] with one process per core; returns per-game rows."""
    import multiprocessing as mp
    from .distributed import _env_setup
    _env_setup()   # inherited by the spawned workers before they initialise jax
    cores = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else [None]
    processes = min(processes or len(cores), len(items))
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    for i in range(processes):
        queue.put(cores[i % len(cores)] if len(cores) >= processes else None)
    with ctx.Pool(processes, initializer=_init, initargs=(cfg, queue)) as pool:
        return list(pool.imap_unordered(_play, items))


def elo(score):
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / score - 1)


def summarize(rows):
    n = len(rows)
    wins = sum(r['reward'] > 0 for r in rows)
    losses = sum(r['reward'] < 0 for r in rows)
    draws = n - wins - losses
    by_map = {}
    for r in rows:
        by_map.setdefault(r['seed'], []).append(1. if r['reward'] > 0 else .5 if r['reward'] == 0 else 0.)
    maps = np.array([np.mean(v) for v in by_map.values()])
    wins_map = {}
    for r in rows:
        wins_map.setdefault(r['seed'], []).append(1. if r['reward'] > 0 else 0.)
    wmaps = np.array([np.mean(v) for v in wins_map.values()])
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(maps), len(maps)) for _ in range(4000)]
    score_boot = np.array([maps[i].mean() for i in idx])
    win_boot = np.array([wmaps[i].mean() for i in idx])
    score = (wins + .5 * draws) / max(n, 1)
    turns = sum(r['search_turns'] for r in rows)
    units = {}
    for r in rows:
        for u, c in r['units'].items():
            units[u] = units.get(u, 0) + c
    total_units = sum(units.values())
    return dict(
        games=n, wins=int(wins), losses=int(losses), truncations=int(draws), win_rate=wins / max(n, 1),
        win_ci95=[float(np.percentile(win_boot, 2.5)), float(np.percentile(win_boot, 97.5))],
        score=score, score_ci95=[float(np.percentile(score_boot, 2.5)), float(np.percentile(score_boot, 97.5))],
        elo=elo(score), elo_ci95=[elo(float(np.percentile(score_boot, 2.5))), elo(float(np.percentile(score_boot, 97.5)))],
        seconds_per_search_turn=sum(r['search_seconds'] for r in rows) / max(turns, 1),
        sub_actions_per_turn=sum(r['sub_actions'] for r in rows) / max(turns, 1),
        lane_steps_per_turn=sum(r['lane_steps'] for r in rows) / max(turns, 1),
        override_rate=sum(r['overrides'] for r in rows) / max(sum(r['plans'] for r in rows), 1),
        replans_per_turn=sum(r['plans'] for r in rows) / max(turns, 1),
        incomplete_lanes=int(sum(r['incomplete_lanes'] for r in rows)),
        mean_turns=float(np.mean([r['turns'] for r in rows])) if rows else 0.,
        mean_turns_decided=float(np.mean([r['turns'] for r in rows if r['reward'] != 0] or [0])),
        game_seconds_mean=float(np.mean([r['seconds'] for r in rows])) if rows else 0.,
        compile_seconds=float(np.mean([r['compile_seconds'] for r in rows])) if rows else 0.,
        unit_mix={UNIT_NAMES.get(int(u), str(u)): round(c / total_units, 4)
                  for u, c in sorted(units.items(), key=lambda kv: -kv[1])} if total_units else {},
    )
