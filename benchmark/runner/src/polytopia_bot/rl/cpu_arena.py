"""Paired-seat checkpoint matches on the frozen engine, stepped on CPU.

Every map seed is played twice (candidate as seat 1, then seat 2), on engine-rs (rust_engine.RustGames, the real
game's rules). Checkpoints from the retired research rules play under these rules too (polytopia_bot.ruleset). Besides
win/loss it records each side's recruit mix, opening (first eight turns) and every
technology gained (turn, tech id), so strategy diversity and tech progress are
measured on the same games as strength.
"""
import json
import os
import time

import numpy as np

OPENING_TURNS = 8
_RULES = json.loads((__import__('pathlib').Path(__file__).parents[1] / 'gpu/rules.json').read_text())
UNIT_NAMES = {int(u['id']): u['name'] for u in _RULES['units']}
TECH_TIER = {int(t['id']): int(t['tier']) for t in _RULES['technologies']}
TECH_NAMES = {int(t['id']): t['name'] for t in _RULES['technologies']}
TECH_TURNS = (10, 20, 30, 40, 50, 60, 80, 100, 120, 150)


def _play_items(candidate_path, opponent_path, items, games, deterministic, seed, fast=False,
                opponent_deterministic=None, maps=None, size=14):
    from .distributed import _env_setup
    _env_setup()
    import jax
    import jax.numpy as jnp
    from . import network
    from .actions import Choice
    from .distributed import checkpoint_template
    from .runner import load_checkpoint
    # Each checkpoint loads with its own architecture, so a grown 18x18 network can meet its 14x14 parent.
    cand = load_checkpoint(candidate_path, checkpoint_template(candidate_path))
    opp = load_checkpoint(opponent_path, checkpoint_template(opponent_path))
    # maps=None keeps today's seeds -> mapgen.reset; a mix name/dict plays maps.stream layouts.
    source = None
    if maps is not None:
        from polytopia_bot.maps.stream import MapSource
        from polytopia_bot.maps.state import stack
        source = MapSource(maps, size)
    from .rust_engine import RustGames
    engine = RustGames(size, games, map_batch=source is not None)   # round limit for the board size
    cache = {}
    def batch(seeds):
        if source is None:
            return seeds
        for s in seeds:
            if s not in cache:
                cache[s] = source.layout(s)
        return stack([(s, cache[s][1]) for s in seeds])
    kind_of = lambda s: cache[s][0] if s in cache else 'classic' 
    make = lambda det: jax.jit(jax.vmap(lambda p, x, m, k: network.sample(p, x, m, k, size=size, deterministic=det)[0],
                                        in_axes=(None, 0, 0, 0)))
    act = make(deterministic)
    act_opponent = act if opponent_deterministic in (None, deterministic) else make(opponent_deterministic)
    queues = [items[i::games] for i in range(games)]
    cursor = [0] * games
    dummy = 299999
    def seed_of(i, at):
        return queues[i][at][0] if at < len(queues[i]) else dummy
    states, prepared = engine.reset(batch([seed_of(i, 0) for i in range(games)]))
    blank = lambda: dict(steps=0, cand_units={}, opp_units={}, cand_research_early=0, first_non_warrior=None,
                         opp_first_non_warrior=None, turn=0, captures=0, ports=0, cand_tech=[], opp_tech=[])
    known = [None] * games  # per lane: tech array [2, 128] at the previous decision of the current game
    per_game = [blank() for _ in range(games)]
    results = []
    key = jax.random.key(seed)
    while any(cursor[i] < len(queues[i]) for i in range(games)):
        current = np.asarray(states.current_player)
        turn = np.asarray(states.turn)
        tech = np.asarray(states.players.tech)
        seats = np.array([queues[i][cursor[i]][1] if cursor[i] < len(queues[i]) else 1 for i in range(games)])
        ours = current == seats
        key, k1, k2 = jax.random.split(key, 3)
        a = act(cand, prepared.features, prepared.masks, jax.random.split(k1, games))
        b = act_opponent(opp, prepared.features, prepared.masks, jax.random.split(k2, games))
        sel = jnp.asarray(ours)
        choice = Choice(*(jnp.where(sel, x, y) for x, y in zip(a, b)))
        fam, tgt = np.asarray(choice.family), np.asarray(choice.target)
        for i in range(games):
            if cursor[i] >= len(queues[i]):
                continue
            g = per_game[i]
            seat_i = queues[i][cursor[i]][1]
            # New technologies since the last decision (research, ruins, ...); starting techs are not counted.
            if known[i] is not None:
                for p in (1, 2):
                    for t in np.flatnonzero(tech[i, p - 1] & ~known[i][p - 1]):
                        g['cand_tech' if p == seat_i else 'opp_tech'].append([int(turn[i]), int(t)])
            known[i] = tech[i].copy()
            g['steps'] += 1
            g['turn'] = int(turn[i])
            g['captures'] += int(fam[i] == 8)
            g['ports'] += int(fam[i] == 3 and tgt[i] == 8)
            if fam[i] == 2:
                unit = int(tgt[i])
                side = 'cand_units' if ours[i] else 'opp_units'
                g[side][unit] = g[side].get(unit, 0) + 1
                field = 'first_non_warrior' if ours[i] else 'opp_first_non_warrior'
                if unit != 2 and g[field] is None:
                    g[field] = int(turn[i])
            if fam[i] == 4 and ours[i] and turn[i] < OPENING_TURNS:
                g['cand_research_early'] += 1
        next_seeds = [seed_of(i, cursor[i] + 1) for i in range(games)]
        tick, status = engine.tick(states, choice, batch(next_seeds))
        if np.any(np.asarray(status) != 0):
            raise RuntimeError('illegal evaluation action')
        done = np.asarray(tick.done)
        for i in np.nonzero(done)[0]:
            if cursor[i] >= len(queues[i]):
                continue
            map_seed, seat = queues[i][cursor[i]]
            r = float(np.asarray(tick.rewards)[i][seat - 1])
            results.append(dict(seed=int(map_seed), seat=int(seat), reward=r, truncated=bool(np.asarray(tick.truncated)[i]),
                                map=kind_of(int(map_seed)), **per_game[i]))
            per_game[i] = blank()
            known[i] = None
            cursor[i] += 1
        states, prepared = tick.states, tick.prepared
    return results


def _worker(args):
    return _play_items(*args)


def match(candidate_path, opponent_path, seeds, *, processes=None, games=8, deterministic=True, seed=1,
          opponent_deterministic=None, maps=None, size=14):
    import multiprocessing as mp
    items = [(int(s), seat) for s in seeds for seat in (1, 2)]
    processes = processes or min(len(items) // games or 1, os.cpu_count() or 1)
    chunks = [items[i::processes] for i in range(processes)]
    started = time.monotonic()
    with mp.get_context('spawn').Pool(processes) as pool:
        parts = pool.map(_worker, [(candidate_path, opponent_path, c, min(games, len(c)), deterministic, seed + i,
                                    False, opponent_deterministic, maps, size) for i, c in enumerate(chunks) if c])
    rows = [r for part in parts for r in part]
    return summarize(rows, seconds=time.monotonic() - started)


def summarize(rows, seconds=None):
    wins = sum(r['reward'] > 0 for r in rows)
    losses = sum(r['reward'] < 0 for r in rows)
    def mix(side):
        total = {}
        for r in rows:
            for u, c in r[side].items():
                total[u] = total.get(u, 0) + c
        n = sum(total.values())
        return {UNIT_NAMES.get(int(u), str(u)): round(c / n, 4) for u, c in sorted(total.items(), key=lambda kv: -kv[1])} if n else {}
    nonwarrior_games = sum(any(int(u) != 2 for u in r['cand_units']) for r in rows)
    first = [r['first_non_warrior'] for r in rows if r['first_non_warrior'] is not None]
    # paired-map bootstrap of the win fraction
    by_map = {}
    for r in rows:
        by_map.setdefault(r['seed'], []).append(1. if r['reward'] > 0 else 0.)
    maps = np.array([np.mean(v) for v in by_map.values()])
    rng = np.random.default_rng(0)
    boots = [maps[rng.integers(0, len(maps), len(maps))].mean() for _ in range(2000)] if len(maps) else [0]
    return dict(games=len(rows), wins=int(wins), losses=int(losses), draws_or_truncations=len(rows) - int(wins) - int(losses),
                truncations=sum(r['truncated'] for r in rows), win_rate=wins / max(len(rows), 1),
                ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                candidate_unit_mix=mix('cand_units'), opponent_unit_mix=mix('opp_units'),
                candidate_non_warrior_games=int(nonwarrior_games),
                candidate_unit_types_per_game=float(np.mean([len(r['cand_units']) for r in rows])) if rows else 0.,
                candidate_first_non_warrior_turn_median=float(np.median(first)) if first else None,
                candidate_early_research_mean=float(np.mean([r['cand_research_early'] for r in rows])) if rows else 0.,
                mean_steps=float(np.mean([r['steps'] for r in rows])) if rows else 0., seconds=seconds,
                mean_turns=float(np.mean([r.get('turn', 0) for r in rows])) if rows else 0.,
                candidate_tech=tech_summary(rows, ('cand_tech',)), opponent_tech=tech_summary(rows, ('opp_tech',)),
                by_map=by_map_type(rows), details=rows)


def tech_summary(rows, sides):
    """Technology progress of the given sides ('cand_tech', 'opp_tech'); one sample per game and side.
    by_turn[T] averages over games still running at turn T (games that ended earlier are not zero-padded)."""
    samples = [(r.get(side) or [], r.get('turn', 0)) for r in rows for side in sides if side in r]
    if not samples:
        return {}
    final = [2 + len(ev) for ev, _ in samples]  # Basic and Hunting at the start
    tier3 = [sum(TECH_TIER.get(t, 0) == 3 for _, t in ev) for ev, _ in samples]
    by_turn = {}
    for T in TECH_TURNS:
        alive = [2 + sum(turn <= T for turn, _ in ev) for ev, last in samples if last >= T]
        if alive:
            by_turn[str(T)] = dict(games=len(alive), mean_techs=float(np.mean(alive)))
    first3 = [min(turn for turn, t in ev if TECH_TIER.get(t, 0) == 3) for ev, _ in samples
              if any(TECH_TIER.get(t, 0) == 3 for _, t in ev)]
    counts = {}
    for ev, _ in samples:
        for t in {t for _, t in ev}:
            counts[t] = counts.get(t, 0) + 1
    share = {TECH_NAMES.get(t, str(t)): round(c / len(samples), 3) for t, c in sorted(counts.items(), key=lambda kv: -kv[1])}
    return dict(samples=len(samples), final_mean=float(np.mean(final)), final_median=float(np.median(final)),
                tier3_mean=float(np.mean(tier3)), share_with_tier3=float(np.mean([x > 0 for x in tier3])),
                first_tier3_turn_median=float(np.median(first3)) if first3 else None, by_turn=by_turn,
                tech_share=share)


def unit_mix(rows, sides):
    total = {}
    for r in rows:
        for side in sides:
            for u, c in r.get(side, {}).items():
                total[int(u)] = total.get(int(u), 0) + c
    n = sum(total.values())
    return {UNIT_NAMES.get(u, str(u)): round(c / n, 4) for u, c in sorted(total.items(), key=lambda kv: -kv[1])} if n else {}


def by_map_type(rows):
    out = {}
    for kind in sorted({r.get('map', 'classic') for r in rows}):
        rs = [r for r in rows if r.get('map', 'classic') == kind]
        seat1 = [r for r in rs if r['seat'] == 1]
        out[kind] = dict(games=len(rs), wins=sum(r['reward'] > 0 for r in rs), losses=sum(r['reward'] < 0 for r in rs),
                         truncations=sum(r['truncated'] for r in rs),
                         mean_turns=float(np.mean([r.get('turn', 0) for r in rs])),
                         mean_captures=float(np.mean([r.get('captures', 0) for r in rs])),
                         games_with_ports=sum(r.get('ports', 0) > 0 for r in rs),
                         seat1_win_rate=float(np.mean([r['reward'] > 0 for r in seat1])) if seat1 else None,
                         candidate_unit_mix=unit_mix(rs, ('cand_units',)), candidate_tech=tech_summary(rs, ('cand_tech',)),
                         both_unit_mix=unit_mix(rs, ('cand_units', 'opp_units')),
                         both_tech=tech_summary(rs, ('cand_tech', 'opp_tech')))
    return out
