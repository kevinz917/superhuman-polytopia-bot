"""Play fresh games under the real rules (engine-rs) for the hands trailer and save every action compactly.

Two checkpoints play each other deterministically from a new seed; nothing comes from the replay library.
Runs on the Studio code and venv that serve the hero bot (engine-rs build + JAX policy), e.g.:

  M=/Users/K.Z./Desktop/projects/polytopia-bot
  PYTHONPATH=$M/.claude/worktrees/studio-main/src $M/.claude/worktrees/studio-main/.venv/bin/python \
      scripts/trailer/gen_games.py OUT_DIR SEED [SEED ...]

Each OUT_DIR/game-SEED.json holds the god-view start state and, per action, the command and what it changed:
units (owner, kind, health by tile), cities (owner, level by tile) and changed tiles (terrain, owner, structure,
resource, road, capital), plus the mover's own value estimate (`value`, -1 loss .. 1 win, from the policy's value
head) and both players' in-game scores after the action (`scores`).
"""

import json
import sys

import numpy as np
import time
from pathlib import Path

M = Path('/Users/K.Z./Desktop/projects/polytopia-bot/artifacts')
HERO = M / 'checkpoint-hero-cont1h/final.npz'
CHAMPION = M / 'checkpoint-scratch-champion/final.npz'
MAX_TURNS = 60


def god(env):
    from polytopia_bot.studio.service import world_view
    vs = world_view(env)['visible_state']
    tiles = {t['tile']: [t.get('terrain', 0), t.get('owner', 0), t.get('structure', 0), t.get('resource', 0),
                         t.get('road', 0), t.get('capital', 0)] for t in vs['tiles']}
    units = {u['tile']: [u['owner'], u['kind'], round(u['health'], 1), int(bool(u.get('veteran')))] for u in vs['units']}
    cities = {c['tile']: [c['owner'], c.get('level', 1) or 1] for c in vs['cities']}
    return tiles, units, cities, vs['turn']


def play(seed, out):
    from polytopia_bot.rl.policy import StructuredPolicy
    from polytopia_bot.studio.environment import RustEnvironment
    env = RustEnvironment(14)
    env.reset(seed=seed, settings={})
    hero_seat = 1 + seed % 2
    policies = {hero_seat: StructuredPolicy(HERO, seed=seed, deterministic=True),
                3 - hero_seat: StructuredPolicy(CHAMPION, seed=seed + 1, deterministic=True)}
    tiles, units, cities, turn = god(env)
    record = {'seed': seed, 'size': 14, 'hero_seat': hero_seat, 'tiles': [tiles[i] for i in range(14 * 14)],
              'units': units, 'cities': cities, 'steps': []}
    start = time.monotonic()
    while True:
        player = int(env.state.current_player)
        obs = env.observe(player)
        decision = policies[player].select(obs)
        result = env.step(action_id=decision.action.action_id, expected_state_version=obs.state_version)
        t2, u2, c2, turn = god(env)
        changed = {k: v for k, v in t2.items() if tiles[k] != v}
        scores = [int(x) for x in np.asarray(env.state.players.score).tolist()]
        step = {'player': player, 'turn': turn, 'command': decision.action.command, 'units': u2,
                'value': round(float(decision.value), 4), 'scores': scores}
        if c2 != cities:
            step['cities'] = c2
        if changed:
            step['tiles'] = changed
        record['steps'].append(step)
        tiles, units, cities = t2, u2, c2
        if result.terminated or result.truncated or turn >= MAX_TURNS:
            record['winner'] = max(result.rewards, key=result.rewards.get) if result.terminated else 0
            record['reason'] = result.reason or 'turn cap'
            break
    env.close()
    path = Path(out) / f'game-{seed}.json'
    path.write_text(json.dumps(record, separators=(',', ':')))
    print(f'{path}: {len(record["steps"])} actions, {turn} turns, winner {record["winner"]} '
          f'(hero seat {hero_seat}), {time.monotonic() - start:.0f} s', flush=True)


if __name__ == '__main__':
    Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)
    for s in sys.argv[2:]:
        play(int(s), sys.argv[1])
