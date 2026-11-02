"""Trusted referee, using the current RustEnvironment and public observations only."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from .common import canonical, sha


from .wasi import BotFault, WasiBot


class ReferenceBot:
    def __init__(self, path, expected_sha):
        if sha(Path(path).read_bytes()) != expected_sha:
            raise ValueError('reference checkpoint digest mismatch')
        from polytopia_bot.rl.policy import StructuredPolicy
        self.bot = StructuredPolicy(path, seed=0, deterministic=True)

    def select(self, obs):
        return self.bot.select(obs).action.action_id

    def close(self):
        pass


def reset_map(env, seed, map_type='classic'):
    """Use the existing variant layouts with the same Rust transition engine."""
    if map_type == 'classic':
        return env.reset(seed=seed, settings={})
    from polytopia_bot.maps.stream import MapSource
    from polytopia_bot.learning.memory import PublicMemory
    batch, _ = MapSource(map_type, size=env.size).batch([seed])
    env.games.batch.reset_layouts(batch.seed, batch.terrain, batch.structure, batch.resource, batch.capitals)
    env._sync()
    env.memory, env._cached = PublicMemory(), {}
    env.observe(2)
    return env.observe(1)


def play_game(a, b, seed, seat, budget, trace_path, *, phase, labels, authoring_deadline=None,
              map_type='classic', cap_policy='unresolved'):
    """a occupies seat; native truncations stay unresolved, guest faults are forfeits.

    Full private action logs remain with the referee. The practice review contains
    only a's own observations and chosen actions, never b's private observation.
    """
    if not os.environ.get('MODAL_TASK_ID'):
        raise RuntimeError('games must run on Modal')
    from polytopia_bot.studio.environment import RustEnvironment
    env = RustEnvironment(size=14)
    bots = {seat: a, 3 - seat: b}
    started = time.monotonic()
    row = dict(a=labels[0], b=labels[1], seed=seed, seat=seat, phase=phase, map_type=map_type,
               status='unresolved', reason='game_timeout', winner=None, actions=0)
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    private = path.with_suffix('.private.jsonl')
    chain = '0' * 64
    try:
        obs = reset_map(env, seed, map_type)
        # Warm reference inference outside the game-clock; compile time isn't a bot's move time.
        for bot in bots.values():
            if isinstance(bot, ReferenceBot):
                bot.select(obs)
        started = time.monotonic()
        game_deadline = started+budget.game_seconds
        if authoring_deadline is not None:
            game_deadline = min(game_deadline,authoring_deadline)
        for bot in bots.values():
            if isinstance(bot,WasiBot): bot.absolute_deadline = game_deadline
        with path.open('x') as public_log, private.open('x') as private_log:
            for _ in range(10_000):
                if time.monotonic() >= game_deadline:
                    break
                player = int(env.state.current_player)
                obs = env.observe(player)
                try:
                    action_id = bots[player].select(obs)
                except BotFault as error:
                    if time.monotonic() >= game_deadline:
                        row.update(reason='authoring_deadline' if authoring_deadline is not None else 'game_timeout')
                    else:
                        row.update(status='forfeit', reason=str(error), winner=3 - player, fault_player=player)
                    break
                entry = dict(player=player, version=obs.state_version, action_id=action_id,
                             turn=obs.visible_state['turn'])
                chain = sha((chain + canonical(entry)).encode())
                private_log.write(canonical(dict(**entry, chain=chain)) + '\n')
                if player == seat:
                    public_log.write(canonical(dict(observation=asdict(obs), action_id=action_id)) + '\n')
                transition = env.step(action_id=action_id, expected_state_version=obs.state_version)
                row['actions'] += 1
                if transition.terminated:
                    winner = max(transition.rewards, key=transition.rewards.get)
                    row.update(status='native', reason=transition.reason, winner=winner)
                    break
                if transition.truncated:
                    row.update(reason='engine_truncation')
                    break
            else:
                row.update(reason='action_limit')
        row.update(turns=int(env.state.turn), seconds=time.monotonic() - started,
                   trace_sha256=sha(path.read_bytes()), private_trace_sha256=sha(private.read_bytes()),
                   action_chain_sha256=chain)
    finally:
        env.close()
        a.close()
        b.close()
    row['result'] = 'unresolved' if row['winner'] is None else 'win' if row['winner'] == seat else 'loss'
    from .protocol import apply_cap_policy
    return apply_cap_policy(row,cap_policy)


def warm_reference(path, expected_sha):
    """Compile/load the fixed opponent before starting the authoring clock."""
    from polytopia_bot.studio.environment import RustEnvironment
    bot = ReferenceBot(path,expected_sha)
    env = RustEnvironment(size=14)
    try:
        bot.select(env.reset(seed=500000,settings={}))
    finally:
        env.close()
    return bot
