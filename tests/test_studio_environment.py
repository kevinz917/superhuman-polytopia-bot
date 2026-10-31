"""Studio's engine-rs environment serializes observations public_state accepts."""
import numpy as np

from polytopia_bot.learning.interface import public_state
from polytopia_bot.studio.environment import RustEnvironment


def test_rust_environment_observes_both_seats():
    # engine-rs unpacks every view field as int32; relations must still reach public_state as bools
    env = RustEnvironment(14)
    env.reset(seed=1810000001, settings={})   # Studio.warmup's seed
    for _ in range(8):
        for player in (1, 2):
            obs = env.observe(player)
            public_state(obs)
            assert all(type(x) is bool for row in obs.visible_state['relations'].values() for x in row)
        obs = env.observe(int(env.state.current_player))
        if not obs.legal_actions:
            break
        env.step(action_id=obs.legal_actions[np.random.default_rng(0).integers(len(obs.legal_actions))].action_id,
                 expected_state_version=obs.state_version)


def test_attack_preview_matches_the_attack():
    # Studio's attack preview plays the attack on a scratch copy; the live game must end up exactly as previewed.
    env = RustEnvironment(14)
    rng = np.random.default_rng(3)
    checked = 0
    for seed in range(4):
        env.reset(seed=seed, settings={})
        for _ in range(3000):
            if env.state.terminated or env.state.truncated:
                break
            player = int(env.state.current_player)
            obs = env.observe(player)
            attacks = [a for a in obs.legal_actions if a.command['kind'] == 2]
            if attacks:
                action = attacks[rng.integers(len(attacks))]
                predicted = {u['tile']: u['after'] for u in env.preview(action.action_id)}
                assert env.observe(player) == obs   # the preview leaves the live game alone
            else:
                # Mostly act, sometimes end the turn (unavailable while a city reward is pending); never resign.
                options = [a for a in obs.legal_actions if a.command['kind'] != 11]
                acts = [a for a in options if a.command['kind'] != 10]
                pool = acts if acts and (rng.random() < 0.85 or len(acts) == len(options)) else options
                action = pool[rng.integers(len(pool))]
            before = {u['tile']: u for u in obs.visible_state['units']}
            env.step(action_id=action.action_id, expected_state_version=obs.state_version)
            if not attacks:
                continue
            now = {u['tile']: u for u in env.observe(player).visible_state['units']}
            source, target = action.command['source'], action.command['target']
            actual = {}
            for tile, u in before.items():
                if tile in now and now[tile]['owner'] == u['owner']:
                    hp = now[tile]['health']
                elif tile == source and target in now and now[target]['owner'] == u['owner']:
                    hp = now[target]['health']   # won and advanced
                else:
                    hp = 0.0
                if hp != u['health']:
                    actual[tile] = hp
            assert actual == predicted
            checked += 1
    assert checked >= 20
