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
