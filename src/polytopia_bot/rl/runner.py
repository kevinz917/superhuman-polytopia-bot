"""Checkpoints (save/load with integrity and rules checks) and the network/run Config they carry.

The single-machine PPO loop that used to live here ran on the retired JAX engine and was removed on 2026-10-27
(docs/rules-alignment-2026-10-27.md); training is rl/distributed.py.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from polytopia_bot.ruleset import BOARD_SIZES, RULESET_ID, compatible, rules_digest
from .data import sha256


@dataclass(frozen=True)
class Config:
    reward_mode: str = 'native'
    architecture: str = 'legacy'
    no_resign: bool = False
    size: int = 14
    seed: int = 20261017
    width: int = 64
    blocks: int = 2
    extra_blocks: int = 0           # identity-initialized pre-activation blocks (network.grow); 0 for older checkpoints
    global_pool: bool = False       # extra blocks carry a global pooling bias (network.init global_pool)
    players: int = 2                # seats the network's features are for (4: rl.observation multi-seat channels)
    environments: int = 8
    horizon: int = 32
    updates: int = 32
    epochs: int = 2
    minibatch: int = 64
    learning_rate: float = 3e-4
    entropy_coef: float = .01
    gamma: float = 1.
    gae_lambda: float = .95
    target_kl: float = .03
    bc_epochs: int = 2
    pool_size: int = 32
    eval_maps: int = 2
    eval_actions: int = 512
    training_seconds: int = 1200

    def validate(self):
        if self.reward_mode not in ('native', 'own-potential-v1'):
            raise ValueError('unknown reward mode')
        if self.architecture not in ('legacy', 'normalized-v2'):
            raise ValueError('unknown architecture')
        if self.size not in BOARD_SIZES or self.extra_blocks < 0 or min(self.width, self.environments, self.horizon, self.updates,
                self.epochs, self.minibatch, self.pool_size, self.eval_maps, self.eval_actions, self.training_seconds) < 1:
            raise ValueError('invalid allocation')
        if self.blocks < 0 or self.bc_epochs < 0 or self.environments * self.horizon % self.minibatch:
            raise ValueError('invalid model or minibatch size')
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or self.learning_rate <= 0 or self.entropy_coef < 0 or self.target_kl <= 0:
            raise ValueError('invalid PPO hyperparameters')


def write_json(path, data):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def finite(tree):
    return all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(tree))


def save_checkpoint(path, params, config, *, stage, optimizer=None):
    leaves, _ = jax.tree.flatten(params)
    arrays = {f'p{i}': np.asarray(v) for i, v in enumerate(leaves)}
    if optimizer is not None:
        arrays.update({f'o{i}': np.asarray(v) for i, v in enumerate(jax.tree.leaves(optimizer))})
    np.savez_compressed(path, **arrays)
    write_json(Path(path).with_suffix('.json'), dict(schema='structured-ppo-v1', config=asdict(config),
               stage=stage, parameter_elements=sum(v.size for v in leaves), sha256=sha256(path),
               ruleset=RULESET_ID, simulator_sha256=rules_digest()))


def load_checkpoint(path, template):
    leaves, structure = jax.tree.flatten(template)
    meta = json.loads(Path(path).with_suffix('.json').read_text())
    if meta['sha256'] != sha256(path) or not compatible(meta):
        raise ValueError('checkpoint integrity/rules mismatch')
    with np.load(path, allow_pickle=False) as archive:
        values = [archive[f'p{i}'] for i in range(len(leaves))]
    if any(x.shape != y.shape for x, y in zip(leaves, values, strict=True)) or not finite(values):
        raise ValueError('invalid checkpoint tensors')
    return structure.unflatten([jnp.asarray(v) for v in values])
