"""Load a structured PPO checkpoint and play through the existing public API."""
from dataclasses import dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from polytopia_bot.environment import LegalAction, Observation
from polytopia_bot.gpu.actions import Masks
from . import network
from .actions import flatten_choice, offsets, shapes
from .observation import encode_view, view_from_public
from .runner import Config, load_checkpoint


@dataclass(frozen=True)
class Decision:
    action: LegalAction
    log_probability: float
    value: float


class StructuredPolicy:
    """Host adapter for public-observation consumers (studio.environment.RustEnvironment).

    Training uses the device-only path instead. This adapter never gets access
    to private state, seeds or opponent technology/bank.
    """
    def __init__(self, checkpoint, *, seed=0, deterministic=False):
        self.path = Path(checkpoint)
        self.config = Config(**json.loads(self.path.with_suffix('.json').read_text())['config'])
        self.key = jax.random.key(seed)
        self.params = None
        self._sample = jax.jit(lambda p, x, m, k: network.sample(p, x,
            m._replace(resign=jnp.bool_(False)) if self.config.no_resign else m, k,
            size=self.config.size, deterministic=deterministic))
        self._encode = jax.jit(lambda view: encode_view(view, self.config.size))

    def select(self, observation: Observation, view=None) -> Decision:
        """`view`: optionally the engine's packed-view dict of the same position (RustEnvironment.view): a network of
        observation version 2 reads the unit facing from it, which the public observation does not carry. Version-1
        networks always encode the public observation."""
        from dataclasses import asdict
        if observation.visible_state['board_size'] != self.config.size or not observation.legal_actions:
            raise ValueError('incompatible board or no legal decisions')
        source = view if view is not None and self.config.obs_version >= 2 else view_from_public(asdict(observation))
        x = self._encode(source)
        if self.params is None:
            # the checkpoint's own input width (its observation version; the features are the newest, which the
            # trunk fits to it)
            from .distributed import feature_shapes
            board, glob = feature_shapes(self.config.players, self.config.obs_version)
            template = network.init(jax.random.key(0), board, glob,
                                    size=self.config.size, width=self.config.width, blocks=self.config.blocks,
                                    architecture=self.config.architecture, extra_blocks=self.config.extra_blocks,
                                    global_pool=self.config.global_pool, block_pool=self.config.block_pool,
                                    attention=self.config.attention)
            self.params = load_checkpoint(self.path, template)
        bounds = offsets(self.config.size)
        flat = np.zeros(bounds[-1], bool)
        catalog = {int(a.action_id): a for a in observation.legal_actions}
        for index in catalog:
            if not 0 <= index < flat.size:
                raise ValueError('expected simulator action IDs, not prompt-local IDs')
            flat[index] = True
        masks = Masks(*(jnp.asarray(flat[bounds[i]:bounds[i + 1]].reshape(shape))
                        for i, shape in enumerate(shapes(self.config.size))))
        self.key, key = jax.random.split(self.key)
        choice, _, logp, value = self._sample(self.params, x, masks, key)
        index = int(flatten_choice(choice, self.config.size))
        return Decision(catalog[index], float(logp), float(value))
