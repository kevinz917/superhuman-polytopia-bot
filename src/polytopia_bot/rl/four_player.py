"""Four-seat networks from two-seat checkpoints (docs/four-player-2026-10-24.md).

A four-seat game's features (observation.encode_view) are the two-seat ones with 'enemy' meaning any other seat
plus new board channels and global entries appended at the end. widen_to_players gives a two-seat network zero
weights for every new input, so on a four-seat position it computes exactly what the two-seat network computes
on the old part of the features; the omniscient critic, which reads the union of every seat's view, keeps the
parent's own-view and next-seat-view weights and starts blind to the third and fourth seats.
"""
import jax
import jax.numpy as jnp

from . import network
from .observation import Features


def widen_to_players(params, players, *, critic=False, obs_version=1):
    """Two-seat policy (or omniscient critic with critic=True) -> the same function on `players`-seat features, both
    of observation version `obs_version` (the version is kept: the multi-seat channels come after the unit facing
    and before the version-3 history channels, so the two-seat features are placed by index)."""
    from .distributed import feature_shapes
    from .observation import channel_index, global_index
    old_board, old_global = feature_shapes(2, obs_version)
    new_board, new_global = feature_shapes(players, obs_version)
    return network.widen_inputs(params, old_board, new_board, old_global, new_global, groups=players if critic else 1,
                                board_index=channel_index(2, obs_version, players, obs_version),
                                global_index=global_index(2, obs_version, players, obs_version))


def convert(policy, critic, *, players=4, size=18, extra_blocks=0, global_pool=False, seed=0, obs_version=1):
    """(policy, critic or None) of a two-seat checkpoint -> the four-seat starting point: widened inputs, the
    board grown to `size` and `extra_blocks` identity residual blocks (with global pooling) added to both."""
    key = jax.random.key(seed)
    out = []
    for k, (net, is_critic) in enumerate(((policy, False), (critic, True))):
        if net is None:
            out.append(None)
            continue
        if 'extra_blocks' in net and (extra_blocks or global_pool):
            raise ValueError('the parent already has extra blocks')
        wide = widen_to_players(net, players, critic=is_critic, obs_version=obs_version)
        if extra_blocks or size * size > wide['parameter_embedding'].shape[1]:
            wide = network.grow(wide, jax.random.fold_in(key, k), size=size, extra_blocks=extra_blocks,
                                global_pool=global_pool)
        out.append(wide)
    return tuple(out)


def old_part(features, players=4, groups=1, obs_version=1):
    """The two-seat part of `players`-seat features (each of `groups` concatenated views, as encode_view computes
    them: the newest observation version): what a two-seat network of version `obs_version` reads from them. Used
    to check widen_to_players."""
    from .distributed import feature_shapes
    from .observation import OBS_VERSION, channel_index, global_index
    stride, gstride = feature_shapes(players, OBS_VERSION)   # the features are the newest version
    bi, gi = channel_index(2, obs_version, players), global_index(2, obs_version, players)
    board = jnp.concatenate([jnp.take(features.board, k * stride + bi, axis=-1) for k in range(groups)], -1)
    glob = jnp.concatenate([jnp.take(features.global_state, k * gstride + gi, axis=-1) for k in range(groups)], -1)
    return Features(board, glob)
