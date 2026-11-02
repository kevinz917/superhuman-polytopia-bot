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


def widen_to_players(params, players, *, critic=False):
    """Two-seat policy (or omniscient critic with critic=True) -> the same function on `players`-seat features."""
    from .distributed import feature_shapes
    old_board, old_global = feature_shapes(2)
    new_board, new_global = feature_shapes(players)
    return network.widen_inputs(params, old_board, new_board, old_global, new_global, groups=players if critic else 1)


def convert(policy, critic, *, players=4, size=18, extra_blocks=0, global_pool=False, seed=0):
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
        wide = widen_to_players(net, players, critic=is_critic)
        if extra_blocks or size * size > wide['parameter_embedding'].shape[1]:
            wide = network.grow(wide, jax.random.fold_in(key, k), size=size, extra_blocks=extra_blocks,
                                global_pool=global_pool)
        out.append(wide)
    return tuple(out)


def old_part(features, players=4, groups=1):
    """The two-seat part of `players`-seat features (each of `groups` concatenated views): what a two-seat network
    reads from them. Used to check widen_to_players."""
    from .distributed import feature_shapes
    old_board, old_global = feature_shapes(2)
    new_board, new_global = feature_shapes(players)
    board = jnp.concatenate([features.board[..., k * new_board:k * new_board + old_board] for k in range(groups)], -1)
    glob = jnp.concatenate([features.global_state[..., k * new_global:k * new_global + old_global]
                            for k in range(groups)], -1)
    return Features(board, glob)
