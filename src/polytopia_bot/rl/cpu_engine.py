"""Packed public views and the policy features computed from them (the engine's game stepping is engine-rs).

The name is historical: this module held the JAX actor engine (CpuGames), retired on 2026-10-27 with the research
rules (docs/rules-alignment-2026-10-27.md). What remains is rules-free: the packed int16 view layout engine-rs writes
(view_layout), its unpacking and the feature encoding the actors and the learner share (view_features), and the
Prepared / Tick records RustGames returns. The view schema is still read from the JAX observation function
(gpu.kernels.observe), which engine-rs's kernels::observe reproduces field for field.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from polytopia_bot.gpu import kernels, mapgen
from .observation import encode_view

# observe() fields that encode_view reads; state_version is excluded on purpose.
VIEW_EXCLUDE = ('state_version',)


class Prepared(NamedTuple):
    features: object
    masks: object
    view: object
    opp_view: object = None   # packed public view of the seat NOT to move (both_views=True); for an omniscient critic


class Tick(NamedTuple):
    states: object
    prepared: Prepared
    done: object
    terminated: object
    truncated: object
    rewards: object          # [games, players] native outcome of the finished game (rank rewards with > 2 seats), else 0
    final_features: object   # per-seat features of a truncated final state (bootstrap)
    final_views: object      # [games, players, width] packed public view per seat of a truncated final state, else 0


# Per-seat relation fields of the view: one entry per seat (engine-rs kernels::observe).
RELATION_FIELDS = ('embassies_received', 'embassies_sent', 'known_players', 'peace', 'peace_offers')


def view_layout(size, players=2):
    """(name, shape) of every packed public-view field, in packing order. players > 2 (engine-rs only): the
    relation fields have one entry per seat and 'alive' (one per seat) is added."""
    template = jax.eval_shape(lambda: kernels.observe(mapgen.reset(jnp.int32(0), size=size), jnp.int32(1)))
    fields = {k: tuple(v.shape) for k, v in template.items() if k not in VIEW_EXCLUDE}
    if players > 2:
        fields.update({k: (players,) for k in RELATION_FIELDS}, alive=(players,))
    return tuple(sorted(fields.items()))


def pack_view(view, layout):
    return jnp.concatenate([jnp.ravel(view[k]).astype(jnp.int16) for k, _ in layout])


def unpack_view(packed, layout):
    """Inverse of pack_view for one sample (vmap for batches); values are exact ints."""
    out, at = {}, 0
    for k, shape in layout:
        n = int(np.prod(shape))
        out[k] = packed[at:at + n].reshape(shape).astype(jnp.int32)
        at += n
    return out


def view_features(packed, layout, size):
    return encode_view(unpack_view(packed, layout), size)
