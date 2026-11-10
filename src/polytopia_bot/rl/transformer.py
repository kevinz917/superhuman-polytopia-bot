"""Tile-token transformer; shares the CNN's pooling, heads and training interface.

Version 1 fixes four attention heads and a 4x SiLU feed-forward expansion.
Fixed 2-D sinusoidal positions retain tile geometry without a board-size-specific
learned table. Fog remains encoded by the unchanged observation encoder.

Version 2 adds a local convolution, learned relative geometry and a global-state
token. It uses the same attention/FFN blocks and heads, with a distinct checkpoint
layout. Its relative-position tables bind it to the configured board size.
"""
import functools
import os

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

from .network import PAD_STEM, PRECISION, TRUNK_DTYPE, affine_layer_norm, layer_norm

HEADS = 4
_remat = os.environ.get('POLYTOPIA_TRANSFORMER_REMAT', '1')
if _remat not in ('0', '1'):
    raise ValueError('POLYTOPIA_TRANSFORMER_REMAT must be 0 or 1')
REMATERIALIZE = _remat == '1'
# Selective rematerialization (Claude challenger, 2026-11-06), only with POLYTOPIA_TRANSFORMER_REMAT=1:
#   full   jax.checkpoint of the whole block (the original behaviour)
#   dots   keep every matmul output and the attention output; recompute only the elementwise work (norms, SiLU,
#          residual adds, casts) inside the backward pass. Same values; less memory than no remat, and the recomputed
#          ops fuse into the backward kernels that read their inputs anyway.
REMAT_POLICY = os.environ.get('POLYTOPIA_TRANSFORMER_REMAT_POLICY', 'full')
if REMAT_POLICY not in ('full', 'dots'):
    raise ValueError('POLYTOPIA_TRANSFORMER_REMAT_POLICY must be full or dots')


def _checkpointed(fn):
    if REMAT_POLICY == 'full':
        return jax.checkpoint(fn)
    policy = jax.checkpoint_policies.save_from_both_policies(
        jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        jax.checkpoint_policies.save_only_these_names('attention_out'))
    return jax.checkpoint(fn, policy=policy)
# Explicit experimental backends; the original manual path remains the default.
# Adapted from OpenClaw's isolated attention probe, commit 11f52f3.
ATTENTION = os.environ.get('POLYTOPIA_TRANSFORMER_ATTENTION', 'manual')
if ATTENTION not in ('manual', 'xla', 'cudnn-bfloat16', 'cudnn-bfloat16-batched', 'tokamax-triton-bfloat16',
                     'cudnn-bfloat16-shared', 'xla-shared'):
    raise ValueError('POLYTOPIA_TRANSFORMER_ATTENTION must be a registered backend')
BATCHED_ATTENTION = ATTENTION == 'cudnn-bfloat16-batched'
ACTIVATION_DTYPE = jnp.dtype(os.environ.get('POLYTOPIA_TRANSFORMER_ACTIVATION_DTYPE', 'float32'))
if ACTIVATION_DTYPE not in (jnp.dtype('float32'), jnp.dtype('bfloat16')):
    raise ValueError('transformer activation dtype must be float32 or bfloat16')
if ACTIVATION_DTYPE == jnp.dtype('bfloat16') and not (BATCHED_ATTENTION and TRUNK_DTYPE == jnp.bfloat16):
    raise ValueError('bfloat16 activation storage requires the batched bfloat16 trunk')


def grow(params: dict, key: jax.Array, *, size: int, blocks: int) -> dict:
    """Append identity residual blocks at fixed width without changing old tensors.

    The two output projections start at zero; their input projections remain random
    so gradients can first train the outputs, then the attention/FFN inputs.
    Optimizer state and the separate critic must be migrated by the caller too.
    """
    from . import network

    previous = params.get('transformer', ())
    if not previous or params.get('blocks') or params.get('extra_blocks') or 'global_token' in params:
        raise ValueError('growth requires an unmixed transformer-v1 checkpoint')
    if type(blocks) is not int or blocks <= len(previous):
        raise ValueError('blocks must exceed the existing transformer depth')
    width = params['stem'].shape[-1]
    board = params['stem'].shape[2]
    glob = params['global']['w'].shape[0] - 2 * width
    initialized = network.init(key, board, glob, size=size, width=width,
                               blocks=blocks - len(previous), architecture='transformer-v1')
    added = []
    for block_params in initialized['transformer']:
        added.append(dict(block_params,
                          out=jax.tree.map(jnp.zeros_like, block_params['out']),
                          ff_out=jax.tree.map(jnp.zeros_like, block_params['ff_out'])))
    return dict(params, transformer=tuple(previous) + tuple(added))


def positions(height: int, width: int, channels: int) -> jax.Array:
    """Row/column sine and cosine features in row-major tile order."""
    frequency = 10000. ** (-jnp.arange(channels // 4, dtype=jnp.float32) / (channels // 4))
    row, col = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing='ij')
    row, col = row.reshape(-1, 1) * frequency, col.reshape(-1, 1) * frequency
    return jnp.concatenate((jnp.sin(row), jnp.cos(row), jnp.sin(col), jnp.cos(col)), axis=-1)


def linear(x: jax.Array, layer: dict) -> jax.Array:
    return (jnp.matmul(x.astype(TRUNK_DTYPE), layer['w'].astype(TRUNK_DTYPE), precision=PRECISION)
            .astype(jnp.float32) + layer['b'])


def attention(q: jax.Array, k: jax.Array, v: jax.Array,
              bias: jax.Array | None = None) -> jax.Array:
    """Attention on [heads, tokens, head_width], preserving the manual default."""
    if ATTENTION == 'manual':
        scores = jnp.matmul(q, k.swapaxes(-1, -2), precision=PRECISION) / jnp.sqrt(q.shape[-1])
        if bias is not None:
            scores = scores + bias
        return jnp.matmul(jax.nn.softmax(scores, axis=-1), v, precision=PRECISION)

    if ATTENTION == 'tokamax-triton-bfloat16':
        # Reuse the independently screened adapter from owner commit 8232ab2.
        # Preserve the real 197 tokens, learned-bias VJP and 32-channel heads.
        import tokamax
        q, k, v = (a.transpose(1, 0, 2).astype(jnp.bfloat16) for a in (q, k, v))
        out = tokamax.dot_product_attention(q, k, v,
            bias=None if bias is None else bias.astype(jnp.bfloat16),
            scale=q.shape[-1] ** -.5, is_causal=False, precision=PRECISION,
            implementation='triton')
        return out.transpose(1, 0, 2).astype(jnp.float32)

    if ATTENTION in ('cudnn-bfloat16-shared', 'xla-shared'):
        return _shared_attention()(q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2),
                                   bias).transpose(1, 0, 2)
    fused = ATTENTION in ('cudnn-bfloat16', 'cudnn-bfloat16-batched')
    dtype = jnp.bfloat16 if fused else jnp.float32
    q, k, v = (a.transpose(1, 0, 2).astype(dtype) for a in (q, k, v))
    tokens = q.shape[0]
    bias = None if bias is None else bias[None].astype(dtype)
    mask = None
    if bias is not None and tokens % 2:
        # JAX 0.11.2 cuDNN biased backward requires even sequence lengths. Keep
        # the real global token; mask only the appended key and discard its query.
        q, k, v = (jnp.pad(a, ((0, 1), (0, 0), (0, 0))) for a in (q, k, v))
        bias = jnp.pad(bias, ((0, 0), (0, 0), (0, 1), (0, 1)))
        mask = (jnp.arange(tokens + 1) < tokens)[None, None, None, :]
    with jax.default_matmul_precision(PRECISION.name.lower()):
        out = jax.nn.dot_product_attention(q, k, v, bias=bias, mask=mask,
                                           implementation='cudnn' if fused else 'xla')
    return out[:tokens].transpose(1, 0, 2).astype(jnp.float32)


def attention_batch(q: jax.Array, k: jax.Array, v: jax.Array,
                    bias: jax.Array) -> jax.Array:
    """Explicit [batch, tokens, heads, width] attention with ONE learned bias.

    An outer vmap around cuDNN attention broadcasts its bias to the batch before
    the primitive. Keeping the batch here lets cuDNN receive [1, heads, T, T],
    and its ordinary reverse-mode rule sums the shared bias gradient.
    """
    if ATTENTION == 'tokamax-triton-bfloat16':
        raise ValueError('Tokamax uses the original vmapped scalar attention path')
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError('batched attention requires equal rank-four Q/K/V')
    tokens = q.shape[1]
    if bias.shape != (q.shape[2], tokens, tokens):
        raise ValueError('batched attention requires one shared per-head bias')
    fused = ATTENTION in ('cudnn-bfloat16', 'cudnn-bfloat16-batched', 'cudnn-bfloat16-shared')
    dtype = jnp.bfloat16 if fused else jnp.float32
    q, k, v = (a.astype(dtype) for a in (q, k, v))
    bias = bias[None].astype(dtype)
    mask = None
    if tokens % 2:
        q, k, v = (jnp.pad(a, ((0, 0), (0, 1), (0, 0), (0, 0))) for a in (q, k, v))
        bias = jnp.pad(bias, ((0, 0), (0, 0), (0, 1), (0, 1)))
        mask = (jnp.arange(tokens + 1) < tokens)[None, None, None, :]
    with jax.default_matmul_precision(PRECISION.name.lower()):
        out = jax.nn.dot_product_attention(q, k, v, bias=bias, mask=mask,
                                           implementation='cudnn' if fused else 'xla')
    return out[:, :tokens].astype(jnp.float32)


@functools.lru_cache(maxsize=None)
def _shared_attention():
    """Per-example attention [tokens, heads, width] whose vmap rule runs the whole vmapped batch through ONE
    attention_batch call with the shared [heads, T, T] bias (Claude challenger, 2026-11-06).

    Callers keep their per-example code (jax.vmap(trunk) in the learner loss, the value pass, the actors and the
    GPU-resident trainer); under vmap, cuDNN receives a [1, heads, T, T] bias and sums its gradient over the batch
    in-kernel, instead of the default batch rule's broadcast bias and dense per-example [B, heads, T, T] gradient.
    Same arithmetic as cudnn-bfloat16-batched; no caller restructuring. A per-example (batched) bias falls back to
    the per-example broadcast path."""
    @jax.custom_batching.custom_vmap
    def attend(q, k, v, bias):
        return attention_batch(q[None], k[None], v[None], bias)[0]

    @attend.def_vmap
    def rule(axis_size, in_batched, q, k, v, bias):
        q, k, v = (a if b else jnp.broadcast_to(a, (axis_size,) + a.shape) for a, b in zip((q, k, v), in_batched))
        if in_batched[3]:
            return jax.vmap(lambda a, b, c, d: attention_batch(a[None], b[None], c[None], d)[0])(q, k, v, bias), True
        return attention_batch(q, k, v, bias), True
    return attend


def block_batch(x: jax.Array, params: dict, attention_bias: jax.Array) -> jax.Array:
    """Experimental batched block; masters and all norm accumulation stay FP32."""
    def project(a, weights):
        return linear(a, weights).astype(ACTIVATION_DTYPE)

    n = params['attention_norm']
    h = affine_layer_norm(x.astype(jnp.float32), n['scale'], n['bias'])
    q, k, v = jnp.split(project(h, params['qkv']), 3, axis=-1)
    shape = (*x.shape[:-1], HEADS, x.shape[-1] // HEADS)
    attended = attention_batch(*(a.reshape(shape) for a in (q, k, v)), attention_bias)
    x = (x.astype(jnp.float32) + linear(attended.reshape(x.shape), params['out'])).astype(ACTIVATION_DTYPE)
    n = params['ff_norm']
    h = affine_layer_norm(x.astype(jnp.float32), n['scale'], n['bias'])
    h = jax.nn.silu(project(h, params['ff_in']).astype(jnp.float32)).astype(ACTIVATION_DTYPE)
    return (x.astype(jnp.float32) + linear(h, params['ff_out'])).astype(ACTIVATION_DTYPE)


def block(x: jax.Array, params: dict, attention_bias: jax.Array | None = None) -> jax.Array:
    n = params['attention_norm']
    h = affine_layer_norm(x, n['scale'], n['bias'])
    q, k, v = jnp.split(linear(h, params['qkv']), 3, axis=-1)
    shape = (x.shape[0], HEADS, x.shape[-1] // HEADS)
    q, k, v = (a.reshape(shape).transpose(1, 0, 2) for a in (q, k, v))
    attended = attention(q, k, v, attention_bias)
    attended = checkpoint_name(attended, "attention_out")
    attended = attended.transpose(1, 0, 2).reshape(x.shape)
    x = x + linear(attended, params['out'])
    n = params['ff_norm']
    h = affine_layer_norm(x, n['scale'], n['bias'])
    return x + linear(jax.nn.silu(linear(h, params['ff_in'])), params['ff_out'])


def relative_attention_bias(table: jax.Array, height: int, width: int) -> jax.Array:
    """Per-head tile bias indexed by query-minus-key offsets; global edges are zero.

    Check the table before gathering: JAX can otherwise silently clip indices
    when a checkpoint is used on a different board size.
    """
    if height < 1 or width < 1 or table.shape != (HEADS, 2 * height - 1, 2 * width - 1):
        raise ValueError('relative-position table does not match the board size')
    tiles = jnp.arange(height * width)
    row, col = tiles // width, tiles % width
    dy = row[:, None] - row[None, :] + height - 1
    dx = col[:, None] - col[None, :] + width - 1
    return jnp.pad(table[:, dy, dx], ((0, 0), (0, 1), (0, 1)))


def _padded_stem(board: jax.Array, stem: jax.Array) -> tuple:
    """POLYTOPIA_PAD_STEM=k (network.PAD_STEM): zero-pad the stem's input channels (board and kernel rows) to a
    multiple of k. The padded channels are zero, so the convolution's value is unchanged; 138 / 276 channels (policy /
    omniscient critic) otherwise keep cuDNN off its tensor-core kernels (Claude challenger, 2026-11-06)."""
    if PAD_STEM and stem.shape[2] % PAD_STEM:
        stem = jnp.pad(stem, ((0, 0), (0, 0), (0, PAD_STEM - stem.shape[2] % PAD_STEM), (0, 0)))
    if board.shape[-1] < stem.shape[2]:   # main's network.trunk may already have padded the board (its CNN path)
        board = jnp.pad(board, ((0, 0),) * (board.ndim - 1) + ((0, stem.shape[2] - board.shape[-1]),))
    return board, stem


def encode_v2(params: dict, board: jax.Array, global_state: jax.Array | None) -> jax.Array:
    """Full-board attention over local tile features and one visible-state token."""
    height, width, _ = board.shape
    if global_state is None or global_state.shape != (params['global_token']['w'].shape[0],):
        raise ValueError('transformer-v2 requires the matching global-state features')
    board, stem = _padded_stem(board, params['stem'])
    x = jax.lax.conv_general_dilated(board[None].astype(TRUNK_DTYPE), stem.astype(TRUNK_DTYPE),
                                   (1, 1), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                   precision=PRECISION)[0].astype(jnp.float32)
    x = jax.nn.silu(layer_norm(x)).reshape(height * width, -1)
    # The observation already contains absolute x/y coordinates. Relative bias
    # supplies displacement; v2 does not add v1's fixed sinusoidal positions.
    context = jax.nn.silu(layer_norm(linear(global_state, params['global_token'])))
    x = jnp.concatenate((x, context[None]), axis=0)
    for weights in params['transformer']:
        bias = relative_attention_bias(weights['relative_bias'], height, width)
        x = (_checkpointed(block) if REMATERIALIZE else block)(x, weights, bias)
    # The context token is not an action tile and must not enter spatial pooling.
    return jax.nn.silu(layer_norm(x[:-1])).reshape(height, width, -1)


def encode_v2_batch(params: dict, board: jax.Array, global_state: jax.Array) -> jax.Array:
    """V2 with an explicit example axis; pooling/heads receive FP32 tile features."""
    if board.ndim != 4 or global_state.shape != (board.shape[0], params['global_token']['w'].shape[0]):
        raise ValueError('batched transformer-v2 requires matching board and global features')
    batch, height, width, _ = board.shape
    board, stem = _padded_stem(board, params['stem'])
    x = jax.lax.conv_general_dilated(board.astype(TRUNK_DTYPE), stem.astype(TRUNK_DTYPE),
                                   (1, 1), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                   precision=PRECISION).astype(jnp.float32)
    x = jax.nn.silu(layer_norm(x)).reshape(batch, height * width, -1).astype(ACTIVATION_DTYPE)
    context = jax.nn.silu(layer_norm(linear(global_state, params['global_token']))).astype(ACTIVATION_DTYPE)
    x = jnp.concatenate((x, context[:, None]), axis=1)
    for weights in params['transformer']:
        bias = relative_attention_bias(weights['relative_bias'], height, width)
        x = (_checkpointed(block_batch) if REMATERIALIZE else block_batch)(x, weights, bias)
    return jax.nn.silu(layer_norm(x[:, :-1].astype(jnp.float32))).reshape(batch, height, width, -1)


def encode(params: dict, board: jax.Array, global_state: jax.Array | None = None) -> jax.Array:
    if 'global_token' in params:
        return encode_v2(params, board, global_state)
    height, width, channels = board.shape
    # Stem is stored as [1,1,in,out] for feature-width inspection compatibility;
    # this is a per-tile linear projection, with no spatial convolution.
    stem = params['stem'][0, 0]
    x = jnp.matmul(board.reshape(-1, channels).astype(TRUNK_DTYPE), stem.astype(TRUNK_DTYPE),
                   precision=PRECISION).astype(jnp.float32)
    x = x + positions(height, width, stem.shape[-1])
    for weights in params['transformer']:
        # Recompute activations on backward to reduce large PPO minibatch memory.
        x = (_checkpointed(block) if REMATERIALIZE else block)(x, weights)
    return jax.nn.silu(layer_norm(x)).reshape(height, width, stem.shape[-1])
