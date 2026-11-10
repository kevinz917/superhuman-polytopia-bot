"""Compact residual CNN, scalar value, and autoregressive masked action heads.

Pure JAX keeps environment, inference and PPO on the same accelerator. No LLM,
PyTorch runtime, pretrained weights, or dependency on the Duper code is needed.
"""
import math
import os
import jax
import jax.numpy as jnp

from .actions import Choice, PathMasks, family_mask, source_mask, target_mask
from .observation import fit_board, fit_global

# Non-spatial target families (unit, structure, tech, ...) choose among at most this many
# parameter ids; only spatial families (move, attack) target tiles, through the conv features.
PARAMETER_TARGETS = 128
# Dense/conv precision. HIGHEST (default) keeps CPU actors and GPU learner on one fp32 path;
# POLYTOPIA_MATMUL_PRECISION=DEFAULT lets the GPU use TF32 tensor cores (throughput studies only).
PRECISION = getattr(jax.lax.Precision, os.environ.get('POLYTOPIA_MATMUL_PRECISION', 'HIGHEST'))
# Head matmuls (dense layers and the target key-query product); defaults to PRECISION. Lets the conv
# torso run TF32 while the logits stay fp32.
HEAD_PRECISION = getattr(jax.lax.Precision, os.environ.get('POLYTOPIA_HEAD_PRECISION', PRECISION.name))
# Conv torso operand dtype. POLYTOPIA_TRUNK_DTYPE=bfloat16 (GPU learner only) runs the convolutions on bf16
# operands and casts each result back to fp32; weights, residual sums, norms and heads stay fp32.
TRUNK_DTYPE = jnp.dtype(os.environ.get('POLYTOPIA_TRUNK_DTYPE', 'float32'))
# Dtype the conv torso's activations are stored in between kernels (POLYTOPIA_ACT_DTYPE). float32 (default) is the
# reference path. bfloat16 keeps each convolution's output, the residual stream and the normalized activations in
# bf16 between kernels (every norm and SiLU still computes in fp32 inside its fused kernel); the actors' policy
# forward is memory-bound on those fp32 tensors (2026-11-06, scripts/bench_forward.py), so this is an actor-side
# inference option: the learner keeps fp32 and PPO's ratio corrects the behaviour policy's rounding.
ACT_DTYPE = jnp.dtype(os.environ.get('POLYTOPIA_ACT_DTYPE', 'float32'))
# POLYTOPIA_PAD_STEM=k: zero-pad the stem convolution's input channels (and its kernel's input rows) to a multiple of k.
# The padded channels are zero, so the convolution's value is unchanged; channel counts that are not a multiple of 8
# keep cuDNN off its tensor-core kernels for that layer. 0 (default): no padding.
PAD_STEM = int(os.environ.get('POLYTOPIA_PAD_STEM', '0'))
# Attention blocks (init attention): dims per head; width 128 = 4 heads.
ATTENTION_HEAD = 32
# Attention kernel (POLYTOPIA_ATTENTION_IMPL, read when the trunk is traced): 'einsum' (XLA, fp32 scores and softmax;
# the reference, and the CPU path of the arenas and the Policy wrapper) or 'sdpa' (jax.nn.dot_product_attention on
# TRUNK_DTYPE q/k/v, implementation 'cudnn': the fused kernel, 56 vs 118 ms per 4096-row policy step on an H100, max
# value difference 2e-4 on a 0.15 scale; the automatic choice fell back to XLA's unfused kernel at 86 ms). 'auto'
# (default) = sdpa/cudnn on a GPU backend, einsum elsewhere, so the learner and the actors' inference servers share
# one kernel and the CPU evaluations keep fp32 attention. Measured 2026-11-01, experiments/modal_attention_bench.py.
ATTENTION_IMPL = os.environ.get('POLYTOPIA_ATTENTION_IMPL', 'auto')
# Rematerialize each attention block under differentiation (jax.checkpoint): the backward pass recomputes the block
# instead of keeping its [tiles, heads, tiles, tiles] scores and softmax per row of the minibatch (the einsum kernel
# OOMed a 96 GB GPU without it, 61.6 GiB in one allocation; the fused kernel never materializes them).
# POLYTOPIA_ATTENTION_REMAT: '1' / '0'; default 'auto' = on for einsum, off for sdpa.
ATTENTION_REMAT = os.environ.get('POLYTOPIA_ATTENTION_REMAT', 'auto')
# Global pooling inside the trunk blocks (init block_pool), two parameter layouts (the checkpoint spec's block_pool):
#  1: farsight's (2026-11-01). Every block: the un-normalized (mean, max) over the board of the block's first
#     convolution -> dense -> added to that convolution BEFORE the block's LayerNorm; zero-initialized. On farsight the
#     bias grew to 3-5x the convolution it was added to within 1.5 B samples (its inputs are unbounded, nothing bounds
#     the dense layer's scale), so one board-wide vector dominated every tile's LayerNorm and starved the spatial path.
#  2: farsight v2. Only the blocks of block_pool_indices (the second of every pair: 1 and 3 of 4) pool, and
#     KataGo's structure sits under the normalization: the first convolution is layer-normalized per tile first,
#     BLOCK_POOL_CHANNELS of its channels are pooled (mean and max over the board), each statistic is layer-normalized
#     across channels, a dense layer maps them to a per-channel bias, and gate * bias joins the normalized tiles before
#     the SiLU. The gate is a scalar at 0 (the block computes what it computes without pooling; Adam moves it by about
#     one learning rate per update, so the bias can only grow at a bounded rate) over randomly initialized dense
#     weights, so the gate's gradient is nonzero from the first step and the dense layer's from the second.
BLOCK_POOL_CHANNELS = 32
BLOCK_POOL_LAYOUTS = (1, 2)


def block_pool_indices(blocks, layout):
    """Trunk blocks that carry a pooling layer: every block in layout 1; in layout 2 the second block of every pair
    (1 and 3 of 4), or the only block of a one-block trunk."""
    if int(layout) == 2:
        return tuple(range(1, blocks, 2)) if blocks > 1 else tuple(range(blocks))
    return tuple(range(blocks))


def init(key, board_channels, global_channels, *, size=14, width=64, blocks=2, architecture='legacy',
         extra_blocks=0, global_pool=False, block_pool=False, attention=0):
    """extra_blocks > 0 adds pre-activation residual blocks after the trunk (key 'extra_blocks'); their
    second convolution starts at zero, so they are the identity until trained (see grow). global_pool: each extra
    block also adds a per-channel bias computed from the mean and max over the board of its first convolution
    (KataGo's global pooling), so every tile sees board-wide context past the trunk's receptive field. block_pool: a
    pooling layer inside the trunk blocks (key 'blocks', a third entry of the pooled blocks), at no extra convolution
    cost; 1 / True = farsight's layout (every block, un-normalized, zero weights), 2 = the gated normalized layout
    (BLOCK_POOL_LAYOUTS above). Both start as the identity, so a block_pool network computes what the same weights
    compute without it (add_block_pool). attention > 0 appends that many pre-LayerNorm transformer blocks over the tiles after the convolutions (key
    'attention'; multi-head self-attention with ATTENTION_HEAD dims per head, then a 4x SiLU MLP) on a learned
    positional embedding ('pos', one row per tile of a `size` board), closed by a final LayerNorm ('attention_norm'):
    every tile reads every other tile directly, where the convolutions only widen the receptive field by one ring per
    layer (lookout, 2026-11-01). Needs normalized-v2."""
    from . import transformer_v3
    if transformer_v3.is_v3(architecture):
        # V3 (rl/transformer_v3.py): shared public encoder; the critic is its own tree (transformer_v3.init_critic)
        if extra_blocks or global_pool or block_pool or attention:
            raise ValueError('transformer-v3 has no CNN growth/hybrid options')
        return transformer_v3.init_policy(key, board_channels, global_channels, size=size, width=width, blocks=blocks,
                                          architecture=architecture)
    if architecture not in ('legacy', 'normalized-v2', 'transformer-v1', 'transformer-v2'):
        raise ValueError('unknown network architecture')
    # transformer-v1/v2 (rl/transformer.py, the V2 campaign's encoder at 849a931): tile tokens instead of the trunk's
    # residual convolutions; same pooled heads as normalized-v2
    transformer = architecture in ('transformer-v1', 'transformer-v2')
    if transformer and (width % 4 or width < 4 or blocks < 1 or extra_blocks or global_pool or block_pool or attention):
        raise ValueError('transformer needs width divisible by 4, blocks >= 1, and no CNN growth/hybrid options')
    def weights(shape, scale=1.):
        nonlocal key
        key, draw = jax.random.split(key)
        fan_in = math.prod(shape[:-1])
        return jax.random.normal(draw, shape) * (scale / math.sqrt(fan_in))
    def linear(a, b, scale=1.):
        return {"w": weights((a, b), scale), "b": jnp.zeros(b)}
    layout = int(block_pool or 0)
    if layout and layout not in BLOCK_POOL_LAYOUTS:
        raise ValueError('unknown block_pool layout')
    if layout == 2 and architecture != 'normalized-v2':
        raise ValueError('block_pool layout 2 needs the normalized-v2 architecture')
    pooled = set(block_pool_indices(blocks, layout)) if layout else set()
    def pool_layer(index):
        if index not in pooled:
            return ()
        if layout == 1:
            return ({'w': jnp.zeros((2 * width, width)), 'b': jnp.zeros(width)},)
        c = min(BLOCK_POOL_CHANNELS, width)
        return ({'w': weights((2 * c, width)), 'b': jnp.zeros(width), 'gate': jnp.zeros(())},)
    params = {
        "stem": weights((1, 1, board_channels, width) if architecture == 'transformer-v1'
                        else (3, 3, board_channels, width), math.sqrt(2)),
        "blocks": (() if transformer else tuple((weights((3, 3, width, width), math.sqrt(2)),
                         weights((3, 3, width, width))) + pool_layer(i) for i in range(blocks))),
        "global": linear(global_channels + width, width),
        "family": linear(width, 17, .01),
        "source": linear(width, 17, .01),
        "query": linear(2 * width, width),
        "family_embedding": weights((17, width), .1),
        "parameter_embedding": weights((17, max(size * size, 128), width), .1),
        "value": linear(width, 1, .1),
    }
    if architecture in ('normalized-v2', 'transformer-v1', 'transformer-v2'):
        # Versioned tensor layout keeps old checkpoints loadable without silently
        # interpreting their weights under a different forward pass.
        params['global'] = linear(global_channels + 2 * width, width)
        params['norm'] = {'scale': jnp.ones(width), 'bias': jnp.zeros(width)}
        params['critic_hidden'] = linear(global_channels + 2 * width, width)
    if transformer:
        params['transformer'] = tuple(dict(qkv=linear(width, 3 * width), out=linear(width, width),
            ff_in=linear(width, 4 * width, math.sqrt(2)), ff_out=linear(4 * width, width),
            attention_norm=dict(scale=jnp.ones(width), bias=jnp.zeros(width)),
            ff_norm=dict(scale=jnp.ones(width), bias=jnp.zeros(width))) for _ in range(blocks))
    if architecture == 'transformer-v2':
        params['global_token'] = linear(global_channels, width)
        params['transformer'] = tuple(dict(block, relative_bias=jnp.zeros((4, 2 * size - 1, 2 * size - 1)))
                                      for block in params['transformer'])
    if extra_blocks:
        params['extra_blocks'] = tuple((weights((3, 3, width, width), math.sqrt(2)), jnp.zeros((3, 3, width, width)))
                                       + ((linear(2 * width, width),) if global_pool else ())
                                       for _ in range(extra_blocks))
    if attention:
        if architecture != 'normalized-v2':
            raise ValueError('attention blocks need the normalized-v2 architecture')
        if width % ATTENTION_HEAD:
            raise ValueError(f'width must be a multiple of {ATTENTION_HEAD} for attention blocks')
        norm = lambda: {'scale': jnp.ones(width), 'bias': jnp.zeros(width)}
        # residual branches (proj, mlp_out) start small (GPT-2's 1/sqrt(2 * depth)) so the residual stream keeps the
        # convolutional features' scale at init
        branch = 1. / math.sqrt(2 * attention)
        params['pos'] = weights((size * size, width), .1 * math.sqrt(size * size))   # ~N(0, .1^2) per entry
        params['attention'] = tuple(dict(norm1=norm(), qkv=linear(width, 3 * width), proj=linear(width, width, branch),
                                         norm2=norm(), mlp_in=linear(width, 4 * width, math.sqrt(2)),
                                         mlp_out=linear(4 * width, width, branch))
                                    for _ in range(attention))
        params['attention_norm'] = norm()
    return params


def grow(params, key, *, size, extra_blocks, global_pool=False):
    """Larger-board network that computes exactly what `params` computes.

    The conv trunk, pooling and heads are board-size independent. parameter_embedding is
    [17, max(size^2, 128), width] only for checkpoint shape compatibility: target_logits reads
    its first PARAMETER_TARGETS rows, so new rows are zero padding. New residual blocks have a
    zero second kernel (identity at init); their first kernel (and global-pooling layer) is freshly initialized."""
    if 'extra_blocks' in params:
        raise ValueError('network already has extra blocks')
    emb = params['parameter_embedding']
    rows = max(size * size, PARAMETER_TARGETS)
    if rows < emb.shape[1]:
        raise ValueError('cannot shrink the parameter embedding')
    width = emb.shape[-1]
    out = dict(params, parameter_embedding=jnp.pad(emb, ((0, 0), (0, rows - emb.shape[1]), (0, 0))))
    if extra_blocks:
        keys = jax.random.split(key, extra_blocks)
        scale = math.sqrt(2) / math.sqrt(9 * width)
        pool = lambda k: ({'w': jax.random.normal(jax.random.fold_in(k, 1), (2 * width, width)) / math.sqrt(2 * width),
                           'b': jnp.zeros(width)},)
        out['extra_blocks'] = tuple((jax.random.normal(k, (3, 3, width, width)) * scale,
                                     jnp.zeros((3, 3, width, width))) + (pool(k) if global_pool else ()) for k in keys)
    return out


def add_block_pool(params, layout=1, key=None):
    """The same network with global pooling in its trunk blocks (init block_pool, parameter layout `layout`): it
    computes exactly what `params` computes (layout 1: zero weights in every block; layout 2: a zero gate over dense
    weights drawn from `key` in the blocks of block_pool_indices), and the new layers get gradients from the first
    step (layout 2: the gate first, the dense layer once the gate has moved)."""
    if any(len(block) != 2 for block in params['blocks']):
        raise ValueError('network already has trunk-block pooling')
    if layout not in BLOCK_POOL_LAYOUTS:
        raise ValueError('unknown block_pool layout')
    if layout == 2 and 'norm' not in params:
        raise ValueError('block_pool layout 2 needs a normalized-v2 network')
    width = params['stem'].shape[-1]
    pooled = set(block_pool_indices(len(params['blocks']), layout))
    key = jax.random.key(0) if key is None else key
    def layer(i, a):
        if i not in pooled:
            return ()
        if layout == 1:
            return ({'w': jnp.zeros((2 * width, width), a.dtype), 'b': jnp.zeros(width, a.dtype)},)
        c = min(BLOCK_POOL_CHANNELS, width)
        w = jax.random.normal(jax.random.fold_in(key, i), (2 * c, width), a.dtype) / math.sqrt(2 * c)
        return ({'w': w, 'b': jnp.zeros(width, a.dtype), 'gate': jnp.zeros((), a.dtype)},)
    return dict(params, blocks=tuple((a, b) + layer(i, a) for i, (a, b) in enumerate(params['blocks'])))


def widen_inputs(params, old_board, new_board, old_global, new_global, groups=1, board_index=None, global_index=None):
    """A network on wider features that computes exactly what `params` computes on the old ones.

    The input is `groups` concatenated feature groups (1 for the policy; the omniscient critic's own view plus
    the other seats' views) of which `params` saw the first min(groups, old groups) at old widths. New features
    sit after the old ones inside each group (observation.encode_view: a newer observation version), so the stem
    and the pooled-input layers ('global', 'critic_hidden', normalized-v2) keep their old rows in place and get
    zero rows for every new feature and for each new group. Where the old features are not a prefix of the new
    group (a two-seat network widened to more seats: observation.channel_index), `board_index` / `global_index`
    give each old feature's position inside the new group."""
    if 'norm' not in params:
        raise ValueError('widen_inputs needs a normalized-v2 network')
    stem = params['stem']
    old_groups = stem.shape[2] // old_board
    if stem.shape[2] != old_groups * old_board or old_groups > groups:
        raise ValueError('stem width does not match the old feature groups')
    def spread(w, axis, old, new, index, prefix=0):
        index = jnp.arange(old) if index is None else jnp.asarray(index)
        if index.shape != (old,) or (old and int(index.max()) >= new):
            raise ValueError('feature index does not match the old and new widths')
        parts = [jnp.take(w, jnp.arange(prefix), axis=axis)] if prefix else []
        for k in range(groups):
            shape = w.shape[:axis] + (new,) + w.shape[axis + 1:]
            block = jnp.zeros(shape, w.dtype)
            if k < old_groups:
                rows = jnp.take(w, prefix + k * old + jnp.arange(old), axis=axis)
                block = jnp.moveaxis(jnp.moveaxis(block, axis, 0).at[index].set(jnp.moveaxis(rows, axis, 0)), 0, axis)
            parts.append(block)
        return jnp.concatenate(parts, axis)
    out = dict(params, stem=spread(stem, 2, old_board, new_board, board_index))
    width = stem.shape[-1]
    for name in ('global', 'critic_hidden'):
        if name in params:
            out[name] = dict(params[name], w=spread(params[name]['w'], 0, old_global, new_global, global_index,
                                                   prefix=2 * width))
    return out


def dense(x, layer):
    # Keep rollout and differently sized learner minibatches on the same FP32
    # arithmetic path. GPU TF32 algorithm choices otherwise shift old logp.
    return jnp.matmul(x, layer["w"], precision=HEAD_PRECISION) + layer["b"]


def _rms(x):
    return jnp.sqrt(jnp.mean(x ** 2))


def trunk(params, features, pool_stats=None):
    """Tile features [tiles, width], the global vector g and the value. pool_stats: a list; every pooling layer
    appends dict(conv_rms, bias_rms, gate) of this position (the RMS of the tiles the bias is added to, of the bias,
    and the gate, None in layout 1): the farsight diagnosis measured bias_rms / conv_rms at 3-5x (see BLOCK_POOL_LAYOUTS);
    the tiles returned are the same with or without it."""
    f32 = jnp.float32
    def conv(x, kernel):
        return jax.lax.conv_general_dilated(x[None].astype(TRUNK_DTYPE), kernel.astype(TRUNK_DTYPE), (1, 1), "SAME",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"), precision=PRECISION)[0].astype(ACT_DTYPE)
    normalized = 'norm' in params
    if ACT_DTYPE == f32:
        activate = (lambda z: jax.nn.silu(layer_norm(z))) if normalized else jax.nn.relu
    else:
        activate = ((lambda z: jax.nn.silu(layer_norm(z.astype(f32))).astype(ACT_DTYPE)) if normalized else
                    (lambda z: jax.nn.relu(z)))
    # the features are always the newest observation version; an older network reads its own channels of them
    if 'v3_stem' in params:
        from . import transformer_v3
        return transformer_v3.trunk(params, features)
    board, stem = fit_board(features.board, params["stem"].shape[2]), params["stem"]
    if PAD_STEM and stem.shape[2] % PAD_STEM:
        extra = PAD_STEM - stem.shape[2] % PAD_STEM
        board = jnp.pad(board, ((0, 0), (0, 0), (0, extra)))
        stem = jnp.pad(stem, ((0, 0), (0, 0), (0, extra), (0, 0)))
    if 'transformer' in params:
        from .transformer import encode
        glob_in = (fit_global(features.global_state, params['global_token']['w'].shape[0])
                   if 'global_token' in params else None)
        x = encode(params, board, glob_in)
    else:
        x = activate(conv(board, stem))
    def branch(h, block):
        """A block's first convolution output -> the activated input of its second convolution, through the block's
        pooling layer when it has one (block[2]): layout 1 and the extra blocks' global_pool add the bias of the
        un-normalized statistics before the norm; layout 2 ('gate') adds the gated bias of the normalized statistics
        of the first BLOCK_POOL_CHANNELS normalized channels after it (BLOCK_POOL_LAYOUTS)."""
        if len(block) == 2:
            return activate(h)
        h = h.astype(f32)   # pooling layers run in fp32 (their outputs go back to ACT_DTYPE below)
        layer = block[2]
        if 'gate' not in layer:
            flat = h.reshape(-1, h.shape[-1])
            bias = dense(jnp.concatenate((flat.mean(0), flat.max(0))), layer)
            if pool_stats is not None:
                pool_stats.append(dict(conv_rms=_rms(h), bias_rms=_rms(bias), gate=None))
            return activate((h + bias).astype(ACT_DTYPE))
        hn = layer_norm(h)
        sub = hn.reshape(-1, hn.shape[-1])[:, :layer['w'].shape[0] // 2]
        pooled = jnp.concatenate((layer_norm(sub.mean(0)), layer_norm(sub.max(0))))
        bias = layer['gate'] * dense(pooled, layer)
        if pool_stats is not None:
            pool_stats.append(dict(conv_rms=_rms(hn), bias_rms=_rms(bias), gate=layer['gate']))
        return jax.nn.silu(hn + bias).astype(ACT_DTYPE)
    for block in params["blocks"]:
        x = activate(x + conv(branch(conv(x, block[0]), block), block[1]))
    for block in params.get("extra_blocks", ()):
        x = x + conv(branch(conv(x, block[0]), block), block[1])
    tiles = x.reshape(-1, x.shape[-1]).astype(f32)
    if 'attention' in params:
        tiles = attend(params, tiles)
    # likewise the global vector: an older network reads its own entries (none of the version-3 enemy techs)
    width = tiles.shape[-1]
    glob = fit_global(features.global_state, params['global']['w'].shape[0] - (2 if normalized else 1) * width)
    if normalized:
        pooled = jnp.concatenate((tiles.mean(0), tiles.max(0), glob))
        g = jax.nn.silu(layer_norm(dense(pooled, params['global'])) * params['norm']['scale'] + params['norm']['bias'])
        critic = jax.nn.silu(layer_norm(dense(pooled, params['critic_hidden'])))
        return tiles, g, dense(critic, params['value'])[0]
    g = jnp.tanh(dense(jnp.concatenate((tiles.mean(0), glob)), params["global"]))
    return tiles, g, dense(g, params["value"])[0]


def layer_norm(x):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(jnp.mean(centered ** 2, axis=-1, keepdims=True) + 1e-5)


# Normalization backend of the transformer blocks (POLYTOPIA_LAYER_NORM; V2 campaign source 849a931): 'original' is the
# expression above; 'tokamax-affine' an opt-in native FP32 affine kernel (needs the tokamax package).
LAYER_NORM = os.environ.get('POLYTOPIA_LAYER_NORM', 'original')
if LAYER_NORM not in ('original', 'tokamax-affine'):
    raise ValueError('POLYTOPIA_LAYER_NORM must select a registered normalization backend')


def affine_layer_norm(x, scale, bias):
    if LAYER_NORM != 'tokamax-affine':
        return layer_norm(x) * scale + bias
    import tokamax
    return tokamax.layer_norm(x, scale, bias, axis=-1, epsilon=1e-5, scale_offset=0., subtract_mean=True,
                              implementation='triton')


def _affine_norm(x, norm):
    return layer_norm(x) * norm['scale'] + norm['bias']


def _trunk_matmul(x, layer):
    """dense on TRUNK_DTYPE operands (the conv torso's precision policy): bf16 inputs and weights, fp32 result."""
    return (jnp.matmul(x.astype(TRUNK_DTYPE), layer['w'].astype(TRUNK_DTYPE), precision=PRECISION).astype(jnp.float32)
            + layer['b'])


def _attention_block(x, block, impl='einsum', head=ATTENTION_HEAD, skip='', sdpa_impl=None):
    n, width = x.shape
    heads = width // head
    qkv = _trunk_matmul(_affine_norm(x, block['norm1']), block['qkv']).reshape(n, 3, heads, head)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                                   # [n, heads, d]
    if impl == 'sdpa':
        mixed = jax.nn.dot_product_attention(q[None].astype(TRUNK_DTYPE), k[None].astype(TRUNK_DTYPE),
                                             v[None].astype(TRUNK_DTYPE), implementation=sdpa_impl)[0]
        mixed = mixed.astype(jnp.float32).reshape(n, width)
    else:
        scores = jnp.einsum('qhd,khd->hqk', q, k, precision=PRECISION) / math.sqrt(head)
        weights = jax.nn.softmax(scores, axis=-1)
        mixed = jnp.einsum('hqk,khd->qhd', weights, v, precision=PRECISION).reshape(n, width)
    x = x if skip == 'attn' else x + _trunk_matmul(mixed, block['proj'])   # skip: benchmarking only
    if skip == 'mlp':
        return x
    return x + _trunk_matmul(jax.nn.silu(_trunk_matmul(_affine_norm(x, block['norm2']), block['mlp_in'])), block['mlp_out'])


def attend(params, tiles, remat=None):
    """Pre-LayerNorm transformer blocks over the tile features [tiles, width] (init attention): tiles + positional
    embedding, then per block x += proj(MHSA(norm1(x))) and x += mlp(norm2(x)); the output is attention_norm(x). The
    softmax runs in fp32; the matmuls follow the conv torso's TRUNK_DTYPE. A board smaller than the embedding (a
    grown parameter table) reads the embedding's first rows. remat (default ATTENTION_REMAT): each block is a
    jax.checkpoint, so its gradients recompute the block's activations; the values are the same either way."""
    n = tiles.shape[0]
    import functools
    impl = os.environ.get('POLYTOPIA_ATTENTION_IMPL', ATTENTION_IMPL)
    if impl == 'auto':
        impl = 'sdpa' if jax.default_backend() == 'gpu' else 'einsum'
    sdpa_impl = os.environ.get('POLYTOPIA_SDPA_IMPL') or ('cudnn' if jax.default_backend() == 'gpu' else None)
    if remat is None:
        flag = os.environ.get('POLYTOPIA_ATTENTION_REMAT', ATTENTION_REMAT)
        remat = (impl == 'einsum') if flag == 'auto' else flag != '0'
    # the flags are read when the trunk is traced (the learner, the servers and the arenas each trace once); a
    # fresh partial per trace keeps jax.checkpoint from reusing a trace made under other flags
    block_fn = functools.partial(_attention_block, impl=impl, head=int(os.environ.get('POLYTOPIA_ATTENTION_HEAD', ATTENTION_HEAD)),
                                 skip=os.environ.get('POLYTOPIA_ATTENTION_BENCH_SKIP', ''), sdpa_impl=sdpa_impl)
    block_fn = jax.checkpoint(block_fn) if remat else block_fn
    x = tiles + params['pos'][:n]
    for block in params['attention']:
        x = block_fn(x, block)
    return _affine_norm(x, params['attention_norm'])


def target_logits(params, tiles, g, family, source):
    pre = dense(jnp.concatenate((tiles[source], g)), params["query"]) + params["family_embedding"][family]
    query = layer_norm(pre) if 'norm' in params else jnp.tanh(pre)
    # Rows past PARAMETER_TARGETS are never legal for non-spatial families; zero keys keep every
    # legal logit identical while letting one network run on any board size.
    rows = max(tiles.shape[0], PARAMETER_TARGETS)
    spatial = jnp.pad(tiles, ((0, rows - tiles.shape[0]), (0, 0)))
    embedding = jnp.pad(params["parameter_embedding"][family][:PARAMETER_TARGETS], ((0, rows - PARAMETER_TARGETS), (0, 0)))
    keys = jnp.where(family < 2, spatial, embedding)
    return jnp.matmul(keys, query, precision=HEAD_PRECISION) / jnp.sqrt(tiles.shape[-1])


def log_distribution(logits, mask):
    # Absorbed evaluation lanes have no legal moves; use a numerical sentinel
    # only there. Active-lane legality is checked by the runner.
    safe = mask.at[0].set(mask[0] | ~jnp.any(mask))
    return jax.nn.log_softmax(jnp.where(safe, logits, -1e9))


def logprob_entropy(logits, mask, selected):
    logp = log_distribution(logits, mask)
    entropy = -jnp.sum(jnp.where(mask, jnp.exp(logp) * logp, 0.))
    return logp[selected], entropy


def evaluate(params, features, choice, masks):
    tiles, g, value = trunk(params, features)
    logits = (dense(g, params["family"]), dense(tiles, params["source"])[:, choice.family],
              target_logits(params, tiles, g, choice.family, choice.source))
    pairs = [logprob_entropy(l, m, a) for l, m, a in zip(logits, masks, choice, strict=True)]
    return sum(p[0] for p in pairs), value, sum(p[1] for p in pairs)


def sample(params, features, masks, key, *, size, deterministic=False):
    tiles, g, value = trunk(params, features)
    choice, path, logp = sample_from(params, tiles, g, masks, key, size=size, deterministic=deterministic)
    return choice, path, logp, value


def sample_from(params, tiles, g, masks, key, *, size, deterministic=False):
    """sample's heads on an already-encoded state (tiles, g): (choice, path masks, logp)."""
    keys = jax.random.split(key, 3)
    def draw(logits, mask, key):
        lp = log_distribution(logits, mask)
        return jnp.argmax(lp).astype(jnp.int32) if deterministic else jax.random.categorical(key, lp).astype(jnp.int32)
    fm = family_mask(masks)
    f = draw(dense(g, params["family"]), fm, keys[0])
    sm = source_mask(masks, f, size)
    s = draw(dense(tiles, params["source"])[:, f], sm, keys[1])
    tm = target_mask(masks, f, s, size)
    t = draw(target_logits(params, tiles, g, f, s), tm, keys[2])
    choice, path = Choice(f, s, t), PathMasks(fm, sm, tm)
    # Reuse the trunk instead of doing another encoder pass.
    logits = (dense(g, params["family"]), dense(tiles, params["source"])[:, f], target_logits(params, tiles, g, f, s))
    logp = sum(logprob_entropy(l, m, a)[0] for l, m, a in zip(logits, path, choice, strict=True))
    return choice, path, logp
