"""Compact residual CNN, scalar value, and autoregressive masked action heads.

Pure JAX keeps environment, inference and PPO on the same accelerator. No LLM,
PyTorch runtime, pretrained weights, or dependency on the Duper code is needed.
"""
import math
import os
import jax
import jax.numpy as jnp

from .actions import Choice, PathMasks, family_mask, source_mask, target_mask

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


def init(key, board_channels, global_channels, *, size=14, width=64, blocks=2, architecture='legacy',
         extra_blocks=0, global_pool=False):
    """extra_blocks > 0 adds pre-activation residual blocks after the trunk (key 'extra_blocks'); their
    second convolution starts at zero, so they are the identity until trained (see grow). global_pool: each extra
    block also adds a per-channel bias computed from the mean and max over the board of its first convolution
    (KataGo's global pooling), so every tile sees board-wide context past the trunk's receptive field."""
    if architecture not in ('legacy', 'normalized-v2'):
        raise ValueError('unknown network architecture')
    def weights(shape, scale=1.):
        nonlocal key
        key, draw = jax.random.split(key)
        fan_in = math.prod(shape[:-1])
        return jax.random.normal(draw, shape) * (scale / math.sqrt(fan_in))
    def linear(a, b, scale=1.):
        return {"w": weights((a, b), scale), "b": jnp.zeros(b)}
    params = {
        "stem": weights((3, 3, board_channels, width), math.sqrt(2)),
        "blocks": tuple((weights((3, 3, width, width), math.sqrt(2)),
                         weights((3, 3, width, width))) for _ in range(blocks)),
        "global": linear(global_channels + width, width),
        "family": linear(width, 17, .01),
        "source": linear(width, 17, .01),
        "query": linear(2 * width, width),
        "family_embedding": weights((17, width), .1),
        "parameter_embedding": weights((17, max(size * size, 128), width), .1),
        "value": linear(width, 1, .1),
    }
    if architecture == 'normalized-v2':
        # Versioned tensor layout keeps old checkpoints loadable without silently
        # interpreting their weights under a different forward pass.
        params['global'] = linear(global_channels + 2 * width, width)
        params['norm'] = {'scale': jnp.ones(width), 'bias': jnp.zeros(width)}
        params['critic_hidden'] = linear(global_channels + 2 * width, width)
    if extra_blocks:
        params['extra_blocks'] = tuple((weights((3, 3, width, width), math.sqrt(2)), jnp.zeros((3, 3, width, width)))
                                       + ((linear(2 * width, width),) if global_pool else ())
                                       for _ in range(extra_blocks))
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


def widen_inputs(params, old_board, new_board, old_global, new_global, groups=1):
    """A network on wider features that computes exactly what `params` computes on the old ones.

    The input is `groups` concatenated feature groups (1 for the policy; the omniscient critic's own view plus
    the other seats' views) of which `params` saw the first min(groups, old groups) at old widths. New features
    sit after the old ones inside each group (observation.encode_view), so the stem and the pooled-input layers
    ('global', 'critic_hidden', normalized-v2) keep their old rows in place and get zero rows for every new
    feature and for each new group."""
    if 'norm' not in params:
        raise ValueError('widen_inputs needs a normalized-v2 network')
    stem = params['stem']
    old_groups = stem.shape[2] // old_board
    if stem.shape[2] != old_groups * old_board or old_groups > groups:
        raise ValueError('stem width does not match the old feature groups')
    def spread(w, axis, old, new, prefix=0):
        parts = [jnp.take(w, jnp.arange(prefix), axis=axis)] if prefix else []
        for k in range(groups):
            block = (jnp.take(w, prefix + k * old + jnp.arange(old), axis=axis) if k < old_groups else
                     jnp.zeros(w.shape[:axis] + (old,) + w.shape[axis + 1:], w.dtype))
            pad = [(0, 0)] * w.ndim
            pad[axis] = (0, new - old)
            parts.append(jnp.pad(block, pad))
        return jnp.concatenate(parts, axis)
    out = dict(params, stem=spread(stem, 2, old_board, new_board))
    width = stem.shape[-1]
    for name in ('global', 'critic_hidden'):
        if name in params:
            out[name] = dict(params[name], w=spread(params[name]['w'], 0, old_global, new_global, prefix=2 * width))
    return out


def dense(x, layer):
    # Keep rollout and differently sized learner minibatches on the same FP32
    # arithmetic path. GPU TF32 algorithm choices otherwise shift old logp.
    return jnp.matmul(x, layer["w"], precision=HEAD_PRECISION) + layer["b"]


def trunk(params, features):
    def conv(x, kernel):
        return jax.lax.conv_general_dilated(x[None].astype(TRUNK_DTYPE), kernel.astype(TRUNK_DTYPE), (1, 1), "SAME",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"), precision=PRECISION)[0].astype(jnp.float32)
    normalized = 'norm' in params
    activate = (lambda z: jax.nn.silu(layer_norm(z))) if normalized else jax.nn.relu
    x = activate(conv(features.board, params["stem"]))
    for a, b in params["blocks"]:
        x = activate(x + conv(activate(conv(x, a)), b))
    for block in params.get("extra_blocks", ()):
        h = conv(x, block[0])
        if len(block) == 3:   # global pooling bias (init global_pool)
            flat = h.reshape(-1, h.shape[-1])
            h = h + dense(jnp.concatenate((flat.mean(0), flat.max(0))), block[2])
        x = x + conv(activate(h), block[1])
    tiles = x.reshape(-1, x.shape[-1])
    if normalized:
        pooled = jnp.concatenate((tiles.mean(0), tiles.max(0), features.global_state))
        g = jax.nn.silu(layer_norm(dense(pooled, params['global'])) * params['norm']['scale'] + params['norm']['bias'])
        critic = jax.nn.silu(layer_norm(dense(pooled, params['critic_hidden'])))
        return tiles, g, dense(critic, params['value'])[0]
    g = jnp.tanh(dense(jnp.concatenate((tiles.mean(0), features.global_state)), params["global"]))
    return tiles, g, dense(g, params["value"])[0]


def layer_norm(x):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(jnp.mean(centered ** 2, axis=-1, keepdims=True) + 1e-5)


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
    return choice, path, logp, value
