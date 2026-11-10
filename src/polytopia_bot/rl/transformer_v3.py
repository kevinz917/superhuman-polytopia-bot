"""Transformer V3: one public encoder shared by the policy and a small privileged critic branch.

Design: docs/v3-compute/design-4a.md and docs/plato-reproduction.md (Claude session E, 2026-11-07). Differences from V2
(rl/transformer.py encode_v2 + a second full critic transformer on both seats' views):

  * stem: a per-tile linear map of the feature planes instead of V2's 3x3 convolution. On the one-hot categorical
    planes (terrain, resource, structure, unit type, owners, facing) a linear map IS a sum of learned categorical
    embeddings, and on the scalar planes it is a scalar projection; unknown/absent/zero stay distinct because the planes
    keep them distinct. The observation ABI is unchanged, so arenas, CPU actors and the distillation adapter read V3
    from the same features. 9x fewer stem MACs; locality comes from the learned relative attention bias.
  * tokens: 1x1 tiles (one token per tile + one global token, 197 at 14x14, as V2) or, with patch 2, one token per
    2x2 tile patch (50 tokens at 14x14). Patch tokens decode back to tiles for the action heads through a learned
    per-sub-position projection plus a skip connection from the tile embedding, so every tile stays addressable.
  * heads: V2's factorization (family from the global token, source = per-tile score per family, target = query-key
    pointer over tiles or parameter embeddings) is kept so Hero1 conditional teacher targets attach unchanged
    (DESIGN.md section 2). V2's source head is already a pointer with one learned query per family and its target
    head already a query-key pointer; V3 only replaces the CNN-style mean/max pooling for `g` with the global token.
  * critic: a separate parameter tree that reads the policy's public tokens (before the final norm) plus a 1x1
    embedding of the other seat's view and runs `critic_blocks` transformer blocks (0 = pooled MLP only). The public
    encoder never reads privileged input; value gradients do train the shared encoder.

Architecture names: 'transformer-v3' [+ '-p2'] [+ '-k3' | '-h<n>'] [+ '-c<k>'] (critic blocks, default 1). '-k3'
replaces the 1x1 stem with V2's 3x3 convolution + LayerNorm + SiLU (local inductive bias before patching; the critic's
privileged embedding stays 1x1).

'-h<n>' (HYBRID trunk, Claude session H, 2026-11-07): the hero CNN's convolutional torso (normalized-v2: a 3x3 stem
convolution, then n residual blocks x <- act(x + conv(act(conv(x)))), act = SiLU(LayerNorm), width W, the hero
CNN's initialization) replaces the stem; its tile output is the tile embedding, so the transformer reads patches of
convolutional features and the action heads keep a skip from them. Everything after the stem (patching, global
token, blocks, heads, privileged critic) is unchanged; the critic's privileged embedding stays 1x1.
"""
import math
import re

import jax
import jax.numpy as jnp

from .network import PRECISION, TRUNK_DTYPE, layer_norm
from .observation import fit_board, fit_global
from .transformer import HEADS, REMATERIALIZE, _checkpointed, block, linear, relative_attention_bias

ARCHITECTURE_PATTERN = re.compile(r'transformer-v3(?P<patch>-p2)?(?:(?P<conv>-k3)|-h(?P<cnn>[1-9]))?(?:-c(?P<critic>\d))?')
DEFAULT_CRITIC_BLOCKS = 1


def parse(architecture: str) -> dict:
    """(patch, conv, cnn_blocks, critic_blocks) of a V3 architecture name; ValueError for anything else. conv: a 3x3
    stem ('-k3', or implied by '-h<n>'); cnn_blocks: the hybrid's residual convolution blocks after it (0 otherwise)."""
    match = ARCHITECTURE_PATTERN.fullmatch(architecture)
    if not match:
        raise ValueError(f'not a transformer-v3 architecture: {architecture}')
    critic, cnn = match.group('critic'), match.group('cnn')
    return dict(patch=2 if match.group('patch') else 1, conv=bool(match.group('conv') or cnn),
                cnn_blocks=int(cnn or 0), critic_blocks=DEFAULT_CRITIC_BLOCKS if critic is None else int(critic))


def is_v3(architecture: str) -> bool:
    return ARCHITECTURE_PATTERN.fullmatch(architecture) is not None


def _grid(size: int, patch: int) -> int:
    if size % patch:
        raise ValueError(f'board size {size} is not divisible by patch {patch}')
    return size // patch


def _block_params(dense, width: int, grid: int) -> dict:
    return dict(qkv=dense(width, 3 * width), out=dense(width, width),
                ff_in=dense(width, 4 * width, 2 ** .5), ff_out=dense(4 * width, width),
                attention_norm=dict(scale=jnp.ones(width), bias=jnp.zeros(width)),
                ff_norm=dict(scale=jnp.ones(width), bias=jnp.zeros(width)),
                relative_bias=jnp.zeros((HEADS, 2 * grid - 1, 2 * grid - 1)))


def _initializer(key):
    state = dict(key=key)

    def weights(shape, scale=1.):
        state['key'], draw = jax.random.split(state['key'])
        return jax.random.normal(draw, shape) * (scale / shape[0] ** .5)

    def dense(a, b, scale=1.):
        return {'w': weights((a, b), scale), 'b': jnp.zeros(b)}
    return weights, dense


def _conv_kernel(weights, shape: tuple, scale: float):
    """A [3, 3, in, out] kernel at network.init's scale (fan-in 3 x 3 x in), so the hybrid's torso starts as the hero CNN."""
    return weights(shape, scale) * (shape[0] / math.prod(shape[:-1])) ** .5


def init_policy(key, board_channels: int, global_channels: int, *, size: int, width: int, blocks: int,
                architecture: str) -> dict:
    """Policy parameters. Head names match rl.network so network.evaluate / sample run V3 unchanged."""
    options = parse(architecture)
    patch = options['patch']
    grid = _grid(size, patch)
    if width % HEADS or blocks < 1:
        raise ValueError('transformer-v3 needs width divisible by the head count and blocks >= 1')
    weights, dense = _initializer(key)
    if options['cnn_blocks']:
        stem = {'w': _conv_kernel(weights, (3, 3, board_channels, width), 2 ** .5), 'b': jnp.zeros(width)}
    elif options['conv']:
        stem = {'w': weights((3, 3, board_channels, width), 2 ** .5) / 3., 'b': jnp.zeros(width)}
    else:
        stem = dense(board_channels, width)
    params = {
        'v3_stem': stem,
        'global_token': dense(global_channels, width),
        'transformer': tuple(_block_params(dense, width, grid) for _ in range(blocks)),
        'v3_final_norm': dict(scale=jnp.ones(width), bias=jnp.zeros(width)),
        'family': dense(width, 17, .01),
        'source': dense(width, 17, .01),
        'query': dense(2 * width, width),
        'family_embedding': weights((17, width), .1),
        'parameter_embedding': weights((17, max(size * size, 128), width), .1),
        # network.target_logits layer-normalizes the query when 'norm' exists (normalized-v2 semantics)
        'norm': {'scale': jnp.ones(width), 'bias': jnp.zeros(width)},
    }
    if options['cnn_blocks']:
        params['v3_conv_blocks'] = tuple((_conv_kernel(weights, (3, 3, width, width), 2 ** .5),
                                          _conv_kernel(weights, (3, 3, width, width), 1.))
                                         for _ in range(options['cnn_blocks']))
    if patch > 1:
        params['v3_patch'] = dense(patch * patch * width, width)
        params['v3_unpatch'] = dense(width, patch * patch * width, .5)
    return params


def init_critic(key, board_channels: int, global_channels: int, *, size: int, width: int, architecture: str) -> dict:
    """Critic-exclusive parameters: privileged (other seat's view) embedding, `critic_blocks` blocks and the value MLP."""
    options = parse(architecture)
    patch = options['patch']
    grid = _grid(size, patch)
    _, dense = _initializer(key)
    critic = {
        'priv_stem': dense(board_channels, width, .5),
        'priv_global': dense(global_channels, width, .5),
        'blocks': tuple(_block_params(dense, width, grid) for _ in range(options['critic_blocks'])),
        'hidden': dense(3 * width, width),
        'value': dense(width, 1, .1),
    }
    if patch > 1:
        critic['priv_patch'] = dense(patch * patch * width, width)
    return critic


def _to_patches(tiles: jax.Array, size: int, patch: int) -> jax.Array:
    """[size*size, c] row-major tiles -> [(size/patch)^2, patch*patch*c] row-major patches."""
    grid, c = size // patch, tiles.shape[-1]
    x = tiles.reshape(grid, patch, grid, patch, c).transpose(0, 2, 1, 3, 4)
    return x.reshape(grid * grid, patch * patch * c)


def _from_patches(patches: jax.Array, size: int, patch: int) -> jax.Array:
    """Inverse of _to_patches."""
    grid = size // patch
    c = patches.shape[-1] // (patch * patch)
    x = patches.reshape(grid, grid, patch, patch, c).transpose(0, 2, 1, 3, 4)
    return x.reshape(size * size, c)


def _patch_of(params: dict) -> int:
    if 'v3_patch' not in params:
        return 1
    width = params['v3_stem']['w'].shape[-1]   # the stem's output width (1x1 [C, W] or 3x3 [3, 3, C, W])
    return int(round((params['v3_patch']['w'].shape[0] // width) ** .5))


def _conv(x: jax.Array, kernel: jax.Array) -> jax.Array:
    """3x3 SAME convolution of [size, size, C] on TRUNK_DTYPE operands with an fp32 result (the CNN torso's policy)."""
    return jax.lax.conv_general_dilated(x[None].astype(TRUNK_DTYPE), kernel.astype(TRUNK_DTYPE), (1, 1), 'SAME',
                                        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                        precision=PRECISION)[0].astype(jnp.float32)


def _activate(x: jax.Array) -> jax.Array:
    return jax.nn.silu(layer_norm(x))


def _stem(stem: dict, merge: dict | None, board: jax.Array, size: int, patch: int, conv_blocks: tuple = ()) -> tuple:
    """Feature planes [size, size, C] -> (tokens [grid^2, W], tile embedding [size^2, W]). conv_blocks: the hybrid's
    residual blocks (normalized-v2's, network.trunk) after a 3x3 stem."""
    if stem['w'].ndim == 4:
        # '-k3' / '-h<n>': a 3x3 convolution of the planes (V2's / the hero CNN's stem), normalized and activated
        x = _activate(_conv(board, stem['w']) + stem['b'])
        for first, second in conv_blocks:
            x = _activate(x + _conv(_activate(_conv(x, first)), second))
        tiles = x.reshape(size * size, -1)
    else:
        tiles = linear(board.reshape(size * size, -1), stem)
    if patch == 1:
        return tiles, tiles
    return linear(_to_patches(tiles, size, patch), merge), tiles


def _run_blocks(x: jax.Array, blocks: tuple, grid: int) -> jax.Array:
    step = _checkpointed(block) if REMATERIALIZE else block
    for weights in blocks:
        x = step(x, weights, relative_attention_bias(weights['relative_bias'], grid, grid))
    return x


def encode(params: dict, board: jax.Array, global_state: jax.Array) -> tuple:
    """Public encoder. Returns (tokens [grid^2 + 1, W] before the final norm, tiles [size^2, W], g [W])."""
    size = board.shape[0]
    patch = _patch_of(params)
    grid = _grid(size, patch)
    board = fit_board(board, params['v3_stem']['w'].shape[-2])
    glob = fit_global(global_state, params['global_token']['w'].shape[0])
    tokens, tile_embedding = _stem(params['v3_stem'], params.get('v3_patch'), board, size, patch,
                                   params.get('v3_conv_blocks', ()))
    context = linear(glob, params['global_token'])
    x = _run_blocks(jnp.concatenate((tokens, context[None]), axis=0), params['transformer'], grid)
    n = params['v3_final_norm']
    out = layer_norm(x) * n['scale'] + n['bias']
    if patch == 1:
        tiles = out[:-1]
    else:
        tiles = _from_patches(linear(out[:-1], params['v3_unpatch']), size, patch) + tile_embedding
    return x, jax.nn.silu(tiles), out[-1]


def trunk(params: dict, features) -> tuple:
    """network.trunk's contract (tiles, g, value). V3's policy has no value head: the value lives in the critic."""
    _, tiles, g = encode(params, features.board, features.global_state)
    return tiles, g, jnp.zeros((), jnp.float32)


def _split_views(features) -> tuple:
    """Omniscient features (distributed.critic_features: own view, then the other seat's, along channels) ->
    (own board, own global, other board, other global)."""
    own_board, opp_board = jnp.split(features.board, 2, axis=-1)
    own_glob, opp_glob = jnp.split(features.global_state, 2, axis=-1)
    return own_board, own_glob, opp_board, opp_glob


def _critic_head(params: dict, critic: dict, tokens: jax.Array, opp_board: jax.Array, opp_glob: jax.Array) -> jax.Array:
    """Value from the public tokens (encoder output before the final norm) and the other seat's view."""
    size = opp_board.shape[0]
    patch = _patch_of(params)
    grid = _grid(size, patch)
    opp_board = fit_board(opp_board, critic['priv_stem']['w'].shape[0])
    opp_glob = fit_global(opp_glob, params['global_token']['w'].shape[0])
    priv, _ = _stem(critic['priv_stem'], critic.get('priv_patch'), opp_board, size, patch)
    priv_context = linear(opp_glob, critic['priv_global'])
    x = tokens + jnp.concatenate((priv, priv_context[None]), axis=0)
    x = layer_norm(_run_blocks(x, critic['blocks'], grid))
    pooled = jnp.concatenate((x[:-1].mean(0), x[:-1].max(0), x[-1]))
    hidden = jax.nn.silu(layer_norm(linear(pooled, critic['hidden'])))
    return linear(hidden, critic['value'])[0]


def value(params: dict, critic: dict, features) -> jax.Array:
    """Critic value of one sample from omniscient features. Only the first view enters the public encoder."""
    own_board, own_glob, opp_board, opp_glob = _split_views(features)
    tokens, _, _ = encode(params, own_board, own_glob)
    return _critic_head(params, critic, tokens, opp_board, opp_glob)


def evaluate(params: dict, critic: dict, features, choice, masks) -> tuple:
    """network.evaluate's (logp, entropy) plus the critic value with ONE public encoder pass (the learner's loss)."""
    from . import network
    own_board, own_glob, opp_board, opp_glob = _split_views(features)
    tokens, tiles, g = encode(params, own_board, own_glob)
    logits = (network.dense(g, params['family']), network.dense(tiles, params['source'])[:, choice.family],
              network.target_logits(params, tiles, g, choice.family, choice.source))
    pairs = [network.logprob_entropy(l, m, a) for l, m, a in zip(logits, masks, choice, strict=True)]
    return sum(p[0] for p in pairs), _critic_head(params, critic, tokens, opp_board, opp_glob), sum(p[1] for p in pairs)


def sample_and_value(params: dict, critic: dict, features, masks, key, *, size: int) -> tuple:
    """The rollout's (choice, path masks, logp, value) for one sample with ONE public encoder pass. `features` are the
    omniscient features (own view first); sampling reads only the own view's encoding."""
    from . import network
    own_board, own_glob, opp_board, opp_glob = _split_views(features)
    tokens, tiles, g = encode(params, own_board, own_glob)
    choice, path, logp = network.sample_from(params, tiles, g, masks, key, size=size)
    return choice, path, logp, _critic_head(params, critic, tokens, opp_board, opp_glob)
