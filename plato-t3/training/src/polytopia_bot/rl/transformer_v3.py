"""Transformer V3: one public encoder shared by the policy and a small privileged critic branch.

Design: (compute half by Claude session E, 2026-11-07). Differences from V2
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
import os
import re

import jax
import jax.numpy as jnp

from .network import PRECISION, TRUNK_DTYPE, layer_norm
from .observation import fit_board, fit_global
from .transformer import HEADS, REMATERIALIZE, _checkpointed, block, linear, relative_attention_bias

ARCHITECTURE_PATTERN = re.compile(r'transformer-v3(?P<patch>-p2)?(?:(?P<conv>-k3)|-h(?P<cnn>[1-9]))?(?:-c(?P<critic>\d))?')
DEFAULT_CRITIC_BLOCKS = 1
# Stage 5 (all sizes, one network): POLYTOPIA_V3_BASE_SIZE=n builds every V3 parameter tree for an n x n board (the
# relative-position tables and parameter_embedding rows), whatever board the run plays. The forward pass then plays
# any size with those weights: a board whose side is not a multiple of the patch is zero-padded up to one (tiles
# cropped back), and attention offsets beyond the table's reach share its edge entries (clamped). At the base size
# both are the identity, so a base-size run computes exactly what it computed before. 0/unset: the board's own size.
BASE_SIZE = int(os.environ.get('POLYTOPIA_V3_BASE_SIZE', '0') or 0)


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


def _template_size(size: int) -> int:
    """The board size the parameter tree is built for (BASE_SIZE when set)."""
    return BASE_SIZE or size


def _padded(size: int, patch: int) -> int:
    """The smallest multiple of the patch >= size (the board the encoder runs on)."""
    return -(-size // patch) * patch


def _pad_board(board: jax.Array, patch: int) -> jax.Array:
    pad = _padded(board.shape[0], patch) - board.shape[0]
    return board if pad == 0 else jnp.pad(board, ((0, pad), (0, pad), (0, 0)))


def _table_bias(table: jax.Array, grid: int) -> jax.Array:
    """relative_attention_bias for a grid x grid token board from a table of any reach: offsets beyond the table's
    +-(g0 - 1) use its edge entries. The table's own grid is the plain lookup."""
    g0 = (table.shape[1] + 1) // 2
    if grid != g0:
        idx = jnp.clip(jnp.arange(2 * grid - 1) - (grid - 1), -(g0 - 1), g0 - 1) + g0 - 1
        table = table[:, idx][:, :, idx]
    return relative_attention_bias(table, grid, grid)


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
    size = _template_size(size)
    grid = _grid(_padded(size, patch), patch)
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
    grid = _grid(_padded(_template_size(size), patch), patch)
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
        x = step(x, weights, _table_bias(weights['relative_bias'], grid))
    return x


def encode(params: dict, board: jax.Array, global_state: jax.Array) -> tuple:
    """Public encoder. Returns (tokens [grid^2 + 1, W] before the final norm, tiles [size^2, W], g [W])."""
    real = board.shape[0]
    patch = _patch_of(params)
    board = _pad_board(board, patch)            # identity unless the side is not a multiple of the patch
    size = board.shape[0]
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
    if real != size:
        tiles = tiles.reshape(size, size, -1)[:real, :real].reshape(real * real, -1)
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
    patch = _patch_of(params)
    opp_board = _pad_board(opp_board, patch)
    size = opp_board.shape[0]
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


def widen_obs_v4(params: dict, critic: dict | None = None, players: int = 2) -> tuple:
    """A V3 network of observation version 3 (Plato) as a version-4 network that computes exactly the
    same thing: every board stem (the policy's 'v3_stem', the critic's 'priv_stem') gets zero rows for the new
    water-resource channels (observation.WATER_RESOURCE_CHANNELS, at the end of each view group); all other weights
    are unchanged. Returns (params, critic). Zero rows give identical outputs on EVERY board until training moves them;
    on boards without Starfish/aqua crop the new channels are zero, so they stay identical whatever the new rows hold."""
    from .observation import WATER_RESOURCE_CHANNELS, group_width
    old, new = group_width(players, 3), group_width(players, 4)
    assert new == old + WATER_RESOURCE_CHANNELS

    def widen(w):
        axis = w.ndim - 2                       # [C, W] dense or [3, 3, C, W] conv: channels are the second-to-last
        channels = w.shape[axis]
        if channels % old:
            raise ValueError(f'stem reads {channels} channels, not whole version-3 groups of {old}')
        parts = []
        for k in range(channels // old):
            parts.append(jax.lax.slice_in_dim(w, k * old, (k + 1) * old, axis=axis))
            shape = list(w.shape)
            shape[axis] = WATER_RESOURCE_CHANNELS
            parts.append(jnp.zeros(shape, w.dtype))
        return jnp.concatenate(parts, axis)

    params = dict(params, v3_stem=dict(params['v3_stem'], w=widen(params['v3_stem']['w'])))
    if critic is not None and 'priv_stem' in critic:
        critic = dict(critic, priv_stem=dict(critic['priv_stem'], w=widen(critic['priv_stem']['w'])))
    return params, critic


def widen_obs_v5(params: dict, critic: dict | None = None, players: int = 2) -> tuple:
    """full-game engine tribes (2026-11-09): a V3 network of observation version 3 or 4 as a version-5 network that computes exactly
    the same thing. Version 3 first gets widen_obs_v4's zero board rows; then the global readers (the policy's
    'global_token', the critic's 'priv_global') get zero rows for the tribe entries (observation.TRIBE_ENTRIES x 2,
    at the end of the global vector). Zero rows give identical outputs on every game until training moves them;
    Bardur-vs-Bardur games have those entries at zero, so they stay identical whatever the new rows hold."""
    from .observation import TRIBE_ENTRIES, global_width, group_width
    if params['v3_stem']['w'].shape[-2] % group_width(players, 4):
        params, critic = widen_obs_v4(params, critic, players)
    old, new = global_width(players, 4), global_width(players, 5)
    assert new == old + 2 * TRIBE_ENTRIES

    def widen(w):
        if w.shape[0] != old:
            raise ValueError(f'global reader reads {w.shape[0]} entries, not the version-4 {old}')
        return jnp.concatenate((w, jnp.zeros((new - old,) + w.shape[1:], w.dtype)), 0)

    params = dict(params, global_token=dict(params['global_token'], w=widen(params['global_token']['w'])))
    if critic is not None and 'priv_global' in critic:
        critic = dict(critic, priv_global=dict(critic['priv_global'], w=widen(critic['priv_global']['w'])))
    return params, critic



def widen_obs_v6(params: dict, critic: dict | None = None, players: int = 2) -> tuple:
    """special tribes (2026-11-09; section 2): a V3 network of observation version 3, 4 or
    5 as a version-6 network that computes exactly the same thing. Versions 3/4 first get widen_obs_v5; then every
    board stem (the policy's 'v3_stem', the critic's 'priv_stem') gets zero rows for the special-tribe channels
    (observation.SPECIAL_CHANNELS, at the end of each view group) and the global readers ('global_token',
    'priv_global') one zero row for the Convert task (observation.SPECIAL_GLOBAL, at the end of the global vector).
    Zero rows give identical outputs on every board until training moves them; in a game of two regular tribes every
    new channel is zero, so it stays identical whatever the new rows hold. The action heads need no change for the
    layout-5 catalogue (widen_actions)."""
    from .observation import SPECIAL_CHANNELS, SPECIAL_GLOBAL, global_width, group_width
    if params['global_token']['w'].shape[0] != global_width(players, 5):
        params, critic = widen_obs_v5(params, critic, players)
    old, new = group_width(players, 5), group_width(players, 6)
    gold, gnew = global_width(players, 5), global_width(players, 6)
    assert new == old + SPECIAL_CHANNELS and gnew == gold + SPECIAL_GLOBAL

    def board(w):
        axis = w.ndim - 2                       # [C, W] dense or [3, 3, C, W] conv: channels are the second-to-last
        channels = w.shape[axis]
        if channels % old:
            raise ValueError(f'stem reads {channels} channels, not whole version-5 groups of {old}')
        parts = []
        for k in range(channels // old):
            parts.append(jax.lax.slice_in_dim(w, k * old, (k + 1) * old, axis=axis))
            shape = list(w.shape)
            shape[axis] = SPECIAL_CHANNELS
            parts.append(jnp.zeros(shape, w.dtype))
        return jnp.concatenate(parts, axis)

    def glob(w):
        if w.shape[0] != gold:
            raise ValueError(f'global reader reads {w.shape[0]} entries, not the version-5 {gold}')
        return jnp.concatenate((w, jnp.zeros((gnew - gold,) + w.shape[1:], w.dtype)), 0)

    params = dict(params, v3_stem=dict(params['v3_stem'], w=board(params['v3_stem']['w'])),
                  global_token=dict(params['global_token'], w=glob(params['global_token']['w'])))
    if critic is not None:
        if 'priv_stem' in critic:
            critic = dict(critic, priv_stem=dict(critic['priv_stem'], w=board(critic['priv_stem']['w'])))
        if 'priv_global' in critic:
            critic = dict(critic, priv_global=dict(critic['priv_global'], w=glob(critic['priv_global']['w'])))
    return params, critic


def widen_actions(params: dict, from_version: int = 2, to_version: int = None) -> dict:
    """A network trained on action layout `from_version` (T2: 2) for layout `to_version` (default actions.ACTION_VERSION,
    5: the recover row grows from 2 to 9 targets). Every layout keeps the families, their order and every older entry
    at its (family, source, target), and the target head scores a non-spatial family's targets with its first
    network.PARAMETER_TARGETS (128) parameter embeddings, which already cover targets 2-8: so the weights are
    returned unchanged, after checking that the head is wide enough. On a board where the new entries are illegal
    (every game of two regular tribes) the masked distributions are identical (tests/test_obs_v6.py)."""
    from .actions import ACTION_VERSION, shapes
    from .network import PARAMETER_TARGETS
    to_version = ACTION_VERSION if to_version is None else to_version
    if not 2 <= from_version <= to_version:
        raise ValueError(f'cannot widen action layout {from_version} to {to_version}')
    old, new = shapes(14, from_version), shapes(14, to_version)
    if len(old) != len(new) or params['parameter_embedding'].shape[0] != len(new):
        raise ValueError('the family count changed')
    widest = max(b for f, (_, b) in enumerate(new) if f >= 2)
    if params['parameter_embedding'].shape[1] < min(widest, PARAMETER_TARGETS):
        raise ValueError(f'the target head holds {params["parameter_embedding"].shape[1]} embeddings, layout '
                         f'{to_version} needs {widest}')
    return params
