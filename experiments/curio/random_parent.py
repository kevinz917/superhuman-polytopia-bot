"""Random-init parent for a from-scratch run, no Modal (the core of experiments/modal_distributed_rl.py random_init_fn).

    python random_parent.py --reference REF.npz --out OUT.npz [--seed 1] [--blocks 4] [--width 128] [--players 2] [--size 14]
                            [--block-pool [LAYOUT]] [--attention N]

The architecture (blocks, width, extra_blocks, architecture, size) comes from the reference checkpoint's json and the
overrides; weights are network.init(key(seed)), saved with runner.save_checkpoint (npz + json sidecar with the
ruleset and sha256) and re-loaded once as a check. network.init is already near-uniform over legal actions (the
Oct 22 scratch check measured path entropy at 98.8% of uniform), so no head rescale is applied.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--blocks', type=int, default=0)
    ap.add_argument('--width', type=int, default=0)
    ap.add_argument('--players', type=int, default=2)
    ap.add_argument('--size', type=int, default=0)
    ap.add_argument('--obs-version', type=int, default=0, help='0 = the newest (rl.observation.OBS_VERSION)')
    ap.add_argument('--block-pool', type=int, nargs='?', const=2, default=0,
                    help='global pooling in the trunk blocks (network.init block_pool): a layout number, 2 when given bare; '
                         '1 = farsight\'s layout (every block, un-normalized)')
    ap.add_argument('--attention', type=int, default=0, help='transformer blocks after the convolutions (network.init attention)')
    args = ap.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import jax
    from polytopia_bot.rl import network
    from polytopia_bot.rl.distributed import network_spec, checkpoint_template, feature_shapes
    from polytopia_bot.rl.runner import Config, load_checkpoint, save_checkpoint
    from polytopia_bot.rl.observation import OBS_VERSION
    obs_version = args.obs_version or OBS_VERSION
    spec = network_spec(args.reference)
    spec.update({k: v for k, v in dict(blocks=args.blocks, width=args.width, size=args.size).items() if v})
    spec.pop('players', None)
    spec.pop('obs_version', None)
    spec.pop('block_pool', None)
    spec.pop('attention', None)
    board_channels, global_features = feature_shapes(args.players, obs_version)
    params = network.init(jax.random.key(args.seed), board_channels, global_features, size=spec['size'],
                          width=spec['width'], blocks=spec['blocks'], architecture=spec['architecture'],
                          extra_blocks=spec['extra_blocks'], block_pool=args.block_pool, attention=args.attention)
    meta = json.loads(Path(args.reference).with_suffix('.json').read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out, params, Config(**{**meta['config'], 'blocks': spec['blocks'], 'width': spec['width'],
                                           'size': spec['size'], 'players': args.players, 'obs_version': obs_version,
                                           'block_pool': args.block_pool, 'attention': args.attention}),
                    stage=f'random-init-seed{args.seed}')
    load_checkpoint(out, checkpoint_template(out))
    n = sum(int(x.size) for x in jax.tree.leaves(params))
    print(json.dumps(dict(out=str(out), spec=network_spec(out), players=args.players, obs_version=obs_version, parameters=n,
                          json=json.loads(out.with_suffix('.json').read_text()).get('stage'))))


if __name__ == '__main__':
    main()
