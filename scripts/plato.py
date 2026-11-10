"""Verify/play the released Plato weights, evaluate paired games, or build training maps.

Run from a repository checkout after installing it: python scripts/plato.py --help.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(checkpoint, decisions=8, seed=500000):
    import jax
    import numpy as np
    from polytopia_bot.rl.distributed import _critic_template, _template_params, network_spec
    from polytopia_bot.rl.policy import StructuredPolicy
    from polytopia_bot.rl.runner import load_checkpoint
    from polytopia_bot.rl.data import sha256
    from polytopia_bot.studio.environment import RustEnvironment

    path = Path(checkpoint)
    spec = network_spec(path)
    params = load_checkpoint(path, _template_params(**spec))
    critic_path = path.with_name(path.stem + '-critic.npz')
    critic = load_checkpoint(critic_path, _critic_template(**spec))
    env = RustEnvironment(spec['size'])
    env.reset(seed=seed, settings={})
    policy = StructuredPolicy(path, seed=0)
    for _ in range(decisions):
        if env.state.terminated or env.state.truncated:
            raise RuntimeError('game ended before the requested verification decisions')
        player = int(env.state.current_player)
        obs = env.observe(player)
        decision = policy.select(obs, env.view(player))
        if not np.isfinite(decision.log_probability):
            raise RuntimeError('nonfinite policy log probability')
        env.step(action_id=decision.action.action_id, expected_state_version=obs.state_version)
    return dict(checkpoint=str(path), sha256=sha256(path), critic_sha256=sha256(critic_path), network=spec,
                policy_parameters=sum(x.size for x in jax.tree.leaves(params)),
                critic_parameters=sum(x.size for x in jax.tree.leaves(critic)), legal_decisions=decisions,
                seed=seed)


def map_table(path, lo, hi):
    import numpy as np
    from polytopia_bot.maps.stream import MapSource
    if lo < 0 or hi <= lo or hi > 2**31 - 1:
        raise ValueError('map seeds must be a nonempty nonnegative int32 interval')
    source = MapSource('classic', 14, players=2)
    arrays = dict(kind=np.zeros(hi - lo, np.int8),
                  **{k: np.empty((hi - lo, 2 if k == 'capitals' else 196), np.int16)
                     for k in ('terrain', 'structure', 'resource', 'capitals')})
    for i, seed in enumerate(range(lo, hi)):
        kind, layout = source.layout(seed)
        arrays['kind'][i] = source.types.index(kind)
        for key in ('terrain', 'structure', 'resource', 'capitals'):
            arrays[key][i] = np.asarray(getattr(layout, key))
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('map-table output must end in .npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, lo=np.int64(lo), hi=np.int64(hi), types=np.array(source.types), **arrays)
    return dict(path=str(path), seeds=hi - lo, lo=lo, hi=hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('verify', help='check policy/critic integrity and make legal Studio moves')
    check.add_argument('--checkpoint', default='weights/plato/final.npz')
    check.add_argument('--decisions', type=int, default=8)
    check.add_argument('--seed', type=int, default=500000)
    arena = sub.add_parser('arena', help='sampled T=1 games on paired maps, both seats')
    arena.add_argument('--checkpoint', default='weights/plato/final.npz')
    arena.add_argument('--opponent', default='weights/hero-1v1/final.npz')
    arena.add_argument('--backend', choices=('rust', 'cpu', 'cuda'), default='rust')
    arena.add_argument('--maps', type=int, default=1)
    arena.add_argument('--seed-base', type=int, default=500000)
    arena.add_argument('--seed', type=int, default=0)
    arena.add_argument('--no-peace', action='store_true')
    arena.add_argument('--max-ticks', type=int, default=10000)
    arena.add_argument('--output', type=Path)
    maps = sub.add_parser('maps', help='generate the deterministic classic 14x14 training map table locally')
    maps.add_argument('--output', default='runs/maps/table-classic-14-2p-200000-300000.npz')
    maps.add_argument('--lo', type=int, default=200000)
    maps.add_argument('--hi', type=int, default=300000)
    args = parser.parse_args()
    if args.command == 'verify':
        if args.decisions < 1:
            parser.error('--decisions must be positive')
        result = verify(args.checkpoint, args.decisions, args.seed)
    elif args.command == 'maps':
        result = map_table(args.output, args.lo, args.hi)
    else:
        from experiments.claude_v3.gpu_arena import run
        if args.maps < 1 or args.max_ticks < 1:
            parser.error('--maps and --max-ticks must be positive')
        result = run(dict(a=args.checkpoint, b=args.opponent, backend=args.backend, maps=args.maps,
                          seed_base=args.seed_base, seed=args.seed, no_peace=args.no_peace, max_ticks=args.max_ticks))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
