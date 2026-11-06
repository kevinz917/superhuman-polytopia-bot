"""Widen a Curio run's checkpoints from observation version 2 to 3 in place (docs/obs-v3-history-2026-11-01.md), and
with --block-pool also add a zero global-pooling bias to every trunk block (network.add_block_pool).

    PYTHONPATH=code/<new>/src venv/bin/python code/<new>/experiments/curio/widen_obs.py \
        --run-dir /data/polytopia/runs/RUN --backup /data/polytopia/runs/RUN/pre-obs3 \
        --parent-in /data/polytopia/ckpt/scratch-init/hero-1v1-init-4x128-obs2-s1.npz \
        --parent-out /data/polytopia/ckpt/scratch-init/hero-1v1-init-4x128-obs3-s1.npz [--block-pool] [--check 64]

Every policy checkpoint of the run (latest.npz, snapshots/update-*.npz: weights plus Adam state over (policy,
critic)), every critic checkpoint (latest-critic.npz, snapshots/update-*-critic.npz) and the run's parent get zero
input rows for the new features (network.widen_inputs), so each computes exactly what it computed before, and their
JSON configs say obs_version 3. The originals are copied to --backup first (the rollback point). --check plays a few
random positions and asserts the widened policy and critic agree with the originals on them. Run it with every
process of the run stopped, on the new code; then point the run's plan.json parent at --parent-out and restart.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def seed_moments(widen, net, old_v, new_v):
    """new_v with every position the widening created (absent from `net`) set to the median of the old tensors' mean
    second moments."""
    import jax
    import jax.numpy as jnp
    mask = widen(jax.tree.map(jnp.ones_like, net))
    fill = float(np.median([float(jnp.mean(x)) for x in jax.tree.leaves(old_v)]))
    return jax.tree.map(lambda x, keep: jnp.where(keep > 0, x, jnp.asarray(fill, x.dtype)), new_v, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--backup', required=True)
    ap.add_argument('--parent-in', required=True)
    ap.add_argument('--parent-out', required=True)
    ap.add_argument('--check', type=int, default=64, help='positions to compare old vs widened networks on (0 = skip)')
    ap.add_argument('--block-pool', action='store_true', help='also add zero pooling layers to the trunk blocks')
    args = ap.parse_args()
    import jax
    from polytopia_bot.rl import network, ppo
    from polytopia_bot.rl.distributed import _critic_template, _template_params, feature_shapes, network_spec
    from polytopia_bot.rl.observation import OBS_VERSION
    from polytopia_bot.rl.runner import Config, load_checkpoint, save_checkpoint
    assert OBS_VERSION == 3
    run = Path(args.run_dir)
    backup = Path(args.backup)
    policies = sorted([run / 'latest.npz'] + list((run / 'snapshots').glob('update-*.npz')))
    policies = [p for p in policies if not p.name.endswith('-critic.npz')]
    critics = [p.with_name(p.stem + '-critic.npz') for p in policies]
    critics = [c for c in critics if c.exists()]
    for p in policies + critics:
        if not p.with_suffix('.json').exists():
            raise SystemExit(f'missing json for {p}')
    specs = {(network_spec(p).get('obs_version', 1), network_spec(p).get('block_pool', False)) for p in policies + critics}
    if specs != {(2, False)}:
        raise SystemExit(f'checkpoints are not all observation version 2 without block_pool: {specs}')
    spec2 = network_spec(policies[0])
    spec3 = dict(spec2, obs_version=3, **(dict(block_pool=True) if args.block_pool else {}))
    players = spec2.get('players', 2)
    b2, g2 = feature_shapes(players, 2)
    b3, g3 = feature_shapes(players, 3)
    t_policy2, t_policy3 = _template_params(**spec2), _template_params(**spec3)
    t_critic2, t_critic3 = _critic_template(**spec2), _critic_template(**spec3)
    pool = network.add_block_pool if args.block_pool else (lambda p: p)
    widen_policy = lambda p: pool(network.widen_inputs(p, b2, b3, g2, g3))
    widen_critic = lambda c: pool(network.widen_inputs(c, b2, b3, g2, g3, groups=players))

    # 1. backup (never overwrite an existing backup: it is the rollback point of a previous attempt)
    backup.mkdir(parents=True, exist_ok=False)
    (backup / 'snapshots').mkdir()
    for p in policies + critics:
        for f in (p, p.with_suffix('.json')):
            shutil.copy2(f, backup / f.relative_to(run))
    print(json.dumps(dict(stage='backed_up', files=2 * (len(policies) + len(critics)), to=str(backup))))

    def config3(path):
        return Config(**{**json.loads(path.with_suffix('.json').read_text())['config'], 'obs_version': 3,
                         'block_pool': bool(args.block_pool)})

    def stage_of(path):
        return json.loads(path.with_suffix('.json').read_text())['stage']

    # 2. critics first (the policies' Adam state spans (policy, critic))
    checks = []
    for c in critics:
        c2 = load_checkpoint(c, t_critic2)
        c3 = widen_critic(c2)
        save_checkpoint(c, c3, config3(c), stage=stage_of(c))
        checks.append(('critic', c, c2, c3))
        print(json.dumps(dict(stage='widened', file=str(c))))
    for p in policies:
        p2 = load_checkpoint(p, t_policy2)
        p3 = widen_policy(p2)
        critic_file = p.with_name(p.stem + '-critic.npz')
        has_critic = critic_file.exists()
        with np.load(p, allow_pickle=False) as archive:
            opt_keys = sorted((k for k in archive.files if k.startswith('o')), key=lambda k: int(k[1:]))
            opt_values = [archive[k] for k in opt_keys]
        opt3 = None
        if opt_values:
            # Adam state (ppo.adam_init): (step, m, v) over (policy, critic or None); widen m and v like the weights
            crit2 = load_checkpoint(backup / critic_file.relative_to(run), t_critic2) if has_critic else None
            template = ppo.adam_init((p2, crit2))
            leaves, structure = jax.tree.flatten(template)
            if len(leaves) != len(opt_values) or any(a.shape != b.shape for a, b in zip(leaves, opt_values)):
                raise SystemExit(f'{p}: Adam state does not match the version-2 (policy, critic) layout')
            step, m, v = structure.unflatten([jax.numpy.asarray(x) for x in opt_values])
            wide = lambda tree: (widen_policy(tree[0]), widen_critic(tree[1]) if tree[1] is not None else None)
            # Zero second moments would give every new weight a 3-6x step for the first few hundred updates (Adam's
            # bias correction is long done at this step count): seed them with the median second moment of the
            # existing tensors instead; first moments start at zero.
            nets = (p2, crit2)
            v3 = tuple(None if n is None else seed_moments(widen, n, old, new) for widen, n, old, new
                       in zip((widen_policy, widen_critic), nets, v, wide(v)))
            opt3 = (step, wide(m), v3)
        save_checkpoint(p, p3, config3(p), stage=stage_of(p), optimizer=opt3)
        checks.append(('policy', p, p2, p3))
        print(json.dumps(dict(stage='widened', file=str(p), adam=opt3 is not None)))
    # 3. the parent (the learner builds its template from it)
    parent_in, parent_out = Path(args.parent_in), Path(args.parent_out)
    if network_spec(parent_in) != spec2:
        raise SystemExit(f'parent {parent_in} is {network_spec(parent_in)}, the run is {spec2}')
    save_checkpoint(parent_out, widen_policy(load_checkpoint(parent_in, t_policy2)), config3(parent_in), stage=stage_of(parent_in))
    print(json.dumps(dict(stage='parent_written', path=str(parent_out), spec=network_spec(parent_out))))
    # 4. reload everything with the version-3 templates and compare on random positions
    for kind, path, old, _ in checks:
        load_checkpoint(path, t_critic3 if kind == 'critic' else t_policy3)
    if args.check:
        import jax.numpy as jnp
        from polytopia_bot.rl.distributed import omniscient_features
        from polytopia_bot.rl.rust_engine import RustGames, split_masks
        from polytopia_bot.rl.actions import shapes
        size = spec2['size']
        g = RustGames(size, 4, both_views=True)
        rng = np.random.default_rng(0)
        shp = shapes(size)
        _, prepared = g.reset(np.arange(4, dtype=np.int32) + 900)
        worst = 0.
        trunk = jax.jit(jax.vmap(lambda p, x: network.trunk(p, x), in_axes=(None, 0)))
        for _ in range(args.check):
            masks = prepared.masks
            choice = [np.zeros(4, np.int32) for _ in range(3)]
            for k in range(4):
                mats = [np.asarray(m[k]).reshape(shp[f]) for f, m in enumerate(masks)]
                fams = [f for f in range(17) if mats[f].any()]
                f = 15 if 15 in fams and rng.random() < .2 else int(rng.choice(fams))
                s = int(rng.choice(np.flatnonzero(mats[f].any(1))))
                choice[0][k], choice[1][k], choice[2][k] = f, s, int(rng.choice(np.flatnonzero(mats[f][s])))
            tick, _ = g.tick(None, choice, np.arange(4, dtype=np.int32) + 1000)
            prepared = tick.prepared
            own = prepared.features
            opp = g._features(prepared.opp_view)
            both = omniscient_features(own, opp)
            for kind, path, old, new in checks:
                x = both if kind == 'critic' else own
                a, b = trunk(old, x), trunk(new, x)
                worst = max(worst, max(float(jnp.max(jnp.abs(u - w))) for u, w in zip(a, b)))
        print(json.dumps(dict(stage='checked', positions=args.check, worst_abs_diff=worst)))
        if not worst < 1e-3:
            raise SystemExit('widened networks differ from the originals')
    print(json.dumps(dict(status='ok', policies=len(policies), critics=len(critics), parent=str(parent_out), backup=str(backup))))


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
