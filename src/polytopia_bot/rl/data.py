"""Optional behavioral-cloning warm start from the existing SFT demonstrations."""
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from .actions import Choice, PathMasks, from_catalog
from .observation import Features, encode_view, view_from_public


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(records, manifest_path, output, *, count=2048, seed=17):
    if count < 0:
        raise ValueError('example count must be nonnegative')
    manifest = json.loads(Path(manifest_path).read_text())
    size = manifest['size']
    groups = [manifest['maps'][k] for k in ('train', 'validation', 'test')]
    for field in ('seed', 'board_hash', 'symmetry_hash'):
        sets = [set(r[field] for r in group) for group in groups]
        if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('training/validation/test overlap')
    allowed = {r['board_hash'] for r in groups[0]}
    # A pure-RL run needs only the frozen split manifest, no demonstrations.
    if count == 0:
        root = Path(output)
        root.mkdir(parents=True, exist_ok=False)
        protocol = dict(schema='polytopia-structured-rl-input-v1', size=size, examples=0,
                        manifest_sha256=sha256(manifest_path), manifest=str(manifest_path),
                        simulator_sha256=manifest['simulator_sha256'],
                        train_seeds=[r['seed'] for r in groups[0]],
                        validation_seeds=[r['seed'] for r in groups[1]], selected=[], data_sha256=None)
        (root / 'input.json').write_text(json.dumps(protocol, indent=2) + '\n')
        return protocol
    # Bounded-memory reservoir samples the entire corpus, not just its opening.
    rng = np.random.default_rng(seed)
    rows = []
    seen = 0
    for line in Path(records).open():
        row = json.loads(line)
        if row['schema'] != 'polytopia-demo-v1' or row['split'] != 'train' or row['source'] != 'teacher' or row['map_hash'] not in allowed:
            raise ValueError('only existing teacher training-split demonstrations are accepted')
        if row['observation']['visible_state']['board_size'] != size:
            raise ValueError('board size differs from manifest')
        seen += 1
        if len(rows) < count:
            rows.append(row)
        else:
            at = rng.integers(seen)
            if at < count:
                rows[at] = row
    if len(rows) != count:
        raise ValueError(f'need {count} examples; found {len(rows)}')
    choices, masks, views = [], [], []
    for r in rows:
        obs = r['observation']
        match = [a for a in obs['legal_actions'] if a['command'] == r['selected_command']]
        if len(match) != 1:
            raise ValueError('teacher command must uniquely match the legal catalog')
        selected = int(match[0]['action_id'])
        choice, path = from_catalog([int(a['action_id']) for a in obs['legal_actions']], selected, size)
        choices.append(choice)
        masks.append(path)
        views.append(view_from_public(obs))
    stack = lambda xs: jax.tree.map(lambda *x: np.stack(x), *xs)
    features = jax.jit(jax.vmap(lambda v: encode_view(v, size)))(stack(views))
    batch = dict(features=features, choice=stack(choices), masks=stack(masks))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    save_batch(root / 'demonstrations.npz', batch)
    # Use the original reserved seeds; final test maps never enter the runner.
    protocol = dict(schema='polytopia-structured-rl-input-v1', size=size, examples=count,
                    records_sha256=sha256(records), manifest_sha256=sha256(manifest_path),
                    source=str(records), manifest=str(manifest_path),
                    simulator_sha256=manifest['simulator_sha256'],
                    train_seeds=[r['seed'] for r in groups[0]],
                    validation_seeds=[r['seed'] for r in groups[1]],
                    selected=[dict(episode_id=r['episode_id'], decision_index=r['decision_index'],
                                   map_hash=r['map_hash'], teacher_hash=r['teacher_hash']) for r in rows])
    protocol['data_sha256'] = sha256(root / 'demonstrations.npz')
    (root / 'input.json').write_text(json.dumps(protocol, indent=2) + '\n')
    return {k: v for k, v in protocol.items() if k not in ('selected', 'train_seeds', 'validation_seeds')}


def save_batch(path, batch):
    arrays = {f'{k}_{name}': np.asarray(value) for k, group in batch.items()
              for name, value in zip(group._fields, group, strict=True)}
    np.savez_compressed(path, **arrays)


def load_batch(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: cls(*(data[f'{k}_{name}'] for name in cls._fields))
                for k, cls in (('features', Features), ('choice', Choice), ('masks', PathMasks))}
