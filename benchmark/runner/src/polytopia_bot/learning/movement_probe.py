"""Prediction-independent sampling and exact parent-token validation for a fit probe."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from .training import file_hash


def stratum(row):
    n = len(row['scores'])
    return row['phase'], 'under64' if n < 64 else '64to127' if n < 128 else '128plus'


def select_positions(rows, count, *, tag, excluded_maps=()):
    """Round-robin over phase/catalog strata, selecting at most one state per map."""
    pools = defaultdict(list)
    for row in rows:
        pools[stratum(row)].append(row)
    for pool in pools.values():
        pool.sort(key=lambda r: hashlib.sha256((tag+':'+r['position_id']).encode()).hexdigest())
    selected = []
    used = set(excluded_maps)
    while len(selected) < count:
        progressed = False
        for key in sorted(pools):
            while pools[key] and pools[key][0]['board_hash'] in used:
                pools[key].pop(0)
            if pools[key]:
                row = pools[key].pop(0)
                selected.append(row)
                used.add(row['board_hash'])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError('insufficient distinct maps for probe selection')
    return selected


def verify_subset(prepared, manifest, parent_run, config):
    """Prove the fit tokens/labels are an unchanged subset of the parent's corpus."""
    provenance = manifest['subset_provenance']
    parent = json.loads(Path(parent_run).read_text())
    if parent['status'] != 'completed':
        raise ValueError('completed parent required')
    parent_manifest = parent['data_manifest']
    if provenance['parent_tokens_sha256'] != parent_manifest['tokens_sha256']:
        raise ValueError('probe parent tokens differ')
    allowed = {'epochs', 'effective_batch_size'}
    if {k:v for k,v in config.items() if k not in allowed} != {k:v for k,v in parent['config'].items() if k not in allowed}:
        raise ValueError('probe changed settings beyond epochs/batch size')
    parent_tokens = Path(provenance['parent_prepared'])/'tokens.jsonl'
    if file_hash(parent_tokens) != provenance['parent_tokens_sha256']:
        raise ValueError('parent tokens changed')
    indices = provenance['parent_indices']
    if len(indices) != 128 or indices != sorted(set(indices)):
        raise ValueError('probe requires 128 distinct ordered parent indices')
    wanted = set(indices)
    with parent_tokens.open() as stream:
        expected = [json.loads(line) for i,line in enumerate(stream) if i in wanted]
    with (Path(prepared)/'tokens.jsonl').open() as stream:
        actual = list(map(json.loads, stream))
    if actual != expected:
        raise ValueError('probe tokens or labels changed from parent')
    return parent
