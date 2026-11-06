"""Print campaign-3 ladder rows (registry Dict) with per-map-type breakdown. Run with Modal's Python."""
import json
import sys
import modal

registry = modal.Dict.from_name('polytopia-distributed-rl-registry', environment_name='staging')
for tag in sys.argv[1:] or ['c3-mixed-v2', 'c3-classic']:
    for r in registry.get(f'{tag}/history', []):
        if 'error' in r:
            print(tag, r['run_id'], r.get('stage'), 'ERROR', r['error'][:200])
            continue
        decisive = r['wins'] + r['losses']
        print(f"{tag:12s} {r['run_id']:13s} {r['stage']:12s} vs {r['reference']:8s} W{r['wins']:3d} L{r['losses']:3d} "
              f"trunc {r['truncations']:3d}  decisive-win {r['wins'] / max(decisive, 1):.2f}  ci {r['ci95']}")
        for kind, m in sorted((r.get('by_map') or {}).items()):
            if len(r.get('by_map') or {}) > 1:
                print(f"      {kind:12s} W{m['wins']:3d} L{m['losses']:3d} T{m['truncations']:3d} turns {m['mean_turns']:5.1f}")
