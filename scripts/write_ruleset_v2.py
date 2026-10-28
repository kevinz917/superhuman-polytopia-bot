"""Write configs/ruleset-v2.json: the v1 manifest extended with the 18x18 board (docs/ruleset-v2.md).

Source hashes are recomputed from the working tree; v1 receipts stay valid for 11/14 boards."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    v1 = json.loads((ROOT / 'configs/ruleset-v1.json').read_text())
    gpu = ROOT / 'src/polytopia_bot/gpu'
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted([*gpu.glob('*.py'), gpu / 'rules.json'])}
    v2 = dict(v1, ruleset_id='bardur-1v1-v2', frozen_date='2026-10-19',
              extends='bardur-1v1-v1', generator_revision='bardur-drylands-dotnet-v2',
              variant_generator_revision='bardur-variants-v2', board_sizes=[11, 14, 18], primary_board_size=14,
              round_limit={'11': 100, '14': 100, '18': 150}, rules_contract='docs/ruleset-v2.md',
              compatible_engine_digests={'7b76e65b74afcd8d4732712b12a2f9e001fafc1a5b2899982b971a9fd78cb134': 'bardur-1v1-v1'},
              source_sha256=sources,
              receipts=dict(v1['receipts'],
                            mapgen_oracle_18='tests/fixtures/corrected-drylands.json.gz (20 Large cases)',
                            cpu_episodes_18='docs/benchmarks/2026-10-19-cpu-native-episodes-18.json',
                            tests_18='docs/benchmarks/2026-10-19-tests-18.txt'))
    (ROOT / 'configs/ruleset-v2.json').write_text(json.dumps(v2, indent=2) + '\n')
    print(f"ruleset-v2: {len(sources)} source hashes")


if __name__ == '__main__':
    main()
