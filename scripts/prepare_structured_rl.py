"""Prepare an audited warm-start subset without changing the SFT corpus."""
import argparse
import json
from polytopia_bot.rl.data import prepare

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--records', default='runs/sft0/main-candidate-sample-v2/records.jsonl')
    p.add_argument('--manifest', default='runs/sft0/splits-v1/maps.json')
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=2048)
    args = p.parse_args()
    print(json.dumps(prepare(args.records, args.manifest, args.output, count=args.count), indent=2))
