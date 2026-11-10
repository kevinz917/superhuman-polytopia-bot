"""Session E budget ledger (cap in the file). reserve NAME USD [note] | reconcile NAME USD [note] | show

Every paid call is reserved at its container-hour BOUND (timeout x GPUs x list price, retries 0) before launch and
reconciled to measured container-seconds afterwards; a reserve that would cross the cap is refused."""
import json
import sys
import time
from pathlib import Path

LEDGER = Path('/Users/Shared/PolytopiaClaude/v3-budget.json')


def committed(entries: list) -> float:
    return sum(e.get('actual_usd', e['reserved_usd']) for e in entries)


def main(argv: list) -> None:
    data = json.loads(LEDGER.read_text())
    cmd = argv[0]
    now = time.strftime('%Y-%m-%d %H:%M %Z')
    if cmd == 'reserve':
        name, usd = argv[1], float(argv[2])
        if committed(data['entries']) + usd > data['cap_usd']:
            sys.exit(f"REFUSED: {committed(data['entries']):.2f} + {usd:.2f} > cap {data['cap_usd']}")
        data['entries'].append(dict(name=name, reserved_usd=usd, reserved_at=now, note=' '.join(argv[3:])))
    elif cmd == 'reconcile':
        name, usd = argv[1], float(argv[2])
        entry = [e for e in data['entries'] if e['name'] == name][-1]
        entry.update(actual_usd=usd, reconciled_at=now, reconcile_note=' '.join(argv[3:]))
    elif cmd != 'show':
        sys.exit(__doc__)
    LEDGER.write_text(json.dumps(data, indent=2))
    open_entries = [e['name'] for e in data['entries'] if 'actual_usd' not in e]
    print(f"committed {committed(data['entries']):.2f} / cap {data['cap_usd']:.2f}; entries {len(data['entries'])}; "
          f"unreconciled {open_entries}")


if __name__ == '__main__':
    main(sys.argv[1:])
