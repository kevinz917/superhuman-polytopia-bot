"""Billed Modal cost per ledger row, from `modal billing report` (hourly, staging).

    python scripts/ledger_spend.py runs/rl-18x18-v1 [--since 2026-10-19] [--write]

Every row with a modal_app_id gets its billed total; --write stores max(billed, reservation) as spend_usd on rows whose app has
stopped (budget.committed() then counts spend instead of the reservation). Rows of live apps keep their
reservation. Billing lags by up to an hour, so totals for just-stopped apps can still grow."""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from polytopia_bot.rl.budget import committed, record  # noqa: E402


def billed(since):
    raw = subprocess.run(['modal', 'billing', 'report', '--start', since, '-r', 'h', '--json'],
                         capture_output=True, text=True, check=True).stdout
    cost, last = defaultdict(float), {}
    for row in json.loads(raw):
        if row.get('environment') == 'staging':
            cost[row['object_id']] += float(row['cost'])
            last[row['object_id']] = max(last.get(row['object_id'], ''), row['interval_start'])
    return cost, last


def app_list():
    raw = subprocess.run(['modal', 'app', 'list', '--env', 'staging', '--json'], capture_output=True, text=True,
                         check=True).stdout
    return {a['app_id']: a for a in json.loads(raw)}


SETTLED_HOURS = 2  # hourly billing rows are complete this long after an app stops


def main():
    folder = Path(sys.argv[1])
    since = sys.argv[sys.argv.index('--since') + 1] if '--since' in sys.argv else '2026-10-19'
    ledger = json.loads((folder / 'budget.json').read_text())
    from datetime import datetime, timezone
    (cost, last), apps = billed(since), app_list()
    live = {a for a, x in apps.items() if x['stopped_at'] is None and x['state'] != 'deployed'}
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # billing hours are UTC
    # Settled: not live, and its last billed hour began more than SETTLED_HOURS ago (stopped apps drop out
    # of `modal app list`, so the billing rows are the evidence).
    settled = {a for a, t in last.items() if a not in live and
               (now - datetime.fromisoformat(t)).total_seconds() > (SETTLED_HOURS + 1) * 3600}
    total_billed = 0.
    for r in ledger['runs']:
        ids = [a for a in [r.get('modal_app_id'), *r.get('extra_app_ids', [])] if a]
        spent = sum(cost.get(a, 0.) for a in ids)
        total_billed += spent
        running = any(a in live for a in ids)
        print(f"{r['run_id']:28s} reserved {r['reserved_usd']:8.2f} billed {spent:8.2f} {'LIVE' if running else ''}")
        if '--write' in sys.argv and ids and not running:
            # Settled apps: billed is the spend. Recently stopped: never less than the row's own bound (lag).
            final = all(a in settled for a in ids)
            record(folder, r['run_id'], spend_usd=round(spent if final else max(spent, float(r['reserved_usd'])), 2),
                   billed_usd=round(spent, 2), billing_settled=final)
    ledger = json.loads((folder / 'budget.json').read_text())
    print(f"billed {total_billed:.2f}; committed against the cap {committed(ledger):.2f} of {ledger['budget_usd']}")


if __name__ == '__main__':
    main()
