"""Atomic local cost ledger for bounded Modal experiments.

Reservations survive success, cancellation and error until an explicit terminal
app-receipt accounting pass. They are bounds with margin, not invoices.
A ledger with budget_usd set refuses any reservation that would take the sum of
reservations (or recorded spend_usd, once a row has it) above that cap; budget_usd
null means no cap. Run IDs are unique. Concurrent clients cannot overwrite another
client's record.
"""
import fcntl
import hashlib
import json
import math
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_CEILING
from pathlib import Path


def mutate(folder, change):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'budget.json'
    with (folder / '.budget.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = json.loads(path.read_text()) if path.exists() else {
            'budget_usd': None, 'reservation_per_run_usd': 5,
            'price_source': 'https://modal.com/pricing', 'runs': []}
        result = change(ledger)
        temp = folder / 'budget.json.tmp'
        temp.write_text(json.dumps(ledger, indent=2)+'\n')
        temp.replace(path)
        return result


def reserve(folder, row):
    def change(ledger):
        if any(r['run_id'] == row['run_id'] for r in ledger['runs']):
            raise ValueError('Run ID already reserved; inspect the existing job')
        if row.get('phase') == 'final_acceptance':
            seed = row.get('final_seed_base')
            if type(seed) is not int or seed < 400000 or (seed-400000)%32:
                raise ValueError('Invalid final partition')
            if any(r.get('phase') == 'final_acceptance' and r.get('final_seed_base') == seed
                   for r in ledger['runs']):
                raise ValueError('Final partition already reserved; never reuse it for selection')
        if row['reserved_usd'] <= 0:
            raise ValueError('reservation must be positive')
        cap = ledger.get('budget_usd')
        if cap is not None:
            held = committed(ledger)
            if held + row['reserved_usd'] > cap + 1e-9:
                raise ValueError(f'budget cap: ${held:.2f} committed + ${row["reserved_usd"]:.2f} exceeds ${cap:.2f}')
        ledger['runs'].append(row)
    mutate(folder, change)


def committed(ledger):
    """Recorded spend where known (spend_usd), else the reservation minus any released capacity."""
    return sum(float(r['spend_usd']) if 'spend_usd' in r else float(r['reserved_usd']) - float(r.get('released_usd', 0))
               for r in ledger['runs'])


def release(folder, run_id, amount, reason):
    """Release part of a live reservation that can no longer be spent (e.g. a dead container's remaining
    hours). Cumulative, never above the reservation; spend_usd, once recorded, supersedes it."""
    def change(ledger):
        row = next(r for r in ledger['runs'] if r['run_id'] == run_id)
        total = float(row.get('released_usd', 0)) + float(amount)
        if amount <= 0 or total > row['reserved_usd']:
            raise ValueError('release must be positive and within the reservation')
        row['released_usd'] = round(total, 2)
        row.setdefault('releases', []).append(dict(amount_usd=float(amount), reason=reason))
    mutate(folder, change)


def record(folder, run_id, **fields):
    def change(ledger):
        found = next(r for r in ledger['runs'] if r['run_id'] == run_id)
        if any(key in fields for key in ('reserved_usd','run_id','phase','final_seed_base')):
            raise ValueError('cannot alter reservation identity, amount or final partition')
        found.update(fields)
    mutate(folder, change)


def account_terminal_app(folder, run_id, app_receipt, *, rate_usd_per_hour=2.584224, margin_usd=1.):
    """Replace a completed single-worker reservation with observed wall cost + margin.

    Caller must verify the declared bounded allocation and a fresh terminal
    app-list receipt. Legacy rows retain the L40S allocation floor; the explicitly
    declared CPU checkpoint-retainer profile has its own conservative floor.
    This is conservative accounting, not an invoice. Running, unknown or
    mismatched applications never release their reservations.
"""
    if app_receipt['state']!='stopped' or int(app_receipt['tasks'])!=0:
        raise ValueError('application is not confirmed terminal')
    if not math.isfinite(rate_usd_per_hour) or rate_usd_per_hour<=0 or not math.isfinite(margin_usd) or margin_usd<1:
        raise ValueError('insufficient rate or overhead margin')
    duration=(datetime.fromisoformat(app_receipt['stopped_at'])-
              datetime.fromisoformat(app_receipt['created_at'])).total_seconds()
    if duration<0:
        raise ValueError('invalid app duration')
    cost = Decimal(str(duration))*Decimal(str(rate_usd_per_hour))/Decimal(3600)+Decimal(str(margin_usd))
    amount = float(cost.quantize(Decimal('.01'), rounding=ROUND_CEILING))
    def change(ledger):
        row=next(r for r in ledger['runs'] if r['run_id']==run_id)
        if row['modal_app_id']!=app_receipt['app_id']:
            raise ValueError('receipt belongs to a different application')
        floor = 2.584224
        if row.get('resource_type') in ('cpu-checkpoint-retainer', 'cpu-checkpoint-verifier'):
            if (row.get('gpu_type') is not None or row.get('cpu_cores') != 1
                    or row.get('memory_gib') != 2 or row.get('nonpreemptible',False)):
                raise ValueError('invalid CPU retention allocation')
            floor = .07
        elif row.get('resource_type') in ('cpu-cloud-comparisons', 'cpu-cloud-qualification'):
            if (row.get('gpu_type') is not None or row.get('cpu_cores') != 6
                    or row.get('memory_gib') != 16 or row.get('nonpreemptible',False)):
                raise ValueError('invalid CPU evaluation allocation')
            floor = .42
        elif row.get('resource_type') == 'cpu-baseline-recovery':
            if (row.get('gpu_type') is not None or row.get('cpu_cores') != 4
                    or row.get('memory_gib') != 12 or row.get('nonpreemptible') is not True):
                raise ValueError('invalid nonpreemptible CPU recovery allocation')
            floor = .86
        if rate_usd_per_hour < max(floor, row.get('rate_usd_per_hour', floor)):
            raise ValueError('accounting rate is below the reserved resource rate')
        if 'terminal_accounting' in row:
            raise ValueError('application already accounted')
        if amount>row['reserved_usd']:
            raise ValueError('observed accounting exceeds reservation; audit required')
        row['original_reserved_usd']=row['reserved_usd']
        row['reserved_usd']=amount
        row['terminal_accounting']=dict(receipt=app_receipt,app_wall_seconds=duration,
            rate_usd_per_hour=rate_usd_per_hour,compute_estimate_usd=duration*rate_usd_per_hour/3600,
            overhead_margin_usd=margin_usd,scope='observed app wall time at maximum declared allocation plus margin; not an invoice')
        return amount
    return mutate(folder,change)


def pool_terminal_overheads(folder, earlier_report, later_report,
                            *, buffer_run_id='campaign-shared-overhead'):
    """Pool duplicated margins only for confirmed-terminal, stably billed apps.

    Keep a bound on full app-wall compute, rounding its hourly rate upward and
    padding timestamp precision by two seconds. A separately reserved $5 buffer
    covers campaign storage/egress and other overhead. This does not turn a
    reservation into spend or release any live application's reservation.
    """
    reports, hashes, costs = [], [], []
    for path in (earlier_report, later_report):
        raw = Path(path).read_bytes()
        report = json.loads(raw)
        start = datetime.fromisoformat(report['start_inclusive'])
        end = datetime.fromisoformat(report['end_exclusive'])
        observed = datetime.fromisoformat(report['observed_at'])
        if (any(t.tzinfo is None for t in (start, end, observed))
                or not start < end <= observed or report['resolution'] != 'h'):
            raise ValueError('invalid completed-hour billing report')
        by_app = {}
        for row in report['rows']:
            value = Decimal(row['cost_usd'])
            if (not value.is_finite() or value < 0
                    or row['environment_name'] != 'staging'
                    or not start <= datetime.fromisoformat(row['interval_start']) < end):
                raise ValueError('invalid billing row')
            by_app[row['object_id']] = by_app.get(row['object_id'], Decimal(0)) + value
        if sum(by_app.values(), Decimal(0)) != Decimal(report['owned_reported_cost_usd']):
            raise ValueError('billing report total mismatch')
        reports.append((start, end, observed))
        hashes.append(hashlib.sha256(raw).hexdigest())
        costs.append(by_app)
    if (reports[1][2] - reports[0][2] < timedelta(hours=1)
            or reports[1][1] < reports[0][1]):
        raise ValueError('billing observations are not sufficiently separated')

    def change(ledger):
        buffer = next((r for r in ledger['runs'] if r['run_id'] == buffer_run_id), None)
        if (buffer is None or buffer.get('resource_type') != 'overhead-buffer'
                or buffer['reserved_usd'] < 5 or buffer.get('modal_app_id')):
            raise ValueError('requires separately reserved campaign overhead buffer')
        pooled, skipped = [], []
        for row in ledger['runs']:
            original = row.get('terminal_accounting')
            if original is None or 'pooled_terminal_accounting' in row:
                continue
            receipt = original['receipt']
            app_id = row['modal_app_id']
            if (receipt['app_id'] != app_id or receipt['state'] != 'stopped'
                    or int(receipt['tasks']) != 0):
                raise ValueError('terminal accounting lacks a matching stopped receipt')
            created = datetime.fromisoformat(receipt['created_at'])
            stopped = datetime.fromisoformat(receipt['stopped_at'])
            duration = (stopped-created).total_seconds()
            if duration < 0 or duration != original['app_wall_seconds']:
                raise ValueError('terminal duration mismatch')
            covered = all(start <= created and end >= stopped for start, end, _ in reports)
            stable = app_id in costs[0] and costs[0].get(app_id) == costs[1].get(app_id)
            if not covered or not stable or reports[1][1] < stopped + timedelta(hours=1):
                skipped.append(row['run_id'])
                continue
            rate = Decimal(str(original['rate_usd_per_hour']))
            if not rate.is_finite() or rate <= 0:
                raise ValueError('invalid terminal allocation rate')
            rate = rate.quantize(Decimal('.01'), rounding=ROUND_CEILING)
            bound = ((Decimal(str(duration))+2)*rate/3600).quantize(
                Decimal('.01'), rounding=ROUND_CEILING)
            previous = Decimal(str(row['reserved_usd']))
            if not costs[1][app_id] <= bound <= previous:
                raise ValueError('stable billing or compute bound exceeds reservation; audit required')
            detail = dict(previous_reserved_usd=float(previous), compute_bound_usd=float(bound),
                rounded_rate_usd_per_hour=float(rate), timestamp_padding_seconds=2,
                stable_reported_cost_usd=str(costs[1][app_id]), billing_report_sha256=hashes,
                shared_overhead_buffer_run_id=buffer_run_id,
                scope='Full app-wall compute bound; overhead pooled separately. Not invoice spend.')
            row['reserved_usd'] = float(bound)
            row['pooled_terminal_accounting'] = detail
            pooled.append(dict(run_id=row['run_id'], **detail))
        released = sum((Decimal(str(r['previous_reserved_usd']))-
                        Decimal(str(r['compute_bound_usd'])) for r in pooled), Decimal(0))
        return dict(pooled=pooled, skipped=skipped, released_usd=float(released),
                    shared_overhead_buffer_run_id=buffer_run_id)
    return mutate(folder, change)
