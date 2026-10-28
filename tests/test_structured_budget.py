from concurrent.futures import ThreadPoolExecutor
import json
import pytest
from polytopia_bot.rl.budget import reserve, record, account_terminal_app, pool_terminal_overheads


def test_concurrent_reservations_never_lose_records_and_have_no_cap(tmp_path):
    def launch(i):
        reserve(tmp_path, dict(run_id=str(i), reserved_usd=50, status='reserved'))
        record(tmp_path, str(i), status='complete')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(launch, range(32)))
    ledger = json.loads((tmp_path / 'budget.json').read_text())
    assert ledger['budget_usd'] is None
    assert len(ledger['runs']) == 32
    assert sum(r['reserved_usd'] for r in ledger['runs']) == 1600
    assert all(r['status'] == 'complete' for r in ledger['runs'])


def test_duplicate_run_is_rejected_and_failures_remain_reserved(tmp_path):
    reserve(tmp_path, dict(run_id='one', reserved_usd=5, status='reserved'))
    record(tmp_path, 'one', status='failed')
    with pytest.raises(ValueError, match='already reserved'):
        reserve(tmp_path, dict(run_id='one', reserved_usd=5))
    rows = json.loads((tmp_path / 'budget.json').read_text())['runs']
    assert rows == [dict(run_id='one', reserved_usd=5, status='failed')]


def receipt(**changes):
    return dict(app_id='app-one',state='stopped',tasks='0',
        created_at='2026-10-18 00:00:00+00:00',stopped_at='2026-10-18 01:00:00+00:00',**changes)


def test_terminal_accounting_releases_only_verified_unused_reservation(tmp_path):
    reserve(tmp_path,dict(run_id='one',modal_app_id='app-one',reserved_usd=5))
    assert account_terminal_app(tmp_path,'one',receipt())==3.59
    row=json.loads((tmp_path/'budget.json').read_text())['runs'][0]
    assert row['original_reserved_usd']==5
    assert row['terminal_accounting']['app_wall_seconds']==3600
    with pytest.raises(ValueError,match='already accounted'):
        account_terminal_app(tmp_path,'one',receipt())


@pytest.mark.parametrize('changes',[dict(state='ephemeral (detached)'),dict(tasks='1'),dict(app_id='other')])
def test_live_or_wrong_app_cannot_release_budget(tmp_path,changes):
    reserve(tmp_path,dict(run_id='one',modal_app_id='app-one',reserved_usd=5))
    evidence=receipt();evidence.update(changes)
    with pytest.raises(ValueError):
        account_terminal_app(tmp_path,'one',evidence)
    assert json.loads((tmp_path/'budget.json').read_text())['runs'][0]['reserved_usd']==5


def test_more_expensive_gpu_cannot_use_default_lower_accounting_rate(tmp_path):
    reserve(tmp_path,dict(run_id='a100',modal_app_id='app-one',reserved_usd=5,
                         gpu_type='A100-80GB',rate_usd_per_hour=3.14))
    with pytest.raises(ValueError,match='reserved resource rate'):
        account_terminal_app(tmp_path,'a100',receipt())
    row=json.loads((tmp_path/'budget.json').read_text())['runs'][0]
    assert row['reserved_usd']==5 and 'terminal_accounting' not in row
    assert account_terminal_app(tmp_path,'a100',receipt(),rate_usd_per_hour=3.14)==4.14


def test_accounting_rounds_fractional_cents_up(tmp_path):
    reserve(tmp_path,dict(run_id='a100',modal_app_id='app-one',reserved_usd=5,
                         rate_usd_per_hour=3.14))
    evidence=receipt();evidence['stopped_at']='2026-10-18 01:00:01+00:00'
    assert account_terminal_app(tmp_path,'a100',evidence,rate_usd_per_hour=3.14)==4.15


@pytest.mark.parametrize('profile', ['cpu-checkpoint-retainer', 'cpu-checkpoint-verifier'])
def test_cpu_retention_profile_has_bounded_rate_and_margin(tmp_path, profile):
    reserve(tmp_path, dict(run_id='retainer', modal_app_id='app-one', reserved_usd=2,
        resource_type=profile, gpu_type=None, cpu_cores=1,
        memory_gib=2, rate_usd_per_hour=.07))
    with pytest.raises(ValueError, match='reserved resource rate'):
        account_terminal_app(tmp_path, 'retainer', receipt(), rate_usd_per_hour=.01)
    assert account_terminal_app(tmp_path, 'retainer', receipt(), rate_usd_per_hour=.07)==1.07


@pytest.mark.parametrize('profile', ['cpu-cloud-comparisons', 'cpu-cloud-qualification'])
def test_cpu_evaluation_profile_cannot_understate_declared_resources(tmp_path, profile):
    reserve(tmp_path, dict(run_id='evaluation', modal_app_id='app-one', reserved_usd=3,
        resource_type=profile, gpu_type=None, cpu_cores=6, memory_gib=16))
    with pytest.raises(ValueError, match='reserved resource rate'):
        account_terminal_app(tmp_path, 'evaluation', receipt(), rate_usd_per_hour=.07)
    assert account_terminal_app(tmp_path, 'evaluation', receipt(), rate_usd_per_hour=.42)==1.42


@pytest.mark.parametrize('fields', [{'gpu_type':'L40S'}, {'cpu_cores':8}, {'memory_gib':32},
                                    {'nonpreemptible':True}])
def test_cpu_evaluation_profile_rejects_resource_mismatch(tmp_path, fields):
    row = dict(run_id='evaluation', modal_app_id='app-one', reserved_usd=3,
        resource_type='cpu-cloud-qualification', gpu_type=None, cpu_cores=6, memory_gib=16)
    row.update(fields)
    reserve(tmp_path, row)
    with pytest.raises(ValueError, match='invalid CPU evaluation allocation'):
        account_terminal_app(tmp_path, 'evaluation', receipt(), rate_usd_per_hour=.42)


def test_nonpreemptible_recovery_includes_threefold_cpu_memory_pricing(tmp_path):
    reserve(tmp_path,dict(run_id='recovery',modal_app_id='app-one',reserved_usd=2,
        resource_type='cpu-baseline-recovery',gpu_type=None,cpu_cores=4,memory_gib=12,
        nonpreemptible=True,rate_usd_per_hour=.86))
    with pytest.raises(ValueError,match='reserved resource rate'):
        account_terminal_app(tmp_path,'recovery',receipt(),rate_usd_per_hour=.29)
    assert account_terminal_app(tmp_path,'recovery',receipt(),rate_usd_per_hour=.86)==1.86


@pytest.mark.parametrize('fields', [{}, {'resource_type':'cpu-checkpoint-retainer',
    'gpu_type':'L40S','cpu_cores':1,'memory_gib':2}])
def test_cpu_rate_cannot_be_used_for_gpu_or_legacy_allocation(tmp_path, fields):
    reserve(tmp_path, dict(run_id='one', modal_app_id='app-one', reserved_usd=5, **fields))
    with pytest.raises(ValueError):
        account_terminal_app(tmp_path, 'one', receipt(), rate_usd_per_hour=.07)
    row=json.loads((tmp_path/'budget.json').read_text())['runs'][0]
    assert row['reserved_usd']==5 and 'terminal_accounting' not in row


def billing_reports(folder, *, later_cost='2.58', earlier_cost='2.58', earlier_end='02'):
    paths=[]
    for hour,cost in [(earlier_end,earlier_cost),('04',later_cost)]:
        path=folder/f'billing-{hour}.json'
        path.write_text(json.dumps(dict(start_inclusive='2026-10-18T00:00:00+00:00',
            end_exclusive=f'2026-10-18T{hour}:00:00+00:00',
            observed_at=f'2026-10-18T{hour}:01:00+00:00', resolution='h',
            owned_reported_cost_usd=cost, rows=[dict(object_id='app-one',
                cost_usd=cost,environment_name='staging',interval_start='2026-10-18T00:00:00+00:00')])) )
        paths.append(path)
    return paths


def test_margin_pooling_preserves_compute_bound_and_is_idempotent(tmp_path):
    reserve(tmp_path,dict(run_id='one',modal_app_id='app-one',reserved_usd=5))
    account_terminal_app(tmp_path,'one',receipt())
    reserve(tmp_path,dict(run_id='campaign-shared-overhead',resource_type='overhead-buffer',reserved_usd=5))
    paths=billing_reports(tmp_path)
    result=pool_terminal_overheads(tmp_path,*paths)
    assert result['released_usd']==.99 and result['pooled'][0]['compute_bound_usd']==2.60
    reserve(tmp_path,dict(run_id='live',modal_app_id='app-live',reserved_usd=2))
    assert pool_terminal_overheads(tmp_path,*paths)['released_usd']==0
    rows=json.loads((tmp_path/'budget.json').read_text())['runs']
    assert sum(r['reserved_usd'] for r in rows)==9.6
    assert next(r for r in rows if r['run_id']=='live')['reserved_usd']==2


def test_margin_pooling_requires_shared_buffer(tmp_path):
    reserve(tmp_path,dict(run_id='one',modal_app_id='app-one',reserved_usd=5))
    account_terminal_app(tmp_path,'one',receipt())
    with pytest.raises(ValueError,match='overhead buffer'):
        pool_terminal_overheads(tmp_path,*billing_reports(tmp_path))


def test_changing_billing_keeps_original_terminal_margin(tmp_path):
    reserve(tmp_path,dict(run_id='one',modal_app_id='app-one',reserved_usd=5))
    account_terminal_app(tmp_path,'one',receipt())
    reserve(tmp_path,dict(run_id='campaign-shared-overhead',resource_type='overhead-buffer',reserved_usd=5))
    result=pool_terminal_overheads(tmp_path,*billing_reports(tmp_path,later_cost='2.59'))
    assert result['released_usd']==0 and result['skipped']==['one']
    assert json.loads((tmp_path/'budget.json').read_text())['runs'][0]['reserved_usd']==3.59


def test_billing_above_compute_bound_requires_audit_without_mutating_ledger(tmp_path):
    reserve(tmp_path,dict(run_id='one',modal_app_id='app-one',reserved_usd=5))
    account_terminal_app(tmp_path,'one',receipt())
    reserve(tmp_path,dict(run_id='campaign-shared-overhead',resource_type='overhead-buffer',reserved_usd=5))
    before=(tmp_path/'budget.json').read_bytes()
    with pytest.raises(ValueError,match='audit required'):
        pool_terminal_overheads(tmp_path,*billing_reports(tmp_path,earlier_cost='3.00',later_cost='3.00'))
    assert (tmp_path/'budget.json').read_bytes()==before


def test_final_partition_reservation_survives_error_and_cannot_be_reused(tmp_path):
    reserve(tmp_path,dict(run_id='final-one',reserved_usd=3,
        phase='final_acceptance',final_seed_base=400000,status='error'))
    before=(tmp_path/'budget.json').read_bytes()
    with pytest.raises(ValueError,match='cannot alter'):
        record(tmp_path,'final-one',final_seed_base=400032)
    with pytest.raises(ValueError,match='cannot alter'):
        record(tmp_path,'final-one',phase='development')
    with pytest.raises(ValueError,match='partition already reserved'):
        reserve(tmp_path,dict(run_id='final-two',reserved_usd=3,
            phase='final_acceptance',final_seed_base=400000))
    assert (tmp_path/'budget.json').read_bytes()==before
    reserve(tmp_path,dict(run_id='final-three',reserved_usd=3,
        phase='final_acceptance',final_seed_base=400032))
