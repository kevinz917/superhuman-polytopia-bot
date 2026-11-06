#!/usr/bin/env python3
"""Read local SFT receipts and logs without launching or changing any job."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'runs/sft0'


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def status():
    result={}
    for name,total in [('fit-result-v1',128),('smoke-result-v1',79),('b200-profile-v1',128),('fit-context-v2-result',128),('repaired-smoke-result-v2',157)]:
        receipt=read(RUNS/name/'execution.json')
        log=RUNS/(name+'.log')
        text=log.read_text() if log.exists() else ''
        progress=re.findall(r'\d+/'+str(total)+r' \[[^\]]+\]',text)
        row=dict(progress=progress[-1] if progress else None,
                 preemptions=text.count('Container terminated due to preemption'),
                 local_failure=any(marker in text for marker in ('uncaught exception raised locally','Stopping app - local client disconnected','RemoteError')))
        if receipt:
            run=receipt.get('run') or {}
            row.update(status=run.get('status'),exit_code=receipt.get('exit_code'),
                       reload_exit_code=receipt.get('reload_exit_code'),elapsed_seconds=receipt.get('elapsed_seconds'),
                       hardware=run.get('hardware'),accuracy_after=run.get('fit_after',{}).get('accuracy'))
        else:row['status']='awaiting_execution_receipt' if log.exists() else 'not_started'
        result[name]=row
    games=read(RUNS/'raw-main-v1/games.json')
    result['raw-main-v1']=dict(completed_episodes=None if games is None else len(games),allocated_episodes=940,
        complete=bool((read(RUNS/'raw-main-v1/report.json') or {}).get('complete')))
    evaluations=[('smoke-positions-v1','report.json'),('base-positions-v2','report.json'),
        ('smoke-games-v1','combined/games/report.json'),('base-games-v2','combined/games/report.json'),
        ('relabel-games-v1','combined/games/report.json'),('repaired-positions-v2','report.json'),
        ('repaired-games-v2','combined/games/report.json'),
        ('repaired-games-v2-recovery1','combined/games/report.json')]
    for name,report_path in evaluations:
        report=read(RUNS/name/report_path)
        execution=read(RUNS/name/'execution.json') or []
        log=RUNS/(name+'.log')
        text=log.read_text() if log.exists() else ''
        apps=re.findall(r'https://modal.com/apps/[^\s]+',text)
        result[name]=dict(app_url=apps[0] if apps else None,
            preemptions=text.count('Container terminated due to preemption'),
            local_failure=any(marker in text for marker in ('uncaught exception raised locally','Stopping app - local client disconnected','RemoteError')),
            app_completed='App completed.' in text,
            returned_shards=len(execution),successful_shards=sum(r.get('exit_code')==0 for r in execution),
            complete=bool(report and report.get('complete')),
            report=None if report is None else {k:v for k,v in report.items() if k not in ('results','games')})
        if report_path=='combined/games/report.json':
            from collections import Counter
            logged={}
            for line in text.splitlines():
                try:row=json.loads(line[line.index('{'):])
                except (ValueError,json.JSONDecodeError):continue
                if isinstance(row,dict) and 'id' in row and 'status' in row:logged[row['id']]=row
            result[name]['logged_outcomes']=dict(games=len(logged),statuses=dict(Counter(r['status'] for r in logged.values())),
                wins=sum(r['status']=='native' and r['reward']>0 for r in logged.values()),
                first_attempt_valid=sum(r['first_attempt_valid'] for r in logged.values()),
                model_decisions=sum(r['model_decisions'] for r in logged.values()),
                note='Partial log counts are not the completed evaluation result.')
    driver=read(RUNS/'repair-driver-status-v1.json')
    if driver:
        result['repair_driver']={k:v for k,v in driver.items() if k!='stages'}
        result['repair_driver']['stages']=[{k:v for k,v in row.items() if k!='args'} for row in driver['stages']]
    recovery_log=RUNS/'relabel-shard3-recovery-v1.log'
    if recovery_log.exists():
        text=recovery_log.read_text()
        apps=re.findall(r'https://modal.com/apps/[^\s]+',text)
        rows={}
        for line in text.splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(row,dict) and 'id' in row and 'status' in row:rows[row['id']]=row
        result['relabel_shard_recovery']=dict(app_url=apps[0] if apps else None,
            completed_games=len(rows),allocated_games=12,
            model_decisions=sum(r['model_decisions'] for r in rows.values()),
            preemptions=text.count('Container terminated due to preemption'),
            local_failure=any(marker in text for marker in ('uncaught exception raised locally','Stopping app - local client disconnected','RemoteError')),
            app_completed='App completed.' in text)
    return result


if __name__=='__main__':print(json.dumps(status(),indent=2))
