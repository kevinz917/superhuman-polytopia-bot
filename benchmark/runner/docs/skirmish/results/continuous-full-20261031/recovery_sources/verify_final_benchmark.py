"""Verify the portable benchmark archive; run from the repository root."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,statistics,math,datetime,shutil
root=Path.cwd();out=root/'docs/skirmish/results/continuous-full-20261031';r=json.loads((out/'results.json').read_text());assert r['status']=='completed_validated';assert r['authored_submissions']==len(r['trials'])==60;assert r['total_final_outcomes']==21600
assert r['models']==12 and r['attempts_per_model']==5
sources=set();groups=defaultdict(list);kinds=Counter()
for t in r['trials']:
 source=out/t['source_file'];assert hashlib.sha256(source.read_bytes()).hexdigest()==t['source_sha256'];assert t['original_trial'] not in sources;sources.add(t['original_trial']);assert t['games']==360 and t['coding_budget_seconds']==2400 and 0<=t['timing']['coding_seconds']<=2400
 receipt=json.loads((out/t['saved_validation_receipt']).read_text());assert receipt['passed'];assert receipt['schedule_sha256']==r['schedule_sha256'];assert receipt['summary']['score']==t['score'];assert receipt['summary']['model']==t['model'];assert math.isclose(t['score'],100*t['rewards']['reward'],abs_tol=1e-12);assert sum(t['champion'].values())==60
 compiled=json.loads((out/t['compilation_receipt_file']).read_text());assert compiled['source_sha256']==t['source_sha256'];assert compiled['wasm_sha256']==t['wasm_sha256'];assert compiled['returncode']==t['compile_returncode']==0;assert len(t['wasm_sha256'])==64
 groups[t['model']].append(t);kinds[t['validation_kind']]+=1
assert len(groups)==12 and all(len(v)==5 for v in groups.values());assert dict(kinds)==r['validation_kinds'];assert sum(t['games'] for t in r['trials'])==21600
for row in r['leaderboard']:
 trials=groups[row['model']];assert row['attempts']==5 and row['games']==1800
 assert math.isclose(row['mean_score'],statistics.mean(t['score'] for t in trials),abs_tol=1e-12);assert math.isclose(row['score_sd'],statistics.stdev(t['score'] for t in trials),abs_tol=1e-12)
 assert row['champion_wdl']=={k:sum(t['champion'][k] for t in trials) for k in ('wins','draws','losses')};assert sum(row['champion_wdl'].values())==300
assert r['leaderboard']==sorted(r['leaderboard'],key=lambda x:(-x['mean_score'],x['model']))
builds=json.loads((out/'build-provenance.json').read_text());assert builds['local']['source_files']==builds['daytona']['source_files'];assert builds['local']['manifest_sha256']==builds['daytona']['manifest_sha256'];assert builds['local']['maps_sha256']==builds['daytona']['maps_sha256']
assert json.loads((out/'compact-qualification-passed.json').read_text())['passed'];assert json.loads((out/'recovery-qualification-passed.json').read_text())['passed'];assert json.loads((out/'cloud-cleanup-inventory.json').read_text())['remaining_campaign_sandboxes']==[]
proof={'passed':True,'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':12,'independent_authors':60,'games':21600,'compilation_receipts':60,'source_hashes_verified':60,'shared_schedule_sha256':r['schedule_sha256'],'validation_kinds':dict(kinds),'checks':['Cohort completeness and unique authored slots','Portable Rust source SHA256 and native compilation receipts','Original coding budget and shared evaluation schedule','Per-bot rewards and model mean/sample standard deviation','Same local/cloud core source, reference-manifest and map pins','Grading recovery and compact replay qualification receipts','Cloud sandbox cleanup receipt']}
(out/'archive-validation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2));destination=out/'recovery_sources'/Path(__file__).name
if Path(__file__).resolve()!=destination.resolve():shutil.copyfile(Path(__file__),destination)
