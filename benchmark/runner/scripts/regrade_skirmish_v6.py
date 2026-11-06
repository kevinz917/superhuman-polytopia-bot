"""Evaluate every frozen pilot artifact under the corrected, separately pinned panel.

No authoring inference, source repair, best-of selection, or old outcomes are reused.
The original feedback/reference protocol remains recorded with each authored build.
"""
import argparse,json,shutil,sys,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from polytopia_bot.skirmish import local,v6,local_v6
from polytopia_bot.skirmish.common import write_json,sha,canonical

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--origin',type=Path,default=ROOT/'artifacts/skirmish-v6/v6-proxy-20261031c')
parser.add_argument('--prefix',default='v61-proxy-20261031')
parser.add_argument('--manifest',type=Path,default=ROOT/'configs/skirmish/local-v6-proxy-pilot.json')
parser.add_argument('--results',type=Path,default=ROOT/'artifacts/skirmish-v61')
args=parser.parse_args()
origin=args.origin.resolve()
old=json.loads((origin/'allocation.json').read_text())
manifest=json.loads(args.manifest.read_text())
prefix=args.prefix;directory=args.results.resolve();root=directory/prefix
root.mkdir(parents=True,exist_ok=False)
cells=v6.cells(prefix,manifest)
by_key={(c['model_label'],c['track'],c['replicate']):c for c in old['cells']}
mapping={}
for c in cells:
 previous=by_key[c['model_label'],c['track'],c['replicate']]
 previous_root=origin/'runs'/previous['run_id']
 report=json.loads((previous_root/'report.json').read_text())
 if report['status'] not in ('completed','failed_submission'):raise ValueError('unfinished original authoring slot')
 protocol=json.loads((previous_root/'protocol.json').read_text())
 if protocol['model']!=c['model'] or protocol['track']!=c['track'] or protocol['source_files']!=old['source_files']:
  raise ValueError('original authoring provenance mismatch')
 for stage in report['stages']+report.get('budget_snapshots',[]):
  folder=previous_root/(f'budget-{stage["snapshot"]}' if 'snapshot' in stage else f'stage-{stage["stage"]}')
  if sha((folder/'main.rs').read_bytes())!=stage['source_sha256']:raise ValueError('frozen source mismatch')
  if stage.get('wasm_sha256') and sha((folder/'bot.wasm').read_bytes())!=stage['wasm_sha256']:raise ValueError('frozen Wasm mismatch')
 shutil.copytree(previous_root,root/'runs'/c['run_id'])
 mapping[c['run_id']]=dict(origin_run_id=previous['run_id'],origin_report_sha256=sha((previous_root/'report.json').read_bytes()),
     origin_protocol_sha256=sha((previous_root/'protocol.json').read_bytes()),
     original_feedback_panel='original six-reference pilot panel, unchanged during all authoring')
references=local.checked_references(manifest,ROOT/'artifacts/skirmish-references')
sources=local.sources()
allocation=dict(schema=v6.SCHEMA,manifest=manifest,cells=cells,source_files=sources,
    coder_image=local.image_id('polytopia-skirmish-coder:local-v6'),referee_image=local.image_id('polytopia-skirmish-referee:local-v6'),
    cpu_slots=6,concurrency=8,references=references)
write_json(root/'allocation.json',allocation)
origin_receipt=dict(origin=str(origin),origin_allocation_sha256=sha((origin/'allocation.json').read_bytes()),runs=mapping,
    evaluation_revision='reference-panel-v6.1',operator_source_sha256=sha(Path(__file__).read_bytes()),
    reason='corrected voluntary-disband priority in calibration candidates; defect recorded before hidden scores; all original builds retained',
    authoring_feedback='original panel preserved; corrected evaluation-only panel never returned to authors',
    submission_source_changed=False,new_model_calls=0,created_unix=time.time())
write_json(root/'authoring-origin.json',origin_receipt)
(root/'frozen-regrade.py').write_bytes(Path(__file__).read_bytes())
report=local_v6.cohort(prefix,manifest,directory,ROOT/'artifacts/skirmish-references',concurrency=8,cpu_slots=6,evaluate_only=True)
report['evaluation_revision']='reference-panel-v6.1';report['authoring_origin']=origin_receipt
write_json(root/'report.json',report)
(root/'report.md').write_text('# Corrected-panel four-model pilot\n\nThe twelve bots were authored once through CLI proxy using the original uniform feedback panel. All frozen stages/builds were transferred by digest; no new model call, repair or best-of selection occurred. Evaluation uses the corrected, separately frozen panel on fresh hidden layouts. These are preliminary one-build results; uncertainty intervals are not estimable.\n\n'+local_v6.render(report))
print(json.dumps({k:report[k] for k in ['status','pilot','games','ranking']},indent=2),flush=True)
