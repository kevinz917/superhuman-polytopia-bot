"""Verify the completed proportional scaling run against its frozen inputs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.training import file_hash
from polytopia_bot.learning.scale_training import BOUNDARIES,CHECKPOINT_STEPS


def main():
 root=ROOT/'runs/movement-scale40k-v1'
 def read(name):return json.loads((root/name).read_text())
 p=read('protocol.json');run=read('training-run.json');state=read('cloud-status.json');summary=read('summary.json');replay=read('replay-integrity.json')
 parent=json.loads((ROOT/'runs/movement-epoch2-v1/training-run.json').read_text())
 if state['status']!='completed' or run['status']!='completed' or not summary['complete']:raise ValueError('incomplete run')
 if run['config']!=parent['config'] or run['data_manifest']!=read('prepared/manifest.json'):raise ValueError('configuration/data changed')
 if run['scale']!=p['training']['scale'] or run['scale']['adapter_files']!=parent['adapter_files']:raise ValueError('warm start changed')
 if run['adapter_before']!=parent['adapter_after'] or run['adapter_after']==run['adapter_before']:raise ValueError('adapter integrity failed')
 if not run['loaded_adapter_exact'] or run['frozen_before']!=run['frozen_after'] or run['frozen_before']!=parent['frozen_after']:raise ValueError('frozen model changed')
 if run['epoch_order_sha256']!=replay['epoch_order_sha256'] or not replay['passed']:raise ValueError('example order differs')
 if run['global_step']!=628 or len(run['logs'])!=628 or run['logs'][-1]['examples_seen']!=40000:raise ValueError('training coverage differs')
 expected=zip(CHECKPOINT_STEPS,BOUNDARIES,strict=True)
 if [(c['step'],c['examples_seen']) for c in run['checkpoints']]!=list(expected):raise ValueError('checkpoint boundaries differ')
 if run['checkpoints'][-1]['adapter_files']!=run['adapter_files']:raise ValueError('final checkpoint differs')
 if set(state['evaluations'])!={'0','157','314','471','628'}:raise ValueError('evaluation checkpoints differ')
 if any(v['exit_code'] or not v['summary']['complete'] or v['summary']['decisions']!=1475 for v in state['evaluations'].values()):raise ValueError('incomplete evaluation')
 if not run['corpus_stress']['passed'] or run['corpus_stress']['optimizer_updates']!=0:raise ValueError('long-input preflight missing')
 for name,sha in p['uploads'].items():
  if file_hash(root/name)!=sha:raise ValueError('uploaded input changed: '+name)
 for name,sha in parent['adapter_files'].items():
  if file_hash(ROOT/'runs/movement-epoch2-v1/adapter-final'/name)!=sha:raise ValueError('preserved main adapter changed')
 out=dict(passed=True,main_preserved=True,exact_main_warm_start=True,frozen_base_unchanged=True,all_40000_examples_once=True,checkpoint_steps=CHECKPOINT_STEPS,checkpoint_examples=BOUNDARIES,completed_evaluations=5,decisions_per_evaluation=1475,
          protocol_sha256=file_hash(root/'protocol.json'),training_receipt_sha256=file_hash(root/'training-run.json'),summary_sha256=file_hash(root/'summary.json'),final_adapter_files=run['adapter_files'])
 (root/'training-integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
