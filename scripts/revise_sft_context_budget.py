#!/usr/bin/env python3
"""Create a distinct main-run context version from an exhaustive length audit."""
import argparse,copy,gzip,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from polytopia_bot.learning.freeze import verify_prompt,inference_config
from polytopia_bot.learning.interface import digest


def revise(output,audit_path,max_length,config_path,prompt_path,stress_path=None):
    old=json.loads((ROOT/config_path).read_text());prompt=json.loads((ROOT/prompt_path).read_text())
    strategy=verify_prompt(prompt,old);audit=json.loads((ROOT/audit_path).read_text())
    if audit['schema']!='sft-sample-length-audit-v1' or audit['count']!=50000:
        raise ValueError('all 50000 main examples must be audited first')
    if audit['config_hash']!=digest(old) or audit['prompt_manifest_hash']!=prompt['manifest_hash']:
        raise ValueError('audit used a different original interface')
    source=ROOT/audit['sample']/'records.jsonl'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=audit['records_sha256']:
        raise ValueError('audited main data changed')
    required=audit['minimum_context_with_repair_headroom'];stress_evidence=None
    if stress_path is not None:
        from polytopia_bot.learning.data import observation_from_dict
        from polytopia_bot.learning.prompt_search import candidate_messages
        from polytopia_bot.learning.training import load_tokenizer
        stress=json.loads((ROOT/stress_path).read_text())
        if stress['schema']!='large-training-input-stress-v1':raise ValueError('unknown stress evidence')
        largest=max(stress['results'],key=lambda r:r['tokens']);raw=ROOT/largest['file']
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=largest['file_sha256']:raise ValueError('stress source changed')
        record=None
        for index,line in enumerate(gzip.open(raw,'rt')):
            if index==largest['index']:record=json.loads(line)['record'];break
        if record is None:raise ValueError('stress public record missing')
        messages=candidate_messages(observation_from_dict(record['observation']),strategy)
        tokenizer=load_tokenizer(old)
        if digest(tokenizer.backend_tokenizer.to_str())!=prompt['tokenizer_hash']:raise ValueError('stress tokenizer changed')
        tokens=len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
        if tokens!=largest['tokens'] or digest(messages)!=largest['prompt_digest']:raise ValueError('stress prompt does not reproduce')
        stress_evidence=dict(**largest,observation=record['observation'],reproduced=True)
        required=max(required,tokens+old['max_new_tokens']+256)
    if required<=old['max_length']:raise ValueError('no measured context/repair-headroom need for expansion')
    if max_length<=old['max_length'] or max_length<required:
        raise ValueError('new capacity must cover every measured input and repair headroom')
    from huggingface_hub import hf_hub_download
    model_config=Path(hf_hub_download(old['model_id'],'config.json',revision=old['revision']))
    if max_length>json.loads(model_config.read_text())['max_position_embeddings']:
        raise ValueError('budget exceeds pinned model context capacity')
    root=ROOT/output;root.mkdir(parents=True,exist_ok=False)
    config=copy.deepcopy(old);config['max_length']=max_length
    for name,value in [('prior-config.json',old),('config.json',config)]:
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    (root/'base-model-config.json').write_bytes(model_config.read_bytes())
    if stress_evidence is not None:
        (root/'stress-evidence.json').write_text(json.dumps(stress_evidence,indent=2)+'\n')
    revised=copy.deepcopy(prompt);revised.pop('manifest_hash')
    revised['inference_config_hash']=digest(inference_config(config))
    revised['context_budget_revision']=dict(previous_manifest_hash=prompt['manifest_hash'],
        previous_max_length=old['max_length'],max_length=max_length,
        reason=('Sample inputs or their reserved repair headroom exceed the prior budget.' if audit['minimum_context_with_repair_headroom']>old['max_length']
                else 'The sampled inputs fit the prior budget, but an independently reproduced reachable raw-training state exceeds it. Keep the complete observation and legal catalog.'),
        sample_overflow_count=audit['overflow_count'],sample_max_input_tokens=audit['tokens']['max'],
        stress_input_tokens=None if stress_evidence is None else stress_evidence['tokens'],
        common_base_and_sft_budget=True,requires_fresh_main_baseline=True,
        unchanged=['Instruction text','Public information','Renderer','Action meanings/response IDs','Tokenizer/chat template','Output-token cap','Repair policy'],
        historical_results='Keep all original 16K training/evaluation receipts unchanged. This is a new main-run version, not a relabeling of old results.')
    paths=[ROOT/prompt_path,ROOT/audit_path,root/'prior-config.json',root/'config.json',root/'base-model-config.json']
    if stress_evidence is not None:paths.extend([ROOT/stress_path,root/'stress-evidence.json'])
    revised['evidence'] += [dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    revised['manifest_hash']=digest(revised);verify_prompt(revised,config)
    (root/'frozen-prompt.json').write_text(json.dumps(revised,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),manifest_hash=revised['manifest_hash'],previous_max_length=old['max_length'],max_length=max_length,
                         requires_main_rerender=True,requires_fresh_main_baseline=True),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--length-audit',type=Path,required=True);p.add_argument('--max-length',type=int,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/sft.json'));p.add_argument('--prompt',type=Path,default=Path('runs/sft0/frozen-prompt-v1.json'))
    p.add_argument('--stress-audit',type=Path)
    a=p.parse_args();revise(a.output,a.length_audit,a.max_length,a.config,a.prompt,a.stress_audit)
