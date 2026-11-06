"""Bounded staging-only unchanged Qwen profile. No optimizer or adapter fit."""
import json
from pathlib import Path
import modal
ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('uv==0.9.25')
    .add_local_file(ROOT/'training/pyproject.toml', '/workspace/training/pyproject.toml', copy=True)
    .add_local_file(ROOT/'training/uv.lock', '/workspace/training/uv.lock', copy=True)
    .run_commands('UV_INDEX_URL=https://pypi.org/simple UV_DEFAULT_INDEX=https://pypi.org/simple uv sync --project /workspace/training --locked --no-dev')
    .env({'PYTHONPATH': '/workspace/src', 'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_dir(ROOT/'src', '/workspace/src', ignore=['**/__pycache__/**'])
    .add_local_file(ROOT/'configs/sft.json', '/workspace/config.json'))
app = modal.App('polytopia-sft0-model-profile', image=image)


@app.function(gpu='A100-80GB', cpu=4, memory=16384, timeout=900, max_containers=1, retries=0, scaledown_window=2)
def profile(positions):
    import subprocess
    import time
    start = time.monotonic()
    Path('/tmp/positions.json').write_text(json.dumps(positions))
    code = r'''
import hashlib, importlib.metadata, json, time
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from polytopia_bot.learning.inference import LocalGenerator
from polytopia_bot.learning.interface import digest
config=json.loads(Path('/workspace/config.json').read_text())
start=time.monotonic()
model_path=Path(snapshot_download(config['model_id'],revision=config['revision'],allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','LICENSE','README.md']))
files={str(p.relative_to(model_path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in model_path.rglob('*') if p.is_file()}
generator=LocalGenerator(config)
load_seconds=time.monotonic()-start
results=[]
for position in json.loads(Path('/tmp/positions.json').read_text()):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 try:
  out=generator(position['messages']);torch.cuda.synchronize()
  try:
   parsed=json.loads(out.text);valid=isinstance(parsed,dict) and set(parsed)=={'action_id'} and parsed['action_id'] in position['catalog']
  except (ValueError,TypeError):valid=False
  results.append(dict(position={k:v for k,v in position.items() if k!='messages'},input_hash=digest(position['messages']),response=out.text,prompt_tokens=len(out.prompt_token_ids),response_tokens=len(out.response_token_ids),prompt_token_ids=out.prompt_token_ids,response_token_ids=out.response_token_ids,first_attempt_valid=valid,teacher_agreement=valid and position['catalog'][parsed['action_id']]==position['teacher_command'],seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
 except Exception as exc:
  results.append(dict(error=f'{type(exc).__name__}: {exc}',input_hash=digest(position['messages'])))
report=dict(stage='unchanged-model-profile-only',config=config,model_files_sha256=files,device=torch.cuda.get_device_name(),load_seconds=load_seconds,chat_template_hash=digest(generator.tokenizer.chat_template),tokenizer_hash=digest(generator.tokenizer.backend_tokenizer.to_str()),packages={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate','tokenizers','huggingface-hub']},results=results)
Path('/tmp/result.json').write_text(json.dumps(report,indent=2))
'''
    subprocess.run(['/workspace/training/.venv/bin/python', '-c', code], check=True, timeout=840)
    result = json.loads(Path('/tmp/result.json').read_text())
    result['function_elapsed_seconds'] = time.monotonic()-start
    return result


@app.local_entrypoint()
def main(input_path: str='runs/sft0/model-profile-input/positions.json', output: str='runs/sft0/model-profile-v0.json'):
    from modal.config import config
    if config.get('environment') != 'staging':
        raise RuntimeError('Pass --env staging')
    out = ROOT/output
    if out.exists():
        raise FileExistsError(out)
    result = profile.remote(json.loads((ROOT/input_path).read_text()))
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('model_files_sha256','results')},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('prompt_token_ids','response_token_ids','position')} for r in result['results']],indent=2))
