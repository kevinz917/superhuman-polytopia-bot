"""Run the single Harbor benchmark locally, with an explicit integration-demo profile."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--demo', action='store_true', help='Same 40-minute coding protocol; 36 final games instead of 360. No ranking claim.')
    p.add_argument('--model', action='append', help='Explicit CLI-proxy model; repeat for multiple models.')
    p.add_argument('--attempts', type=int, default=1)
    p.add_argument('--concurrency', type=int, default=2)
    args = p.parse_args()
    models = args.model or ['openai/gpt-6-astra','anthropic/claude-opus-5.5']
    from polytopia_bot.skirmish.inference import endpoint
    for model in models:
        if endpoint(dict(model=model,provider='cliproxyapi')).backend != 'cliproxyapi':
            raise ValueError('CLI proxy is required')
    package = args.package.resolve()
    build = json.loads((package/'build.json').read_text())
    config = dict(job_name=args.name, jobs_dir=str(ROOT/'artifacts/harbor-runs'),
        n_attempts=args.attempts, n_concurrent_trials=args.concurrency, skip_credential_checks=True,
        environment=dict(type='docker', delete=True, cpu_enforcement_policy='limit', memory_enforcement_policy='limit'),
        tasks=[dict(path=build['task'])], agents=[dict(import_path='polytopia_bot.skirmish.harbor_agent:SkirmishAgent',
            model_name=model, kwargs=dict(build_receipt=str(package/'build.json'))) for model in models],
        verifier=dict(env={'SKIRMISH_EVALUATION_PROFILE':'demo' if args.demo else 'standard'}))
    from harbor.models.job.config import JobConfig
    JobConfig.model_validate(config)
    config_path = package/f'{args.name}.json'
    if config_path.exists():
        raise ValueError('job name already allocated; never silently re-author')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    from scripts.check_skirmish_proxy import mini_check
    probes = []
    for model in models:
        probes.append(dict(mini_check(model), requested_model=model))
    (package/f'{args.name}.probes.json').write_text(json.dumps(probes,indent=2)+'\n')
    env = dict(os.environ, DOCKER_CONTEXT=os.environ.get('SKIRMISH_DOCKER_CONTEXT','colima-polytopia-skirmish'),
               PYTHONPATH=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    executable = Path(sys.executable).with_name('harbor')
    subprocess.run([str(executable),'run','--config',str(config_path)], env=env, cwd=ROOT, check=True)
    from scripts.summarize_skirmish_harbor import summarize
    result = summarize(ROOT/'artifacts/harbor-runs'/args.name)
    print(json.dumps(dict(status=result['status'],outcomes=result['total_final_outcomes'],
        models=[t['model'] for t in result['trials']]),indent=2))


if __name__ == '__main__':
    main()
