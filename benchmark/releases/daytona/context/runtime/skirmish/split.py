"""Plan explicit routes once; a failed proxy never creates a paid fallback."""
import copy
from .inference import proxy_model


def route_models(models, proxy_catalog, *, proxy_exclusions=()):
    result = []
    seen = set()
    for original in models:
        row = copy.deepcopy(original)
        model = row['model']
        if model in seen:
            raise ValueError('duplicate model in campaign')
        seen.add(model)
        # Match exact IDs; never substitute a newer/different model by similarity.
        if proxy_model(model) in proxy_catalog and model not in proxy_exclusions:
            row.update(provider='cliproxyapi', execution_backend='docker')
            row.setdefault('parameters', {}).pop('provider', None)
        else:
            row.update(provider='openrouter', execution_backend='daytona')
            provider = row.setdefault('parameters', {}).setdefault('provider', {})
            provider['allow_fallbacks'] = False
            if model in proxy_exclusions:
                row['routing_reason'] = 'proxy tool-continuation preflight failed; OpenRouter selected before authoring'
        result.append(row)
    return result


def job_config(name, package, models, attempts, concurrency, jobs_dir, *, demo=False):
    import json
    from pathlib import Path
    package = Path(package).resolve()
    receipt = json.loads((package/'build.json').read_text())
    backends = {r['execution_backend'] for r in models}
    if len(backends) != 1:
        raise ValueError('each Harbor job has one execution backend')
    backend = backends.pop()
    if backend == 'docker':
        env = dict(import_path='polytopia_bot.skirmish.harbor_environments:SkirmishDockerEnvironment',
            delete=True, cpu_enforcement_policy='limit',memory_enforcement_policy='limit',
            kwargs=dict(cpu_slots=6,grader_workers=4))
    else:
        env = dict(import_path='polytopia_bot.skirmish.harbor_environments:SkirmishDaytonaEnvironment',
                   delete=True,kwargs=dict(snapshot_ids={receipt[k]['name']:receipt[k]['id']
                    for k in ['coder_snapshot','verifier_snapshot']}))
    return dict(job_name=name,jobs_dir=str(jobs_dir),n_attempts=attempts,
        n_concurrent_trials=min(concurrency,len(models)*attempts),skip_credential_checks=True,
        environment=env,tasks=[dict(path=receipt['task'])],
        agents=[dict(import_path='polytopia_bot.skirmish.harbor_agent:SkirmishAgent',
            model_name=r['model'], kwargs=dict(provider=r['provider'],parameters=r['parameters'],
            execution_backend=backend,build_receipt=str(package/'build.json'),local_cpu_slots=6)) for r in models],
        verifier=dict(env={'SKIRMISH_EVALUATION_PROFILE':'demo' if demo else 'standard'}))
