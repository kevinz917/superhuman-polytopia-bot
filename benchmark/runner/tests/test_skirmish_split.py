import asyncio
import copy
import json
from pathlib import Path

import pytest

from polytopia_bot.skirmish.split import route_models, job_config


def test_routes_use_exact_catalog_and_never_mutate_roster():
    models = [dict(label='a',model='openai/gpt-6-astra',parameters={'provider':{'only':['openai']}}),
              dict(label='b',model='qwen/exact-model',parameters={'provider':{'only':['alibaba']}})]
    original = copy.deepcopy(models)
    rows = route_models(models, {'gpt-6-astra','exact-model-newer'})
    assert [(r['execution_backend'],r['provider']) for r in rows] == [
        ('docker','cliproxyapi'),('daytona','openrouter')]
    assert rows[1]['parameters']['provider'] == dict(only=['alibaba'],allow_fallbacks=False)
    assert models == original
    with pytest.raises(ValueError,match='duplicate'):
        route_models([models[0],models[0]],set())


def test_native_jobs_preserve_one_protocol_and_explicit_routes(tmp_path):
    from harbor.models.job.config import JobConfig
    (tmp_path/'build.json').write_text(json.dumps({'task':str(tmp_path/'task'),'coder_snapshot':{'name':'coder','id':'1'},'verifier_snapshot':{'name':'verifier','id':'2'}}))
    for backend,provider in [('docker','cliproxyapi'),('daytona','openrouter')]:
        model = dict(model='example/model',execution_backend=backend,provider=provider,parameters={})
        cfg = job_config('job',tmp_path,[model],5,32,tmp_path/'jobs')
        JobConfig.model_validate(cfg)
        assert cfg['n_attempts']==5 and cfg['n_concurrent_trials']==5
        assert cfg['verifier']['env']['SKIRMISH_EVALUATION_PROFILE']=='standard'
        assert cfg['agents'][0]['kwargs']['provider']==provider
        assert cfg['agents'][0]['kwargs']['execution_backend']==backend
    with pytest.raises(ValueError,match='one execution backend'):
        job_config('job',tmp_path,[dict(execution_backend='docker'),dict(execution_backend='daytona')],1,2,tmp_path)


def test_cpu_admission_is_bounded_and_cancellation_releases(tmp_path):
    from polytopia_bot.skirmish.harbor_environments import slot
    async def exercise():
        active = 0
        maximum = 0
        async def worker():
            nonlocal active,maximum
            async with slot(tmp_path,2):
                active += 1
                maximum = max(maximum,active)
                try:
                    await asyncio.sleep(.05)
                finally:
                    active -= 1
        await asyncio.gather(*(worker() for _ in range(6)))
        assert maximum == 2 and active == 0
        entered = asyncio.Event()
        async def cancelled():
            async with slot(tmp_path,1):
                entered.set()
                await asyncio.sleep(60)
        task = asyncio.create_task(cancelled())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with asyncio.timeout(1):
            async with slot(tmp_path,1):
                pass
    asyncio.run(exercise())


def test_signed_native_anthropic_tool_continuation_is_lossless():
    from polytopia_bot.skirmish.anthropic_wire import request_payload, response_envelope
    signed = [dict(type='thinking',thinking='private planning',signature='signature+/=='),
              dict(type='redacted_thinking',data='opaque+/=='),
              dict(type='tool_use',id='bash-1',name='bash',input={'command':'printf ok'})]
    raw = dict(id='test',model='claude-test',content=signed,stop_reason='tool_use',
               usage=dict(input_tokens=2,cache_creation_input_tokens=3,cache_read_input_tokens=5,output_tokens=7))
    envelope = response_envelope(raw)
    message = envelope['choices'][0]['message']
    assert envelope['usage']['prompt_tokens']==10
    tool = dict(function=dict(name='bash',description='shell',parameters={'type':'object'}))
    payload = request_payload([dict(role='system',content='task'),dict(role='user',content='start'),message,
        dict(role='tool',tool_call_id='bash-1',content='ok')],
        dict(max_tokens=32768,reasoning_effort='high'),'claude-test',tool)
    assert payload['messages'][1]['content']==signed
    assert payload['messages'][2]['content']==[dict(type='tool_result',tool_use_id='bash-1',content='ok')]
    assert payload['thinking']==dict(type='adaptive') and payload['output_config']==dict(effort='high')
    assert raw['content']==signed


def test_export_validation_hashes_original_bytes_for_compressed_traces(tmp_path):
    import gzip
    import hashlib
    import importlib.util
    path = Path(__file__).resolve().parents[1]/'scripts/summarize_skirmish_harbor.py'
    spec = importlib.util.spec_from_file_location('skirmish_summary',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = b'{"observation":{},"action_id":1}\n'*10000
    plain = tmp_path/'game.jsonl'
    plain.write_bytes(data)
    compressed = tmp_path/'game.jsonl.gz'
    compressed.write_bytes(gzip.compress(data))
    assert module.trace_digest(plain)==module.trace_digest(compressed)==hashlib.sha256(data).hexdigest()


def test_daytona_ack_backpressure_preserves_command_deadlines_and_never_retries(monkeypatch):
    from types import SimpleNamespace
    import polytopia_bot.skirmish.harbor_environments as module
    async def exercise():
        module._startup_gate = module._submission_gate = None
        active = maximum = calls = 0
        command = SimpleNamespace(command='timeout 8 python long_command.py',run_async=True)
        async def original(session_id,req,timeout=None):
            nonlocal active,maximum,calls
            assert req is command and req.command.startswith('timeout 8 ')
            assert timeout == 120
            calls += 1
            active += 1
            maximum = max(maximum,active)
            try:
                await asyncio.sleep(.01)
                if session_id=='failed':
                    raise TimeoutError('ambiguous submission')
                return session_id
            finally:
                active -= 1
        process = SimpleNamespace(execute_session_command=original)
        env = module.SkirmishDaytonaEnvironment.__new__(module.SkirmishDaytonaEnvironment)
        async def create(self,*args,**kwargs):
            self._sandbox=SimpleNamespace(process=process)
        monkeypatch.setattr(module.DaytonaEnvironment,'_create_sandbox',create)
        await env._create_sandbox()
        await asyncio.gather(*(process.execute_session_command(str(i),command,timeout=8) for i in range(35)))
        assert maximum<=8 and calls==35
        with pytest.raises(TimeoutError):
            await process.execute_session_command('failed',command,timeout=8)
        assert calls==36
        module._startup_gate = module._submission_gate = None
    asyncio.run(exercise())


def test_long_controllers_cannot_starve_sdk_file_transport_threads():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from polytopia_bot.skirmish.harbor_agent import CONTROLLERS
    async def exercise():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        release = threading.Event()
        controllers = [loop.run_in_executor(CONTROLLERS,release.wait) for _ in range(3)]
        try:
            async with asyncio.timeout(1):
                assert await asyncio.to_thread(lambda:'file transport remains available')
        finally:
            release.set()
            await asyncio.gather(*controllers)
    asyncio.run(exercise())
