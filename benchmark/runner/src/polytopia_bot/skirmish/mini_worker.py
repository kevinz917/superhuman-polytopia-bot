"""Trusted mini-SWE-agent process; the model's bash runs only in an offline Modal sandbox.

The parent kills this process at the stage deadline, including model HTTP retries.
No key, controller filesystem, or environment is passed to the coding sandbox.
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import time

MINI_VERSION = '2.4.6'
FORMAT_ERROR_TEMPLATE = '''{{ error }}
{% if finish_reason | default('') == 'length' %}
Your response reached the per-call output-token limit before a complete bash action was available.
Reasoning counts toward that limit. Use a compact bash tool call and split large file writes across calls.
{% else %}
Use the bash tool for actions. To finish, call bash with: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
{% endif %}'''


def openrouter_model(*, timing=None, route=None, deadline_path=None, **kwargs):
    """Keep upstream requests/tools; billing metadata can never stop a trial."""
    from minisweagent.models.openrouter_model import OpenRouterModel
    from minisweagent.exceptions import TimeExceeded

    class BenchmarkModel(OpenRouterModel):
        abort_exceptions=[*OpenRouterModel.abort_exceptions,TimeExceeded]
        def _query(self, messages, **kw):
            if route is None:
                return super()._query(messages, **kw)
            from minisweagent.models.utils.actions_toolcall import BASH_TOOL
            from .providers import request
            payload = dict(self.config.model_kwargs | kw, model=route.model,
                           messages=messages, tools=[BASH_TOOL])
            if route.backend == 'openrouter':
                payload.setdefault('usage', {'include': True})
            try:
                timeout=60.
                if deadline_path:
                    deadline=json.loads(Path(deadline_path).read_text())['deadline_unix']
                    timeout=max(.01,deadline-time.time())
                if route.backend == 'cliproxyapi' and route.model.startswith('claude-'):
                    from .anthropic_wire import request_payload, response_envelope
                    native = request_payload(messages,self.config.model_kwargs|kw,route.model,BASH_TOOL)
                    return response_envelope(request(route.base_url+'/messages',route.credential(),
                        native,timeout,anthropic=True))
                return request(route.base_url + '/chat/completions',route.credential(),payload,timeout)
            except RuntimeError as error:
                from minisweagent.models.openrouter_model import OpenRouterAuthenticationError, OpenRouterRateLimitError
                if 'model_api_http_401' in str(error):
                    raise OpenRouterAuthenticationError('Selected inference endpoint rejected its credential') from None
                if 'model_api_http_429' in str(error):
                    raise OpenRouterRateLimitError('Selected inference endpoint rate limited the request') from None
                raise

        def serialize(self):
            value = super().serialize()
            if route is not None:
                value['info']['inference'] = route.receipt()
                value['info']['inference']['api_protocol'] = (
                    'anthropic' if route.backend=='cliproxyapi' and route.model.startswith('claude-') else 'chat')
            return value

        def query(self, *args, **kw):
            import uuid
            call_id=uuid.uuid4().hex
            if timing:timing.event('api_start',id=call_id)
            outcome='completed'
            try:
                return super().query(*args,**kw)
            except BaseException as error:
                outcome='interrupted' if isinstance(error,TimeExceeded) else 'error'
                raise
            finally:
                if timing:timing.event('api_end',id=call_id,outcome=outcome)

        def _calculate_cost(self, response):
            # This is optional bookkeeping, not validation or a spend limit.
            # Preserve the original usage in the trajectory; unknown cost adds 0
            # to the partial reported total and must not reject the model response.
            try:
                cost = float((response.get('usage') or {}).get('cost') or 0)
                return {'cost': cost if math.isfinite(cost) else 0.0}
            except (TypeError, ValueError, AttributeError, OverflowError):
                return {'cost': 0.0}

    kwargs['cost_tracking']='ignore_errors'
    model = BenchmarkModel(**kwargs)
    if route is not None:
        # Upstream constructor reads OPENROUTER_API_KEY even for compatible APIs.
        # Our transport obtains only the selected credential at dispatch time.
        model._api_key = ''
    return model


def coding_model(config, *, timing=None):
    """Mini-SWE parser/loop, with the same host-side routing as native adapters."""
    from .inference import endpoint, parameters
    route = endpoint(config)
    return openrouter_model(model_name=route.model, model_kwargs=parameters(config, route),
                            format_error_template=FORMAT_ERROR_TEMPLATE,timing=timing,route=route,
                            deadline_path=config.get('start_path') if config.get('v6_track') or config.get('continuous') else None)


def continued_history(messages):
    """Close interrupted/submit tool calls before appending the next stage's user message.

    Upstream Submitted exits before execute_actions writes its tool observations. The
    original trajectory stays untouched; these boundary receipts belong to the new stage.
    """
    history = [m for m in messages if m.get('role') != 'exit']
    pending = {}
    for message in history:
        if message.get('role') == 'assistant':
            for call in message.get('tool_calls') or []:
                pending[call['id']] = call
        elif message.get('role') == 'tool':
            pending.pop(message.get('tool_call_id'),None)
    return [*history, *[dict(role='tool',tool_call_id=call_id,content=
        'The previous coding stage ended before this tool result was recorded. '
        'The controller froze the current source. Inspect the workspace and the supplied '
        'stage feedback; no execution outcome is asserted by this boundary receipt.') for call_id in pending]]

# Runs as coder, and kills the whole shell process group at the command deadline.
SHELL_RUNNER = '''import json,os,signal,subprocess,sys,time,tempfile
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text())
command,deadline=request[:2]
seconds=min(request[2] if len(request)>2 else 60.,deadline-time.time())
if seconds<=0:
 print(json.dumps(dict(output='',returncode=-1,exception_info='command deadline before launch')));sys.exit(0)
script=tempfile.NamedTemporaryFile(mode='w',suffix='.sh',delete=False);script.write(command);script.close()
p=subprocess.Popen(['/bin/bash',script.name],cwd='/bot',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
try:
 out,_=p.communicate(timeout=seconds); code=p.returncode; error=''
except subprocess.TimeoutExpired:
 os.killpg(p.pid,signal.SIGKILL); out,_=p.communicate(); code=-1; error='command deadline'
os.unlink(script.name)
print(json.dumps(dict(output=out.decode('utf-8','replace')[-1000000:],returncode=code,exception_info=error)))
'''


class ModalBashEnvironment:
    def __init__(self, sandbox, deadline, timing=None):
        self.sandbox, self.deadline = sandbox, deadline
        self.timing = timing
        self.command_seconds = 60.

    def execute(self, action, **kwargs):
        from minisweagent.exceptions import TimeExceeded, Submitted
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise TimeExceeded(dict(role='exit', content='TimeExceeded', extra=dict(exit_status='TimeExceeded')))
        seconds = max(.05, min(self.command_seconds, remaining))
        import uuid
        call_id=uuid.uuid4().hex
        if self.timing:self.timing.event('tool_start',id=call_id)
        # Transfer command bytes through the filesystem, never argv: Modal caps
        # total exec arguments at 64 KiB, below legitimate Rust file writes.
        command_path=f'/tmp/skirmish-command-{call_id}.json'
        self.sandbox.filesystem.write_text(json.dumps([action['command'],self.deadline,self.command_seconds]),command_path)
        if time.time()>=self.deadline:
            raise TimeExceeded(dict(role='exit',content='TimeExceeded',extra=dict(exit_status='TimeExceeded')))
        proc = self.sandbox.exec('/usr/sbin/runuser', '-u', 'coder', '--', 'env', 'CARGO_HOME=/home/coder/.cargo',
                                 'python', '-c', SHELL_RUNNER,
                                 command_path, timeout=math.ceil(min(self.command_seconds,max(.05,self.deadline-time.time())))+5)
        try:
            raw = proc.stdout.read()
            proc.wait()
            output = json.loads(raw)
        except Exception as error:
            output = dict(output='', returncode=-1, exception_info=f'{type(error).__name__}: command interrupted')
        finally:
            if self.timing:self.timing.event('tool_end',id=call_id)
        if time.time() >= self.deadline:
            raise TimeExceeded(dict(role='exit', content='TimeExceeded', extra=dict(exit_status='TimeExceeded')))
        lines = output['output'].lstrip().splitlines()
        if lines and lines[0].strip() == 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' and output['returncode'] == 0:
            raise Submitted(dict(role='exit', content='', extra=dict(exit_status='Submitted', submission='')))
        output['output'] += f'\n[Benchmark: {max(0,int(self.deadline-time.time()))} seconds remain in this coding stage.]'
        return output

    def get_template_vars(self, **kwargs):
        return dict(system='Linux', release='Modal', version='', machine='x86_64', cwd='/bot', **kwargs)

    def serialize(self):
        return dict(info=dict(environment_type=type(self).__name__, backend=getattr(self.sandbox, 'backend', 'modal'),
                              network=False, cpu=2, memory_mib=2048))



def continuing_agent_class(base,cfg):
    """Durable atomic conversation receipts, including before the first API call."""
    class ContinuingAgent(base):
        def save(self,path,*extra):
            if path is None:return super().save(path,*extra)
            path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
            data=super().save(temporary,*extra)
            temporary.replace(path)
            return data

        def add_messages(self,*messages):
            if not self.messages and cfg.get('history'):
                history=json.loads(Path(cfg['history']).read_text())['messages']
                messages=(*continued_history(history),self.model.format_message(role='user',content=cfg['task']))
            added=super().add_messages(*messages)
            self.save(self.config.output_path)
            return added
    return ContinuingAgent


def main():
    import minisweagent
    from minisweagent.agents.default import DefaultAgent
    cfg = json.load(sys.stdin)
    assert minisweagent.__version__ == MINI_VERSION
    if cfg.get('sandbox_backend') == 'docker':
        from .docker_sandbox import DockerSandbox
        sandbox = DockerSandbox.from_id(cfg['sandbox_id'])
    elif cfg.get('sandbox_backend') == 'harbor':
        from .harbor_bridge import HarborSandbox
        sandbox = HarborSandbox.from_id(cfg['sandbox_id'])
    else:
        import modal
        sandbox = modal.Sandbox.from_id(cfg['sandbox_id'])
    from .telemetry import TimingLog
    timing=TimingLog(cfg.get('timing'))
    env = ModalBashEnvironment(sandbox, cfg.get('deadline_unix',float('inf')),timing)
    if cfg.get('cpu_slots_directory'):
        from .local_scheduler import bash_environment
        env = bash_environment(sandbox, env.deadline, timing, cfg)
    if cfg.get('continuous'):
        env.command_seconds = 1800.
    # Resume the exact linear conversation, removing only the prior stage's exit record.
    # The agent's reasoning, bash tool parsing, observations and step loop remain upstream.
    ContinuingAgent=continuing_agent_class(DefaultAgent,cfg)
    # Legacy cloud worker payloads omitted provider and were always OpenRouter.
    # Local callers default to coding-plan routing; explicit configs win everywhere.
    import os
    if 'provider' not in cfg and os.environ.get('MODAL_TASK_ID'):
        cfg['provider'] = 'openrouter'
    model = coding_model(cfg, timing=timing)
    agent = ContinuingAgent(model, env, system_template=cfg['system'], instance_template='{{ task }}',
                            step_limit=0, cost_limit=0, wall_time_limit_seconds=0 if cfg.get('cpu_slots_directory') else cfg['seconds'],
                            max_consecutive_format_errors=0 if cfg.get('continuous') else 10,
                            output_path=Path(cfg['trajectory']))
    # All imports, sandbox attachment and agent construction are outside the coding
    # clock. No model call or coder command occurs before the trusted start receipt.
    if cfg.get('ready_path'):
        from .common import write_json
        write_json(cfg['ready_path'],dict(ready_unix=time.time()))
        waiting=time.monotonic()
        while not Path(cfg['start_path']).exists():
            if time.monotonic()-waiting>120:raise TimeoutError('controller never started stage')
            time.sleep(.01)
        start=json.loads(Path(cfg['start_path']).read_text())
        env.deadline=start['deadline_unix']
        timing.event('stage_start',started_unix=start['started_unix'],deadline_unix=env.deadline)
    import signal
    from minisweagent.exceptions import TimeExceeded
    def cutoff(signum,frame):
        raise TimeExceeded(dict(role='exit',content='TimeExceeded',extra=dict(exit_status='TimeExceeded')))
    signal.signal(signal.SIGTERM,cutoff)
    try:
        agent.run(cfg['task'])
    finally:
        agent.save(Path(cfg['trajectory']))


if __name__ == '__main__':
    main()
