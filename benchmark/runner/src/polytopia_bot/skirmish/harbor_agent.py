"""One authoring protocol: local/proxy or Daytona/OpenRouter, selected explicitly."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path
import subprocess
import threading

from harbor.agents.base import BaseAgent
from harbor.environments.docker.docker import _sanitize_docker_compose_project_name
from .common import write_json
from .continuous import CODING_SECONDS, SCHEMA, source_files
from .docker_sandbox import DockerSandbox, docker
from .inference import endpoint
from .local import run_stage

# Long-lived authoring controllers must not occupy asyncio's small default pool:
# SDK file transfers and short setup/cleanup work use that pool too.
CONTROLLERS = ThreadPoolExecutor(max_workers=128,thread_name_prefix='skirmish-controller')


class SkirmishAgent(BaseAgent):
    def __init__(self, *args, build_receipt=None, provider='cliproxyapi',
                 parameters=None, execution_backend='docker', local_cpu_slots=6,
                 inference_base_url=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_receipt = Path(build_receipt) if build_receipt else None
        self.execution_backend = execution_backend
        self.inference_provider = provider
        self.parameters = parameters or dict(reasoning=dict(effort='high'), max_tokens=32768)
        self.local_cpu_slots = local_cpu_slots
        self.inference_base_url = inference_base_url
        self.bridge = None

    @staticmethod
    def name():
        return 'polytopia-skirmish'

    def version(self):
        return SCHEMA

    async def setup(self, environment):
        try:
            await self._setup(environment)
        except BaseException:
            if self.bridge:
                await asyncio.to_thread(self.bridge.close)
                self.bridge = None
            raise
        finally:
            if self.execution_backend == 'daytona' and hasattr(environment,'release_startup'):
                environment.release_startup()

    async def _setup(self, environment):
        if not self.model_name:
            raise ValueError('an explicit model is required')
        if (self.execution_backend, self.inference_provider) not in (
                ('docker','cliproxyapi'), ('daytona','openrouter')):
            raise ValueError('allowed routes: local CLI proxy or Daytona OpenRouter; no fallback')
        if self.execution_backend == 'docker':
            project = _sanitize_docker_compose_project_name(environment.session_id)
            def attach():
                output = subprocess.run(docker('ps','-q','--filter',f'label=com.docker.compose.project={project}',
                    '--filter','label=com.docker.compose.service=main'), check=True, capture_output=True, text=True).stdout.split()
                if len(output) != 1:
                    raise RuntimeError('cannot uniquely identify Harbor agent container')
                return DockerSandbox.from_id(output[0])
            self.sandbox = await asyncio.to_thread(attach)
        else:
            from .harbor_environments import SkirmishDaytonaEnvironment
            from .harbor_bridge import HarborBridge, HarborSandbox
            if not isinstance(environment, SkirmishDaytonaEnvironment):
                raise ValueError('Daytona route requires the pinned benchmark environment')
            self.bridge = HarborBridge(environment)
            await self.bridge.prepare()
            self.sandbox = HarborSandbox(self.bridge.path)
        def isolation_check():
            # Harbor normally makes log mounts writable for installed agents.
            # This agent/controller lives on the host: the coder must not edit
            # its host clock, conversation, or future verifier receipt mounts.
            # Preserve mount ownership so the trusted host can still write.
            command = ('mkdir -p /logs/agent /logs/verifier && chmod 755 /logs && chmod 700 /logs/agent /logs/verifier && '
                'test ! -e /refs && test ! -e /suite/maps.json && '
                '/usr/sbin/runuser -u coder -- /bin/sh -c '
                "'test ! -w /verify && test ! -w /logs && test ! -w /logs/agent && test ! -w /logs/verifier && ! sudo -n true 2>/dev/null'")
            process = self.sandbox.exec('/bin/sh','-c',command,timeout=30)
            process.wait()
            if process.returncode:
                raise RuntimeError('coder isolation contract failed')
        await asyncio.to_thread(isolation_check)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config = dict(provider=self.inference_provider, model=self.model_name,
            parameters=self.parameters, continuous=True)
        if self.inference_base_url:
            self.config['base_url'] = self.inference_base_url
        route = endpoint(self.config)
        route.credential()
        self.stop_event = threading.Event()
        pinned = source_files()
        if self.build_receipt and json.loads(self.build_receipt.read_text())['source_files'] != pinned:
            raise ValueError('source drift since Harbor package build; rebuild with a new package')
        wire = 'anthropic' if route.backend=='cliproxyapi' and route.model.startswith('claude-') else 'chat'
        write_json(self.logs_dir/'protocol.json', dict(schema=SCHEMA, coding_seconds=CODING_SECONDS,
            inference={**route.receipt(),'api_protocol':wire}, source_files=pinned, submitted_source='current main.rs at cutoff',
            practice='on demand; compilation, play, audit and trace reading all charged'))

    async def run(self, instruction, environment, context):
        try:
            return await self._run(instruction, environment, context)
        finally:
            if self.bridge:
                await asyncio.to_thread(self.bridge.close)
                self.bridge = None

    async def _run(self, instruction, environment, context):
        payload = dict(self.config, seconds=CODING_SECONDS, task=instruction,
            system='You are a coding agent building an offline Rust game bot. Use bash for actions. '
                   'Follow the supplied benchmark protocol. Work in /bot. '
                   'When finished, call bash with: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT',
            trajectory=str(self.logs_dir/'trajectory.json'))
        slots = Path.home()/'.cache/polytopia-skirmish/cpu-slots' if self.execution_backend=='docker' else None
        work = asyncio.get_running_loop().run_in_executor(CONTROLLERS,partial(
            run_stage,self.sandbox,self.logs_dir,0,payload,slots,self.local_cpu_slots,
            stop_event=self.stop_event))
        try:
            reason, timing = await asyncio.shield(work)
        except asyncio.CancelledError:
            self.stop_event.set()
            await asyncio.shield(work)
            raise
        if reason in ('agent_error','cancelled'):
            raise RuntimeError(f'authoring infrastructure failure: {reason}')
        # Read root-owned capture, never trust an agent-supplied manifest or Wasm.
        capture = json.loads(await asyncio.to_thread(self.sandbox.filesystem.read_text,
            '/verify/cutoffs/stage-0/capture.json'))
        trajectory = json.loads((self.logs_dir/'trajectory.json').read_text())
        exit_status = trajectory.get('info',{}).get('exit_status')
        if exit_status == 'TimeExceeded':
            reason = 'deadline'
        elif exit_status != 'Submitted' and reason != 'deadline':
            raise RuntimeError(f'agent exited without submission or deadline: {exit_status}')
        usage = dict(input_tokens=0, output_tokens=0, completed_responses=0)
        costs = []
        for message in trajectory['messages']:
            response = message.get('extra',{}).get('response') or {}
            if response:
                u = response.get('usage') or {}
                usage['input_tokens'] += u.get('prompt_tokens',0)
                usage['output_tokens'] += u.get('completion_tokens',0)
                usage['completed_responses'] += 1
                costs.append(u.get('cost'))
        context.n_input_tokens = usage['input_tokens']
        context.n_output_tokens = usage['output_tokens']
        known = all(isinstance(c,(int,float)) for c in costs)
        context.cost_usd = sum(costs) if self.inference_provider=='openrouter' and costs and known else None
        context.metadata = dict(protocol=SCHEMA, timing=timing, usage=usage, end_reason=reason,
                                capture=capture, coding_budget_seconds=CODING_SECONDS,
                                execution_backend=self.execution_backend,
                                completed_call_cost_usd=context.cost_usd)
        write_json(self.logs_dir/'authoring.json', context.metadata)
