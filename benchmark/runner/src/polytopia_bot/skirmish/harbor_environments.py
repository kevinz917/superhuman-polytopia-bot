"""Native Harbor environments with explicit local admission and Daytona snapshots."""
import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
import fcntl
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.daytona.environment import DaytonaEnvironment

_startup_gate = None
_submission_gate = None


def daytona_gates():
    global _startup_gate, _submission_gate
    if _startup_gate is None:
        _startup_gate = asyncio.Semaphore(4)
        _submission_gate = asyncio.Semaphore(8)
    return _startup_gate, _submission_gate


@asynccontextmanager
async def slot(directory, count):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    streams = [(directory/f'{i}.lock').open('a') for i in range(count)]
    acquired = None
    try:
        while acquired is None:
            for stream in streams:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    acquired = stream
                    break
                except BlockingIOError:
                    continue
            if acquired is None:
                await asyncio.sleep(.2)
        yield
    finally:
        if acquired is not None:
            fcntl.flock(acquired, fcntl.LOCK_UN)
        for stream in streams:
            stream.close()


class SkirmishDockerEnvironment(DockerEnvironment):
    def __init__(self, *args, cpu_slots=6, grader_workers=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_slots = cpu_slots
        self.grader_workers = grader_workers
        self.leases = AsyncExitStack()

    async def _run_docker_compose_command(self, command, *args, **kwargs):
        # Many later trials share the same immutable local image IDs. Harbor's
        # default `down --rmi local` can remove one after an early trial finishes.
        # Keep images; still remove the trial's containers, networks and volumes.
        if command and command[0] == 'down' and '--rmi' in command:
            command = list(command)
            index = command.index('--rmi')
            del command[index:index+2]
        return await super()._run_docker_compose_command(command,*args,**kwargs)

    async def start(self, force_build):
        try:
            if self.environment_dir.name == 'tests':
                directory = Path.home()/'.cache/polytopia-skirmish/cpu-slots'
                # Reserve at most four of six CPU slots for graders. Authoring
                # tool calls use the same flock pool, so API-waiting sessions
                # can remain highly concurrent without oversubscribing CPUs.
                await self.leases.enter_async_context(slot(directory/'graders', self.grader_workers))
                await self.leases.enter_async_context(slot(directory, self.cpu_slots))
            await super().start(force_build)
        except BaseException:
            await self.leases.aclose()
            raise

    async def stop(self, delete):
        try:
            await super().stop(delete)
        finally:
            await self.leases.aclose()


class SkirmishDaytonaEnvironment(DaytonaEnvironment):
    async def start(self, force_build):
        startup, _ = daytona_gates()
        await startup.acquire()
        self._startup_lease = True
        try:
            await super().start(force_build)
            if self.environment_dir.name == 'tests':
                self.release_startup()
        except BaseException:
            self.release_startup()
            raise

    def release_startup(self):
        if getattr(self,'_startup_lease',False):
            self._startup_lease = False
            daytona_gates()[0].release()

    async def stop(self, delete):
        try:
            await super().stop(delete)
        finally:
            self.release_startup()

    async def _create_sandbox(self, *args, **kwargs):
        await super()._create_sandbox(*args,**kwargs)
        process = self._sandbox.process
        original = process.execute_session_command
        _, submissions = daytona_gates()
        async def submit(session_id, req, timeout=None):
            # This is the HTTP acknowledgement, not the execution deadline.
            # Native Harbor still embeds the original `timeout` in the remote
            # shell, and our independent cutoff watchdog remains authoritative.
            # Never retry submissions: an ambiguous timeout could duplicate work.
            async with submissions:
                return await original(session_id,req,timeout=120)
        process.execute_session_command = submit

    async def _resolve_start_sandbox_params(self, daytona, resources, *, force_build):
        image = self.task_env_config.docker_image or ''
        if not image.startswith('daytona-snapshot:'):
            raise ValueError('Daytona benchmark requires a versioned snapshot package')
        name = image.removeprefix('daytona-snapshot:')
        snapshot = await daytona.snapshot.get(name)
        expected_id = self._kwargs.get('snapshot_ids',{}).get(name)
        if expected_id and snapshot.id != expected_id:
            raise ValueError('Daytona snapshot identity changed since package build')
        if snapshot.cpu != self._effective_cpus or snapshot.mem*1024 != self._effective_memory_mb:
            raise ValueError('Daytona snapshot resource shape differs from the benchmark contract')
        params = self._snapshot_sandbox_params(name)
        params.ttl_minutes = 180
        return params
