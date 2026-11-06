"""Local Docker isolation implementing the small sandbox API used by Mini-SWE.

Coder containers have no host mounts, network, Docker socket or API credentials.
The trusted host controls lifecycle and copies only explicitly selected files.
"""
import json
import math
import io
import os
from pathlib import Path
import subprocess
import uuid


def docker(*args):
    context = os.environ.get('SKIRMISH_DOCKER_CONTEXT', 'colima-polytopia-skirmish')
    return ['docker', '--context', context, *map(str, args)]


class Filesystem:
    def __init__(self, name):
        self.name = name

    def write_bytes(self, data, path):
        script = 'import sys,uuid;from pathlib import Path;p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+uuid.uuid4().hex);t.write_bytes(sys.stdin.buffer.read());t.replace(p)'
        subprocess.run(docker('exec', '-i', self.name, 'python', '-c', script, path),
                       input=data, check=True, capture_output=True, timeout=30)

    def write_text(self, data, path):
        self.write_bytes(data.encode(), path)

    def read_bytes(self, path):
        return subprocess.run(docker('exec', self.name, 'cat', path), check=True,
                              capture_output=True, timeout=30).stdout

    def read_text(self, path):
        return self.read_bytes(path).decode()


class DockerSandbox:
    backend = 'docker'

    def __init__(self, name):
        self.object_id = name
        self.filesystem = Filesystem(name)

    @classmethod
    def create(cls, image):
        name = 'polytopia-coder-'+uuid.uuid4().hex
        subprocess.run(docker('run', '-d', '--name', name, '--label', 'polytopia.skirmish.role=coder',
            '--network', 'none', '--cpus', '2', '--memory', '2g', '--memory-swap', '2g',
            '--pids-limit', '512', '--cap-drop', 'ALL', '--cap-add', 'KILL',
            '--security-opt', 'no-new-privileges=true', '--workdir', '/bot',
            image, 'sleep', 'infinity'), check=True, capture_output=True, timeout=60)
        return cls(name)

    @classmethod
    def from_id(cls, name):
        data = json.loads(subprocess.run(docker('inspect', name), check=True,
                          capture_output=True, text=True, timeout=10).stdout)[0]
        if data['Config'].get('Labels', {}).get('polytopia.skirmish.role') != 'coder':
            raise ValueError('Refusing to attach an unrelated Docker container')
        return cls(name)

    def exec(self, *args, timeout=60):
        options = []
        # Docker controls uid, so no SETUID/SETGID capabilities are needed.
        if args[:5] == ('/usr/sbin/runuser', '-u', 'coder', '--', 'env'):
            options = ['--user', 'coder']
            args = args[4:]
        process = subprocess.Popen(docker('exec', *options, self.object_id,
            'timeout', '--signal=KILL', str(math.ceil(timeout)), *args),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return ExecProcess(process)

    def terminate(self):
        subprocess.run(docker('rm', '-f', self.object_id), capture_output=True, timeout=30)


class LocalVolume:
    """Host files are already durable; no cloud storage calls occur."""
    def commit(self):
        pass


class ExecProcess:
    def __init__(self, process):
        self.process = process
        self.stdout = process.stdout
        self.stderr = io.StringIO('')  # stderr is merged to prevent pipe deadlocks

    @property
    def returncode(self):
        return self.process.returncode

    def wait(self):
        return self.process.wait()
