"""Private host-only bridge from the trusted Mini worker to its Harbor environment.

The Unix socket is never mounted in the coder. Harbor owns cloud authentication,
resource allocation, offline execution and artifact transfer. No SDK key or model
credential is serialized into RPC requests or copied into the remote environment.
"""
import asyncio
import base64
import io
import json
import math
import os
from pathlib import Path
import shlex
import socket
import socketserver
import tempfile
import threading
import uuid


class HarborBridge:
    def __init__(self, environment):
        self.environment = environment
        self.loop = asyncio.get_running_loop()
        self.directory = tempfile.TemporaryDirectory(prefix='skirmish-rpc-')
        os.chmod(self.directory.name, 0o700)
        self.path = str(Path(self.directory.name)/'bridge.sock')
        self.operations = {}
        self.lock = threading.Lock()
        bridge = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    request = json.loads(self.rfile.readline())
                    result = bridge.dispatch(request)
                    response = dict(ok=True, result=result)
                except Exception as error:
                    # Do not relay provider exception bodies or authentication data.
                    response = dict(ok=False, error=type(error).__name__)
                self.wfile.write((json.dumps(response)+'\n').encode())

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

        self.server = Server(self.path, Handler)
        os.chmod(self.path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    async def prepare(self):
        result = await self.environment.exec(
            'mkdir -p /verify/transfers && chmod 700 /verify/transfers && test "$(id -u)" = 0',
            user='root', timeout_sec=30)
        if result.return_code:
            raise RuntimeError('trusted Harbor root execution unavailable')

    async def write(self, path, data):
        with tempfile.TemporaryDirectory(prefix='skirmish-upload-') as folder:
            source = Path(folder)/'data'
            source.write_bytes(base64.b64decode(data))
            remote = '/verify/transfers/'+uuid.uuid4().hex
            await self.environment.upload_file(source, remote)
            code = ('from pathlib import Path;import os,sys;'
                    'p=Path(sys.argv[2]);p.parent.mkdir(parents=True,exist_ok=True);'
                    't=p.with_name(p.name+"."+sys.argv[3]);'
                    't.write_bytes(Path(sys.argv[1]).read_bytes());os.replace(t,p);'
                    'Path(sys.argv[1]).unlink()')
            command = shlex.join(['python','-c',code,remote,path,uuid.uuid4().hex])
            result = await self.environment.exec(command, user='root', timeout_sec=30)
            if result.return_code:
                raise RuntimeError('trusted file upload failed')

    async def read(self, path):
        with tempfile.TemporaryDirectory(prefix='skirmish-download-') as folder:
            destination = Path(folder)/'data'
            await self.environment.download_file(path, destination)
            return base64.b64encode(destination.read_bytes()).decode()

    def dispatch(self, request):
        action = request['action']
        if action in ('read','write'):
            coroutine = (self.read(request['path']) if action == 'read' else
                         self.write(request['path'], request['data']))
            return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout=180)
        if action == 'exec':
            timeout = math.ceil(request['timeout'])
            command = shlex.join(['timeout','--signal=KILL',str(timeout), *request['args']])
            future = asyncio.run_coroutine_threadsafe(self.environment.exec(
                command, user='root', timeout_sec=timeout+10), self.loop)
            identifier = uuid.uuid4().hex
            with self.lock:
                self.operations[identifier] = future
            return identifier
        if action == 'wait':
            with self.lock:
                future = self.operations[request['id']]
            result = future.result(timeout=max(180,request['timeout']+30))
            with self.lock:
                self.operations.pop(request['id'], None)
            return dict(stdout=result.stdout, stderr=result.stderr, returncode=result.return_code)
        raise ValueError('unknown trusted bridge operation')

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        with self.lock:
            for future in self.operations.values():
                future.cancel()
            self.operations.clear()
        self.directory.cleanup()


class HarborSandbox:
    backend = 'harbor'

    def __init__(self, path):
        self.object_id = path
        self.filesystem = BridgeFilesystem(self)

    @classmethod
    def from_id(cls, path):
        if not Path(path).is_socket():
            raise RuntimeError('trusted Harbor bridge unavailable')
        return cls(path)

    def request(self, **request):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(max(225,request.get('timeout',180)+45))
            connection.connect(self.object_id)
            stream = connection.makefile('rwb')
            stream.write((json.dumps(request)+'\n').encode())
            stream.flush()
            response = json.loads(stream.readline())
        if not response['ok']:
            raise RuntimeError('trusted Harbor transport: '+response['error'])
        return response['result']

    def exec(self, *args, timeout=60):
        identifier = self.request(action='exec', args=list(args), timeout=timeout)
        return BridgeProcess(self, identifier, timeout)


class BridgeFilesystem:
    def __init__(self, sandbox):
        self.sandbox = sandbox

    def write_bytes(self, data, path):
        self.sandbox.request(action='write', path=path, data=base64.b64encode(data).decode())

    def write_text(self, data, path):
        self.write_bytes(data.encode(), path)

    def read_bytes(self, path):
        return base64.b64decode(self.sandbox.request(action='read', path=path))

    def read_text(self, path):
        return self.read_bytes(path).decode()


class BridgeProcess:
    def __init__(self, sandbox, identifier, timeout):
        self.sandbox, self.identifier, self.timeout = sandbox, identifier, timeout
        self.result = None
        self.stdout = self
        self.stderr = io.StringIO('')

    def wait(self):
        if self.result is None:
            self.result = self.sandbox.request(action='wait', id=self.identifier, timeout=self.timeout)
        return self.result['returncode']

    def read(self):
        self.wait()
        return self.result['stdout']+self.result['stderr']

    @property
    def returncode(self):
        return None if self.result is None else self.result['returncode']
