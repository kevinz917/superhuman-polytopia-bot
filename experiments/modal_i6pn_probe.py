"""Lever 5 probe (docs/levers-egress-2026-10-22.md): can region-pinned containers talk over i6pn, and is that
traffic billed as egress? One server + N clients per arm, tiny CPU containers; each client streams `mb` MB.

    modal run --env staging experiments/modal_i6pn_probe.py --arms us,none --clients 4 --mb 2048

Read the egress line afterwards: modal billing report --start <date> -r h --show-resources (app polytopia-i6pn-probe).
"""
import json
import os
import socket
import time

import modal

if os.environ.get('MODAL_ENVIRONMENT', 'staging') != 'staging':
    raise RuntimeError('staging only')
app = modal.App('polytopia-i6pn-probe')
image = modal.Image.debian_slim(python_version='3.12')
board = modal.Dict.from_name('polytopia-i6pn-probe', create_if_missing=True)
PORT = 7090


def _where():
    return {k: os.environ.get(k) for k in ('MODAL_CLOUD_PROVIDER', 'MODAL_REGION')}


def _i6pn_address():
    return socket.getaddrinfo('i6pn.modal.local', None, socket.AF_INET6)[0][4][0]


def _serve(key, clients, seconds):
    addr = _i6pn_address()
    server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('::', PORT))
    server.listen(64)
    server.settimeout(seconds)
    board[key] = dict(address=addr, **_where())
    got = []
    try:
        for _ in range(clients):
            conn, peer = server.accept()
            total, t = 0, time.monotonic()
            conn.settimeout(120)
            while True:
                chunk = conn.recv(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
            got.append(dict(peer=peer[0], bytes=total, seconds=time.monotonic() - t))
            conn.close()
    except OSError as error:
        got.append(dict(error=repr(error)))
    return dict(server=dict(address=addr, **_where()), received=got)


def _client(key, mb):
    deadline = time.time() + 300
    while key not in board and time.time() < deadline:
        time.sleep(1)
    entry = board[key]
    out = dict(client=dict(**_where(), own=_i6pn_address()), server=entry)
    try:
        sock = socket.create_connection((entry['address'], PORT), timeout=20)
        payload = os.urandom(1 << 20)
        t = time.monotonic()
        for _ in range(mb):
            sock.sendall(payload)
        sock.close()
        out.update(connected=True, mb=mb, mb_per_s=mb / (time.monotonic() - t))
    except OSError as error:
        out.update(connected=False, error=repr(error))
    return out


@app.function(image=image, cpu=1, memory=1024, timeout=1200, i6pn=True, region='us')
def server_us(key: str, clients: int, seconds: int):
    return _serve(key, clients, seconds)


@app.function(image=image, cpu=1, memory=1024, timeout=1200, i6pn=True, region='us')
def client_us(key: str, mb: int):
    return _client(key, mb)


@app.function(image=image, cpu=1, memory=1024, timeout=1200, i6pn=True)
def server_none(key: str, clients: int, seconds: int):
    return _serve(key, clients, seconds)


@app.function(image=image, cpu=1, memory=1024, timeout=1200, i6pn=True)
def client_none(key: str, mb: int):
    return _client(key, mb)


@app.local_entrypoint()
def main(arms: str = 'us,none', clients: int = 4, mb: int = 2048):
    fns = dict(us=(server_us, client_us), none=(server_none, client_none))
    calls = {}
    for arm in arms.split(','):
        key = f'{arm}-{int(time.time())}'
        try:
            board.pop(key)
        except KeyError:
            pass
        server, client = fns[arm]
        calls[arm] = (server.spawn(key, clients, 600), [client.spawn(key, mb) for _ in range(clients)])
    for arm, (s, cs) in calls.items():
        rows = []
        for c in cs:
            try:
                rows.append(c.get(timeout=900))
            except Exception as error:
                rows.append(dict(error=repr(error)))
        try:
            srv = s.get(timeout=120)
        except Exception as error:
            srv = dict(error=repr(error))
        print(json.dumps(dict(arm=arm, server=srv, clients=rows)), flush=True)
