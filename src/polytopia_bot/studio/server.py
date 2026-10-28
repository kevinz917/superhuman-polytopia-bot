"""Run with python -m polytopia_bot.studio.server (loopback only)."""
import argparse
import base64
import gzip
import json
import mimetypes
import os
from pathlib import Path
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Bound local inference resources independently of the campaign.
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')

from .service import Studio, Conflict, STATIC, ROOT, RULES, TERRAIN, RESOURCE, ABILITY, REWARD, dump


def serve(studio,port,web=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args):
            if args and ('/api/jobs/' in str(args[0])):return
            super().log_message(fmt,*args)

        def send(self,value,status=200,ctype='application/json',attachment=None):
            data=dump(value).encode() if ctype=='application/json' else value
            self.send_response(status)
            self.send_header('Content-Type',ctype)
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','same-origin')
            if attachment:self.send_header('Content-Disposition',f'attachment; filename="{attachment}"')
            if 'gzip' in self.headers.get('Accept-Encoding','') and len(data)>2048:
                data=gzip.compress(data);self.send_header('Content-Encoding','gzip')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass

        def route(self,post=False):
            parsed=urlparse(self.path); path=parsed.path
            q={k:v[0] for k,v in parse_qs(parsed.query).items()}
            host=self.headers.get('Host','')
            if host not in (f'127.0.0.1:{port}',f'localhost:{port}'):raise PermissionError('Loopback host required')
            data={}
            if post:
                origin=self.headers.get('Origin')
                if origin and origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
                    raise PermissionError('Cross-origin mutations are disabled')
                if self.headers.get('X-Studio-Client')!='1':raise PermissionError('X-Studio-Client: 1 required')
                length=int(self.headers.get('Content-Length','0'))
                if length>48*1024*1024:raise ValueError('Upload exceeds 48 MiB')
                data=json.loads(self.rfile.read(length) or b'{}')
            if path=='/api/meta':
                return self.send({'bot':studio.bot,'ready':studio.ready,'warmup_job':studio.warmup_job,
                                 'search':{'available':studio.search_workers is not None,'ready':studio.search_ready},
                                 'rules':RULES,'terrain':TERRAIN,'resources':RESOURCE,'abilities':ABILITY,'rewards':REWARD})
            if path=='/api/library':
                return self.send({'replays':studio.library(),'games':studio.session_library()})
            if path=='/api/games' and post:return self.send(studio.submit(lambda:studio.create(data)),202)
            if path=='/api/import' and post:
                return self.send(studio.submit(lambda:(studio.reconstruct_trace if data.get('reconstruct') else studio.import_trace)(base64.b64decode(data['data'],validate=True),str(data.get('name','Imported replay'))[:120])),202)
            match=re.fullmatch(r'/api/jobs/([a-f0-9]{32})',path)
            if match:return self.send(studio.jobs[match[1]])
            match=re.fullmatch(r'/api/(games|replays)/([a-f0-9]{32})(?:/(action|advance|observation|save|export|close|settings|series))?',path)
            if match:
                kind,ident,operation=match.groups()
                if kind=='replays':
                    if operation=='export':return self.send(studio.replays[ident],attachment='bardur-replay.json')
                    if operation=='series':return self.send(studio.replay_series(ident))
                    return self.send(studio.replay_frame(ident,q.get('frame',0),q.get('view','actor')))
                g=studio.games.get(ident)
                if g is None:
                    # Restore runs in the single engine executor, never on HTTP threads.
                    return self.send(studio.submit(lambda:{'restored':True,**studio.describe(studio.restore(ident))}),202)
                if operation=='settings':
                    if not post:raise ValueError('POST required')
                    return self.send(studio.set_search(ident,data))
                if operation in ('action','advance','save','close'):
                    if not post:raise ValueError('POST required')
                    with studio.lock:
                        if g['busy']:raise Conflict('An action is already pending')
                        g['busy']=True
                    def work():
                        try:
                            # advance locks per engine step itself, so a search can run while frames are served.
                            if operation=='advance':return studio.advance(ident,data)
                            with studio.lock:
                                if operation=='action':return studio.action(ident,data)
                                if operation=='advance':return studio.advance(ident,data)
                                if operation=='save':return studio.save(g)
                                result=studio.save(g);g['env'].close();del studio.games[ident];(studio.storage/'sessions'/f'{ident}.json').unlink(missing_ok=True);return result
                        finally:g['busy']=False
                    return self.send(studio.submit(work),202)
                if operation=='observation':
                    with studio.lock:return self.send(studio.observation(ident))
                if operation=='export':
                    with studio.lock:studio.save(g)
                    return self.send(studio.replays[ident],attachment='bardur-replay.json')
                return self.send(studio.public_frame(g,q.get('frame'),q.get('view')))
            if path.startswith('/assets/'):
                name=path.removeprefix('/assets/')
                # Publisher sprites for the legacy page (tools/studio/setup.py).
                if not re.fullmatch(r'[a-z0-9-]+\.(png|json)',name):raise KeyError('Asset not found')
                target=ROOT/'artifacts/mobile-assets'/name
                if not target.is_file():raise KeyError('Asset not installed')
                return self.send(json.loads(target.read_text())) if name.endswith('.json') else self.send(target.read_bytes(),ctype='image/png')
            if path.startswith('/api/'):raise KeyError('Unknown endpoint')
            # The React build (web/dist) is the interface; the original page stays at /legacy/.
            root=STATIC
            if web and not path.startswith('/legacy'):root=web
            else:path=path.removeprefix('/legacy') or '/'
            target=(root/('index.html' if path in ('','/') else path.lstrip('/'))).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():raise KeyError('Not found')
            return self.send(target.read_bytes(),ctype=mimetypes.guess_type(target.name)[0] or 'application/octet-stream')

        def handle_request(self,post=False):
            try:self.route(post)
            except Conflict as e:self.send({'error':str(e)},409)
            except PermissionError as e:self.send({'error':str(e)},403)
            except KeyError as e:self.send({'error':str(e)},404)
            except (ValueError,TypeError) as e:self.send({'error':str(e)},400)
            except Exception as e:
                import traceback;traceback.print_exc();self.send({'error':str(e)},500)
        def do_GET(self):self.handle_request()
        def do_POST(self):self.handle_request(True)

    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Studio ready at http://127.0.0.1:{port}',flush=True)
    server.serve_forever()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--storage',type=Path,default=ROOT/'artifacts/studio')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'weights/champion/final.npz')
    p.add_argument('--import-trace',type=Path)
    p.add_argument('--web',type=Path,default=ROOT/'web/dist',help='Built React interface; falls back to the legacy page')
    p.add_argument('--name',default='Bardur · Champion',help='Bot display name')
    p.add_argument('--short',default='Champion',help='Short bot name for game titles')
    p.add_argument('--training',default='Self-play PPO from random weights',help='One-line training provenance shown in match details')
    p.add_argument('--no-warmup',action='store_true',
                   help='Replay browsing only: skip compiling the engine and policy at startup')
    p.add_argument('--featured',help='Replay id to open by default and list first')
    p.add_argument('--search',action='store_true',
                   help='Offer leak-free robust turn search as a per-game opponent mode (switchable mid-game)')
    p.add_argument('--search-workers',type=int,default=max(1,(os.cpu_count() or 2)-2),
                   help='Threads for parallel search lanes (default: all cores but two)')
    p.add_argument('--opponent',action='append',default=[],metavar='NAME=PATH',
                   help='Named rival checkpoint a watch game can seat against the bot (raw policy); repeatable')
    args=p.parse_args()
    opponents={}
    for spec in args.opponent:
        if '=' not in spec: raise SystemExit(f'--opponent {spec}: expected NAME=PATH')
        label,path=spec.split('=',1); opponents[label]=Path(path)
    studio=Studio(args.storage,args.checkpoint,name=args.name,short=args.short,training=args.training,
                  search_workers=args.search_workers if args.search else None,opponents=opponents)
    if args.featured:
        if args.featured not in studio.replays: raise SystemExit(f'--featured {args.featured}: no such replay')
        studio.featured=args.featured
    if args.import_trace:
        studio.import_trace(args.import_trace.read_bytes(),'Champion vs imported trace · recorded match')
    studio.warmup_job=None if args.no_warmup else studio.submit(studio.warmup)['job_id']
    if args.search and not args.no_warmup:studio.submit(studio.warm_search)
    web=args.web if (args.web/'index.html').is_file() else None
    print('Interface:',web or 'legacy static page',flush=True)
    serve(studio,args.port,web)

if __name__=='__main__':main()
