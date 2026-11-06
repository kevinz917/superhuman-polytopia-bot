"""The identical, small model-facing interface for every provider."""
from dataclasses import asdict
import json
import math
from pathlib import Path, PurePosixPath
import time

from .common import REFERENCE, REFERENCE_SHA, PRACTICE_START, canonical, sha, write_json
from .game import ReferenceBot, WasiBot, play_game


def schema(name, description, properties, required=()):
    return dict(name=name, description=description, parameters=dict(type='object', properties=properties,
                required=list(required), additionalProperties=False))


TOOLS = [
    schema('budget_status', 'Read neutral, factual time, revision and practice allowances.', {}),
    schema('finish', 'Freeze your current compiled bot when ready. Required when no useful revisions remain.', {}),
    schema('read_file', 'Read /bot/src/main.rs or files beneath /reference (README.md, rules.json, engine-rs/src/*.rs).',
           {'path': {'type': 'string'}}, ['path']),
    schema('write_source', 'Replace /bot/src/main.rs with complete Rust source. Only std and serde_json are available.',
           {'source': {'type': 'string'}}, ['source']),
    schema('compile', 'Compile current Rust source offline. Call after every edit. Returns errors or immutable build hash.', {}),
    schema('campaign', 'Play the latest compiled bot against hero-1v1, both seats on fresh practice maps. Returns game IDs.',
           {'maps': {'type': 'integer', 'minimum': 1, 'maximum': 4}}, ['maps']),
    schema('review', 'Read a page of your own public observations/actions from a practice game. Never opponent private state.',
           {'game_id': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0},
            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 8}}, ['game_id', 'offset', 'limit']),
]


class AuthoringTools:
    def __init__(self, sandbox, root, budget, volume, deadline):
        self.sandbox, self.root, self.budget, self.volume = sandbox, Path(root), budget, volume
        self.deadline = deadline
        self.source = ''
        self.compiled_source = None
        self.artifact = None
        self.games = {}
        self.practice_count = 0
        self.reference = Path('/results/references/hero-1v1/final.npz')
        self.reference_bot = None
        self.revisions = 0
        self.finished = False

    def remaining(self):
        return self.deadline - time.monotonic()

    def read(self, path, mode='r'):
        return (self.sandbox.filesystem.read_bytes(path) if 'b' in mode
                else self.sandbox.filesystem.read_text(path))

    def write(self, source):
        if not isinstance(source, str) or len(source.encode()) > 200_000:
            raise ValueError('source must be a string up to 200000 bytes')
        changed = source != self.source
        if changed and self.budget.max_revisions is not None and self.revisions >= self.budget.max_revisions:
            raise ValueError('revision budget exhausted; compile/test the current version, then finish')
        self.sandbox.filesystem.write_text(source, '/bot/src/main.rs')
        if self.remaining() <= 0:
            raise TimeoutError('write completed after deadline; source not accepted')
        if changed:
            self.revisions += 1
            with (self.root / 'revisions.jsonl').open('a') as log:
                log.write(canonical(dict(revision=self.revisions, source_sha256=sha(source.encode()),
                                        remaining_seconds=max(0, self.remaining()))) + '\n')
        self.source = source
        return {'source_sha256': sha(source.encode()), 'revision': self.revisions}

    def compile(self):
        if self.remaining() <= 0:
            raise TimeoutError('authoring time is over')
        seconds = max(1, min(90, math.floor(self.remaining())))
        script = ('import subprocess,json; '
                  'p=subprocess.run(["cargo","build","--release","--offline","--locked","--target","wasm32-wasip1"],'
                  'cwd="/bot",capture_output=True,text=True); '
                  'print(json.dumps({"returncode":p.returncode,"output":(p.stdout+p.stderr)[-12000:]}))')
        process = self.sandbox.exec('python', '-c', script, timeout=seconds)
        stdout = process.stdout.read()
        process.wait()
        if self.remaining() <= 0:
            raise TimeoutError('compile finished after deadline; build not accepted')
        out = json.loads(stdout)
        if out['returncode'] == 0:
            data = self.read('/bot/target/wasm32-wasip1/release/skirmish-bot.wasm', 'rb')
            if len(data) > 8 * 1024 * 1024 or not data.startswith(b'\0asm'):
                raise ValueError('invalid Wasm artifact')
            digest = sha(data)
            build = self.root / 'builds' / digest
            build.mkdir(parents=True, exist_ok=True)
            (build / 'bot.wasm').write_bytes(data)
            (build / 'main.rs').write_text(self.source)
            (build / 'Cargo.lock').write_text(self.read('/bot/Cargo.lock'))
            self.compiled_source = sha(self.source.encode())
            self.artifact = build / 'bot.wasm'
            out.update(wasm_sha256=digest, source_sha256=self.compiled_source)
            self.volume.commit()
        return out

    def campaign(self, maps):
        if type(maps) is not int or not 1 <= maps <= self.budget.max_campaign_maps:
            raise ValueError('maps outside this run budget')
        if self.artifact is None or sha(self.source.encode()) != self.compiled_source:
            raise ValueError('compile the current source first')
        if self.practice_count + maps * 2 > self.budget.max_practice_games:
            raise ValueError('practice game budget exhausted')
        if self.reference_bot is None:
            self.reference_bot = ReferenceBot(self.reference, REFERENCE_SHA)
        rows = []
        for _ in range(maps):
            seed = PRACTICE_START + self.practice_count // 2
            for seat in (1, 2):
                if self.remaining() < 3:
                    return {'games': rows, 'stopped': 'authoring_deadline'}
                ident = f'practice-{self.practice_count:04d}'
                self.practice_count += 1
                trace = self.root / 'practice' / f'{ident}.jsonl'
                from dataclasses import replace
                game_budget = replace(self.budget, game_seconds=min(self.budget.game_seconds, max(1, int(self.remaining()))))
                row = play_game(WasiBot(self.artifact, self.budget.fuel, self.budget.move_seconds),
                                self.reference_bot, seed, seat, game_budget, trace,
                                phase='practice', labels=['candidate', REFERENCE],authoring_deadline=self.deadline)
                row['game_id'] = ident
                self.games[ident] = trace
                write_json(trace.with_suffix('.summary.json'), row)
                rows.append(row)
                self.volume.commit()
        return {'games': rows, 'wins': sum(r['result'] == 'win' for r in rows),
                'losses': sum(r['result'] == 'loss' for r in rows),
                'remaining_seconds': max(0, int(self.remaining()))}

    def budget_status(self):
        return dict(constraint=self.budget.constraint, seconds_remaining=max(0, int(self.remaining())),
                    revisions_used=self.revisions, revisions_remaining=None if self.budget.max_revisions is None
                    else self.budget.max_revisions-self.revisions,
                    practice_games_remaining=self.budget.max_practice_games-self.practice_count,
                    runtime=dict(move_seconds=self.budget.move_seconds,game_seconds=self.budget.game_seconds,
                                 fuel_per_game=self.budget.fuel,memory_mib=256))

    def dispatch(self, name, args):
        if self.remaining() <= 0:
            raise TimeoutError('authoring time is over')
        if self.finished:
            raise ValueError('submission already finished')
        if name == 'budget_status':
            return self.budget_status()
        if name == 'finish':
            if self.artifact is None or sha(self.source.encode()) != self.compiled_source:
                raise ValueError('compile the current source before finishing')
            self.finished = True
            return {'finished': True, 'wasm_sha256': self.artifact.parent.name}
        if name == 'write_source':
            return self.write(args['source'])
        if name == 'read_file':
            path = args['path']
            p = PurePosixPath(path)
            if '..' in p.parts or not (path == '/bot/src/main.rs' or path in ('/reference/README.md', '/reference/rules.json')
                or path.startswith('/reference/engine-rs/src/') and path.endswith('.rs')):
                raise ValueError('only the starter, public rule table and Rust engine reference are readable')
            value = self.read(path)
            return {'content': value[:60000], 'truncated': len(value) > 60000}
        if name == 'compile':
            return self.compile()
        if name == 'campaign':
            return self.campaign(args['maps'])
        if name == 'review':
            ident, offset, limit = args['game_id'], args['offset'], args['limit']
            if ident not in self.games or type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 8:
                raise ValueError('invalid practice game/page')
            rows = []
            with self.games[ident].open() as f:
                for i, line in enumerate(f):
                    if i >= offset + limit:
                        break
                    if i >= offset:
                        rows.append(json.loads(line))
            return {'game_id': ident, 'offset': offset, 'rows': rows, 'next_offset': offset + len(rows)}
        raise ValueError('unknown tool')
