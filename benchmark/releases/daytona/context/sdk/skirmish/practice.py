"""Public, on-demand practice. No champion weights or held-out maps are installed."""
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from .authoring_tools import PUBLIC_MAPS, compile_bot
from .continuous import EVALUATION
from .common import sha
from .local_referee import play_game_streamed, audit_trace_streamed
from .wasi import WasiBot


class PublicPracticeBot:
    """A documented greedy development baseline, independent of the hidden panel."""
    def close(self):
        pass

    def select(self, obs):
        v = obs.visible_state
        p, size = obs.player, v['board_size']
        units = {u['tile']: u for u in v['units']}
        targets = [c['tile'] for c in v['cities'] if c['owner'] != p]
        targets += [t['tile'] for t in v['tiles'] if not t['known']]
        targets = targets or [size*size//2]
        def distance(tile):
            return min(max(abs(tile%size-t%size), abs(tile//size-t//size)) for t in targets)
        def score(action):
            c = action.command
            k, src, dst, param = (c[x] for x in ('kind','source','target','parameter'))
            if k == 11: return -10000
            if k == 10: return -1000
            if k == 9: return 1000
            if k == 8: return 900
            if k == 2: return 300-units.get(dst, {}).get('health', 10)
            if k == 5: return 150
            if k == 4: return 100 + (20 if param in (2,3,4) else 0) - len(units)
            if k == 6: return 80
            if k == 7: return 50 + (15 if param in (1,2,3,4) else 0)
            if k == 1: return 30 + 15*(distance(src)-distance(dst))
            if k == 3: return -10000 if param == 8 else 10
            return 0
        return max(obs.legal_actions, key=lambda a: (score(a), a.action_id)).action_id


def practice_match(map_id='drylands-0', seat=1):
    if map_id not in PUBLIC_MAPS or seat not in (1, 2):
        raise ValueError('choose a public map and seat 1 or 2')
    compiled = compile_bot()
    root = Path('/bot/practice')/uuid.uuid4().hex
    root.mkdir(parents=True)
    family, seed = PUBLIC_MAPS[map_id]
    # The trusted clock is readable but not writable by the coding user.
    clock = Path('/verify/stage-start.json')
    remaining = json.loads(clock.read_text())['deadline_unix']-time.time() if clock.exists() else 1800
    if remaining <= 0:
        raise ValueError('coding budget exhausted')
    row = play_game_streamed(WasiBot(compiled['wasm'], EVALUATION['fuel'], 2.), PublicPracticeBot(),
        seed, seat, SimpleNamespace(game_seconds=min(1800, remaining)), root/'game.jsonl',
        phase='public_practice', labels=['your-bot','public-greedy-v1'], map_type=family,
        cap_policy='engine_draw', evaluation=EVALUATION,
        authoring_deadline=time.monotonic()+remaining)
    audited = audit_trace_streamed(row, root/'game.jsonl')
    report = {k: row[k] for k in ('status','reason','result','winner','actions','turns','seconds','seat')}
    report.update(map_id=map_id, opponent='public-greedy-v1', source_sha256=sha(Path('/bot/src/main.rs').read_bytes()),
                  wasm_sha256=compiled['sha256'], trace_path=str(root/'game.jsonl'), audited_actions=audited)
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def inspect_trace(path, offset=0, limit=1):
    if offset < 0 or not 1 <= limit <= 20:
        raise ValueError('offset must be nonnegative; limit must be 1..20')
    result = []
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            if index >= offset:
                result.append(json.loads(line))
            if len(result) == limit:
                break
    return result
