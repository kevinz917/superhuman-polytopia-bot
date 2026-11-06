"""Resumed games must reproduce an uninterrupted game exactly (real engine; run on Modal)."""
import asyncio
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from polytopia_bot.learning.async_gameplay import play_async_games
from polytopia_bot.learning.async_resume import resume_async_games
from polytopia_bot.learning.interface import Generation
from polytopia_bot.learning.thinking_readable import render_readable_decision

CUT = {'resume-a': 9, 'resume-b': 23}


class FakeModel:
    """Deterministic in (job, decision) only, like seeded sampling."""
    def __init__(self, copy_dir=None):
        self.pending, self.copy_dir = {}, copy_dir
        self.root = None

    async def generate(self, messages, *, job, decision):
        if self.copy_dir and decision == CUT.get(job['id']):
            # The checkpoint on disk now reflects the state just before this decision.
            src = self.root/'private'/(job['id']+'.checkpoint.json')
            shutil.copy(src, self.copy_dir/(job['id']+'.checkpoint.json'))
        names = sorted(json.loads(messages[1]['content'])['action_names'], key=lambda a: int(a[1:]))
        names = [a for a in names if json.loads(messages[1]['content'])['action_names'][a] != 'resign']
        pick = names[int(hashlib.sha256(f"{job['id']}:{decision}".encode()).hexdigest(), 16) % len(names)]
        text = 'checking</think>{"action_id":"%s"}' % pick
        await asyncio.sleep(0)
        return Generation(text, [1, 2, 3], [7, 8]), dict(finish_reason='stop', stop_reason=None)


def rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(l) for l in f]


def strip(row):
    if 'game_result' in row:
        row = dict(row, game_result={k: v for k, v in row['game_result'].items() if k != 'seconds'})
    return row


def test_resume_reproduces_uninterrupted_games():
    jobs = [dict(id='resume-a', seed=100691, seat=1, opponent='balanced', rng_index=0, board_hash='x'),
            dict(id='resume-b', seed=100694, seat=2, opponent='balanced', rng_index=7, board_hash='x')]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); cuts = tmp/'cuts'; cuts.mkdir()
        straight = FakeModel(cuts); straight.root = tmp/'straight'
        full = asyncio.run(play_async_games(tmp/'straight', jobs, straight, thinking=True,
                                            prompt_renderer=render_readable_decision))
        entries = []
        for job in jobs:
            cp = json.loads((cuts/(job['id']+'.checkpoint.json')).read_text())
            assert cp['counters']['model_decisions'] == CUT[job['id']] - 1
            entries.append(dict(job=job, counters=cp['counters'], snapshot=cp['snapshot']))
        resumed = asyncio.run(resume_async_games(tmp/'resumed', entries, FakeModel(), thinking=True,
                                                 prompt_renderer=render_readable_decision))
        by_id = {g['id']: g for g in full['games']}
        for g in resumed['games']:
            ref = by_id[g['id']]
            assert {k: v for k, v in g.items() if k != 'seconds'} == {k: v for k, v in ref.items() if k != 'seconds'}, (g, ref)
        for e in entries:
            jid = e['job']['id']; n = e['counters']['decisions']
            a = rows(tmp/'straight'/(jid+'.jsonl.gz')); b = rows(tmp/'resumed'/(jid+'.jsonl.gz'))
            assert len(a) > n + 5, (jid, len(a), n)
            assert [strip(r) for r in a[:n]+b] == [strip(r) for r in a], jid
        return dict(games={g['id']: (g['status'], g['round'], g['model_decisions']) for g in full['games']},
                    cuts={e['job']['id']: e['counters'] for e in entries})


if __name__ == '__main__':
    print(json.dumps(test_resume_reproduces_uninterrupted_games()))
