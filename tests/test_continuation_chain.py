"""Two interruptions, three segments: the stitched final run must equal an uninterrupted
run and pass the unchanged merge/summary. Real engine and opponent; run on Modal."""
import asyncio
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path[:0] = ['/workspace/scripts']
from continue_thinking_games import finalize, prepare, segment_dir  # noqa: E402
from polytopia_bot.learning.async_gameplay import atomic_json, play_async_games  # noqa: E402
from polytopia_bot.learning.async_resume import resume_async_games  # noqa: E402
from polytopia_bot.learning.interface import Generation  # noqa: E402
from polytopia_bot.learning.thinking_readable import render_readable_decision  # noqa: E402

JOBS = [dict(id=f'seed-{s}-seat-{t}', seed=s, board_hash=f'b{s}', seat=t, candidate='fake', strategy='',
             opponent='balanced', rng_index=i) for i, (s, t) in enumerate([(100691, 1), (100691, 2), (100694, 1), (100694, 2)])]
CUT0 = {'seed-100691-seat-1': 5, 'seed-100694-seat-1': 12, 'seed-100694-seat-2': 7}   # interrupted in segment 0
CUT1 = {'seed-100691-seat-1': 20, 'seed-100694-seat-2': 15}                          # interrupted again in segment 1
SEED = 17


class FakeModel:
    """Deterministic in (job, decision); blocks forever at a cut to mimic a killed worker."""
    def __init__(self, folder, thinking, cuts=None):
        self.pending, self.thinking, self.cuts = {}, thinking, cuts or {}
        self.blocked = set()
        folder.mkdir(parents=True, exist_ok=True)
        self.log = (folder/'generation-timing.jsonl').open('x')

    async def generate(self, messages, *, job, decision):
        if self.cuts.get(job['id']) == decision:
            self.blocked.add(job['id'])
            await asyncio.Event().wait()
        body = json.loads(messages[1]['content'])['action_names']
        names = sorted((a for a in body if body[a] != 'resign'), key=lambda a: int(a[1:]))
        pick = names[int(hashlib.sha256(f"{job['id']}:{decision}".encode()).hexdigest(), 16) % len(names)]
        text = ('checking</think>' if self.thinking else '') + '{"action_id":"%s"}' % pick
        timing = dict(id=job['id'], decision=decision, input_tokens=3, output_tokens=2, request_seconds=0.01,
                      first_token_seconds=0.0, finish_reason='stop', stop_reason=None, thinking_closed=self.thinking,
                      max_tokens=None, thinking_token_budget=None, seed=SEED+job['rng_index']*100000+decision)
        self.log.write(json.dumps(timing)+'\n'); self.log.flush()
        await asyncio.sleep(0)
        return Generation(text, [1, 2, 3], [7, 8]), timing


def runtime(thinking):
    return dict(model_id='fake', revision='r', adapter=None, thinking=thinking, max_tokens=None,
                thinking_token_budget=None, max_model_len=262144, device='cpu', tokenizer_hash='t',
                chat_template_hash='c', packages={}, serving={}, constrain_actions=True, harness='h')


async def run_segment(coro_factory, model, expected):
    """Run until every game finished or blocked; cancel to mimic a timeout. True if it completed."""
    task = asyncio.create_task(coro_factory())
    while not task.done():
        await asyncio.sleep(0.5)
        if model.blocked and len(model.blocked) == expected:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return False
    task.result()
    return True


def strip(line):
    row = json.loads(line)
    if 'game_result' in row:
        row['game_result'] = {k: v for k, v in row['game_result'].items()
                              if k not in ('seconds', 'continued_from_checkpoint', 'final_segment',
                                           'final_segment_seconds', 'resumed_at')}
    return row


def read(path):
    with gzip.open(path, 'rt') as f:
        return [strip(l) for l in f]


def test_two_interruptions_stitch_exactly():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); root = tmp/'run'
        maps = sorted({j['seed'] for j in JOBS})
        protocol = dict(jobs=JOBS, replicas_per_mode=2, revision='r', seed=SEED,
                        pairing=dict(games_per_mode=4, maps=2))
        root.mkdir(); (root/'protocol.json').write_text(json.dumps(protocol))
        shard_jobs = {s: [j for j in JOBS if j['seed'] in set(maps[s::2])] for s in (0, 1)}
        for s, jobs in shard_jobs.items():
            base = root/'shards'/str(s); base.mkdir(parents=True)
            (base/'protocol.json').write_text(json.dumps(dict(protocol, jobs=jobs, shard=s)))
            # Thinking-off arm: complete, with the receipts a normal worker leaves.
            off = FakeModel(base/'off', False)
            report = asyncio.run(play_async_games(base/'off'/'games', jobs, off, thinking=False,
                                                  prompt_renderer=render_readable_decision))
            atomic_json(base/'off'/'report.json', report); atomic_json(base/'off'/'runtime.json', runtime(False))
            atomic_json(base/'off-execution.json', dict(exit_code=0, worker_seconds=1.0))
            # Thinking-on segment 0: killed while some games wait on the model.
            on = FakeModel(base/'on', True, CUT0)
            atomic_json(base/'on'/'runtime.json', runtime(True))
            expected = sum(j['id'] in CUT0 for j in jobs)
            assert not asyncio.run(run_segment(lambda: play_async_games(base/'on'/'games', jobs, on, thinking=True,
                                   prompt_renderer=render_readable_decision), on, expected))
        assumed = {'0:0': 10.0, '0:1': 10.0}
        for k, cuts in ((1, CUT1), (2, {})):
            prepare(root, {}, segment=k)
            for s in (0, 1):
                folder = segment_dir(root, k)/'shards'/str(s)
                entries = json.loads((folder/'entries.json').read_text())
                if not entries:
                    continue
                model = FakeModel(folder/'on', True, cuts)
                atomic_json(folder/'on'/'runtime.json', runtime(True))
                expected = sum(e['job']['id'] in cuts for e in entries)
                done = asyncio.run(run_segment(lambda: resume_async_games(folder/'on'/'games', entries, model,
                                   thinking=True, prompt_renderer=render_readable_decision), model, expected))
                if done:
                    atomic_json(folder/'on'/'report.json', json.loads((folder/'on'/'games'/'report.json').read_text()))
                    atomic_json(folder/'on-execution.json', dict(exit_code=0, worker_seconds=2.0))
                else:
                    assumed[f'{k}:{s}'] = 20.0
        segments = finalize(root, assumed)
        # Reference: the same thinking-on games played without interruption.
        ref = tmp/'ref'
        for s, jobs in shard_jobs.items():
            model = FakeModel(ref/str(s), True)
            asyncio.run(play_async_games(ref/str(s)/'games', jobs, model, thinking=True,
                                         prompt_renderer=render_readable_decision))
        for s, jobs in shard_jobs.items():
            for j in jobs:
                a = read(root/'final'/'on'/'games'/(j['id']+'.jsonl.gz'))
                b = read(ref/str(s)/'games'/(j['id']+'.jsonl.gz'))
                assert a == b, j['id']
        summary = json.loads((root/'final'/'summary.json').read_text())
        assert summary['modes']['on']['allocated'] == 4 and summary['modes']['off']['allocated'] == 4
        report = json.loads((root/'final'/'on'/'report.json').read_text())
        return dict(segments=segments, statuses={g['id']: (g['status'], g['round'], g['model_decisions']) for g in report['games']},
                    gpu_worker_seconds=report['gpu_worker_seconds'], on_calls=summary['modes']['on']['calls'])


if __name__ == '__main__':
    print(json.dumps(test_two_interruptions_stitch_exactly()))
