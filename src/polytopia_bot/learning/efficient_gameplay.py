"""Overlap independent CPU game lanes with one persistent inference process.

Each lane uses the frozen gameplay implementation and preserves its own batch
boundaries and job order. Queues carry only public model messages/responses;
private engine state remains inside the owning CPU process.
"""
import json
import multiprocessing
from pathlib import Path
import queue
import time


class QueueGenerator:
    def __init__(self, lane, events, responses):
        self.lane, self.events, self.responses = lane, events, responses
        self.sequence = 0

    def batch(self, messages):
        request = self.sequence
        self.sequence += 1
        self.events.put(('batch', self.lane, request, messages, time.monotonic()))
        reply = self.responses.get()
        if reply[0] == 'error':
            raise RuntimeError('inference server failed: ' + reply[1])
        _, returned_request, generations = reply
        if returned_request != request or len(generations) != len(messages):
            raise ValueError('response routed to wrong request or lane')
        return generations


def _lane(lane, output, jobs, batch_size, seconds_limit, events, responses,
          environment_factory, opponent_factory):
    from .gameplay import play_games
    try:
        report = play_games(output, jobs, QueueGenerator(lane, events, responses),
            batch_size=batch_size, seconds_limit=seconds_limit,
            environment_factory=environment_factory, opponent_factory=opponent_factory)
        events.put(('done', lane, report))
    except BaseException as error:
        events.put(('failed', lane, type(error).__name__ + ': ' + str(error)))
        raise


def play_pipelined(output, lane_jobs, generator, *, batch_size=8, seconds_limit=3600,
                   environment_factory=None, opponent_factory=None):
    if not 1 <= len(lane_jobs) <= 8 or not all(lane_jobs):
        raise ValueError('one to eight nonempty CPU lanes required')
    jobs = [job for lane in lane_jobs for job in lane]
    if len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('job IDs must be unique across lanes')
    if batch_size < 1 or seconds_limit <= 0:
        raise ValueError('positive batch size and time ceiling required')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context('spawn')
    events = context.Queue(maxsize=2 * len(lane_jobs))
    responses = [context.Queue(maxsize=1) for _ in lane_jobs]
    workers = []
    reports = {}
    started = time.monotonic()
    batches = 0
    inference_seconds = 0.
    queue_wait_seconds = 0.
    try:
        for lane, allocated in enumerate(lane_jobs):
            output_path = root / f'shard-{lane}' / 'games'
            process = context.Process(target=_lane, args=(lane, output_path, allocated,
                batch_size, seconds_limit, events, responses[lane],
                environment_factory, opponent_factory))
            process.start()
            workers.append(process)
        with (root / 'dispatch.jsonl').open('x') as dispatch:
            while len(reports) < len(workers):
                if time.monotonic() - started > seconds_limit + 60:
                    raise TimeoutError('bounded pipeline time ceiling reached')
                try:
                    event = events.get(timeout=.1)
                except queue.Empty:
                    if any(p.exitcode not in (None, 0) for p in workers):
                        raise RuntimeError('CPU lane exited without a complete report')
                    continue
                kind, lane, *payload = event
                if kind == 'failed':
                    raise RuntimeError(f'CPU lane {lane} failed: {payload[0]}')
                if kind == 'done':
                    if lane in reports:
                        raise ValueError('duplicate lane report')
                    reports[lane] = payload[0]
                    continue
                request, messages, queued_at = payload
                before = time.monotonic()
                queue_wait_seconds += before - queued_at
                generations = generator.batch(messages)
                seconds = time.monotonic() - before
                inference_seconds += seconds
                batches += 1
                responses[lane].put(('result', request, generations))
                dispatch.write(json.dumps(dict(lane=lane, request=request,
                    size=len(messages), queue_wait_seconds=before-queued_at,
                    model_seconds=seconds)) + '\n')
        for worker in workers:
            worker.join(timeout=10)
            if worker.exitcode != 0:
                raise RuntimeError('CPU lane did not exit successfully')
        ordered = [reports[i] for i in range(len(lane_jobs))]
        report = dict(schema='pipelined-gameplay-v1', allocated_games=len(jobs),
            finished_games=sum(r['finished_games'] for r in ordered),
            games=[g for r in ordered for g in r['games']],
            unstarted_jobs=[j for r in ordered for j in r['unstarted_jobs']],
            complete=all(r['complete'] for r in ordered),
            elapsed_seconds=time.monotonic()-started,
            pipeline=dict(cpu_lanes=len(lane_jobs), batch_size=batch_size,
                batches=batches, model_seconds=inference_seconds,
                queue_wait_seconds=queue_wait_seconds,
                note='Lane batch timings include queue wait. Parent dispatch records actual model-call time once; do not sum lane batch times as GPU usage.'))
        (root / 'report.json').write_text(json.dumps(report, indent=2))
        return report
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
        for q in responses + [events]:
            q.close()
