"""Continue unfinished async games from their last committed checkpoint.

Mirrors ``async_gameplay.play_async_games`` line for line, except that each game
starts from a restored engine snapshot and its saved counters instead of
``env.reset``. Decision numbers, and therefore per-decision sampling seeds,
continue exactly where the interrupted worker stopped. The original runner is
left untouched so the hashed source receipts of the interrupted run stay valid.
"""
import asyncio
import base64
from dataclasses import asdict
import gzip
from pathlib import Path
import time
import traceback

from .async_gameplay import atomic_json
from .decision_harness import render_decision
from .interface import canonical
from .interface import ResponseError
from .thinking_gameplay import parse_thinking_action


async def resume_async_games(output, entries, generator, *, thinking,
                             environment_factory=None, opponent_factory=None,
                             prompt_renderer=render_decision):
    """``entries``: [{job, counters, snapshot (base64)}] from original checkpoints."""
    jobs = [e['job'] for e in entries]
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('duplicate game IDs')
    if any(j['seat'] not in (1, 2) or not j['id'] or Path(j['id']).name != j['id']
           or j['id'] in ('.', '..') for j in jobs):
        raise ValueError('invalid job')
    if environment_factory is None:
        from .adapter import GPUEnvironment
        environment_factory = lambda: GPUEnvironment(14)
    if opponent_factory is None:
        from .teacher import ScriptedTeacher
        opponent_factory = ScriptedTeacher
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'private').mkdir()
    atomic_json(root/'jobs.json', jobs)
    started = time.monotonic()
    completed, progress = [], {}

    def persist_progress():
        atomic_json(root/'progress.json', dict(elapsed_seconds=time.monotonic()-started,
            finished=len(completed), allocated=len(jobs), active=list(progress.values()),
            generating=list(generator.pending.values())))

    async def monitor():
        while True:
            persist_progress()
            await asyncio.sleep(15)

    async def game(entry):
        job = entry['job']
        env = environment_factory()
        opponent = opponent_factory(job['opponent'])
        opened = time.monotonic()
        counters = dict(entry['counters'])
        obs = None
        trace = gzip.open(root/(job['id']+'.jsonl.gz'), 'xt', compresslevel=1)

        def write(row):
            trace.write(canonical(row)+'\n')
            trace.flush()

        def checkpoint():
            atomic_json(root/'private'/(job['id']+'.checkpoint.json'), dict(job=job,
                counters=counters, snapshot=base64.b64encode(env.snapshot()).decode()))
            progress[job['id']] = dict(id=job['id'], round=obs.visible_state['turn'], **counters)

        def finish(status, reason, reward=0.):
            # ``seconds`` covers only this continuation segment.
            row = dict(**job, **counters, status=status, reason=reason, reward=reward,
                       round=None if obs is None else obs.visible_state['turn'],
                       seconds=time.monotonic()-opened)
            write(dict(game_result=row))
            completed.append(row)
            progress.pop(job['id'], None)
            atomic_json(root/'games.json', completed)
            persist_progress()
            print(canonical(dict(event='GAME_RESULT', **row)), flush=True)
            return row

        try:
            obs = env.restore(base64.b64decode(entry['snapshot'], validate=True))
            checkpoint()
            while True:
                await asyncio.sleep(0)
                if obs.player == job['seat']:
                    counters['model_decisions'] += 1
                    counters['attempts'] += 1
                    prompt = prompt_renderer(obs)
                    generated, timing = await generator.generate(prompt.messages,
                        job=job, decision=counters['model_decisions'])
                    counters['max_prompt_tokens'] = max(counters['max_prompt_tokens'],
                                                        len(generated.prompt_token_ids))
                    entry_row = dict(messages=prompt.messages, generation=asdict(generated), timing=timing)
                    if timing['finish_reason'] == 'repetition_abort':
                        entry_row['error'] = timing['stop_reason']
                        write(dict(observation=asdict(obs), policy='model', attempts=[entry_row],
                                   generation_failure=True))
                        return finish('generation_incomplete', timing['stop_reason'])
                    if timing['finish_reason'] == 'length':
                        entry_row['error'] = 'native_context_exhausted'
                        write(dict(observation=asdict(obs), policy='model', attempts=[entry_row],
                                   context_failure=True))
                        return finish('context_incomplete', 'native_context_exhausted')
                    try:
                        action = parse_thinking_action(generated.text, prompt,
                            current_version=obs.state_version, thinking=thinking)
                    except ResponseError as exc:
                        entry_row['error'] = str(exc)
                        write(dict(observation=asdict(obs), policy='model', attempts=[entry_row], forfeit=True))
                        return finish('forfeit', 'invalid_response_single_call', -1.)
                    counters['first_attempt_valid'] += 1
                    policy, diagnostic = 'model', [entry_row]
                else:
                    choice = opponent.select(obs)
                    action, policy, diagnostic = choice.action, 'scripted_opponent', asdict(choice)
                transition = env.step(action_id=action.action_id, expected_state_version=obs.state_version)
                counters['decisions'] += 1
                write(dict(observation=asdict(obs), policy=policy,
                    selected_command=dict(action.command), diagnostic=diagnostic,
                    after=[asdict(env.observe(p)) for p in (1, 2)],
                    outcome=dict(terminated=transition.terminated, truncated=transition.truncated,
                                 rewards=transition.rewards, reason=transition.reason)))
                if transition.terminated or transition.truncated:
                    obs = transition.actor_observation
                    checkpoint()
                    return finish('native' if transition.terminated else 'truncated',
                                  transition.reason, transition.rewards[job['seat']])
                obs = env.observe(transition.next_player)
                checkpoint()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            (root/(job['id']+'.failure.txt')).write_text(traceback.format_exc())
            return finish('infrastructure_error', f'{type(exc).__name__}: {exc}')
        finally:
            trace.close()
            env.close()

    watcher = asyncio.create_task(monitor())
    try:
        await asyncio.gather(*(game(e) for e in entries))
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        persist_progress()
    report = dict(schema='async-gameplay-resume-v1', allocated_games=len(jobs), finished_games=len(completed),
                  games=completed, elapsed_seconds=time.monotonic()-started,
                  complete=len(completed)==len(jobs) and all(g['status'] in ('native','forfeit','truncated') for g in completed))
    atomic_json(root/'report.json', report)
    return report
