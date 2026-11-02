"""Batched full-game evaluation with independent public decision contexts.

Batching combines requests from separate games. It never changes a game's
action order, substitutes a scripted model move, or hides a failed decision.
"""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time
from .interface import canonical, parse_response, render, ResponseError
from .prompt_search import candidate_messages


def play_games(output, jobs, generator, *, batch_size=8, seconds_limit=3600,
               environment_factory=None, opponent_factory=None, prompt_renderer=None,
               output_parser=None, max_attempts=3, model_action_limit=None):
    """Jobs specify id, seed, board_hash, seat, candidate, strategy, opponent.

    Caller must freeze and verify experimental prerequisites before invoking.
    This layer is also usable with fake environments for coordination tests.
    """
    if batch_size < 1 or seconds_limit <= 0 or max_attempts < 1 or len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('invalid game allocation')
    if any(j['seat'] not in (1, 2) for j in jobs):
        raise ValueError('invalid learner seat')
    if environment_factory is None:
        from .adapter import GPUEnvironment
        environment_factory = lambda: GPUEnvironment(14)
    if opponent_factory is None:
        from .teacher import ScriptedTeacher, WeakBaseline
        opponent_factory = lambda name: WeakBaseline() if name == 'weak' else ScriptedTeacher(name)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'private').mkdir()
    (root/'jobs.json').write_text(json.dumps(jobs, indent=2)+'\n')
    started = time.monotonic()
    active, completed, next_job = [], [], 0

    def finish(game, *, status, reason, reward=0.):
        row = {**game['job'], 'status': status, 'reason': reason, 'reward': reward,
               'decisions': game['decisions'], 'model_decisions': game['model_decisions'],
               'attempts': game['attempts'], 'first_attempt_valid': game['first_attempt_valid'],
               'max_prompt_tokens': game['max_prompt_tokens'],
               'round': game['obs'].visible_state['turn'],
               'seconds': time.monotonic()-game['started']}
        game['trace'].write(canonical({'game_result': row})+'\n')
        game['trace'].close()
        game['env'].close()
        completed.append(row)
        (root/'games.json').write_text(json.dumps(completed, indent=2)+'\n')
        game['finished'] = True
        print(canonical(row), flush=True)

    def execute(game, action, *, policy, diagnostic):
        obs = game['obs']
        transition = game['env'].step(action_id=action.action_id,
                                     expected_state_version=obs.state_version)
        game['decisions'] += 1
        game['trace'].write(canonical(dict(observation=asdict(obs), policy=policy,
            selected_command=dict(action.command), diagnostic=diagnostic,
            after=[asdict(game['env'].observe(p)) for p in (1, 2)],
            outcome=dict(terminated=transition.terminated, truncated=transition.truncated,
                         rewards=transition.rewards, reason=transition.reason)))+'\n')
        if transition.terminated or transition.truncated:
            game['obs'] = transition.actor_observation
            finish(game, status='native' if transition.terminated else 'truncated',
                   reason=transition.reason, reward=transition.rewards[game['job']['seat']])
        elif game['decisions'] >= 10000:
            finish(game, status='truncated', reason='research_action_limit')
        else:
            game['obs'] = game['env'].observe(transition.next_player)
        game['pending'] = None

    try:
        while next_job < len(jobs) or active:
            if time.monotonic()-started >= seconds_limit:
                for game in active:
                    finish(game, status='budget_incomplete', reason='occupied_time_ceiling')
                break
            while len(active) < batch_size and next_job < len(jobs):
                job = jobs[next_job]
                next_job += 1
                # Job IDs become filenames, so reject paths even in trusted manifests.
                if not job['id'] or Path(job['id']).name != job['id'] or job['id'] in ('.', '..'):
                    raise ValueError('unsafe game id')
                env = environment_factory()
                obs = env.reset(seed=job['seed'], settings={})
                (root/'private'/f"{job['id']}.snapshot").write_bytes(env.snapshot())
                active.append(dict(job=job, env=env, obs=obs, opponent=opponent_factory(job['opponent']),
                    trace=gzip.open(root/f"{job['id']}.jsonl.gz", 'xt', compresslevel=1),
                    started=time.monotonic(), decisions=0, model_decisions=0, attempts=0,
                    first_attempt_valid=0, max_prompt_tokens=0, pending=None, finished=False))
            requests = []
            for game in active:
                while not game['finished'] and game['obs'].player != game['job']['seat']:
                    if time.monotonic()-started >= seconds_limit:
                        break
                    choice = game['opponent'].select(game['obs'])
                    execute(game, choice.action, policy='scripted_opponent', diagnostic=asdict(choice))
                if game['finished'] or time.monotonic()-started >= seconds_limit:
                    continue
                if model_action_limit is not None and game['model_decisions'] >= model_action_limit:
                    finish(game, status='budget_incomplete', reason='model_decision_ceiling')
                    continue
                if game['pending'] is None:
                    game['model_decisions'] += 1
                    prompt = (prompt_renderer or render)(game['obs'])
                    game['pending'] = dict(prompt=prompt, attempts=[],
                        messages=prompt.messages if prompt_renderer else candidate_messages(game['obs'], game['job']['strategy']))
                requests.append(game)
            if requests:
                batch_start = time.monotonic()
                outputs = generator.batch([g['pending']['messages'] for g in requests])
                batch_seconds = time.monotonic()-batch_start
                if len(outputs) != len(requests):
                    raise ValueError('generation batch returned the wrong number of responses')
                for game, generation in zip(requests, outputs):
                    pending = game['pending']
                    game['attempts'] += 1
                    game['max_prompt_tokens'] = max(game['max_prompt_tokens'], len(generation.prompt_token_ids))
                    entry = dict(messages=pending['messages'], generation=asdict(generation),
                                 batch_size=len(requests), batch_seconds=batch_seconds)
                    try:
                        action = (output_parser or parse_response)(generation.text, pending['prompt'],
                                                current_version=game['obs'].state_version)
                    except ResponseError as exc:
                        entry['error'] = str(exc)
                        pending['attempts'].append(entry)
                        if str(exc) in ('context_overflow', 'unfinished_thinking'):
                            game['trace'].write(canonical(dict(observation=asdict(game['obs']),
                                policy='model', attempts=pending['attempts'], budget_failure=True))+'\n')
                            finish(game, status='budget_incomplete', reason=str(exc))
                        elif len(pending['attempts']) == max_attempts:
                            game['trace'].write(canonical(dict(observation=asdict(game['obs']),
                                policy='model', attempts=pending['attempts'], forfeit=True))+'\n')
                            finish(game, status='forfeit', reason='invalid_response_after_two_repairs' if max_attempts==3
                                   else 'invalid_response_single_call', reward=-1.)
                        else:
                            pending['messages'] = pending['messages']+[
                                dict(role='assistant', content=generation.text),
                                dict(role='user', content=f'Invalid response: {exc}. Return one legal action_id JSON object.')]
                        continue
                    game['first_attempt_valid'] += not pending['attempts']
                    pending['attempts'].append(entry)
                    execute(game, action, policy='model', diagnostic=pending['attempts'])
                progress = dict(elapsed_seconds=time.monotonic()-started, finished=len(completed),
                    active=[dict(id=g['job']['id'], round=g['obs'].visible_state['turn'],
                                 model_decisions=g['model_decisions'], decisions=g['decisions'])
                            for g in active if not g['finished']])
                temporary = root/'progress.tmp'
                temporary.write_text(json.dumps(progress))
                temporary.replace(root/'progress.json')
                if sum(g['model_decisions'] for g in requests) % (8*len(requests)) == 0:
                    print(canonical(dict(event='GAME_PROGRESS',**progress)),flush=True)
            active = [g for g in active if not g['finished']]
    except Exception as exc:
        import traceback
        (root/'failure-traceback.txt').write_text(traceback.format_exc())
        for game in active:
            if not game['finished']:
                finish(game, status='infrastructure_error', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for game in active:
            if not game['trace'].closed:
                game['trace'].close()
                game['env'].close()
    report = dict(schema='gameplay-v1', allocated_games=len(jobs), finished_games=len(completed),
        unstarted_jobs=[j['id'] for j in jobs[next_job:]], games=completed,
        elapsed_seconds=time.monotonic()-started,
        complete=len(completed)==len(jobs) and all(g['status'] in ('native','truncated','forfeit') for g in completed))
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
