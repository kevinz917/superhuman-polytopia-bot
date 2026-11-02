"""Record a structured-RL expert's decisions as readable-harness SFT examples.

The expert plays through the public GPUEnvironment interface. Every expert
decision is rendered with the same prompt the language model sees at evaluation
(``render_readable_decision``); the label is the prompt-local ``aN`` of the
expert's chosen legal action, verified by a parse round trip.
"""
import gzip
import hashlib
import json
from pathlib import Path
import time

from .interface import parse_response
from .thinking_readable import render_readable_decision

FAMILIES = {1: 'move', 2: 'attack', 3: 'ability', 4: 'recruit', 5: 'harvest', 6: 'build', 7: 'research',
            8: 'capture', 9: 'city_reward', 10: 'end_turn', 11: 'resign'}


def response_text(label):
    return json.dumps({'action_id': label}, separators=(',', ':'))


def play_expert_game(job, expert_checkpoint, output, *, environment_factory=None, opponent_factory=None,
                     policy_factory=None):
    """Play one game; write ``<id>.jsonl.gz`` (header, examples, result) and return the result row."""
    if environment_factory is None:
        from .adapter import GPUEnvironment
        environment_factory = lambda: GPUEnvironment(14)
    if opponent_factory is None:
        from .teacher import ScriptedTeacher
        opponent_factory = ScriptedTeacher
    if policy_factory is None:
        from polytopia_bot.rl.policy import StructuredPolicy
        policy_factory = lambda path: StructuredPolicy(path, deterministic=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env, opponent, expert = environment_factory(), opponent_factory(job['opponent']), policy_factory(expert_checkpoint)
    started = time.monotonic()
    path = output/(job['id']+'.jsonl.gz')
    system = None
    counts = dict(decisions=0, expert_decisions=0)
    with gzip.open(path.with_suffix('.tmp'), 'wt', compresslevel=6) as f:
        try:
            obs = env.reset(seed=job['seed'], settings={})
            while True:
                if obs.player == job['seat']:
                    prompt = render_readable_decision(obs)
                    chosen = expert.select(obs).action
                    labels = [k for k, a in prompt.catalog.items() if a.action_id == chosen.action_id]
                    if len(labels) != 1:
                        raise ValueError('expert action is not exactly one prompt action')
                    label = labels[0]
                    if parse_response(response_text(label), prompt, current_version=obs.state_version).action_id != chosen.action_id:
                        raise ValueError('label round trip failed')
                    if system is None:
                        system = prompt.messages[0]['content']
                        f.write(json.dumps(dict(header=dict(job=job, system=system,
                            system_sha256=hashlib.sha256(system.encode()).hexdigest())))+'\n')
                    elif prompt.messages[0]['content'] != system:
                        raise ValueError('system prompt changed within a game')
                    counts['expert_decisions'] += 1
                    f.write(json.dumps(dict(decision=counts['expert_decisions'], turn=obs.visible_state['turn'],
                        family=FAMILIES.get(chosen.command['kind'], str(chosen.command['kind'])),
                        legal=len(prompt.catalog), user=prompt.messages[1]['content'], label=label))+'\n')
                    action = chosen
                else:
                    action = opponent.select(obs).action
                transition = env.step(action_id=action.action_id, expected_state_version=obs.state_version)
                counts['decisions'] += 1
                if transition.terminated or transition.truncated:
                    final = transition.actor_observation
                    seat_view = env.observe(job['seat']).visible_state
                    result = dict(**job, **counts, status='native' if transition.terminated else 'truncated',
                                  reason=transition.reason, reward=transition.rewards[job['seat']],
                                  round=final.visible_state['turn'],
                                  final_cities=sum(c['owner'] == job['seat'] for c in seat_view['cities']),
                                  seconds=time.monotonic()-started)
                    break
                obs = env.observe(transition.next_player)
        finally:
            env.close()
        f.write(json.dumps(dict(result=result))+'\n')
    path.with_suffix('.tmp').replace(path)
    return result


def read_game(path):
    header, examples, result = None, [], None
    with gzip.open(path, 'rt') as f:
        for line in f:
            row = json.loads(line)
            if 'header' in row:
                header = row['header']
            elif 'result' in row:
                result = row['result']
            else:
                examples.append(row)
    if header is None or result is None:
        raise ValueError(f'incomplete expert game file {path}')
    return header, examples, result
