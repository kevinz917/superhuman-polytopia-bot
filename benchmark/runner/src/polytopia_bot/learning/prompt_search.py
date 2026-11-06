"""Bounded public development traces and predeclared prompt-search artifacts."""
from collections import defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from .interface import canonical, digest, render

# No instance IDs, action answers, extra tools or mechanics edits in candidates.
STRATEGIES = {
    'original': '',
    'capture': 'Prioritize winning by capturing enemy cities while protecting your own. Capture an available settlement when useful. Move combat-capable units toward known enemy cities or neutral settlements. When none is known, explore unknown borders. Avoid aimless movement and premature end turns. Spend on economy and recruitment when they support expansion or survival.',
    'economy': 'Build a sustainable expansion economy. Inspect your stars, city population progress and visible resources before spending. Compare legal harvest and building opportunities, useful research, and needed recruits. Prefer investments with near-term use. Resolve city rewards deliberately. Keep units exploring and capturing; income without progress toward enemy cities is insufficient to win.',
    'frontier': 'For each mobile unit, identify a visible settlement, enemy threat, or unexplored frontier as an objective. Use tile coordinates (tile modulo board_size, tile divided by board_size) to compare locations. Choose a legal move that makes useful progress. Consult recent own actions to avoid pointless backtracking. Never invent terrain or enemy positions in unknown tiles.',
    'combat': 'Inspect visible enemy units and threatened friendly cities before committing resources. Prefer useful attacks, recovering endangered units, reinforcement and captures over low-value spending. Use observed health, unit definitions and terrain to assess trades without assuming hidden enemy actions. Keep pressure toward enemy cities; do not heal or recruit indefinitely while neglecting capture opportunities.',
    'turn_plan': 'Before choosing one action, identify your most urgent objective for this turn: resolve a reward, preserve a threatened city, capture a settlement, improve usable income, or explore and advance. Check whether the chosen action supports that objective and leaves resources for immediate necessities. Execute only one legal action. End the turn when remaining choices offer no useful progress.',
    'catalog_first': 'Read the legal_actions columns before comparing rows. Kind 9 resolves a city reward, kind 8 captures a settlement, kind 2 attacks, kind 1 moves, and kind 10 ends the turn. Inspect the public state to decide which available action advances expansion, defense or income. Select the corresponding row and copy only its action_id into the required JSON. The first row is not automatically the best choice.',
}


def candidate_messages(observation, strategy):
    prompt = render(observation)
    messages = [dict(m) for m in prompt.messages]
    if strategy:
        messages[0]['content'] += '\nStrategic guidance:\n' + strategy
    return messages


def _development_game(directory, seed, seat):
    from .adapter import GPUEnvironment
    from .teacher import ScriptedTeacher, WeakBaseline
    root = Path(directory)
    teacher, opponent = ScriptedTeacher(), WeakBaseline()
    env = GPUEnvironment(14)
    obs = env.reset(seed=seed, settings={})
    result = None
    try:
        with gzip.open(root/f'seed-{seed}-seat-{seat}.jsonl.gz','xt',compresslevel=1) as trace:
            for index in range(10000):
                policy=teacher if obs.player==seat else opponent
                selected=policy.select(obs)
                transition=env.step(action_id=selected.action.action_id,expected_state_version=obs.state_version)
                trace.write(canonical(dict(observation=asdict(obs),selected_command=dict(selected.action.command),
                    policy_hash=selected.teacher_hash,teacher_side=obs.player==seat,index=index))+'\n')
                if transition.terminated or transition.truncated:break
                obs=env.observe(transition.next_player)
        result=dict(seed=seed,seat=seat,decisions=index+1,terminated=transition.terminated,
            truncated=transition.truncated or not transition.terminated,rewards=transition.rewards,
            teacher_hash=teacher.teacher_hash,opponent_hash=opponent.teacher_hash)
        (root/f'seed-{seed}-seat-{seat}.outcome.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        return result
    finally:
        env.close()


def prepare_development(output, manifest):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    from .data import observation_from_dict
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    maps=manifest['maps']['prompt_search']
    if len(maps)!=10:raise ValueError('exactly ten reserved search maps required')
    protocol=dict(schema='prompt-development-v1',map_manifest_hash=manifest['manifest_hash'],maps=maps,
        selection='Five public positions per map/seat, prefer distinct selected-action families with deterministic hash ties; teacher agreement is diagnostic only.',
        candidates=STRATEGIES,screen_first_attempts=700,repairs_max=2,
        confirmation_maps='Reserved separately; not used here.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    games=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(_development_game,str(root),m['seed'],seat) for m in maps for seat in (1,2)]
        for future in as_completed(futures):games.append(future.result())
    positions=[]
    for game in sorted(games,key=lambda g:(g['seed'],g['seat'])):
        path=root/f"seed-{game['seed']}-seat-{game['seat']}.jsonl.gz"
        rows=[json.loads(line) for line in gzip.open(path,'rt')]
        groups=defaultdict(list)
        for row in rows:
            if row['teacher_side']:groups[row['selected_command']['kind']].append(row)
        # Predeclared rotation spreads coverage across kinds, without model scores.
        kinds=sorted(groups)
        offset=(game['seed']+game['seat'])%len(kinds)
        kinds=kinds[offset:]+kinds[:offset]
        selected=[]
        for kind in kinds[:5]:
            selected.append(min(groups[kind],key=lambda r:digest(r['observation'])))
        for row in sorted(rows,key=lambda r:digest(r['observation'])):
            if len(selected)>=5:break
            if row['teacher_side'] and row not in selected:selected.append(row)
        if len(selected)!=5:raise ValueError('insufficient development positions; do not duplicate')
        for row in selected:
            obs=observation_from_dict(row['observation'])
            nonresign=[a for a in obs.legal_actions if a.command['kind']!=11]
            # Only mechanically forced actions/rewards are correctness checks.
            acceptable=[dict(a.command) for a in nonresign] if len(nonresign)==1 or all(a.command['kind']==9 for a in nonresign) else None
            positions.append(dict(position_id=f"{game['seed']}-{game['seat']}-{row['index']}",
                observation=row['observation'],teacher_command=row['selected_command'],
                mechanically_acceptable=acceptable,input_digest=digest(row['observation'])))
    assert len(positions)==100
    (root/'positions.json').write_text(json.dumps(positions,indent=2)+'\n')
    (root/'games.json').write_text(json.dumps(games,indent=2)+'\n')
    return dict(positions=100,games=len(games),positions_sha256=digest(positions))
