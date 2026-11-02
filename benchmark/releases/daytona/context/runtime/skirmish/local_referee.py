"""Local streamed referee, derived from the completed v5.1 evaluator.

Same Rust engine, action protocol, fuel and trace audit. No Modal imports/calls.
"""
import os,json,time,hashlib
from pathlib import Path

def file_sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
 return digest.hexdigest()

def play_game_streamed(a, b, seed, seat, budget, trace_path, *, phase, labels, authoring_deadline=None,
              map_type='classic', cap_policy='unresolved', evaluation=None, initial_snapshot=None):
    """Run the original move protocol with bounded memory trace hashing."""
    if os.environ.get('POLYTOPIA_EXECUTION_BACKEND') != 'local-docker':
        raise RuntimeError('local referee must run in its isolated Docker worker')
    from polytopia_bot.studio.environment import RustEnvironment
    from polytopia_bot.skirmish.game import ReferenceBot, WasiBot, BotFault, reset_map
    from polytopia_bot.skirmish.common import canonical, sha
    from dataclasses import asdict
    cfg = evaluation or {}
    env = RustEnvironment(size=cfg.get('size',14), max_rounds=cfg.get('max_rounds') if cfg.get('exact_rounds') else None, max_actions=cfg.get('max_actions',10000))
    bots = {seat: a, 3 - seat: b}
    started = time.monotonic()
    row = dict(a=labels[0], b=labels[1], seed=seed, seat=seat, phase=phase, map_type=map_type,
               status='unresolved', reason='game_timeout', winner=None, actions=0, evaluation=cfg,
               action_counts={'1':0,'2':0}, initial_snapshot=initial_snapshot)
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    private = path.with_suffix('.private.jsonl')
    chain = '0' * 64
    try:
        obs = env.restore(initial_snapshot.encode()) if initial_snapshot else reset_map(env, seed, map_type)
        # Warm reference inference outside the game-clock; compile time isn't a bot's move time.
        for bot in bots.values():
            if isinstance(bot, ReferenceBot):
                bot.select(obs)
        started = time.monotonic()
        game_deadline = started+budget.game_seconds
        if authoring_deadline is not None:
            game_deadline = min(game_deadline,authoring_deadline)
        for bot in bots.values():
            if isinstance(bot,WasiBot): bot.absolute_deadline = game_deadline
        with path.open('x') as public_log, private.open('x') as private_log:
            for _ in range(cfg.get('max_actions',10000)):
                if time.monotonic() >= game_deadline:
                    break
                player = int(env.state.current_player)
                obs = env.observe(player)
                try:
                    action_id = bots[player].select(obs)
                except BotFault as error:
                    if time.monotonic() >= game_deadline:
                        row.update(reason='authoring_deadline' if authoring_deadline is not None else 'game_timeout')
                    else:
                        row.update(status='forfeit', reason=str(error), winner=3 - player, fault_player=player)
                    break
                entry = dict(player=player, version=obs.state_version, action_id=action_id,
                             turn=obs.visible_state['turn'])
                chain = sha((chain + canonical(entry)).encode())
                private_log.write(canonical(dict(**entry, chain=chain)) + '\n')
                if player == seat:
                    public_log.write(canonical(dict(observation=asdict(obs), action_id=action_id)) + '\n')
                transition = env.step(action_id=action_id, expected_state_version=obs.state_version)
                row['actions'] += 1
                row['action_counts'][str(player)] += 1
                if transition.terminated:
                    winner = max(transition.rewards, key=transition.rewards.get)
                    row.update(status='native', reason=transition.reason, winner=winner)
                    break
                if transition.truncated:
                    row.update(reason='engine_truncation', cap_kind='round_limit' if int(env.state.turn)>=cfg.get('max_rounds',101) else 'action_limit')
                    break
            else:
                row.update(reason='action_limit', cap_kind='action_limit')
        row.update(turns=int(env.state.turn), seconds=time.monotonic() - started,
                   trace_sha256=file_sha(path), private_trace_sha256=file_sha(private),
                   action_chain_sha256=chain,
                   final_public={str(p):dict(cities=sum(c['owner']==p for c in env.observe(p).visible_state['cities']),
                     own_units=sum(u['owner']==p for u in env.observe(p).visible_state['units'])) for p in (1,2)})
    finally:
        env.close()
        a.close()
        b.close()
    row['result'] = 'unresolved' if row['winner'] is None else 'win' if row['winner'] == seat else 'loss'
    from polytopia_bot.skirmish.protocol import apply_cap_policy
    return apply_cap_policy(row,cap_policy)

def audit_trace_streamed(row, path):
    from dataclasses import asdict
    from polytopia_bot.skirmish.common import canonical, sha
    from polytopia_bot.skirmish.game import reset_map
    from polytopia_bot.studio.environment import RustEnvironment
    private=path.with_suffix('.private.jsonl')
    assert file_sha(path)==row['trace_sha256']
    assert file_sha(private)==row['private_trace_sha256']
    cfg=row.get('evaluation',{});env=RustEnvironment(size=cfg.get('size',14),max_rounds=cfg.get('max_rounds') if cfg.get('exact_rounds') else None,max_actions=cfg.get('max_actions',10000));count=0;chain='0'*64
    try:
        with path.open() as public_file,private.open() as private_file:
            public=iter(json.loads(line) for line in public_file)
            env.restore(row['initial_snapshot'].encode()) if row.get('initial_snapshot') else reset_map(env,row['seed'],row['map_type'])
            for line in private_file:
                entry=json.loads(line);expected=entry.pop('chain')
                chain=sha((chain+canonical(entry)).encode());assert chain==expected
                obs=env.observe(int(env.state.current_player))
                assert obs.player==entry['player'] and obs.state_version==entry['version']
                assert obs.visible_state['turn']==entry['turn']
                if obs.player==row['seat']:assert next(public)==dict(observation=asdict(obs),action_id=entry['action_id'])
                env.step(action_id=entry['action_id'],expected_state_version=obs.state_version);count+=1
            assert next(public,None) is None
            assert count==row['actions'] and chain==row['action_chain_sha256']
            assert int(env.state.turn)==row['turns']
            if row['status']=='native':
                assert bool(env.state.terminated)
                assert list(env.games.batch.terminal_rewards()[0])[row['winner']-1]>0
            elif row['status']=='draw':
                assert row['winner'] is None and row['reason'] in ('engine_truncation','action_limit')
                if row['reason']=='engine_truncation':assert bool(env.state.truncated)
            elif row['status']=='forfeit':assert not bool(env.state.terminated) and row['winner']==3-row['fault_player']
    finally:env.close()
    return count
