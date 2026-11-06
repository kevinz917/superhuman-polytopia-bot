"""Predeclared teacher qualification and auditable public episode traces."""
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time
from .adapter import GPUEnvironment
from .interface import canonical, digest
from .teacher import ScriptedTeacher, WeakBaseline


def source_receipt(output):
    root = Path(__file__).resolve().parents[3]
    files = [*sorted((root / 'src/polytopia_bot/learning').glob('*.py')),
             root / 'configs/ruleset-v1.json', root / 'configs/sft.json']
    hashes = {}
    for path in files:
        relative = path.relative_to(root)
        target = output / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = path.read_bytes()
        target.write_bytes(payload)
        hashes[str(relative)] = hashlib.sha256(payload).hexdigest()
    return hashes


def qualify(output, manifest, *, map_start=0, map_count=100):
    """100 validation maps x both seats; stop only on irrecoverable gate failure."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    teacher, opponent = ScriptedTeacher(), WeakBaseline()
    maps = manifest['maps']['validation'][map_start:map_start+map_count]
    if len(maps) != map_count or not 1 <= map_count <= 100 or manifest['size'] != 14:
        raise ValueError('qualification shards require reserved paired 14x14 validation maps')
    protocol = dict(schema='teacher-qualification-v1', map_manifest_hash=manifest['manifest_hash'],
        maps=maps, split='validation', teacher_hash=teacher.teacher_hash, opponent_hash=opponent.teacher_hash,
        teacher='T-main-v0 balanced', opponent='weak-public-v0', games_required=200,
        zero_illegal_decisions=True, teacher_decisions_min=10000,
        native_completion_min=.95, truncation_strict_max=.05, win_equivalent_min=.60,
        early_stop='10 truncations or any illegal action makes the fixed 200-game gate unattainable',
        sources=source_receipt(root))
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    env, games, errors, decisions = GPUEnvironment(14), [], [], 0
    started = time.monotonic()
    (root / 'private').mkdir()
    try:
        for board in maps:
            for seat in (1, 2):
                obs = env.reset(seed=board['seed'], settings={})
                name = f"seed-{board['seed']}-seat-{seat}"
                (root / 'private' / f'{name}.snapshot.json').write_bytes(env.snapshot())
                families = Counter()
                game_start, teacher_decisions = time.monotonic(), 0
                with gzip.open(root / f'{name}.jsonl.gz', 'xt', compresslevel=1) as trace:
                    for index in range(10000):
                        policy = teacher if obs.player == seat else opponent
                        selected = policy.select(obs)
                        row = dict(observation=asdict(obs), input_digest=digest(asdict(obs)), selection=asdict(selected))
                        try:
                            transition = env.step(action_id=selected.action.action_id, expected_state_version=obs.state_version)
                        except Exception as exc:
                            row['error'] = f'{type(exc).__name__}: {exc}'
                            trace.write(canonical(row)+'\n')
                            raise
                        teacher_decisions += obs.player == seat
                        families[str(selected.action.command['kind'])] += 1
                        # Keep both full public observation logs separately from bounded
                        # prompt memory. No hidden snapshot enters this public trace.
                        row['after'] = [asdict(env.observe(p)) for p in (1, 2)]
                        row['outcome'] = dict(terminated=transition.terminated, truncated=transition.truncated,
                                             rewards=transition.rewards, reason=transition.reason)
                        trace.write(canonical(row)+'\n')
                        if transition.terminated or transition.truncated:
                            break
                        obs = env.observe(transition.next_player)
                decisions += teacher_decisions
                result = dict(seed=board['seed'], board_hash=board['board_hash'], seat=seat,
                    decisions=index+1, teacher_decisions=teacher_decisions, round=int(env.state.turn),
                    terminated=transition.terminated, truncated=transition.truncated or not transition.terminated,
                    reason=transition.reason or 'research_action_limit', reward=transition.rewards[seat],
                    families=dict(families), seconds=time.monotonic()-game_start)
                games.append(result)
                (root / 'games.json').write_text(json.dumps(games, indent=2)+'\n')
                print(json.dumps(dict(game=len(games), **result)), flush=True)
                if sum(g['truncated'] for g in games) >= 10:
                    errors.append('Truncation gate cannot pass even if all remaining games complete.')
                    break
            if errors:
                break
    except Exception as exc:
        import traceback
        (root / 'failure-traceback.txt').write_text(traceback.format_exc())
        errors.append(f'{type(exc).__name__}: {exc}')
        print(errors[-1], flush=True)
    finally:
        env.close()
    completed = sum(g['terminated'] for g in games)
    truncations = sum(g['truncated'] for g in games)
    wins = sum(g['reward'] > 0 for g in games)
    # Qualification denominators remain the registered 200, even after early stop.
    gates = dict(all_games_run=len(games)==200, zero_execution_errors=not errors,
        teacher_decisions=decisions>=10000, native_completion=completed>=190,
        truncation=truncations<10, strength=wins>=120)
    report = dict(protocol_sha256=digest(protocol), games_run=len(games), native_completions=completed,
        truncations=truncations, wins=wins, teacher_decisions=decisions, errors=errors,
        gates=gates, teacher_qualified=all(gates.values()), elapsed_seconds=time.monotonic()-started)
    (root / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


def _qualification_shard(output, manifest, start):
    return qualify(output, manifest, map_start=start, map_count=25)


def qualify_parallel(output, manifest):
    """Four isolated CPU workers, same 100 paired maps and unchanged policies."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema='teacher-qualification-parallel-v1', workers=4,
        map_manifest_hash=manifest['manifest_hash'], maps=manifest['maps']['validation'][:100],
        source_sha256=source_receipt(root), total_games=200, teacher_decisions_min=10000,
        native_completion_min=.95, truncation_strict_max=.05, win_equivalent_min=.60)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.monotonic()
    reports=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(_qualification_shard,str(root/f'worker-{i}'),manifest,i*25) for i in range(4)]
        for future in as_completed(futures):
            reports.append(future.result())
    games=[]
    for i in range(4):
        path=root/f'worker-{i}'/'games.json'
        if path.exists():games.extend(json.loads(path.read_text()))
    games.sort(key=lambda r:(r['seed'],r['seat']))
    (root/'games.json').write_text(json.dumps(games,indent=2)+'\n')
    errors=[error for r in reports for error in r['errors']]
    completed=sum(g['terminated'] for g in games)
    truncated=sum(g['truncated'] for g in games)
    wins=sum(g['reward']>0 for g in games)
    decisions=sum(g['teacher_decisions'] for g in games)
    gates=dict(all_games_run=len(games)==200,zero_execution_errors=not errors,
        teacher_decisions=decisions>=10000,native_completion=completed>=190,truncation=truncated<10,strength=wins>=120)
    report=dict(protocol_sha256=digest(protocol),games_run=len(games),native_completions=completed,
        truncations=truncated,wins=wins,teacher_decisions=decisions,errors=errors,gates=gates,
        teacher_qualified=all(gates.values()),elapsed_seconds=time.monotonic()-started)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report
