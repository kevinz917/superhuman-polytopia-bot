"""Validate and publish a complete five-bot follow-up without changing the original archive.

No model calls or new games. Run only after native Harbor has completed successfully.
The output contains audited scores, sources, conversations, compact action replays,
practice records, and the exact source package used for this follow-up.
"""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.summarize_skirmish_harbor import summarize, trace_digest
from polytopia_bot.skirmish.common import canonical, sha, write_json
from polytopia_bot.skirmish.continuous import rewards, SCHEMA, CODING_SECONDS
from polytopia_bot.skirmish.inference import proxy_model

ENTRY = re.compile(rb'"action_id":"([0-9]+)","chain":"[0-9a-f]{64}","player":([12]),"turn":([0-9]+),"version":([0-9]+)')
SECRET = re.compile(rb'sk-or-v1-[0-9a-f]{32,}|sk-ant-[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def pack(files, destination):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, raw in sorted(files.items()):
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert not SECRET.search(raw), 'Secret pattern in '+name
            item = tarfile.TarInfo(name)
            item.size, item.mode, item.mtime = len(raw), 0o644, 0
            archive.addfile(item, io.BytesIO(raw))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress(stream.getvalue(), compresslevel=9, mtime=0))


def combined_summary(run, record):
    """Validate native trial receipts across explicitly recorded interrupted/replacement jobs."""
    assert record['original_campaign'] == run.name
    assert len(record['retained_original_trials']) == 3 and len(record['replacement_trials']) == 2
    assert len(record['stopped_trials']) == 2
    replacement = run.parent/record['replacement_campaign']
    locations = {name: run/name for name in record['retained_original_trials']}
    locations.update({name: replacement/name for name in record['replacement_trials']})
    assert len(locations) == 5
    trials = []
    for name, base in locations.items():
        native = json.loads((base/'result.json').read_text())
        assert native['exception_info'] is None and native['finished_at']
        author = json.loads((base/'agent/authoring.json').read_text())
        protocol = json.loads((base/'agent/protocol.json').read_text())
        grading = json.loads((base/'verifier/summary.json').read_text())
        plan = json.loads((base/'verifier/evaluation-plan.json').read_text())
        rows = json.loads((base/'verifier/games.json').read_text())
        assert protocol['schema'] == SCHEMA and protocol['coding_seconds'] == CODING_SECONDS
        assert author['coding_budget_seconds'] == CODING_SECONDS and author['timing']['coding_seconds'] <= CODING_SECONDS
        assert len(rows) == len(plan['jobs']) == grading['rewards']['games'] == 360
        assert grading['profile'] == 'standard'
        for row in rows:
            planned = plan['jobs'][row['job_index']]
            assert row['job_sha256'] == planned['job_sha256'] == sha(canonical({k:v for k,v in planned.items() if k != 'job_sha256'}).encode())
            assert row['audit']['passed'] and row['result'] != 'unresolved'
        assert dict(rewards(rows, 360), evaluation_is_demo=0) == grading['rewards']
        capture = author['capture']
        assert sha((base/'artifacts/submission/main.rs').read_bytes()) == capture['source_sha256'] == grading['submission']['source_sha256']
        hero = [r for r in rows if r['opponent_label'] == 'hero-1v1']
        trials.append(dict(model=native['config']['agent']['model_name'], trial=name,
            score=grading['score'], games=360, profile='standard',
            quality=grading['submission']['submission_quality'], compile_returncode=grading['submission']['returncode'],
            champion={key:sum(r['result']==singular for r in hero) for key,singular in (('wins','win'),('draws','draw'),('losses','loss'))},
            practice_matches=len(list((base/'artifacts/practice').glob('*/report.json'))),
            coding_budget_seconds=CODING_SECONDS, timing=author['timing'], usage=author['usage'],
            inference=protocol['inference'], replay_actions=grading['replay_actions'], rewards=grading['rewards']))
    for name in record['stopped_trials']:
        base = run/name
        native = json.loads((base/'result.json').read_text())
        assert native['exception_info'] is not None and native['verifier'] is None
        assert (base/'artifacts/submission/capture.json').exists()
    return dict(schema=SCHEMA, trials=trials, total_final_outcomes=1800,
                replay_actions=sum(t['replay_actions'] for t in trials)), locations


def export(run, package, destination, model, label, restart_record=None):
    record = json.loads(restart_record.read_text()) if restart_record else None
    if record:
        summary, locations = combined_summary(run, record)
    else:
        summary = summarize(run)
        locations = {t['trial']:run/t['trial'] for t in summary['trials']}
    trials = summary['trials']
    assert len(trials) == 5 and {t['model'] for t in trials} == {model}
    assert all(t['profile'] == 'standard' and t['games'] == 360 for t in trials)
    assert summary['total_final_outcomes'] == 1800
    build = json.loads((package/'build.json').read_text())
    original = ROOT/'docs/skirmish/results/continuous-full-20261031'
    old_build = json.loads((original/'build-provenance.json').read_text())['local']
    assert build['manifest_sha256'] == old_build['manifest_sha256']
    assert build['maps_sha256'] == old_build['maps_sha256']
    assert build['coder_image'] == old_build['coder_image']
    # Only grading/scheduling infrastructure may differ from the frozen game.
    allowed = {'src/polytopia_bot/skirmish/local_eval.py', 'src/polytopia_bot/skirmish/harbor_verifier.py',
               'scripts/build_skirmish_harbor.py', 'scripts/run_skirmish_harbor.py',
               'scripts/check_skirmish_harbor.py'}
    differences = {k for k, v in old_build['source_files'].items() if build['source_files'].get(k) != v}
    assert differences <= allowed, 'Unexpected frozen protocol/game source drift: '+str(differences-allowed)
    assert all(sha((ROOT/k).read_bytes()) == v for k, v in build['source_files'].items())
    destination.mkdir(parents=True, exist_ok=True)
    if record:
        write_json(destination/'restart-record.json', record)
        for name in record['stopped_trials']:
            base = run/name
            files = {str(p.relative_to(base)):p.read_bytes() for p in (base/'agent').rglob('*') if p.is_file()}
            for name_in_trial in ('config.json','result.json','exception.txt','trial.log','artifacts/submission/main.rs','artifacts/submission/capture.json'):
                if (base/name_in_trial).exists():
                    files[name_in_trial] = (base/name_in_trial).read_bytes()
            pack(files, destination/'interrupted'/name/'authoring.tar.gz')
    shutil.copyfile(package/'source.tar.gz', destination/'harness-source.tar.gz')
    write_json(destination/'build-provenance.json', build)
    for source, name in (
        (package/(run.name+'.probes.json'), 'transport-probe.json'),
        (ROOT/'artifacts/harbor-campaigns'/run.name/'parallel-qualification/qualification.json', 'parallel-qualification.json'),
        (ROOT/'artifacts/harbor-runs/fable52-parallel-fixture-20261110/integration-check.json', 'integration-check.json'),
        (ROOT/'artifacts/harbor-campaigns'/run.name/'operator-context.json', 'operator-context.json'),
        (ROOT/'artifacts/harbor-campaigns'/run.name/'grader-admission.json', 'grader-admission.json'),
        (ROOT/'artifacts/harbor-campaigns'/run.name/'original-archive-recheck.json', 'original-archive-recheck.json'),
    ):
        if source.exists():
            shutil.copyfile(source, destination/name)
    spec = importlib.util.spec_from_file_location('article_analysis', ROOT/'scripts/analyze_skirmish_article.py')
    article = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(article)
    cells_models = []
    totals = [Counter() for _ in range(5)]
    native_trials = []
    cache_totals = Counter()
    actual_games = virtual_losses = 0
    expected_schedule = json.loads((original/'results.json').read_text())['schedule_sha256']
    for trial in trials:
        base = locations[trial['trial']]
        logs = base/'verifier'
        assert json.loads((base/'agent/protocol.json').read_text())['source_files'] == build['source_files']
        rows = json.loads((logs/'games.json').read_text())
        plan = json.loads((logs/'evaluation-plan.json').read_text())
        jobs = plan['jobs']
        receipt = json.loads((logs/'submission.json').read_text())
        source = (base/'artifacts/submission/main.rs').read_bytes()
        assert sha(source) == receipt['source_sha256'] == json.loads((base/'agent/authoring.json').read_text())['capture']['source_sha256']
        source_name = 'sources/'+trial['trial']+'.rs'
        (destination/'sources').mkdir(exist_ok=True)
        (destination/source_name).write_bytes(source)
        calculated = dict(rewards(rows, 360), evaluation_is_demo=0)
        assert calculated == trial['rewards']
        assert abs(trial['score']-100*calculated['reward']) < 1e-10
        schedule = [(j['opponent'], j['map_type'], j['seed'], j['seat'], j['evaluation']) for j in jobs]
        assert sha(canonical(schedule).encode()) == expected_schedule
        trajectory = json.loads((base/'agent/trajectory.json').read_text())
        initial_users = [m['content'] for m in trajectory['messages'] if m.get('role') == 'user']
        assert initial_users and initial_users[0].strip() == (ROOT/'harbor/tasks/polytopia-skirmish/instruction.md').read_text().strip()
        response_models = set()
        for message in trajectory['messages']:
            response = message.get('extra', {}).get('response') or {}
            if response:
                response_models.add(response.get('model'))
                usage = (response.get('usage') or {}).get('anthropic_usage') or {}
                for k in ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens', 'output_tokens'):
                    cache_totals[k] += usage.get(k, 0)
        assert response_models == {proxy_model(model)}, 'Unexpected response model: '+str(response_models)
        units, early_units, techs, builds, families = [Counter() for _ in range(5)]
        cells = defaultdict(Counter)
        games = []
        for row in rows:
            assert row['audit']['passed'] and row['result'] in ('win', 'draw', 'loss')
            if row['audit'].get('virtual'):
                virtual_losses += 1
            else:
                actual_games += 1
            cell = cells[row['opponent_label'], row['map_type']]
            cell[row['result']] += 1
            cell['faults'] += row['status'] == 'forfeit'
            cell['turns'] += row.get('turns', 0)
            entries = []
            if not row['audit'].get('virtual'):
                rel = Path(row['trace_relative']).relative_to('games')
                public = logs/'traces'/rel
                assert trace_digest(public) == row['trace_sha256']
                plain = public.with_suffix('') if public.suffix == '.gz' else public
                private = plain.with_suffix('.private.jsonl').read_bytes()
                assert sha(private) == row['private_trace_sha256']
                entries = [[int(p), int(v), int(t), a.decode()] for a, p, t, v in ENTRY.findall(private)]
                assert len(entries) == row['actions'] == row['audit']['actions']
                for player, version, turn, action in entries:
                    if player != row['seat']:
                        continue
                    family, parameter = article.decode(int(action))
                    families[family] += 1
                    if family == 2:
                        units[parameter] += 1
                        if turn <= 20:
                            early_units[parameter] += 1
                    elif family == 3:
                        builds[parameter] += 1
                    elif family == 4:
                        techs[parameter] += 1
            games.append(dict(summary=row, actions=[e[3] for e in entries], entries=entries))
        assert len(cells) == 18 and all(sum(c[k] for k in ('win', 'draw', 'loss')) == 20 for c in cells.values())
        trial_article = dict(id=trial['trial'], score=trial['score'], source=source_name,
            sourceSha256=receipt['source_sha256'], wasmSha256=receipt.get('wasm_sha256'),
            practiceMatches=trial['practice_matches'], activeMinutes=trial['timing']['coding_seconds']/60,
            cells=[dict(opponent=op, map=family, wins=c['win'], draws=c['draw'], losses=c['loss'],
                        faults=c['faults'], turns=c['turns']) for (op, family), c in sorted(cells.items())])
        cells_models.append(trial_article)
        for total, counts in zip(totals, (units, early_units, techs, builds, families), strict=True):
            total.update(counts)
        target = destination/'runs'/trial['trial']
        target.mkdir(parents=True, exist_ok=True)
        final = {name: json.loads((logs/name).read_text()) for name in
                 ('summary.json', 'games.json', 'evaluation-plan.json', 'submission.json', 'reward.json')}
        if (logs/'bot.wasm').exists():
            wasm = (logs/'bot.wasm').read_bytes()
            assert sha(wasm) == receipt['wasm_sha256']
            final['wasm_base64'] = base64.b64encode(wasm).decode()
        replay = dict(schema='skirmish-compact-replay-v1', source_sha256=receipt['source_sha256'],
            referee_source_sha256=build['source_files']['src/polytopia_bot/skirmish/local_referee.py'], games=games, final=final)
        (target/'final-replay.json.gz').write_bytes(gzip.compress(canonical(replay).encode(), compresslevel=9, mtime=0))
        files = {str(p.relative_to(base)): p.read_bytes() for p in (base/'agent').rglob('*') if p.is_file()}
        for name in ('config.json', 'result.json', 'trial.log'):
            if (base/name).exists():
                files[name] = (base/name).read_bytes()
        files['artifacts/submission/main.rs'] = source
        pack(files, target/'authoring.tar.gz')
        practices, practice_files = [], {}
        for folder in sorted((base/'artifacts/practice').glob('*')):
            if not folder.is_dir():
                continue
            practice_record = dict(id=folder.name, completed=(folder/'report.json').exists(), files={})
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    raw = path.read_bytes()
                    practice_record['files'][path.name] = dict(sha256=sha(raw), bytes=len(raw))
                    if path.name != 'game.jsonl' or not practice_record['completed']:
                        practice_files[folder.name+'/'+path.name] = raw
            practices.append(practice_record)
        practice_files['index.json'] = (json.dumps(practices, indent=2)+'\n').encode()
        pack(practice_files, target/'practice.tar.gz')
        native_trials.append(dict(trial, source_file=source_name, source_sha256=receipt['source_sha256'],
            wasm_sha256=receipt.get('wasm_sha256'), original_trial=str(base.relative_to(ROOT)),
            graded_trial=str(base.relative_to(ROOT)), archive='runs/'+trial['trial'], response_models=sorted(response_models)))
        print(trial['trial'], trial['score'], '360 audited games archived', flush=True)
    scores = [t['score'] for t in trials]
    units, early_units, techs, builds, families = totals
    naval = Counter({kind: units.pop(kind, 0) for kind in (41, 42, 44)})
    for kind in (41, 42, 44):
        early_units.pop(kind, None)
    model_data = dict(id=model, label=label, mean=statistics.mean(scores), sd=statistics.stdev(scores),
        trials=sorted(cells_models, key=lambda t: t['id']), navalUpgrades=article.named(naval, article.UNIT_NAMES),
        recruits=article.named(units, article.UNIT_NAMES), earlyRecruits=article.named(early_units, article.UNIT_NAMES),
        research=article.named(techs, article.TECH_NAMES), builds=article.named(builds, article.STRUCTURE_NAMES),
        actionFamilies={str(k): v for k, v in families.items()})
    write_json(destination/'article-model.json', model_data)
    hero = {key: sum(t['champion'][key] for t in trials) for key in ('wins', 'draws', 'losses')}
    report = dict(status='completed_validated', campaign=run.name, protocol=summary['schema'], model=model,
        completed_at=datetime.now(timezone.utc).isoformat(), attempts_per_model=5, authored_submissions=5,
        coding_seconds=2400, total_final_outcomes=1800, mean_score=model_data['mean'], score_sd=model_data['sd'],
        actual_played_games=actual_games, virtual_compile_failure_losses=virtual_losses,
        champion_wdl=hero, schedule_sha256=expected_schedule, trials=native_trials,
        usage=dict(cache_totals), completed_call_cost_usd=None, inference='local CLI proxy; subscription usage',
        usage_scope='Completed responses from the five scored attempts; interrupted sessions remain in their separate authoring archives.',
        selection_rule=('Three retained original submissions plus two replacements after externally requested interruptions; every scheduled final outcome from these five completed attempts counts.' if record else 'All five first independently authored submissions, including every scheduled final outcome.'),
        comparison=dict(original_campaign='continuous-full-20261031', identical_coder_image=True,
            identical_maps_and_opponents=True, identical_game_and_bot_limits=True,
            changed_infrastructure_sources=sorted(differences), grading_workers_per_container=4,
            max_grader_containers=3, grader_cpus_per_container=4,
            grading_admission='One four-worker verifier while authoring remains active; up to three after all sources are frozen.',
            caveat='Follow-up on the original fixed panel. Maps and references were published after the original campaign; they remain unavailable inside this offline authoring sandbox.'))
    if record:
        report['replacement_campaign'] = record['replacement_campaign']
        report['interrupted_attempts'] = record['stopped_trials']
        report['total_authoring_sessions'] = 7
        report['excluded_interrupted_submissions'] = 2
        report['comparison']['selection_deviation'] = 'Two original sessions were stopped during unbounded filesystem searches and replaced. The interrupted sessions are archived and excluded. This differs from the original uninterrupted five-attempt procedure.'
    write_json(destination/'results.json', report)
    write_json(destination/'archive-manifest.json', dict(schema='polytopia-bench-extension-archive-v1',
        files={str(p.relative_to(destination)):dict(sha256=sha(p.read_bytes()), bytes=p.stat().st_size)
               for p in sorted(destination.rglob('*')) if p.is_file() and p.name not in ('archive-manifest.json', 'archive-validation.json')}))
    write_json(destination/'archive-validation.json', dict(passed=True, authored_submissions=5, final_games=1800,
        actual_played_games=actual_games, virtual_compile_failure_losses=virtual_losses,
        replay_actions=summary['replay_actions'], original_public_and_private_hashes_checked=True,
        source_and_wasm_hashes_checked=True, same_original_schedule=True, response_model_verified=True))
    lines = ['# Claude Fable 5.2: Polytopia Bench follow-up', '',
        f"Five independent local CLI-proxy attempts: **{model_data['mean']:.2f} ± {model_data['sd']:.2f} panel points /100** (mean ± sample SD).",
        f"Champion results: {hero['wins']} wins, {hero['draws']} draws, {hero['losses']} losses across 300 games.", '',
        'Each completed attempt received one continuous 40-minute authoring allowance, followed by the original six-opponent, three-map-family, ten-layout, both-seat schedule (360 games per bot). All 1,800 outcomes and replay audits from these five completed attempts are retained. No post-cutoff source repair.', '',
        report['comparison']['caveat'], '',
        'The immutable original coder/toolchain image and all game/reference hashes match the October 31 campaign. Independent final games now run in four spawned processes per verifier, with up to three local verifiers. No OpenRouter, Daytona, or paid API fallback was used; subscription dollar cost and quota percentage are unavailable.', '',
        '| Submission | Panel points | Champion W–D–L | Active minutes | Practice games | Submission quality |',
        '|---|---:|---:|---:|---:|---|']
    if record:
        lines[2:2] = ['**Restart disclosure.** '+report['comparison']['selection_deviation']+' See [restart record](restart-record.json) and `interrupted/` for the original conversations and cutoff sources. These interruptions are not reported as model losses or as spontaneous harness failures.', '']
    for t in native_trials:
        h=t['champion']
        lines.append(f"| [{t['trial']}]({t['source_file']}) | {t['score']:.2f} | {h['wins']}–{h['draws']}–{h['losses']} | {t['timing']['coding_seconds']/60:.2f} | {t['practice_matches']} | {t['quality']} |")
    lines += ['', 'Scores, hashes and provenance: [results.json](results.json). Article-ready aggregates: [article-model.json](article-model.json).', '',
        '`runs/` contains each exact authoring conversation, compact final replay, and practice archive. Completed practice observations can be reconstructed from the private action log and report using the pinned engine. `harness-source.tar.gz` contains the exact follow-up harness sources; `build-provenance.json` pins their hashes. The original benchmark archive is unchanged.']
    (destination/'README.md').write_text('\n'.join(lines)+'\n')
    # Include the readable report in the inventory after creating it.
    manifest = json.loads((destination/'archive-manifest.json').read_text())
    manifest['files']['README.md'] = dict(sha256=sha((destination/'README.md').read_bytes()), bytes=(destination/'README.md').stat().st_size)
    write_json(destination/'archive-manifest.json', manifest)
    print(json.dumps({k:report[k] for k in ('status', 'mean_score', 'score_sd', 'champion_wdl', 'total_final_outcomes')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='anthropic/claude-fable-5.2')
    parser.add_argument('--label', default='Claude Fable 5.2')
    parser.add_argument('--restart-record', type=Path)
    args = parser.parse_args()
    export(args.run.resolve(), args.package.resolve(), args.output.resolve(), args.model, args.label,
           args.restart_record.resolve() if args.restart_record else None)
