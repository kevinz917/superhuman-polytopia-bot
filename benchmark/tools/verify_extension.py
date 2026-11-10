"""Verify a follow-up's inventory, every private replay chain, and all scores offline."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import base64
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

from verify_archive import canonical, sha, read_member

ROOT = Path(__file__).resolve().parents[1]


def verify(folder):
    inventory = json.loads((folder/'archive-manifest.json').read_text())
    for relative, info in inventory['files'].items():
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        path = folder/relative
        assert path.stat().st_size == info['bytes'] < 100_000_000, relative
        assert sha(path.read_bytes()) == info['sha256'], relative
    report = json.loads((folder/'results.json').read_text())
    article = json.loads((folder/'article-model.json').read_text())
    build = json.loads((folder/'build-provenance.json').read_text())
    original = json.loads((ROOT/'releases/local/build.json').read_text())
    original_report = json.loads((ROOT/'runner/docs/skirmish/results/continuous-full-20261031/results.json').read_text())
    assert report['schedule_sha256'] == original_report['schedule_sha256']
    assert build['coder_image'] == original['coder_image']
    for key in ('maps_sha256', 'manifest_sha256'):
        assert build[key] == original[key]
    with tarfile.open(folder/'harness-source.tar.gz') as archive:
        for relative, expected in build['source_files'].items():
            assert sha(read_member(archive, relative)) == expected, relative
    assert len(report['trials']) == report['attempts_per_model'] == 5
    assert report['total_final_outcomes'] == 1800
    article_trials = {t['id']: t for t in article['trials']}
    scores, private_rebuilt, actions, virtual_losses = [], 0, 0, 0
    for trial in report['trials']:
        source = (folder/trial['source_file']).read_bytes()
        assert sha(source) == trial['source_sha256']
        base = folder/trial['archive']
        payload = json.loads(gzip.decompress((base/'final-replay.json.gz').read_bytes()))
        assert payload['source_sha256'] == trial['source_sha256']
        final = payload['final']
        assert final['submission.json']['source_sha256'] == trial['source_sha256']
        if trial.get('wasm_sha256'):
            assert sha(base64.b64decode(final['wasm_base64'])) == trial['wasm_sha256']
        plan, rows = final['evaluation-plan.json']['jobs'], final['games.json']
        assert len(plan) == len(rows) == len(payload['games']) == 360
        schedule = [(j['opponent'], j['map_type'], j['seed'], j['seat'], j['evaluation']) for j in plan]
        assert sha(canonical(schedule).encode()) == report['schedule_sha256']
        assert {r['job_index'] for r in rows} == set(range(360))
        records = {g['summary']['job_index']: g for g in payload['games']}
        cells = Counter()
        for row in rows:
            assert row['audit']['passed'] and row['result'] in ('win', 'draw', 'loss')
            job = plan[row['job_index']]
            assert row['job_sha256'] == job['job_sha256'] == sha(canonical({k: v for k, v in job.items() if k != 'job_sha256'}).encode())
            record = records[row['job_index']]
            assert record['summary'] == row and len(record['entries']) == row['actions']
            assert record['actions'] == [e[3] for e in record['entries']]
            cells[row['opponent_label'], row['map_type'], row['result']] += 1
            if row['audit'].get('virtual'):
                virtual_losses += 1
            if not row['audit'].get('virtual'):
                chain, digest = '0'*64, hashlib.sha256()
                for player, version, turn, action in record['entries']:
                    entry = dict(player=player, version=version, turn=turn, action_id=action)
                    chain = sha((chain+canonical(entry)).encode())
                    digest.update((canonical(dict(**entry, chain=chain))+'\n').encode())
                assert digest.hexdigest() == row['private_trace_sha256']
                assert chain == row['action_chain_sha256']
                private_rebuilt += 1
                actions += row['actions']
        score = 100*sum({'win': 1, 'draw': .5, 'loss': 0}[r['result']] for r in rows)/360
        assert abs(score-trial['score']) < 1e-10
        scores.append(score)
        assert final['reward.json'] == trial['rewards'] == final['summary.json']['rewards']
        a = article_trials[trial['trial']]
        for cell in a['cells']:
            for singular, plural in (('win','wins'), ('draw','draws'), ('loss','losses')):
                assert cell[plural] == cells[cell['opponent'],cell['map'],singular]
        with tarfile.open(base/'authoring.tar.gz') as archive:
            author = json.loads(read_member(archive, 'agent/authoring.json'))
            assert json.loads(read_member(archive, 'agent/protocol.json'))['source_files'] == build['source_files']
            assert author['capture']['source_sha256'] == trial['source_sha256']
            assert author['coding_budget_seconds'] == 2400 and author['timing']['coding_seconds'] <= 2400
            assert sha(read_member(archive, 'artifacts/submission/main.rs')) == trial['source_sha256']
            trajectory = json.loads(read_member(archive, 'agent/trajectory.json'))
            users = [m['content'] for m in trajectory['messages'] if m.get('role') == 'user']
            assert users and users[0].strip() == (ROOT/'runner/harbor/tasks/polytopia-skirmish/instruction.md').read_text().strip()
            models = {m['extra']['response'].get('model') for m in trajectory['messages'] if m.get('extra',{}).get('response')}
            assert models == {'claude-fable-5-2'}
        with tarfile.open(base/'practice.tar.gz') as archive:
            for practice in json.loads(read_member(archive, 'index.json')):
                for name, info in practice['files'].items():
                    if name == 'game.jsonl' and practice['completed']:
                        continue
                    raw = read_member(archive, practice['id']+'/'+name)
                    assert len(raw) == info['bytes'] and sha(raw) == info['sha256']
        print('Verified', trial['trial'], f'{score:.2f}', flush=True)
    assert abs(statistics.mean(scores)-report['mean_score']) < 1e-10
    assert abs(statistics.stdev(scores)-report['score_sd']) < 1e-10
    assert abs(article['mean']-report['mean_score']) < 1e-10
    assert abs(article['sd']-report['score_sd']) < 1e-10
    assert private_rebuilt == report['actual_played_games']
    assert virtual_losses == report['virtual_compile_failure_losses']
    assert private_rebuilt+virtual_losses == 1800
    interrupted_checked = 0
    if report.get('interrupted_attempts'):
        record = json.loads((folder/'restart-record.json').read_text())
        assert record['stopped_trials'] == report['interrupted_attempts']
        assert len(record['stopped_trials']) == report['excluded_interrupted_submissions'] == 2
        assert set(t['trial'] for t in report['trials']) == set(record['retained_original_trials']+record['replacement_trials'])
        for name in record['stopped_trials']:
            with tarfile.open(folder/'interrupted'/name/'authoring.tar.gz') as archive:
                native = json.loads(read_member(archive, 'result.json'))
                capture = json.loads(read_member(archive, 'artifacts/submission/capture.json'))
                assert native['exception_info'] is not None and native['verifier'] is None
                assert sha(read_member(archive, 'artifacts/submission/main.rs')) == capture['source_sha256']
                assert archive.getmember('agent/trajectory.json').size > 0
            interrupted_checked += 1
    proof = dict(passed=True, checked_at=datetime.now(timezone.utc).isoformat(),
        authored_submissions=5, final_games=1800, replay_actions=actions,
        actual_played_games=private_rebuilt, virtual_compile_failure_losses=virtual_losses,
        private_game_hashes_reconstructed=private_rebuilt, authoring_trajectories=5,
        original_public_and_private_hashes_checked=True, source_and_wasm_hashes_checked=True,
        same_original_schedule=True, response_model_verified=True, file_hashes_verified=len(inventory['files']),
        same_task_instruction_verified=True,
        interrupted_archives_checked=interrupted_checked,
        public_observation_validation='Original native replay audits and export-time public SHA256 checks retained; this verifier reconstructs every private log byte without new model calls.')
    (folder/'archive-validation.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('extension', type=Path, nargs='?', default=ROOT/'extensions/fable52-20261110')
    verify(parser.parse_args().extension.resolve())
