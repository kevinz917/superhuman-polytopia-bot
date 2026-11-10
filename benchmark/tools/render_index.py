"""Render the current score index while retaining the original campaign archive."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = json.loads((ROOT/'runner/docs/skirmish/results/continuous-full-20261031/article-data.json').read_text())
    extension = ROOT/'extensions/fable52-20261110'
    result = json.loads((extension/'results.json').read_text())
    validation = json.loads((extension/'archive-validation.json').read_text())
    model = json.loads((extension/'article-model.json').read_text())
    assert result['status'] == 'completed_validated' and validation['passed']
    assert result['schedule_sha256'] == base['provenance']['scheduleSha256']
    assert result['total_final_outcomes'] == 1800 and len(model['trials']) == 5
    models = sorted([*base['models'], model], key=lambda m: (-m['mean'], m['id']))
    assert len(models) == 13 and sum(len(m['trials']) for m in models) == 65
    lines = ['# Polytopia Bench: current results', '',
        '**13 models · 65 completed coding attempts · 23,400 audited final games.**', '',
        'Each attempt has one continuous 40-minute allowance to write and practice a Rust bot. The frozen submission plays six opponents × three map families × ten layouts × both seats: 360 games. Scores are panel points /100 (win 1, draw ½), reported as mean ± sample SD over five attempts. They are not expert win rates.', '',
        'The October 31 archive remains intact. Claude Fable 5.2 was added on November 10 using the identical frozen coder image, game/reference hashes, maps, seats and runtime limits. The original evaluation maps have since been published; they remain outside the follow-up’s offline coding sandbox. The follow-up uses parallel grading and the local CLI subscription proxy.', '',
        '**Fable 5.2 restart disclosure:** '+result['comparison']['selection_deviation']+' Its result combines three retained original attempts and two replacements. The interrupted conversations and cutoff sources are included in the follow-up archive.', '',
        '| Model | Panel points /100 | Champion wins /300 | Evaluated |',
        '|---|---:|---:|---|']
    for m in models:
        wins = sum(c['wins'] for t in m['trials'] for c in t['cells'] if c['opponent'] == 'hero-1v1')
        latest = m['id'] == model['id']
        link = 'extensions/fable52-20261110/README.md' if latest else 'runner/docs/skirmish/results/continuous-full-20261031/README.md'
        date = '2026-11-10' if latest else '2026-10-31'
        lines.append(f"| [{m['label']}]({link}) | {m['mean']:.2f} ± {m['sd']:.2f} | {wins} / 300 | {date} |")
    lines += ['', '- [Original October 31 archive: harness, conversations, sources and replays](README.md)',
        '- [Fable 5.2 follow-up: all five submissions, conversations, replays and validation receipts](extensions/fable52-20261110/README.md)',
        '- [Interactive leaderboard and article](https://thekevinz.com/blog/polytopia-bench)', '',
        'Verify the follow-up locally without Docker or model calls:', '', '```sh',
        'python3 benchmark/tools/verify_extension.py', '```', '',
        'The original archive has its own verifier: `python3 benchmark/tools/verify_archive.py`. Rebuild this index with `python3 benchmark/tools/render_index.py`.']
    (ROOT/'INDEX.md').write_text('\n'.join(lines)+'\n')
    print('Rendered 13-model index, 65 attempts and 23,400 games.')


if __name__ == '__main__':
    main()
