"""Read-only, Modal-only evidence for the Qwen3.5 thinking on/off + Qwen3-Instruct comparison.

  modal run --env staging scripts/analyze_thinking_comparison.py --out evidence.json

Thinking-on games are stitched across the original run and its continuation segments
(continuation, continuation-2) using each segment's entries.json trace_rows. Games that
were still running when the user stopped the run are reported as ``stopped_unfinished``
with their last observed state; they are never counted as wins or losses.
"""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name('polytopia-sft0-training', environment_name='staging').read_only()
app = modal.App('polytopia-thinking-comparison-evidence')
image = modal.Image.debian_slim(python_version='3.12').add_local_file(
    ROOT/'src/polytopia_bot/gpu/rules.json', '/rules.json')
RUN = 'qwen35-thinking-30paired-v1'
INSTRUCT = 'qwen3-instruct-2507-30paired-v2'
FAM = {1: 'move', 2: 'attack', 3: 'ability', 4: 'recruit', 5: 'harvest', 6: 'build', 7: 'research',
       8: 'capture', 9: 'city_reward', 10: 'end_turn', 11: 'resign'}
KEYWORDS = {
    'self_doubt_wait': ['Wait,', 'Wait.', 'wait,'],
    'format_check': ['action_id', 'Return only', 'JSON', 'no prose'],
    'enemy_or_threat': ['enemy', 'threat', 'opponent'],
    'capital_or_city_attack': ['capital', 'capture city', 'take the city', 'siege'],
    'future_plan': ['next turn', 'later', 'future', 'plan', 'long-term', 'long term'],
    'economy': ['economy', 'stars', 'income', 'population'],
    'expansion': ['expand', 'village', 'settlement'],
    'defense': ['defend', 'defense', 'protect', 'safe'],
}


def segments_for(root, shard):
    """[(segment folder holding on/, entries trace_rows of the NEXT segment or None)]"""
    import json
    segs = [root/'shards'/str(shard)/'on', root/'continuation'/'shards'/str(shard)/'on',
            root/'continuation-2'/'shards'/str(shard)/'on']
    entries = []
    for nxt in ['continuation', 'continuation-2']:
        p = root/nxt/'shards'/str(shard)/'entries.json'
        entries.append({e['job']['id']: e['trace_rows'] for e in json.loads(p.read_text())} if p.exists() else {})
    return segs, entries


@app.function(image=image, cpu=2, memory=8192, timeout=3600, volumes={'/runs': volume})
def analyze_game(arm, job):
    import json, zlib
    from collections import Counter, defaultdict
    from pathlib import Path
    rules = json.load(open('/rules.json'))
    NAMES = {4: {u['id']: u['name'] for u in rules['units']}, 7: {t['id']: t['name'] for t in rules['technologies']},
             6: {s['id']: s['name'] for s in rules['structures']},
             3: {1: 'burn_forest', 2: 'clear_forest', 3: 'grow_forest', 4: 'destroy', 7: 'recover', 8: 'disband',
                 9: 'heal', 16: 'promote', 18: 'break_peace', 19: 'respond_peace', 20: 'embassy', 21: 'offer_peace',
                 22: 'remove_embassy'}}
    jid, seat = job['id'], job['seat']

    def rows(path, limit=None):
        if not path.exists():
            return
        dec, buf, n = zlib.decompressobj(31), b'', 0
        with open(path, 'rb') as f:
            while chunk := f.read(1 << 20):
                try:
                    buf += dec.decompress(chunk)
                except zlib.error:
                    break
                *lines, buf = buf.split(b'\n')
                for line in lines:
                    if limit is not None and n >= limit:
                        return
                    try:
                        r = json.loads(line)
                    except ValueError:
                        return
                    n += 1
                    yield r

    result, parts = None, []
    if arm == 'instruct':
        g = Path('/runs')/INSTRUCT/'instruct'/'games'
        parts = [(g/(jid+'.jsonl.gz'), None)]
        listing = json.loads((g/'games.json').read_text()) if (g/'games.json').exists() else json.loads((g/'report.json').read_text())['games']
        result = {x['id']: x for x in listing}.get(jid)
    elif arm == 'off':
        root = Path('/runs')/RUN
        for s in (0, 1):
            p = root/'shards'/str(s)/'off'/'games'/(jid+'.jsonl.gz')
            if p.exists():
                parts = [(p, None)]
                result = {x['id']: x for x in json.loads((p.parent/'games.json').read_text())}.get(jid)
    else:
        root = Path('/runs')/RUN
        for s in (0, 1):
            if not (root/'shards'/str(s)/'on'/'games'/(jid+'.jsonl.gz')).exists():
                continue
            segs, entries = segments_for(root, s)
            for k, seg in enumerate(segs):
                done = {x['id']: x for x in json.loads((seg/'games'/'games.json').read_text())} if (seg/'games'/'games.json').exists() else {}
                nxt = entries[k] if k < len(entries) else {}
                if jid in done:
                    parts.append((seg/'games'/(jid+'.jsonl.gz'), None)); result = dict(done[jid], final_segment=k); break
                if jid in nxt:
                    parts.append((seg/'games'/(jid+'.jsonl.gz'), nxt[jid])); continue
                if (seg/'games'/(jid+'.jsonl.gz')).exists():
                    parts.append((seg/'games'/(jid+'.jsonl.gz'), None))   # stopped by user mid-segment
                break
    status = (result['status'], result['reason'], result['reward']) if result else ('stopped_unfinished', 'stopped_by_user', None)

    out = dict(arm=arm, id=jid, seed=job['seed'], seat=seat, status=status, segments=len(parts))
    acts, spend, gain = Counter(), Counter(), Counter()
    names = defaultdict(Counter); techs = []
    turns = {}; owned = {1: set(), 2: set()}
    combat = Counter(); end_turn = dict(n=0, stars=0, with_5plus=0, skipped=Counter())
    first_of_family = [0, 0, 0.]   # hits, eligible, expected-by-chance
    forced = dict(n=0, tokens=0)
    reasoning = dict(n=0, tokens=[], by_family=defaultdict(list), keywords=Counter(), closed=0)
    samples = []
    for path, limit in parts:
        for r in rows(path, limit):
            if 'selected_command' not in r:
                continue
            obs = r['observation']; actor = obs['player']; vs = obs['visible_state']; turn = vs['turn']
            cmd = r['selected_command']; fam = FAM.get(cmd['kind'], str(cmd['kind']))
            who = 'model' if actor == seat else 'opp'
            after = r['after']
            if cmd['kind'] == 2:
                tgt = cmd['target']
                hit = [u for u in vs['units'] if u['tile'] == tgt and u['owner'] == 3-actor]
                alive = [u for u in after[actor-1]['visible_state']['units'] if u['tile'] == tgt and u['owner'] == 3-actor]
                combat[f'{who}_attacks'] += 1
                if hit and not alive:
                    combat[f'{who}_kills'] += 1
            if cmd['kind'] == 8:
                combat[f'{who}_captures'] += 1
            now = {p: {c['tile'] for c in after[p-1]['visible_state']['cities'] if c['owner'] == p} for p in (1, 2)}
            for p in (1, 2):
                taken = now[p] & owned[3-p]
                if taken:
                    combat[('model' if p == seat else 'opp')+'_takes_city'] += len(taken)
            owned = now
            me, op = after[seat-1]['visible_state'], after[2-seat]['visible_state']
            turns[turn] = (sum(c['owner'] == seat for c in me['cities']), sum(c['owner'] == 3-seat for c in op['cities']),
                           sum(u['owner'] == seat for u in me['units']), sum(u['owner'] == 3-seat for u in op['units']),
                           me['stars'], sum(c['level'] for c in me['cities'] if c['owner'] == seat))
            if who != 'model':
                continue
            acts[fam] += 1
            delta = vs['stars'] - after[seat-1]['visible_state']['stars']
            if delta > 0:
                spend[fam] += delta
            elif delta < 0:
                gain[fam] += -delta
            if cmd['kind'] in NAMES:
                nm = str(NAMES[cmd['kind']].get(cmd['parameter'], cmd['parameter']))
                names[fam][nm] += 1
                if cmd['kind'] == 7:
                    techs.append(nm)
            legal = obs['legal_actions']
            legal_fams = Counter(FAM.get(a['command']['kind']) for a in legal)
            same = [a for a in legal if a['command']['kind'] == cmd['kind']]
            if len(same) > 1:
                first_of_family[1] += 1; first_of_family[2] += 1/len(same)
                if dict(same[0]['command']) == dict(cmd):
                    first_of_family[0] += 1
            if fam == 'end_turn':
                end_turn['n'] += 1; end_turn['stars'] += vs['stars']; end_turn['with_5plus'] += vs['stars'] >= 5
                for k, v in legal_fams.items():
                    if k not in ('end_turn', 'resign') and v:
                        end_turn['skipped'][k] += 1
            diag = r.get('diagnostic') or []
            gen = diag[-1]['generation'] if diag and isinstance(diag, list) and 'generation' in diag[-1] else None
            ntok = len(gen['response_token_ids']) if gen else 0
            if set(legal_fams) <= {'end_turn', 'resign'}:
                forced['n'] += 1; forced['tokens'] += ntok
            if arm == 'on' and gen:
                text = gen['text']; think = text.split('</think>')[0]
                reasoning['n'] += 1; reasoning['tokens'].append(ntok); reasoning['by_family'][fam].append(ntok)
                reasoning['closed'] += '</think>' in text
                for key, words in KEYWORDS.items():
                    if any(w in think for w in words):
                        reasoning['keywords'][key] += 1
                reasoning['keywords']['wait_count_total'] += think.count('Wait')
                if (len(samples) < 6 and (fam in ('capture', 'attack', 'end_turn', 'research') or set(legal_fams) <= {'end_turn', 'resign'})
                        and turn in (3, 12, 20, 28) and not any(s['family'] == fam for s in samples)):
                    samples.append(dict(turn=turn, family=fam, tokens=ntok, legal_families=dict(legal_fams),
                                        cities=turns[turn][:2], head=think[:700], tail=think[-1100:]))
    last = max(turns) if turns else None
    out.update(final_round=last, model_moves=sum(acts.values()), actions=dict(acts), spend=dict(spend), gain=dict(gain),
               names={k: dict(v) for k, v in names.items()}, first5_techs=techs[:5], combat=dict(combat),
               end_turn=dict(end_turn, skipped=dict(end_turn['skipped'])), first_of_family=first_of_family,
               forced=forced, turns={t: turns[t] for t in sorted(turns)},
               peak_cities=max((v[0] for v in turns.values()), default=0), last_state=turns.get(last))
    if arm == 'on':
        tk = sorted(reasoning['tokens'])
        out['reasoning'] = dict(n=reasoning['n'], closed=reasoning['closed'], total_tokens=sum(tk),
                                median_tokens=tk[len(tk)//2] if tk else None,
                                by_family_median={k: sorted(v)[len(v)//2] for k, v in reasoning['by_family'].items()},
                                by_family_total={k: sum(v) for k, v in reasoning['by_family'].items()},
                                keywords=dict(reasoning['keywords']))
        out['samples'] = samples
    return out


@app.local_entrypoint()
def main(out: str = 'evidence.json'):
    import json
    protocol = json.loads(b''.join(volume.read_file(RUN+'/protocol.json')))
    work = [(arm, j) for arm in ['on', 'off', 'instruct'] for j in protocol['jobs']]
    results = list(analyze_game.starmap(work))
    Path(out).write_text(json.dumps(results))
    print('games analyzed:', len(results))
