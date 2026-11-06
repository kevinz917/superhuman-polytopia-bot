"""Early-game openings (first EARLY turns) from public game traces (*.jsonl.gz).

Per game and side: summons and research in turns [0, EARLY), the first
non-Warrior unit (whole game), early action-family counts, cities at turn EARLY,
army at turn 10, and which non-Warrior units were already summonable early.
classify() turns one side's record into an opening type; summarize() counts
types and signatures over games. Command kinds follow the public interface:
1 move, 2 attack, 3 ability, 4 summon, 5 harvest, 6 build, 7 research, 8 capture.
Usage: analyze_early_game.py <trace.jsonl.gz> ... > early.json
"""
import collections
import gzip
import json
import sys
from pathlib import Path

RULES = json.loads((Path(__file__).resolve().parents[1] / 'src/polytopia_bot/gpu/rules.json').read_text())
UNIT = {u['id']: u['name'] for u in RULES['units']}
TECH = {t['id']: t['name'] for t in RULES['technologies']}
FAMILY = {1: 'move', 2: 'attack', 3: 'ability', 4: 'summon', 5: 'harvest', 6: 'build', 7: 'research',
          8: 'capture', 9: 'city_reward', 10: 'end_turn', 11: 'resign'}
SUMMON, RESEARCH = 4, 7
EARLY, ARMY_TURN = 8, 10

UNIT_OPENING = {'Rider': 'rider-rush', 'Knight': 'rider-rush', 'Archer': 'archer',
                'Defender': 'defender-turtle', 'Catapult': 'siege', 'Swordsman': 'heavy',
                'MindBender': 'mindbender'}
# Techs that unlock a non-Warrior land unit.
UNLOCKS = ('Riding', 'Chivalry', 'Archery', 'Strategy', 'Mathematics', 'Smithery', 'Philosophy')


def early(lines, window=EARLY):
    header = json.loads(lines[0])
    seat = header['challenger_seat'] if 'challenger_seat' in header else header['seat']
    side = {seat: 'challenger', 3 - seat: 'opponent'}
    out = {s: dict(summons=[], research=[], first_non_warrior=None, families=collections.Counter(),
                   early_summonable=set(), cities_at_window=None, army_at_10=None)
           for s in side.values()}
    for raw in lines[1:]:
        step = json.loads(raw)
        if 'result' in step:
            continue
        obs = step['observation']
        who, player = side[obs['player']], obs['player']
        d = out[who]
        v = obs['visible_state']
        turn = v['turn']
        if turn >= window and d['cities_at_window'] is None:
            d['cities_at_window'] = sum(1 for c in v['cities'] if c['owner'] == player)
        if turn >= ARMY_TURN and d['army_at_10'] is None:
            d['army_at_10'] = dict(collections.Counter(UNIT.get(u['kind'], u['kind'])
                                                      for u in v['units'] if u['owner'] == player))
        c = step['action']['command']
        if turn < window:
            d['families'][FAMILY.get(c['kind'], c['kind'])] += 1
            for a in obs['legal_actions']:
                if a['command']['kind'] == SUMMON and UNIT.get(a['command']['parameter']) != 'Warrior':
                    d['early_summonable'].add(UNIT.get(a['command']['parameter'], a['command']['parameter']))
        if c['kind'] == SUMMON:
            unit = UNIT.get(c['parameter'], c['parameter'])
            if turn < window:
                d['summons'].append((turn, unit))
            if unit != 'Warrior' and d['first_non_warrior'] is None:
                d['first_non_warrior'] = (turn, unit)
        elif c['kind'] == RESEARCH and turn < window:
            d['research'].append((turn, TECH.get(c['parameter'], c['parameter'])))
    for d in out.values():
        d['families'] = dict(d['families'])
        d['early_summonable'] = sorted(d['early_summonable'])
    return dict(seed=header['initial_seed'], seat=seat, **out)


def classify(side):
    """Opening type of one side's early record."""
    for _, unit in side['summons']:
        if unit in UNIT_OPENING:
            return UNIT_OPENING[unit]
    for _, tech in side['research']:
        if tech in UNLOCKS:
            return f'warriors + {tech} tech'
    warriors = sum(u == 'Warrior' for _, u in side['summons'])
    return 'warrior-rush' if warriors >= 3 else 'economy-first'


def signature(side):
    """Finer opening: first two techs + early non-Warrior units in order."""
    techs = '>'.join(t for _, t in side['research'][:2]) or '-'
    units = [u for _, u in side['summons'] if u != 'Warrior']
    return f"{techs} | {'+'.join(units[:3]) or 'warriors'}"


def summarize(records):
    n = len(records) or 1
    types = collections.Counter(classify(r) for r in records)
    sigs = collections.Counter(signature(r) for r in records)
    first_tech = collections.Counter((r['research'][0][1] if r['research'] else '-') for r in records)
    early_units = collections.Counter(u for r in records for _, u in r['summons'])
    total = sum(early_units.values()) or 1
    fnw = [r['first_non_warrior'] for r in records if r['first_non_warrior']]
    fnw_unit = collections.Counter(u for _, u in fnw)
    turns = sorted(t for t, _ in fnw)
    summonable = collections.Counter(u for r in records for u in r['early_summonable'])
    cities = [r['cities_at_window'] for r in records if r['cities_at_window'] is not None]
    fam = collections.Counter()
    for r in records:
        fam.update(r['families'])
    return dict(
        sides=len(records),
        opening_types=dict(types.most_common()), distinct_opening_types=len(types),
        distinct_signatures=len(sigs), top_signatures=dict(sigs.most_common(6)),
        first_tech=dict(first_tech.most_common()),
        early_summons_per_game=round(sum(early_units.values()) / n, 2),
        early_unit_share={k: round(v / total, 3) for k, v in early_units.most_common()},
        games_with_non_warrior_ever=len(fnw),
        first_non_warrior_unit=dict(fnw_unit.most_common()),
        first_non_warrior_turn_median=turns[len(turns) // 2] if turns else None,
        early_non_warrior_summonable_games=dict(summonable.most_common()),
        cities_at_turn8_mean=round(sum(cities) / len(cities), 2) if cities else None,
        early_actions_per_game={k: round(v / n, 1) for k, v in fam.most_common()})


if __name__ == '__main__':
    games = [early(gzip.open(p, 'rt').read().splitlines()) for p in sys.argv[1:]]
    print(json.dumps(dict(challenger=summarize([g['challenger'] for g in games]),
                          opponent=summarize([g['opponent'] for g in games]), games=games), indent=2))
