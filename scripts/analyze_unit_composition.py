"""Unit-type and technology usage from public game traces (*.jsonl.gz).

Each trace: a header line {initial_seed, challenger_seat}, then one line per
public decision with the acting player's observation and chosen command.
Command kind 4 = summon (parameter = unit id), 7 = research (parameter = tech id).
Usage: analyze_unit_composition.py <trace.jsonl.gz> ... > summary.json
"""
import collections
import gzip
import json
import sys
from pathlib import Path

RULES = json.loads((Path(__file__).resolve().parents[1] / 'src/polytopia_bot/gpu/rules.json').read_text())
UNIT = {u['id']: u['name'] for u in RULES['units']}
TECH = {t['id']: t['name'] for t in RULES['technologies']}
TIER = {t['id']: t['tier'] for t in RULES['technologies']}
SUMMON, RESEARCH = 4, 7


def game(lines):
    header = json.loads(lines[0])
    # Match traces name the evaluated seat challenger_seat; baseline traces call it seat.
    seat = header['challenger_seat'] if 'challenger_seat' in header else header['seat']
    side = {seat: 'challenger', 3 - seat: 'opponent'}
    out = {s: dict(summons=collections.Counter(), legal=collections.Counter(), research=[], last=None) for s in side.values()}
    result, turns = 0., 0
    for raw in lines[1:]:
        step = json.loads(raw)
        if 'result' in step:
            result = step['result']['reward']
            continue
        obs = step['observation']
        who = side[obs['player']]
        v = obs['visible_state']
        turns = max(turns, v['turn'])
        out[who]['last'] = v
        for kind in {a['command']['parameter'] for a in obs['legal_actions'] if a['command']['kind'] == SUMMON}:
            out[who]['legal'][UNIT.get(kind, kind)] += 1
        c = step['action']['command']
        if c['kind'] == SUMMON:
            out[who]['summons'][UNIT.get(c['parameter'], c['parameter'])] += 1
        elif c['kind'] == RESEARCH:
            out[who]['research'].append((v['turn'], TECH.get(c['parameter'], c['parameter'])))
    report = dict(seed=header['initial_seed'], seat=seat, turns=turns,
                  outcome='win' if result > 0 else 'loss' if result < 0 else 'unresolved')
    for who, d in out.items():
        player = seat if who == 'challenger' else 3 - seat
        v = d['last'] or dict(technology=[], units=[], cities=[])
        own = [u for u in v['units'] if u['owner'] == player]
        report[who] = dict(
            summons=dict(d['summons']), summon_legal_decisions=dict(d['legal']), research=d['research'],
            techs_final=[TECH.get(t, t) for t in v['technology']],
            max_tier=max((TIER.get(t, 0) for t in v['technology']), default=0),
            army_final=dict(collections.Counter(UNIT.get(u['kind'], u['kind']) for u in own)),
            cities_final=sum(1 for c in v['cities'] if c['owner'] == player))
    return report


def aggregate(games):
    def side(who, subset):
        n = len(subset) or 1
        summons = collections.Counter()
        legal = collections.Counter()
        tech_freq = collections.Counter()
        for g in subset:
            summons.update(g[who]['summons'])
            legal.update({k: 1 for k in g[who]['summon_legal_decisions']})
            tech_freq.update(set(g[who]['techs_final']))
        total = sum(summons.values()) or 1
        first_research = [g[who]['research'][0][0] for g in subset if g[who]['research']]
        return dict(
            games=len(subset),
            summon_share={k: round(v / total, 3) for k, v in summons.most_common()},
            summons_per_game=round(sum(summons.values()) / n, 2),
            games_where_unit_was_ever_summonable={k: v for k, v in legal.most_common()},
            games_with_non_warrior_summon=sum(any(k != 'Warrior' for k in g[who]['summons']) for g in subset),
            distinct_unit_types_per_game=round(sum(len(g[who]['summons']) for g in subset) / n, 2),
            research_actions_per_game=round(sum(len(g[who]['research']) for g in subset) / n, 2),
            techs_final_per_game=round(sum(len(g[who]['techs_final']) for g in subset) / n, 2),
            max_tier_distribution=dict(collections.Counter(g[who]['max_tier'] for g in subset)),
            median_first_research_turn=sorted(first_research)[len(first_research) // 2] if first_research else None,
            tech_owned_share={k: round(v / n, 3) for k, v in tech_freq.most_common()},
            cities_final_mean=round(sum(g[who]['cities_final'] for g in subset) / n, 2))
    wins = [g for g in games if g['outcome'] == 'win']
    losses = [g for g in games if g['outcome'] == 'loss']
    return dict(games=len(games), wins=len(wins), losses=len(losses),
                mean_turns=round(sum(g['turns'] for g in games) / max(len(games), 1), 1),
                challenger=side('challenger', games), opponent=side('opponent', games),
                challenger_in_wins=side('challenger', wins), challenger_in_losses=side('challenger', losses))


if __name__ == '__main__':
    games = [game(gzip.open(p, 'rt').read().splitlines()) for p in sys.argv[1:]]
    print(json.dumps(dict(aggregate=aggregate(games), games=games), indent=2))
