"""Markdown tables from experiments/modal_distributed_rl.py::tech_report outputs (runs/rl-18x18-v1/*.json).

    python scripts/tech_report_table.py runs/rl-18x18-v1/l18-baseline-c3-14.json [more.json ...]
"""
import json
import sys


def table(path, side='both'):
    d = json.load(open(path))
    print(f"\n{d['tag']}: {d['size']}x{d['size']}, {'sampled' if d['sampled'] else 'deterministic'}, "
          f"{d['maps_per_type']} maps x 2 seats per type, seeds {d['seed_base']}+100k")
    print('| Type | Games | W–L–trunc | Mean turns | Techs (final) | Tier-3 | Techs at T20 / T40 / T60 / T80 | Top units |')
    print('|---|---|---|---|---|---|---|---|')
    tot = dict(games=0, trunc=0, turns=0., techs=0., tier3=0.)
    for t, r in d['types'].items():
        if 'error' in r:
            print(f'| {t} | error: {r["error"][:80]} |')
            continue
        b = r['by_map'][t]
        tech, units = b[f'{side}_tech'], b[f'{side}_unit_mix']
        at = lambda T: f"{tech['by_turn'][T]['mean_techs']:.1f}" if T in tech['by_turn'] else '—'
        u = ', '.join(f'{k} {v:.0%}' for k, v in list(units.items())[:4])
        print(f"| {t} | {r['games']} | {r['wins']}–{r['losses']}–{r['truncations']} ({r['truncations'] / r['games']:.0%} cap) "
              f"| {r['mean_turns']:.0f} | {tech['final_mean']:.1f} | {tech['tier3_mean']:.1f} "
              f"| {at('20')} / {at('40')} / {at('60')} / {at('80')} | {u} |")
        tot['games'] += r['games']
        tot['trunc'] += r['truncations']
        tot['turns'] += r['mean_turns'] * r['games']
        tot['techs'] += tech['final_mean'] * r['games']
        tot['tier3'] += tech['tier3_mean'] * r['games']
    g = max(tot['games'], 1)
    print(f"| all | {tot['games']} | {tot['trunc']} cap ({tot['trunc'] / g:.0%}) | {tot['turns'] / g:.0f} "
          f"| {tot['techs'] / g:.1f} | {tot['tier3'] / g:.1f} | | |")


if __name__ == '__main__':
    for p in sys.argv[1:]:
        table(p)
