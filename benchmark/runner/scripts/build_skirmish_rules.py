"""Build the identical detailed offline Markdown rule book from the actual engine sources."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def build():
    sections = ['''# Complete Polytopia Skirmish rule book

This is the implementation-defined rule book for polytopia-2.17.3, Bardur 1v1,
14x14, drylands/lakes/continents. It contains the bot protocol, current rules
alignment explanation, every rule-table entry, full Rust implementation, map
variants, and public observation serializer. The legal action catalog is always
authoritative for what is available now. The engine source and tables resolve
mechanical edge cases; this benchmark does not substitute older research rules.

## How to read this book

Start with the protocol and alignment's gameplay sections, then consult the tables
for numeric IDs/stats. Rust masks define legality; core dispatch defines actions;
military/pathfinding cover damage, retaliation and movement; economy/structures/
capture cover population, income, construction and city rewards; discovery and
infiltration describe exploration, ruins and hidden units; diplomacy describes
relations and peace. All these sources appear below without truncation.

Bardur starts with Hunting, one capital, a Warrior and five stars. Expand by
exploring villages and capturing them, grow cities through resources and buildings,
research unlocks, recruit armies within city capacity, and defeat the opponent by
capturing their last city. End turn collects income and refreshes units. Fog limits
observations; do not assume unseen enemy resources, units, or technologies.

Feedback/final map seeds are controlled by the referee. Layout generation is an
approximation of map distributions, separate from transition rules. No checkpoint,
opponent policy or held-out map allocation appears in this book.
''']
    sections.append('''
## Gameplay mechanics reference

### Actions, economy and cities

One observation is supplied per atomic action, not per whole turn. You may take many
actions before end turn. Mandatory city rewards block ordinary actions until selected.
Stars pay for technologies, units, resources and construction. Harvesting an animal,
fish or fruit costs two stars and adds one population when the required technology
is known. Farms/mines and population multipliers use the exact costs and prerequisites
in the tables and structure implementation below. Population progress to the next
level is current city level + 1. Level growth increases production and triggers a
reward. City capacity and occupied city tiles constrain recruitment.

City reward command.parameter codes: 1 City Wall, 2 Park, 3 Workshop, 4 Explorer,
5 Border Growth, 6 Giant, 7 Resources (+5 stars), 8 Population (+3).
At level 2 choose Workshop/Explorer; level 3 Wall/Resources; level 4 Border/Population;
subsequent levels Park/Giant. Consult the actual legal catalog for each pending choice.
Capture commands transfer villages/cities; see capture.rs for unit action flags,
territory transfer, home-unit capacity and last-city victory. Capturing and recruitment
are separate from moving; arriving at a city alone does not execute capture.

Research price is 4 + ownedCityCount * technologyTier. Philosophy multiplies by
0.66666 with ceiling. Only reachable, unknown and affordable technologies appear.
Researching before expansion can therefore cost less. Technologies expose unit,
building and ability unlocks, but some table entries belong to tribes not playable
in this Bardur track. Follow the successor graph from known Basic/Hunting and the
legal catalog; a listed unit is not necessarily recruitable in this game.

### Combat and unit state

Health affects combat strength. Let af=attackStat*attackerHP/attackerMaxHP and
df=defenseStat*defenderHP/defenderMaxHP*defenseBonus. Normal damage is rounded
(af/(af+df))*attackStat*4.5; retaliation, if the defender survives and can retaliate,
uses (df/(af+df))*defenseStat*4.5. Exact rounding, range, terrain/city defense,
Surprise, Convert, splash and advance rules are in kernels.rs/military.rs.
Veterancy adds five maximum health; promotion heals. Recovery restores four health
in friendly territory and two elsewhere; Heal Others is a separate action.
Dash permits attacking after a move; Escape permits moving after attacking; Persist
allows kill chains. Non-Dash units generally cannot move then attack. Newly recruited,
embarked, healed or exhausted units follow the explicit moved/attacked flags.
Mind Bender conversion is not ordinary zero-damage combat. Attack outcomes include
retaliation and possible occupation of the defeated unit's tile.

### Geography, fog and navy

Distance is Chebyshev on the square grid (diagonals count). Movement uses terrain,
roads, mountains, water, bridges, occupied tiles and visible zone of control.
Entering a forest ends movement unless traveling road-to-road; mountain entry and
water require their unlocks. Exact reachability comes from the supplied legal moves.
Forests/mountains and city fortification also affect combat defense. Exploration
reveals tiles, public units/resources and contacts; public memory persists while
current visibility and remembered facts remain separate. Never infer that an unknown
tile is empty. Ruins, explorers, giant pushes and infiltration have seeded outcomes.

Ports are built in eligible owned shallow water. Moving a land unit into a port
embarks it; upgrades require the appropriate tech, funds and friendly territory.
Naval types have their own movement/attack/skills; carried land-unit health and cost
are tracked. Disembarking restores the passenger. Continents typically requires
naval transport; lakes mixes land and water. Diplomacy and peace affect hostility,
movement, embassies and cooldowns as specified by diplomacy.rs.

The following readable tables mirror every entry, including unreachable special-tribe
entries for completeness. The full JSON and source appendices retain every field.
''')
    rules = json.loads((ROOT/'src/polytopia_bot/gpu/rules.json').read_text())
    columns = {'units':['id','name','cost','health','attack','defense','movement','range','skills'],
               'technologies':['id','name','tier','requires','next','units','structures'],
               'structures':['id','name','cost','population','stars','terrain','resource','adjacent','per_city','per_tribe']}
    for group, fields in columns.items():
        lines = ['\n## '+group.title(), '', '| '+' | '.join(fields)+' |',
                 '| '+' | '.join('---' for _ in fields)+' |']
        for row in rules[group]:
            lines.append('| '+' | '.join(str(row.get(key,'')) for key in fields)+' |')
        sections.append('\n'.join(lines)+'\n')
    files = [ROOT/'skirmish/mini/PROTOCOL.md', ROOT/'docs/rules-alignment-2026-10-27.md',
             ROOT/'src/polytopia_bot/gpu/rules.json']
    files += sorted((ROOT/'engine-rs/src').glob('*.rs'))
    files += [ROOT/'src/polytopia_bot/maps/variants.py', ROOT/'src/polytopia_bot/maps/stream.py',
              ROOT/'src/polytopia_bot/learning/adapter.py', ROOT/'src/polytopia_bot/learning/interface.py',
              ROOT/'src/polytopia_bot/learning/memory.py', ROOT/'src/polytopia_bot/environment.py',
              ROOT/'src/polytopia_bot/studio/environment.py']
    for path in files:
        name = str(path.relative_to(ROOT))
        text = path.read_text()
        # The old learning adapter also contains a retired JAX simulator; include only rules-free serializer.
        if name.endswith('learning/adapter.py'):
            text = text[text.index('def serialize('):text.index('class GPUEnvironment')]
        language = {'.py':'python','.rs':'rust','.json':'json','.md':'markdown'}[path.suffix]
        sections.append(f'\n## {name}\n\nSource SHA-256: {hashlib.sha256(path.read_bytes()).hexdigest()}\n\n'
                        f'````{language}\n{text}\n````\n')
    return '\n'.join(sections)

if __name__ == '__main__':
    output = ROOT/'skirmish/mini/RULES.md'
    output.write_text(build())
    print(f'{output}: {output.stat().st_size} bytes')
