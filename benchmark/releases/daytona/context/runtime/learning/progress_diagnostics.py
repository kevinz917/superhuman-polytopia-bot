"""Descriptive public-state progress indicators, never a reward or outcome."""


def public_progress(rows, seat):
    known=set();initial=None;maximum_cities=0;initial_cities=None
    moves=reversals=captures=0;previous_move=None
    progressed=False;streak=longest=0;end_turns=0
    def observe(obs):
        nonlocal initial,initial_cities,maximum_cities,progressed
        if obs['player']!=seat:return
        state=obs['visible_state']
        visible={tile['tile'] for tile in state['tiles'] if tile['known']}
        cities=sum(city['owner']==seat for city in state['cities'])
        if initial is None:initial=len(visible);initial_cities=cities
        elif visible-known:progressed=True
        known.update(visible);maximum_cities=max(maximum_cities,cities)
    for row in rows:
        if 'game_result' in row:continue
        observe(row['observation'])
        if row.get('forfeit'):continue
        for obs in row['after']:observe(obs)
        if row['policy']!='model':continue
        action=row['selected_command'];kind=action['kind']
        if kind==1:
            edge=(action['source'],action['target']);moves+=1
            reversals+=previous_move is not None and edge==previous_move[::-1]
            previous_move=edge
        if kind==8:captures+=1;progressed=True
        if kind==10:
            end_turns+=1;streak=0 if progressed else streak+1
            longest=max(longest,streak);progressed=False
    return dict(movement_actions=moves,consecutive_movement_reversals=reversals,
        captures=captures,newly_seen_tiles=len(known)-(initial or 0),
        initial_owned_cities=initial_cities,max_owned_cities=maximum_cities,
        end_turns=end_turns,longest_end_turn_streak_without_discovery_or_capture=longest,
        interpretation='Descriptive indicators only. Waiting, combat, or economic development can be useful without discovery or capture; this is not an automatic stalled-game verdict.')
