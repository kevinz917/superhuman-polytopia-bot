"""Integration checks against the separately running studio; no campaign writes."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/studio'))
from client import Client
c=Client();obs=c.start(seed=1810000222)
version=obs['state_version']
move=next(a for a in obs['legal_actions'] if a['command']['kind']==1)
c.act(move['action_id'],version)
after=c.observe();assert after['state_version']==version+1
for ident,v in [(move['action_id'],version),('-999',after['state_version'])]:
    try:c.act(ident,v)
    except RuntimeError:pass
    else:raise AssertionError('Rejected action accepted')
    assert c.observe()['state_version']==after['state_version']
end=next(a for a in after['legal_actions'] if a['command']['kind']==10)
c.act(end['action_id'],after['state_version'])
try:c.act(end['action_id'],after['state_version']+1)
except RuntimeError as e:assert 'bot' in str(e)
else:raise AssertionError('Wrong seat accepted')
while not c.observe()['your_turn']:c.opponent()
state=c.observe();public=c.request(f'/api/games/{c.game_id}?view=all');assert public['perspective']=='1'
export=c.request(f'/api/games/{c.game_id}/export');assert export['available_views']==['1'];assert 'seed' not in export['provenance']
assert all(list(f['views'])==['1'] for f in export['frames'])
Path('artifacts/restart-check.json').write_text(json.dumps({'game_id':c.game_id,'observation':state,'frames':public['frames']}))
print('PASS: legal move, stale rejection, invalid rejection, wrong-seat rejection, bot turn, fog restriction, redacted export; restart fixture saved')
