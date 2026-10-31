"""One public action boundary for the browser, external agents and trained bots.

The local studio never writes to campaign directories. Full spectator frames stay
server-side during human/agent games. Only RustEnvironment (engine-rs) owns game mechanics.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import base64
import gzip
import hashlib
import json
import secrets
import threading
import time
import uuid

from polytopia_bot.ruleset import RULESET_ID

SCHEMA = 'bardur-replay-v1'
ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).with_name('static')
RULES = json.loads((Path(__file__).parents[1] / 'gpu/rules.json').read_text())
TERRAIN = {0:'Unknown',1:'Shallow water',2:'Ocean',3:'Field',4:'Mountain',5:'Forest',6:'Ice',7:'Wetland',8:'Flooded field'}
RESOURCE = {0:'None',1:'Animal',2:'Crop',3:'Fish',4:'Whale',5:'Metal',6:'Fruit',7:'Fungi',8:'Starfish'}
ABILITY = {1:'Burn forest',2:'Clear forest',3:'Grow forest',4:'Destroy',7:'Recover',8:'Disband',9:'Heal others',16:'Promote',18:'Break peace',19:'Respond to peace',20:'Establish embassy',21:'Offer peace',22:'Remove embassy'}
REWARD = {1:'City wall',2:'Park',3:'Workshop',4:'Explorer',5:'Border growth',6:'Super unit',7:'Resources',8:'Population growth'}


def dump(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(dump(value))
    tmp.replace(path)


def compact(obs):
    d = asdict(obs)
    # Legal actions remain on the observation endpoint, not in every replay frame.
    d.pop('remembered_state')
    d.pop('legal_actions')
    return d


def action_label(command):
    k,p = command['kind'],command['parameter']
    names = lambda group: {x['id']:x['name'] for x in RULES[group]}
    if k == 3:
        if p == 19: return 'Accept peace' if command.get('target') == 1 else 'Reject peace'
        return ABILITY.get(p, f'Ability {p}')
    if k == 4: return 'Train ' + names('units').get(p, str(p))
    if k == 6: return 'Build ' + names('structures').get(p, str(p))
    if k == 7: return 'Research ' + names('technologies').get(p, str(p))
    if k == 9: return REWARD.get(p, f'City reward {p}')
    return {1:'Move',2:'Attack',5:'Harvest',8:'Capture',10:'End turn',11:'Resign'}.get(k,f'Action {k}')


def labeled(action):
    a = asdict(action) if not isinstance(action,dict) else dict(action)
    return {**a, 'label':action_label(a['command'])}


def with_hud(obs, env, player):
    import jax
    # City borders are drawn on the public map; expose membership of known tiles only.
    ruling=jax.device_get(env.state.tiles.city).tolist()
    tiles=obs['visible_state']['tiles']
    territory={}
    for c in obs['visible_state']['cities']:
        if c['owner']:territory[str(c['tile'])]=[t['tile'] for t in tiles if t['known'] and ruling[t['tile']]==c['tile']]
    # Score and city count are public in the mobile leaderboard, fog or not.
    structure,owner=jax.device_get((env.state.tiles.structure,env.state.tiles.owner))
    scores=jax.device_get(env.state.players.score).tolist()
    obs['hud']={'score':int(env.state.players.score[player-1]),
                'income':env.income(player),
                'territory':territory,
                'players':[{'player':p,'score':int(scores[p-1]),'cities':int(((structure==1)&(owner==p)).sum())}
                           for p in range(1,len(scores)+1)]}
    return obs


def world_view(env):
    # Rendering-only omniscience. Never passed to a policy, never stepped.
    import jax
    import numpy as np
    from polytopia_bot.learning.adapter import serialize
    s = env.state
    render = s._replace(tiles=s.tiles._replace(explored=np.ones_like(s.tiles.explored)),
                       units=s.units._replace(invisible=np.zeros_like(s.units.invisible)))
    d = compact(serialize(env.view(int(s.current_player), render), size=env.size, legal=[]))
    cities = jax.device_get(s.cities)
    for c in d['visible_state']['cities']:
        for name in ('level','progress','population','production'):
            c[name] = int(getattr(cities,name)[c['tile']])
    return d


def visible_animation(previous, current, event):
    """Allow only motion/combat whose endpoints and acting units are public.

    No action ID, ability parameter, hidden source, or policy metric is emitted.
    Uses the already-redacted observations, never private engine state.
    """
    command=event.get('action',{}).get('command',{})
    kind,source,target=(command.get(k) for k in ('kind','source','target'))
    if kind not in (1,2) or not isinstance(source,int) or not isinstance(target,int):return None
    before,after=previous['visible_state'],current['visible_state']
    if any(i<0 or i>=len(v['tiles']) or not v['tiles'][i].get('known')
           for i in (source,target) for v in (before,after)):return None
    old={u['tile']:u for u in before['units']}
    new={u['tile']:u for u in after['units']}
    actor=old.get(source)
    if not actor or actor['owner']!=event['player']:return None
    if kind==1 and (source in new or new.get(target,{}).get('owner')!=actor['owner']):return None
    if kind==2 and (target not in old or old[target]['owner']==actor['owner']):return None
    return {'kind':kind,'source':source,'target':target}


class Conflict(ValueError):
    pass


# Leak-free robust turn search (docs/test-time-search-c3-2026-10-19.md): K=16 candidate turns, each scored over
# 64 sampled opponent replies with sequential halving, planned on worlds sampled from the bot's own view.
SEARCH = dict(lanes=16, depth=1, rollout_temperature=1., replies=64, determinize=True, parallel=True)


class Studio:
    def __init__(self, storage, checkpoint, *, name='Bardur · Champion', short='Champion',
                 training='Self-play PPO from random weights', max_games=8, search_workers=None, opponents=None):
        self.storage = Path(storage).resolve()
        self.storage.mkdir(parents=True, exist_ok=True)
        (self.storage/'replays').mkdir(exist_ok=True)
        self.checkpoint = Path(checkpoint).resolve()
        self.meta = json.loads(self.checkpoint.with_suffix('.json').read_text())
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        if digest != self.meta['sha256']: raise ValueError('checkpoint SHA-256 mismatch')
        self.short, self.max_games = short, max_games
        self.bot = {'name':name,'checkpoint_sha256':digest,
                    'training':training, 'mode':'deterministic',
                    'parameters':self.meta['parameter_elements'], 'ruleset':RULESET_ID}
        # Named rival checkpoints a watch game can seat against the served bot (raw policy, never searched).
        self.opponents, self.opponent_policies = {}, {}
        for label, path in (opponents or {}).items():
            path = Path(path).resolve()
            meta = json.loads(path.with_suffix('.json').read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
                raise ValueError(f'opponent {label}: checkpoint SHA-256 mismatch')
            self.opponents[str(label)] = {'checkpoint': path, 'checkpoint_sha256': meta['sha256'],
                                          'parameters': meta['parameter_elements']}
        if self.opponents: self.bot['opponents'] = list(self.opponents)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='studio-engine')
        self.lock = threading.RLock()
        self.games, self.jobs, self.policies, self.replays = {}, {}, {}, {}
        self.featured = None  # replay id the interface opens by default
        self.ready = False
        # search_workers=None: no search mode; otherwise the opponent can switch to TurnSearcher per game.
        self.search_workers, self.searcher, self.search_ready = search_workers, None, False
        if search_workers is not None:
            self.bot['search'] = {**{k:v for k,v in SEARCH.items() if k!='parallel'},'workers':search_workers}
        for path in (self.storage/'replays').glob('*.json.gz'):
            try:
                with gzip.open(path,'rt') as f: replay=json.load(f)
                self.replays[path.stem.removesuffix('.json')] = replay
            except (ValueError,OSError): pass

    def submit(self, fn):
        ident = uuid.uuid4().hex
        self.jobs[ident] = {'id':ident,'status':'queued'}
        def execute():
            self.jobs[ident]['status']='running'
            try:
                value=fn()
                self.jobs[ident].update(status='done', result=value)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.jobs[ident].update(status='error',error=str(e))
        self.pool.submit(execute)
        return {'job_id':ident}

    def warmup(self):
        from polytopia_bot.studio.environment import RustEnvironment
        from polytopia_bot.rl.policy import StructuredPolicy
        env=RustEnvironment(14)
        obs=env.reset(seed=1810000001,settings={})
        policy=StructuredPolicy(self.checkpoint, seed=101, deterministic=True)
        choice=policy.select(obs)
        env.step(action_id=choice.action.action_id,expected_state_version=obs.state_version)
        self.policies[1]=policy
        self.policies[2]=StructuredPolicy(self.checkpoint,seed=102,deterministic=True)
        self.policies[2].select(env.observe(int(env.state.current_player)))
        env.close()
        self.ready=True
        return {'ready':True}

    def opponent_policy(self, label):
        """Deterministic raw policy of a named rival checkpoint, compiled on first use (engine pool only)."""
        if label not in self.opponent_policies:
            from polytopia_bot.rl.policy import StructuredPolicy
            self.opponent_policies[label] = StructuredPolicy(self.opponents[label]['checkpoint'], seed=103, deterministic=True)
        return self.opponent_policies[label]

    def warm_search(self):
        """Builds and compiles the searcher. Runs in the engine pool: compiling patches engine modules."""
        if self.searcher is not None or self.search_workers is None:return {'search_ready':self.search_ready}
        import jax
        from polytopia_bot.studio.environment import RustEnvironment
        from polytopia_bot.rl.distributed import _template_params
        from polytopia_bot.rl.runner import load_checkpoint
        from polytopia_bot.rl.search import TurnSearcher
        start=time.monotonic()
        params=load_checkpoint(self.checkpoint,_template_params(14))
        searcher=TurnSearcher(params,size=14,workers=self.search_workers or None,**SEARCH)
        env=RustEnvironment(14);env.reset(seed=1810000001,settings={})
        searcher.compile(env.state,jax.random.key(0))
        env.close()
        self.searcher,self.search_ready=searcher,True
        print(f'Search compiled in {time.monotonic()-start:.0f} s',flush=True)
        return {'search_ready':True}

    def plan_turn(self,g):
        """Searches the rest of the bot's turn from the live state; returns its sub-actions as catalog IDs."""
        import jax
        import numpy as np
        from polytopia_bot.rl.actions import offsets, shapes
        if self.searcher is None:self.warm_search()
        env=g['env'];version=int(env.state.state_version)
        start=time.monotonic()
        final,plans=self.searcher.play_turn(env.state,jax.random.fold_in(jax.random.key(g['seed']),version))
        bounds,widths=offsets(14),[b for _,b in shapes(14)]
        ids=[str(int(bounds[f]+s*widths[f]+t)) for p in plans for f,s,t in np.asarray(p.actions)[:int(p.count)].tolist()]
        g['queue'],g['queue_version']=ids,version
        g['queue_final']=int(final.state_version)
        return {'seconds':time.monotonic()-start,'sub_actions':len(ids),'plans':len(plans),
                'overrides':sum(int(p.best)!=0 for p in plans),'value':float(np.asarray(plans[0].scores)[int(plans[0].best)]) if plans else None}

    def frame(self,g, event=None):
        env=g['env']
        return {'index':len(g['frames']), 'current_player':int(env.state.current_player),
                'terminated':bool(env.state.terminated),'truncated':bool(env.state.truncated),
                'views':{'1':with_hud(compact(env.observe(1)),env,1),
                         '2':with_hud(compact(env.observe(2)),env,2),
                         'all':with_hud(world_view(env),env,int(env.state.current_player))},
                'event':event}

    def create(self, data):
        from polytopia_bot.studio.environment import RustEnvironment
        mode=data.get('mode','play')
        if mode not in ('play','watch','agent'): raise ValueError('mode must be play, watch or agent')
        seat=int(data.get('seat',1))
        if seat not in (1,2): raise ValueError('seat must be 1 or 2')
        # New interactive maps have their own namespace, outside research splits.
        seed=int(data.get('seed',1810000000+secrets.randbelow(1000000)))
        if not 0 <= seed < 2**31: raise ValueError('seed outside int32 range')
        if not self.ready: self.warmup()
        if len(self.games)>=self.max_games: raise ValueError('Too many games are open; close one first')
        search=bool(data.get('search',self.search_workers is not None)) and mode!='agent'
        if search and self.search_workers is None: raise ValueError('Search is not enabled on this server')
        opponent=str(data['opponent']) if data.get('opponent') else None
        opponent_seat=int(data.get('opponent_seat',2))
        if opponent is not None:
            if mode!='watch': raise ValueError('A named opponent applies to watch games only')
            if opponent not in self.opponents: raise ValueError('Unknown opponent')
            if opponent_seat not in (1,2): raise ValueError('opponent_seat must be 1 or 2')
        env=RustEnvironment(14)
        env.reset(seed=seed, settings={})
        ident=uuid.uuid4().hex
        if mode!='watch': title=f'You vs {self.short}'
        elif opponent is None: title=f'{self.short} self-play'
        else: title=f'{self.short} (blue) vs {opponent} (red)' if opponent_seat==2 else f'{opponent} (blue) vs {self.short} (red)'
        g={'id':ident,'mode':mode,'seat':seat,'seed':seed,'env':env,'frames':[],'search':search,
           'opponent':opponent,'opponent_seat':opponent_seat,
           'created':time.time(), 'busy':False,'error':None,'title':title}
        g['frames'].append(self.frame(g))
        self.games[ident]=g
        self.persist(g)
        return {'game_id':ident, 'mode':mode,'seat':seat,'search':search,'opponent':opponent,'opponent_seat':opponent_seat}

    def persist(self,g):
        # Durable private state is never served by the HTTP static-file handler.
        atomic(self.storage/'sessions'/f"{g['id']}.json", {'id':g['id'],'mode':g['mode'],
               'seat':g['seat'],'seed':g['seed'],'created':g['created'],'title':g['title'],'search':g.get('search',False),
               'opponent':g.get('opponent'),'opponent_seat':g.get('opponent_seat',2),
               'summary':self.describe(g), 'snapshot':base64.b64encode(g['env'].snapshot()).decode()})

    def restore(self,ident):
        if ident in self.games:return self.games[ident]
        path=self.storage/'sessions'/f'{ident}.json'
        if not path.is_file():raise KeyError('Game not found')
        from polytopia_bot.studio.environment import RustEnvironment
        d=json.loads(path.read_text()); d.pop('summary',None); env=RustEnvironment(14)
        env.restore(base64.b64decode(d.pop('snapshot')))
        g={**d,'env':env,'frames':[],'busy':False,'error':None}
        # Rehydrate previous frames from append-only journal.
        journal=self.storage/'sessions'/f'{ident}.frames.jsonl'
        if journal.exists():
            for line in journal.read_text().splitlines():
                try: frame=json.loads(line)
                except ValueError: break  # A partial trailing write after interruption.
                if frame['views']['1']['state_version']<=int(env.state.state_version):
                    g['frames'].append(frame)
            journal.write_text(''.join(dump(f)+'\n' for f in g['frames']))
        if not g['frames'] or g['frames'][-1]['views']['1']['state_version']!=int(env.state.state_version):
            g['frames'].append(self.frame(g))
        self.games[ident]=g
        return g

    def _apply(self,g,action_id,version,source,metrics=None):
        env=g['env']; obs=env.observe(int(env.state.current_player))
        if version!=obs.state_version: raise Conflict('State changed; refresh before acting')
        action=next((a for a in obs.legal_actions if a.action_id==str(action_id)),None)
        if action is None: raise Conflict('That action is not legal in the current state')
        result=env.step(action_id=action.action_id,expected_state_version=version)
        event={'player':obs.player,'turn':obs.visible_state['turn'],'action':labeled(action),
               'source':source,'metrics':metrics,'rewards':result.rewards,'reason':result.reason}
        frame=self.frame(g,event)
        with self.lock: g['frames'].append(frame)
        journal=self.storage/'sessions'/f"{g['id']}.frames.jsonl"
        with journal.open('a') as f:
            if len(g['frames'])==2: f.write(dump(g['frames'][0])+'\n')
            f.write(dump(frame)+'\n')
        self.persist(g)
        if result.terminated or result.truncated: self.save(g)
        return frame

    def action(self,ident,data):
        g=self.restore(ident)
        if g['mode']=='watch':raise Conflict('Spectator games are controlled by the bot')
        if int(g['env'].state.current_player)!=g['seat']:raise Conflict('It is the bot’s turn')
        self._apply(g,str(data['action_id']),int(data['expected_state_version']),g['mode'])
        return self.describe(g)

    def set_search(self,ident,data):
        # No engine work and no lock: a plan in flight is dropped when it is next consulted.
        g=self.restore(ident)
        if not isinstance(data.get('search'),bool):raise ValueError('search must be true or false')
        if data['search'] and self.search_workers is None:raise ValueError('Search is not enabled on this server')
        g['search'],g['queue']=data['search'],[]
        self.persist(g)
        return self.describe(g)

    def advance(self,ident,data):
        """Plays up to `count` bot sub-actions. Search mode plans the whole remaining turn once, outside the
        studio lock (other requests keep flowing), then replays it one sub-action per step."""
        with self.lock:g=self.restore(ident)
        if g['error']:g['error']=None
        for _ in range(max(1,min(int(data.get('count',1)),8))):
            env=g['env']
            if bool(env.state.terminated|env.state.truncated):break
            player=int(env.state.current_player)
            if g['mode']!='watch' and player==g['seat']:break
            if g.get('opponent') and player==g.get('opponent_seat',2):
                # The named rival plays its raw deterministic policy; the served bot's search queue is left alone
                # (it is keyed by state version, so the bot replans on its next turn).
                with self.lock:
                    obs=env.observe(player);start=time.monotonic()
                    decision=self.opponent_policy(g['opponent']).select(obs)
                    self._apply(g,decision.action.action_id,obs.state_version,'checkpoint',
                                {'value':decision.value,'log_probability':decision.log_probability,
                                 'seconds':time.monotonic()-start,'policy':g['opponent']})
                continue
            metrics=None
            if g.get('search'):
                if not g.get('queue') or g.get('queue_version')!=int(env.state.state_version):
                    metrics={'search':self.plan_turn(g)}
                    metrics['seconds']=metrics['search'].pop('seconds')
                    metrics['value']=metrics['search'].pop('value')
            with self.lock:
                obs=env.observe(player)
                if g.get('search') and g.get('queue') and g.get('queue_version')==obs.state_version:
                    action_id=g['queue'].pop(0)
                    try:
                        self._apply(g,action_id,obs.state_version,'search',metrics)
                        g['queue_version']=int(env.state.state_version)
                        if not g['queue'] and g['queue_version']!=g.get('queue_final'):
                            print(f"search replay ended at version {g['queue_version']}, planned {g.get('queue_final')}",flush=True)
                        continue
                    except Conflict as e:
                        # The plan disagreed with the live game's catalog. Fall back to the policy.
                        print(f'search action {action_id} rejected ({e}); policy plays this step',flush=True)
                        g['queue']=[]
                start=time.monotonic()
                decision=self.policies[player].select(obs)
                self._apply(g,decision.action.action_id,obs.state_version,'checkpoint',
                            {'value':decision.value,'log_probability':decision.log_probability,'seconds':time.monotonic()-start})
        return self.describe(g)

    def describe(self,g):
        f=g['frames'][-1]
        return {'id':g['id'],'title':g['title'],'mode':g['mode'],'seat':g['seat'],
                'frames':len(g['frames']),'current_player':f['current_player'],
                'terminated':f['terminated'],'truncated':f['truncated'],
                'bot':self.bot,'busy':g['busy'],'error':g['error'],'search':bool(g.get('search')),
                'opponent':g.get('opponent'),'opponent_seat':g.get('opponent_seat',2)}

    def public_frame(self,g,index=None,view=None):
        with self.lock:
            desc=self.describe(g)
            f=g['frames'][max(0,min(int(index) if index is not None else len(g['frames'])-1,len(g['frames'])-1))]
        # A human or external agent cannot reveal the opponent by changing a URL.
        completed=desc['terminated'] or desc['truncated']
        requested=str(view or (g['seat'] if g['mode']!='watch' else 'all'))
        if g['mode']!='watch' and not completed:requested=str(g['seat'])
        key=str(f['current_player']) if requested=='actor' else requested
        if key not in f['views']:raise ValueError('Unavailable perspective')
        event=f['event']
        if g['mode']!='watch' and not completed and event and event['player']!=g['seat']:
            # Keep privileged action payloads redacted. A separate cue may name
            # endpoints only when they are already visible to this seat.
            cue=None
            if f['index']>0:
                prior=g['frames'][f['index']-1]['views'][str(g['seat'])]
                cue=visible_animation(prior,f['views'][str(g['seat'])],event)
            event={'player':event['player'],'turn':event['turn'],'source':event.get('source','checkpoint'),
                   'action':{'label':'Opponent acted','command':{}},'reason':event.get('reason')}
            if cue:event['animation']=cue
        return {**desc,'frame':{**{k:v for k,v in f.items() if k not in ('views','event')},'event':event},
                'observation':f['views'][key],'perspective':requested,
                'available_views':['1','2','all','actor'] if g['mode']=='watch' or completed else [str(g['seat'])],
                'event':event, 'frame_index':f['index']}

    def observation(self,ident):
        g=self.restore(ident)
        seat=g['seat']
        obs=g['env'].observe(seat)
        return {'schema':'bardur-agent-v1','game_id':ident,**asdict(obs),
                'your_turn':int(g['env'].state.current_player)==seat,
                'terminated':bool(g['env'].state.terminated),'truncated':bool(g['env'].state.truncated)}

    def preview(self,ident,action_id):
        """What an attack would do, for the attack preview: the health of each unit it changes."""
        g=self.restore(ident); env=g['env']
        if g['mode']=='watch' or int(env.state.current_player)!=g['seat']:raise Conflict('Previews are for your own turn')
        obs=env.observe(g['seat'])
        action=next((a for a in obs.legal_actions if a.action_id==str(action_id)),None)
        if action is None or action.command['kind']!=2:raise ValueError('Only a legal attack can be previewed')
        return {'action_id':action.action_id,'state_version':int(obs.state_version),'units':env.preview(action.action_id)}

    def save(self,g):
        completed=bool(g['env'].state.terminated|g['env'].state.truncated)
        frames=list(g['frames'])
        views=['1','2','all','actor']
        if g['mode']!='watch' and not completed:
            views=[str(g['seat'])]
            frames=[]
            for i in range(len(g['frames'])):
                public=self.public_frame(g,i)
                frames.append({**public['frame'],'event':public['event'],
                               'views':{str(g['seat']):public['observation']}})
        replay={'schema':SCHEMA,'title':g['title'],'created':g['created'],'bot':self.bot,
                'ruleset':RULESET_ID,'available_views':views,'frames':frames,
                'complete':completed,'provenance':{**({'seed':g['seed']} if completed or g['mode']=='watch' else {}),'source':'studio-live','checkpoint_sha256':self.bot['checkpoint_sha256'],
                    **({'search':True} if g.get('search') else {}),
                    **({'opponent':{'name':g['opponent'],'seat':g.get('opponent_seat',2),
                                    'checkpoint_sha256':self.opponents[g['opponent']]['checkpoint_sha256']}} if g.get('opponent') else {})}}
        self.write_replay(g['id'],replay)
        return {'replay_id':g['id']}

    def write_replay(self,ident,replay):
        path=self.storage/'replays'/f'{ident}.json.gz'
        temp=path.with_suffix('.tmp')
        with gzip.open(temp,'wt') as f:json.dump(replay,f,separators=(',',':'))
        temp.replace(path)
        self.replays[ident]=replay

    def import_trace(self,raw,name='Imported game'):
        digest=hashlib.sha256(raw).hexdigest()
        if raw.startswith(b'\x1f\x8b'):
            import io
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:raw=f.read(128*1024*1024+1)
        if len(raw)>128*1024*1024:raise ValueError('Replay expands beyond 128 MiB')
        text=raw.decode()
        try: whole=json.loads(text)
        except json.JSONDecodeError: whole=None
        if isinstance(whole,dict) and whole.get('schema')==SCHEMA:
            if not whole.get('frames') or len(whole['frames'])>20000:raise ValueError('Invalid frame count')
            for f in whole['frames']:
                if not isinstance(f.get('views'),dict):raise ValueError('Missing replay views')
                for obs in f['views'].values():
                    v=obs['visible_state']
                    if v['board_size'] not in (11,14) or len(v['tiles'])!=v['board_size']**2:raise ValueError('Unsupported board')
            replay=whole
        else:
            records=[json.loads(line) for line in text.splitlines() if line.strip()]
            frames=[]; header=records[0]; previous=None; result=None
            for r in records:
                if 'result' in r: result=r['result'];continue
                if 'observation' not in r:continue
                obs=dict(r['observation']);obs.pop('legal_actions',None);obs.pop('remembered_state',None)
                if obs['visible_state']['board_size'] not in (11,14):raise ValueError('Unsupported trace size')
                frames.append({'index':len(frames),'current_player':obs['player'],
                               'terminated':False,'truncated':False,'views':{'actor':obs},'event':previous})
                a=r.get('action')
                previous={'player':obs['player'],'turn':obs['visible_state']['turn'],
                          'action':labeled(a),'source':'recorded'} if a and 'command' in a else None
            if not frames:raise ValueError('No compatible public observations in trace')
            replay={'schema':SCHEMA,'title':name,'created':time.time(),'ruleset':RULESET_ID,
                    'available_views':['actor'],'frames':frames,'complete':result is not None,
                    'result':result,'terminal_frame_recorded':False,
                    'provenance':{'source':'public-observation-trace','trace_sha256':digest,'header':header},
                    'note':'Recorded acting-player observations. Final post-action board was not recorded; final outcome is shown separately.'}
        ident=uuid.uuid4().hex
        self.write_replay(ident,replay)
        return {'replay_id':ident,'frames':len(replay['frames'])}

    def replay_frame(self,ident,index=0,view='actor'):
        # frame_index is the list position: replays recorded on variant maps stored index 1 for frame 0.
        r=self.replays[ident];position=max(0,min(int(index),len(r['frames'])-1));f=r['frames'][position]
        if view not in r['available_views']:view=r['available_views'][0]
        key=str(f['current_player']) if view=='actor' and 'actor' not in f['views'] else view
        return {'id':ident,'title':r['title'],'mode':'replay','frames':len(r['frames']),
                'frame_index':position,'frame':{k:v for k,v in f.items() if k!='views'},
                'observation':f['views'][key],'perspective':view,'available_views':r['available_views'],
                'event':f.get('event'),'bot':r.get('bot'),'result':r.get('result'),
                'note':r.get('note'),'provenance':r.get('provenance'),
                'terminated':f['terminated'],'truncated':f['truncated']}

    def replay_series(self,ident):
        """Per-turn economy and army of every player, for the replay side panel.

        One sample per player turn, taken at its first frame (after income), plus the final frame. Needs every
        seat's view and the spectator view, so it is only offered for replays that already show everything.
        Army value = star cost of each unit scaled by its remaining health."""
        r=self.replays[ident]
        if 'series' in r:return r['series']
        seats=sorted(int(v) for v in r['available_views'] if v.isdigit())
        if len(seats)<2 or 'all' not in r['available_views']:
            return {'available':False}
        units={u['id']:u for u in RULES['units']}
        def army(view,p):
            return sum(units[u['kind']]['cost']*u['health']/(units[u['kind']]['health']+5*u['veteran'])
                       for u in view['units'] if u['owner']==p and u['kind'] in units and units[u['kind']]['cost']>0)
        def sample(f,i,p):
            mine,world=f['views'][str(p)],f['views']['all']['visible_state']
            cities=[c for c in world['cities'] if c['owner']==p]
            return {'frame':i,'turn':world['turn'],'stars':mine['visible_state']['stars'],
                    'income':mine.get('hud',{}).get('income'),'score':mine.get('hud',{}).get('score'),
                    'cities':len(cities),'population':sum(c.get('population',0) for c in cities),
                    'units':sum(u['owner']==p for u in world['units']),'army':round(army(world,p),1)}
        players={p:[] for p in seats};seen=set()
        frames=r['frames']
        for i,f in enumerate(frames):
            p=f['current_player'];turn=f['views']['all']['visible_state']['turn']
            if p in players and (p,turn) not in seen:
                seen.add((p,turn));players[p].append(sample(f,i,p))
        last=len(frames)-1
        for p in players:
            if players[p] and players[p][-1]['frame']!=last:players[p].append({**sample(frames[last],last,p),'final':True})
        r['series']={'available':True,'players':{str(p):s for p,s in players.items()}}
        return r['series']

    def library(self):
        return [{'id':ident,'title':r['title'],'frames':len(r['frames']),'complete':r.get('complete',False),
                 'created':r.get('created',0),'source':r.get('provenance',{}).get('source'),
                 'available_views':r['available_views'],'featured':ident==self.featured} for ident,r in self.replays.items()]

    def reconstruct_trace(self,raw,name='Verified match'):
        """Replay real logged actions through engine-rs, checking every input (traces recorded before 2026-10-27
        used the retired rules and do not replay).

        This produces full spectator / either-seat frames and the actual terminal
        board. It never samples a new action or evaluates a training checkpoint.
        """
        import io
        from polytopia_bot.studio.environment import RustEnvironment
        digest=hashlib.sha256(raw).hexdigest()
        stream=gzip.open(io.BytesIO(raw),'rt') if raw.startswith(b'\x1f\x8b') else io.StringIO(raw.decode())
        header=json.loads(next(stream))
        if 'initial_seed' not in header:raise ValueError('Reconstruction requires initial_seed header')
        first=None
        for line in stream:
            record=json.loads(line)
            if 'observation' in record:first=record;break
        if first is None:raise ValueError('No observations in trace')
        env=RustEnvironment(first['observation']['visible_state']['board_size'])
        env.reset(seed=int(header['initial_seed']),settings={})
        g={'env':env,'frames':[]}
        g['frames'].append(self.frame(g))
        count=0;footer=None;transition=None
        def apply(record):
            nonlocal count,transition
            obs=env.observe(int(env.state.current_player))
            if asdict(obs)!=record['observation']:raise ValueError(f'Observation mismatch before action {count}')
            a=record['action']
            catalog={x.action_id:asdict(x) for x in obs.legal_actions}
            if catalog.get(a['action_id'])!=a:raise ValueError(f'Action mismatch at {count}')
            transition=env.step(action_id=a['action_id'],expected_state_version=obs.state_version)
            if transition.terminated!=record['terminated'] or transition.truncated!=record['truncated']:
                raise ValueError(f'Outcome mismatch at {count}')
            rewardkey='challenger_reward' if 'challenger_reward' in record else 'reward'
            seat=header.get('seat',header.get('challenger_seat'))
            if rewardkey in record and seat and transition.rewards[seat]!=record[rewardkey]:raise ValueError('Reward mismatch')
            event={'player':obs.player,'turn':obs.visible_state['turn'],'action':labeled(a),
                   'source':'verified-recording','rewards':transition.rewards,'reason':transition.reason}
            g['frames'].append(self.frame(g,event));count+=1
        try:
            apply(first)
            for line in stream:
                r=json.loads(line)
                if footer is not None:raise ValueError('Records after result footer')
                if 'result' in r:footer=r['result']
                elif 'observation' in r:apply(r)
            if footer is None:raise ValueError('Missing result footer')
            for key,actual in {'actions':count,'terminated':transition.terminated,'truncated':transition.truncated}.items():
                if key in footer and footer[key]!=actual:raise ValueError(f'Footer mismatch: {key}')
            replay={'schema':SCHEMA,'title':name,'created':time.time(),'ruleset':RULESET_ID,
                    'available_views':['all','1','2','actor'],'frames':g['frames'],
                    'complete':transition.terminated or transition.truncated,'result':footer,
                    'terminal_frame_recorded':True,'bot':self.bot,
                    'provenance':{'source':'verified-engine-replay','trace_sha256':digest,'header':header,
                                  'verified_actions':count,'checkpoint_selection_replayed':False},
                    'note':'Every logged public observation, action and transition was replayed through engine-rs (the real game rules). Original policy selections were not rerun.'}
            ident=uuid.uuid4().hex;self.write_replay(ident,replay)
            return {'replay_id':ident,'frames':len(g['frames']),'verified_actions':count}
        finally:env.close();stream.close()

    def session_library(self):
        result={ident:self.describe(g) for ident,g in list(self.games.items())}
        for path in (self.storage/'sessions').glob('*.json'):
            try:
                d=json.loads(path.read_text())
                if d['id'] not in result and 'summary' in d:result[d['id']]=d['summary']
            except (OSError,ValueError,KeyError):pass
        return list(result.values())
