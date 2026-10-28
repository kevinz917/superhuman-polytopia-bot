"""Dependency-free adapter for policies that consume public observations.

Use Client.start() or attach to the game shown by the browser. All actions use
optimistic version checking. Built-in opponent moves are requested explicitly.
"""
import json
import time
from urllib.request import Request,urlopen

class Client:
    def __init__(self,base='http://127.0.0.1:8765',game_id=None):
        self.base=base.rstrip('/');self.game_id=game_id
    def request(self,path,data=None):
        req=Request(self.base+path,headers={'Content-Type':'application/json','X-Studio-Client':'1'},
                    data=None if data is None else json.dumps(data).encode())
        with urlopen(req,timeout=180) as r:result=json.load(r)
        if 'job_id' in result:
            ident=result['job_id']
            while True:
                time.sleep(.1);job=self.request('/api/jobs/'+ident)
                if job['status']=='done':
                    return self.request(path,data) if job['result'].get('restored') else job['result']
                if job['status']=='error':raise RuntimeError(job['error'])
        return result
    def start(self,seat=1,seed=None):
        data={'mode':'agent','seat':seat}
        if seed is not None:data['seed']=seed
        self.game_id=self.request('/api/games',data)['game_id']
        return self.observe()
    def observe(self):return self.request(f'/api/games/{self.game_id}/observation')
    def act(self,action_id,state_version):
        return self.request(f'/api/games/{self.game_id}/action',
                            {'action_id':str(action_id),'expected_state_version':state_version})
    def opponent(self):return self.request(f'/api/games/{self.game_id}/advance',{'count':8})
    def run(self,select,max_decisions=1000):
        """select(public_observation) -> action_id; no snapshot/seed is supplied."""
        for _ in range(max_decisions):
            obs=self.observe()
            if obs['terminated'] or obs['truncated']:return obs
            if not obs['your_turn']:self.opponent();continue
            action=select(obs)
            self.act(action,obs['state_version'])
        return self.observe()
