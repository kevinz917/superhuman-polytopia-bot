"""Real-engine horizon, action-budget and tactical probes inside the local VM."""
from pathlib import Path
from types import SimpleNamespace
import json

class EndBot:
    def select(self,obs):return next(a.action_id for a in obs.legal_actions if a.command['kind']==10)
    def close(self):pass

class SpendBot:
    def select(self,obs):
        return next((a.action_id for a in obs.legal_actions if a.command['kind'] not in (10,11)),obs.legal_actions[0].action_id)
    def close(self):pass


def run(directory='/results/checks'):
    from .local_referee import play_game_streamed,audit_trace_streamed
    from polytopia_bot.studio.environment import RustEnvironment
    from .game import reset_map
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    budget=SimpleNamespace(fuel=4*10**12,move_seconds=2.,game_seconds=30)
    receipts=[]
    for rounds in [1,2,5]:
        cfg=dict(max_rounds=rounds,max_actions=1000,exact_rounds=True,size=14)
        path=root/f'rounds-{rounds}.jsonl'
        row=play_game_streamed(EndBot(),EndBot(),700000011,1,budget,path,phase='checks',labels=['end','end'],map_type='drylands',cap_policy='engine_draw',evaluation=cfg)
        assert row['status']=='draw' and row['turns']==rounds and row['actions']==2*rounds,row
        assert row['action_counts']=={'1':rounds,'2':rounds}
        assert audit_trace_streamed(row,path)==row['actions']
        receipts.append(dict(check='exact_rounds',rounds=rounds,actions=row['actions'],passed=True))
    path=root/'action-cap.jsonl';cfg=dict(max_rounds=200,max_actions=12,exact_rounds=True,size=14)
    row=play_game_streamed(SpendBot(),SpendBot(),700000012,1,budget,path,phase='checks',labels=['spend','spend'],map_type='drylands',cap_policy='engine_draw',evaluation=cfg)
    assert row['actions']==12 and row['status']=='draw' and row['cap_kind']=='action_limit',row
    audit_trace_streamed(row,path)
    receipts.append(dict(check='shared_action_budget',passed=True,attribution=row['action_counts'],policy='draw; separately flagged, no claim of exploit resistance'))
    # Targeted fixture: a capture must be selected before routine economy/movement.
    from .reference_panel import StyleBot
    from polytopia_bot.environment import Observation,LegalAction
    obs=Observation(1,1,dict(board_size=14,turn=10,stars=5,technology=[1],tiles=[dict(tile=i,known=True,owner=0) for i in range(196)],units=[],cities=[dict(tile=20,owner=0)]),{},[
        LegalAction('capture',dict(kind=8,source=20,target=-1,parameter=0)),LegalAction('end',dict(kind=10,source=-1,target=-1,parameter=0))])
    for style in ['cavalry','naval','siege']:
        assert StyleBot(style).select(obs)=='capture'
    receipts.append(dict(check='scripted_capture_priority',passed=True))
    # Every board-size and map-family combination must expose correct public size/IDs.
    for size in [11,14,18]:
        env=RustEnvironment(size=size,max_rounds=2,max_actions=100)
        try:
            for family in ['pangea','archipelago','highlands']:
                obs=reset_map(env,700000013,family)
                assert obs.visible_state['board_size']==size
                assert len(obs.visible_state['tiles'])==size*size
                action=next(a for a in obs.legal_actions if a.command['kind']==10)
                env.step(action_id=action.action_id,expected_state_version=obs.state_version)
        finally:env.close()
    receipts.append(dict(check='transfer_board_abi',passed=True,sizes=[11,14,18]))
    (root/'report.json').write_text(json.dumps(dict(passed=True,checks=receipts),indent=2)+'\n')
    return receipts

if __name__=='__main__':print(json.dumps(run()))
