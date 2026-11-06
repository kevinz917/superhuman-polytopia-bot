from dataclasses import asdict
from polytopia_bot.learning.data import fixture_observation
from polytopia_bot.learning.interface import Generation
from polytopia_bot.learning.offline import evaluate


def test_saved_positions_repair_forfeit_and_semantic_agreement(tmp_path):
    obs=fixture_observation()
    positions=[dict(position_id=str(i),observation=asdict(obs),teacher_command=dict(obs.legal_actions[0].command)) for i in range(2)]
    class Generator:
        calls=0
        def batch(self,messages):
            self.calls+=1
            if self.calls==1:return [Generation('bad',[1],[2]),Generation('bad',[1],[2])]
            if self.calls==2:
                assert all(len(m)==4 for m in messages)
                return [Generation('{"action_id":"a0"}',[1],[2]),Generation('bad',[1],[2])]
            assert len(messages)==1 and len(messages[0])==6
            return [Generation('bad',[1],[2])]
    r=evaluate(tmp_path/'eval',positions,Generator())
    assert r['complete'] and r['decisions']==2 and r['attempts']==5
    assert r['first_attempt_valid']==0 and r['eventually_valid']==1 and r['teacher_agreement']==1
