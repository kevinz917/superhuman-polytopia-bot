import gzip
import json
import pytest
from polytopia_bot.learning.efficient_gameplay import play_pipelined
from polytopia_bot.learning.gameplay import play_games
from polytopia_bot.learning.interface import Generation
from test_learning_gameplay import Game, jobs


class Valid:
    def batch(self, messages):
        return [Generation('{"action_id":"a0"}', [1], [2]) for _ in messages]


class Invalid:
    def batch(self, messages):
        return [Generation('bad', [1], [2]) for _ in messages]


def semantic_trace(path):
    result=[]
    for line in gzip.open(path,'rt'):
        row=json.loads(line)
        if 'game_result' in row:row['game_result'].pop('seconds')
        for attempt in row.get('diagnostic',[]) if row.get('policy')=='model' else []:
            attempt.pop('batch_seconds')
        result.append(row)
    return result


def test_cpu_lanes_preserve_public_trace_actions_and_batch_boundaries(tmp_path):
    allocated=jobs(4);lanes=[allocated[::2],allocated[1::2]]
    result=play_pipelined(tmp_path/'pipe',lanes,Valid(),environment_factory=Game)
    assert result['complete'] and result['finished_games']==4
    assert result['pipeline']['batches']==4
    for lane,group in enumerate(lanes):
        expected=play_games(tmp_path/f'serial{lane}',group,Valid(),environment_factory=Game)
        for job in group:
            assert semantic_trace(tmp_path/f'pipe/shard-{lane}/games/{job["id"]}.jsonl.gz')==semantic_trace(tmp_path/f'serial{lane}/{job["id"]}.jsonl.gz')


def test_invalid_responses_keep_three_attempt_limit(tmp_path):
    result=play_pipelined(tmp_path/'pipe',[jobs(2)],Invalid(),environment_factory=Game)
    assert result['complete']
    assert all(g['status']=='forfeit' and g['attempts']==3 and g['decisions']==0 for g in result['games'])


def test_server_failure_cancels_lanes_without_marking_complete(tmp_path):
    class Broken:
        def batch(self,messages):raise RuntimeError('deliberate server failure')
    with pytest.raises(RuntimeError,match='deliberate server failure'):
        play_pipelined(tmp_path/'pipe',[jobs()],Broken(),environment_factory=Game)
    assert not (tmp_path/'pipe/report.json').exists()
