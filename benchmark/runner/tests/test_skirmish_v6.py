"""Protocol-level v6 regressions: independent builds, shared maps, caps and controls."""
import copy,json
from pathlib import Path
import pytest
from polytopia_bot.skirmish import v6
from polytopia_bot.skirmish.local import checked_references


def manifest():
    m=json.loads(Path('configs/skirmish/local-v6-proxy-pilot.json').read_text())
    m['models']=m['models'][:2];m['bootstrap_resamples']=2000
    return m


def frozen(cells):
    return {c['run_id']:dict(status='completed',stages=[dict(stage=i,returncode=0,source_sha256='source',wasm_sha256='wasm',submission_quality='custom_submission') for i in range(v6.final_stage(c['track'])+1)]) for c in cells}


def test_full_manifest_requires_more_builds_and_maps_but_pilot_is_explicit():
    m=manifest();assert v6.validate_manifest(m)
    m['pilot']=False
    with pytest.raises(ValueError,match='replicates'):v6.validate_manifest(m)
    full=json.loads(Path('configs/skirmish/local-v6.json').read_text());assert v6.validate_manifest(full)


def test_segmented_control_has_identical_time_and_no_feedback():
    m=manifest()
    tracks={t:v6.track_config(m,t) for t in v6.TRACKS}
    assert {c['stage_seconds']*(1+c['revision_rounds']) for c in tracks.values()}=={600}
    assert tracks['feedback']['supplied_feedback'] and not tracks['segmented-control']['supplied_feedback']
    assert tracks['feedback']['stage_seconds']==tracks['segmented-control']['stage_seconds']


def test_plan_keeps_broken_final_and_all_maps_and_builds():
    m=manifest();cs=v6.cells('test',m);runs=frozen(cs)
    bad=next(c for c in cs if c['track']=='segmented-control')
    runs[bad['run_id']]['stages'][-1].update(returncode=1,wasm_sha256=None,submission_quality='compile_failure')
    maps=v6.make_maps(1,sample=lambda pop,n:list(range(pop.start,pop.start+n)))
    p=v6.make_plan(cs,runs,maps,m['references'],m)
    assert p['primary_games']==2*9*6*3*2
    assert len(p['invalid_stages'])==1
    assert all(j['evaluation']['exact_rounds'] for j in p['jobs'])
    final_bad=[j for j in p['jobs'] if j['candidate']['run_id']==bad['run_id'] and j['candidate']['stage']==3]
    assert final_bad and all(j['invalid_submission'] for j in final_bad)
    assert all(j['candidate']['replicate']==j['opponent']['replicate'] for j in p['jobs'] if j['phase']=='peer')
    assert v6.cells('test',m)==cs


def test_bootstrap_pairs_maps_but_never_pairs_independent_model_build_indices():
    rows=[]
    # Identical aggregate means but different independently authored builds.
    # Incorrectly pairing replicate labels would give a zero-width contrast.
    for model in ['a','b']:
        for rep in [1,2,3]:
            for seed in [1,2,3]:
                rows.append(dict(model_label=model,track='independent',replicate=rep,map_type='drylands',seed=seed,result='win' if rep==1 else 'loss'))
    s=v6.bootstrap_scores(rows,resamples=2000)
    diff=[a-b for a,b in zip(s['a','independent'],s['b','independent'])]
    low,high=v6.interval(diff)
    assert low<0<high
    # Layout draws ARE shared across models, so deterministic identical map effects cancel.
    for r in rows:r['result']='win' if r['seed']==1 else 'loss'
    s=v6.bootstrap_scores(rows,resamples=2000)
    assert s['a','independent']==s['b','independent']


def test_diagnostic_partitions_are_disjoint_from_hidden_and_practice():
    for part in ['heldout','transfer','calibration','feedback']:
        rows=v6.make_maps(3,partition=part)
        assert len({r['seed'] for r in rows})==9
        if part=='heldout':assert all(10**9<=r['seed']<1500000000 for r in rows)
        if part=='calibration':assert all(700000000<=r['seed']<800000000 for r in rows)


def test_unresolved_blocks_score_and_all_virtual_compilation_losses_stay_in_denominator():
    m=manifest();cs=v6.cells('score',m);p=v6.make_plan(cs,frozen(cs),v6.make_maps(1),m['references'],m)
    rows=[]
    for j in p['jobs']:
        a=j['candidate'];rows.append(dict(job_index=j['index'],job_sha256=j['job_sha256'],phase=j['phase'],
            model_label=a['model_label'],track=a['track'],replicate=a['replicate'],stage=a['stage'],
            opponent_label=j['opponent']['label'],map_type=j['map_type'],seed=j['seed'],seat=j['seat'],
            submission_quality=a['submission_quality'],result='loss',status='native',reason='elimination',winner=3-j['seat'],audit=dict(passed=True)))
    r=v6.summarize(p,cs,rows,m);assert r['status']=='completed'
    assert all(x['score']==0 for rs in r['ranking'].values() for x in rs)
    assert all(x['ci95'] is None for rs in r['ranking'].values() for x in rs)
    assert not r['uncertainty']['intervals_estimable']
    assert all(x['ci95'] is None for x in r['contrasts'])
    rows[0].update(result='unresolved',status='unresolved',winner=None)
    r=v6.summarize(p,cs,rows,m);assert r['status']=='provisional' and not r['contrasts']
    rows[0]['audit']['passed']=False
    with pytest.raises(ValueError,match='unaudited'):v6.summarize(p,cs,rows,m)


def test_guest_fuel_exit_is_preserved_when_it_dies_during_observation_write(monkeypatch):
    from unittest.mock import Mock
    from polytopia_bot.skirmish.wasi import WasiBot,BotFault
    from polytopia_bot.environment import Observation
    bot=WasiBot.__new__(WasiBot);bot.first=False;bot.seconds=2;bot.proc=Mock();bot.proc.wait.return_value=42
    monkeypatch.setattr('select.select',lambda *args: ([],[bot.proc.stdin],[]))
    def broken(*args):raise BrokenPipeError()
    monkeypatch.setattr('os.write',broken)
    with pytest.raises(BotFault,match='fuel_exhausted'):
        bot.select(Observation(1,0,{}, {}, []))


def test_initial_conversation_is_durable_before_a_blocking_first_request(tmp_path):
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.exceptions import TimeExceeded
    from polytopia_bot.skirmish.mini_worker import continuing_agent_class
    path=tmp_path/'stage.json'
    class Model:
        def format_message(self,role,content,**kw):return dict(role=role,content=content,**kw)
        def get_template_vars(self):return {}
        def serialize(self):return {}
        def query(self,messages):
            saved=json.loads(path.read_text())
            assert [m['role'] for m in saved['messages']]==['system','user']
            raise TimeExceeded(dict(role='exit',content='deadline',extra=dict(exit_status='TimeExceeded')))
    class Env:
        def get_template_vars(self):return {}
        def serialize(self):return {}
    cls=continuing_agent_class(DefaultAgent,{})
    agent=cls(Model(),Env(),system_template='system',instance_template='{{ task }}',output_path=path,cost_limit=0)
    agent.run('first task')
    assert json.loads(path.read_text())['info']['exit_status']=='TimeExceeded'
    assert not path.with_suffix('.json.tmp').exists()
    # Even a stage with no completed API response can resume the exact saved history.
    resumed=tmp_path/'resumed.json';cfg=dict(history=str(path),task='next task')
    cls=continuing_agent_class(DefaultAgent,cfg)
    agent=cls(Model(),Env(),system_template='system',instance_template='{{ task }}',output_path=resumed,cost_limit=0)
    # Check the receipt directly without making another request.
    agent.add_messages(dict(role='system',content='ignored on continuation'))
    data=json.loads(resumed.read_text())
    assert [m['content'] for m in data['messages'] if m['role']=='user']==['first task','next task']


def test_stage_cancellation_is_never_retried_as_an_api_failure(monkeypatch):
    from unittest.mock import patch
    from minisweagent.exceptions import TimeExceeded
    from polytopia_bot.skirmish.mini_worker import coding_model
    monkeypatch.setenv('CLIPROXY_API_KEY','fixture')
    model=coding_model(dict(provider='cliproxyapi',model='gpt-fixture'))
    error=TimeExceeded(dict(role='exit',content='cutoff',extra=dict(exit_status='TimeExceeded')))
    with patch.object(model,'_query',side_effect=error) as call,pytest.raises(TimeExceeded):model.query([])
    assert call.call_count==1


def test_v6_api_deadline_uses_remaining_stage_time_instead_of_hidden_sixty_second_cap(tmp_path,monkeypatch):
    import time
    from unittest.mock import patch
    from polytopia_bot.skirmish.mini_worker import coding_model
    monkeypatch.setenv('CLIPROXY_API_KEY','fixture')
    path=tmp_path/'clock.json';path.write_text(json.dumps(dict(deadline_unix=time.time()+125)))
    model=coding_model(dict(provider='cliproxyapi',model='gpt-fixture',v6_track={},start_path=str(path)))
    # v6_track carries its explicit stage shape in real payloads.
    model=coding_model(dict(provider='cliproxyapi',model='gpt-fixture',v6_track=dict(revision_rounds=3),start_path=str(path)))
    with patch('polytopia_bot.skirmish.providers.request',return_value={}) as call:model._query([])
    assert 120<call.call_args.args[3]<126


def test_reference_styles_do_not_disband_their_army_as_routine_development():
    from polytopia_bot.environment import Observation,LegalAction
    from polytopia_bot.skirmish.reference_panel import StyleBot
    obs=Observation(1,5,dict(board_size=14,turn=20,stars=0,technology=[2],tiles=[],units=[],cities=[]),{},[
        LegalAction('disband',dict(kind=3,source=10,target=-1,parameter=8)),
        LegalAction('end',dict(kind=10,source=-1,target=-1,parameter=0))])
    for style in ['cavalry','naval','siege']:assert StyleBot(style).select(obs)=='end'


def test_independent_rebuild_uses_captured_source_and_never_writes_coder_owned_directory(tmp_path,monkeypatch):
    from unittest.mock import Mock
    from polytopia_bot.skirmish import local
    cell=dict(run_id='fixture');stage=dict(stage=0,source_sha256='s',wasm_sha256='w')
    folder=tmp_path/'runs/fixture/stage-0';folder.mkdir(parents=True);(folder/'main.rs').write_text('fn main() {}')
    sb=Mock();monkeypatch.setattr(local.DockerSandbox,'create',lambda _:sb)
    def compile_captured(sandbox,destination,volume):
        assert (destination/'main.rs').read_text()=='fn main() {}'
        sb.filesystem.write_bytes.assert_not_called()
        return dict(source_sha256='s',wasm_sha256='w')
    monkeypatch.setattr(local,'freeze',compile_captured)
    local.rebuild_all(tmp_path,[cell],{'fixture':dict(stages=[stage])},'image',tmp_path/'slots',1)
    sb.terminate.assert_called_once()


def test_feedback_reports_only_public_fields_and_real_sandbox_paths():
    from polytopia_bot.skirmish.local import public_feedback_row
    row=dict(a='candidate',b='hero',result='loss',reason='elimination',winner=2,
        trace_relative='trusted/archive/game.jsonl',private_trace_sha256='private',
        action_chain_sha256='private-chain',initial_snapshot='private-state',
        final_public={'unseen':'facts'},seconds=5,unknown_future_private_field='secret')
    result=public_feedback_row(row,'/feedback/round-2',3)
    assert result==dict(a='candidate',b='hero',result='loss',reason='elimination',winner=2,
        trace_file='03.jsonl',trace_path='/feedback/round-2/03.jsonl')


@pytest.mark.parametrize('override,expected',[(None,10**12),('4000000000000',4*10**12)])
def test_authoring_protocol_check_reports_scoped_fuel_and_keeps_one_process(tmp_path,monkeypatch,override,expected):
    from unittest.mock import Mock
    from polytopia_bot.skirmish.authoring_tools import check_bot
    if override is None:monkeypatch.delenv('POLYTOPIA_SKIRMISH_CHECK_FUEL',raising=False)
    else:monkeypatch.setenv('POLYTOPIA_SKIRMISH_CHECK_FUEL',override)
    observations=tmp_path/'public.jsonl'
    observation=dict(player=1,state_version=0,visible_state={},remembered_state={},
        legal_actions=[dict(action_id='end',command={'kind':10})])
    observations.write_text((json.dumps(observation)+'\n')*2)
    instance=Mock();instance.select.return_value='end';factory=Mock(return_value=instance)
    monkeypatch.setattr('polytopia_bot.skirmish.wasi.WasiBot',factory)
    result=check_bot('bot.wasm',observations)
    factory.assert_called_once_with('bot.wasm',fuel=expected,seconds=2)
    assert instance.select.call_count==2 and instance.close.call_count==1
    assert result['decisions']==2 and result['runtime_limits']==dict(fuel=expected,move_seconds=2)
