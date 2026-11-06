import pytest
torch=pytest.importorskip('torch')
from polytopia_bot.learning import trainer_qualification as tq
from polytopia_bot.learning.expert_trainer import check_benchmark, check_variant

VARIANT=dict(name='v',packed=True,backend='segmented_sdpa',token_budget=8192,checkpointing=False)


def test_compare_uses_float64_cosine():
    g=torch.full((1000,),1e-23)
    out=tq.compare(1.,g,1.,g.clone())
    assert out['cosine']==pytest.approx(1.,abs=1e-12) and out['relative_gradient_error']==0 and out['passed']


def arms(**deltas):
    ref=[1.,.8]+[.5]*30
    out=dict(ref=dict(status='passed',losses=ref,tokens_per_second=100.))
    for n,d in deltas.items():out[n]=dict(status='passed',losses=[x+d for x in ref],tokens_per_second=140.)
    return out


def test_trajectory_and_fp32_gates():
    a=arms(noise=.004,good=.005,bad=.01);a['good']['losses'][1]+=.03   # a large warm-up deviation is not gated
    strict=dict(ref=dict(passed=True),noise=dict(passed=True),good=dict(passed=False),bad=dict(passed=False))
    fp32=dict(ref=dict(loss_delta=.029),noise=dict(loss_delta=.029),good=dict(loss_delta=.0005),bad=dict(loss_delta=.05))
    good=tq.assess('good','ref',a,strict,fp32,2)
    assert not good['strict']['passed'] and good['fp32_anchored']['passed'] and good['trajectory']['passed'] and good['revised']['passed']
    assert good['trajectory']['noise_floor']==pytest.approx(.004) and set(good['trajectory']['noise_arms'])=={'noise'}
    bad=tq.assess('bad','ref',a,strict,fp32,2)
    assert not bad['fp32_anchored']['passed'] and not bad['trajectory']['passed'] and not bad['revised']['passed']


def test_trajectory_needs_a_gate_passing_noise_arm():
    a=arms(good=.001)
    out=tq.assess('good','ref',a,dict(ref=dict(passed=True),good=dict(passed=True)),dict(ref=dict(loss_delta=0),good=dict(loss_delta=0)),2)
    assert out['strict']['passed'] and not out['trajectory']['passed'] and not out['revised']['passed']


def receipt(strict_ok, revised_ok, speedup=1.4):
    a=arms(good=.001)
    gates=dict(strict=dict(passed=strict_ok,loss=1.,loss_delta=0.,relative_gradient_error=0.,cosine=1.),revised=dict(passed=revised_ok))
    base=dict(source='movement-scale40k-v1/protocol.json',speedup=speedup,selected=dict(name='qualified'))
    return tq.receipt('good',VARIANT,'ref',base,a,gates,dict(config={}))


def test_receipt_is_accepted_only_under_the_gate_it_passed():
    r=receipt(False,True)
    assert r['speedup']==pytest.approx(1.4*1.4) and r['speedup_vs_reference']==pytest.approx(1.4)
    assert check_benchmark(r,'revised')['backend']=='segmented_sdpa'
    with pytest.raises(ValueError):check_benchmark(r,'strict')
    assert check_benchmark(receipt(True,False),'strict')['token_budget']==8192
    with pytest.raises(ValueError):check_benchmark(receipt(True,False),'revised')
    with pytest.raises(ValueError):check_benchmark(receipt(True,True),'lenient')


def test_legacy_receipts_pass_strict_only():
    legacy=dict(parity=dict(passed=True),speedup=1.5,selected=dict(VARIANT))
    assert check_benchmark(legacy)['name']=='v'
    with pytest.raises(ValueError):check_benchmark(legacy,'revised')
    with pytest.raises(ValueError):check_benchmark(dict(legacy,speedup=.9))


@pytest.mark.parametrize('bad',[dict(packed=False),dict(backend='eager'),dict(checkpointing='sometimes'),
                                dict(checkpointing='adaptive'),dict(checkpointing='adaptive',adaptive_tokens=0),dict(token_budget=0)])
def test_check_variant_rejects(bad):
    with pytest.raises(ValueError):check_variant(dict(VARIANT,**bad))


def test_check_variant_accepts_new_runtimes():
    check_variant(dict(VARIANT,backend='segmented_cudnn',token_budget=16384,liger_swiglu=True))
    check_variant(dict(VARIANT,checkpointing='adaptive',adaptive_tokens=12288))


def test_assemble_from_raw_json_round_trip():
    import json
    ref=[2.,1.]+[.5]*30
    traj=lambda name,d,tps:dict(name=name,status='passed',losses=[x+d for x in ref],tokens_per_second=tps,runtime={})
    probe=lambda d,e,c:dict(loss=1.+d,reference_loss=1.,loss_delta=d,relative_gradient_error=e,cosine=c,passed=d<.01 and e<.03 and c>.999)
    arms=[dict(VARIANT,name='qualified'),dict(VARIANT,name='b16384',token_budget=16384),
          dict(VARIANT,name='swiglu',token_budget=16384,liger_swiglu=True),
          dict(VARIANT,name='cudnn',backend='segmented_cudnn',token_budget=16384,liger_swiglu=True)]
    raw=dict(context=dict(config={'max_length':24576},warmup_steps=2,probe_setup={},tag='t'),arms=arms,candidates=['swiglu','cudnn'],
             base=dict(source=dict(path='p',sha256='s'),speedup=1.53,selected=arms[0]),seconds=1.,
             probes=dict(strict=dict(lengths=[1],results=dict(qualified=probe(0,0,1),b16384=probe(0,.015,.9999),
                                                               swiglu=probe(0,.016,.9999),cudnn=probe(.018,.87,.59))),
                         fp32=dict(lengths=[1],results=dict(fp32_math_truth=dict(loss=1.),qualified=probe(.029,1.4,.36),
                                                             b16384=probe(.029,1.4,.36),swiglu=probe(.029,1.4,.36),cudnn=probe(.0005,1.45,.32)))),
             trajectories=dict(qualified=traj('qualified',0,53.1),b16384=traj('b16384',.0036,53.2),
                               swiglu=traj('swiglu',.0032,53.8),cudnn=traj('cudnn',.0025,74.3)))
    summary,receipts=tq.assemble(json.loads(json.dumps(raw)))
    assert set(receipts)=={'swiglu','cudnn'} and json.dumps(summary)
    assert check_benchmark(receipts['swiglu'],'strict')['liger_swiglu']
    assert receipts['swiglu']['gates']['revised']['passed']   # noise floor: b16384 only
    with pytest.raises(ValueError):check_benchmark(receipts['cudnn'],'strict')
    assert check_benchmark(receipts['cudnn'],'revised')['backend']=='segmented_cudnn'
    assert receipts['cudnn']['speedup_vs_reference']==pytest.approx(74.3/53.1)
    assert set(summary['candidates']['cudnn']['trajectory']['noise_arms'])=={'b16384','swiglu'}
