"""Expert-trainer runtime qualification: two parity gates per candidate variant.

strict   same probe batch at the initial adapter, against the qualified bf16 path:
         |dloss| < 0.01, relative gradient error < 0.03, cosine > 0.999 (the original gate).
         It checks that a variant shares the qualified path's bf16 rounding; any change of
         attention kernel fails it, whether more or less accurate.
revised  (a) fp32-anchored loss: against an fp32 reference (fp32 weights, no autocast, math SDPA),
             the candidate's |dloss| is at most the qualified path's |dloss| + 0.01; and
         (b) trajectory: over identical training steps from the same adapter, the candidate's mean
             |loss - qualified loss| after warm-up is within TRAJECTORY_TOLERANCE x the noise floor,
             the largest such deviation among other arms that pass the strict gate.
See docs/sft-trainer-benchmark-2026-10-19.md section 3. The trainer applies the gate the run names.
"""

STRICT = dict(loss=.01, relative_gradient_error=.03, cosine=.999)
FP32_LOSS_TOLERANCE = .01
TRAJECTORY_TOLERANCE = 1.5
SCHEMA = 'expert-trainer-qualification-v1'


def compare(ref_loss, ref_grad, loss, grad):
    """Strict-gate metrics. Float64: a float32 cosine of these tiny gradients underflows."""
    import torch
    a, b = ref_grad.double(), grad.double()
    out = dict(loss=loss, reference_loss=ref_loss, loss_delta=abs(loss-ref_loss),
               relative_gradient_error=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a)),
               cosine=float(a.dot(b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))))
    out['passed'] = (out['loss_delta'] < STRICT['loss'] and out['relative_gradient_error'] < STRICT['relative_gradient_error']
                     and out['cosine'] > STRICT['cosine'])
    return out


def trajectory_deviation(losses, reference, warmup):
    if len(losses) != len(reference) or len(losses) <= warmup:
        raise ValueError('trajectories must cover the same steps beyond warm-up')
    d = [abs(a-b) for a, b in zip(losses, reference)]
    measured = d[warmup:]
    return dict(max_abs=max(d), max_abs_after_warmup=max(measured), mean_abs=sum(measured)/len(measured))


def assess(candidate, reference, arms, strict, fp32, warmup):
    """Both gates for one candidate arm.

    arms: name -> dict(status, losses, tokens_per_second); strict: name -> compare() vs reference;
    fp32: name -> dict(loss_delta) vs the fp32 reference.
    """
    ok = lambda name: arms.get(name, {}).get('status') == 'passed'
    out = dict(strict=dict(strict.get(candidate, {}), passed=bool(strict.get(candidate, {}).get('passed')) and ok(candidate)))
    if candidate in fp32 and reference in fp32:
        c, r = fp32[candidate]['loss_delta'], fp32[reference]['loss_delta']
        out['fp32_anchored'] = dict(candidate_loss_delta=c, reference_loss_delta=r, tolerance=FP32_LOSS_TOLERANCE,
                                    passed=c <= r+FP32_LOSS_TOLERANCE)
    else:
        out['fp32_anchored'] = dict(passed=False, error='missing fp32 probe')
    noise = {n: trajectory_deviation(a['losses'], arms[reference]['losses'], warmup) for n, a in arms.items()
             if n not in (candidate, reference) and ok(n) and strict.get(n, {}).get('passed')}
    if ok(candidate) and ok(reference) and noise:
        dev = trajectory_deviation(arms[candidate]['losses'], arms[reference]['losses'], warmup)
        floor = max(x['mean_abs'] for x in noise.values())
        out['trajectory'] = dict(dev, noise_floor=floor, noise_arms=noise, tolerance=TRAJECTORY_TOLERANCE,
                                 steps=len(arms[candidate]['losses']), warmup_steps=warmup,
                                 passed=dev['mean_abs'] <= TRAJECTORY_TOLERANCE*floor)
    else:
        out['trajectory'] = dict(passed=False, error='candidate, reference or noise-floor arm missing')
    out['revised'] = dict(passed=out['fp32_anchored']['passed'] and out['trajectory']['passed'])
    return out


def receipt(candidate, variant, reference, base, arms, gates, context):
    """A benchmark receipt the expert trainer accepts (selected / parity / speedup) plus both gates.

    speedup is chained to the base receipt's baseline so it keeps its original meaning.
    """
    rel = arms[candidate]['tokens_per_second']/arms[reference]['tokens_per_second']
    parity = {k: gates['strict'].get(k) for k in ('loss', 'loss_delta', 'relative_gradient_error', 'cosine')}
    return dict(schema=SCHEMA, training_only=True, **context,
                selected=dict(variant, packed=True, status='passed' if gates['strict']['passed'] else 'strict_parity_failed'),
                parity=dict(parity, passed=gates['strict']['passed']), gates=gates,
                speedup=base['speedup']*rel, speedup_vs_reference=rel, reference=reference,
                base_receipt=dict(source=base['source'], speedup=base['speedup'], selected=base['selected']),
                arm=arms[candidate])


def assemble(raw):
    """qualification summary and one receipt per candidate whose arm and the reference arm both trained."""
    arms, trajectories = raw['arms'], raw['trajectories']; ref = arms[0]['name']
    strict = raw['probes']['strict']['results']
    fp32 = {n: dict(loss_delta=x['loss_delta']) for n, x in raw['probes']['fp32']['results'].items() if 'loss_delta' in x}
    summary = dict(schema=SCHEMA+'-summary', **raw['context'], reference=ref, base=raw['base'], arms=arms,
                   probes=raw['probes'], trajectories=trajectories, candidates={}, seconds=raw.get('seconds'))
    receipts = {}
    for name in raw['candidates']:
        gates = assess(name, ref, trajectories, strict, fp32, raw['context']['warmup_steps'])
        summary['candidates'][name] = gates
        if trajectories[name].get('status') == 'passed' and trajectories[ref].get('status') == 'passed':
            variant = next(v for v in arms if v['name'] == name)
            receipts[name] = receipt(name, variant, ref, raw['base'], trajectories, gates, raw['context'])
            gates['receipt'] = f'receipt-{name}.json'
    return summary, receipts
