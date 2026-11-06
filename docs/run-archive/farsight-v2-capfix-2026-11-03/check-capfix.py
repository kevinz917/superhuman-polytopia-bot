"""10-minute status check for farsight-v2-capfix: learner trace (last 10 rows), fleet, keeper errors. Prints ALERT lines."""
import sys, json, time, collections, pathlib
sys.path.insert(0, 'experiments/daytona'); sys.argv = ['hero.py']
import hero
run = 'farsight-v2-capfix'
now = time.strftime('%H:%M')
rows = hero.sandboxes(run=run)
c = collections.Counter((s['labels'].get('role'), s.get('state')) for s in rows)
actors = [s for s in rows if s['labels'].get('role') == 'actor' and hero._alive(s)]
learners = [s for s in rows if s['labels'].get('role') == 'learner' and hero._alive(s)]
alerts = []
if not learners: alerts.append('NO LEARNER')
if len(actors) < 4: alerts.append(f'ONLY {len(actors)} ACTORS')
line = f'{now} fleet learner={len(learners)} actors={len(actors)} evals={c.get(("eval","started"),0)}'
if learners:
    try:
        out = hero.sh(learners[0], f'tail -n 10 /workspace/run/{run}/training.jsonl', check=False, timeout=90)
        rs = [json.loads(x) for x in out.strip().splitlines() if x.strip().startswith('{')]
        if rs:
            b = rs[-1]; ac = b['actors']
            kl = sum(r['approx_kl'] for r in rs) / len(rs); lr = min(r['lr_scale'] for r in rs)
            line += (f" | u{b['update']} {b['samples']/1e9:.2f}B kl10 {kl:.4f} ema {b['kl_ema']:.4f} lr_scale {lr:.2f} clip {b['clip_fraction']:.3f} "
                     f"ent {b['entropy']:.2f} | capped {ac['selfplay_truncations']}/{ac['selfplay_games']} | recv/s {b['received_samples_per_second']:.0f}")
            if kl > 0.015: alerts.append(f'KL HIGH {kl:.4f}')
            if lr < 0.5: alerts.append(f'LR_SCALE LOW {lr:.2f}')
            if b['received_samples_per_second'] < 50000: alerts.append(f"INTAKE LOW {b['received_samples_per_second']:.0f}")
        else:
            line += ' | no training rows yet'
    except Exception as e:
        line += f' | learner read failed {repr(e)[:80]}'
log = pathlib.Path('runs/hero-daytona/keep-capfix.log').read_text().splitlines()[-40:]
errs = [l for l in log if 'actor_error' in l or 'learner_launch_failed' in l or 'stalled' in l or 'supervisor' in l]
if errs: alerts.append(f'KEEPER: {errs[-1][:160]}')
tail = [l for l in log if '"learners"' in l]
print(line + (f" | keeper {tail[-1][:90]}" if tail else ''))
for a in alerts: print('ALERT', a)
