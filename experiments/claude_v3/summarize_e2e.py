"""Warm throughput of GPU-resident runs from their logs: median samples/s over updates 3+, rollout vs train split.

    python summarize_e2e.py LOG [LOG ...]
"""
import json
import statistics
import sys

H100_PER_HOUR = 3.95 + 4 * 0.0472 + 32 * 0.008   # 1xH100, 4 CPU, 32 GiB (the run's request)


def summarize(path: str, skip: int = 2, per_hour: float = H100_PER_HOUR) -> dict:
    ups = [json.loads(l[l.index('{'):]) for l in open(path) if '"stage": "update"' in l]
    warm = ups[skip:]
    if not warm:
        return dict(log=path, updates=len(ups))
    sps = [u['samples_per_second'] for u in warm]
    med = statistics.median(sps)
    train = statistics.median(u['train_seconds'] for u in warm)
    total = statistics.median(u['seconds'] for u in warm)
    return dict(log=path.split('/')[-1], updates=len(ups), warm=len(warm), samples_per_s=round(med),
                samples_per_hour_M=round(med * 3600 / 1e6, 1), usd_per_B=round(per_hour / (med * 3600) * 1e9, 1),
                update_s=round(total, 2), train_s=round(train, 2), rollout_s=round(total - train, 2),
                forced_close=round(statistics.median(u['forced_close_fraction'] for u in warm), 4),
                entropy_last=round(warm[-1]['entropy'], 3), approx_kl_last=round(warm[-1]['approx_kl'], 4),
                value_loss_last=round(warm[-1].get('value_loss', float('nan')), 4))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(json.dumps(summarize(p)))
