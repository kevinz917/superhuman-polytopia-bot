"""Session H arena matches (prereg.md), written once both H arms have finished: two arena calls, one per GPU.

    python3 make_matches.py H4_UPDATES P1W_UPDATES OUT_DIR

A: the hybrid at equal dollars vs the CNN bar and every E no_peace V3 arm (E-best is read off these).
B: the p1 arm (equal dollars and equal samples) and the hybrid's equal-samples match vs the CNN.
Snapshots are every 10 updates; equal samples takes the snapshot nearest to and >= the shorter arm's final update."""
import json
import math
import sys
from pathlib import Path

H = '/results/distributed'
E = '/e/distributed'
D_NP = f'{E}/np-D-cnn-4x128'
D_NP_UPDATES = 643
H4, P1W = f'{H}/h-h4-v3p2h4-6x128', f'{H}/h-p1w-v3-6x128'
SNAPSHOT_EVERY = 10


def snapshot_at_least(run: str, updates: int) -> str:
    return f'{run}/snapshots/update-{SNAPSHOT_EVERY * math.ceil(updates / SNAPSHOT_EVERY)}.npz'


def match(name: str, a: str, b: str, maps: int) -> dict:
    return dict(name=name, a=a, b=b, maps=maps, timeout=1500)


def main(h4_updates: int, p1w_updates: int, out: Path) -> None:
    a = [match('h4-vs-Dnp-final', f'{H4}/final.npz', f'{D_NP}/final.npz', 1024),
         match('h4-vs-Bw-final', f'{H4}/final.npz', f'{E}/np-Bw-v3p2-6x128/final.npz', 1024),
         match('h4-vs-Bk3-final', f'{H4}/final.npz', f'{E}/np-Bk3-v3p2k3-6x128/final.npz', 1024),
         match('h4-vs-B-final', f'{H4}/final.npz', f'{E}/np-B-v3p2-6x128/final.npz', 512)]
    if h4_updates < D_NP_UPDATES:
        equal = match(f'h4-final-vs-Dnp-u{SNAPSHOT_EVERY * math.ceil(h4_updates / SNAPSHOT_EVERY)}',
                      f'{H4}/final.npz', snapshot_at_least(D_NP, h4_updates), 512)
    else:
        equal = match(f'h4-u{SNAPSHOT_EVERY * math.ceil(D_NP_UPDATES / SNAPSHOT_EVERY)}-vs-Dnp-final',
                      snapshot_at_least(H4, D_NP_UPDATES), f'{D_NP}/final.npz', 512)
    b = [match('p1w-vs-Dnp-final', f'{P1W}/final.npz', f'{D_NP}/final.npz', 1024),
         match('p1w-vs-Bw-final', f'{P1W}/final.npz', f'{E}/np-Bw-v3p2-6x128/final.npz', 512),
         match(f'p1w-final-vs-Bw-u{SNAPSHOT_EVERY * math.ceil(p1w_updates / SNAPSHOT_EVERY)}', f'{P1W}/final.npz',
               snapshot_at_least(f'{E}/np-Bw-v3p2-6x128', p1w_updates), 512),
         equal]
    out.mkdir(parents=True, exist_ok=True)
    (out / 'arena-A.json').write_text(json.dumps(a, indent=1))
    (out / 'arena-B.json').write_text(json.dumps(b, indent=1))
    print(json.dumps(dict(A=[m['name'] for m in a], B=[m['name'] for m in b])))


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]))
