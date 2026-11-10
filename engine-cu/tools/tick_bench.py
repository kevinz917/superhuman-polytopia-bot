"""Engine-alone tick time at trainer scale: G games (default 4096) on one GPU in device-pointer mode, random legal play
sampled on the GPU with torch, the thread-per-game kernel (k_tick) against the warp-per-game kernel (k_tick_warp,
PC_CUDA_WARP_TICK=1), same seed, so the two runs play the same games and their outputs are compared bitwise every tick
(a per-tick checksum of masks, views, flags and rewards) on top of the timing. Run on Modal (experiments/
modal_gpu_engine.py::tick_bench); never on the Mac.

    python engine-cu/tools/tick_bench.py --size 14 --games 4096 --ticks 300 --warmup 30 --kernels thread,warp

Per kernel: ms per tick (mean / median / p90 of pc_tick wall time with every pointer on the device: the memsets and
the kernel; the action sampling is outside the timed region), and with POLYTOPIA_PC_PROFILE=1 (lib*_prof.so) the
phase split (GpuGames.profile_read). Prints one `TICKBENCH {json}` line per kernel and a final `TICKBENCH_CMP {json}`.
Random play: a uniform nonzero mask byte, then a uniform set bit in it (not uniform over actions; enough to move the
games along), end turn with probability --end-boost when legal. Finished games restart from the same layouts.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def sample(masks, gen, offsets, widths, mask_width, end_boost):
    """(family, source, target) int32 [G] on the GPU from packed masks [G, mb] uint8 (numpy.packbits order)."""
    import torch
    G = masks.shape[0]
    nz = masks != 0
    score = torch.rand(masks.shape, generator=gen, device=masks.device).masked_fill_(~nz, -1.0)
    byte = score.argmax(dim=1)
    val = masks.gather(1, byte[:, None])[:, 0].to(torch.int32)
    shifts = torch.arange(7, -1, -1, device=masks.device, dtype=torch.int32)   # bit k of the byte = 0x80 >> k
    bits = (val[:, None] >> shifts[None, :]) & 1
    bscore = torch.rand((G, 8), generator=gen, device=masks.device).masked_fill_(bits == 0, -1.0)
    flat = byte.to(torch.int64) * 8 + bscore.argmax(dim=1)
    end_at = mask_width - 2
    end_legal = ((masks[:, end_at // 8].to(torch.int32) >> (7 - end_at % 8)) & 1) == 1
    boost = (torch.rand((G,), generator=gen, device=masks.device) < end_boost) & end_legal
    flat = torch.where(boost, torch.full_like(flat, end_at), flat)
    fam = torch.searchsorted(offsets, flat, right=True) - 1
    within = flat - offsets[fam]
    w = widths[fam]
    return fam.to(torch.int32).contiguous(), (within // w).to(torch.int32).contiguous(), (within % w).to(torch.int32).contiguous()


def run(kernel, args, layouts_host, seeds_host):
    import torch
    from polytopia_bot.rl.actions import offsets as action_offsets, shapes
    from polytopia_bot.rl.gpu_engine import GpuGames, mask_width
    os.environ["PC_CUDA_WARP_TICK"] = "1" if kernel == "warp" else "0"
    dev = torch.device("cuda", 0)
    env = GpuGames(size=args.size, games=args.games, players=args.players, device=0)
    info = env.build_info()
    env.reset(seeds_host, layouts_host)
    out = env.alloc_device_outputs(dev)
    terrain, structure, resource, caps = (torch.as_tensor(np.ascontiguousarray(a, dtype=np.int32), device=dev) for a in layouts_host)
    seeds = torch.as_tensor(seeds_host, device=dev)
    offs = torch.as_tensor(np.asarray(action_offsets(args.size)[:-1], np.int64), device=dev)
    widths = torch.as_tensor(np.asarray([b for _, b in shapes(args.size)], np.int64), device=dev)
    mw = mask_width(args.size * args.size)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    # the reset states' masks (pc_legal_masks, host copy) for the first draw
    masks = torch.as_tensor(env.legal_masks(packed=True), device=dev)
    times, sums, finished, nonok = [], [], 0, 0
    for k in range(args.warmup + args.ticks):
        f, s, t = sample(masks, gen, offs, widths, mw, args.end_boost)
        if k == args.warmup:
            env.profile_reset()
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        env.tick_device(f, s, t, seeds, terrain, structure, resource, caps, True, out)
        dt = time.perf_counter() - t0
        if k >= args.warmup:
            times.append(dt * 1e3)
        masks = out["masks"]
        finished += int(out["done"].sum().item())
        nonok += int((out["status"] != 0).sum().item())
        if args.check:
            ck = [int(out[name].to(torch.int64).sum().item()) for name in ("masks", "view", "opp_view", "final_view", "done", "status")]
            ck.append(int(torch.round(out["rewards"].double() * 1e6).sum().item()))
            sums.append(ck)
    prof = env.profile_read()
    env.close()
    t = np.asarray(times)
    row = dict(kernel=kernel, build=info, size=args.size, games=args.games, players=args.players, ticks=args.ticks,
               ms_mean=round(float(t.mean()), 3), ms_median=round(float(np.median(t)), 3),
               ms_p90=round(float(np.percentile(t, 90)), 3), ticks_per_s=round(args.games / (float(t.mean()) / 1e3)),
               finished=finished, nonok=nonok, profile=prof)
    print("TICKBENCH " + json.dumps(row), flush=True)
    return row, sums


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=14)
    ap.add_argument("--games", type=int, default=4096)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--end-boost", type=float, default=0.12)
    ap.add_argument("--kernels", default="thread,warp")
    ap.add_argument("--no-check", dest="check", action="store_false", help="skip the per-tick output checksums")
    args = ap.parse_args(argv)
    from polytopia_engine_rs import GameBatch
    from polytopia_bot.gpu import ENGINE_REVISION
    from polytopia_bot.rl.gpu_engine import layouts_from_rust
    from polytopia_bot.ruleset import rules_json, training_round_limit
    batch = GameBatch(args.size, args.games, rules_json(), ENGINE_REVISION, training_round_limit(args.size), True, 10_000,
                      args.players, False)
    seeds = np.arange(args.games, dtype=np.int32) + 1_000_000 * (args.seed + 1)
    layouts = layouts_from_rust(batch, seeds)
    rows, sums = {}, {}
    for kernel in args.kernels.split(","):
        rows[kernel], sums[kernel] = run(kernel, args, layouts, seeds)
    cmp = dict(kernels=list(rows))
    if len(rows) == 2:
        a, b = list(rows)
        cmp["speedup"] = round(rows[a]["ms_mean"] / rows[b]["ms_mean"], 3)
        if args.check:
            diff = [k for k, (x, y) in enumerate(zip(sums[a], sums[b])) if x != y]
            cmp["checksums_equal"] = not diff
            cmp["first_differing_tick"] = diff[0] if diff else None
    print("TICKBENCH_CMP " + json.dumps(cmp), flush=True)
    return 0 if cmp.get("checksums_equal", True) else 1


if __name__ == "__main__":
    sys.exit(main())
