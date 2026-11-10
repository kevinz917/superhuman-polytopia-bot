"""The C/CUDA engine (engine-cu/, rl.gpu_engine.GpuGames) vs engine-rs (polytopia_engine_rs.GameBatch): bitwise-equal
status, done flags, rewards, packed masks, packed views and every State field in lockstep.

Both engines reset from the same GameBatch.layout(seed) layouts; every tick samples one legal action per game from
the reference's packed masks (uniform, end-turn boosted so games reach the round cap, a forced resignation every k
ticks when it is legal), feeds the same (family, source, target, seeds, layouts) to both and compares the tick
outputs bitwise; every `export_every` ticks it exports every State field of every game from both and compares those.
The module doubles as a CLI for Modal runs (experiments/modal_gpu_engine.py):

    python tests/test_gpu_engine.py --size 14 --games 24 --ticks 2000 --seed 0 --players 2 --end-boost 0.12 \\
        --resign-every 97 --export-every 50 [--device 0] [--cpu-vs-cuda] [--bench] \\
        [--policy /results/distributed/<run>/latest.npz --policy-temp 1.0 --epsilon 0.05]

--device 0: the candidate is libpc_cuda.so on CUDA device 0 (host-pointer mode); --cpu-vs-cuda: the reference is
libpc_cpu.so instead of Rust (CPU oracle vs CUDA). Prints `RESULT ok ...` or `RESULT FAIL ...` (exit 1). RESULT rows
carry the C build(s) played (`c_build=libpc_cuda_14.so/warp`: rl.gpu_engine.default_so_path picks the PC_MAXSIZE=14
build for boards <= 14, POLYTOPIA_PC_VARIANT=full the full one; PC_CUDA_WARP_TICK=1 the warp-per-game tick kernel).

--profile (2026-11-06): with POLYTOPIA_PC_PROFILE=1 (the lib*_prof.so builds) print one `PROFILE {json}` line per C
engine after the run: the tick's phases per game (GpuGames.profile_read: step, reset, masks per family, observe ...).

--policy PATH (a checkpoint .npz, or a run directory holding latest.npz): policy play. Uniform-random play never
leaves the opening (few units, no techs past tier 1, games end only at the round cap), so the gate never sees the
positions a trained policy reaches: long games, late techs, many units, captures, boats. In policy mode the actions
come from rl.network.sample on CPU JAX over the reference tick's `view` (the mover's packed int16 view ->
cpu_engine.view_features) and its packed masks (np.unpackbits -> rust_engine.split_masks), vmapped over the games with
one PRNG key per tick; with probability --epsilon a game plays a uniform-random legal action instead (end-turn boosted
as in random play), so odd actions stay in the mix, and the forced resignation applies on top. The checkpoint loads
through the production path (distributed.network_spec / checkpoint_template, runner.load_checkpoint): the features
are always the newest observation version and network.trunk fits them to the checkpoint's own version (fit_board /
fit_global), so observation-version 1, 2 and 3 checkpoints all play. The checkpoint's seat count must match
--players. --policy-temp T divides the logits (1 = the production sampler, 0 = deterministic argmax).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SIZE, GAMES, TICKS = 11, 8, 150
TICK_KEYS = ("status", "done", "terminated", "truncated", "rewards", "masks", "view", "opp_view", "final_view")
END, RESIGN = 15, 16   # rl.actions END / RESIGN families (one entry each: the last two catalog entries)


class Divergence(AssertionError):
    def __init__(self, tick, game, field, index, ref, cand, context=""):
        self.info = dict(tick=tick, game=game, field=field, index=index, rust=ref, c=cand)
        self.context = context
        super().__init__(f"tick={tick} game={game} field={field} index={index} rust={ref!r} c={cand!r} {context}")


def _norm(a):
    a = np.asarray(a)
    return a.view(np.uint8) if a.dtype == np.bool_ else a


def first_diff(a, b):
    """None when bitwise equal (bool and uint8 0/1 count as the same dtype), else (index, a-value, b-value, count)."""
    a, b = _norm(a), _norm(b)
    if a.shape != b.shape:
        return ("shape", a.shape, b.shape, 1)
    if a.dtype != b.dtype:
        return ("dtype", str(a.dtype), str(b.dtype), 1)
    same = a == b
    if a.dtype.kind == 'f':
        same |= np.isnan(a) & np.isnan(b)
    if same.all():
        return None
    bad = np.argwhere(~same)
    idx = tuple(int(i) for i in bad[0])
    return (idx, a[idx].item(), b[idx].item(), int(len(bad)))


class RustRef:
    """engine-rs GameBatch behind the GpuGames tick/export shape (packed masks, both views)."""

    def __init__(self, size, games, players, max_turns, no_resign, no_peace, max_actions):
        from polytopia_engine_rs import GameBatch
        from polytopia_bot.gpu import ENGINE_REVISION
        from polytopia_bot.ruleset import rules_json
        self.batch = GameBatch(size, games, rules_json(), ENGINE_REVISION, max_turns, no_resign, max_actions, players, no_peace)
        self.players, self.games, self.view_width = players, games, self.batch.view_width

    def reset(self, seeds, layouts):
        self.batch.reset_layouts(seeds, *layouts)

    def tick(self, f, s, t, seeds, layouts, both_views=True):
        return dict(self.batch.tick(f, s, t, seeds, both_views, layouts, True))

    def legal_masks(self):
        return self.batch.legal_masks(packed=True)

    def observe(self):
        return self.batch.observe()

    def observe_others(self):
        opp = self.batch.observe_opponent()
        return opp.reshape(self.games, self.players - 1, self.view_width) if self.players > 2 else opp

    def export_all(self):
        return dict(self.batch.export())


class CRef:
    """GpuGames behind the same shape."""

    def __init__(self, size, games, players, max_turns, no_resign, no_peace, max_actions, device):
        from polytopia_bot.rl.gpu_engine import GpuGames
        self.g = GpuGames(size, games, players, max_turns if max_turns is not None else 0, no_resign, no_peace, max_actions,
                          device=device)
        self.players, self.games, self.view_width = players, games, self.g.view_width
        info = self.g.build_info()
        self.build = f"{self.g.so_path.name}/{info['kernel']}" if info else self.g.so_path.name

    def reset(self, seeds, layouts):
        self.g.reset(seeds, layouts)

    def tick(self, f, s, t, seeds, layouts, both_views=True):
        return self.g.tick(f, s, t, seeds, layouts, both_views)

    def legal_masks(self):
        return self.g.legal_masks(packed=True)

    def observe(self):
        return self.g.observe()

    def observe_others(self):
        """GameBatch.observe_opponent: the other seats in turn order after the mover, from per-seat pc_observe calls."""
        cp = np.stack([self.g.export(g)["current_player"] for g in range(self.games)]).reshape(self.games)
        np_ = self.players
        by_seat = {seat: self.g.observe(seat) for seat in range(1, np_ + 1)}   # [G, vw] each
        out = np.empty((self.games, np_ - 1, self.view_width), np.int16)
        for k in range(1, np_):
            seats = (cp - 1 + k) % np_ + 1
            for g in range(self.games):
                out[g, k - 1] = by_seat[int(seats[g])][g]
        return out.reshape(self.games, self.view_width) if np_ == 2 else out

    def export_all(self):
        return self.g.export_all()


def sample_actions(masks, size, mask_width, rng, end_boost, resign_game=None):
    """One legal action per game from packed masks (numpy.packbits order): uniform over legal entries, the end-turn
    entry with probability end_boost when legal, and for `resign_game` the resign entry when legal."""
    from polytopia_bot.rl.actions import unflatten
    bits = np.unpackbits(np.asarray(masks, np.uint8), axis=1)[:, :mask_width]
    end_at, resign_at = mask_width - 2, mask_width - 1
    fam, src, tgt = [], [], []
    for g in range(bits.shape[0]):
        legal = np.flatnonzero(bits[g])
        if len(legal) == 0:
            raise RuntimeError(f"game {g}: no legal action")
        if resign_game == g and bits[g, resign_at]:
            pick = resign_at
        elif bits[g, end_at] and rng.random() < end_boost:
            pick = end_at
        else:
            pick = int(legal[rng.integers(len(legal))])
        c = unflatten(pick, size)
        fam.append(c.family); src.append(c.source); tgt.append(c.target)
    return np.array(fam, np.int32), np.array(src, np.int32), np.array(tgt, np.int32)


class PolicySampler:
    """Actions of a trained checkpoint for the lockstep gate: rl.network.sample on CPU JAX over the reference's packed
    view and masks, vmapped over the games (the production actors' path: cpu_engine.view_features + split_masks)."""

    def __init__(self, path, size, players, games, view_width, mask_width, seed, temp=1.0):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        import jax.numpy as jnp
        from polytopia_bot.rl import network
        from polytopia_bot.rl.actions import Choice, family_mask, source_mask, target_mask
        from polytopia_bot.rl.cpu_engine import view_features, view_layout
        from polytopia_bot.rl.distributed import checkpoint_template, network_spec
        from polytopia_bot.rl.runner import load_checkpoint
        path = Path(path)
        if path.is_dir():
            path = path / "latest.npz"
        self.path = path
        self.name = f"{path.parent.name}/{path.stem}"
        meta = json.loads(path.with_suffix(".json").read_text())
        self.config = meta["config"]
        self.spec = network_spec(path)
        if self.config.get("players", 2) != players:
            raise ValueError(f"checkpoint {path} is a {self.config.get('players', 2)}-seat network; the harness plays {players} seats")
        if self.config.get("size", size) != size:
            print(json.dumps(dict(note="checkpoint board size differs from the harness board; the network is board-size independent",
                                  checkpoint_size=self.config.get("size"), size=size)), flush=True)
        t0 = time.perf_counter()
        self.params = load_checkpoint(path, checkpoint_template(path))
        self.load_s = round(time.perf_counter() - t0, 2)
        layout = view_layout(size, players)
        if sum(int(np.prod(s)) for _, s in layout) != view_width:
            raise RuntimeError("packed view width differs between the engine and cpu_engine.view_layout")
        self.size, self.games, self.mask_width, self.temp = size, games, mask_width, float(temp)
        self.key = jax.random.key(seed)

        def act(params, view, masks, key):
            features = view_features(view, layout, size)
            if self.temp == 1.0:
                return network.sample(params, features, masks, key, size=size)[0]
            if self.temp <= 0.0:
                return network.sample(params, features, masks, key, size=size, deterministic=True)[0]
            # network.sample with the logits of every head divided by the temperature
            tiles, g, _ = network.trunk(params, features)
            keys = jax.random.split(key, 3)

            def draw(logits, mask, k):
                return jax.random.categorical(k, network.log_distribution(logits / self.temp, mask)).astype(jnp.int32)
            fm = family_mask(masks)
            f = draw(network.dense(g, params["family"]), fm, keys[0])
            sm = source_mask(masks, f, size)
            s = draw(network.dense(tiles, params["source"])[:, f], sm, keys[1])
            tm = target_mask(masks, f, s, size)
            t = draw(network.target_logits(params, tiles, g, f, s), tm, keys[2])
            return Choice(f, s, t)
        self._act = jax.jit(jax.vmap(act, in_axes=(None, 0, 0, 0)))
        self._jax = jax
        self.compile_s = None

    def __call__(self, view, masks, tick):
        """(family, source, target) int32 arrays for every game from the packed masks and the mover's packed view."""
        from polytopia_bot.rl.rust_engine import split_masks
        bits = np.unpackbits(np.asarray(masks, np.uint8), axis=1)[:, :self.mask_width]
        split = split_masks(np.ascontiguousarray(bits), self.size)
        keys = self._jax.random.split(self._jax.random.fold_in(self.key, tick), self.games)
        t0 = time.perf_counter()
        choice = self._act(self.params, np.ascontiguousarray(np.asarray(view, np.int16)), split, keys)
        f, s, t = (np.asarray(a, np.int32) for a in choice)
        if self.compile_s is None:
            self.compile_s = round(time.perf_counter() - t0, 2)
        return f, s, t


def mix_actions(policy_actions, random_actions, masks, mask_width, rng, epsilon, resign_game):
    """Per game the policy's action, a uniform-random one with probability epsilon, and the forced resignation on top
    (when legal); returns the arrays and the number of games that took the policy's action."""
    f, s, t = (a.copy() for a in policy_actions)
    rf, rs, rt = random_actions
    random_pick = rng.random(len(f)) < epsilon
    if resign_game is not None:
        bits = np.unpackbits(np.asarray(masks[resign_game:resign_game + 1], np.uint8), axis=1)[0, :mask_width]
        if bits[mask_width - 1]:
            random_pick[resign_game] = True   # sample_actions already picked the resign entry for this game
    f[random_pick], s[random_pick], t[random_pick] = rf[random_pick], rs[random_pick], rt[random_pick]
    return f, s, t, int((~random_pick).sum())


def compare_tick(k, ref, cand, context):
    for key in TICK_KEYS:
        if key not in ref and key not in cand:
            continue
        if (key in ref) != (key in cand):
            raise Divergence(k, -1, key, "present", key in ref, key in cand, context)
        d = first_diff(ref[key], cand[key])
        if d:
            idx, a, b, count = d
            game = idx[0] if isinstance(idx, tuple) and len(idx) else -1
            raise Divergence(k, game, key, idx, a, b, f"{context} ({count} differing entries)")


def compare_exports(k, ref, cand, context):
    if list(ref.keys()) != list(cand.keys()):
        raise Divergence(k, -1, "export keys", "order", list(ref.keys()), list(cand.keys()), context)
    for name in ref:
        d = first_diff(ref[name], cand[name])
        if d:
            idx, a, b, count = d
            game = idx[0] if isinstance(idx, tuple) and len(idx) else -1
            raise Divergence(k, game, name, idx, a, b, f"{context} ({count} differing entries)")


def play(size=SIZE, games=GAMES, ticks=TICKS, seed=0, players=2, end_boost=0.12, resign_every=97, export_every=50,
         device=-1, cpu_vs_cuda=False, no_resign=True, no_peace=False, max_turns=None, bench=False, verbose=False,
         policy=None, policy_temp=1.0, epsilon=0.05, profile=False):
    """Lockstep play; raises Divergence with the first differing value. Returns the RESULT stats. policy: a checkpoint
    path (PolicySampler) whose actions replace the uniform-random ones except for an epsilon share of the games."""
    from polytopia_bot.rl.actions import offsets
    from polytopia_bot.rl.gpu_engine import layouts_from_rust, mask_width
    from polytopia_bot.ruleset import training_round_limit
    limit = training_round_limit(size) if max_turns is None else (None if max_turns == 0 else int(max_turns))
    n = size * size
    mw = mask_width(n)
    assert mw == int(offsets(size)[-1]), "gpu_engine.family_widths disagrees with rl.actions.shapes"
    rng = np.random.default_rng(seed)
    if cpu_vs_cuda:
        ref = CRef(size, games, players, limit, no_resign, no_peace, 10_000, -1)
        cand = CRef(size, games, players, limit, no_resign, no_peace, 10_000, device if device >= 0 else 0)
        rust = RustRef(size, games, players, limit, no_resign, no_peace, 10_000)   # layouts only
        layout_batch = rust.batch
    else:
        ref = RustRef(size, games, players, limit, no_resign, no_peace, 10_000)
        cand = CRef(size, games, players, limit, no_resign, no_peace, 10_000, device)
        layout_batch = ref.batch
    base = 1_000_000 * (seed + 1)
    seeds = np.arange(games, dtype=np.int32) + base
    layouts = layouts_from_rust(layout_batch, seeds)
    ref.reset(seeds, layouts)
    cand.reset(seeds, layouts)
    sampler = None
    if policy:
        sampler = PolicySampler(policy, size, players, games, ref.view_width, mw, seed, policy_temp)
    stats = dict(size=size, games=games, players=players, ticks=0, steps=0, finished=0, terminated=0, truncated=0,
                 nonok=0, exports=0, resigns=0, device=device, mode="cpu-vs-cuda" if cpu_vs_cuda else "rust-vs-c",
                 policy=sampler.name if sampler else "random", policy_actions=0, max_turn=0,
                 c_build=",".join(e.build for e in (ref, cand) if isinstance(e, CRef)))
    if sampler:
        stats.update(policy_temp=policy_temp, epsilon=epsilon, obs_version=sampler.config.get("obs_version", 1),
                     width=sampler.spec["width"], blocks=sampler.spec["blocks"])
    if not bench:
        first = dict(masks=ref.legal_masks(), view=ref.observe(), opp_view=ref.observe_others())
        second = dict(masks=cand.legal_masks(), view=cand.observe(), opp_view=cand.observe_others())
        compare_tick(-1, first, second, "after reset")
        compare_exports(-1, ref.export_all(), cand.export_all(), "after reset")
        stats["exports"] += 1
        masks, view = first["masks"], first["view"]
    else:
        masks, view = ref.legal_masks(), (ref.observe() if sampler else None)
    next_seed = base + games
    t_ref = t_cand = t_policy = 0.0
    t0 = time.monotonic()
    for k in range(ticks):
        resign_game = (k % games) if (resign_every and k % resign_every == resign_every - 1) else None
        f, s, t = sample_actions(masks, size, mw, rng, end_boost, resign_game)
        if sampler:
            tp = time.perf_counter()
            f, s, t, took = mix_actions(sampler(view, masks, k), (f, s, t), masks, mw, rng, epsilon, resign_game)
            t_policy += time.perf_counter() - tp
            stats["policy_actions"] += took
        if resign_game is not None and f[resign_game] == RESIGN:
            stats["resigns"] += 1
        reset_seeds = np.arange(games, dtype=np.int32) + next_seed
        reset_layouts = layouts_from_rust(layout_batch, reset_seeds)
        t1 = time.perf_counter()
        out_ref = ref.tick(f, s, t, reset_seeds, reset_layouts)
        t2 = time.perf_counter()
        out_cand = cand.tick(f, s, t, reset_seeds, reset_layouts)
        t3 = time.perf_counter()
        t_ref += t2 - t1
        t_cand += t3 - t2
        if not bench:
            context = f"action family={f.tolist()} source={s.tolist()} target={t.tolist()} status_rust={np.asarray(out_ref['status']).tolist()}"
            compare_tick(k, out_ref, out_cand, context)
            if export_every and k % export_every == export_every - 1:
                compare_exports(k, ref.export_all(), cand.export_all(), context)
                stats["exports"] += 1
        masks, view = out_ref["masks"], out_ref["view"]
        done = np.asarray(out_ref["done"])
        stats["finished"] += int(done.sum())
        stats["terminated"] += int(np.asarray(out_ref["terminated"]).sum())
        stats["truncated"] += int(np.asarray(out_ref["truncated"]).sum())
        stats["nonok"] += int((np.asarray(out_ref["status"]) != 0).sum())
        stats["ticks"] += 1
        stats["steps"] += games
        next_seed += games
        if sampler and (k % 50 == 49 or k == ticks - 1):
            # the turn index of the running games (export "turn" is per game): how deep into the game the gate gets
            turns = np.asarray(ref.export_all()["turn"]).reshape(-1) if not bench else np.zeros(1, np.int32)
            stats["max_turn"] = max(stats["max_turn"], int(turns.max()))
        if verbose and k % 200 == 199:
            print(json.dumps(dict(tick=k + 1, seconds=round(time.monotonic() - t0, 1),
                                  **{a: stats[a] for a in ("finished", "terminated", "truncated", "nonok", "max_turn")})), flush=True)
    stats["seconds"] = round(time.monotonic() - t0, 2)
    stats["ref_ms_per_tick"] = round(1e3 * t_ref / max(1, ticks), 4)
    stats["c_ms_per_tick"] = round(1e3 * t_cand / max(1, ticks), 4)
    if sampler:
        stats["policy_ms_per_tick"] = round(1e3 * t_policy / max(1, ticks), 2)
        stats["policy_compile_s"] = sampler.compile_s
        stats["policy_load_s"] = sampler.load_s
    stats["ref_steps_per_s"] = round(games * ticks / t_ref) if t_ref else None
    stats["c_steps_per_s"] = round(games * ticks / t_cand) if t_cand else None
    if profile:
        for role, e in (("ref", ref), ("cand", cand)):
            if isinstance(e, CRef):
                prof = e.g.profile_read()
                print("PROFILE " + json.dumps(dict(role=role, build=e.build, size=size, players=players,
                                                   policy=stats["policy"], profile=prof)), flush=True)
    return stats


def test_gpu_engine_matches_rust():
    import pytest
    pytest.importorskip("polytopia_engine_rs")
    from polytopia_bot.rl.gpu_engine import default_so_path
    if not default_so_path(-1).exists():
        pytest.skip("engine-cu not built (make -C engine-cu cpu)")
    stats = play(export_every=25)
    assert stats["steps"] == GAMES * TICKS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=14)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--end-boost", type=float, default=0.12)
    ap.add_argument("--resign-every", type=int, default=97)
    ap.add_argument("--export-every", type=int, default=50)
    ap.add_argument("--device", type=int, default=-1, help="-1: libpc_cpu.so; else libpc_cuda.so on that CUDA device")
    ap.add_argument("--cpu-vs-cuda", action="store_true", help="reference = libpc_cpu.so instead of engine-rs")
    ap.add_argument("--no-resign", type=int, default=1, help="mask option (RustGames default 1); 0 lets forced resigns through")
    ap.add_argument("--no-peace", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=None, help="None: the board's training round limit; 0: none")
    ap.add_argument("--bench", action="store_true", help="time the two engines' ticks, no comparison")
    ap.add_argument("--policy", default="", help="checkpoint .npz (or a run directory with latest.npz): policy play (module docstring)")
    ap.add_argument("--policy-temp", type=float, default=1.0, help="logit temperature of the policy (1 = production sampler, 0 = argmax)")
    ap.add_argument("--epsilon", type=float, default=0.05, help="per game, the probability of a uniform-random legal action instead of the policy's")
    ap.add_argument("--profile", action="store_true", help="print the C engines' phase profile (POLYTOPIA_PC_PROFILE=1 builds)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    pin = os.environ.get("PIN_CORE")
    if pin and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(pin)})
    try:
        out = play(args.size, args.games, args.ticks, args.seed, args.players, args.end_boost, args.resign_every,
                   args.export_every, args.device, args.cpu_vs_cuda, bool(args.no_resign), bool(args.no_peace),
                   args.max_turns, args.bench, verbose=not args.quiet, policy=args.policy or None,
                   policy_temp=args.policy_temp, epsilon=args.epsilon, profile=args.profile)
        out.update(seed=args.seed, end_boost=args.end_boost, resign_every=args.resign_every, export_every=args.export_every,
                   bench=args.bench)
        print("RESULT ok " + " ".join(f"{k}={v}" for k, v in out.items()), flush=True)
        return 0
    except Divergence as e:
        policy = f"{Path(args.policy).parent.name}/{Path(args.policy).stem}" if args.policy else "random"
        print(f"RESULT FAIL " + " ".join(f"{k}={v}" for k, v in e.info.items()) + f" seed={args.seed} size={args.size} "
              f"players={args.players} policy={policy} context={e.context}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
