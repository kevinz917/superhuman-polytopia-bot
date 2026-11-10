"""GPU-resident PPO on the CUDA engine with the PRODUCTION JAX network and PPO code (rl.network, rl.ppo,
rl.cpu_engine.view_features): thousands of games stepped by one kernel into torch device tensors, every tensor handed
to JAX zero-copy through dlpack, features / masks / sampling / GAE / update all jitted on the same card. torch is only
the allocator and the bridge to the engine (GpuGames.tick_device); no CPU actors, no sockets, no batch assembly. Runs
on Modal (experiments/modal_gpu_engine.py::jax_train / ::launch), never locally.

Two modes:

  * the recipe trainer (default): rl.gpu_resident.GpuResidentTrainer runs the hero-1v1 recipe (--config
    experiments/hero/hero-1v1.json: omniscient critic, same-seat 32-decision segments, turn-clock GAE, cap score,
    opponent pool, LR schedule, production checkpoint layout under /results/distributed/{run-id}); --parent continues a
    checkpoint (its architecture), else --width/--blocks/--architecture initialise a fresh network:

        python experiments/gpu_resident_train.py --config experiments/hero/hero-1v1.json \\
            --parent /results/distributed/hero-1v1-cont1h/latest.npz \\
            --critic-path /results/distributed/hero-1v1-cont1h/latest-critic.npz --run-id gpu-smoke --seconds 600 \\
            --games 4096 --sets 4 --ring 160 --grad-accum 4

    --devices N trains data parallel on N GPUs of one machine (rl.gpu_resident module docstring, "Several GPUs"): each
    device runs its own --games games (per device) and ring, the update's rows are summed over the devices, the
    minibatch is split over them and the gradients are pmean'd before one Adam step (production's learner step).
    Example, 4 x H100 (per-device minibatch 4096 = production's 4-GPU hero learner, no accumulation needed):

        python experiments/gpu_resident_train.py --devices 4 --parent ... --critic-path ... --run-id gpu-dp4 --seconds 600

    The critic values of the rows are computed during the rollout and kept in the ring (rl.gpu_resident module
    docstring, value_mode); --value-at-assembly restores the production-shaped value pass at assembly, --stale-values
    skips the re-score of rows valued by the previous critic, --value-check runs both and reports max |diff|.
    Checkpoints are written by a background thread (host copy on the loop, compression / files / commit off it).
    Each set's encode + critic + sampling is one fused jit per step (timing key `fused`); --unfused keeps the separate
    calls for A/B. The start line prints games_per_set: --sets 2 (2048 per set) halves the per-set fixed cost against a
    partly exposed engine, --sets 4 (default) hides it.

  * --policy-only: the policy-only self-play benchmark below (the 97.5k samples/s H100 number), unchanged:

        python experiments/gpu_resident_train.py --policy-only --games 4096 --size 14 --steps 32 --seconds 120 \\
            --width 128 --blocks 4 --architecture normalized-v2 --minibatch 4096 --epochs 1 --bf16-trunk --check-dlpack

--sets N overlaps the engine with the policy: the games are split into N independent GpuGames batches (each its own
PcBatch and CUDA stream, torch output buffers, layout rows and seed range; games // N each), and the rollout is a round
robin: while the other sets' ticks run (pc_tick_async), set k is synced, encoded, valued, sampled in JAX, bookkept and
its tick launched, then set k+1's, ... so each set's JAX work hides behind N-1 kernels (recipe default 4: three sets'
JAX work covers one set's tick). GAE and the update see one batch of games (a set is just a range of game columns).
--sets 1 (policy-only default) is the sequential loop: act, synchronous tick, bookkeeping, encode.

Conventions copied from rl/distributed.py (the hero recipe, experiments/hero/hero-1v1.json):
  * features: view_features(packed int16 view, view_layout(size, 2), size) = observation.encode_view (OBS_VERSION 3);
    the learner re-encodes them from the stored views per minibatch (to_batch).
  * masks in the loss: the PathMasks (family 17 / source n / target max(n, 128)) network.sample returns, exactly what
    the learner unpacks from the rows' family_mask / source_mask / target_mask; the dense engine masks are only
    unpacked (jnp.unpackbits + rust_engine.split_masks in jnp) for sampling.
  * GAE (--gae seat, default): per-seat sequences as SeatTrajectories / segment_gae: next_value is the SAME seat's
    next decision value, the seat's terminal reward lands on its own last decision, a truncation (round cap) bootstraps
    the seat's value of the final state (engine final_view; hero's cap_score_reward needs the score, which the view
    does not carry), and the turn clock (turn_gamma .99 / turn_lambda .95: gamma and lambda apply only on end-turn
    decisions, 1 inside a turn; --no-turn-clock: gamma .995 / lambda .99 per decision). At the segment boundary the
    seat to move bootstraps on the value of the new state and the other seat on its negation (zero-sum stand-in for
    the next decision of that seat, which production waits for). --gae alternating: ppo.advantages, the alternating-
    seat formulation (same_player between consecutive decisions, next_value = the next decision's value, sign flip
    on a seat change), as the torch smoke did.
  * advantages normalized over the whole batch with the advantage floor (ppo.normalize_advantages, floor .05), then
    ppo.make_update (clip .2, value_coef .5, entropy .01, Adam 3e-4, grad clip .5) over minibatches, `epochs` passes.
  * the policy's own value head is the critic (hero trains an omniscient critic on both views; both_views=False here).
Prints one line per report interval and a final `RESULT {...}` JSON line.
"""
import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-only", action="store_true", help="the policy-only self-play benchmark (the old loop)")
    # ---- recipe trainer (rl.gpu_resident)
    ap.add_argument("--config", default="experiments/hero/hero-1v1.json", help="LearnerConfig JSON (the recipe)")
    ap.add_argument("--parent", default="", help="policy checkpoint to continue (sets the architecture); '' = random init")
    ap.add_argument("--critic-path", default=None, help="critic checkpoint (overrides the config's critic_path)")
    ap.add_argument("--run-id", default=None, help="run folder name under --run-root (default gpu-<mmdd-HHMM>)")
    ap.add_argument("--run-root", default="/results/distributed")
    ap.add_argument("--ring", type=int, default=160, help="steps an open segment's rows stay addressable before a forced cut "
                                                          "(the lever against forced_close_fraction; ring memory is in the start line)")
    ap.add_argument("--batch", type=int, default=None, help="rows per update (default: the config's; smokes on small GPUs)")
    ap.add_argument("--grad-accum", type=int, default=None, help="micro-batches per minibatch (same step, less memory): 4 = 4096-row "
                                                                 "micro-batches of the 16384 minibatch, 2 = 8192, 1 = the whole minibatch")
    ap.add_argument("--value-at-assembly", action="store_true", help="recipe: the production value pass at assembly instead of the "
                                                                     "in-rollout critic values (value_mode assembly)")
    ap.add_argument("--stale-values", action="store_true", help="recipe: keep the in-rollout values of rows valued by the previous "
                                                                "critic instead of re-scoring them at assembly (value_mode rollout-stale)")
    ap.add_argument("--value-check", action="store_true", help="recipe: also run the assembly value pass and report max |diff| per update")
    ap.add_argument("--unfused", action="store_true", help="recipe: the separate encode / critic / sample jits per set per step instead "
                                                           "of the one fused call (A/B timing; same arithmetic)")
    ap.add_argument("--checkpoint-every", type=int, default=None)
    ap.add_argument("--snapshot-every", type=int, default=None)
    ap.add_argument("--map-table", default=None, help="precomputed map table .npz (default: /results/maps/table-...)")
    ap.add_argument("--updates", type=int, default=0, help="stop after this many updates (config training_updates)")
    ap.add_argument("--block", type=int, default=24, help="lanes per opponent block")
    ap.add_argument("--fp32-trunk", action="store_true", help="recipe mode: fp32 conv torso instead of the learner's bf16")
    ap.add_argument("--dp-threads", type=int, default=None, choices=(0, 1),
                    help="recipe, --devices > 1: one host thread per device for the rollout and its share of the assembly "
                         "(1, the default) or the lockstep loop on the main thread (0, A/B); ignored with one device")
    ap.add_argument("--coordinator", default="", help="multi-process: jax.distributed coordinator host:port (rank 0)")
    ap.add_argument("--num-processes", type=int, default=1, help="multi-process: one process per container")
    ap.add_argument("--process-id", type=int, default=0, help="multi-process: this process's rank")
    ap.add_argument("--devices", type=int, default=1, help="recipe: data-parallel GPUs in this process (0 = all visible). "
                                                          "--games / --sets / --ring are PER DEVICE; the update takes batch rows "
                                                          "summed over the devices and splits the minibatch over them "
                                                          "(production's per-device minibatch // N, pmean'd gradients)")
    # ---- shared / policy-only
    ap.add_argument("--games", type=int, default=4096)
    ap.add_argument("--sets", type=int, default=None, help="independent engine batches stepped in a round robin (games // sets each; "
                                                            "default 4 in recipe mode, 1 policy-only)")
    ap.add_argument("--size", type=int, default=14)
    ap.add_argument("--steps", type=int, default=32, help="policy-only: decisions per game per PPO iteration (hero segment 32)")
    ap.add_argument("--iters", type=int, default=0, help="policy-only: stop after this many iterations (0: by --seconds)")
    ap.add_argument("--seconds", type=float, default=120.0, help="stop after this wall time (recipe: training_seconds)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--architecture", default="normalized-v2",
                    help="legacy | normalized-v2 | transformer-v1 | transformer-v2 | transformer-v3[-p2][-c<k>] "
                         "(recipe mode, fresh networks; policy-only mode: legacy / normalized-v2)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--minibatch", type=int, default=None, help="decisions per PPO minibatch (recipe: the config's 16384; "
                                                                 "policy-only default 4096)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--pool", type=int, default=512, help="policy-only: layouts generated once (game g uses row g %% pool)")
    ap.add_argument("--bf16-trunk", action="store_true", help="POLYTOPIA_TRUNK_DTYPE=bfloat16 (the hero learner's setting; "
                                                              "recipe mode default)")
    ap.add_argument("--matmul-precision", default="DEFAULT", choices=("DEFAULT", "HIGH", "HIGHEST"),
                    help="POLYTOPIA_MATMUL_PRECISION for the conv torso (hero learner: DEFAULT = TF32)")
    ap.add_argument("--head-precision", default="HIGHEST", choices=("HIGHEST", "DEFAULT"),
                    help="POLYTOPIA_HEAD_PRECISION for the dense heads and the target key-query product")
    ap.add_argument("--gae", default="seat", choices=("seat", "alternating"))
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.99)
    ap.add_argument("--turn-gamma", type=float, default=0.99)
    ap.add_argument("--turn-lambda", type=float, default=0.95)
    ap.add_argument("--no-turn-clock", action="store_true")
    ap.add_argument("--advantage-floor", type=float, default=0.05)
    ap.add_argument("--report-every", type=int, default=1)
    ap.add_argument("--check-dlpack", action="store_true")
    ap.add_argument("--jax-mem-fraction", default=None, help="XLA_PYTHON_CLIENT_MEM_FRACTION: JAX's share of the card (torch owns the "
                                                              "row ring and the engine's buffers). Given: overrides the environment; "
                                                              "default .45 recipe (.5 policy-only) unless the environment sets it")
    ap.add_argument("--seed", type=int, default=None, help="recipe: overrides the config seed; policy-only default 0")
    return ap.parse_args(argv)


def main(argv=None, commit=None):
    args = parse_args(argv)
    recipe = not args.policy_only
    # torch and JAX share the card: no preallocation, JAX capped at a fraction. Read by jax at import.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.jax_mem_fraction:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.jax_mem_fraction)
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".45" if recipe else ".5")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    # rl.network reads these at import
    os.environ["POLYTOPIA_MATMUL_PRECISION"] = args.matmul_precision
    os.environ["POLYTOPIA_HEAD_PRECISION"] = args.head_precision
    if args.bf16_trunk or (recipe and not args.fp32_trunk):
        os.environ["POLYTOPIA_TRUNK_DTYPE"] = "bfloat16"
    if recipe and args.num_processes > 1:
        # multi-container data parallel (docs/claude-gpu-resident-multinode-design.md): before any JAX device use
        import jax
        jax.distributed.initialize(coordinator_address=args.coordinator, num_processes=args.num_processes,
                                   process_id=args.process_id)
    if recipe:
        return recipe_main(args, commit)
    return policy_only_main(args)


def recipe_main(args, commit=None):
    """The hero recipe on one GPU (rl.gpu_resident.GpuResidentTrainer); prints one JSON line per update and RESULT."""
    import torch
    import jax
    if not torch.cuda.is_available():
        print("RESULT " + json.dumps({"ok": False, "error": "no CUDA device (torch)"}))
        sys.exit(2)
    if jax.default_backend() != "gpu":
        print("RESULT " + json.dumps({"ok": False, "error": f"jax backend {jax.default_backend()} is not gpu"}))
        sys.exit(2)
    from polytopia_bot.rl.gpu_resident import GpuResidentTrainer
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = ROOT / cfg_path
    config = json.loads(cfg_path.read_text())
    if args.updates:
        config["training_updates"] = int(args.updates)
    run_id = args.run_id or time.strftime("gpu-%m%d-%H%M")
    run_dir = Path(args.run_root) / run_id
    network = None if args.parent else dict(width=args.width, blocks=args.blocks, architecture=args.architecture, size=args.size)
    if args.value_at_assembly and args.stale_values:
        raise SystemExit("--value-at-assembly and --stale-values exclude each other")
    value_mode = "assembly" if args.value_at_assembly else "rollout-stale" if args.stale_values else "rollout"
    trainer = GpuResidentTrainer(config, run_dir, args.parent, critic_path=args.critic_path, training_seconds=args.seconds,
                                 games=args.games, sets=args.sets or 4, ring=args.ring, minibatch=args.minibatch,
                                 batch=args.batch, grad_accum=args.grad_accum, checkpoint_every=args.checkpoint_every,
                                 snapshot_every=args.snapshot_every, map_table=args.map_table, seed=args.seed,
                                 block=args.block, network=network, value_mode=value_mode, value_check=args.value_check,
                                 fused=not args.unfused, devices=args.devices, dp_threads=args.dp_threads, log=lambda s: print(s, flush=True),
                                 commit=commit or (lambda: None))
    try:
        trainer.run()
    finally:
        trainer.close()
    print("RESULT " + json.dumps(trainer.result), flush=True)
    return trainer.result


def policy_only_main(args):
    """The policy-only self-play benchmark (module docstring): the policy's own value head is the critic, both_views off."""
    args.minibatch = args.minibatch or 4096
    args.sets = args.sets or 1
    args.seed = args.seed or 0
    import numpy as np
    import torch
    import torch.utils.dlpack
    import jax
    import jax.numpy as jnp
    import jax.dlpack

    if not torch.cuda.is_available():
        print("RESULT " + json.dumps({"ok": False, "error": "no CUDA device (torch)"}))
        sys.exit(2)
    if jax.default_backend() != "gpu":
        print("RESULT " + json.dumps({"ok": False, "error": f"jax backend {jax.default_backend()} is not gpu"}))
        sys.exit(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    from polytopia_engine_rs import GameBatch
    from polytopia_bot.gpu import ENGINE_REVISION
    from polytopia_bot.gpu.actions import Masks
    from polytopia_bot.ruleset import rules_json
    from polytopia_bot.rl import network, ppo
    from polytopia_bot.rl.actions import END, Choice, PathMasks, leaf_shapes
    from polytopia_bot.rl.cpu_engine import view_features, view_layout
    from polytopia_bot.rl.gpu_engine import GpuGames, layouts_from_rust

    G, size, T, sets = args.games, args.size, args.steps, args.sets
    if sets < 1 or G % sets:
        raise ValueError("--games must be a positive multiple of --sets")
    Gs = G // sets
    n = size * size
    N = T * G
    mb = args.minibatch
    if N % mb:
        print(f"note: {N % mb} of {N} decisions per iteration are dropped (steps*games not a multiple of minibatch)")
    turn_clock = not args.no_turn_clock

    # ---- dlpack bridge (zero-copy both ways) ----
    def to_jax(t):
        """torch CUDA tensor -> JAX array aliasing the same memory (JAX syncs its stream on torch's current stream)."""
        return jax.dlpack.from_dlpack(t)

    def to_torch(a):
        jax.block_until_ready(a)
        return torch.utils.dlpack.from_dlpack(a)

    # ---- layouts: a pool generated once on the CPU by engine-rs mapgen, uploaded once; game g resets on row g % P ----
    t0 = time.perf_counter()
    P = min(args.pool, G)
    layout_batch = GameBatch(size, 1, rules_json(), ENGINE_REVISION)
    pool = layouts_from_rust(layout_batch, np.arange(P, dtype=np.int32) + 1_000, exact=False)
    layout_ms = (time.perf_counter() - t0) * 1000.0

    # ---- view layout (production) ----
    layout = view_layout(size, 2)
    vw = sum(int(np.prod(s)) for _, s in layout)
    offsets, at = {}, 0
    for name, shape in layout:
        offsets[name] = at
        at += int(np.prod(shape))
    player_at = offsets["player"]
    # rollout views of every set: set k owns the game columns [k*Gs, (k+1)*Gs) (one tensor, handed to JAX once per update)
    buf_views = torch.zeros((T, G, vw), dtype=torch.int16, device=device)

    class GameSet:
        """One GpuGames batch (its own PcBatch, CUDA stream, output tensors, layout rows and seed range) and the
        per-step state the rollout keeps for it."""

        def __init__(self, k):
            self.k = k
            lo = k * Gs
            self.env = GpuGames(size=size, games=Gs, players=2, device=0)
            if vw != self.env.view_width:
                raise RuntimeError(f"packed view width differs: production layout {vw} vs C engine {self.env.view_width}")
            rows = (np.arange(Gs) + lo) % P                       # the same rows game g gets with one set
            host_layouts = tuple(np.ascontiguousarray(a[rows]) for a in pool)
            self.dev_layouts = tuple(torch.from_numpy(a).to(device) for a in host_layouts)
            seeds_host = np.arange(Gs, dtype=np.int32) + lo       # disjoint ranges; += G per iteration keeps them so
            self.env.reset(seeds_host, host_layouts)
            self.seeds = torch.from_numpy(seeds_host).to(device)
            self.out = self.env.alloc_device_outputs(device)
            self.out["opp_view"] = None                           # both_views=False: the mover's view only
            # the first state's masks and views come from the host-pointer calls once, into the same device tensors the
            # engine writes afterwards (one aliasing path for every step)
            self.out["masks"].copy_(torch.from_numpy(self.env.legal_masks(packed=True)))
            self.out["view"].copy_(torch.from_numpy(self.env.observe()))
            self.inputs = tuple(torch.zeros(Gs, dtype=torch.int32, device=device) for _ in range(3))
            self.views = buf_views[:, lo:lo + Gs]                 # [T, Gs, vw] view into the shared rollout tensor
            self.ep_len = jnp.zeros(Gs, jnp.float32)
            self.features = self.player = None                    # encoded current state (set by finish)
            self.in_flight = False                                # a pc_tick_async queued and not yet synced
            self.pending = None                                   # the step index whose outputs the queued tick produces
            self.store = None

    gsets = [GameSet(k) for k in range(sets)]
    mask_width = gsets[0].env.mask_width
    mask_bytes = gsets[0].env.mask_bytes
    torch.cuda.synchronize()

    # ---- JAX side: features, masks, network, GAE, update (all jitted) ----
    encode_one = lambda v: view_features(v, layout, size)

    @jax.jit
    def encode(views):
        """[Gs, vw] int16 -> (Features with a leading game axis, mover seat [Gs] int32 from the view's player field)."""
        return jax.vmap(encode_one)(views), views[:, player_at].astype(jnp.int32)

    shapes = jax.eval_shape(lambda: encode_one(jnp.zeros(vw, jnp.int16)))
    board_channels, global_channels = shapes.board.shape[-1], shapes.global_state.shape[-1]
    key = jax.random.key(args.seed)
    key, k_init = jax.random.split(key)
    params = network.init(k_init, board_channels, global_channels, size=size, width=args.width, blocks=args.blocks,
                          architecture=args.architecture)
    opt = ppo.adam_init(params)
    n_params = int(sum(x.size for x in jax.tree.leaves(params)))

    def split_masks(bits):
        """[B, mask_width] bool -> Masks of [B, ...] (rust_engine.split_masks in jnp)."""
        leaves, at = [], 0
        for shape in leaf_shapes(size):
            k = int(np.prod(shape)) if shape else 1
            leaves.append(bits[:, at:at + k].reshape((bits.shape[0],) + tuple(shape)))
            at += k
        assert at == mask_width, (at, mask_width)
        return Masks(*leaves)

    sample_one = functools.partial(network.sample, size=size)

    @jax.jit
    def act(params, features, packed, key):
        """Dense masks from the packed bits, then the production sampler per game: choice, path masks, logp, value."""
        bits = jnp.unpackbits(packed, axis=1)[:, :mask_width].astype(bool)
        masks = split_masks(bits)
        keys = jax.random.split(key, Gs)
        return jax.vmap(sample_one, in_axes=(None, 0, 0, 0))(params, features, masks, keys)

    @jax.jit
    def value_of(params, features):
        return jax.vmap(lambda f: network.trunk(params, f)[2])(features)

    @jax.jit
    def final_values(params, final_views):
        """[Gs, 2, vw] final views of truncated games (zero rows elsewhere) -> each seat's value [Gs, 2]."""
        flat = final_views.reshape(2 * Gs, vw)
        v = jax.vmap(lambda x: network.trunk(params, encode_one(x))[2])(flat)
        return v.reshape(Gs, 2)

    @jax.jit
    def post_step(stats, ep_len, rewards, done_u8, term_u8, trunc_u8, status):
        """Running statistics and fresh copies of this step's engine outputs (the engine overwrites its tensors)."""
        done, term, trunc = (x.astype(bool) for x in (done_u8, term_u8, trunc_u8))
        donef = done.astype(jnp.float32)
        ep_len = ep_len + 1
        stats = dict(stats)
        stats["finished"] = stats["finished"] + done.sum()
        stats["ep_len_sum"] = stats["ep_len_sum"] + (ep_len * donef).sum().astype(jnp.int32)
        stats["seat1_reward"] = stats["seat1_reward"] + (rewards[:, 0] * donef).sum()
        stats["terminated"] = stats["terminated"] + term.sum()
        stats["truncated"] = stats["truncated"] + trunc.sum()
        stats["nonok"] = stats["nonok"] + (status != 0).sum()
        stats["decisions"] = stats["decisions"] + Gs
        ep_len = ep_len * (1.0 - donef)
        # the seats' terminal rewards of a natively ended game (episode_outcome: a truncation bootstraps, no reward)
        rew2 = jnp.where(term[:, None], rewards, 0.0)
        return stats, ep_len, rew2, done, term, trunc, jnp.any(trunc)

    def seat_gae(value, rew2, player, end_turn, done, term, boot, next_player, next_value):
        """Per-seat GAE of alternating-seat rows [T, G] (distributed.gae / segment_gae): reverse scan with a per-seat
        carry (last advantage, next value, pending terminal reward). boot [T, G, 2]: each seat's value of the final
        state where the game truncated at that step (zero elsewhere)."""
        if turn_clock:
            g = jnp.where(end_turn, args.turn_gamma, 1.0).astype(jnp.float32)
            gl = jnp.where(end_turn, args.turn_gamma * args.turn_lambda, 1.0).astype(jnp.float32)
        else:
            g = jnp.full(value.shape, args.gamma, jnp.float32)
            gl = jnp.full(value.shape, args.gamma * args.lam, jnp.float32)
        m0 = (next_player - 1)[:, None]
        seat = jnp.arange(2)[None, :]
        # boundary bootstrap: the seat to move on the new state's value, the other seat on its negation (zero-sum)
        nv0 = jnp.where(seat == m0, next_value[:, None], -next_value[:, None])
        carry = (jnp.zeros((G, 2), jnp.float32), nv0, jnp.zeros((G, 2), jnp.float32))

        def step(carry, row):
            A, NV, R = carry
            v, r2, m, gt, glt, d, te, b = row
            d2 = d[:, None]
            # the game ended with this decision: the rows after it (already scanned) belong to the next episode
            A = jnp.where(d2, 0.0, A)
            NV = jnp.where(d2, jnp.where(te[:, None], 0.0, b), NV)
            R = jnp.where(d2, r2, R)
            mine = seat == (m - 1)[:, None]
            pick = lambda x: jnp.sum(jnp.where(mine, x, 0.0), 1)
            delta = pick(R) + gt * pick(NV) - v
            adv = delta + glt * pick(A)
            A = jnp.where(mine, adv[:, None], A)
            NV = jnp.where(mine, v[:, None], NV)
            R = jnp.where(mine, 0.0, R)
            return (A, NV, R), adv

        _, adv = jax.lax.scan(step, carry, (value, rew2, player, g, gl, done, term, boot), reverse=True)
        return adv, adv + value

    def alternating_gae(value, rew2, player, family, done, term, trunc, boot, next_player, next_value):
        """ppo.advantages on the alternating-seat rows (the torch smoke's formulation)."""
        reward = jnp.take_along_axis(rew2, (player - 1)[..., None], -1)[..., 0]
        nxt_v = jnp.concatenate([value[1:], next_value[None]], 0)
        nxt_p = jnp.concatenate([player[1:], next_player[None]], 0)
        boot_m = jnp.take_along_axis(boot, (player - 1)[..., None], -1)[..., 0]
        nxt_v = jnp.where(trunc, boot_m, nxt_v)          # truncations bootstrap the final position
        same = jnp.where(trunc, True, nxt_p == player)
        if turn_clock:
            gamma, lam = ppo.turn_discounts(family, END, args.turn_gamma, args.turn_lambda)
        else:
            gamma, lam = args.gamma, args.lam
        return ppo.advantages(reward, value, nxt_v, same, term, trunc, gamma=gamma, lam=lam)

    @jax.jit
    def gae(value, rew2, player, family, done, term, trunc, boot, next_player, next_value):
        end_turn = family == END
        if args.gae == "seat":
            adv, ret = seat_gae(value, rew2, player, end_turn, done, term, boot, next_player, next_value)
        else:
            adv, ret = alternating_gae(value, rew2, player, family, done, term, trunc, boot, next_player, next_value)
        norm = ppo.normalize_advantages(adv.reshape(-1), floor=args.advantage_floor)
        return norm, ret.reshape(-1), adv

    update = ppo.make_update(learning_rate=args.lr, entropy_coef=args.entropy)

    @jax.jit
    def train_step(params, opt, views, path, choice, logp, adv, ret, idx):
        """One minibatch: features re-encoded from the stored views (learner to_batch), path masks, production loss."""
        v = views.reshape(N, vw)[idx]
        features = jax.vmap(encode_one)(v)
        batch = dict(features=features, masks=jax.tree.map(lambda a: a.reshape((N,) + a.shape[2:])[idx], path),
                     choice=jax.tree.map(lambda a: a.reshape(N)[idx], choice),
                     logp=logp.reshape(N)[idx], advantage=adv[idx], returns=ret[idx])
        return update(params, opt, batch)

    # ---- dlpack check (set 0) ----
    if args.check_dlpack:
        torch.cuda.synchronize()
        out0 = gsets[0].out
        v_j, m_j = to_jax(out0["view"]), to_jax(out0["masks"])
        v_np, m_np = out0["view"].cpu().numpy(), out0["masks"].cpu().numpy()
        ok = np.array_equal(np.asarray(v_j), v_np) and np.array_equal(np.asarray(m_j), m_np)
        f_a, p_a = encode(v_j)
        f_b, p_b = encode(jnp.asarray(v_np))
        ok = ok and np.allclose(np.asarray(f_a.board), np.asarray(f_b.board)) and \
            np.allclose(np.asarray(f_a.global_state), np.asarray(f_b.global_state)) and np.array_equal(np.asarray(p_a), np.asarray(p_b))
        a = jnp.arange(Gs, dtype=jnp.int32) * 3 + 1
        t = to_torch(a)
        ok = ok and t.dtype == torch.int32 and t.is_cuda and np.array_equal(t.cpu().numpy(), np.asarray(a))
        fam_copy = torch.zeros(Gs, dtype=torch.int32, device=device)
        fam_copy.copy_(t)
        ok = ok and np.array_equal(fam_copy.cpu().numpy(), np.asarray(a))
        print("DLPACK ok" if ok else "DLPACK MISMATCH", f"view={v_j.shape} {v_j.dtype} masks={m_j.shape} {m_j.dtype} "
              f"features={tuple(f_a.board.shape)} {f_a.board.dtype} choice->torch={tuple(t.shape)} {t.dtype}", flush=True)
        if not ok:
            sys.exit(3)

    # ---- running statistics (on device; one transfer per report) ----
    i32 = lambda: jnp.zeros((), jnp.int32)
    stats = dict(finished=i32(), ep_len_sum=i32(), nonok=i32(), decisions=i32(), terminated=i32(), truncated=i32(),
                 seat1_reward=jnp.zeros((), jnp.float32))
    gpu_name = torch.cuda.get_device_name(0)
    print(f"gpu_resident_train: {gpu_name} (jax: {jax.devices()[0].device_kind}) games={G} sets={sets} size={size} steps={T} "
          f"width={args.width} blocks={args.blocks} arch={args.architecture} params={n_params} channels={board_channels}+{global_channels} "
          f"vw={vw} mask_bytes={mask_bytes} trunk={os.environ.get('POLYTOPIA_TRUNK_DTYPE', 'float32')} "
          f"matmul={args.matmul_precision} heads={args.head_precision} gae={args.gae} turn_clock={turn_clock} minibatch={mb} "
          f"epochs={args.epochs} layouts={P} ({layout_ms:.0f} ms)", flush=True)

    def timed(fn):
        t0 = time.perf_counter()
        res = fn()
        jax.block_until_ready(res)
        return res, (time.perf_counter() - t0) * 1000.0

    STORE_KEYS = ("family", "source", "target", "fam_mask", "src_mask", "tgt_mask", "logp", "value", "player", "rew2",
                  "done", "term", "trunc", "boot")
    clock = dict(tick=0.0, engine_wait=0.0, step=0.0, encode=0.0, sample=0.0, gae=0.0, update=0.0)
    calls = dict(tick=0, engine_wait=0, step=0, encode=0, sample=0)

    def finish(gs):
        """Complete the set's queued tick (if any), bookkeep its outputs for the step it produced, and encode the new
        state. All JAX work on the set's output tensors is dispatched here and completes before the set's next tick is
        launched (launch blocks on the sampled actions; JAX runs its computations in dispatch order)."""
        nonlocal stats
        if gs.in_flight:
            t0 = time.perf_counter()
            gs.env.sync()
            clock["engine_wait"] += (time.perf_counter() - t0) * 1000.0
            calls["engine_wait"] += 1
            gs.in_flight = False
        if gs.pending is not None:
            out = gs.out
            stats, gs.ep_len, rew2, done, term, trunc, any_trunc = post_step(
                stats, gs.ep_len, to_jax(out["rewards"]), to_jax(out["done"]), to_jax(out["terminated"]),
                to_jax(out["truncated"]), to_jax(out["status"]))
            if bool(any_trunc):
                boot = final_values(params, to_jax(out["final_view"]))
            else:
                boot = jnp.zeros((Gs, 2), jnp.float32)
            for k, v in (("rew2", rew2), ("done", done), ("term", term), ("trunc", trunc), ("boot", boot)):
                gs.store[k].append(v)
            gs.pending = None
        views_j = to_jax(gs.out["view"])
        (gs.features, gs.player), ms = timed(lambda: encode(views_j))
        clock["encode"] += ms
        calls["encode"] += 1

    def launch(gs, t, k_act):
        """Sample the set's decisions from its current state, record them for step t, and start its tick: synchronous
        with one set (today's loop), queued on the set's stream with several (the next set's JAX work overlaps it)."""
        masks_j = to_jax(gs.out["masks"])
        (choice, path, logp, value), ms = timed(lambda: act(params, gs.features, masks_j, k_act))
        clock["sample"] += ms
        calls["sample"] += 1
        for k, v in (("family", choice.family), ("source", choice.source), ("target", choice.target),
                     ("fam_mask", path.family), ("src_mask", path.source), ("tgt_mask", path.target),
                     ("logp", logp), ("value", value), ("player", gs.player)):
            gs.store[k].append(v)
        gs.views[t].copy_(gs.out["view"])                          # before the tick overwrites the view tensor
        for dst, src in zip(gs.inputs, (choice.family, choice.source, choice.target)):
            dst.copy_(to_torch(src))
        t0 = time.perf_counter()
        # tick_device synchronizes torch's current stream (the copies above) before the engine reads the inputs
        gs.env.tick_device(*gs.inputs, gs.seeds, *gs.dev_layouts, False, gs.out, sync=(sets == 1))
        gs.pending = t
        if sets == 1:
            clock["tick"] += (time.perf_counter() - t0) * 1000.0
            calls["tick"] += 1
        else:
            gs.in_flight = True

    t_start = time.perf_counter()
    it = 0
    steady_t = steady_decisions = None
    last_metrics = {}
    for gs in gsets:
        finish(gs)   # the initial state's features
    while True:
        if args.iters and it >= args.iters:
            break
        if not args.iters and time.perf_counter() - t_start >= args.seconds:
            break
        # ---------------- rollout ----------------
        for gs in gsets:
            gs.store = {k: [] for k in STORE_KEYS}
        for t in range(T):
            t_step = time.perf_counter()
            key, *ks = jax.random.split(key, sets + 1)
            # ping-pong: finish(set k) waits for its own tick (queued one step ago) while the other sets' ticks run; its
            # encode + act then overlap those. With one set finish() runs right after the synchronous tick.
            for gs, k_act in zip(gsets, ks):
                if sets > 1:
                    finish(gs)
                launch(gs, t, k_act)
                if sets == 1:
                    finish(gs)
            clock["step"] += (time.perf_counter() - t_step) * 1000.0
            calls["step"] += 1
        for gs in gsets:
            if sets > 1:
                finish(gs)   # the last ticks of the segment: their outputs and the new state for the bootstrap
            gs.seeds += G    # fresh RNG seeds for the games that reset during the next iteration (layouts stay per game)

        # ---------------- GAE (per game; the sets are game columns [T, k*Gs:(k+1)*Gs]) ----------------
        def run_gae():
            S = {k: jnp.concatenate([jnp.stack(gs.store[k]) for gs in gsets], axis=1) for k in STORE_KEYS}
            next_value = jnp.concatenate([value_of(params, gs.features) for gs in gsets])
            next_player = jnp.concatenate([gs.player for gs in gsets])
            adv, ret, raw = gae(S["value"], S["rew2"], S["player"], S["family"], S["done"], S["term"], S["trunc"],
                                S["boot"], next_player, next_value)
            return S, adv, ret, raw
        (S, adv, ret, raw_adv), ms = timed(run_gae)
        clock["gae"] += ms

        # ---------------- update: `epochs` passes over T*G decisions in minibatches ----------------
        def run_update():
            nonlocal params, opt, key
            views_all = to_jax(buf_views)
            path = PathMasks(S["fam_mask"], S["src_mask"], S["tgt_mask"])
            choice = Choice(S["family"], S["source"], S["target"])
            acc, count = None, 0
            for _ in range(args.epochs):
                key, k_perm = jax.random.split(key)
                perm = jax.random.permutation(k_perm, N)
                for start in range(0, N - mb + 1, mb):
                    params, opt, metrics = train_step(params, opt, views_all, path, choice, S["logp"], adv, ret,
                                                      perm[start:start + mb])
                    acc = metrics if acc is None else jax.tree.map(jnp.add, acc, metrics)
                    count += 1
            return jax.tree.map(lambda a: a / count, acc), params
        (metrics, _), ms = timed(run_update)
        clock["update"] += ms
        last_metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
        it += 1
        if it == 1:   # the first iteration carries every compile; steady-state throughput starts here
            steady_t, steady_decisions = time.perf_counter(), int(stats["decisions"])

        if it % args.report_every == 0:
            s = {k: float(v) for k, v in jax.device_get(stats).items()}
            elapsed = time.perf_counter() - t_start
            fin = max(s["finished"], 1.0)
            raw_std = float(jnp.std(raw_adv))
            engine = (f"tick={clock['tick'] / calls['tick']:.1f}ms" if sets == 1 else
                      f"step={clock['step'] / calls['step']:.1f}ms engine_wait={clock['engine_wait'] / max(calls['engine_wait'], 1):.1f}ms")
            print(f"iter {it} t={elapsed:.0f}s samples/s={s['decisions'] / elapsed:.0f} {engine} "
                  f"encode={clock['encode'] / calls['encode']:.1f}ms sample={clock['sample'] / calls['sample']:.1f}ms "
                  f"gae={clock['gae'] / it:.0f}ms update={clock['update'] / it:.0f}ms "
                  f"torch_mem={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB finished={int(s['finished'])} "
                  f"ep_len={s['ep_len_sum'] / fin:.0f} term={int(s['terminated'])} trunc={int(s['truncated'])} "
                  f"nonok={s['nonok'] / max(s['decisions'], 1):.2e} seat1_reward={s['seat1_reward'] / fin:+.3f} "
                  f"adv_std={raw_std:.3f} loss={last_metrics['loss']:.4f} pg={last_metrics['policy_loss']:.4f} "
                  f"vf={last_metrics['value_loss']:.4f} ent={last_metrics['entropy']:.3f} kl={last_metrics['approx_kl']:.4f} "
                  f"clip={last_metrics['clip_fraction']:.3f} gnorm={last_metrics['grad_norm']:.3f}", flush=True)

    elapsed = time.perf_counter() - t_start
    s = {k: float(v) for k, v in jax.device_get(stats).items()}
    fin = max(s["finished"], 1.0)
    jax_stats = jax.local_devices()[0].memory_stats() or {}
    jax_peak = jax_stats.get("peak_bytes_in_use", 0) / 2**30
    steady = None
    if steady_t is not None and it > 1:
        steady = round((s["decisions"] - steady_decisions) / max(time.perf_counter() - steady_t, 1e-9))
    per = lambda k, c: round(clock[k] / max(c, 1), 2)
    result = {
        "ok": True, "gpu": gpu_name, "jax_device": jax.devices()[0].device_kind, "games": G, "sets": sets, "size": size,
        "steps": T, "iters": it, "width": args.width, "blocks": args.blocks, "architecture": args.architecture,
        "trunk_dtype": os.environ.get("POLYTOPIA_TRUNK_DTYPE", "float32"), "matmul_precision": args.matmul_precision,
        "head_precision": args.head_precision,
        "params": n_params, "channels": {"board": int(board_channels), "global": int(global_channels)},
        "minibatch": mb, "epochs": args.epochs, "gae": args.gae, "turn_clock": turn_clock,
        "seconds": round(elapsed, 1), "decisions": int(s["decisions"]),
        "samples_per_s": round(s["decisions"] / max(elapsed, 1e-9)), "samples_per_s_steady": steady,
        "tick_ms": per("tick", calls["tick"]) if sets == 1 else None,
        "step_ms": per("step", calls["step"]), "engine_wait_ms": per("engine_wait", calls["engine_wait"]) if sets > 1 else None,
        "encode_ms": per("encode", calls["encode"]), "sample_ms": per("sample", calls["sample"]),
        "gae_ms": per("gae", it), "update_ms": per("update", it),
        "rollout_s_per_iter": round(clock["step"] / 1000.0 / max(it, 1), 2),
        "gpu_mem_gib": {"torch_max_allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        "jax_peak_in_use": round(jax_peak, 2), "jax_in_use": round(jax_stats.get("bytes_in_use", 0) / 2**30, 2)},
        "finished": int(s["finished"]), "terminated": int(s["terminated"]), "truncated": int(s["truncated"]),
        "mean_episode_len": round(s["ep_len_sum"] / fin, 1), "nonok_fraction": s["nonok"] / max(s["decisions"], 1.0),
        "seat1_mean_reward": round(s["seat1_reward"] / fin, 4),
        "metrics": {k: round(v, 5) for k, v in last_metrics.items()},
    }
    print("RESULT " + json.dumps(result), flush=True)
    for gs in gsets:
        gs.env.close()


if __name__ == "__main__":
    main()
