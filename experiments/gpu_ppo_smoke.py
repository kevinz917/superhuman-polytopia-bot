"""GPU-resident PPO smoke on the CUDA engine (engine-cu libpc_cuda.so through GpuGames.tick_device): thousands of
games stepped by one kernel, legal masks and packed views written straight into torch tensors, features built on the
GPU, a factored policy (family -> source -> target) sampled from the masks, PPO on the same device. No CPU actors, no
staging: the only host work per iteration is the ctypes call and the report. Runs on Modal (modal_gpu_engine.py
::ppo_smoke), never locally.

    python experiments/gpu_ppo_smoke.py --games 4096 --size 14 --steps 32 --seconds 600 --width 64 --blocks 2 --bf16

Two seats only (the view layout below is the 2-seat packing of engine-rs kernels.rs VIEW_FIELDS). The features are a
throughput / learning smoke, not observation.encode_view. One policy plays both seats; the reward of a decision is the
mover's terminal reward of the stepped state (rewards[g, mover-1]); the GAE bootstrap flips sign when the next decision
is the opponent's (zero-sum two-seat games). Prints one line per report interval and a final `RESULT {...}` JSON line.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------------------------
# packed view layout: engine-rs kernels.rs observe_impl, field by field in packing order (VIEW_FIELDS sorted order),
# with the 2-seat widths: relations np entries each, 128-bit sets, scalars, per-tile fields n, own_city_rewards [n, 9],
# hist_enemy_unit [HIST_TURNS, n]. Total (40 + 3) * n + 397 + 128 for two seats (kernels.rs view_width).
# ---------------------------------------------------------------------------------------------------------------------
HIST_TURNS = 3


def view_layout(n, np_=2):
    """[(field, length)] in packing order; a field of length k*n is a per-tile field with k entries per tile."""
    fields = []
    if np_ > 2:
        fields.append(("alive", np_))
    fields += [("capital", n), ("detect", n), ("embassies_received", np_), ("embassies_sent", np_), ("enemy_tech", 128),
               ("explored", n), ("hist_enemy_built", n), ("hist_enemy_trained", n), ("hist_enemy_unit", HIST_TURNS * n),
               ("known_players", np_), ("own_built", 128), ("own_city_connected", n), ("own_city_level", n),
               ("own_city_population", n), ("own_city_production", n), ("own_city_progress", n), ("own_city_rewards", 9 * n),
               ("own_city_riot", n), ("own_monuments", 128), ("own_unit_attacked", n), ("own_unit_boosted", n),
               ("own_unit_frozen", n), ("own_unit_home", n), ("own_unit_invisible", n), ("own_unit_kills", n),
               ("own_unit_moved", n), ("own_unit_passenger", n), ("own_unit_poison", n), ("owner", n), ("peace", np_),
               ("peace_offers", np_), ("player", 1), ("resource", n), ("road", n), ("stars", 1), ("structure", n),
               ("structure_level", n), ("tech", 128), ("terrain", n), ("turn", 1), ("unit_direction", n), ("unit_health", n),
               ("unit_owner", n), ("unit_type", n), ("unit_veteran", n)]
    return fields


def view_offsets(n, np_=2):
    at, out = 0, {}
    for name, length in view_layout(n, np_):
        out[name] = (at, length)
        at += length
    return out, at


# feature plan: (field, kind, arg) -> channels. kind: onehot (arg = classes, values outside [0, classes) give no hot),
# bool (0/1, -1 -> 0), num (value / arg), own (== mover), enemy (> 0 and != mover), present (value > 0)
TILE_FEATURES = (
    ("terrain", "onehot", 8), ("explored", "bool", None), ("resource", "onehot", 9), ("structure", "onehot", 64),
    ("structure_level", "num", 20.0), ("road", "bool", None), ("capital", "bool", None),
    ("owner", "own", None), ("owner", "enemy", None),
    ("unit_type", "onehot", 48), ("unit_owner", "own", None), ("unit_owner", "enemy", None), ("unit_health", "num", 20.0),
    ("unit_veteran", "bool", None), ("unit_direction", "onehot", 9),
    ("own_city_connected", "bool", None), ("own_city_level", "num", 20.0), ("own_city_population", "num", 20.0),
    ("own_city_production", "num", 20.0), ("own_city_progress", "num", 20.0), ("own_city_riot", "bool", None),
    ("own_city_rewards", "num9", 20.0),
    ("own_unit_attacked", "bool", None), ("own_unit_boosted", "bool", None), ("own_unit_frozen", "bool", None),
    ("own_unit_invisible", "bool", None), ("own_unit_moved", "bool", None), ("own_unit_poison", "bool", None),
    ("own_unit_home", "numn", None), ("own_unit_kills", "num", 20.0), ("own_unit_passenger", "num", 20.0),
    ("detect", "bool", None),
    ("hist_enemy_built", "bool", None), ("hist_enemy_trained", "bool", None), ("hist_enemy_unit", "bool3", None),
)
GLOBAL_WIDTH = 3 + 128   # turn / 100, log1p(stars) / 8, seat (player - 1), tech bits


def feature_channels():
    c = 0
    for _, kind, arg in TILE_FEATURES:
        c += arg if kind == "onehot" else 9 if kind == "num9" else HIST_TURNS if kind == "bool3" else 1
    return c


class ViewEncoder:
    """Packed int16 views [B, vw] -> board [B, C, size, size] float32 and global [B, GLOBAL_WIDTH], all on the GPU."""

    def __init__(self, size, device):
        self.size, self.n = size, size * size
        self.offsets, width = view_offsets(self.n, 2)
        self.width = width
        self.device = device
        self.channels = feature_channels()

    def field(self, views, name):
        at, length = self.offsets[name]
        return views[:, at:at + length]

    def encode(self, views):
        B, n = views.shape[0], self.n
        v = views.to(torch.int32)
        player = self.field(v, "player")                       # [B, 1] 1-based seat of the mover
        chans = []
        for name, kind, arg in TILE_FEATURES:
            x = self.field(v, name)
            if kind == "onehot":
                idx = x.clamp(0, arg - 1)
                oh = F.one_hot(idx.long(), arg).to(torch.float32)   # [B, n, arg]
                oh = oh * ((x >= 0) & (x < arg)).unsqueeze(-1)
                chans.append(oh.permute(0, 2, 1))
            elif kind == "bool":
                chans.append((x > 0).to(torch.float32).unsqueeze(1))
            elif kind == "num":
                chans.append((x.clamp(min=0).to(torch.float32) / arg).unsqueeze(1))
            elif kind == "numn":
                chans.append((x.clamp(min=0).to(torch.float32) / float(n)).unsqueeze(1))
            elif kind == "own":
                chans.append((x == player).to(torch.float32).unsqueeze(1))
            elif kind == "enemy":
                chans.append(((x > 0) & (x != player)).to(torch.float32).unsqueeze(1))
            elif kind == "num9":                                   # [B, n * 9] tile-major -> [B, 9, n]
                chans.append((x.view(B, n, 9).clamp(min=0).to(torch.float32) / arg).permute(0, 2, 1))
            elif kind == "bool3":                                  # [B, 3 * n] turn-major -> [B, 3, n]
                chans.append((x.view(B, HIST_TURNS, n) > 0).to(torch.float32))
            else:
                raise ValueError(kind)
        board = torch.cat(chans, dim=1).reshape(B, self.channels, self.size, self.size)
        turn = self.field(v, "turn").to(torch.float32) / 100.0
        stars = torch.log1p(self.field(v, "stars").clamp(min=0).to(torch.float32)) / 8.0
        seat = (player - 1).to(torch.float32)
        tech = (self.field(v, "tech") > 0).to(torch.float32)
        glob = torch.cat([turn, stars, seat, tech], dim=1)
        return board, glob, player.squeeze(1)


# ---------------------------------------------------------------------------------------------------------------------
# legal masks: packed bytes [B, mask_bytes] (numpy.packbits order) -> family / source / target path masks on the GPU
# ---------------------------------------------------------------------------------------------------------------------
class MaskTables:
    def __init__(self, size, device):
        from polytopia_bot.rl.actions import offsets, shapes
        self.size, self.n = size, size * size
        self.shapes = shapes(size)                                 # 17 x (rows, row_width)
        self.offsets = [int(o) for o in offsets(size)]             # 18 bounds
        self.mask_width = self.offsets[-1]
        self.families = len(self.shapes)
        self.target_width = max(self.n, 128)
        self.rows = torch.tensor([r for r, _ in self.shapes], dtype=torch.int64, device=device)
        self.row_width = torch.tensor([w for _, w in self.shapes], dtype=torch.int64, device=device)
        self.family_offset = torch.tensor(self.offsets[:-1], dtype=torch.int64, device=device)
        self.shift = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
        self.device = device

    def unpack(self, packed):
        """[B, mask_bytes] uint8 -> [B, mask_width] bool."""
        bits = (packed.unsqueeze(-1) >> self.shift) & 1
        return bits.reshape(packed.shape[0], -1)[:, :self.mask_width].bool()

    def family_and_source(self, bits):
        """family mask [B, 17] (any legal bit in the family's range) and the per-family source mask (row-any bits padded
        to n rows) [B, 17, n]."""
        B, n = bits.shape[0], self.n
        fam = torch.zeros(B, self.families, dtype=torch.bool, device=bits.device)
        src = torch.zeros(B, self.families, n, dtype=torch.bool, device=bits.device)
        for f, (rows, width) in enumerate(self.shapes):
            m = bits[:, self.offsets[f]:self.offsets[f + 1]].view(B, rows, width)
            row_any = m.any(dim=2)                                 # [B, rows]
            src[:, f, :rows] = row_any
            fam[:, f] = row_any.any(dim=1)
        return fam, src

    def target(self, bits, family, source):
        """row bits of (family, source) padded to target_width: [B, target_width] bool (gather, no per-game loop)."""
        B = bits.shape[0]
        width = self.row_width[family]                             # [B]
        base = self.family_offset[family] + source.to(torch.int64) * width
        cols = torch.arange(self.target_width, device=bits.device).unsqueeze(0)   # [1, TW]
        valid = cols < width.unsqueeze(1)
        idx = (base.unsqueeze(1) + cols).clamp(max=self.mask_width - 1)
        return bits.gather(1, idx) & valid


# ---------------------------------------------------------------------------------------------------------------------
# policy: conv trunk, factored heads (family | source tile for the family | target over the family's row), value
# ---------------------------------------------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c1 = nn.Conv2d(width, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x):
        y = F.relu(self.c1(x))
        return F.relu(x + self.c2(y))


class Policy(nn.Module):
    def __init__(self, channels, global_width, width, blocks, families, size, target_width):
        super().__init__()
        self.size, self.n, self.families, self.target_width = size, size * size, families, target_width
        self.stem = nn.Conv2d(channels, width, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(blocks)])
        self.glob = nn.Linear(global_width, width)
        self.source_head = nn.Conv2d(width, families, 1)           # per-tile source logits for every family
        self.family_head = nn.Sequential(nn.Linear(3 * width, width), nn.ReLU(), nn.Linear(width, families))
        self.target_head = nn.Linear(4 * width, families * target_width)   # pooled + global + source tile -> row logits
        self.value_head = nn.Sequential(nn.Linear(3 * width, width), nn.ReLU(), nn.Linear(width, 1))

    def trunk(self, board, glob):
        x = F.relu(self.stem(board))
        g = F.relu(self.glob(glob))
        x = x + g.unsqueeze(-1).unsqueeze(-1)
        for b in self.blocks:
            x = b(x)
        B, W = x.shape[0], x.shape[1]
        tiles = x.flatten(2)                                       # [B, W, n]
        pooled = torch.cat([tiles.mean(2), tiles.amax(2), g], dim=1)   # [B, 3W]
        return tiles, pooled

    def heads(self, tiles, pooled, path_masks, actions=None):
        """path_masks: (family [B, 17], source [B, 17, n], target fn(family, source) -> [B, TW]). actions None: sample;
        else [B, 3] int64 to evaluate. Returns actions, logp, entropy, value, target mask used."""
        B = tiles.shape[0]
        neg = torch.finfo(torch.float32).min
        fam_mask, src_masks, target_fn = path_masks
        fam_logits = self.family_head(pooled).float().masked_fill(~fam_mask, neg)
        fam_dist = torch.distributions.Categorical(logits=fam_logits)
        family = fam_dist.sample() if actions is None else actions[:, 0]
        src_all = self.source_head(tiles.unflatten(2, (self.size, self.size))).flatten(2).float()   # [B, 17, n]
        ar = torch.arange(B, device=tiles.device)
        src_logits = src_all[ar, family].masked_fill(~src_masks[ar, family], neg)
        src_dist = torch.distributions.Categorical(logits=src_logits)
        source = src_dist.sample() if actions is None else actions[:, 1]
        src_feat = tiles[ar, :, source]                             # [B, W]
        tgt_all = self.target_head(torch.cat([pooled, src_feat], dim=1)).view(B, self.families, self.target_width).float()
        tgt_mask = target_fn(family, source)
        tgt_logits = tgt_all[ar, family].masked_fill(~tgt_mask, neg)
        tgt_dist = torch.distributions.Categorical(logits=tgt_logits)
        target = tgt_dist.sample() if actions is None else actions[:, 2]
        logp = fam_dist.log_prob(family) + src_dist.log_prob(source) + tgt_dist.log_prob(target)
        entropy = fam_dist.entropy() + src_dist.entropy() + tgt_dist.entropy()
        value = self.value_head(pooled).float().squeeze(1)
        acts = torch.stack([family, source, target], dim=1)
        return acts, logp, entropy, value, tgt_mask


# ---------------------------------------------------------------------------------------------------------------------
def cuda_ms(fn):
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=4096)
    ap.add_argument("--size", type=int, default=14)
    ap.add_argument("--steps", type=int, default=32, help="decisions per game per PPO iteration")
    ap.add_argument("--iters", type=int, default=0, help="stop after this many iterations (0: by --seconds)")
    ap.add_argument("--seconds", type=float, default=300.0, help="stop after this wall time")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--minibatch", type=int, default=4096, help="decisions per PPO minibatch")
    ap.add_argument("--report-every", type=int, default=1, help="iterations between report lines")
    ap.add_argument("--pool", type=int, default=512, help="layouts generated once (game g uses row g %% pool)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("RESULT " + json.dumps({"ok": False, "error": "no CUDA device"}))
        sys.exit(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda", 0)
    torch.backends.cudnn.benchmark = True
    torch.distributions.Distribution.set_default_validate_args(False)   # the argument checks would sync the host every step

    from polytopia_engine_rs import GameBatch
    from polytopia_bot.gpu import ENGINE_REVISION
    from polytopia_bot.ruleset import rules_json
    from polytopia_bot.rl.gpu_engine import GpuGames, layouts_from_rust

    G, size, T = args.games, args.size, args.steps
    n = size * size
    # ---- layouts: a pool generated once on the CPU by engine-rs mapgen, uploaded once; game g resets on row g % P ----
    t0 = time.perf_counter()
    P = min(args.pool, G)
    layout_batch = GameBatch(size, 1, rules_json(), ENGINE_REVISION)
    pool = layouts_from_rust(layout_batch, np.arange(P, dtype=np.int32) + 1_000, exact=False)   # 4 arrays [P, ...]
    rows = np.arange(G) % P
    host_layouts = tuple(np.ascontiguousarray(a[rows]) for a in pool)                          # [G, n] x3, [G, 2]
    dev_layouts = tuple(torch.from_numpy(a).to(device) for a in host_layouts)
    layout_ms = (time.perf_counter() - t0) * 1000.0

    # ---- engine ----
    env = GpuGames(size=size, games=G, players=2, device=0)
    assert env.view_width == view_offsets(n, 2)[1], (env.view_width, view_offsets(n, 2)[1])
    seeds_host = np.arange(G, dtype=np.int32)
    env.reset(seeds_host, host_layouts)
    seeds = torch.from_numpy(seeds_host).to(device)
    out = env.alloc_device_outputs(device)
    # the first state's masks and views come from the host-pointer calls once; every later one is written on the device
    masks = torch.from_numpy(env.legal_masks(packed=True)).to(device)
    views = torch.from_numpy(env.observe()).to(device)

    enc = ViewEncoder(size, device)
    tables = MaskTables(size, device)
    assert env.mask_width == tables.mask_width
    policy = Policy(enc.channels, GLOBAL_WIDTH, args.width, args.blocks, tables.families, size, tables.target_width).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    n_params = sum(p.numel() for p in policy.parameters())
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16)

    # ---- rollout storage (raw views are re-encoded per minibatch; path masks are stored, not the packed bytes) ----
    vw = env.view_width
    buf_views = torch.zeros((T, G, vw), dtype=torch.int16, device=device)
    buf_fam = torch.zeros((T, G, tables.families), dtype=torch.bool, device=device)
    buf_src = torch.zeros((T, G, tables.families, n), dtype=torch.bool, device=device)
    buf_tgt = torch.zeros((T, G, tables.target_width), dtype=torch.bool, device=device)
    buf_act = torch.zeros((T, G, 3), dtype=torch.int64, device=device)
    buf_logp = torch.zeros((T, G), dtype=torch.float32, device=device)
    buf_val = torch.zeros((T, G), dtype=torch.float32, device=device)
    buf_rew = torch.zeros((T, G), dtype=torch.float32, device=device)
    buf_done = torch.zeros((T, G), dtype=torch.float32, device=device)
    buf_player = torch.zeros((T, G), dtype=torch.int32, device=device)

    # ---- running statistics (all on device; one .item() batch per report) ----
    ep_len = torch.zeros(G, dtype=torch.float32, device=device)
    stat = {k: torch.zeros((), dtype=torch.float64, device=device)
            for k in ("finished", "ep_len_sum", "nonok", "decisions", "seat1_reward", "terminated", "truncated")}
    print(f"gpu_ppo_smoke: {torch.cuda.get_device_name(0)} games={G} size={size} steps={T} width={args.width} "
          f"blocks={args.blocks} bf16={args.bf16} channels={enc.channels} vw={vw} mask_bytes={env.mask_bytes} "
          f"params={n_params} layouts={P} ({layout_ms:.0f} ms)", flush=True)

    t_start = time.perf_counter()
    it = 0
    t_tick = t_fwd = t_upd = 0.0
    tick_calls = fwd_calls = 0
    last_losses = {}
    while True:
        if args.iters and it >= args.iters:
            break
        if not args.iters and time.perf_counter() - t_start >= args.seconds:
            break
        # ---------------- rollout ----------------
        for t in range(T):
            with torch.no_grad(), autocast:
                def forward():
                    bits = tables.unpack(masks)
                    fam_mask, src_masks = tables.family_and_source(bits)
                    board, glob, player = enc.encode(views)
                    tiles, pooled = policy.trunk(board, glob)
                    acts, logp, ent, value, tgt_mask = policy.heads(tiles, pooled, (fam_mask, src_masks, lambda f, s: tables.target(bits, f, s)))
                    return fam_mask, src_masks, player, acts, logp, value, tgt_mask
                (fam_mask, src_masks, player, acts, logp, value, tgt_mask), ms = cuda_ms(forward)
            t_fwd += ms
            fwd_calls += 1
            buf_views[t] = views
            buf_fam[t], buf_src[t], buf_tgt[t] = fam_mask, src_masks, tgt_mask
            buf_act[t], buf_logp[t], buf_val[t], buf_player[t] = acts, logp, value, player
            family, source, target = (acts[:, k].to(torch.int32).contiguous() for k in range(3))

            def tick():
                env.tick_device(family, source, target, seeds, *dev_layouts, False, out)
                return None
            _, ms = cuda_ms(tick)
            t_tick += ms
            tick_calls += 1
            masks, views = out["masks"], out["view"]
            done = out["done"].bool()
            donef = done.float()
            # reward of the decision = the mover's terminal reward of the stepped state (zero while the game runs)
            rew = out["rewards"].gather(1, (player - 1).long().unsqueeze(1)).squeeze(1)
            buf_rew[t] = rew
            buf_done[t] = donef
            ep_len += 1
            stat["finished"] += donef.sum()
            stat["ep_len_sum"] += (ep_len * donef).sum()
            stat["seat1_reward"] += (out["rewards"][:, 0] * donef).sum()
            stat["terminated"] += out["terminated"].sum()
            stat["truncated"] += out["truncated"].sum()
            stat["nonok"] += (out["status"] != 0).sum()
            stat["decisions"] += G
            ep_len = ep_len * (1.0 - donef)
        seeds += G   # fresh RNG seeds for the games that reset during the next iteration (layouts stay per game)

        # ---------------- GAE (zero-sum two seats: the next decision's value is negated when the seat changes) -------
        with torch.no_grad(), autocast:
            bits = tables.unpack(masks)
            board, glob, player_next = enc.encode(views)
            tiles, pooled = policy.trunk(board, glob)
            next_value = policy.value_head(pooled).float().squeeze(1)
        adv = torch.zeros_like(buf_rew)
        lastgae = torch.zeros(G, device=device)
        for t in reversed(range(T)):
            if t == T - 1:
                nv, nplayer = next_value, player_next
            else:
                nv, nplayer = buf_val[t + 1], buf_player[t + 1]
            sign = torch.where(nplayer == buf_player[t], 1.0, -1.0)
            nonterminal = 1.0 - buf_done[t]
            delta = buf_rew[t] + args.gamma * sign * nv * nonterminal - buf_val[t]
            lastgae = delta + args.gamma * args.lam * nonterminal * lastgae
            adv[t] = lastgae
        ret = adv + buf_val

        # ---------------- PPO update: 1 epoch over T*G decisions in minibatches ----------------
        def update():
            N = T * G
            perm = torch.randperm(N, device=device)
            f_views = buf_views.view(N, vw)
            f_fam, f_src, f_tgt = buf_fam.view(N, -1), buf_src.view(N, tables.families, n), buf_tgt.view(N, -1)
            f_act, f_logp, f_adv, f_ret = buf_act.view(N, 3), buf_logp.view(N), adv.view(N), ret.view(N)
            f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
            sums = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0}
            mbs = 0
            for start in range(0, N, args.minibatch):
                idx = perm[start:start + args.minibatch]
                with autocast:
                    board, glob, _ = enc.encode(f_views[idx])
                    tiles, pooled = policy.trunk(board, glob)
                    tgt_mask = f_tgt[idx]
                    _, logp, ent, value, _ = policy.heads(tiles, pooled, (f_fam[idx], f_src[idx], lambda f, s: tgt_mask), f_act[idx])
                ratio = torch.exp(logp - f_logp[idx])
                a = f_adv[idx]
                pg = torch.max(-a * ratio, -a * ratio.clamp(0.8, 1.2)).mean()
                vf = 0.5 * F.mse_loss(value, f_ret[idx])
                loss = pg + 0.5 * vf - args.entropy * ent.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    sums["pg"] += pg.detach(); sums["vf"] += vf.detach(); sums["ent"] += ent.mean().detach()
                    sums["kl"] += (f_logp[idx] - logp).mean().detach()
                    sums["clip"] += ((ratio - 1).abs() > 0.2).float().mean().detach()
                mbs += 1
            return {k: float(v) / mbs for k, v in sums.items()}
        last_losses, ms = cuda_ms(update)
        t_upd += ms
        it += 1

        if it % args.report_every == 0:
            s = {k: float(v) for k, v in stat.items()}
            elapsed = time.perf_counter() - t_start
            fin = max(s["finished"], 1.0)
            print(f"iter {it} t={elapsed:.0f}s samples/s={s['decisions'] / elapsed:.0f} "
                  f"tick={t_tick / tick_calls:.1f}ms fwd={t_fwd / fwd_calls:.1f}ms update={t_upd / it:.0f}ms "
                  f"mem={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB finished={int(s['finished'])} "
                  f"ep_len={s['ep_len_sum'] / fin:.0f} term={int(s['terminated'])} trunc={int(s['truncated'])} "
                  f"nonok={s['nonok'] / max(s['decisions'], 1):.2e} seat1_reward={s['seat1_reward'] / fin:+.3f} "
                  f"pg={last_losses['pg']:.4f} vf={last_losses['vf']:.4f} ent={last_losses['ent']:.3f} "
                  f"kl={last_losses['kl']:.4f} clip={last_losses['clip']:.3f}", flush=True)

    elapsed = time.perf_counter() - t_start
    s = {k: float(v) for k, v in stat.items()}
    fin = max(s["finished"], 1.0)
    result = {
        "ok": True, "gpu": torch.cuda.get_device_name(0), "games": G, "size": size, "steps": T, "iters": it,
        "width": args.width, "blocks": args.blocks, "bf16": args.bf16, "params": n_params, "channels": enc.channels,
        "seconds": round(elapsed, 1), "decisions": int(s["decisions"]), "samples_per_s": round(s["decisions"] / max(elapsed, 1e-9)),
        "tick_ms": round(t_tick / max(tick_calls, 1), 2), "forward_ms": round(t_fwd / max(fwd_calls, 1), 2),
        "update_ms": round(t_upd / max(it, 1), 1), "rollout_s_per_iter": round((t_tick + t_fwd) / 1000.0 / max(it, 1), 2),
        "gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "finished": int(s["finished"]), "terminated": int(s["terminated"]), "truncated": int(s["truncated"]),
        "mean_episode_len": round(s["ep_len_sum"] / fin, 1), "nonok_fraction": s["nonok"] / max(s["decisions"], 1.0),
        "seat1_mean_reward": round(s["seat1_reward"] / fin, 4), "losses": {k: round(v, 5) for k, v in last_losses.items()},
    }
    print("RESULT " + json.dumps(result), flush=True)
    env.close()


if __name__ == "__main__":
    main()
