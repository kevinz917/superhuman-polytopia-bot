"""The C/CUDA engine (engine-cu/) behind the shape of rust_engine.RustGames: a batch of games stepped by
libpc_cpu.so (CPU oracle) or libpc_cuda.so (one thread per game) through the C ABI of engine-cu/src/pc_api.h.

GpuGames.tick returns the dict engine-rs GameBatch.tick(packed_masks=True) returns (same keys, dtypes, shapes), so
the lockstep harness (tests/test_gpu_engine.py) compares the two bitwise. Rules come from gpu/rules.json packed into
PcRulesInput (pack_rules); layouts come from engine-rs GameBatch.layout (layouts_from_rust), since the C engine has
no map generator. NumPy only (no JAX): the wrapper imports in the GPU trainer and in plain harness containers.
"""
import ctypes
import json
import os
from pathlib import Path

import numpy as np

from polytopia_bot.gpu import ENGINE_REVISION
from polytopia_bot.ruleset import rules_json, training_round_limit

ROOT = Path(__file__).resolve().parents[3]
BUILD_DIR = Path(os.environ.get('POLYTOPIA_PC_BUILD', ROOT / 'engine-cu' / 'build'))
# skill flags in engine-rs rules.rs const order (FLOAT .. INFILTRATE); PcRulesInput.unit_skills[kind][k]
SKILL_NAMES = ("Float", "Carry", "Scout", "Creep", "Hide", "Independent", "Surprise", "Splash", "Stomp", "Infiltrate")
# the six other skill strings from_json tests, in rules.h PC_UF_* order; PcRulesInput.unit_flags[kind][k]
FLAG_NAMES = ("Dash", "Fortify", "Stiff", "Persist", "Escape", "Static")
UNIT_NAME_MAX, TECH_NEXT_MAX = 32, 128   # rules.h PC_UNIT_NAME_MAX, PC_TECH_NEXT_MAX
RULES_ERRORS = {1: "unit id / unit_order", 2: "tech id / tech_order", 3: "structure id / struct_order",
                4: "cost table span > PC_COST_TABLE_MAX", 5: "tech_tree overflow", 6: "tech_next_list disagrees with tech_next"}
MAXP = 4
HIST_TURNS = 3
NO_ROUND_LIMIT_C = 1_000_000   # pc.h PC_NO_ROUND_LIMIT (engine-rs mapgen NO_ROUND_LIMIT is i32::MAX; see the report)

c_f32_128, c_i32_128, c_u8_128 = ctypes.c_float * 128, ctypes.c_int32 * 128, ctypes.c_uint8 * 128


class PcRulesInput(ctypes.Structure):
    """engine-cu/src/rules.h PcRulesInput, field for field (primary tables of gpu/rules.json)."""
    _fields_ = [
        ("unit_health", c_f32_128), ("unit_attack", c_f32_128), ("unit_defense", c_f32_128),
        ("unit_movement", c_i32_128), ("unit_range", c_i32_128), ("unit_cost", c_i32_128),
        ("unit_skills", (ctypes.c_uint8 * 10) * 128),
        ("unit_exists", c_u8_128),
        ("unit_order", c_i32_128), ("unit_count", ctypes.c_int32),
        ("tech_tier", c_i32_128), ("tech_requires", c_i32_128),
        ("tech_next", c_u8_128 * 128),
        ("tech_structures", c_u8_128 * 128),
        ("tech_exists", c_u8_128),
        ("tech_order", c_i32_128), ("tech_count", ctypes.c_int32),
        ("struct_cost", c_i32_128), ("struct_population", c_i32_128), ("struct_score", c_i32_128),
        ("struct_stars", c_i32_128), ("struct_resource", c_i32_128), ("struct_tribe", c_i32_128),
        ("struct_terrain", (ctypes.c_uint8 * 16) * 128),
        ("struct_adjacent", c_u8_128 * 128),
        ("struct_per_city", c_u8_128), ("struct_per_tribe", c_u8_128), ("struct_exists", c_u8_128),
        ("struct_order", c_i32_128), ("struct_count", ctypes.c_int32),
        ("engine_revision", ctypes.c_char * 64),
        # appended after the design text (rules.h): from_json inputs the fields above cannot carry
        ("unit_name", (ctypes.c_char * UNIT_NAME_MAX) * 128),      # JSON "name", NUL-terminated (core_names -> unit_supported)
        ("unit_flags", (ctypes.c_uint8 * 6) * 128),                # FLAG_NAMES order
        ("tech_next_list", (ctypes.c_int32 * TECH_NEXT_MAX) * 128),   # JSON "next" verbatim, in JSON order (drives tech_tree)
        ("tech_next_n", c_i32_128),
    ]


class PcTickOut(ctypes.Structure):
    _fields_ = [
        ("masks", ctypes.POINTER(ctypes.c_uint8)), ("view", ctypes.POINTER(ctypes.c_int16)),
        ("opp_view", ctypes.POINTER(ctypes.c_int16)), ("final_view", ctypes.POINTER(ctypes.c_int16)),
        ("done", ctypes.POINTER(ctypes.c_uint8)), ("terminated", ctypes.POINTER(ctypes.c_uint8)),
        ("truncated", ctypes.POINTER(ctypes.c_uint8)), ("rewards", ctypes.POINTER(ctypes.c_float)),
        ("status", ctypes.POINTER(ctypes.c_int32)), ("scores", ctypes.POINTER(ctypes.c_int32)),
        ("device_ptrs", ctypes.c_int),
    ]


def pack_rules(text=None):
    """gpu/rules.json (text; default polytopia_bot.ruleset.rules_json()) -> PcRulesInput.

    Null JSON values become -1 (tech requires, structure cost / population / score / stars / resource / tribe);
    pc_rules_finalize applies rules.rs's defaults (cost -1, the others 0, tribe null = allowed). Table order is kept
    in *_order / *_count (ids in JSON order; technologies with id < 0 are skipped, as every rules.rs loop does)."""
    data = json.loads(rules_json() if text is None else text)
    r = PcRulesInput()
    for i in range(128):
        r.tech_tier[i] = 1                 # rules.rs default tech_tier: [1; 128]
        r.tech_requires[i] = -1
        r.struct_cost[i] = -1              # rules.rs default cost: [-1; 128]
        r.struct_tribe[i] = -1
    for table in ('units', 'technologies', 'structures'):
        ids = [int(row['id']) for row in data[table]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"rules.json {table}: duplicate ids")
    count = 0
    for u in data['units']:
        i = int(u['id'])
        if not 0 <= i < 128:
            raise ValueError("unit id out of range")
        r.unit_health[i], r.unit_attack[i], r.unit_defense[i] = float(u['health']), float(u['attack']), float(u['defense'])
        r.unit_movement[i], r.unit_range[i], r.unit_cost[i] = int(u['movement']), int(u['range']), int(u['cost'])
        skills = set(u['skills'])
        for k, name in enumerate(SKILL_NAMES):
            r.unit_skills[i][k] = int(name in skills)
        for k, name in enumerate(FLAG_NAMES):
            r.unit_flags[i][k] = int(name in skills)
        name = str(u['name']).encode()
        if len(name) >= UNIT_NAME_MAX:
            raise ValueError(f"unit name too long for PcRulesInput: {u['name']}")
        r.unit_name[i].value = name
        r.unit_exists[i] = 1
        r.unit_order[count] = i
        count += 1
    r.unit_count = count
    count = 0
    for t in data['technologies']:
        i = int(t['id'])
        if i < 0:
            continue   # {"id": -1, "name": "BeyondComprehension"}: every from_json loop skips it
        if i >= 128:
            raise ValueError("tech id out of range")
        r.tech_tier[i] = int(t['tier'])
        r.tech_requires[i] = -1 if t['requires'] is None else int(t['requires'])
        if len(t['next']) > TECH_NEXT_MAX:
            raise ValueError("tech next list too long for PcRulesInput")
        for j, k in enumerate(t['next']):
            r.tech_next_list[i][j] = int(k)   # raw, JSON order
            if 0 <= int(k) < 128:
                r.tech_next[i][int(k)] = 1
        r.tech_next_n[i] = len(t['next'])
        for s in t['structures']:
            if 0 <= int(s) < 128:
                r.tech_structures[i][int(s)] = 1
        r.tech_exists[i] = 1
        r.tech_order[count] = i
        count += 1
    r.tech_count = count
    count = 0
    opt = lambda v: -1 if v is None else int(v)
    for s in data['structures']:
        i = int(s['id'])
        if not 0 <= i < 128:
            raise ValueError("structure id out of range")
        r.struct_cost[i], r.struct_population[i], r.struct_score[i] = opt(s['cost']), opt(s['population']), opt(s['score'])
        r.struct_stars[i], r.struct_resource[i], r.struct_tribe[i] = opt(s['stars']), opt(s['resource']), opt(s['tribe'])
        for t in s['terrain']:
            if 0 <= int(t) < 16:
                r.struct_terrain[i][int(t)] = 1
        for a in s['adjacent']:
            if 0 <= int(a) < 128:
                r.struct_adjacent[i][int(a)] = 1
        r.struct_per_city[i], r.struct_per_tribe[i] = int(bool(s['per_city'])), int(bool(s['per_tribe']))
        r.struct_exists[i] = 1
        r.struct_order[count] = i
        count += 1
    r.struct_count = count
    r.engine_revision = data['engine_revision'].encode()[:63]
    return r


def family_widths(n):
    """engine-rs core.rs family_widths: catalog entries per mask family (rl/actions.py shapes products)."""
    return (n * n, n * n, n * 128, n * 128, 128, n, n * 3, n * 9, n, n * 2, n, n, n, n, 18, 1, 1)


def mask_width(n):
    return sum(family_widths(n))


# (name, dtype, shape(n, np), min_players): engine-rs lib.rs::export dict keys in order (pc_api.h PC_EXPORT_FIELDS)
_tile_i32 = lambda n, np: (n,)
_tile_bool = _tile_i32
_seat_i32 = lambda n, np: (np,)
_pair = lambda n, np: (np, np)
_seat128 = lambda n, np: (np, 128)
_scalar = lambda n, np: ()
EXPORT_FIELDS = (
    ("tiles.terrain", np.int32, _tile_i32, 2), ("tiles.owner", np.int32, _tile_i32, 2),
    ("tiles.capital", np.int32, _tile_i32, 2), ("tiles.city", np.int32, _tile_i32, 2),
    ("tiles.road", np.bool_, _tile_bool, 2), ("tiles.structure", np.int32, _tile_i32, 2),
    ("tiles.resource", np.int32, _tile_i32, 2), ("tiles.explored", np.bool_, lambda n, np: (np, n), 2),
    ("tiles.structure_level", np.int32, _tile_i32, 2), ("tiles.structure_turn", np.int32, _tile_i32, 2),
    ("tiles.route", np.bool_, _tile_bool, 2),
    ("units.owner", np.int32, _tile_i32, 2), ("units.kind", np.int32, _tile_i32, 2),
    ("units.health", np.float32, _tile_i32, 2), ("units.veteran", np.bool_, _tile_bool, 2),
    ("units.moved", np.bool_, _tile_bool, 2), ("units.attacked", np.bool_, _tile_bool, 2),
    ("units.kills", np.int32, _tile_i32, 2), ("units.poison", np.bool_, _tile_bool, 2),
    ("units.frozen", np.bool_, _tile_bool, 2), ("units.boosted", np.bool_, _tile_bool, 2),
    ("units.home", np.int32, _tile_i32, 2), ("units.created", np.int32, _tile_i32, 2),
    ("units.converted", np.bool_, _tile_bool, 2), ("units.attacks_performed", np.int32, _tile_i32, 2),
    ("units.prev_x", np.int32, _tile_i32, 2), ("units.prev_y", np.int32, _tile_i32, 2),
    ("units.last_attack", np.int32, _tile_i32, 2), ("units.passenger", np.int32, _tile_i32, 2),
    ("units.invisible", np.bool_, _tile_bool, 2), ("units.direction", np.int32, _tile_i32, 2),
    ("cities.owner", np.int32, _tile_i32, 2), ("cities.level", np.int32, _tile_i32, 2),
    ("cities.population", np.int32, _tile_i32, 2), ("cities.progress", np.int32, _tile_i32, 2),
    ("cities.production", np.int32, _tile_i32, 2), ("cities.border", np.int32, _tile_i32, 2),
    ("cities.rewards", np.int32, lambda n, np: (n, 9), 2), ("cities.territory", np.bool_, lambda n, np: (n, n), 2),
    ("cities.connected", np.bool_, _tile_bool, 2), ("cities.order", np.int32, _tile_i32, 2),
    ("cities.infiltrated", np.int32, _tile_i32, 2), ("cities.riot", np.bool_, _tile_bool, 2),
    ("players.stars", np.int32, _seat_i32, 2), ("players.score", np.int32, _seat_i32, 2),
    ("players.tech", np.bool_, _seat128, 2), ("players.tech_turn", np.int32, _seat128, 2),
    ("players.known", np.bool_, _pair, 2), ("players.kills", np.int32, _seat_i32, 2),
    ("players.casualties", np.int32, _seat_i32, 2), ("players.pacifist", np.int32, _seat_i32, 2),
    ("players.attacked", np.bool_, _seat_i32, 2), ("players.built", np.bool_, _seat128, 2),
    ("players.killed_turn", np.int32, _seat_i32, 2), ("players.resigned_turn", np.int32, _seat_i32, 2),
    ("players.killer", np.int32, _seat_i32, 2), ("players.peace", np.bool_, _pair, 2),
    ("players.embassy", np.bool_, _pair, 2), ("players.peace_offer", np.bool_, _pair, 2),
    ("players.truce_breaker", np.int32, _pair, 2), ("players.hostile_until", np.int32, _pair, 2),
    ("players.peace_broken", np.int32, _pair, 2), ("players.monuments", np.bool_, _seat128, 2),
    ("players.out_order", np.int32, _seat_i32, 3),
    ("history.seen_unit", np.bool_, lambda n, np: (np, HIST_TURNS, n), 2),
    ("history.prev_code", np.int32, lambda n, np: (np, n), 2), ("history.built", np.bool_, lambda n, np: (np, n), 2),
    ("history.trained", np.bool_, lambda n, np: (np, n), 2), ("history.enemy_tech", np.bool_, _seat128, 2),
    ("current_player", np.int32, _scalar, 2), ("turn", np.int32, _scalar, 2), ("version", np.int32, _scalar, 2),
    ("state_version", np.int32, _scalar, 2), ("actions", np.int32, _scalar, 2), ("terminated", np.bool_, _scalar, 2),
    ("truncated", np.bool_, _scalar, 2), ("fog", np.bool_, _scalar, 2), ("reference_max_turns", np.int32, _scalar, 2),
    ("core_supported", np.bool_, _scalar, 2), ("seed", np.uint32, _scalar, 2), ("initial_seed", np.uint32, _scalar, 2),
)
assert len(EXPORT_FIELDS) == 81


def export_fields(players):
    """(name, dtype, shape-fn) of the fields a batch with `players` seats exports, in order."""
    return [(name, dt, fn) for name, dt, fn, minp in EXPORT_FIELDS if players >= minp]


# pc.h PC_PH_* phase order of pc_profile_read (engine-cu/src/pc_api.h)
PROFILE_PHASES = ("step", "outputs", "final_views", "reset", "mask_pre", "mask_move", "mask_attack", "mask_summon",
                  "mask_build", "mask_research", "mask_economy", "mask_unit", "mask_diplo", "legal_copy", "observe_mover",
                  "observe_others")
PROFILE_OUT, BUILD_INFO_OUT = 24, 4


def default_so_path(device, size=None):
    """The engine-cu build for a batch: libpc_cpu*.so (device < 0) or libpc_cuda*.so. Boards <= 14 take the
    PC_MAXSIZE=14 build (`*_14.so`, smaller per-game state) when it exists; POLYTOPIA_PC_VARIANT=full keeps the full
    build, =14 insists on the small one. POLYTOPIA_PC_PROFILE=1 takes the phase-profile build (`*_prof.so`)."""
    stem = 'libpc_cpu' if device < 0 else 'libpc_cuda'
    variant = os.environ.get('POLYTOPIA_PC_VARIANT', 'auto')
    prof = '_prof' if os.environ.get('POLYTOPIA_PC_PROFILE', '0') not in ('', '0') else ''
    small = BUILD_DIR / f'{stem}_14{prof}.so'
    if variant == '14' or (variant == 'auto' and size is not None and size <= 14 and small.exists()):
        return small
    return BUILD_DIR / f'{stem}{prof}.so'


def _load(so_path):
    lib = ctypes.CDLL(str(so_path))
    P = ctypes.POINTER
    i32p, u8p, i16p, vp = P(ctypes.c_int32), P(ctypes.c_uint8), P(ctypes.c_int16), ctypes.c_void_p
    lib.pc_create.restype = ctypes.c_void_p
    lib.pc_create.argtypes = [ctypes.c_int] * 8
    lib.pc_destroy.restype = None
    lib.pc_destroy.argtypes = [vp]
    lib.pc_set_rules.restype = ctypes.c_int
    lib.pc_set_rules.argtypes = [vp, P(PcRulesInput)]
    for name in ("pc_mask_bytes", "pc_view_width"):
        getattr(lib, name).restype = ctypes.c_int
        getattr(lib, name).argtypes = [vp]
    lib.pc_sizes.restype = ctypes.c_int
    lib.pc_sizes.argtypes = [vp, P(ctypes.c_int), P(ctypes.c_int), P(ctypes.c_int)]
    lib.pc_reset.restype = ctypes.c_int
    lib.pc_reset.argtypes = [vp, i32p, i32p, i32p, i32p, i32p]
    lib.pc_tick.restype = ctypes.c_int
    lib.pc_tick.argtypes = [vp, i32p, i32p, i32p, i32p, i32p, i32p, i32p, i32p, ctypes.c_int, P(PcTickOut)]
    # pc_tick_async / pc_sync (pc_api.h): the tick queued on the batch's stream, complete after pc_sync
    lib.pc_tick_async.restype = ctypes.c_int
    lib.pc_tick_async.argtypes = lib.pc_tick.argtypes
    lib.pc_sync.restype = ctypes.c_int
    lib.pc_sync.argtypes = [vp]
    for name in ("pc_export", "pc_import"):
        getattr(lib, name).restype = ctypes.c_int
        getattr(lib, name).argtypes = [vp, ctypes.c_int, P(ctypes.c_void_p)]
    lib.pc_legal_masks.restype = ctypes.c_int
    lib.pc_legal_masks.argtypes = [vp, u8p]
    lib.pc_observe.restype = ctypes.c_int
    lib.pc_observe.argtypes = [vp, ctypes.c_int, i16p]
    # 2026-11-06 additions (absent from older builds)
    for name, args in (("pc_build_info", [vp, i32p]), ("pc_profile_read", [vp, P(ctypes.c_uint64)]),
                       ("pc_profile_reset", [vp])):
        if hasattr(lib, name):
            getattr(lib, name).restype = ctypes.c_int
            getattr(lib, name).argtypes = args
    return lib


def _ptr(a, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype))


def _i32(a):
    return np.ascontiguousarray(np.asarray(a), dtype=np.int32)


class GpuGames:
    """A fixed number of independent games stepped by the C engine (device -1: libpc_cpu.so; else libpc_cuda.so on
    that CUDA device, host-pointer mode). Arguments as RustGames / GameBatch: max_turns None = the training round
    limit of the board, 0 = no limit, N = truncate after N rounds."""

    def __init__(self, size=14, games=24, players=2, max_turns=None, no_resign=True, no_peace=False, max_actions=10_000,
                 so_path=None, device=-1):
        self.size, self.games, self.players, self.device = size, games, players, device
        self.n = size * size
        limit = training_round_limit(size) if max_turns is None else (None if max_turns == 0 else int(max_turns))
        self.max_turns = limit
        self.so_path = Path(so_path) if so_path else default_so_path(device, size)
        self.lib = _load(self.so_path)
        self.handle = self.lib.pc_create(size, games, players, 0 if limit is None else limit, int(no_resign), int(no_peace),
                                         int(max_actions), int(device))
        if not self.handle:
            raise ValueError("pc_create failed (size must be 11, 14 or 18; players 2..4)")
        self.rules = pack_rules()
        if self.rules.engine_revision.decode() != ENGINE_REVISION:
            raise RuntimeError("rule table and pinned engine revision differ")
        rc = self.lib.pc_set_rules(self.handle, ctypes.byref(self.rules))
        if rc != 0:
            raise RuntimeError(f"pc_set_rules failed: {rc} ({RULES_ERRORS.get(rc, 'driver error')})")
        self.mask_bytes = self.lib.pc_mask_bytes(self.handle)
        self.view_width = self.lib.pc_view_width(self.handle)
        self.mask_width = mask_width(self.n)
        if self.mask_bytes != (self.mask_width + 7) // 8:
            raise RuntimeError(f"packed mask width differs: C {self.mask_bytes} bytes vs {self.mask_width} entries")
        self.fields = export_fields(players)

    def build_info(self):
        """pc_build_info: the build's board limit, profile flag, tick kernel (0 thread / 1 warp per game), threads per
        block; None for a build without the call."""
        if not hasattr(self.lib, "pc_build_info"):
            return None
        out = np.zeros(BUILD_INFO_OUT, np.int32)
        self._check(self.lib.pc_build_info(self.handle, _ptr(out, ctypes.c_int32)), "pc_build_info")
        return dict(so=self.so_path.name, max_size=int(out[0]), profile=bool(out[1]), kernel="warp" if out[2] else "thread",
                    threads_per_block=int(out[3]))

    def profile_read(self):
        """The phase profile of a PROFILE=1 build (pc_profile_read), None otherwise: per phase the counter sum over the
        games (`cycles` on CUDA, nanoseconds on CPU) and the mean per game-tick in microseconds, its share of the
        phases' total, and the tick time (CUDA: k_tick alone) per tick in ms."""
        if not hasattr(self.lib, "pc_profile_read"):
            return None
        out = np.zeros(PROFILE_OUT, np.uint64)
        rc = self.lib.pc_profile_read(self.handle, _ptr(out, ctypes.c_uint64))
        if rc == -6:   # PC_E_NO_PROFILE
            return None
        self._check(rc, "pc_profile_read")
        ticks, games, khz, tick_ns = (int(out[k]) for k in (16, 17, 18, 19))
        per = max(1, ticks * games)
        to_us = (lambda c: c / khz * 1e3) if khz else (lambda c: c / 1e3)
        total = int(out[:len(PROFILE_PHASES)].sum())
        phases = {name: dict(sum=int(out[k]), us_per_game_tick=round(to_us(int(out[k]) / per), 3),
                             share=round(int(out[k]) / total, 4) if total else 0.0)
                  for k, name in enumerate(PROFILE_PHASES)}
        masks = sum(int(out[k]) for k, name in enumerate(PROFILE_PHASES) if name.startswith("mask_"))
        return dict(ticks=ticks, games=games, unit="cycles" if khz else "ns", clock_khz=khz,
                    tick_ms=round(tick_ns / max(1, ticks) / 1e6, 4), kernel="warp" if out[20] else "thread",
                    threads_per_block=int(out[21]), max_size=int(out[22]),
                    masks_share=round(masks / total, 4) if total else 0.0, phases=phases)

    def profile_reset(self):
        if hasattr(self.lib, "pc_profile_reset"):
            rc = self.lib.pc_profile_reset(self.handle)
            if rc not in (0, -6):
                self._check(rc, "pc_profile_reset")

    def close(self):
        if getattr(self, "handle", None):
            self.lib.pc_destroy(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _layouts(self, layouts):
        terrain, structure, resource, caps = (_i32(a) for a in layouts)
        if terrain.shape != (self.games, self.n) or structure.shape != (self.games, self.n) or resource.shape != (self.games, self.n) \
                or caps.shape != (self.games, self.players):
            raise ValueError("layout arrays must be [games, n] and capitals [games, players]")
        return terrain, structure, resource, caps

    def _check(self, rc, what):
        if rc != 0:
            raise RuntimeError(f"{what} failed: {rc}")

    def reset(self, seeds, layouts):
        """Fresh games from layouts (terrain, structure, resource, capitals) = layouts_from_rust(batch, seeds)."""
        seeds = _i32(seeds)
        if seeds.shape != (self.games,):
            raise ValueError("seeds must have one entry per game")
        t, s, r, c = self._layouts(layouts)
        self._check(self.lib.pc_reset(self.handle, *(_ptr(a, ctypes.c_int32) for a in (seeds, t, s, r, c))), "pc_reset")

    def tick(self, family, source, target, seeds, layouts, both_views=True):
        """GameBatch.tick(family, source, target, seeds, both_views, layouts, packed_masks=True): the same dict."""
        f, s, t, seeds = (_i32(a) for a in (family, source, target, seeds))
        if f.shape != (self.games,) or s.shape != (self.games,) or t.shape != (self.games,) or seeds.shape != (self.games,):
            raise ValueError("choice and seed arrays must have one entry per game")
        terrain, structure, resource, caps = self._layouts(layouts)
        G, np_, vw = self.games, self.players, self.view_width
        others = np_ - 1
        masks = np.zeros((G, self.mask_bytes), np.uint8)
        view = np.empty((G, vw), np.int16)
        opp = np.empty((G * others if both_views else 0, vw), np.int16)
        final_view = np.zeros((G, np_, vw), np.int16)
        done, terminated, truncated = (np.zeros(G, np.uint8) for _ in range(3))
        rewards = np.zeros((G, np_), np.float32)
        status = np.zeros(G, np.int32)
        scores = np.zeros((G, np_), np.int32)
        out = PcTickOut(masks=_ptr(masks, ctypes.c_uint8), view=_ptr(view, ctypes.c_int16),
                        opp_view=_ptr(opp, ctypes.c_int16) if both_views else None, final_view=_ptr(final_view, ctypes.c_int16),
                        done=_ptr(done, ctypes.c_uint8), terminated=_ptr(terminated, ctypes.c_uint8),
                        truncated=_ptr(truncated, ctypes.c_uint8), rewards=_ptr(rewards, ctypes.c_float),
                        status=_ptr(status, ctypes.c_int32), scores=_ptr(scores, ctypes.c_int32), device_ptrs=0)
        rc = self.lib.pc_tick(self.handle, _ptr(f, ctypes.c_int32), _ptr(s, ctypes.c_int32), _ptr(t, ctypes.c_int32),
                              _ptr(seeds, ctypes.c_int32), _ptr(terrain, ctypes.c_int32), _ptr(structure, ctypes.c_int32),
                              _ptr(resource, ctypes.c_int32), _ptr(caps, ctypes.c_int32), int(both_views), ctypes.byref(out))
        self._check(rc, "pc_tick")
        d = {"masks": masks, "view": view}
        if both_views:
            d["opp_view"] = opp.reshape(G, others, vw) if np_ > 2 else opp
        d.update(final_view=final_view, done=done.view(bool), terminated=terminated.view(bool), truncated=truncated.view(bool),
                 rewards=rewards, status=status, scores=scores)
        return d

    # ---- device-pointer mode (the CUDA driver with PcTickOut.device_ptrs = 1; torch tensors stay on the GPU) ----

    def alloc_device_outputs(self, device):
        """The preallocated torch CUDA tensors tick_device fills: masks uint8 [G, mask_bytes], view int16 [G, vw],
        opp_view int16 [G, np-1, vw], final_view int16 [G, np, vw], done / terminated / truncated uint8 [G],
        rewards float32 [G, np], status int32 [G], scores int32 [G, np] (players.score before any reset). `device` is a torch device (or index / string)."""
        import torch
        G, np_, vw = self.games, self.players, self.view_width
        dev = torch.device(device)
        z = lambda shape, dt: torch.zeros(shape, dtype=dt, device=dev)
        return {"masks": z((G, self.mask_bytes), torch.uint8), "view": z((G, vw), torch.int16),
                "opp_view": z((G, np_ - 1, vw), torch.int16), "final_view": z((G, np_, vw), torch.int16),
                "done": z((G,), torch.uint8), "terminated": z((G,), torch.uint8), "truncated": z((G,), torch.uint8),
                "rewards": z((G, np_), torch.float32), "status": z((G,), torch.int32), "scores": z((G, np_), torch.int32)}

    def tick_device(self, family, source, target, seeds, terrain, structure, resource, caps, both_views, out, sync=True):
        """pc_tick with every pointer in CUDA device memory (PcTickOut.device_ptrs = 1): nothing is staged, the engine
        reads the decisions and layouts from, and writes masks / views / flags / rewards / status into, torch tensors.

        family / source / target / seeds: int32 [G]; terrain / structure / resource: int32 [G, n]; caps: int32 [G, np];
        all contiguous torch tensors on this batch's CUDA device. `out`: the dict of alloc_device_outputs (opp_view may
        be None when both_views is false; the driver only checks it when both_views is set). The engine runs on its own
        stream and synchronizes it before returning, so the outputs are complete when this returns; torch's current
        stream is synchronized first so the inputs torch just produced are complete when the kernel reads them.
        sync=False: pc_tick_async, the tick is queued on the batch's stream and this returns at once; the outputs are
        complete after sync(). The input and output tensors must stay untouched until then."""
        import torch
        if self.device < 0:
            raise RuntimeError("tick_device needs the CUDA driver (GpuGames(device=<cuda ordinal>))")
        G, n, np_, vw = self.games, self.n, self.players, self.view_width

        def check(t, name, shape, dtype):
            if not isinstance(t, torch.Tensor) or not t.is_cuda:
                raise TypeError(f"{name} must be a CUDA torch tensor")
            if t.dtype != dtype or tuple(t.shape) != tuple(shape) or not t.is_contiguous():
                raise ValueError(f"{name}: expected contiguous {dtype} {tuple(shape)}, got {t.dtype} {tuple(t.shape)} "
                                 f"contiguous={t.is_contiguous()}")
            return t

        i32 = torch.int32
        inputs = [check(family, "family", (G,), i32), check(source, "source", (G,), i32), check(target, "target", (G,), i32),
                  check(seeds, "seeds", (G,), i32), check(terrain, "terrain", (G, n), i32),
                  check(structure, "structure", (G, n), i32), check(resource, "resource", (G, n), i32),
                  check(caps, "caps", (G, np_), i32)]
        spec = {"masks": ((G, self.mask_bytes), torch.uint8, ctypes.c_uint8), "view": ((G, vw), torch.int16, ctypes.c_int16),
                "final_view": ((G, np_, vw), torch.int16, ctypes.c_int16), "done": ((G,), torch.uint8, ctypes.c_uint8),
                "terminated": ((G,), torch.uint8, ctypes.c_uint8), "truncated": ((G,), torch.uint8, ctypes.c_uint8),
                "rewards": ((G, np_), torch.float32, ctypes.c_float), "status": ((G,), torch.int32, ctypes.c_int32),
                "scores": ((G, np_), torch.int32, ctypes.c_int32)}
        ptrs = {}
        for name, (shape, dt, ct) in spec.items():
            t = check(out[name], name, shape, dt)
            ptrs[name] = ctypes.cast(t.data_ptr(), ctypes.POINTER(ct))
        opp = out.get("opp_view")
        if both_views:
            check(opp, "opp_view", (G, np_ - 1, vw), torch.int16)
        opp_ptr = ctypes.cast(opp.data_ptr(), ctypes.POINTER(ctypes.c_int16)) if isinstance(opp, torch.Tensor) and opp.numel() else None
        tick_out = PcTickOut(opp_view=opp_ptr, device_ptrs=1, **ptrs)
        torch.cuda.current_stream(inputs[0].device).synchronize()
        args = [ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_int32)) for t in inputs]
        if sync:
            rc = self.lib.pc_tick(self.handle, *args, int(bool(both_views)), ctypes.byref(tick_out))
            self._check(rc, "pc_tick(device_ptrs)")
        else:
            rc = self.lib.pc_tick_async(self.handle, *args, int(bool(both_views)), ctypes.byref(tick_out))
            self._check(rc, "pc_tick_async(device_ptrs)")
        return out

    def sync(self):
        """Block until the tick queued by tick_device(sync=False) is complete (pc_sync; the tick's errors surface here)."""
        self._check(self.lib.pc_sync(self.handle), "pc_sync")

    def legal_masks(self, packed=True):
        """[G, mask_bytes] packed bits (numpy.packbits order), or unpacked [G, mask_width] 0/1 uint8."""
        masks = np.zeros((self.games, self.mask_bytes), np.uint8)
        self._check(self.lib.pc_legal_masks(self.handle, _ptr(masks, ctypes.c_uint8)), "pc_legal_masks")
        if packed:
            return masks
        return np.unpackbits(masks, axis=1)[:, :self.mask_width]

    def observe(self, player=None):
        """Packed view of `player` (1-based; None = the player to move) for every game: [G, vw] int16."""
        view = np.zeros((self.games, self.view_width), np.int16)
        self._check(self.lib.pc_observe(self.handle, 0 if player is None else int(player), _ptr(view, ctypes.c_int16)), "pc_observe")
        return view

    def _field_arrays(self, d=None):
        arrays = []
        for name, dt, fn in self.fields:
            shape = fn(self.n, self.players)
            if d is None:
                a = np.zeros(shape, dt)
            else:
                a = np.ascontiguousarray(np.asarray(d[name]), dtype=dt)
                if a.shape != tuple(shape):
                    raise ValueError(f"field {name}: expected shape {shape}, got {a.shape}")
            arrays.append(a)
        ptrs = (ctypes.c_void_p * len(arrays))(*(a.ctypes.data_as(ctypes.c_void_p).value for a in arrays))
        return arrays, ptrs

    def export(self, game):
        """Every State field of one game: {"tiles.terrain": array, ...} in lib.rs export order (no game axis)."""
        arrays, ptrs = self._field_arrays()
        self._check(self.lib.pc_export(self.handle, int(game), ptrs), "pc_export")
        return {name: a for (name, _, _), a in zip(self.fields, arrays)}

    def export_all(self):
        """Every game, arrays with a leading game axis (the GameBatch.export layout)."""
        per = [self.export(g) for g in range(self.games)]
        return {name: np.stack([p[name] for p in per]) for name, _, _ in self.fields}

    def import_(self, game, d):
        """Inverse of export: overwrite one game from a dict of every field (missing keys are an error)."""
        arrays, ptrs = self._field_arrays(d)
        self._check(self.lib.pc_import(self.handle, int(game), ptrs), "pc_import")


def layouts_from_rust(batch, seeds, exact=False):
    """(terrain, structure, resource, capitals) int32 arrays [G, n] x3 and [G, players] from engine-rs
    GameBatch.layout(seed, exact) for each seed: what pc_reset / pc_tick take (GameBatch.reset_layouts / tick layouts)."""
    seeds = _i32(seeds)
    rows = [batch.layout(int(s), exact) for s in seeds]
    stack = lambda key: np.ascontiguousarray(np.stack([np.asarray(r[key], np.int32) for r in rows]), dtype=np.int32)
    return stack("terrain"), stack("structure"), stack("resource"), stack("capitals")
