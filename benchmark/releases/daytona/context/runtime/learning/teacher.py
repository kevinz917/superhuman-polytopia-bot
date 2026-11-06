"""T-main-v0: deterministic public-state heuristics, not a qualified expert.

Scores are preferences, not simulated returns or exact combat predictions.
Only the supplied public observation and immutable public rule tables are read.
"""
from dataclasses import dataclass
import json
from pathlib import Path

from polytopia_bot.environment import Observation, LegalAction
from .interface import canonical, digest, render

VERSION = "t-main-v0"


@dataclass(frozen=True)
class TeacherDecision:
    action: LegalAction
    response: str
    scores: list[dict]
    teacher_hash: str


class WeakBaseline:
    """Unstrategic legal-action opponent, reproducible from public state only."""
    def __init__(self):
        self.teacher_hash = digest(["weak-public-v0", Path(__file__).read_text()])

    def select(self, observation: Observation) -> TeacherDecision:
        prompt = render(observation)
        candidates = {k: a for k, a in prompt.catalog.items() if a.command["kind"] != 11} or prompt.catalog
        scores = [{"action_id": k, "command": dict(a.command), "score": 0,
                   "reason": "unstrategic deterministic legal choice",
                   "tie": digest([self.teacher_hash, prompt.public, a.command])} for k, a in candidates.items()]
        scores.sort(key=lambda row: row["tie"])
        winner = scores[0]["action_id"]
        return TeacherDecision(prompt.catalog[winner], canonical({"action_id": winner}), scores, self.teacher_hash)


class ScriptedTeacher:
    def __init__(self, style="balanced"):
        if style not in ("balanced", "economy", "pressure"):
            raise ValueError("unknown teacher style")
        self.style = style
        table = Path(__file__).resolve().parents[1] / "gpu" / "rules.json"
        raw = json.loads(table.read_text())
        self.units = {u["id"]: u for u in raw["units"]}
        self.structures = {s["id"]: s for s in raw["structures"]}
        self.tech = {t["id"]: t for t in raw["technologies"]}
        self.teacher_hash = digest({"version": VERSION, "style": style, "rules": raw,
                                    "source": Path(__file__).read_text()})

    def select(self, observation: Observation) -> TeacherDecision:
        prompt = render(observation)
        v, player = prompt.public["visible"], observation.player
        size, stars = v["board_size"], v["stars"]
        tiles = {t["tile"]: t for t in v["tiles"]}
        units = {u["tile"]: u for u in v["units"]}
        own = [u for u in units.values() if u["owner"] == player]
        enemy = [u for u in units.values() if u["owner"] != player]
        cities = [c for c in v["cities"] if c["owner"] == player]
        objectives = {c["tile"] for c in v["cities"] if c["owner"] != player}
        objectives |= {t["tile"] for t in tiles.values() if t.get("structure") in (1, 2) and t.get("owner", 0) != player}
        def distance(a, b):
            return max(abs(a % size - b % size), abs(a // size - b // size))
        def threat(tile):
            return sum(distance(tile, u["tile"]) <= self.units.get(u["kind"], {}).get("range", 1) + 1 for u in enemy)
        def nearest(tile):
            return min((distance(tile, t) for t in objectives), default=0)
        def frontier(tile):
            return sum(not t["known"] and distance(tile, t["tile"]) <= 1 for t in tiles.values())
        economy_weight = 1.35 if self.style == "economy" else 1.0
        combat_weight = 1.35 if self.style == "pressure" else 1.0

        def score(c):
            k, src, dst, param = (c[x] for x in ("kind", "source", "target", "parameter"))
            if k == 11:
                return -10000., "avoid resignation"
            if k == 10:
                return 0., "end turn when remaining actions are unhelpful"
            if k == 9:
                # Workshop, resources, population, then superunit at later levels.
                return float({3: 210, 4: 180, 7: 210, 1: 175, 8: 210, 5: 190, 6: 215, 2: 180}.get(param, 170)), "resolve mandatory city reward"
            if k == 8:
                return 200., "capture reachable settlement"
            if k == 2 and src in units and dst in units:
                attacker, defender = units[src], units[dst]
                a = self.units.get(attacker["kind"], {})
                d = self.units.get(defender["kind"], {})
                # Deliberately approximate: no hidden terrain, branch simulator,
                # damage oracle, or assumed exact kill is used.
                strength = a.get("attack", 0) * attacker["health"] / max(a.get("health", 10), 1)
                resistance = d.get("defense", 0) * defender["health"] / max(d.get("health", 10), 1)
                value = 35 + 15 * (strength - resistance) + 10 * (1 - defender["health"] / max(d.get("health", 10), 1))
                return value * combat_weight, "approximate public combat trade"
            if k == 3 and param == 16:
                return 120., "promote veteran"
            if k == 3 and param in (7, 9):
                if param == 9:
                    return 65., "heal friendly units"
                u = units.get(src)
                if u:
                    maximum = self.units.get(u["kind"], {}).get("health", 10) + (5 if u.get("veteran") else 0)
                    return 80 * (1 - u["health"] / maximum) + 8 * threat(src), "recover wounded unit"
            if k == 5:
                return 72 * economy_weight, "harvest legal population resource"
            if k == 6:
                st = self.structures.get(param, {})
                cost = st.get("cost") or 0
                benefit = 12 * (st.get("population") or 0) + 15 * (st.get("stars") or 0)
                return (benefit - 2 * cost + (10 if param == 3 else 0)) * economy_weight, "build using public population/income preference"
            if k == 7:
                # Count visible resources compatible with unlocked structures.
                tech = self.tech.get(param, {})
                unlocked = [self.structures.get(i, {}) for i in tech.get("structures", [])]
                matches = sum(t.get("owner") == player and any(st.get("resource") not in (None, 0) and st.get("resource") == t.get("resource") for st in unlocked) for t in tiles.values())
                priority = {6: 12, 15: 12, 16: 10, 20: 8, 23: 8, 1: 8, 18: 5, 10: 6, 13: 3, 4: 5}.get(param, 1)
                return (priority + 8 * matches - 12 * (stars < 8)) * economy_weight, "research visible economic opportunities and mobility"
            if k == 4:
                cost = self.units.get(param, {}).get("cost", 99)
                demand = max(0, 2 * len(cities) - len(own))
                return (20 * demand + 20 * threat(src) - 2 * cost) * combat_weight, "recruit for expansion and visible threats"
            if k == 1:
                progress = nearest(src) - nearest(dst)
                capture_bonus = 100 if dst in objectives else 0
                reveal = frontier(dst) - frontier(src)
                risk = threat(dst)
                return capture_bonus + 12 * progress + 4 * reveal + (12 if not tiles[dst]["known"] else 0) - 8 * risk, "advance toward visible objectives or exploration frontier"
            return -5., "legal but no positive heuristic for this action"

        scores = []
        # Tie-break on semantic command and public input, independent of engine IDs
        # and catalog order. No map seed, hidden state hash, or global RNG is used.
        public_key = {k: value for k, value in prompt.public.items() if k not in ("legal_actions", "state_version")}
        for response_id, action in prompt.catalog.items():
            value, reason = score(action.command)
            scores.append({"action_id": response_id, "command": dict(action.command), "score": round(value, 6),
                           "reason": reason, "tie": digest([self.teacher_hash, public_key, action.command])})
        scores.sort(key=lambda r: (r["score"], r["tie"]), reverse=True)
        winner = scores[0]["action_id"]
        return TeacherDecision(prompt.catalog[winner], canonical({"action_id": winner}), scores, self.teacher_hash)
