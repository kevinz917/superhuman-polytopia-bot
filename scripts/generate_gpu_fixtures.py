#!/usr/bin/env python3
"""Regenerate golden cases from Rust. Run with PYTHONPATH=src; never uses JAX.

Cases include real generated starts plus explicitly modified rule fixtures.
These are primitive/transition parity tests, not full-game replay validation.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random
import subprocess

from polytopia_bot.gpu import ENGINE_REVISION
from polytopia_bot.oracle import Oracle

ROOT = Path(__file__).resolve().parents[1]


def coord(i, size):
    return {"idx": i, "x": i % size, "y": i // size}


def main():
    repo = ROOT / "vendor/Polyfish"
    rev = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if rev != ENGINE_REVISION:
        raise SystemExit("wrong oracle source revision")
    cases = []
    with Oracle() as oracle:
        rules = oracle.request(op="rules")
        assert rules["engine_revision"] == ENGINE_REVISION
        (ROOT / "src/polytopia_bot/gpu/rules.json").write_text(json.dumps(rules, indent=2) + "\n")
        for size in (11, 14):
            base = oracle.request(op="reset", size=size, seed=7)["state"]
            scenarios = []
            for seed in (0, 1, 7, 19):
                scenarios.append((f"generated-{seed}", oracle.request(op="reset", size=size, seed=seed)["state"]))
            for kind in (2, 3, 4, 5, 8, 9, 11, 12):
                s = deepcopy(base)
                s["settings"]["turn"] = 3
                for tile in s["tiles"].values():
                    tile["explorers"] = [1, 2]
                    tile["hasRoad"] = tile["coords"]["idx"] % 3 != 0
                    tile.pop("unitOwnerId", None)
                for player in s["tribes"].values():
                    player["stars"] = 100
                    player["knownPlayers"] = [3 - player["id"]]
                    for tech in (0, 20, 21, 22):
                        if not any(t["type"] == tech for t in player["tech_vanilla"]):
                            player["tech_vanilla"].append({"type": tech, "discovered": True, "discoveredTurn": 1})
                unit = s["tribes"]["1"]["units"][0]
                unit.update(type=kind, health=5.0, veteran=(kind == 4), effects=[1] if kind == 5 else [])
                other = s["tribes"]["2"]["units"][0]
                other.update(health=4.0, moved=False, attacked=False, effects=[0])
                center = (size // 2) * size + size // 2
                unit["coords"] = coord(center, size)
                other["coords"] = coord(center + 2, size)
                for p in s["tribes"].values():
                    for u in p["units"]:
                        tile = s["tiles"][str(u["coords"]["idx"])]
                        tile["unitOwnerId"] = p["id"]
                        tile["type"] = 3
                # Hidden enemy still affects source ZOC; retain a regression case.
                if kind == 3:
                    s["tiles"][str(center + 2)]["explorers"] = [2]
                scenarios.append((f"land-unit-{kind}", s))
            # Turn cap is a truncation in the adapter, not a score-based victory.
            cap = deepcopy(base)
            cap["settings"].update(turn=45, currentPlayerTurnId=2)
            scenarios.append(("turn-cap", cap))
            # Pending city choices must suppress ordinary actions.
            pending = deepcopy(base)
            pending["tribes"]["1"]["cities"][0]["level"] = 2
            scenarios.append(("pending-city-choice", pending))
            for name, raw in scenarios:
                result = oracle.request(op="query", state=raw)
                legal = result["legal"]
                selected = []
                for kind in (10, 7, 3, 1):
                    matches = [a for a in legal if a["moveType"] == kind and (kind != 3 or a.get("type") == 7)]
                    if kind == 1:
                        matches = [a for a in matches if max(abs(a["src"] % size - a["target"] % size), abs(a["src"] // size - a["target"] // size)) == 1]
                        # Initial moves near corners can reveal lighthouses;
                        # those are explicitly unsupported by this first slice.
                        def no_lighthouse(a):
                            p = raw["settings"]["currentPlayerTurnId"]
                            for key, st in raw["structures"].items():
                                if st and st["type"] == 47 and p not in raw["tiles"][key]["explorers"]:
                                    i = int(key)
                                    for pos in (a["src"], a["target"]):
                                        radius = 2 if raw["tiles"][str(pos)]["type"] == 4 else 1
                                        if max(abs(i % size - pos % size), abs(i // size - pos // size)) <= radius:
                                            return False
                            return True
                        matches = [a for a in matches if no_lighthouse(a)]
                    if matches:
                        a = matches[0]
                        selected.append({"action": a, "after": oracle.request(op="step", state=raw, action=a)["state"]})
                explored = oracle.request(op="explore", state=raw, player=1)["state"]
                cases.append({"name": f"{size}-{name}", "before": raw, "legal": legal, "transitions": selected, "explored_player1": explored})
        rng = random.Random(42)
        combat = []
        # Includes exact ties, zero forces, fractional HP, and near-lethal cases.
        for _ in range(4096):
            am, dm = rng.choice((10, 15, 20, 40)), rng.choice((10, 15, 20, 40))
            combat.append([rng.choice((0, 1, 2, 3, 3.5, 4, 5)), rng.randint(1, am * 2) / 2, am,
                           rng.choice((0, 0.5, 1, 2, 3, 4)), rng.randint(1, dm * 2) / 2, dm,
                           rng.choice((0.5, 0.75, 1, 1.5, 2, 4))])
        damage = oracle.request(op="combat", cases=combat)["results"]
    data = {"revision": ENGINE_REVISION, "rules_version": 115, "cases": cases,
            "combat": {"inputs": combat, "expected": damage}}
    path = ROOT / "tests/fixtures/polyfish-core.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(data, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    print(json.dumps({"path": str(path), "board_cases": len(cases), "transitions": sum(len(c["transitions"]) for c in cases), "combat_cases": len(combat)}))


if __name__ == "__main__":
    main()
