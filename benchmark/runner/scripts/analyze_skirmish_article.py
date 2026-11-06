"""Build a seed-free article dataset from the 60 accepted benchmark submissions.

Run from the benchmark repository root. Reads the original audited action logs or
their hash-checked compact exports; never runs games or makes model calls.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gzip
import hashlib
import json
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/skirmish/results/continuous-full-20261031"
CAMPAIGN = ROOT / "artifacts/harbor-campaigns/continuous-split-full-20261031T075011Z"
REPORT = json.loads((RESULTS / "results.json").read_text())
RULES_PATH = ROOT / "src/polytopia_bot/gpu/rules.json"
RULES = json.loads(RULES_PATH.read_text())
UNIT_NAMES = {u["id"]: u["name"] for u in RULES["units"]}
TECH_NAMES = {u["id"]: u["name"] for u in RULES["technologies"]}
STRUCTURE_NAMES = {u["id"]: u["name"] for u in RULES["structures"]}
BOUNDS = [0]
# Exact catalog shapes in the pinned rl/actions.py for a 14x14 board.
SHAPES = [(196, 196), (196, 196), (196, 128), (196, 128), (1, 128),
          (196, 1), (196, 3), (196, 9), *[(196, 1)] * 6, (1, 6), (1, 1), (1, 1)]
for a, b in SHAPES:
    BOUNDS.append(BOUNDS[-1] + a * b)
PRIVATE_ENTRY = re.compile(rb'"action_id":"([0-9]+)","chain":"[0-9a-f]{64}","player":([12]),"turn":([0-9]+),"version":([0-9]+)')
LABELS = {
    "openai/gpt-6-astra": "GPT-6 Astra",
    "openai/gpt-6.1-sol": "GPT-6.1 Sol",
    "google/gemini-4-argon": "Gemini 4 Argon",
    "anthropic/claude-opus-5.5": "Claude Opus 5.5",
    "anthropic/claude-fable-5-1": "Claude Fable 5.1",
    "moonshotai/kimi-k3.1-pro": "Kimi K3.1 Pro",
    "z-ai/glm-5.3": "GLM 5.3",
    "qwen/qwen3.8-max-0902": "Qwen 3.8 Max",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash",
    "deepseek/deepseek-v4-pro-0813": "DeepSeek V4 Pro",
    "minimax/minimax-m3.1-flash": "MiniMax M3.1 Flash",
}
OPPONENTS = [
    {"id": "reference-weak", "label": "Weak frozen bot"},
    {"id": "reference-intermediate", "label": "Intermediate frozen bot"},
    {"id": "hero-cavalry-25", "label": "Cavalry + 25% champion turns"},
    {"id": "hero-naval-50", "label": "Naval + 50% champion turns"},
    {"id": "scripted-siege", "label": "Scripted siege"},
    {"id": "hero-1v1", "label": "Full RL champion"},
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode(action):
    assert 0 <= action < BOUNDS[-1]
    family = bisect_right(BOUNDS, action) - 1
    return family, (action - BOUNDS[family]) % SHAPES[family][1]


def named(counter, names):
    return [{"name": names.get(k, str(k)), "count": n}
            for k, n in sorted(counter.items(), key=lambda x: (-x[1], x[0]))]


def analyze(trial):
    receipt = json.loads((RESULTS / trial["saved_validation_receipt"]).read_text())
    native = None
    if trial["validation_kind"] == "lossless_replay":
        for folder in (CAMPAIGN / "validated-compact-trials").iterdir():
            validation = folder / "validated.json"
            if not validation.exists():
                continue
            d = json.loads(validation.read_text())
            if d.get("compact_blob_sha256") == receipt["compact_blob_sha256"]:
                raw = (folder / "replay.json.gz").read_bytes()
                assert digest(raw) == receipt["compact_blob_sha256"]
                payload = json.loads(gzip.decompress(raw))
                native = payload["final"]
                entries_by_job = {g["summary"]["job_index"]: g["entries"] for g in payload["games"]}
                break
        assert native is not None
    else:
        base = ROOT / trial["graded_trial"] / "verifier"
        native = {"games.json": json.loads((base / "games.json").read_text()),
                  "submission.json": json.loads((base / "submission.json").read_text())}
    assert native["submission.json"]["source_sha256"] == trial["source_sha256"]
    assert native["submission.json"]["wasm_sha256"] == trial["wasm_sha256"]
    rows = native["games.json"]
    assert len(rows) == 360 and all(r["audit"]["passed"] for r in rows)
    cells = defaultdict(Counter)
    units, early_units, techs, builds, families = (Counter() for _ in range(5))
    for row in rows:
        assert row["result"] in ("win", "draw", "loss")
        key = row["opponent_label"], row["map_type"]
        cell = cells[key]
        cell[row["result"]] += 1
        cell["faults"] += row["status"] == "forfeit"
        cell["turns"] += row["turns"]
        if trial["validation_kind"] == "lossless_replay":
            entries = [(p, t, int(action)) for p, version, t, action in entries_by_job[row["job_index"]]]
        else:
            rel = Path(row["trace_relative"]).relative_to("games")
            public = base / "traces" / rel
            plain = public.with_suffix("") if public.suffix == ".gz" else public
            private = plain.with_suffix(".private.jsonl").read_bytes()
            assert digest(private) == row["private_trace_sha256"]
            entries = [(int(p), int(t), int(action)) for action, p, t, version in PRIVATE_ENTRY.findall(private)]
        assert len(entries) == row["actions"]
        for player, turn, action in entries:
            if player != row["seat"]:
                continue
            family, parameter = decode(action)
            families[family] += 1
            if family == 2:
                units[parameter] += 1
                if turn <= 20:
                    early_units[parameter] += 1
            elif family == 3:
                builds[parameter] += 1
            elif family == 4:
                techs[parameter] += 1
    assert len(cells) == 18 and all(sum(c[k] for k in ("win", "draw", "loss")) == 20 for c in cells.values())
    assert abs(100 * sum(r["result"] == "win" or .5 * (r["result"] == "draw") for r in rows) / 360 - trial["score"]) < 1e-10
    output = {
        "id": trial["trial"], "score": trial["score"], "source": trial["source_file"],
        "sourceSha256": trial["source_sha256"], "wasmSha256": trial["wasm_sha256"],
        "practiceMatches": trial["practice_matches"], "activeMinutes": trial["timing"]["coding_seconds"] / 60,
        "cells": [{"opponent": opp, "map": map_type, "wins": c["win"], "draws": c["draw"],
                   "losses": c["loss"], "faults": c["faults"], "turns": c["turns"]}
                  for (opp, map_type), c in sorted(cells.items())],
    }
    print(trial["model"], trial["trial"], "360 audited games analyzed", flush=True)
    return trial["model"], output, units, early_units, techs, builds, families


def main():
    builds = json.loads((RESULTS / "build-provenance.json").read_text())
    pins = builds["local"]["source_files"]
    for relative in ("src/polytopia_bot/rl/actions.py", "src/polytopia_bot/studio/environment.py", "src/polytopia_bot/gpu/rules.json"):
        assert digest((ROOT / relative).read_bytes()) == pins[relative]
    models = {}
    for row in REPORT["leaderboard"]:
        models[row["model"]] = {"id": row["model"], "label": LABELS[row["model"]],
                                "mean": row["mean_score"], "sd": row["score_sd"], "trials": []}
    counts = {model: [Counter() for _ in range(5)] for model in models}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for model, trial, *counters in pool.map(analyze, REPORT["trials"]):
            models[model]["trials"].append(trial)
            for dest, counter in zip(counts[model], counters, strict=True):
                dest.update(counter)
    for model, data in models.items():
        units, early_units, techs, builds, families = counts[model]
        data["trials"].sort(key=lambda t: t["id"])
        # The engine puts Raft upgrades in the summon catalog too. They are not
        # additional recruits (military.rs UPGRADES), so keep them separate.
        naval = Counter({kind: units.pop(kind, 0) for kind in (41, 42, 44)})
        for kind in (41, 42, 44):
            early_units.pop(kind, None)
        data["navalUpgrades"] = named(naval, UNIT_NAMES)
        data["recruits"] = named(units, UNIT_NAMES)
        data["earlyRecruits"] = named(early_units, UNIT_NAMES)
        data["research"] = named(techs, TECH_NAMES)
        data["builds"] = named(builds, STRUCTURE_NAMES)
        data["actionFamilies"] = {str(k): v for k, v in families.items()}
        assert len(data["trials"]) == 5
        assert abs(statistics.mean(t["score"] for t in data["trials"]) - data["mean"]) < 1e-10
    output = {
        "schema": "polytopia-bench-article-v1", "date": "2026-10-31", "models": list(models.values()),
        "opponents": OPPONENTS, "maps": ["drylands", "lakes", "continents"],
        "provenance": {"benchmarkCommit": "50280698b5d663b2a1ff59c133b577e5fa70dc9c",
                       "resultsSha256": digest((RESULTS / "results.json").read_bytes()),
                       "scheduleSha256": REPORT["schedule_sha256"], "games": 21600, "authors": 60,
                       "codingMinutes": 40, "rulesSha256": digest(RULES_PATH.read_bytes()),
                       "recruitmentDefinition": "Candidate-player train actions across all 1,800 final games per model. Excludes spawned units, conversions and boat upgrades. Early recruits use recorded engine turn <=20.",
                       "actionLogValidation": "Original full private SHA256 checked, or accepted compact blob SHA256 checked; 360 audited receipts and source/Wasm hashes checked per submission.",
                       "unitMixIsDescriptive": "Recruit shares are action-weighted over the fixed panel, not a causal explanation of score."},
    }
    dest = RESULTS / "article-data.json"
    dest.write_text(json.dumps(output, indent=2) + "\n")
    print("Saved", dest)
    for model, data in models.items():
        total = sum(x["count"] for x in data["recruits"])
        print(data["label"], [(x["name"], round(100*x["count"]/total, 1)) for x in data["recruits"][:4]])


if __name__ == "__main__":
    main()
