"""Verify the synced benchmark without Docker, model APIs or the original worktree."""
import base64
from collections import Counter
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runner/docs/skirmish/results/continuous-full-20261031"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_member(archive, name):
    return archive.extractfile(name).read()


def verify():
    manifest = json.loads((ROOT / "archive-manifest.json").read_text())
    report = json.loads((RESULTS / "results.json").read_text())
    for relative, info in manifest["files"].items():
        path = ROOT / relative
        assert path.is_file() and path.stat().st_size == info["bytes"], relative
        assert sha(path.read_bytes()) == info["sha256"], relative
        assert info["bytes"] < 100_000_000, "File exceeds ordinary GitHub file size limit"
    for relative, digest in manifest["frozen_source_files"].items():
        assert sha((ROOT / "runner" / relative).read_bytes()) == digest, relative
    assert report["authored_submissions"] == manifest["authored_submissions"] == 60
    assert report["total_final_outcomes"] == manifest["final_games"] == 21600
    assert len(manifest["trials"]) == 60
    by_trial = {row["trial"]: row for row in report["trials"]}
    counts, scores = Counter(), {}
    private_rebuilt = practice_completed = practice_directories = 0
    original_compact_blobs = 0
    expected_schedule = report["schedule_sha256"]
    for exported in manifest["trials"]:
        old = by_trial[exported["trial"]]
        assert old["model"] == exported["model"]
        counts[old["model"]] += 1
        scores.setdefault(old["model"], []).append(old["score"])
        source = (RESULTS / old["source_file"]).read_bytes()
        assert sha(source) == old["source_sha256"] == exported["source_sha256"]
        folder = ROOT / exported["path"]
        receipt = json.loads((RESULTS / old["saved_validation_receipt"]).read_text())
        assert receipt["passed"] and receipt["schedule_sha256"] == expected_schedule
        blob = (folder / "final-replay.json.gz").read_bytes()
        if old["validation_kind"] == "lossless_replay":
            assert sha(blob) == receipt["compact_blob_sha256"]
            original_compact_blobs += 1
        payload = json.loads(gzip.decompress(blob))
        final = payload["final"]
        assert payload["source_sha256"] == old["source_sha256"]
        assert sha(base64.b64decode(final["wasm_base64"])) == old["wasm_sha256"] == exported["wasm_sha256"]
        plan, rows = final["evaluation-plan.json"]["jobs"], final["games.json"]
        assert len(plan) == len(rows) == len(payload["games"]) == 360
        schedule = [(j["opponent"], j["map_type"], j["seed"], j["seat"], j["evaluation"]) for j in plan]
        assert sha(canonical(schedule).encode()) == expected_schedule
        assert {r["job_index"] for r in rows} == set(range(360))
        records = {r["summary"]["job_index"]: r for r in payload["games"]}
        for row in rows:
            assert row["audit"]["passed"] and row["result"] in ("win", "draw", "loss")
            job = plan[row["job_index"]]
            assert row["job_sha256"] == job["job_sha256"] == sha(canonical({k: v for k, v in job.items() if k != "job_sha256"}).encode())
            record = records[row["job_index"]]
            assert record["summary"] == row and len(record["entries"]) == row["actions"]
            if old["validation_kind"] == "full_export":
                # Newly compacted full exports must reproduce every original
                # private byte hash and chain. Existing compact blobs retain
                # their exact already-validated receipt digest instead.
                chain, digest = "0" * 64, hashlib.sha256()
                for player, version, turn, action in record["entries"]:
                    entry = dict(player=player, version=version, turn=turn, action_id=action)
                    chain = sha((chain + canonical(entry)).encode())
                    digest.update((canonical(dict(**entry, chain=chain)) + "\n").encode())
                assert digest.hexdigest() == row["private_trace_sha256"]
                assert chain == row["action_chain_sha256"]
                private_rebuilt += 1
        points = sum({"win": 1, "draw": .5, "loss": 0}[r["result"]] for r in rows)
        assert math.isclose(100 * points / 360, old["score"], abs_tol=1e-12)
        assert final["reward.json"] == old["rewards"] == final["summary.json"]["rewards"]
        with tarfile.open(folder / "authoring.tar.gz") as archive:
            author = json.loads(read_member(archive, "agent/authoring.json"))
            assert author["capture"]["source_sha256"] == old["source_sha256"]
            assert author["coding_budget_seconds"] == 2400 and author["timing"]["coding_seconds"] <= 2400
            assert archive.getmember("agent/trajectory.json").size > 0
            assert sha(read_member(archive, "artifacts/submission/main.rs")) == old["source_sha256"]
        with tarfile.open(folder / "practice.tar.gz") as archive:
            practices = json.loads(read_member(archive, "index.json"))
            assert len(practices) == exported["practice_directories"]
            for practice in practices:
                practice_directories += 1
                for name, info in practice["files"].items():
                    if name == "game.jsonl" and practice.get("public_encoding", "").startswith("pinned engine replay"):
                        continue
                    raw = read_member(archive, practice["id"] + "/" + name)
                    assert sha(raw) == info["sha256"] and len(raw) == info["bytes"]
                if practice["completed"]:
                    practice_completed += 1
                    practice_report = json.loads(read_member(archive, practice["id"] + "/report.json"))
                    private = read_member(archive, practice["id"] + "/game.private.jsonl")
                    assert len(private.splitlines()) == practice_report["audited_actions"] == practice_report["actions"]
        print("Verified", old["model"], old["trial"], flush=True)
    assert len(counts) == 12 and set(counts.values()) == {5}
    for row in report["leaderboard"]:
        assert math.isclose(row["mean_score"], statistics.mean(scores[row["model"]]), abs_tol=1e-12)
        assert math.isclose(row["score_sd"], statistics.stdev(scores[row["model"]]), abs_tol=1e-12)
    failed = ROOT / manifest["failed_authoring"]["path"]
    assert manifest["failed_authoring"]["excluded_from_scores"]
    with tarfile.open(failed / "authoring.tar.gz") as archive:
        assert archive.getmember("agent/trajectory.json").size > 0
    refs = json.loads((ROOT / "releases/local/manifest.json").read_text())["references"]
    for ref in refs:
        if "file" in ref:
            assert sha((ROOT / "references" / ref["file"]).read_bytes()) == ref.get("wasm_sha256", ref.get("checkpoint_sha256"))
    proof = dict(passed=True, checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 models=12, authored_submissions=60, final_games=21600,
                 frozen_source_files=len(manifest["frozen_source_files"]),
                 original_validated_compact_blobs_preserved=original_compact_blobs,
                 newly_encoded_private_game_hashes_reconstructed=private_rebuilt,
                 authoring_trajectories=60, failed_authoring_trajectory_preserved=True,
                 completed_practice_records=practice_completed, practice_directories=practice_directories,
                 file_hashes_verified=len(manifest["files"]),
                 public_observation_validation="Original native audits retained; no new full-game simulations run during archive verification.")
    json_write = ROOT / "archive-validation.json"
    json_write.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    verify()
