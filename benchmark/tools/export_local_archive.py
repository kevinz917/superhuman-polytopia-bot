"""Export the official local campaign into a portable Git-sized benchmark archive.

Run with --source pointing to the original codex/polytopia-skirmish worktree.
Only the official cohort and its failed Gemini authoring predecessor are included.
No containers, API calls, or model inference are started.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor

DEST = Path(__file__).resolve().parents[1]
ENTRY = re.compile(rb'"action_id":"([^"]*)","chain":"[0-9a-f]{64}","player":([12]),"turn":([0-9]+),"version":([0-9]+)')
SENSITIVE = re.compile(rb'sk-or-v1-[0-9a-f]{32,}|sk-ant-[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def pack_files(files, path):
    """Deterministic tar+gzip of supplied relative names and exact published bytes."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, raw in sorted(files.items()):
            rel = Path(name)
            assert not rel.is_absolute() and ".." not in rel.parts
            item = tarfile.TarInfo(name)
            item.size, item.mode, item.mtime = len(raw), 0o644, 0
            archive.addfile(item, io.BytesIO(raw))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(stream.getvalue(), compresslevel=9, mtime=0))


def export(source):
    source = source.resolve()
    result_dir = source / "docs/skirmish/results/continuous-full-20261031"
    report = json.loads((result_dir / "results.json").read_text())
    campaign = source / "artifacts/harbor-campaigns" / report["campaign"]
    provenance = json.loads((result_dir / "build-provenance.json").read_text())
    pins = provenance["local"]["source_files"]
    assert provenance["daytona"]["source_files"] == pins
    assert report["authored_submissions"] == 60 and report["total_final_outcomes"] == 21600
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0")
    selected = set(pins)
    for name in filter(None, tracked):
        if (name.startswith(("src/", "engine-rs/", "skirmish/", "harbor/", "configs/skirmish/", "docs/skirmish/", "licenses/"))
                or name.startswith("scripts/") and "skirmish" in Path(name).name
                or name.startswith("tests/test_skirmish")
                or name.startswith("experiments/modal_skirmish")
                or name in ("README.md", "pyproject.toml", "THIRD_PARTY_NOTICES.md")):
            selected.add(name)
    frozen = DEST / "runner"
    for name in sorted(selected):
        raw = (source / name).read_bytes()
        assert not SENSITIVE.search(raw), "Credential pattern in source: " + name
        path = frozen / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        shutil.copymode(source / name, path)
    assert all(sha((frozen / name).read_bytes()) == value for name, value in pins.items())
    for backend in ("local", "daytona"):
        old = source / "artifacts/harbor-packages" / f"split-{backend}-release-v6-20261031"
        shutil.copytree(old, DEST / "releases" / backend, dirs_exist_ok=True)
    shutil.copytree(source / "artifacts/skirmish-references", DEST / "references", dirs_exist_ok=True)
    (DEST / "campaign").mkdir(exist_ok=True)
    for name in ("accepted-results.json", "grading-recovery-jobs.json", "infrastructure-retry.json", "monitor-status.json"):
        shutil.copyfile(campaign / name, DEST / "campaign" / name)
    monitor_tools = ["monitor_full_campaign.py", "monitor_campaign_probe.py", "watch_grading_recovery.py",
                     "checkpoint_cloud_compact_api.py", "validate_compact_results.py", "validate_finished_trials.py",
                     "track_accepted_results.py", "checkpoint_full_grading.py", "compact_grading_checkpoint.py",
                     "grading_recovery.py", "restore_compact_replay.py", "finalize_full_benchmark.py", "verify_final_benchmark.py"]
    (DEST / "tools/campaign").mkdir(exist_ok=True)
    for name in monitor_tools:
        shutil.copyfile(source / "artifacts" / name, DEST / "tools/campaign" / name)

    # Credentials are read only to prevent publishing them in author/tool logs.
    secrets = []
    key = Path.home() / ".config/polytopia-skirmish/openrouter.key"
    if key.exists():
        secrets.append(key.read_bytes().strip())
    for name in ("OPENROUTER_API_KEY", "CLIPROXY_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DAYTONA_API_KEY", "DAYTONA_JWT_TOKEN"):
        if os.environ.get(name):
            secrets.append(os.environ[name].encode())
    proxy_key = Path(os.environ.get("CLIPROXY_KEY_FILE", "~/.cli-proxy-api/client-keys.json")).expanduser()
    if proxy_key.exists():
        secrets.append(json.loads(proxy_key.read_text())["api_key"].encode())
    import sys
    sys.path.insert(0, str(source))
    from scripts.check_skirmish_daytona import credentials
    auth = credentials()
    secrets += [auth[k].encode() for k in ("api_key", "jwt_token") if auth.get(k)]
    redactions = []

    def publish(raw, name):
        original = raw
        for secret in secrets:
            if secret and len(secret) > 12:
                raw = raw.replace(secret, b"<REDACTED_CREDENTIAL>")
        raw = SENSITIVE.sub(b"<REDACTED_CREDENTIAL>", raw)
        if raw != original:
            redactions.append(dict(path=name, original_sha256=sha(original), published_sha256=sha(raw)))
        return raw

    compact = {}
    for path in (campaign / "validated-compact-trials").glob("*/validated.json"):
        value = json.loads(path.read_text())
        compact[value["compact_blob_sha256"]] = path.parent / "replay.json.gz"

    def author_archive(trial, target):
        files = {}
        for path in (trial / "agent").rglob("*"):
            if path.is_file():
                name = str(path.relative_to(trial))
                files[name] = publish(path.read_bytes(), trial.name + "/" + name)
        for name in ("config.json", "result.json", "exception.txt", "trial.log"):
            path = trial / name
            if path.is_file():
                files[name] = publish(path.read_bytes(), trial.name + "/" + name)
        submission = trial / "artifacts/submission"
        for path in submission.rglob("*"):
            if path.is_file():
                files[str(path.relative_to(trial))] = publish(path.read_bytes(), str(path))
        pack_files(files, target / "authoring.tar.gz")

    def practice_archive(trial, target):
        files, records = {}, []
        for folder in sorted((trial / "artifacts/practice").iterdir()) if (trial / "artifacts/practice").exists() else []:
            if not folder.is_dir():
                continue
            public, private, result = (folder / name for name in ("game.jsonl", "game.private.jsonl", "report.json"))
            record = {"id": folder.name, "completed": result.exists(), "files": {}}
            for path in (private, result):
                if path.exists():
                    raw = path.read_bytes()
                    files[folder.name + "/" + path.name] = raw
                    record["files"][path.name] = dict(sha256=sha(raw), bytes=len(raw))
            if public.exists():
                with public.open("rb") as stream:
                    public_sha = hashlib.file_digest(stream, "sha256").hexdigest()
                record["files"][public.name] = dict(sha256=public_sha, bytes=public.stat().st_size)
                if not result.exists():
                    # Interrupted attempts lack their map/seat receipt. Keep exact
                    # public bytes for these instead of promising reconstruction.
                    files[folder.name + "/" + public.name] = public.read_bytes()
                    record["public_encoding"] = "exact public bytes in tar"
                else:
                    record["public_encoding"] = "pinned engine replay from exact private log and report.map_id/seat"
            records.append(record)
        files["index.json"] = (json.dumps(records, indent=2) + "\n").encode()
        pack_files(files, target / "practice.tar.gz")
        return records

    def final_archive(item):
        original = source / item["original_trial"]
        target = DEST / "runs" / item["model"].replace("/", "__") / item["trial"]
        target.mkdir(parents=True, exist_ok=True)
        receipt = json.loads((result_dir / item["saved_validation_receipt"]).read_text())
        if item["validation_kind"] == "lossless_replay":
            raw = compact[receipt["compact_blob_sha256"]].read_bytes()
            assert sha(raw) == receipt["compact_blob_sha256"]
            payload = json.loads(gzip.decompress(raw))
            final_blob = raw  # Preserve the exact already-accepted blob digest.
        else:
            logs = source / item["graded_trial"] / "verifier"
            names = ("summary.json", "games.json", "evaluation-plan.json", "submission.json", "reward.json")
            final = {name: json.loads((logs / name).read_text()) for name in names}
            final["wasm_base64"] = base64.b64encode((logs / "bot.wasm").read_bytes()).decode()
            games = []
            for row in final["games.json"]:
                relative = Path(row["trace_relative"]).relative_to("games")
                public = logs / "traces" / relative
                plain = public.with_suffix("") if public.suffix == ".gz" else public
                raw_private = plain.with_suffix(".private.jsonl").read_bytes()
                assert sha(raw_private) == row["private_trace_sha256"]
                entries = [[int(p), int(v), int(t), a.decode()] for a, p, t, v in ENTRY.findall(raw_private)]
                assert len(entries) == row["actions"]
                games.append(dict(summary=row, actions=[e[3] for e in entries], entries=entries))
            payload = dict(schema="skirmish-compact-replay-v1", source_sha256=item["source_sha256"],
                           referee_source_sha256=pins["src/polytopia_bot/skirmish/local_referee.py"], games=games, final=final)
            final_blob = gzip.compress(canonical(payload).encode(), compresslevel=9, mtime=0)
        assert payload["source_sha256"] == item["source_sha256"] and len(payload["games"]) == 360
        assert sha(base64.b64decode(payload["final"]["wasm_base64"])) == item["wasm_sha256"]
        (target / "final-replay.json.gz").write_bytes(final_blob)
        author_archive(original, target)
        practice = practice_archive(original, target)
        print(item["model"], item["trial"], "exported", flush=True)
        return dict(model=item["model"], trial=item["trial"], original_trial=item["original_trial"],
                    graded_trial=item["graded_trial"], path=str(target.relative_to(DEST)), source_sha256=item["source_sha256"],
                    wasm_sha256=item["wasm_sha256"], games=360, practice_directories=len(practice),
                    completed_practice=sum(p["completed"] for p in practice))

    with ThreadPoolExecutor(max_workers=3) as pool:
        trials = list(pool.map(final_archive, report["trials"]))
    failed = source / "artifacts/harbor-runs" / (report["campaign"] + "-daytona") / "task__aUYod56"
    failed_target = DEST / "runs/failed-authoring/task__aUYod56"
    failed_target.mkdir(parents=True, exist_ok=True)
    author_archive(failed, failed_target)
    failed_practice = practice_archive(failed, failed_target)
    main_base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=DEST.parent, text=True).strip()
    manifest = dict(schema="polytopia-benchmark-main-archive-v1", source_commit=commit, main_base_commit=main_base,
                    campaign=report["campaign"], models=12, authored_submissions=60, final_games=21600,
                    frozen_source_files=pins, trials=trials, failed_authoring=dict(path=str(failed_target.relative_to(DEST)),
                    excluded_from_scores=True, practice_directories=len(failed_practice)), credential_redactions=redactions,
                    storage=dict(final_public_observations="lossless pinned-engine replay; original SHA256 retained",
                    practice_public_observations="completed reports use pinned-engine replay; interrupted logs retain exact bytes",
                    excluded="Container/VM images, virtual environments, caches, duplicate expanded traces and unrelated pilot runtime directories"))
    cache_parts = {"__pycache__", ".pytest_cache", "artifacts", "target", "node_modules", ".venv"}
    manifest["files"] = {str(path.relative_to(DEST)): dict(sha256=sha(path.read_bytes()), bytes=path.stat().st_size)
                         for path in sorted(DEST.rglob("*")) if path.is_file()
                         and path.name not in ("archive-manifest.json", "archive-validation.json")
                         and not cache_parts.intersection(path.relative_to(DEST).parts)}
    json_write(DEST / "archive-manifest.json", manifest)
    print("EXPORTED", len(trials), "trials; MB", round(sum(f["bytes"] for f in manifest["files"].values()) / 1e6, 1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    export(parser.parse_args().source)
