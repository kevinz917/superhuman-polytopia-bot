"""Replayable demonstrations and explicitly masked SFT examples."""
from dataclasses import asdict
import json
from pathlib import Path

from polytopia_bot.environment import Observation, LegalAction
from .interface import SCHEMA, canonical, command, digest, render
from .teacher import ScriptedTeacher

RECORD_SCHEMA = "polytopia-demo-v1"


def observation_from_dict(row):
    if set(row) != {"player", "state_version", "visible_state", "remembered_state", "legal_actions"}:
        raise ValueError("invalid observation envelope")
    actions = []
    for a in row["legal_actions"]:
        if set(a) != {"action_id", "command"}:
            raise ValueError("invalid action envelope")
        actions.append(LegalAction(a["action_id"], command(a["command"])))
    obs = Observation(row["player"], row["state_version"], row["visible_state"], row["remembered_state"], actions)
    render(obs)
    return obs


def demonstrate(obs, teacher, *, episode_id, map_hash, decision_index, source="teacher", split="train"):
    selection = teacher.select(obs)
    return {"schema": RECORD_SCHEMA, "source": source, "split": split, "episode_id": episode_id,
            "map_hash": map_hash, "decision_index": decision_index, "observation": asdict(obs),
            "teacher_hash": selection.teacher_hash, "selected_command": dict(selection.action.command),
            "teacher_scores": selection.scores}


def example(row, *, strategy=""):
    required = {"schema", "source", "split", "episode_id", "map_hash", "decision_index", "observation", "teacher_hash", "selected_command", "teacher_scores"}
    if set(row) != required or row["schema"] != RECORD_SCHEMA:
        raise ValueError("invalid demonstration record")
    if row["split"] != "train":
        raise ValueError("only training-split demonstrations may enter SFT")
    if row["source"] not in ("teacher", "fixture"):
        raise ValueError("unknown demonstration source")
    for field in ("episode_id", "map_hash", "teacher_hash"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError("missing data provenance")
    if type(row["decision_index"]) is not int or row["decision_index"] < 0:
        raise ValueError("invalid decision index")
    obs = observation_from_dict(row["observation"])
    prompt = render(obs)
    selected = command(row["selected_command"])
    matches = [k for k, a in prompt.catalog.items() if a.command == selected]
    if len(matches) != 1:
        raise ValueError("teacher target is not in the complete legal catalog")
    public = {k: v for k, v in prompt.public.items() if k != "state_version"}
    messages = prompt.messages
    if strategy:
        from .prompt_search import candidate_messages
        messages = candidate_messages(obs, strategy)
    return messages, canonical({"action_id": matches[0]}), digest(public)


def tokenize_example(tokenizer, messages, response, *, max_length, max_new_tokens):
    """Prove prefix identity; supervise JSON and <|im_end|>, never the prompt."""
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": response}],
                                         tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError("chat template does not preserve the generation prefix")
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos is None or eos == tokenizer.unk_token_id:
        raise ValueError("tokenizer must define <|im_end|>")
    try:
        end = full.index(eos, len(prefix)) + 1
    except ValueError:
        raise ValueError("missing assistant end-of-message token") from None
    ids = full[:end]
    if tokenizer.decode(ids[len(prefix):end - 1], skip_special_tokens=False) != response:
        raise ValueError("response token roundtrip mismatch")
    if len(ids) > max_length or len(ids) - len(prefix) > max_new_tokens:
        raise ValueError("example exceeds context/response budget; no truncation permitted")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + ids[len(prefix):]}


class ResponseCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        size = max(len(r["input_ids"]) for r in rows)
        result = {k: [] for k in ("input_ids", "attention_mask", "labels")}
        for r in rows:
            padding = size - len(r["input_ids"])
            for k, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
                result[k].append(r[k] + [pad] * padding)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}


def iter_records(path):
    count = 0
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    example(row)
                except (ValueError, KeyError, TypeError) as e:
                    raise ValueError(f"invalid record at line {number}: {e}") from e
                count += 1
                yield row
    if not count:
        raise ValueError("empty corpus")


class CorpusAudit:
    def __init__(self, *, allow_fixtures=False, split_manifest=None, strategy=""):
        self.allow_fixtures = allow_fixtures
        self.strategy = strategy
        self.groups = None
        if split_manifest is not None:
            self.groups = [set(split_manifest[k]) for k in ("train", "validation", "test")]
            if any(self.groups[i] & self.groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                raise ValueError("map hashes overlap across data splits")
        self.seen, self.episodes = {}, {}
        self.sources, self.teachers, self.maps = set(), set(), set()
        self.count, self.duplicates = 0, 0

    def accept(self, row):
        messages, response, key = example(row, strategy=self.strategy)
        if row["source"] == "fixture" and not self.allow_fixtures:
            raise ValueError("synthetic fixtures cannot enter a real SFT corpus")
        if self.groups is not None and row["map_hash"] not in self.groups[0]:
            raise ValueError("map absent from frozen training split")
        episode = row["episode_id"]
        if episode in self.episodes and self.episodes[episode] != row["map_hash"]:
            raise ValueError("episode associated with different maps")
        self.episodes[episode] = row["map_hash"]
        self.sources.add(row["source"])
        self.teachers.add(row["teacher_hash"])
        self.maps.add(row["map_hash"])
        self.count += 1
        target = canonical(row["selected_command"])
        if key in self.seen:
            if self.seen[key] != target:
                raise ValueError("conflicting teacher labels for identical public input")
            self.duplicates += 1
            return None
        self.seen[key] = target
        return row, messages, response

    def report(self):
        return {"input_records": self.count, "unique_records": len(self.seen), "duplicates": self.duplicates,
                "episodes": len(self.episodes), "maps": len(self.maps), "sources": sorted(self.sources),
                "teacher_hashes": sorted(self.teachers), "observation_schema": SCHEMA}


def audit_records(rows, *, allow_fixtures=False, split_manifest=None):
    audit = CorpusAudit(allow_fixtures=allow_fixtures, split_manifest=split_manifest)
    unique = [accepted for row in rows if (accepted := audit.accept(row)) is not None]
    return unique, audit.report()


class JsonlDataset:
    """Random-access token records; keep offsets rather than all tokens in RAM."""
    def __init__(self, path):
        self.path = str(path)
        self.offsets = []
        with open(self.path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        offset = self.offsets[index]
        with open(self.path, "rb") as f:
            f.seek(offset)
            return json.loads(f.readline())


def fixture_observation():
    # Deliberately synthetic public position; no claim of simulator reachability.
    return Observation(1, 0, {"board_size": 3, "turn": 1, "stars": 5, "technology": [0, 15],
        "tiles": [{"tile": i, "known": True, "terrain": 3, "owner": 1 if i == 4 else 0,
                   "structure": 1 if i == 5 else 0, "resource": 0} if i < 6 else {"tile": i, "known": False} for i in range(9)],
        "units": [{"tile": 4, "owner": 1, "kind": 2, "health": 10}],
        "cities": [{"tile": 4, "owner": 1, "level": 1, "production": 1}]}, {},
        [LegalAction("engine-move", {"kind": 1, "source": 4, "target": 5, "parameter": 0}),
         LegalAction("engine-end", {"kind": 10, "source": -1, "target": -1, "parameter": 0})])


def write_fixtures(path, count=8):
    from dataclasses import replace
    teacher = ScriptedTeacher()
    base = fixture_observation()
    rows = [demonstrate(replace(base, state_version=i, visible_state={**base.visible_state, "stars": i + 2}), teacher,
                        episode_id=f"fixture-{i}", map_hash=digest(["synthetic", i]), decision_index=0, source="fixture") for i in range(count)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as f:
        f.write("".join(canonical(r) + "\n" for r in rows))
    return len(rows)
