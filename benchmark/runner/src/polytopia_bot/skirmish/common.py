"""Pure metadata operations; simulation and compilation run on Modal."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

SCHEMA = 'polytopia-skirmish-v2'
REFERENCE = 'hero-1v1'
REFERENCE_SHA = 'e6f4e1977e7f9cdcfbbe17358d87e49d638d82150496a6d4099e0cd389b0934f'
PRACTICE_START, PRACTICE_END = 500_000, 599_999
FINAL_START, FINAL_END = 1_000_000_000, 2_000_000_000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}', value):
        raise ValueError('identifier must contain only letters, numbers, underscore and hyphen (1–96)')
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


@dataclass(frozen=True)
class Budget:
    seconds: int = 3600
    constraint: str = 'time'
    max_revisions: int | None = None
    max_requests: int = 300
    max_output_tokens: int = 12000
    max_practice_games: int = 128
    max_campaign_maps: int = 4
    move_seconds: float = 2.0
    game_seconds: int = 180
    fuel: int = 1_000_000_000_000

    def validate(self):
        if self.constraint not in ('time', 'revisions', 'combined'):
            raise ValueError('unknown active constraint')
        if self.max_revisions is not None and (type(self.max_revisions) is not int or not 1 <= self.max_revisions <= 100):
            raise ValueError('revision allowance must be 1..100')
        if self.constraint != 'time' and self.max_revisions is None:
            raise ValueError('revision/combined tracks require max_revisions')
        if not 30 <= self.seconds <= 7200 or not 1 <= self.max_requests <= 1000:
            raise ValueError('invalid authoring budget')
        if not 1 <= self.max_output_tokens <= 32768 or not 2 <= self.max_practice_games <= 512:
            raise ValueError('invalid API/practice budget')
        if not 1 <= self.max_campaign_maps <= 16 or not 0 < self.move_seconds <= 30:
            raise ValueError('invalid campaign/decision budget')
        if not 10 <= self.game_seconds <= 1800 or not 1000 <= self.fuel <= 1_000_000_000_000:
            raise ValueError('invalid game/fuel budget')
        return self


def paired_schedule(labels, seeds):
    labels, seeds = list(labels), list(seeds)
    if len(labels) < 2 or len(set(labels)) != len(labels) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('need distinct entrants and nonempty distinct maps')
    for label in labels:
        identifier(label)
    return [dict(a=a, b=b, seed=seed, seat=seat)
            for i, a in enumerate(labels) for b in labels[i + 1:]
            for seed in seeds for seat in (1, 2)]


def validate_seed(seed, phase):
    bounds = (PRACTICE_START, PRACTICE_END) if phase == 'practice' else (FINAL_START, FINAL_END)
    if phase not in ('practice', 'final') or not bounds[0] <= seed <= bounds[1]:
        raise ValueError('map outside the phase partition')
    return seed
