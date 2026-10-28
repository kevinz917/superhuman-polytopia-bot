"""Engine adapter contract only. This module implements no game mechanics."""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class LegalAction:
    action_id: str
    command: Mapping[str, Any]


@dataclass(frozen=True)
class Observation:
    player: int
    state_version: int
    visible_state: Mapping[str, Any]
    remembered_state: Mapping[str, Any]
    legal_actions: Sequence[LegalAction]


@dataclass(frozen=True)
class Transition:
    actor_observation: Observation
    next_player: int | None
    rewards: Mapping[int, float]
    terminated: bool
    truncated: bool
    reason: str | None


class Environment(Protocol):
    """One logical game; a worker may host multiple instances."""

    def reset(self, *, seed: int, settings: Mapping[str, Any]) -> Observation: ...

    def observe(self, player: int) -> Observation: ...

    def step(self, *, action_id: str, expected_state_version: int) -> Transition: ...

    def snapshot(self) -> bytes:
        """Private trusted-controller state. Never expose to a policy."""
        ...

    def restore(self, snapshot: bytes) -> Observation: ...

    def close(self) -> None: ...
