"""Versioned feature and action contract shared by every tic-tac-toe path."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .ttt_engine import EMPTY, GameState, other

N_ACTIONS = 9
ACTION_ORDER: tuple[int, ...] = tuple(range(N_ACTIONS))
FEATURE_VERSION = "ttt-features-v1"

FEATURE_LAYOUT: tuple[str, ...] = tuple(
    f"cell_{index}_{role}"
    for index in range(N_ACTIONS)
    for role in ("empty", "mine", "opp")
) + ("fullness",)

FEATURE_DIM = len(FEATURE_LAYOUT)


@dataclass(frozen=True, slots=True)
class Normalization:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.mean) != FEATURE_DIM or len(self.scale) != FEATURE_DIM:
            raise ValueError(f"normalization vectors must have {FEATURE_DIM} elements")
        if any(value <= 0.0 for value in self.scale):
            raise ValueError("normalization scales must be positive")

    def encode(self, features: Sequence[float]) -> tuple[float, ...]:
        if len(features) != FEATURE_DIM:
            raise ValueError(f"feature vector must have {FEATURE_DIM} elements")
        return tuple(
            (float(value) - mean) / scale
            for value, mean, scale in zip(features, self.mean, self.scale, strict=True)
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "scale": list(self.scale)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[float]]) -> Normalization:
        return cls(tuple(map(float, payload["mean"])), tuple(map(float, payload["scale"])))


IDENTITY_NORMALIZATION = Normalization(
    mean=(0.0,) * FEATURE_DIM,
    scale=(1.0,) * FEATURE_DIM,
)


def extract_features(state: GameState, perspective: int) -> tuple[float, ...]:
    """Return the FEATURE_DIM vector from the point of view of ``perspective``.

    Cells are encoded relative to the controlled player (empty / mine / opponent)
    so a single network plays both X and O symmetrically. The final feature is the
    fraction of empty cells, a cheap progress signal.
    """
    opponent = other(perspective)
    values: list[float] = []
    for cell in state.board:
        values.append(1.0 if cell == EMPTY else 0.0)
        values.append(1.0 if cell == perspective else 0.0)
        values.append(1.0 if cell == opponent else 0.0)
    empties = sum(1 for cell in state.board if cell == EMPTY)
    values.append(empties / N_ACTIONS)
    if len(values) != FEATURE_DIM:
        raise AssertionError("feature contract drift")
    return tuple(values)


def legal_mask_tuple(state: GameState) -> tuple[bool, ...]:
    return tuple(cell == EMPTY for cell in state.board)


def feature_hash() -> str:
    payload = json.dumps(
        {"version": FEATURE_VERSION, "layout": FEATURE_LAYOUT}, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()
