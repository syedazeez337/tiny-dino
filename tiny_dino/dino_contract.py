"""Versioned feature and action contract shared by every Dino path."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .dino_engine import DINO_X, DinoEngine, EngineState, Obstacle

Action = Literal["run", "jump", "duck"]
ACTION_ORDER: tuple[Action, ...] = ("run", "jump", "duck")
FEATURE_VERSION = "dino-features-v1"
FEATURE_LAYOUT = (
    "speed", "dino_y", "dino_vy", "grounded", "rising", "falling", "ducking",
    "control_run", "control_jump", "control_duck", "frames_since_control",
    "obstacle_0_rel_x", "obstacle_0_width", "obstacle_0_height", "obstacle_0_y",
    "obstacle_0_small", "obstacle_0_large", "obstacle_0_bird",
    "obstacle_1_rel_x", "obstacle_1_width", "obstacle_1_height", "obstacle_1_y",
    "obstacle_1_small", "obstacle_1_large", "obstacle_1_bird",
    "obstacle_gap", "obstacle_0_missing", "obstacle_1_missing",
)
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
        return tuple((float(value) - mean) / scale for value, mean, scale in zip(features, self.mean, self.scale, strict=True))

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "scale": list(self.scale)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[float]]) -> Normalization:
        return cls(tuple(map(float, payload["mean"])), tuple(map(float, payload["scale"])))


def _obstacle_features(obstacle: Obstacle | None, state: EngineState) -> tuple[float, ...]:
    if obstacle is None:
        return (0.0,) * 7
    return (
        obstacle.x - state.distance - DINO_X, obstacle.width, obstacle.height, obstacle.y,
        float(obstacle.kind == "small"), float(obstacle.kind == "large"), float(obstacle.kind == "bird"),
    )


def extract_features(state: EngineState) -> tuple[float, ...]:
    first = _obstacle_features(state.obstacle, state)
    second = _obstacle_features(state.next_obstacle, state)
    gap = 0.0
    if state.obstacle is not None and state.next_obstacle is not None:
        gap = state.next_obstacle.x - (state.obstacle.x + state.obstacle.width)
    values = (
        state.speed, state.dino_y, state.dino_vy,
        float(state.motion == "grounded"), float(state.motion == "rising"),
        float(state.motion == "falling"), float(state.ducking),
        float(state.control == "run"), float(state.control == "jump"), float(state.control == "duck"),
        float(min(state.frames_since_control, 60)), *first, *second, gap,
        float(state.obstacle is None), float(state.next_obstacle is None),
    )
    if len(values) != FEATURE_DIM:
        raise AssertionError("feature contract drift")
    return values


def normalize(features: Sequence[float], norm: Normalization) -> tuple[float, ...]:
    return norm.encode(features)


def feature_hash() -> str:
    payload = json.dumps({"version": FEATURE_VERSION, "layout": FEATURE_LAYOUT}, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def apply_action(engine: DinoEngine, action: Action) -> None:
    engine.step(action)
