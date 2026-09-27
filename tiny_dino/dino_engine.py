"""Deterministic fixed-step runner physics for the Tiny Dino specialist."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import inf
from typing import Literal, cast

FRAME_HZ = 60
FRAME_SECONDS = 1.0 / FRAME_HZ
DINO_ENGINE_REVISION = "dino-engine-v2"
ENGINE_SCHEMA = 1
GROUND_Y = 0.0
DINO_X = 60.0
DINO_WIDTH = 44.0
DINO_HEIGHT = 47.0
DINO_DUCK_HEIGHT = 25.0
GRAVITY = -2160.0
JUMP_VELOCITY = 600.0
MIN_JUMP_VELOCITY = 300.0
DROP_VELOCITY = -300.0
JUMP_HOLD_GRAVITY_SCALE = 0.5
START_SPEED = 360.0
MAX_SPEED = 780.0
ACCELERATION = 3.6
OBSTACLE_KINDS = ("small", "large", "bird")
OBSTACLE_GEOMETRY = {"small": (17.0, 35.0), "large": (25.0, 50.0), "bird": (46.0, 30.0)}
BIRD_HEIGHTS = (25.0, 50.0, 75.0)
FIRST_OBSTACLE_DELAY_PX = 600.0
GAP_MIN_FRAMES = 30
GAP_MAX_FRAMES = 90

ControlState = Literal["run", "jump", "duck"]
Motion = Literal["grounded", "rising", "apex", "falling", "landing", "ducking"]


def _number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError("expected numeric engine-state field")
    return float(value)


def _integer(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError("expected integer engine-state field")
    return value


def derive_motion(dino_y: float, dino_vy: float, ducking: bool) -> Motion:
    if ducking and dino_y <= 0.0:
        return "ducking"
    if dino_y > 0.0 and dino_vy > 0.0:
        return "rising"
    if dino_y > 0.0 and dino_vy == 0.0:
        return "apex"
    if dino_y > 0.0 and dino_vy < 0.0:
        return "falling"
    return "grounded"


@dataclass(frozen=True, slots=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    def overlaps(self, other: Box) -> bool:
        return (
            self.x < other.x + other.width
            and self.x + self.width > other.x
            and self.y < other.y + other.height
            and self.y + self.height > other.y
        )

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True, slots=True)
class Obstacle:
    kind: str
    x: float
    width: float
    height: float
    y: float

    def as_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "x": self.x, "width": self.width, "height": self.height, "y": self.y}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Obstacle:
        kind = str(payload["kind"])
        if kind not in OBSTACLE_KINDS:
            raise ValueError(f"unknown obstacle kind: {kind}")
        return cls(kind, _number(payload["x"]), _number(payload["width"]), _number(payload["height"]), _number(payload["y"]))


@dataclass(frozen=True, slots=True)
class EngineConfig:
    gravity: float = GRAVITY
    jump_velocity: float = JUMP_VELOCITY
    min_jump_velocity: float = MIN_JUMP_VELOCITY
    jump_hold_scale: float = JUMP_HOLD_GRAVITY_SCALE
    start_speed: float = START_SPEED
    max_speed: float = MAX_SPEED
    acceleration: float = ACCELERATION
    first_obstacle_delay_px: float = FIRST_OBSTACLE_DELAY_PX
    gap_min_frames: int = GAP_MIN_FRAMES
    gap_max_frames: int = GAP_MAX_FRAMES
    bird_heights: tuple[float, ...] = BIRD_HEIGHTS


@dataclass(frozen=True, slots=True)
class EngineState:
    frame: int
    distance: float
    speed: float
    accel_state: int
    dino_y: float
    dino_vy: float
    motion: Motion
    ducking: bool
    control: ControlState
    frames_since_control: int
    obstacle: Obstacle | None
    next_obstacle: Obstacle | None
    rng_state: int
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": ENGINE_SCHEMA, "engine_revision": DINO_ENGINE_REVISION,
            "frame": self.frame, "distance": self.distance, "speed": self.speed,
            "accel_state": self.accel_state, "dino_y": self.dino_y, "dino_vy": self.dino_vy,
            "motion": self.motion, "ducking": self.ducking, "control": self.control,
            "frames_since_control": self.frames_since_control,
            "obstacle": None if self.obstacle is None else self.obstacle.as_dict(),
            "next_obstacle": None if self.next_obstacle is None else self.next_obstacle.as_dict(),
            "rng_state": self.rng_state, "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EngineState:
        if payload.get("schema") != ENGINE_SCHEMA or payload.get("engine_revision") != DINO_ENGINE_REVISION:
            raise ValueError("incompatible engine state")
        obstacle = payload.get("obstacle")
        next_obstacle = payload.get("next_obstacle")
        return cls(
            _integer(payload["frame"]), _number(payload["distance"]), _number(payload["speed"]),
            _integer(payload["accel_state"]), _number(payload["dino_y"]), _number(payload["dino_vy"]),
            cast(Motion, payload["motion"]), bool(payload["ducking"]),
            cast(ControlState, payload["control"]), _integer(payload["frames_since_control"]),
            None if obstacle is None else Obstacle.from_dict(cast(Mapping[str, object], obstacle)),
            None if next_obstacle is None else Obstacle.from_dict(cast(Mapping[str, object], next_obstacle)),
            _integer(payload["rng_state"]), _integer(payload["seed"]),
        )


def clearance(boxes: Sequence[Box]) -> float:
    if len(boxes) < 2:
        return inf
    dino = boxes[0]
    best = inf
    for other in boxes[1:]:
        horizontal = max(other.x - (dino.x + dino.width), dino.x - (other.x + other.width))
        vertical = max(other.y - (dino.y + dino.height), dino.y - (other.y + other.height))
        best = min(best, max(horizontal, vertical) if not dino.overlaps(other) else -min(dino.width, other.width))
    return best


class DinoEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self._config = config or EngineConfig()
        self._ready = False
        self._obstacle: Obstacle | None = None
        self._next_obstacle: Obstacle | None = None

    def reset(self, seed: int) -> None:
        self._frame, self._distance, self._speed = 0, 0.0, self._config.start_speed
        self._accel_state, self._dino_y, self._dino_vy = 1, 0.0, 0.0
        self._motion, self._ducking, self._control = "grounded", False, "run"
        self._frames_since_control, self._seed = 0, seed
        self._rng_state = (seed ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF or 1
        width, height = OBSTACLE_GEOMETRY["small"]
        self._obstacle = Obstacle("small", self._config.first_obstacle_delay_px, width, height, 0.0)
        self._next_obstacle = self._spawn(self._obstacle.x + self._obstacle.width)
        self._collided, self._ready = False, True

    def _next_rng(self) -> int:
        x = self._rng_state & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self._rng_state = x & 0xFFFFFFFFFFFFFFFF
        return self._rng_state

    def _spawn(self, anchor_x: float) -> Obstacle:
        kind = OBSTACLE_KINDS[self._next_rng() % len(OBSTACLE_KINDS)]
        width, height = OBSTACLE_GEOMETRY[kind]
        y = self._config.bird_heights[self._next_rng() % len(self._config.bird_heights)] if kind == "bird" else 0.0
        span = self._config.gap_max_frames - self._config.gap_min_frames + 1
        gap_frames = self._config.gap_min_frames + self._next_rng() % span
        return Obstacle(kind, anchor_x + self._speed * gap_frames / FRAME_HZ, width, height, y)

    def step(self, action: ControlState) -> None:
        if not self._ready:
            raise RuntimeError("engine has not been reset")
        if self._collided:
            raise RuntimeError("cannot step a terminal engine")
        if action not in ("run", "jump", "duck"):
            raise ValueError(f"invalid control: {action}")
        grounded = self._dino_y <= 0.0
        control_changed = action != self._control
        if control_changed:
            self._control, self._frames_since_control = action, 0
        else:
            self._frames_since_control += 1
        self._ducking = action == "duck" and grounded
        if action == "jump":
            if grounded and self._dino_vy <= 0.0 and control_changed:
                self._dino_vy = self._config.jump_velocity
            scale = self._config.jump_hold_scale if self._dino_vy > 0.0 else 1.0
        else:
            self._dino_vy = min(self._dino_vy, self._config.min_jump_velocity)
            if action == "duck" and not grounded:
                self._dino_vy = min(self._dino_vy, DROP_VELOCITY)
            scale = 1.0
        self._dino_vy += self._config.gravity * scale * FRAME_SECONDS
        self._dino_y += self._dino_vy * FRAME_SECONDS
        if self._dino_y <= 0.0:
            self._dino_y, self._dino_vy = 0.0, 0.0
        self._frame += 1
        self._speed = min(self._config.max_speed, self._config.start_speed + self._config.acceleration * self._frame * FRAME_SECONDS)
        self._accel_state = 0 if self._speed >= self._config.max_speed else 1
        self._distance += self._speed * FRAME_SECONDS
        self._advance_obstacles()
        self._motion = derive_motion(self._dino_y, self._dino_vy, self._ducking)
        self._collided = self._collides()

    def _advance_obstacles(self) -> None:
        if self._obstacle is not None and self._obstacle.x + self._obstacle.width < self._distance:
            self._obstacle, self._next_obstacle = self._next_obstacle, None
        if self._next_obstacle is None:
            anchor = self._obstacle.x + self._obstacle.width if self._obstacle else self._distance + self._config.first_obstacle_delay_px
            self._next_obstacle = self._spawn(anchor)

    def collision_boxes(self) -> tuple[Box, ...]:
        dino = Box(DINO_X, self._dino_y, DINO_WIDTH, DINO_DUCK_HEIGHT if self._ducking else DINO_HEIGHT)
        boxes = [dino]
        for obstacle in (self._obstacle, self._next_obstacle):
            if obstacle is not None:
                boxes.append(Box(obstacle.x - self._distance, obstacle.y, obstacle.width, obstacle.height))
        return tuple(boxes)

    def _collides(self) -> bool:
        boxes = self.collision_boxes()
        return any(boxes[0].overlaps(other) for other in boxes[1:])

    def snapshot(self) -> EngineState:
        if not self._ready:
            raise RuntimeError("engine has not been reset")
        return EngineState(self._frame, self._distance, self._speed, self._accel_state, self._dino_y, self._dino_vy, cast(Motion, self._motion), self._ducking, cast(ControlState, self._control), self._frames_since_control, self._obstacle, self._next_obstacle, self._rng_state, self._seed)

    def restore(self, state: EngineState) -> None:
        if state.control not in ("run", "jump", "duck"):
            raise ValueError("invalid control")
        self._frame, self._distance, self._speed, self._accel_state = state.frame, state.distance, state.speed, state.accel_state
        self._dino_y, self._dino_vy, self._motion = state.dino_y, state.dino_vy, state.motion
        self._ducking, self._control, self._frames_since_control = state.ducking, state.control, state.frames_since_control
        self._obstacle, self._next_obstacle = state.obstacle, state.next_obstacle
        self._rng_state, self._seed, self._collided, self._ready = state.rng_state, state.seed, False, True
        self._collided = self._collides()

    def is_terminal(self) -> bool:
        return self._collided

    def clone(self) -> DinoEngine:
        clone = DinoEngine(self._config)
        clone.restore(self.snapshot())
        return clone
