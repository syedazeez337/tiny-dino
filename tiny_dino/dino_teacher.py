"""Exact forward-simulation teacher, never imported by live play."""
from __future__ import annotations

from dataclasses import dataclass

from .dino_contract import ACTION_ORDER, Action
from .dino_engine import DinoEngine, EngineState, clearance


@dataclass(frozen=True, slots=True)
class TeacherVerdict:
    acceptable: tuple[Action, ...]
    survival: dict[Action, bool]
    min_margin: dict[Action, float]
    clearance_frame: dict[Action, int | None]
    rejection_reason: dict[Action, str | None]
    outside_horizon: bool
    horizon_frames: int

class Teacher:
    def __init__(self, engine: DinoEngine, lookahead_frames: int) -> None:
        if lookahead_frames < 1:
            raise ValueError("lookahead_frames must be positive")
        self.engine, self.lookahead_frames = engine, lookahead_frames

    def evaluate(self, state: EngineState) -> TeacherVerdict:
        original = self.engine.snapshot()
        survival: dict[Action, bool] = {}
        margins: dict[Action, float] = {}
        cleared: dict[Action, int | None] = {}
        reasons: dict[Action, str | None] = {}
        try:
            for action in ACTION_ORDER:
                self.engine.restore(state)
                target = state.obstacle
                margin = float("inf")
                clear_frame: int | None = None
                for offset in range(self.lookahead_frames):
                    chosen = action if offset == 0 else self._continuation(self.engine.snapshot())
                    self.engine.step(chosen)
                    margin = min(margin, clearance(self.engine.collision_boxes()))
                    if self.engine.is_terminal():
                        break
                    current = self.engine.snapshot().obstacle
                    if target is not None and current != target:
                        clear_frame = offset + 1
                        break
                alive = not self.engine.is_terminal()
                survival[action], margins[action], cleared[action] = alive, margin, clear_frame
                reasons[action] = None if alive else "collision"
        finally:
            self.engine.restore(original)
        acceptable = tuple(action for action in ACTION_ORDER if survival[action])
        return TeacherVerdict(acceptable, survival, margins, cleared, reasons, not acceptable, self.lookahead_frames)

    @staticmethod
    def _continuation(state: EngineState) -> Action:
        if state.dino_y > 0.0:
            if state.dino_vy <= 0.0 and state.obstacle is not None:
                relative_x = state.obstacle.x - state.distance - 60.0
                if relative_x < -20.0:
                    return "duck"
            return "jump" if state.dino_vy > 180.0 and state.frames_since_control < 8 else "run"
        obstacle = state.obstacle
        if obstacle is None:
            return "run"
        relative_x = obstacle.x - state.distance - 60.0
        trigger = max(85.0, state.speed * 0.34)
        if relative_x > trigger:
            return "run"
        if obstacle.kind == "bird" and obstacle.y <= 25.0:
            return "duck"
        if obstacle.kind in {"small", "large"}:
            return "jump"
        return "run"
