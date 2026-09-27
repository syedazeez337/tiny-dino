import pytest

from tiny_dino.dino_engine import (
    DINO_HEIGHT,
    FRAME_SECONDS,
    MAX_SPEED,
    Box,
    DinoEngine,
    EngineConfig,
    EngineState,
    clearance,
    derive_motion,
)


def _run(seed: int, actions: list[str]) -> list[EngineState]:
    engine = DinoEngine(EngineConfig(first_obstacle_delay_px=1_000_000.0))
    engine.reset(seed)
    states = [engine.snapshot()]
    for action in actions:
        engine.step(action)  # type: ignore[arg-type]
        states.append(engine.snapshot())
    return states


ACTIONS = ["run", "jump", "jump", "run", "run", "duck", "run"] * 20


def test_fixed_step_determinism() -> None:
    assert _run(7, ACTIONS) == _run(7, ACTIONS)


def test_frame_advances_once_per_step() -> None:
    assert [s.frame for s in _run(1, ["run"] * 5)] == [0, 1, 2, 3, 4, 5]


def test_speed_clamps_at_max() -> None:
    engine = DinoEngine()
    engine.reset(3)
    for _ in range(20_000):
        if engine.is_terminal():
            engine.reset(3)
        engine.step("jump")
    assert engine.snapshot().speed <= MAX_SPEED


def test_jump_arc_rises_then_grounds() -> None:
    states = _run(5, ["jump"] + ["run"] * 120)
    ys = [state.dino_y for state in states]
    assert ys[0] == 0.0 and max(ys) > 20.0 and ys[-1] == 0.0
    assert states[1].motion == "rising" and states[-1].motion == "grounded"


def test_clone_and_restore_are_exact_and_isolated() -> None:
    engine = DinoEngine()
    engine.reset(11)
    for action in ["run", "jump", "run", "run"]:
        engine.step(action)  # type: ignore[arg-type]
    snapshot = engine.snapshot()
    clone = engine.clone()
    engine.step("duck")
    assert clone.snapshot() == snapshot
    engine.restore(snapshot)
    assert engine.snapshot() == snapshot


def test_hand_traced_hit_and_safe_path() -> None:
    dino = Box(60.0, 0.0, 44.0, DINO_HEIGHT)
    far = Box(200.0, 0.0, 17.0, 35.0)
    near = Box(80.0, 0.0, 17.0, 35.0)
    assert not dino.overlaps(far)
    assert dino.overlaps(near)
    assert clearance((dino, far)) == pytest.approx(96.0)
    assert clearance((dino, near)) < 0.0


def test_reset_is_reproducible() -> None:
    first, second = DinoEngine(), DinoEngine()
    first.reset(42)
    second.reset(42)
    assert first.snapshot() == second.snapshot()


def test_derive_motion_boundaries() -> None:
    assert derive_motion(0.0, 0.0, True) == "ducking"
    assert derive_motion(50.0, 600.0, False) == "rising"
    assert derive_motion(120.0, 0.0, False) == "apex"
    assert derive_motion(50.0, -600.0, False) == "falling"
    assert derive_motion(0.0, 0.0, False) == "grounded"


def test_state_round_trips_through_dict() -> None:
    engine = DinoEngine()
    engine.reset(9)
    engine.step("jump")
    state = engine.snapshot()
    assert EngineState.from_dict(state.as_dict()) == state
    assert FRAME_SECONDS == pytest.approx(1.0 / 60.0)


def test_holding_jump_does_not_retrigger_on_landing() -> None:
    engine = DinoEngine(EngineConfig(first_obstacle_delay_px=1_000_000.0))
    engine.reset(3)
    engine.step("jump")
    airborne_frames = 0
    while engine.snapshot().dino_y > 0.0:
        airborne_frames += 1
        engine.step("jump")
    assert airborne_frames > 1
    engine.step("jump")
    assert engine.snapshot().dino_y == 0.0


def test_ducking_in_air_triggers_a_fast_drop() -> None:
    engine = DinoEngine()
    engine.reset(7)
    engine.step("jump")
    while engine.snapshot().dino_vy > 0.0:
        engine.step("run")
    before = engine.snapshot().dino_vy
    engine.step("duck")
    assert engine.snapshot().dino_vy < min(before, -300.0)
    assert not engine.snapshot().ducking
