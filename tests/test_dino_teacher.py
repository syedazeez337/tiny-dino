from dataclasses import replace

from tiny_dino.dino_engine import DinoEngine, EngineConfig
from tiny_dino.dino_teacher import Teacher


def test_teacher_is_deterministic_and_does_not_mutate_engine() -> None:
    engine = DinoEngine(EngineConfig(first_obstacle_delay_px=180.0))
    engine.reset(7)
    state = engine.snapshot()
    teacher = Teacher(engine, lookahead_frames=90)
    first = teacher.evaluate(state)
    second = teacher.evaluate(state)
    assert first == second
    assert engine.snapshot() == state
    assert first.acceptable


def test_teacher_reports_all_three_actions() -> None:
    engine = DinoEngine(EngineConfig(first_obstacle_delay_px=1_000_000.0))
    engine.reset(1)
    verdict = Teacher(engine, 12).evaluate(engine.snapshot())
    assert verdict.acceptable == ("run", "jump", "duck")
    assert set(verdict.survival) == {"run", "jump", "duck"}


def test_teacher_fast_drops_after_clearing_an_obstacle() -> None:
    engine = DinoEngine()
    engine.reset(1)
    state = replace(engine.snapshot(), dino_y=80.0, dino_vy=-120.0, distance=700.0)
    assert Teacher._continuation(state) == "duck"
