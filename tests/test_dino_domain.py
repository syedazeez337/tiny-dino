from dataclasses import replace

from tiny_dino.dino_domain import DomainSpec, domain_manifest, enumerate_domain
from tiny_dino.dino_engine import DINO_X, DinoEngine, Obstacle
from tiny_dino.dino_teacher import Teacher


def test_domain_is_deterministic_and_covers_families() -> None:
    spec = DomainSpec.small()
    first = enumerate_domain(spec)
    second = enumerate_domain(spec)
    assert first == second
    assert {case.family for case in first} >= {"single", "bird", "paired"}
    assert domain_manifest(first, spec) == domain_manifest(second, spec)
    assert len(domain_manifest(first, spec)["sha256"]) == 64


def test_domain_lower_bound_excludes_only_unavoidable_large_obstacles() -> None:
    spec = DomainSpec.small()
    assert min(spec.distances) == 75.0
    engine = DinoEngine(); engine.reset(0); teacher = Teacher(engine, 90)
    for distance in (60.0, 65.0, 70.0):
        state = replace(
            engine.snapshot(),
            obstacle=Obstacle("large", DINO_X + distance, 25.0, 50.0, 0.0),
            next_obstacle=None,
        )
        assert teacher.evaluate(state).acceptable == ()
