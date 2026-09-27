"""Deterministic bounded scenario domain for Tiny Dino certification."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal

from .dino_engine import DINO_X, OBSTACLE_GEOMETRY, DinoEngine, EngineState, Obstacle

DOMAIN_VERSION = "dino-domain-v1"

@dataclass(frozen=True, slots=True)
class DomainSpec:
    speeds: tuple[float, ...]
    distances: tuple[float, ...]
    gaps: tuple[float, ...]
    control_ages: tuple[int, ...]

    @classmethod
    def small(cls) -> DomainSpec:
        # Dense timing-boundary coverage is essential: three isolated distances let a
        # classifier fit the samples while missing the jump transition between them.
        return cls(
            (360.0, 570.0, 780.0),
            # Below 75 px, some grounded large-obstacle states have no physically
            # survivable action. They are invalid initial conditions, not policy
            # choices, so the actionable certification lattice starts at 75 px.
            tuple(float(distance) for distance in range(75, 225, 5)),
            (180.0, 300.0),
            (0, 15, 30, 60),
        )

@dataclass(frozen=True, slots=True)
class CertCase:
    case_id: str
    state: EngineState
    provenance: Literal["reachable", "synthetic_boundary"]
    family: str

def enumerate_domain(spec: DomainSpec) -> list[CertCase]:
    base_engine = DinoEngine(); base_engine.reset(0); base = base_engine.snapshot()
    cases: list[CertCase] = []
    index = 0
    for speed in spec.speeds:
        for distance in spec.distances:
            for kind in ("small", "large", "bird"):
                width, height = OBSTACLE_GEOMETRY[kind]
                for y in ((25.0, 50.0, 75.0) if kind == "bird" else (0.0,)):
                    obstacle = Obstacle(kind, DINO_X + distance, width, height, y)
                    for control_age in spec.control_ages:
                        state = replace(
                            base,
                            speed=speed,
                            frames_since_control=control_age,
                            obstacle=obstacle,
                            next_obstacle=None,
                        )
                        cases.append(CertCase(f"case-{index:06d}", state, "synthetic_boundary", "bird" if kind == "bird" else "single")); index += 1
                        for gap in spec.gaps:
                            nxt = Obstacle("small", obstacle.x + width + gap, 17.0, 35.0, 0.0)
                            cases.append(CertCase(f"case-{index:06d}", replace(state, next_obstacle=nxt), "synthetic_boundary", "paired")); index += 1
    return cases

def domain_manifest(cases: Sequence[CertCase], spec: DomainSpec) -> dict[str, object]:
    rows = [{"id": c.case_id, "family": c.family, "provenance": c.provenance, "state": c.state.as_dict()} for c in cases]
    canonical = json.dumps({"version": DOMAIN_VERSION, "spec": asdict(spec), "cases": rows}, sort_keys=True, separators=(",", ":"))
    return {"version": DOMAIN_VERSION, "case_count": len(cases), "sha256": hashlib.sha256(canonical.encode()).hexdigest(), "spec": asdict(spec)}
