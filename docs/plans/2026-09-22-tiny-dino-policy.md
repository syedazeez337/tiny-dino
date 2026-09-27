# Tiny Dino Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one small, genuinely unassisted neural policy that plays a deterministic, Chromium-compatible Dino runner, choose every live action itself, and certify it over a frozen bounded domain plus preregistered held-out and soak suites — proving zero unsafe actions and zero crashes on that domain while never claiming the endless game is solved. Expose the result through the existing Play / Arena / Proof / replay stack without changing the preserved Nim implementation, artifacts, or checkpoint hashes.

**Architecture:** Six bounded Python units share one source of truth for game rules, feature extraction, action semantics, and collision decisions: a deterministic fixed-step **engine**, an exact forward-simulation **teacher** (unavailable to the live path), a **dataset generator** with hashed manifests, a small **policy** classifier over a fixed normalized feature vector, a **certification/arena** layer that evaluates a frozen artifact without altering its actions, and a FastAPI + React **web application** that consumes serialized frames and telemetry rather than reimplementing physics. New code is additive and flat (`rlvr_curriculum/dino_*.py`), mirroring the existing `nim_*.py` modules; the Nim Python product, artifacts, and hashes are frozen.

**Tech Stack:** Python 3.13, PyTorch (CUDA 12.8 wheel), NumPy, pytest, Ruff, strict mypy, uv; FastAPI, uvicorn; React 19, Vite 7, Vitest 3, TypeScript 5.8, ESLint 9.

**Spec:** `docs/superpowers/specs/2026-09-22-tiny-dino-policy-design.md` (commit `33f0ea3`).

## Global Constraints

- **Local only.** Windows target: RTX 3050 Laptop GPU (6 GB), i5-13450HX, 16 GB RAM. No paid services, no cloud API, no authentication, database, leaderboard, analytics, chat, or unrelated UI framework.
- **Deterministic engine.** 60 FPS fixed-step advance (exactly `1/60` s per step) with explicit `reset`, `step`, `snapshot`, `restore`, and collision interfaces; exact replay from a serialized initial state plus actions.
- **Unassisted policy.** The policy chooses every live action from structured state features. Zero runtime planner, solver, or heuristic interventions. The teacher is reachable only from dataset / certification / arena / replay paths — never from `/api/play/*`.
- **Everything hashes.** Every corpus, domain, and certification manifest carries a stable hash; the shipped artifact is the exact artifact whose SHA-256 appears on the Proof page.
- **Latency gate.** One forward pass must run at substantially less than one 60 FPS frame, target p99 `< 16.67 ms` and p95 well under it, on the target machine. Latency is reported separately from policy correctness.
- **Calibration before long training.** Every long run is preceded by a frozen `calibration.json`; the orchestrator checks it first. If the forecast exceeds the budget the run is refused — gates are never weakened to fit.
- **Claim language is constrained** to the spec §11 wording. The endless stochastic game is not claimed solved.
- **Integrity first.** Missing or hash-mismatched checkpoint, or engine/checkpoint schema mismatch, refuses startup. Non-finite feature/logit/probability stops the run and persists a violation. Invalid action stops; it is never substituted. Interrupted certification resumes without counting incomplete cases as passed. Failed jobs and exceptions surface diagnostics, and completed runs are never overwritten.
- **Preserve Nim.** Nim Python modules, artifacts, reports, and checkpoint hashes remain untouched and reproducible. Repository gates (`pytest`, `ruff`, `mypy`, and the frontend gates) stay green at every milestone.
- **Licensing.** Record the exact reference commit of `https://github.com/virajbhartiya/laya-vs-jev.git` before implementing. No vendored model weights. Any adapted Chromium code/assets retain the BSD-3-Clause notice; adapted repository code keeps its original attribution and license.

## Review Focus

- **Planner leakage:** no module under the live-play import graph may import `dino_teacher` (or any solver/heuristic). An import-graph test enforces this. `dino_arena` and `dino_web` are inside that graph, so the arena is teacher-free by construction: it reports crashes, not certified safe/unsafe verdicts (those come only from M6 certification).
- **Action integrity:** the live move must equal the frozen artifact's greedy action; an unsafe or malformed choice stops the run and writes a violation, and the teacher is not invoked as a fallback.
- **Split isolation:** train, dev, held-out, and DAgger corpora are disjoint by course seed and scenario family, never by random row; held-out seeds are frozen before selection.
- **Calibration honesty:** the frozen `calibration.json` bounds batch/epochs/forecast/VRAM/latency and is inspected before the long run; the fallback ladder never weakens a gate.
- **Selection discipline:** candidates are chosen lexicographically — pass certification, pass held-out/soak, minimize deployed bytes, minimize p95 latency — never by training accuracy alone.
- **Quantization parity:** INT8 PTQ that changes any certification action forces quantization-aware training and recertification; the shipped hash equals the certified hash.
- **Preservation:** the Nim baseline manifest test stays green and no `nim*` file appears in `git status`.

## Execution Model

Each milestone is **one reviewable CommandCode run** in an isolated worktree, executed with `deepseek/deepseek-v4-flash` in `--yolo` mode. Setup:

```powershell
git worktree add ..\tiny-dino -b feature/tiny-dino-policy
cd ..\tiny-dino
$env:UV_CACHE_DIR='.uv-cache'
uv sync
uv run pytest -q   # baseline must be green before M0
```

Strictly ordered. Per milestone: **failing test (RED) → implement → focused green → repository gates → commit.** The orchestrator inspects diff and artifacts and re-runs the listed commands before accepting. CommandCode may edit only this worktree: no push, publish, deploy, history rewrite, Nim-checkpoint changes, or completion claims from its own prose. Implementation commits and generated experimental artifacts stay separate. No agent or bot attribution is added to commits.

## Shared conventions

- All new modules are fully typed (`from __future__ import annotations`, PEP 604 unions, `slots=True` dataclasses). No new `disallow_untyped_calls` override is added to `pyproject.toml`.
- All JSON artifacts are written with `rlvr_curriculum.io.atomic_write_json`; JSONL with `rlvr_curriculum.io.append_jsonl`.
- Fixed vector length `FEATURE_DIM = 28` and action order `("run", "jump", "duck")` are the single source of truth used by engine, teacher, dataset, training, certification, arena, and web.
- Every hash is lowercase hex SHA-256 over the canonical serialization defined at its definition site.

## Module Map

All new modules live flat under `rlvr_curriculum/`:

| Milestone | Module(s) | Tests |
|---|---|---|
| M0 | `docs/superpowers/dino-provenance.md`, `docs/superpowers/dino-provenance.json`, `docs/superpowers/nim-baseline.sha256` | `tests/test_nim_baseline.py` |
| M1 | `dino_engine.py` | `tests/test_dino_engine.py` |
| M2 | `dino_contract.py` | `tests/test_dino_contract.py` |
| M3 | `dino_teacher.py`, `dino_domain.py` | `tests/test_dino_teacher.py`, `tests/test_dino_domain.py` |
| M4 | `dino_dataset.py` | `tests/test_dino_dataset.py` |
| M5 | `dino_model.py`, `dino_train.py` | `tests/test_dino_model.py`, `tests/test_dino_train.py` |
| M6 | `dino_certify.py` | `tests/test_dino_certify.py` |
| M7 | `dino_dagger.py` | `tests/test_dino_dagger.py`, `tests/test_dino_selection.py` |
| M8 | `dino_arena.py`, `dino_web.py` | `tests/test_dino_arena.py`, `tests/test_dino_web.py` |
| M9 | `web/src/**` | `web/src/test/*.test.tsx` |
| M10 | `tests/test_dino_end_to_end.py`, `reports/dino/**` | `reports/dino/evidence.*` |

Supporting paths: `configs/dino-*.json`, `artifacts/dino/**`, `reports/dino/**`, `web/qa/dino-*.png`.

---

### M0 — Provenance, Isolation, and Nim Preservation Guard

**Files:** Create `docs/superpowers/dino-provenance.md`, `docs/superpowers/dino-provenance.json`, `docs/superpowers/nim-baseline.sha256`, `tests/test_nim_baseline.py`. Modify: none.

**Interfaces:** The provenance JSON is the machine-readable pin that later milestones read; the baseline manifest pins Nim.

```python
# docs/superpowers/dino-provenance.json  (schema written by this milestone)
# {
#   "schema": 1,
#   "reference_repo": "https://github.com/virajbhartiya/laya-vs-jev.git",
#   "reference_commit": "<40 lowercase hex chars>",
#   "source_path": "<path inside the reference repo that was pinned>",
#   "source_sha256": "<64 lowercase hex chars>",
#   "recorded_on": "<YYYY-MM-DD>",
#   "license_verdict": "adapted" | "rejected",
#   "adapted_elements": [],          # one string per adapted element, [] when none
#   "rejected_elements": ["assisted gameplay"],
#   "checkpoint_sha256": "<64 lowercase hex chars or empty on a clean checkout>"
# }

def build_nim_baseline(repo_root: Path) -> dict[str, str]:
    """Map every ``rlvr_curriculum/nim*.py`` and ``configs/nim-*.json`` path to its
    SHA-256 at the pinned commit. Return ``{relative_path: hex_digest}`` sorted by key."""

def load_provenance(repo_root: Path) -> dict[str, object]:
    """Read ``docs/superpowers/dino-provenance.json`` and assert the field set above;
    raise ``ValueError`` if ``reference_commit`` is not 40 hex chars or
    ``source_sha256`` is not 64 hex chars."""
```

- [ ] RED: write `tests/test_nim_baseline.py` to pin the SHA-256 of every `rlvr_curriculum/nim*.py` and `configs/nim-*.json` at commit `087e873`, assert working-tree hashes and the `nim*.py` file set match, verify the ignored checkpoint digest only when present (mirror the `needs_model` skip pattern in `tests/test_nim_arena.py`), and assert `artifacts/nim/**` and `reports/**` are uninvolved. Include a provenance-shape test.

```python
# tests/test_nim_baseline.py
import hashlib, json, subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "superpowers" / "nim-baseline.sha256"
PROVENANCE = ROOT / "docs" / "superpowers" / "dino-provenance.json"
PINNED_COMMIT = "087e873"
CKPT = ROOT / "artifacts" / "nim" / "action-policy" / "full" / "full-policy-only-seed101.pt"
needs_model = pytest.mark.skipif(not CKPT.exists(), reason="specialist checkpoint absent")

def _git_show(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{PINNED_COMMIT}:{path}"], cwd=ROOT, check=True,
        stdout=subprocess.PIPE,
    ).stdout

def _parse_manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, _, path = line.partition("  ")
        entries[path.strip()] = digest.strip()
    return entries

def test_pinned_paths_match_the_recorded_manifest() -> None:
    entries = _parse_manifest()
    assert entries, "manifest is empty"
    for path, digest in entries.items():
        assert hashlib.sha256(_git_show(path)).hexdigest() == digest, path
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path

def test_nim_file_set_is_frozen() -> None:
    pinned = {p for p in _parse_manifest() if p.startswith("rlvr_curriculum/nim")}
    on_disk = {f"rlvr_curriculum/{p.name}" for p in (ROOT / "rlvr_curriculum").glob("nim*.py")}
    assert pinned == on_disk

def test_provenance_record_is_complete() -> None:
    payload = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert len(payload["reference_commit"]) == 40
    assert len(payload["source_sha256"]) == 64
    assert payload["license_verdict"] in {"adapted", "rejected"}

@needs_model
def test_ignored_checkpoint_digest_when_present() -> None:
    recorded = json.loads(PROVENANCE.read_text(encoding="utf-8"))["checkpoint_sha256"]
    assert len(recorded) == 64
    assert hashlib.sha256(CKPT.read_bytes()).hexdigest() == recorded
```

(`needs_model` skips on a clean checkout exactly as `tests/test_nim_arena.py` does; `checkpoint_sha256` is the only provenance field allowed to be empty on a clean checkout because the frozen artifact is not committed.)

Run `uv run pytest tests/test_nim_baseline.py -v` — expect failure (`nim-baseline.sha256` and `dino-provenance.json` missing).

- [ ] GREEN: generate `docs/superpowers/nim-baseline.sha256` from commit `087e873` with `build_nim_baseline`, write `docs/superpowers/dino-provenance.json` and the human-readable `docs/superpowers/dino-provenance.md` recording the exact reference commit, date, adapted vs rejected elements, and the license verdict (add a BSD-3-Clause `NOTICE` if anything is adapted).

```powershell
uv run python -c "import pathlib, rlvr_curriculum as p; from tests.test_nim_baseline import build_nim_baseline; \
  m = build_nim_baseline(pathlib.Path('.')); \
  pathlib.Path('docs/superpowers/nim-baseline.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in sorted(m.items())), encoding='utf-8')"
```

```python
def build_nim_baseline(repo_root: Path) -> dict[str, str]:
    tracked = [
        *sorted(p.name for p in (repo_root / "rlvr_curriculum").glob("nim*.py")),
    ]
    tracked = [f"rlvr_curriculum/{name}" for name in tracked]
    tracked += [f"configs/{p.name}" for p in sorted((repo_root / "configs").glob("nim-*.json"))]
    baseline: dict[str, str] = {}
    for path in tracked:
        blob = subprocess.run(
            ["git", "show", f"087e873:{path}"], cwd=repo_root, check=True, stdout=subprocess.PIPE
        ).stdout
        baseline[path] = hashlib.sha256(blob).hexdigest()
    return dict(sorted(baseline.items()))
```

Run the test again — expect all green.

- [ ] Verify: `uv run pytest tests/test_nim_baseline.py -v`, then `uv run pytest -q`, `uv run ruff check rlvr_curriculum tests run_experiment.py`, `uv run mypy rlvr_curriculum` — zero failures.
- [ ] Commit: `chore: pin nim baseline and record dino provenance`.

**Gate:** Manifest test green; all pre-existing Nim tests green; concrete reference commit and an explicit license verdict recorded; no `nim*` file appears in `git status`.

---

### M1 — Deterministic Dino Engine

**Files:** Create `rlvr_curriculum/dino_engine.py`, `tests/test_dino_engine.py`, `tests/fixtures/dino/reference_trajectories.json`.

**Interfaces:** Pinned constants, the full state dataclasses, and the engine. All values below are the source of truth; later milestones import them unchanged.

```python
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Literal

FRAME_HZ: int = 60
FRAME_SECONDS: float = 1.0 / FRAME_HZ
DINO_ENGINE_REVISION: str = "dino-engine-v1"
ENGINE_SCHEMA: int = 1

# World geometry (px, y-up; ground is y = 0; x grows to the right).
GROUND_Y: float = 0.0
DINO_X: float = 60.0
DINO_WIDTH: float = 44.0
DINO_HEIGHT: float = 47.0
DINO_DUCK_HEIGHT: float = 25.0

# Physics, Chromium t-rex runner constants converted from per-frame to per-second.
GRAVITY: float = -2160.0            # -0.6 px/frame^2
JUMP_VELOCITY: float = 600.0        # 10 px/frame
MIN_JUMP_VELOCITY: float = 300.0    # release clamp -> variable jump height
JUMP_HOLD_GRAVITY_SCALE: float = 0.5
START_SPEED: float = 360.0          # 6 px/frame
MAX_SPEED: float = 780.0            # 13 px/frame
ACCELERATION: float = 3.6           # 0.001 px/frame^2

OBSTACLE_KINDS: tuple[str, ...] = ("small", "large", "bird")
OBSTACLE_GEOMETRY: dict[str, tuple[float, float]] = {
    "small": (17.0, 35.0), "large": (25.0, 50.0), "bird": (46.0, 30.0),
}
BIRD_HEIGHTS: tuple[float, ...] = (25.0, 50.0, 75.0)
FIRST_OBSTACLE_DELAY_PX: float = 600.0
GAP_MIN_FRAMES: int = 30
GAP_MAX_FRAMES: int = 90
CLEARANCE_TAIL_FRAMES: int = 30

ControlState = Literal["run", "jump", "duck"]
Motion = Literal["grounded", "rising", "apex", "falling", "landing", "ducking"]

def derive_motion(dino_y: float, dino_vy: float, ducking: bool) -> Motion:
    """ducking if ``ducking`` and grounded; apex if airborne with vy == 0.0;
    rising if vy > 0.0; falling if vy < 0.0 and airborne; else grounded."""

@dataclass(frozen=True, slots=True)
class Box:
    x: float          # left edge, screen space, px
    y: float          # bottom edge above ground, px
    width: float
    height: float

    def overlaps(self, other: Box) -> bool:
        """Strict AABB overlap: touching edges do not collide."""

    def as_dict(self) -> dict[str, float]:
        """``{"x": .., "y": .., "width": .., "height": ..}``."""

@dataclass(frozen=True, slots=True)
class Obstacle:
    kind: str         # one of OBSTACLE_KINDS
    x: float          # world x of the left edge, px
    width: float
    height: float
    y: float          # bottom edge above ground (0.0 grounded, >0.0 for birds)

    def as_dict(self) -> dict[str, object]:
        """``{"kind", "x", "width", "height", "y"}`` with floats as-is."""

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Obstacle:
        """Inverse of ``as_dict``; raise ``ValueError`` on an unknown ``kind``."""

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
    distance: float                 # world scroll, px
    speed: float                    # px/s
    accel_state: int                # 0 = at max speed, 1 = still accelerating
    dino_y: float                   # bottom of the dino box above ground, px
    dino_vy: float                  # px/s, positive up
    motion: Motion
    ducking: bool
    control: ControlState
    frames_since_control: int       # reset to 0 on any control change
    obstacle: Obstacle | None
    next_obstacle: Obstacle | None
    rng_state: int
    seed: int

    def as_dict(self) -> dict[str, object]:
        """Canonical serialization: ``schema``, ``engine_revision``, then every declared
        field read by name (``obstacle``/``next_obstacle`` via ``Obstacle.as_dict`` or
        ``None``). Read the fields by name — these state dataclasses are ``slots=True``
        and therefore have no ``__dict__``."""

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EngineState:
        """Inverse of ``as_dict``. Raise ``ValueError`` when ``schema != ENGINE_SCHEMA``
        or ``engine_revision != DINO_ENGINE_REVISION``."""

def clearance(boxes: Sequence[Box]) -> float:
    """Signed AABB separation between ``boxes[0]`` (the dino) and the nearest other
    box: positive px when disjoint, negative when overlapping, and ``inf`` when no
    obstacle box is present. Overlap is detected by ``Box.overlaps``."""

class DinoEngine:
    def __init__(self, config: EngineConfig = EngineConfig()) -> None:
        """Store the config; the engine is unusable until ``reset`` is called."""

    def reset(self, seed: int) -> None:
        """frame=0, distance=0, speed=config.start_speed, accel_state=0, dino_y=0,
        dino_vy=0, motion='grounded', ducking=False, control='run',
        frames_since_control=0, obstacle=None, next_obstacle=None, collided=False,
        rng_state derived deterministically from ``seed`` (never 0), then spawn the
        first obstacle at ``first_obstacle_delay_px``."""

    def step(self, action: ControlState) -> None:
        """Advance exactly ``FRAME_SECONDS``. Raise ``RuntimeError`` if already
        terminal and ``ValueError`` if ``action`` is not a ControlState."""

    def snapshot(self) -> EngineState:
        """Return a fully detached ``EngineState`` (obstacles copied); mutating the
        engine afterwards must not change the snapshot."""

    def restore(self, state: EngineState) -> None:
        """Set every internal field from ``state``; no validation of motion beyond
        ``ValueError`` on an unknown ``control``/``motion``."""

    def collision_boxes(self) -> tuple[Box, ...]:
        """``(dino_box, *obstacle_boxes)`` in screen space; the dino box height is
        ``DINO_DUCK_HEIGHT`` when ducking else ``DINO_HEIGHT`` and obstacle boxes are
        ``Obstacle`` geometry shifted by ``-distance``."""

    def is_terminal(self) -> bool:
        """True once a collision has been registered."""

    def clone(self) -> DinoEngine:
        """Independent engine restored from this engine's snapshot (config shared)."""
```

The single-step semantics (unchanged by any milestone) are:

```python
dt = FRAME_SECONDS
grounded = self._dino_y <= 0.0
if action != self._control:
    self._frames_since_control = 0
    self._control = action
else:
    self._frames_since_control += 1
self._ducking = action == "duck" and grounded
if action == "jump":
    if grounded and self._dino_vy <= 0.0:
        self._dino_vy = self._config.jump_velocity
    scale = self._config.jump_hold_scale if self._dino_vy > 0.0 else 1.0
else:
    if self._dino_vy > self._config.min_jump_velocity:
        self._dino_vy = self._config.min_jump_velocity
    scale = 1.0
self._dino_vy += self._config.gravity * scale * dt
self._dino_y += self._dino_vy * dt
if self._dino_y <= 0.0:
    self._dino_y = 0.0
    self._dino_vy = 0.0
self._frame += 1
self._speed = min(self._config.max_speed,
                  self._config.start_speed + self._config.acceleration * self._frame * dt)
self._accel_state = 0 if self._speed >= self._config.max_speed else 1
self._distance += self._speed * dt
self._advance_obstacles()
self._motion = derive_motion(self._dino_y, self._dino_vy, self._ducking)
if self._collides():
    self._collided = True
```

Obstacle spawning and the deterministic RNG (xorshift64; snapshot-safe because only `rng_state: int` is stored):

```python
def _next_rng(self) -> int:
    x = self._rng_state & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    self._rng_state = x
    return x

def _spawn(self, anchor_x: float) -> Obstacle:
    kind = OBSTACLE_KINDS[self._next_rng() % len(OBSTACLE_KINDS)]
    width, height = OBSTACLE_GEOMETRY[kind]
    if kind == "bird":
        heights = self._config.bird_heights
        y = heights[self._next_rng() % len(heights)]
    else:
        y = 0.0
    span = self._config.gap_max_frames - self._config.gap_min_frames + 1
    gap_frames = self._config.gap_min_frames + self._next_rng() % span
    gap_px = self._speed * gap_frames / FRAME_HZ
    return Obstacle(kind=kind, x=anchor_x + gap_px, width=width, height=height, y=y)

def _advance_obstacles(self) -> None:
    active, nxt = self._obstacle, self._next_obstacle
    if active is not None and active.x + active.width - self._distance < 0.0:
        active, nxt = nxt, None
    if nxt is None:
        anchor = active.x + active.width if active is not None else self._distance + self._config.first_obstacle_delay_px
        nxt = self._spawn(anchor)
    self._obstacle, self._next_obstacle = active, nxt

def _collides(self) -> bool:
    boxes = self.collision_boxes()
    return any(boxes[0].overlaps(other) for other in boxes[1:])
```

- [ ] RED: write `tests/test_dino_engine.py` asserting fixed-step determinism from `(seed, actions)`, `frame += 1` per step, speed clamps, a jump arc that rises → falls → grounds, duck grounded-only, box overlap on a hand-traced hit and no overlap on a safe path, snapshot isolation plus exact `restore`, clones diverging only by action, and `reset(seed)` reproducibility.

```python
# tests/test_dino_engine.py
import pytest
from rlvr_curriculum.dino_engine import (
    DINO_HEIGHT, FRAME_SECONDS, MAX_SPEED, EngineState, Box, DinoEngine, clearance,
    derive_motion,
)

def _run(seed: int, actions: list[str]) -> list[EngineState]:
    engine = DinoEngine()
    engine.reset(seed)
    states = [engine.snapshot()]
    for action in actions:
        engine.step(action)                       # type: ignore[arg-type]
        states.append(engine.snapshot())
    return states

ACTIONS = ["run", "jump", "jump", "run", "run", "duck", "run"] * 20  # type: ignore[misc]

def test_fixed_step_determinism() -> None:
    assert _run(7, ACTIONS) == _run(7, ACTIONS)

def test_frame_advances_once_per_step() -> None:
    states = _run(1, ["run"] * 5)
    assert [s.frame for s in states] == [0, 1, 2, 3, 4, 5]

def test_speed_clamps_at_max() -> None:
    engine = DinoEngine()
    engine.reset(3)
    for _ in range(20_000):
        engine.step("run")
    assert engine.snapshot().speed == pytest.approx(MAX_SPEED)
    assert engine.snapshot().accel_state == 0

def test_jump_arc_rises_then_grounds() -> None:
    states = _run(5, ["jump"] + ["run"] * 120)
    ys = [s.dino_y for s in states]
    assert ys[0] == 0.0 and max(ys) > 100.0 and ys[-1] == 0.0
    assert states[1].motion == "rising" and states[-1].motion == "grounded"

def test_clone_and_restore_are_exact_and_isolated() -> None:
    engine = DinoEngine()
    engine.reset(11)
    for action in ["run", "jump", "run", "run"]:
        engine.step(action)                       # type: ignore[arg-type]
    snap = engine.snapshot()
    clone = engine.clone()
    engine.step("duck")
    assert clone.snapshot() == snap
    engine.restore(snap)
    assert engine.snapshot() == snap

def test_hand_traced_hit_and_safe_path() -> None:
    engine = DinoEngine()
    engine.reset(0)
    safe = Box(60.0, 0.0, 44.0, DINO_HEIGHT)
    assert not safe.overlaps(Box(200.0, 0.0, 17.0, 35.0))
    assert safe.overlaps(Box(80.0, 0.0, 17.0, 35.0))
    assert clearance((safe, Box(200.0, 0.0, 17.0, 35.0))) == pytest.approx(200.0 - 104.0)
    assert clearance((safe, Box(80.0, 0.0, 17.0, 35.0))) < 0.0

def test_reset_is_reproducible() -> None:
    first = DinoEngine()
    first.reset(42)
    second = DinoEngine()
    second.reset(42)
    assert first.snapshot() == second.snapshot()

def test_derive_motion_boundaries() -> None:
    assert derive_motion(0.0, 0.0, True) == "ducking"
    assert derive_motion(50.0, 600.0, False) == "rising"
    assert derive_motion(120.0, 0.0, False) == "apex"
    assert derive_motion(50.0, -600.0, False) == "falling"
    assert derive_motion(0.0, 0.0, False) == "grounded"

def test_step_is_rejected_after_a_terminal_frame() -> None:
    engine = DinoEngine()
    engine.reset(0)
    while not engine.is_terminal():
        engine.step("run")
    with pytest.raises(RuntimeError):
        engine.step("jump")

def test_state_round_trips_through_dict() -> None:
    engine = DinoEngine()
    engine.reset(9)
    engine.step("jump")
    state = engine.snapshot()
    assert EngineState.from_dict(state.as_dict()) == state
    assert FRAME_SECONDS == pytest.approx(1.0 / 60.0)
```

Run `uv run pytest tests/test_dino_engine.py -v` — expect `ModuleNotFoundError: No module named 'rlvr_curriculum.dino_engine'`.

- [ ] GREEN: implement `dino_engine.py` with the pinned constants, dataclasses, and class above. Add differential fixtures `tests/fixtures/dino/reference_trajectories.json` from the pinned revision: a JSON list of `{"seed": int, "actions": list[str], "frames": list[EngineState.as_dict()]}` produced by the pinned reference; when fixtures cannot be legally produced, write the same file with `{"fallback": "pinned_constants"}` and rely on the invariant tests above. Run the test again — expect all green.

```json
// tests/fixtures/dino/reference_trajectories.json  (differential form)
[
  {"seed": 0, "actions": ["run", "jump", "run"],
   "frames": [{"schema": 1, "engine_revision": "dino-engine-v1",
               "frame": 0, "distance": 0.0, "speed": 360.0, "accel_state": 1,
               "dino_y": 0.0, "dino_vy": 0.0, "motion": "grounded", "ducking": false,
               "control": "run", "frames_since_control": 0, "obstacle": null,
               "next_obstacle": null, "rng_state": 12345, "seed": 0}]}
]
```

- [ ] Verify: `uv run pytest tests/test_dino_engine.py -v` (including a 10,000-step seeded determinism property) and `uv run pytest -q`; write `artifacts/dino/engine.json` with `DINO_ENGINE_REVISION` via `io.atomic_write_json`.

```python
atomic_write_json(Path("artifacts/dino/engine.json"), {
    "schema": 1, "engine_revision": DINO_ENGINE_REVISION, "frame_hz": FRAME_HZ,
    "constants": {"gravity": GRAVITY, "jump_velocity": JUMP_VELOCITY,
                  "start_speed": START_SPEED, "max_speed": MAX_SPEED,
                  "acceleration": ACCELERATION},
})
```

- [ ] Commit: `feat: add deterministic fixed-step dino engine`.

**Gate:** All engine tests green; differential fixtures (or documented fallback) present; engine revision written to `engine.json`; no `nim*` regression.

---

### M2 — State/Action Contract: Features and Normalization

**Files:** Create `rlvr_curriculum/dino_contract.py`, `tests/test_dino_contract.py`.

**Interfaces:** The action type, the shared feature contract, the shared error types, and the sole action mutator. Nothing later may re-derive these.

```python
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .dino_engine import ControlState, DinoEngine, EngineState

Action: TypeAlias = ControlState
ACTION_ORDER: tuple[Action, ...] = ("run", "jump", "duck")
N_ACTIONS: int = 3
FEATURE_VERSION: str = "dino-features-v1"
FEATURE_DIM: int = 28

FEATURE_LAYOUT: tuple[str, ...] = (
    "speed", "dino_y", "dino_vy",
    "grounded", "rising", "falling", "ducking",
    "control_run", "control_jump", "control_duck", "frames_since_control",
    "ob0_rel_x", "ob0_width", "ob0_height", "ob0_y",
    "ob0_small", "ob0_large", "ob0_bird",
    "ob1_rel_x", "ob1_width", "ob1_height", "ob1_y",
    "ob1_small", "ob1_large", "ob1_bird",
    "gap", "missing0", "missing1",
)  # len(FEATURE_LAYOUT) == FEATURE_DIM == 28 is asserted at import time

class DinoContractError(ValueError):
    """Base class for every invalid-value failure raised by the dino contract."""

class InvalidActionError(DinoContractError):
    """Raised when an action name or action index falls outside ACTION_ORDER."""

class NonFiniteError(DinoContractError):
    """Raised when a feature, logit, or probability is not finite."""

class FeatureMismatchError(DinoContractError):
    """Raised when a feature vector or normalization has the wrong length."""

class IntegrityError(RuntimeError):
    """Raised when a checkpoint hash, schema, or revision tag fails validation."""

class CalibrationRefused(RuntimeError):
    """Raised when the frozen calibration forbids the requested long run."""

def action_index(action: Action) -> int:
    """Position of ``action`` in ACTION_ORDER; raise ``InvalidActionError`` otherwise."""

def action_from_index(index: int) -> Action:
    """``ACTION_ORDER[index]``; raise ``InvalidActionError`` unless ``0 <= index < 3``."""

@dataclass(frozen=True, slots=True)
class Normalization:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def encode(self, features: Sequence[float]) -> tuple[float, ...]:
        """Elementwise ``(v - mean) / scale``. Raise ``FeatureMismatchError`` when
        lengths differ from ``len(mean) == len(scale) == FEATURE_DIM`` or when any
        ``scale`` entry is 0.0."""

    def to_dict(self) -> dict[str, list[float]]:
        """``{"mean": [28 floats], "scale": [28 floats]}``."""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[float]]) -> Normalization:
        """Inverse of ``to_dict``; raise ``FeatureMismatchError`` unless both lists
        have length FEATURE_DIM."""

def extract_features(state: EngineState) -> tuple[float, ...]:
    """Return the FEATURE_DIM vector in FEATURE_LAYOUT order. Pure: it reads a
    snapshot and never mutates the engine. Missing obstacles contribute zeros for
    geometry and 1.0 for their mask; ``gap`` is 0.0 unless both obstacles exist."""

def normalize(features: Sequence[float], norm: Normalization) -> tuple[float, ...]:
    """Alias for ``norm.encode(features)`` (single documented normalization path)."""

def feature_hash() -> str:
    """SHA-256 over ``f"{FEATURE_VERSION}:{FEATURE_DIM}:" + ",".join(FEATURE_LAYOUT)``;
    stable across imports and sensitive to any layout change."""

def apply_action(engine: DinoEngine, action: Action) -> None:
    """The only control-semantics mutator. Raise ``InvalidActionError`` unless
    ``action`` is in ACTION_ORDER, then call ``engine.step(action)``."""
```

Feature layout (28): speed (1) · dino_y (1) · dino_vy (1) · grounded/rising/falling/ducking (4) · control one-hot (3) · frames since last control transition (1) · two obstacles × (rel_x, width, height, y, type one-hot small/large/bird = 7) = 14 · gap (1) · two missing-obstacle masks (2). `grounded` is `motion == "grounded"`; `rising` covers `{"rising", "apex"}`; `falling` covers `{"falling", "landing"}`.

- [ ] RED: write `tests/test_dino_contract.py` asserting layout length and order, masks zeroing absent geometry, the transition counter resetting on control change, exact and invertible normalization, a stable and sensitive `feature_hash()`, and that `apply_action` is the only control-semantics mutator.

```python
# tests/test_dino_contract.py
import pytest
from rlvr_curriculum.dino_contract import (
    ACTION_ORDER, FEATURE_DIM, FEATURE_LAYOUT, InvalidActionError, Normalization,
    apply_action, extract_features, feature_hash, action_index, action_from_index,
)
from rlvr_curriculum.dino_engine import DinoEngine

def test_layout_is_the_fixed_28_slot_contract() -> None:
    assert len(FEATURE_LAYOUT) == FEATURE_DIM == 28
    assert ACTION_ORDER == ("run", "jump", "duck")

def test_missing_obstacle_geometry_is_zeroed_and_masked() -> None:
    engine = DinoEngine()
    engine.reset(0)
    state = engine.snapshot()
    if state.obstacle is None:
        values = extract_features(state)
        assert values[11:18] == (0.0,) * 7
        assert values[27] == 1.0

def test_feature_extraction_does_not_mutate_the_engine() -> None:
    engine = DinoEngine()
    engine.reset(2)
    for action in ["run", "jump", "run", "duck"]:
        engine.step(action)                       # type: ignore[arg-type]
    before = engine.snapshot()
    extract_features(before)
    assert engine.snapshot() == before

def test_transition_counter_resets_on_control_change() -> None:
    engine = DinoEngine()
    engine.reset(1)
    engine.step("run")
    engine.step("run")
    assert engine.snapshot().frames_since_control == 2
    engine.step("jump")
    assert engine.snapshot().frames_since_control == 0

def test_normalization_round_trips_and_rejects_bad_shapes() -> None:
    norm = Normalization(mean=(0.0,) * FEATURE_DIM, scale=(2.0,) * FEATURE_DIM)
    encoded = norm.encode((2.0,) + (0.0,) * (FEATURE_DIM - 1))
    assert encoded[0] == pytest.approx(1.0)
    assert Normalization.from_dict(norm.to_dict()) == norm
    with pytest.raises(Exception):
        norm.encode((0.0,) * (FEATURE_DIM - 1))

def test_feature_hash_is_stable_and_sensitive() -> None:
    assert feature_hash() == feature_hash()
    assert len(feature_hash()) == 64

def test_apply_action_is_the_single_semantics_mutator() -> None:
    engine = DinoEngine()
    engine.reset(4)
    with pytest.raises(InvalidActionError):
        apply_action(engine, "fly")               # type: ignore[arg-type]
    assert action_from_index(action_index("duck")) == "duck"
```

Run `uv run pytest tests/test_dino_contract.py -v` — expect import failure.

- [ ] GREEN: implement `dino_contract.py`; `extract_features` reads a `snapshot()` and never mutates the engine.

```python
def extract_features(state: EngineState) -> tuple[float, ...]:
    grounded = 1.0 if state.motion == "grounded" else 0.0
    rising = 1.0 if state.motion in {"rising", "apex"} else 0.0
    falling = 1.0 if state.motion in {"falling", "landing"} else 0.0
    ducking = 1.0 if state.ducking else 0.0
    control = tuple(1.0 if state.control == name else 0.0 for name in ACTION_ORDER)
    values: list[float] = [
        state.speed, state.dino_y, state.dino_vy,
        grounded, rising, falling, ducking, *control,
        float(state.frames_since_control),
    ]
    for obstacle in (state.obstacle, state.next_obstacle):
        if obstacle is None:
            values += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            values += [
                obstacle.x - state.distance, obstacle.width, obstacle.height, obstacle.y,
                1.0 if obstacle.kind == "small" else 0.0,
                1.0 if obstacle.kind == "large" else 0.0,
                1.0 if obstacle.kind == "bird" else 0.0,
            ]
    both = state.obstacle is not None and state.next_obstacle is not None
    gap = (
        state.next_obstacle.x - (state.obstacle.x + state.obstacle.width)
        if both else 0.0
    )
    values += [gap, 1.0 if state.obstacle is None else 0.0,
               1.0 if state.next_obstacle is None else 0.0]
    if len(values) != FEATURE_DIM:
        raise FeatureMismatchError(f"feature vector has {len(values)} slots")
    return tuple(values)
```

Run the test again — expect all green.

- [ ] Verify: `uv run pytest tests/test_dino_contract.py -v`; assert `feature_hash()` is stable across imports; confirm features from a snapshot leave the engine bit-identical.
- [ ] Commit: `feat: define dino state-action contract and normalization`.

**Gate:** Contract tests green; `feature_hash()` asserted stable and sensitive; normalization round-trips through the checkpoint dict form.

---

### M3 — Teacher and Frozen Certification Domain

**Files:** Create `rlvr_curriculum/dino_teacher.py`, `rlvr_curriculum/dino_domain.py`, `tests/test_dino_teacher.py`, `tests/test_dino_domain.py`.

**Interfaces:** The exact teacher and the enumeration of the frozen certification domain.

```python
# rlvr_curriculum/dino_teacher.py
from __future__ import annotations
import math
from dataclasses import dataclass
from collections.abc import Mapping
from .dino_contract import ACTION_ORDER, Action, apply_action, extract_features
from .dino_engine import CLEARANCE_TAIL_FRAMES, DinoEngine, EngineState, clearance

REJECTION_COLLISION: str = "collision"
REJECTION_REASON_NONE: str = ""
DEFAULT_LOOKAHEAD_FRAMES: int = 600

@dataclass(frozen=True, slots=True)
class TeacherVerdict:
    acceptable: tuple[Action, ...]           # sorted by ACTION_ORDER
    survival: dict[Action, bool]             # no collision within the horizon
    min_margin: dict[Action, float]          # signed clearance, px
    clearance_frame: dict[Action, int | None]  # frame of min margin, None if rejected
    rejection_reason: dict[Action, str]      # "" | REJECTION_COLLISION
    outside_horizon: bool                    # a next obstacle lies beyond the horizon
    horizon_frames: int                      # horizon actually simulated

    def as_dict(self) -> dict[str, object]:
        """JSON-safe mirror: acceptable as a list, per-action dicts keyed by action."""

    def acceptable_mask(self) -> tuple[bool, bool, bool]:
        """``tuple(a in self.acceptable for a in ACTION_ORDER)`` (length 3)."""

class Teacher:
    def __init__(self, engine: DinoEngine, lookahead_frames: int = DEFAULT_LOOKAHEAD_FRAMES) -> None:
        """Raise ``ValueError`` when ``lookahead_frames < 1``."""

    def evaluate(self, state: EngineState) -> TeacherVerdict:
        """Snapshot the engine, restore ``state``, simulate every candidate action,
        then restore the original snapshot in a ``finally`` block (bit-identical
        before/after). Accept exactly the actions that survive the horizon with
        ``min_margin > 0.0``."""

    def _horizon(self, state: EngineState) -> int:
        """``min(lookahead_frames, frames_for_active_obstacle_to_pass +
        CLEARANCE_TAIL_FRAMES)``; with no active obstacle use ``CLEARANCE_TAIL_FRAMES``."""

    def _simulate(self, action: Action, horizon: int) -> tuple[bool, float, int | None, str]:
        """Clone the engine, apply ``action`` for ``horizon`` frames, tracking the
        minimum ``clearance``. Return ``(survived, min_margin, clearance_frame,
        rejection_reason)``."""
```

```python
# rlvr_curriculum/dino_domain.py
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, itertools
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .dino_contract import ACTION_ORDER, Action, FEATURE_VERSION
from .dino_engine import (
    BIRD_HEIGHTS, MAX_SPEED, START_SPEED, ControlState, DinoEngine, EngineState,
    Motion, Obstacle, derive_motion,
)
from .dino_teacher import DEFAULT_LOOKAHEAD_FRAMES, Teacher

DOMAIN_VERSION: str = "dino-domain-v1"
CertProvenance = Literal["reachable", "synthetic_boundary"]

@dataclass(frozen=True, slots=True)
class DomainSpec:
    version: str
    speeds: tuple[float, ...]
    kinds: tuple[str, ...]
    bird_heights: tuple[float, ...]
    distances: tuple[float, ...]
    gaps: tuple[float, ...]
    motions: tuple[Motion, ...]
    controls: tuple[ControlState, ...]
    duck_transition_frames: tuple[int, ...]
    seed: int
    lookahead_frames: int

    def as_dict(self) -> dict[str, object]:
        """``asdict`` with tuples coerced to lists for JSON."""

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DomainSpec:
        """Inverse of ``as_dict``; raise ``ValueError`` on a missing key."""

DEFAULT_DOMAIN_SPEC: DomainSpec = DomainSpec(
    version=DOMAIN_VERSION,
    speeds=(START_SPEED, 540.0, MAX_SPEED),
    kinds=("small", "large", "bird"),
    bird_heights=BIRD_HEIGHTS,
    distances=(120.0, 200.0, 300.0),
    gaps=(90.0, 300.0, 540.0),
    motions=("grounded", "rising", "apex", "falling", "landing", "ducking"),
    controls=("run", "jump", "duck"),
    duck_transition_frames=(0, 1, 3),
    seed=20260922,
    lookahead_frames=DEFAULT_LOOKAHEAD_FRAMES,
)

@dataclass(frozen=True, slots=True)
class CertCase:
    case_id: str
    state: EngineState
    provenance: CertProvenance
    family: str

def build_state(
    spec: DomainSpec, *, speed: float, kind: str, bird_height: float,
    distance: float, gap: float, motion: Motion, control: ControlState,
    frames_since_control: int,
) -> EngineState:
    """Construct a deterministic ``EngineState`` lattice point: the active obstacle
    sits ``distance`` px ahead, the next sits one ``gap`` further, and the dino
    kinematics follow ``motion`` (grounded/ducking -> y=0.0 vy=0.0;
    rising/falling -> on the engine's jump arc; apex -> y at apex, vy=0.0;
    landing -> y=0.0, vy<0.0)."""

def enumerate_domain(spec: DomainSpec) -> list[CertCase]:
    """Cartesian product of the spec lattice; keep only states whose teacher verdict
    has at least one acceptable action, dedupe by ``case_id``, and return them sorted
    by ``case_id``. Provenance is ``"reachable"`` iff the requested ``motion`` equals
    ``derive_motion(state.dino_y, state.dino_vy, state.ducking)``, else
    ``"synthetic_boundary"``."""

def domain_manifest(cases: Sequence[CertCase], spec: DomainSpec) -> dict[str, Any]:
    """``{"version": DOMAIN_VERSION, "feature_version": FEATURE_VERSION,
    "spec": spec.as_dict(), "count": n, "reachable": r, "synthetic_boundary": s,
    "families": {family: count}, "case_ids_sha256": sha256 of "\\n".join(sorted ids)}``."""

def write_domain_manifest(cases: Sequence[CertCase], spec: DomainSpec, path: Path) -> dict[str, Any]:
    """``atomic_write_json(path, domain_manifest(cases, spec))`` and return the manifest."""
```

- [ ] RED: write `tests/test_dino_teacher.py` covering obvious labels, a jump-too-late case, a duck-under-bird boundary, multiple acceptable actions, a two-obstacle interaction, snapshot isolation (bit-identical before/after), determinism, and `outside_horizon`. Write `tests/test_dino_domain.py` covering every obstacle type/legal size, bird heights, the full legal speed range, min/typical/max gaps, single and paired sequences, the six motion states, reachable vs `synthetic_boundary` tagging, and a stable order-independent manifest hash.

```python
# tests/test_dino_teacher.py
import pytest
from rlvr_curriculum.dino_contract import extract_features
from rlvr_curriculum.dino_domain import DEFAULT_DOMAIN_SPEC, build_state
from rlvr_curriculum.dino_engine import DinoEngine
from rlvr_curriculum.dino_teacher import REJECTION_COLLISION, Teacher

def _teacher() -> Teacher:
    return Teacher(DinoEngine(), DEFAULT_DOMAIN_SPEC.lookahead_frames)

def test_snapshot_isolation_is_bit_identical() -> None:
    engine = DinoEngine()
    engine.reset(5)
    engine.step("run")
    teacher = Teacher(engine, 120)
    before = engine.snapshot()
    teacher.evaluate(before)
    assert engine.snapshot() == before

def test_ground_obstacle_rejects_run_and_accepts_jump() -> None:
    state = build_state(DEFAULT_DOMAIN_SPEC, speed=360.0, kind="small",
                        bird_height=25.0, distance=120.0, gap=300.0,
                        motion="grounded", control="run", frames_since_control=0)
    verdict = _teacher().evaluate(state)
    assert "jump" in verdict.acceptable and "run" not in verdict.acceptable
    assert verdict.rejection_reason["run"] == REJECTION_COLLISION
    assert verdict.min_margin["jump"] > 0.0

def test_low_bird_accepts_duck_and_rejects_run() -> None:
    state = build_state(DEFAULT_DOMAIN_SPEC, speed=360.0, kind="bird",
                        bird_height=25.0, distance=120.0, gap=300.0,
                        motion="grounded", control="run", frames_since_control=0)
    verdict = _teacher().evaluate(state)
    assert "duck" in verdict.acceptable

def test_evaluation_is_deterministic() -> None:
    state = build_state(DEFAULT_DOMAIN_SPEC, speed=360.0, kind="large",
                        bird_height=50.0, distance=200.0, gap=300.0,
                        motion="grounded", control="run", frames_since_control=0)
    assert _teacher().evaluate(state) == _teacher().evaluate(state)
```

```python
# tests/test_dino_domain.py
from rlvr_curriculum.dino_domain import (
    DEFAULT_DOMAIN_SPEC, DOMAIN_VERSION, enumerate_domain, domain_manifest,
)

def test_domain_covers_every_kind_and_motion() -> None:
    cases = enumerate_domain(DEFAULT_DOMAIN_SPEC)
    assert cases, "domain is empty"
    families = {case.family for case in cases}
    assert {"small", "large", "bird"} <= families
    assert {"grounded", "rising", "apex", "falling", "landing", "ducking"} <= {
        case.family.split("_", 1)[1] for case in cases
    }
    assert {case.provenance for case in cases} == {"reachable", "synthetic_boundary"}

def test_manifest_is_stable_and_order_independent() -> None:
    cases = enumerate_domain(DEFAULT_DOMAIN_SPEC)
    first = domain_manifest(cases, DEFAULT_DOMAIN_SPEC)
    second = domain_manifest(list(reversed(cases)), DEFAULT_DOMAIN_SPEC)
    assert first["case_ids_sha256"] == second["case_ids_sha256"]
    assert first["version"] == DOMAIN_VERSION
    assert first["count"] == first["reachable"] + first["synthetic_boundary"]
```

Run `uv run pytest tests/test_dino_teacher.py tests/test_dino_domain.py -v` — expect import failures.

- [ ] GREEN: implement `dino_teacher.py` (snapshot → simulate every candidate over a bounded lookahead → restore) and `dino_domain.py`; write `artifacts/dino/domain/domain-manifest.json`.

```python
def evaluate(self, state: EngineState) -> TeacherVerdict:
    horizon = self._horizon(state)
    saved = self._engine.snapshot()
    try:
        self._engine.restore(state)
        results = {action: self._simulate(action, horizon) for action in ACTION_ORDER}
    finally:
        self._engine.restore(saved)
    acceptable = tuple(a for a in ACTION_ORDER if results[a][0] and results[a][1] > 0.0)
    outside = state.next_obstacle is not None and (
        state.next_obstacle.x - state.distance > state.speed * horizon / 60.0
    )
    return TeacherVerdict(
        acceptable=acceptable,
        survival={a: results[a][0] for a in ACTION_ORDER},
        min_margin={a: results[a][1] for a in ACTION_ORDER},
        clearance_frame={a: results[a][2] for a in ACTION_ORDER},
        rejection_reason={a: results[a][3] for a in ACTION_ORDER},
        outside_horizon=outside,
        horizon_frames=horizon,
    )

def _simulate(self, action: Action, horizon: int) -> tuple[bool, float, int | None, str]:
    probe = self._engine.clone()
    best = math.inf
    best_frame: int | None = None
    for frame in range(horizon):
        apply_action(probe, action)
        margin = clearance(probe.collision_boxes())
        if margin <= 0.0:
            return False, margin, None, REJECTION_COLLISION
        if margin < best:
            best, best_frame = margin, frame
    return True, best, best_frame, REJECTION_REASON_NONE
```

```python
def enumerate_domain(spec: DomainSpec) -> list[CertCase]:
    engine = DinoEngine()
    teacher = Teacher(engine, spec.lookahead_frames)
    found: dict[str, CertCase] = {}
    for speed, kind, height, distance, gap, motion, control, frames in itertools.product(
        spec.speeds, spec.kinds, spec.bird_heights, spec.distances, spec.gaps,
        spec.motions, spec.controls, spec.duck_transition_frames,
    ):
        if kind != "bird" and height != spec.bird_heights[0]:
            continue
        if control != "duck" and frames != spec.duck_transition_frames[0]:
            continue
        state = build_state(spec, speed=speed, kind=kind, bird_height=height,
                            distance=distance, gap=gap, motion=motion, control=control,
                            frames_since_control=frames)
        if not teacher.evaluate(state).acceptable:
            continue
        family = f"{kind}_{motion}"
        provenance = (
            "reachable" if motion == derive_motion(state.dino_y, state.dino_vy, state.ducking)
            else "synthetic_boundary"
        )
        case_id = hashlib.sha256(
            f"{family}|{provenance}|{speed}|{height}|{distance}|{gap}|{control}|{frames}".encode()
        ).hexdigest()[:24]
        found.setdefault(case_id, CertCase(case_id, state, provenance, family))
    return sorted(found.values(), key=lambda case: case.case_id)
```

Run the tests again — expect all green. Write `artifacts/dino/domain/domain-manifest.json` with `write_domain_manifest`.

- [ ] Verify: focused tests plus an **import-graph test** asserting `dino_teacher` is not imported by any live module (`dino_web`, `dino_arena`, `cli` play path).

```python
# tests/test_dino_teacher.py (appended)
def test_teacher_is_outside_the_live_import_graph() -> None:
    import ast, pathlib
    live = ["dino_web", "dino_arena", "cli"]   # certification legitimately imports the teacher
    for name in live:
        module = pathlib.Path("rlvr_curriculum") / f"{name}.py"
        if not module.exists():                # dino_web/dino_arena land in M8
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"))
        imported = {n.module or "" for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom)}
        assert not any(m.endswith("dino_teacher") for m in imported), name
```

- [ ] Commit: `feat: add dino teacher and frozen certification domain`.

**Gate:** Teacher and domain tests green; manifest at `artifacts/dino/domain/domain-manifest.json`; import-graph test proves the teacher is outside the live path.

---

### M4 — Dataset, Manifests, and Leak-Free Splits

**Files:** Create `rlvr_curriculum/dino_dataset.py`, `tests/test_dino_dataset.py`, `configs/dino-corpus.json`.

**Interfaces:** The corpus generator and its manifests. The generator is the only consumer of the teacher besides certification.

```python
from __future__ import annotations
import hashlib, math
from dataclasses import dataclass, asdict
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from .dino_contract import (ACTION_ORDER, FEATURE_DIM, FEATURE_VERSION, Action,
                            apply_action, extract_features)
from .dino_engine import DinoEngine, EngineState
from .dino_teacher import Teacher

DATASET_VERSION: str = "dino-corpus-v1"
FAMILIES: tuple[str, ...] = (
    "empty_track", "single_ground", "single_bird",
    "pair_ground", "pair_mixed", "boundary_timing",
)
SPLITS: tuple[str, ...] = ("train", "dev", "heldout", "dagger")

@dataclass(frozen=True, slots=True)
class DatasetSpec:
    version: str
    families: tuple[str, ...]
    max_empty_run_fraction: float
    train_seeds: tuple[int, ...]
    dev_seeds: tuple[int, ...]
    heldout_seeds: tuple[int, ...]
    dagger_seeds: tuple[int, ...]
    perturbation_radius: float
    frames_per_rollout: int
    lookahead_frames: int

    def as_dict(self) -> dict[str, object]:
        """``asdict`` with tuples coerced to lists."""

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DatasetSpec:
        """Inverse of ``as_dict``; raise ``ValueError`` on a missing key or when the
        four seed tuples overlap."""

@dataclass(frozen=True, slots=True)
class Example:
    features: tuple[float, ...]                 # raw FEATURE_DIM vector, not normalized
    acceptable_mask: tuple[bool, bool, bool]    # ACTION_ORDER order
    course_seed: int
    family: str                                 # one of FAMILIES
    boundary: bool                              # perturbation/boundary perturbation

    def as_dict(self) -> dict[str, object]:
        """``{"features": [28 floats], "acceptable_mask": [3 bools], "course_seed":
        int, "family": str, "boundary": bool}`` with features rounded to 6 decimals
        for hashing stability."""

@dataclass(frozen=True, slots=True)
class Corpus:
    examples: tuple[Example, ...]
    spec: DatasetSpec
    manifests: dict[str, Any]                   # see generate_corpus

def load_dataset_spec(path: Path) -> DatasetSpec:
    """Read ``configs/dino-corpus.json`` (``{"version", "families", "max_empty_run_fraction",
    "train_seeds", "dev_seeds", "heldout_seeds", "dagger_seeds", "perturbation_radius",
    "frames_per_rollout", "lookahead_frames"}``) and validate it."""

def split_of(example: Example, spec: DatasetSpec) -> str:
    """``"train"``/``"dev"``/``"heldout"``/``"dagger"`` by ``course_seed`` membership;
    raise ``ValueError`` when the seed belongs to no split."""

def generate_corpus(spec: DatasetSpec) -> Corpus:
    """Deterministic rollout per (family, seed) using the teacher for labels; store
    raw features; cap empty-track ``run`` states; compute the split normalization."""

def corpus_digest(corpus: Corpus) -> str:
    """SHA-256 over the spec version plus every example's ``as_dict`` in sorted order."""

def write_corpus(corpus: Corpus, output_dir: Path) -> dict[str, Any]:
    """Write ``corpus.jsonl`` (append_jsonl) and ``corpus-manifest.json``
    (atomic_write_json); return the manifest."""
```

`generate_corpus` returns `Corpus.manifests` with exactly these keys:

```python
{
  "version": DATASET_VERSION,
  "feature_version": FEATURE_VERSION,
  "families": {family: count},
  "split_counts": {"train": n, "dev": n, "heldout": n, "dagger": n},
  "empty_run_fraction": float,            # <= spec.max_empty_run_fraction
  "normalization": {"mean": [28 floats], "scale": [28 floats]},  # from train only
  "digest": corpus_digest(corpus),
}
```

Normalization is `mean = sum(x)/n`, `scale = max(std(x), 1e-6)` computed per slot over the train split only.

- [ ] RED: write `tests/test_dino_dataset.py` asserting ≥1 example per family, trivial empty-track `run` states ≤ `max_empty_run_fraction`, **split isolation** by `course_seed` + `family` (train ↔ dev ↔ held-out ↔ dagger disjoint), balanced acceptable-action classes, determinism, hash sensitivity, and an atomic write that leaves no temporary files.

```python
# tests/test_dino_dataset.py
import dataclasses
import json
from pathlib import Path
from rlvr_curriculum.dino_dataset import (
    FAMILIES, DatasetSpec, SPLITS, corpus_digest, generate_corpus, load_dataset_spec,
    split_of, write_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_dataset_spec(ROOT / "configs" / "dino-corpus.json")

def test_every_family_is_present() -> None:
    corpus = generate_corpus(SPEC)
    assert {example.family for example in corpus.examples} == set(FAMILIES)

def test_empty_run_fraction_is_capped() -> None:
    corpus = generate_corpus(SPEC)
    assert corpus.manifests["empty_run_fraction"] <= SPEC.max_empty_run_fraction

def test_splits_are_disjoint_by_seed() -> None:
    corpus = generate_corpus(SPEC)
    buckets: dict[str, set[int]] = {name: set() for name in SPLITS}
    for example in corpus.examples:
        buckets[split_of(example, SPEC)].add(example.course_seed)
    assert not (buckets["train"] & buckets["dev"])
    assert not (buckets["train"] & buckets["heldout"])
    assert not (buckets["heldout"] & buckets["dagger"])

def test_corpus_is_deterministic_and_hash_sensitive() -> None:
    first = generate_corpus(SPEC)
    second = generate_corpus(SPEC)
    assert corpus_digest(first) == corpus_digest(second)
    changed = dataclasses.replace(SPEC, perturbation_radius=SPEC.perturbation_radius + 1.0)
    assert corpus_digest(generate_corpus(changed)) != corpus_digest(first)

def test_write_is_atomic_and_leaves_no_temporaries(tmp_path: Path) -> None:
    corpus = generate_corpus(SPEC)
    manifest = write_corpus(corpus, tmp_path)
    assert (tmp_path / "corpus-manifest.json").exists()
    assert json.loads((tmp_path / "corpus-manifest.json").read_text())["digest"] == manifest["digest"]
    assert not list(tmp_path.glob(".*.tmp"))
```

Run `uv run pytest tests/test_dino_dataset.py -v` — expect import failure.

- [ ] GREEN: implement `dino_dataset.py` reusing `io.atomic_write_json` / `io.append_jsonl`; cap empty-track `run` states and split by seed/family.

```python
def generate_corpus(spec: DatasetSpec) -> Corpus:
    engine = DinoEngine()
    teacher = Teacher(engine, spec.lookahead_frames)
    examples: list[Example] = []
    for family in spec.families:
        for seed in (*spec.train_seeds, *spec.dev_seeds, *spec.heldout_seeds, *spec.dagger_seeds):
            engine.reset(seed)
            for _ in range(spec.frames_per_rollout):
                state = engine.snapshot()
                verdict = teacher.evaluate(state)
                if not verdict.acceptable:
                    break
                mask = verdict.acceptable_mask()
                examples.append(Example(extract_features(state), mask, seed, family,
                                        boundary=verdict.outside_horizon))
                apply_action(engine, _scripted_action(verdict.acceptable))
    kept, fraction = _cap_empty_runs(examples, spec.max_empty_run_fraction)
    corpus = Corpus(tuple(kept), spec, {
        "version": DATASET_VERSION, "feature_version": FEATURE_VERSION,
        "families": _counts(kept, "family"),
        "split_counts": {name: sum(split_of(e, spec) == name for e in kept) for name in SPLITS},
        "empty_run_fraction": fraction,
        "normalization": _normalization(kept, spec),
        "digest": "",
    })
    corpus.manifests["digest"] = corpus_digest(corpus)
    return corpus

def _scripted_action(acceptable: Sequence[Action]) -> Action:
    """Highest-priority acceptable action in ACTION_ORDER (deterministic)."""
    return next(action for action in ACTION_ORDER if action in acceptable)
```

Run the test again — expect all green.

- [ ] Verify: `uv run pytest tests/test_dino_dataset.py -v`; confirm `corpus_digest` is reproducible and that held-out seeds are disjoint from train and DAgger seeds.
- [ ] Commit: `feat: add deterministic dino corpus with leak-free splits`.

**Gate:** Dataset tests green; digest reproducible; split isolation holds by construction.

---

### M5 — Candidates, Distillation, and Compute Calibration

**Files:** Create `rlvr_curriculum/dino_model.py`, `rlvr_curriculum/dino_train.py`, `tests/test_dino_model.py`, `tests/test_dino_train.py`, `configs/dino-train-{mlp32,mlp64,mlp96,transformer}.json`. Modify `rlvr_curriculum/cli.py` (add `dino-calibrate`, `dino-train` plus `--corpus`/`--calibration` arguments).

**Interfaces:** The model candidates, the set-valued loss, training, checkpoint I/O, inference, and calibration.

```python
# rlvr_curriculum/dino_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn

from .dino_contract import FEATURE_DIM, N_ACTIONS

@dataclass(frozen=True, slots=True)
class ModelSpec:
    kind: Literal["mlp", "transformer"]
    hidden: int
    layers: int

CANDIDATES: tuple[ModelSpec, ...] = (
    ModelSpec("mlp", 32, 2), ModelSpec("mlp", 64, 3), ModelSpec("mlp", 96, 3),
)

class ActionMLP(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 32, layers: int = 2,
                 out_dim: int = N_ACTIONS) -> None:
        """``nn.Sequential`` of ``layers`` pairs ``(nn.Linear(prev, hidden), nn.ReLU())``
        followed by ``nn.Linear(hidden, out_dim)``; ``prev`` starts at ``in_dim``."""
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[B, FEATURE_DIM] -> logits [B, N_ACTIONS] (no softmax)."""

class ActionTransformer(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 32, layers: int = 2,
                 heads: int = 4, out_dim: int = N_ACTIONS) -> None:
        """``nn.Linear(1, hidden)`` token embedding plus ``nn.TransformerEncoder``
        (``layers`` layers of ``width=hidden``, ``nhead=heads``, ``dropout=0.0``),
        a ``nn.LayerNorm(hidden)``, and a ``nn.Linear(hidden, out_dim)`` head."""
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[B, FEATURE_DIM] -> logits [B, N_ACTIONS]; treats each feature as one token,
        mean-pools the encoder output, then applies a linear head."""

def build_model(spec: ModelSpec) -> nn.Module:
    """``ActionMLP`` for kind 'mlp', ``ActionTransformer`` for 'transformer'."""

def parameter_count(model: nn.Module) -> int:
    """``sum(p.numel() for p in model.parameters())``."""

def set_valued_loss(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Scalar ``-mean_b log sum_{a in A*} softmax(logits_b)[a]`` with ``allowed`` a
    boolean [B, N_ACTIONS] mask. Implementation mirrors ``nim_train.set_valued_loss``
    (``logsumexp`` over masked log-softmax). Raise ``ValueError`` when any row has no
    allowed action."""
```

```python
# rlvr_curriculum/dino_train.py  (key signatures)
@dataclass(frozen=True, slots=True)
class TrainingSpec:
    version: str
    candidate: ModelSpec
    epochs: int
    batch: int
    learning_rate: float
    weight_decay: float
    seeds: tuple[int, ...]
    device: str                      # "auto" | "cpu" | "cuda"
    max_seconds: int
    corpus_dir: Path
    calibration: Path

def load_training_spec(path: Path) -> TrainingSpec:
    """Read ``configs/dino-train-*.json`` into a ``TrainingSpec``; raise ``ValueError``
    on an unknown ``candidate.kind``, a non-positive ``epochs``/``batch``, or a missing
    key."""

def save_dino_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                         step: int, extra: dict[str, Any]) -> None:
    """Thin wrapper over ``checkpoint.save_checkpoint`` (schema 1)."""

def load_dino_model(checkpoint: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load schema-1 checkpoint, rebuild the model from ``extra['model_spec']``, load
    weights, ``eval()``, ``to(device)``. Raise ``IntegrityError`` when the schema is
    not 1, ``feature_version != FEATURE_VERSION``, ``engine_revision``,
    ``source_revision`` or ``corpus_digest`` is missing, or ``len(normalization.mean)
    != FEATURE_DIM``."""

@torch.no_grad()
def predict_actions(model: nn.Module, features: torch.Tensor, device: torch.device
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized ``features`` [B, FEATURE_DIM] -> ``(indices int64 [B],
    probabilities float32 [B, N_ACTIONS])``. Raise ``NonFiniteError`` when any
    probability is not finite. Restores the model's training flag."""

@torch.no_grad()
def predict_one(model: nn.Module, features: Sequence[float], device: torch.device
                ) -> tuple[int, tuple[float, float, float]]:
    """Single-row convenience returning ``(index, (p_run, p_jump, p_duck))``."""

def train_candidate(spec: TrainingSpec, corpus: Corpus, output_dir: Path) -> dict[str, Any]:
    """Train one seed per ``spec.seeds``; return the report described below."""

def calibrate(config: TrainingSpec, corpus: Corpus) -> dict[str, Any]:
    """Measure throughput and latency, apply the batch fallback ladder, and return the
    calibration payload described below (never modifies ``config``)."""

def write_calibration(payload: Mapping[str, object], output_dir: Path) -> None:
    """``atomic_write_json(output_dir / "calibration.json", payload)``. Raise
    ``FileExistsError`` when a calibration with a different fingerprint exists."""

def check_calibration(spec: TrainingSpec) -> dict[str, Any]:
    """Read ``spec.calibration``. Raise ``CalibrationRefused`` when ``refused`` is
    true, ``forecast_seconds > limits.max_seconds``,
    ``peak_vram_bytes > limits.max_vram_bytes``, or the fingerprint, candidate, or
    corpus digest does not match ``spec``. Return the payload otherwise."""
```

`train_candidate` returns:

```python
{
  "schema": 1, "status": "complete", "candidate": {"kind": str, "hidden": int, "layers": int},
  "seed": int, "params": int, "deployed_bytes": int, "checkpoint": "<name>.pt",
  "feature_version": FEATURE_VERSION, "feature_hash": feature_hash(),
  "normalization": {"mean": [28 floats], "scale": [28 floats]},
  "engine_revision": DINO_ENGINE_REVISION, "source_revision": REFERENCE_COMMIT,
  "corpus_digest": hex, "epochs": int, "curve": [{"epoch": int, "loss": float}],
  "train_loss": float, "wall_seconds": float,
  "latency_ms": {"p50": float, "p95": float, "p99": float},
}
```

`calibrate` returns:

```python
{
  "schema": 1, "fingerprint": hex, "candidate": {"kind": str, "hidden": int, "layers": int},
  "corpus_digest": hex,
  "batch": 4096 | 2048 | 1024, "epochs": int, "forecast_seconds": float,
  "peak_vram_bytes": int, "steps_per_second": float,
  "latency_ms": {"cpu": {"p50": float, "p95": float, "p99": float},
                 "cuda": {"p50": float, "p95": float, "p99": float} | None},
  "limits": {"max_seconds": 1800, "max_vram_bytes": 5_400_000_000,
             "target_p99_ms": 16.67},
  "refused": bool, "reason": str,
}
```

`REFERENCE_COMMIT` comes from `docs/superpowers/dino-provenance.json` via `load_provenance`. `save_dino_checkpoint` `extra` always carries `feature_version`, `normalization`, `engine_revision`, `source_revision`, `corpus_digest`, `model_spec`, `feature_hash`.

`configs/dino-train-mlp32.json` (the other three differ only in `candidate`):

```json
{
  "version": "dino-train-v1",
  "candidate": {"kind": "mlp", "hidden": 32, "layers": 2},
  "epochs": 40,
  "batch": 4096,
  "learning_rate": 0.002,
  "weight_decay": 0.01,
  "seeds": [11],
  "device": "auto",
  "max_seconds": 1800,
  "corpus_dir": "artifacts/dino/corpus",
  "calibration": "artifacts/dino/calibration/calibration.json"
}
```

- [ ] RED: write `tests/test_dino_model.py` and `tests/test_dino_train.py` asserting MLP shapes and parameter counts, determinism, a buildable transformer comparison, `set_valued_loss` ≈ 0 on a two-way split and `log 2` on disallowed mass (mirroring the Nim `test_set_valued_loss_*` cases), that training lowers loss, a checkpoint round-trip carrying normalization + `feature_version` + `engine_revision` + `source_revision`, and that the loader raises on schema/hash mismatch.

```python
# tests/test_dino_model.py
import math
import pytest
import torch
from rlvr_curriculum.dino_contract import FEATURE_DIM, N_ACTIONS
from rlvr_curriculum.dino_model import (
    CANDIDATES, ActionMLP, ActionTransformer, build_model, parameter_count,
    set_valued_loss,
)

def test_mlp_shapes_and_parameter_counts() -> None:
    expected = {32: 2083, 64: 10371, 96: 21699}
    for spec in CANDIDATES:
        model = build_model(spec)
        assert isinstance(model, ActionMLP)
        assert model(torch.zeros(4, FEATURE_DIM)).shape == (4, N_ACTIONS)
        assert parameter_count(model) == expected[spec.hidden]

def test_transformer_candidate_builds_and_runs() -> None:
    model = ActionTransformer(FEATURE_DIM, 32, 2)
    assert model(torch.zeros(4, FEATURE_DIM)).shape == (4, N_ACTIONS)

def test_set_valued_loss_matches_split_and_disallowed_mass() -> None:
    allowed = torch.zeros(1, N_ACTIONS, dtype=torch.bool)
    allowed[0, 1] = allowed[0, 2] = True          # jump and duck are acceptable
    split = torch.full((1, N_ACTIONS), -50.0)
    split[0, 1] = split[0, 2] = 0.0
    assert set_valued_loss(split, allowed).item() == pytest.approx(0.0, abs=1e-3)
    half = torch.full((1, N_ACTIONS), -50.0)
    half[0, 1] = half[0, 0] = 0.0                 # half the mass on a disallowed action
    assert set_valued_loss(half, allowed).item() == pytest.approx(math.log(2.0), abs=1e-3)
```

```python
# tests/test_dino_train.py
from pathlib import Path
import pytest
import torch
from rlvr_curriculum.dino_contract import FEATURE_DIM, IntegrityError, Normalization
from rlvr_curriculum.dino_dataset import generate_corpus, load_dataset_spec
from rlvr_curriculum.dino_model import ModelSpec
from rlvr_curriculum.dino_train import (
    TrainingSpec, load_dino_model, train_candidate,
)

ROOT = Path(__file__).resolve().parents[1]

def _tiny_spec(tmp_path: Path) -> TrainingSpec:
    return TrainingSpec(
        version="dino-train-v1", candidate=ModelSpec("mlp", 32, 2), epochs=2, batch=64,
        learning_rate=0.01, weight_decay=0.0, seeds=(11,), device="cpu", max_seconds=600,
        corpus_dir=tmp_path, calibration=tmp_path / "calibration.json",
    )

def test_training_lowers_loss_and_round_trips(tmp_path: Path) -> None:
    corpus = generate_corpus(load_dataset_spec(ROOT / "configs" / "dino-corpus.json"))
    report = train_candidate(_tiny_spec(tmp_path), corpus, tmp_path)
    assert report["curve"][-1]["loss"] < report["curve"][0]["loss"]
    model, extra = load_dino_model(tmp_path / "mlp32-seed11.pt", torch.device("cpu"))
    assert extra["feature_version"] and len(extra["normalization"]["mean"]) == FEATURE_DIM
    assert extra["engine_revision"] and extra["source_revision"]

def test_loader_rejects_schema_mismatch(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"schema": 99, "extra": {}}, bad)
    with pytest.raises(IntegrityError):
        load_dino_model(bad, torch.device("cpu"))
```

Run `uv run pytest tests/test_dino_model.py tests/test_dino_train.py -v` — expect import failure.

- [ ] GREEN: implement `dino_model.py` and `dino_train.py`, add the four configs, and wire `dino-calibrate` / `dino-train` into `cli.py`.

```python
def set_valued_loss(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    if allowed.ndim != 2 or logits.shape != allowed.shape:
        raise ValueError("allowed must be a [B, N] boolean mask matching logits")
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("every row needs at least one acceptable action")
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return -torch.logsumexp(log_probs.masked_fill(~allowed, float("-inf")), dim=-1).mean()

def _fallback_batch(spec: TrainingSpec, corpus: Corpus, device: torch.device) -> dict[str, Any]:
    for batch in (4096, 2048, 1024):
        seconds = _measure_step_seconds(spec, corpus, device, batch)
        forecast = seconds * _steps_for(spec, corpus, batch)
        if forecast <= 1800 and _peak_vram(device) <= 5_400_000_000:
            return {"batch": batch, "forecast_seconds": forecast, "refused": False, "reason": ""}
    return {"batch": 1024, "forecast_seconds": forecast, "refused": True,
            "reason": "forecast exceeds 1800 s at every batch; reduce hidden"}
```

```python
# cli.py additions (inside main)
if args.command == "dino-calibrate":
    if not args.config or not args.corpus:
        parser.error("dino-calibrate requires --config and --corpus")
    from .dino_dataset import generate_corpus, load_dataset_spec
    from .dino_train import calibrate, load_training_spec, write_calibration
    payload = calibrate(load_training_spec(args.config), generate_corpus(load_dataset_spec(args.corpus)))
    write_calibration(payload, args.output)
    print(f"batch={payload['batch']} forecast={payload['forecast_seconds']:.1f}s refused={payload['refused']}")
    return
if args.command == "dino-train":
    if not args.config or not args.corpus:
        parser.error("dino-train requires --config and --corpus")
    from .dino_dataset import generate_corpus, load_dataset_spec
    from .dino_train import check_calibration, load_training_spec, train_candidate
    spec = load_training_spec(args.config)
    check_calibration(spec)                       # refuses before any long training
    print(train_candidate(spec, generate_corpus(load_dataset_spec(args.corpus)), args.output)["checkpoint"])
    return
```

Run the tests again — expect all green.

- [ ] VERIFY (calibration gate, mandatory before any long training): run `uv run tiny-rlvr dino-calibrate --config configs/dino-train-mlp32.json --corpus configs/dino-corpus.json --output artifacts/dino/calibration`; measure corpus throughput, SFT step time per spec at batch {4096, 2048, 1024}, peak VRAM via `torch.cuda.max_memory_allocated()`, and single-state latency p50/p95/p99 on CPU and CUDA. Apply the fallback ladder batch 4096 → 2048 → 1024, then reduce `hidden` — **never weaken a gate**. Freeze `artifacts/dino/calibration/calibration.json` (batch, epochs, forecast, VRAM, latency, fingerprint) and refuse the long run if forecast > **1800 s** or peak VRAM > **5.4 GB**. Then run `uv run tiny-rlvr dino-train --config configs/dino-train-mlp32.json --corpus configs/dino-corpus.json --output artifacts/dino/candidates/mlp32`.
- [ ] Commit: `feat: add dino policy candidates, distillation, and compute calibration`.

**Gate:** Model/train tests green; p99 `< 16.67 ms`; candidate extras carry feature version, normalization, engine revision, source revision, and corpus digest; the frozen calibration artifact is inspected before the long run.

---

### M6 — Certification: Frozen Domain, Held-Out, Soak, and Latency

**Files:** Create `rlvr_curriculum/dino_certify.py`, `tests/test_dino_certify.py`, `configs/dino-preregistration.json`. Modify `rlvr_curriculum/cli.py` (add `dino-certify`).

**Interfaces:** Certification over the frozen domain plus held-out, soak, and latency reports.

```python
# rlvr_curriculum/dino_certify.py
from __future__ import annotations
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .dino_contract import (ACTION_ORDER, FEATURE_DIM, FEATURE_VERSION, Action,
                            InvalidActionError, NonFiniteError, Normalization,
                            action_from_index, extract_features)
from .dino_domain import (DEFAULT_DOMAIN_SPEC, CertCase, DomainSpec, enumerate_domain,
                          domain_manifest)
from .dino_engine import DINO_ENGINE_REVISION, DinoEngine
from .dino_teacher import Teacher
from .dino_train import load_dino_model, predict_one

CLAIM: str = (
    "This unassisted checkpoint produced zero unsafe choices over the complete "
    "frozen certification domain and zero crashes across the preregistered "
    "held-out and soak suites."
)

def load_preregistration(path: Path) -> dict[str, Any]:
    """Read ``configs/dino-preregistration.json``; raise ``ValueError`` unless
    ``version == "dino-prereg-v1"``, ``frozen_before_run is True``, and the keys
    ``domain_manifest``, ``heldout_seeds``, ``soak_hours``, ``soak_seed``,
    ``latency_budget_ms`` are present."""

def certify(checkpoint: Path, preregistration: Path, output_dir: Path) -> dict[str, Any]:
    """Exhaustively check ``enumerate_domain(DEFAULT_DOMAIN_SPEC)`` against the frozen
    artifact. Refuse to overwrite a completed ``certification.json``
    (``FileExistsError``). Resume from ``certification-progress.jsonl`` without
    counting incomplete cases as passed. Evaluate the teacher only AFTER the greedy
    action is selected. Write ``certification.json``, ``certification.md``, and
    ``certification-failures.csv`` via ``io``."""

def certify_heldout(checkpoint: Path, preregistration: Path) -> dict[str, Any]:
    """Run the preregistered held-out seed suite (random / min-gap / max-speed /
    bird-heavy / mixed) end-to-end; return the held-out block described below."""

def soak(checkpoint: Path, hours: float, seed: int) -> dict[str, Any]:
    """Deterministic headless soak for ``round(hours * 3600 * FRAME_HZ)`` frames;
    return ``{"kind": "soak", "seed", "hours", "frames", "crashes", "unsafe"}``."""

def latency_report(checkpoint: Path, device: torch.device) -> dict[str, float]:
    """N=2000 single-state forward passes; return
    ``{"n": float, "p50": .., "p95": .., "p99": ..}`` in milliseconds."""

def write_report(report: Mapping[str, object], output_dir: Path) -> None:
    """Write ``certification.json`` (atomic_write_json) plus ``certification.md`` and
    ``certification-failures.csv`` (same prefix as the report's ``kind``)."""
```

Report fields: domain size; unsafe/malformed/nondeterminism/non-finite/feature-mismatch counts; held-out seeds and frames; soak frames; crashes (held-out + soak); runtime interventions (`0 by construction`); architecture; parameters; deployed bytes; engine revision; source revision; domain and corpus manifest hashes; checkpoint SHA-256; device; timestamp; `n_failing` and every counterexample.

```python
# top-level shape of the report returned by certify(checkpoint, preregistration, output_dir)
{
  "schema": 1, "kind": "certification", "status": "CERTIFIED ON FROZEN DOMAIN" | "NOT CERTIFIED",
  "checkpoint": "<name>", "checkpoint_sha256": hex, "device": str, "timestamp": iso8601,
  "engine_revision": DINO_ENGINE_REVISION, "source_revision": str,
  "feature_version": FEATURE_VERSION,
  "architecture": {"kind": str, "hidden": int, "layers": int},
  "params": int, "deployed_bytes": int,
  "corpus_digest": hex, "domain_version": str, "domain_manifest_sha256": hex,
  "domain_size": int, "synthetic_boundary": int, "states_checked": int, "resumed_from": int,
  "unsafe": int, "malformed": int, "nondeterminism": int, "non_finite": int,
  "feature_mismatch": int, "runtime_interventions": 0, "crashes": int,
  "heldout": {"seeds": [9001, 9002, 9003], "frames": int, "crashes": int},
  "soak": {"frames": int, "crashes": int, "hours": float, "seed": int},
  "latency_ms": {"p50": float, "p95": float, "p99": float},
  "n_failing": int, "failures": [counterexample],   # one entry per counterexample
  "claim": CLAIM,
}
# counterexample:
{"case_id": str, "family": str, "provenance": str, "action": str,
 "probabilities": [float, float, float], "acceptable": ["jump"], "min_margin": float,
 "reason": "unsafe" | "malformed" | "nondeterminism" | "non_finite" | "feature_mismatch",
 "state": EngineState.as_dict()}
```

`configs/dino-preregistration.json`:

```json
{
  "version": "dino-prereg-v1",
  "frozen_before_run": true,
  "domain_manifest": "<sha256 of artifacts/dino/domain/domain-manifest.json>",
  "heldout_seeds": [9001, 9002, 9003, 9004, 9005],
  "soak_hours": 4.0,
  "soak_seed": 20260922,
  "latency_budget_ms": {"p95": 8.0, "p99": 16.67}
}
```

- [ ] RED: write `tests/test_dino_certify.py` asserting exhaustive coverage over `enumerate_domain`, that the teacher is evaluated **after** selection (spy) and the live move equals the artifact argmax, that a bad hash/schema raises (no partial pass), that a NaN logit produces a violation and fails, that an interrupted run resumes without counting incomplete cases as passed, determinism, held-out disjointness, and a deterministic soak.

```python
# tests/test_dino_certify.py
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from rlvr_curriculum.dino_contract import IntegrityError
from rlvr_curriculum.dino_certify import certify, load_preregistration, soak
from rlvr_curriculum.dino_dataset import generate_corpus, load_dataset_spec
from rlvr_curriculum.dino_domain import DEFAULT_DOMAIN_SPEC, enumerate_domain
from rlvr_curriculum.dino_model import ModelSpec
from rlvr_curriculum.dino_teacher import TeacherVerdict
from rlvr_curriculum.dino_train import TrainingSpec, train_candidate

ROOT = Path(__file__).resolve().parents[1]

def _write_prereg(tmp_path: Path) -> Path:
    path = tmp_path / "dino-preregistration.json"
    path.write_text(json.dumps({
        "version": "dino-prereg-v1", "frozen_before_run": True,
        "domain_manifest": "0" * 64, "heldout_seeds": [9001],
        "soak_hours": 0.001, "soak_seed": 1,
        "latency_budget_ms": {"p95": 8.0, "p99": 16.67},
    }), encoding="utf-8")
    return path

def _tiny_checkpoint(tmp_path: Path) -> Path:
    """One-seed, one-epoch MLP32 checkpoint written by train_candidate."""
    corpus = generate_corpus(load_dataset_spec(ROOT / "configs" / "dino-corpus.json"))
    spec = TrainingSpec("dino-train-v1", ModelSpec("mlp", 32, 2), 1, 32, 0.01, 0.0,
                        (11,), "cpu", 600, tmp_path, tmp_path / "calibration.json")
    return tmp_path / train_candidate(spec, corpus, tmp_path)["checkpoint"]

def _verdict() -> TeacherVerdict:
    """A permissive verdict used by the ordering spy (every action acceptable)."""
    return TeacherVerdict(
        acceptable=("run", "jump", "duck"),
        survival={"run": True, "jump": True, "duck": True},
        min_margin={"run": 10.0, "jump": 10.0, "duck": 10.0},
        clearance_frame={"run": 0, "jump": 0, "duck": 0},
        rejection_reason={"run": "", "jump": "", "duck": ""},
        outside_horizon=False, horizon_frames=60,
    )

def test_certification_covers_the_frozen_domain(tmp_path: Path) -> None:
    report = certify(_tiny_checkpoint(tmp_path), _write_prereg(tmp_path), tmp_path / "cert")
    assert report["states_checked"] == len(enumerate_domain(DEFAULT_DOMAIN_SPEC))
    assert report["runtime_interventions"] == 0
    assert report["n_failing"] == report["unsafe"] + report["malformed"] + report["non_finite"]

def test_teacher_is_evaluated_after_selection(tmp_path: Path) -> None:
    order: list[str] = []
    with patch("rlvr_curriculum.dino_certify.predict_one", side_effect=lambda *a, **k: order.append("select") or (0, (0.6, 0.3, 0.1))), \
         patch("rlvr_curriculum.dino_certify.Teacher.evaluate", side_effect=lambda *a, **k: order.append("teach") or _verdict()):
        certify(_tiny_checkpoint(tmp_path), _write_prereg(tmp_path), tmp_path / "cert2")
    assert order[0] == "select"

def test_bad_schema_raises_without_partial_pass(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"schema": 7, "extra": {}}, bad)
    with pytest.raises(IntegrityError):
        certify(bad, _write_prereg(tmp_path), tmp_path / "cert3")

def test_soak_is_deterministic(tmp_path: Path) -> None:
    first = soak(_tiny_checkpoint(tmp_path), hours=0.001, seed=1)
    second = soak(_tiny_checkpoint(tmp_path), hours=0.001, seed=1)
    assert first == second and first["frames"] > 0
```

Run `uv run pytest tests/test_dino_certify.py -v` — expect import failure.

- [ ] GREEN: implement `dino_certify.py` mirroring the Nim `_certify_action` accounting; freeze `configs/dino-preregistration.json` with `"frozen_before_run": true` before final selection and refuse overwrite.

```python
def certify(checkpoint: Path, preregistration: Path, output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "certification.json"
    if report_path.exists():
        raise FileExistsError(f"certification already complete: {report_path}")
    prereg = load_preregistration(preregistration)
    model, extra = load_dino_model(checkpoint, _device())
    normalization = Normalization.from_dict(extra["normalization"])
    teacher = Teacher(DinoEngine(), DEFAULT_DOMAIN_SPEC.lookahead_frames)
    cases = enumerate_domain(DEFAULT_DOMAIN_SPEC)
    done = _read_progress(output_dir / "certification-progress.jsonl")
    failures: list[dict[str, Any]] = []
    for case in cases:
        if case.case_id in done:
            continue
        features = normalization.encode(extract_features(case.state))
        index, probabilities = predict_one(model, features, _device())   # selection first
        action = action_from_index(index)
        verdict = teacher.evaluate(case.state)                           # teacher afterwards
        if action not in verdict.acceptable:
            failures.append(_counterexample(case, action, probabilities, verdict, "unsafe"))
        _append_progress(output_dir / "certification-progress.jsonl", case.case_id)
    report = _build_report(checkpoint, extra, cases, failures, prereg, output_dir)
    write_report(report, output_dir)
    return report
```

Run the test again — expect all green.

- [ ] Verify: `uv run tiny-rlvr dino-certify --checkpoint <artifact> --output artifacts/dino/certification`; confirm `n_failing == 0`, interventions `0`, and resume behavior on a deliberately interrupted run.
- [ ] Commit: `feat: add frozen-domain dino certification, held-out, and soak`.

**Gate:** Certification tests green; report exposes every required field and counterexample; preregistration frozen before final selection.

---

### M7 — DAgger, Lexicographic Selection, and INT8

**Files:** Create `rlvr_curriculum/dino_dagger.py`, `tests/test_dino_dagger.py`, `tests/test_dino_selection.py`. Modify `rlvr_curriculum/cli.py` (add `dino-dagger`, `dino-select`).

**Interfaces:** DAgger iterations, priority weighting, lexicographic selection, and quantization parity.

```python
# rlvr_curriculum/dino_dagger.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from .dino_contract import ACTION_ORDER, Action
from .dino_dataset import Corpus, Example

@dataclass(frozen=True, slots=True)
class DaggerConfig:
    iterations: int
    rollout_frames: int
    max_new_examples: int
    priority: dict[str, float]      # keys: "crash", "unsafe", "low_margin", "disagreement"

DEFAULT_DAGGER_PRIORITY: dict[str, float] = {
    "crash": 4.0, "unsafe": 2.0, "low_margin": 1.0, "disagreement": 1.5,
}

@dataclass(frozen=True, slots=True)
class Visit:
    example: Example
    crashes: int
    unsafe: int
    min_margin: float
    disagreement: bool

    def key(self) -> str:
        """Stable example identity: sha256 over rounded features + mask + seed + family."""

    def as_dict(self) -> dict[str, object]:
        """``{"features": [28 floats], "acceptable_mask": [3 bools], "course_seed": int,
        "family": str, "boundary": bool, "crashes": int, "unsafe": int,
        "min_margin": float, "disagreement": bool}``."""

def collect_visited_states(candidate: Path, config: DaggerConfig, seed: int) -> list[Visit]:
    """Run the unassisted candidate for ``config.rollout_frames`` frames. The action
    loop never calls the teacher; the teacher labels each visited state only
    afterwards, supplying ``unsafe``/``min_margin``/``disagreement``."""

def priority_weights(visits: Sequence[Visit], priority: Mapping[str, float]) -> dict[str, float]:
    """``{visit.key(): weight}`` with
    ``weight = 1.0 + priority['crash']*crashes + priority['unsafe']*unsafe
    + priority['low_margin']*max(0.0, 12.0 - min_margin)
    + priority['disagreement']*int(disagreement)``."""

def dagger_iteration(candidate: Path, corpus: Corpus, config: DaggerConfig,
                     output_dir: Path, iteration: int) -> dict[str, Any]:
    """One DAgger round: collect -> weight -> keep the top ``max_new_examples`` ->
    retrain. Return ``{"iteration", "visited", "kept", "crashes", "unsafe",
    "digest", "aggregated_digest", "checkpoint"}``."""

def select_candidate(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Lexicographic minimum over
    ``(not passed_certification, not passed_heldout_soak, deployed_bytes, p95_ms)``;
    raise ``ValueError`` on an empty sequence."""

def quantize_int8(checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    """Dynamic INT8 PTQ of the MLP; return ``{"checkpoint", "sha256", "size_bytes",
    "baseline_sha256", "changed", "n_changed_actions"}`` where ``changed`` is True when
    any certification action differs from the FP32 baseline."""

def quantization_aware_train(spec: TrainingSpec, corpus: Corpus, output_dir: Path) -> Path:
    """QAT fallback required when the ``changed`` flag returned by ``quantize_int8`` is
    True, i.e. INT8 post-training quantization altered a certification action."""

def recertify_if_changed(quantized: Path, baseline: Mapping[str, Any],
                         preregistration: Path, output_dir: Path) -> dict[str, Any]:
    """Run ``certify`` on the quantized artifact; raise ``IntegrityError`` when the
    quantized SHA-256 is not the certified SHA-256."""
```

- [ ] RED: write `tests/test_dino_dagger.py` asserting the rollout is unassisted (no teacher in the loop; spy) and that priorities increase with crashes, unsafe choices, low margins, and teacher/model disagreement, and that the aggregated digest is stable and disjoint from held-out. Write `tests/test_dino_selection.py` asserting a passing smaller model beats a passing larger one, that a failing small never beats a passing larger, that INT8 PTQ parity sets `changed=True` when any certification action changes (requiring QAT + recertify), and that the shipped hash equals the certified hash.

```python
# tests/test_dino_selection.py
import pytest
from rlvr_curriculum.dino_dagger import select_candidate

def _report(name: str, certified: bool, heldout: bool, bytes_: int, p95: float) -> dict:
    return {"checkpoint": name, "passed_certification": certified,
            "passed_heldout_soak": heldout, "deployed_bytes": bytes_, "latency_ms": {"p95": p95}}

def test_lexicographic_order_prefers_smaller_passing_model() -> None:
    chosen = select_candidate([
        _report("big", True, True, 900_000, 3.0),
        _report("small", True, True, 120_000, 4.0),
    ])
    assert chosen["checkpoint"] == "small"

def test_a_failing_model_never_beats_a_passing_one() -> None:
    chosen = select_candidate([
        _report("tiny", False, False, 10_000, 1.0),
        _report("big", True, True, 900_000, 3.0),
    ])
    assert chosen["checkpoint"] == "big"
```

Run `uv run pytest tests/test_dino_dagger.py tests/test_dino_selection.py -v` — expect import failure.

- [ ] GREEN: implement `dino_dagger.py`; selection is lexicographic — pass certification, pass held-out/soak, minimize deployed bytes, minimize p95 latency — never by accuracy.

```python
def select_candidate(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not reports:
        raise ValueError("no candidate reports to select from")
    def order(report: Mapping[str, Any]) -> tuple[bool, bool, int, float]:
        return (
            not bool(report["passed_certification"]),
            not bool(report["passed_heldout_soak"]),
            int(report["deployed_bytes"]),
            float(report["latency_ms"]["p95"]),
        )
    return min(reports, key=order)
```

Run the tests again — expect all green.

- [ ] Verify: run `dino-dagger` then `dino-select`; inspect the selected report's ordering evidence; if PTQ changed any action, run QAT and recertify before selecting.
- [ ] Commit: `feat: add dino dagger, lexicographic selection, and int8 quantization`.

**Gate:** DAgger/selection tests green; chosen candidate wins on the lexicographic order; certified hash equals shipped hash.

---

### M8 — FastAPI Integration and Arena

**Files:** Create `rlvr_curriculum/dino_arena.py`, `rlvr_curriculum/dino_web.py`, `tests/test_dino_arena.py`, `tests/test_dino_web.py`. Modify `rlvr_curriculum/cli.py` (add `dino-arena`, `dino-web`, `dino-play`).

**Interfaces:** `dino_arena.py` owns the frozen suites and the load-once runtime; `dino_web.py` owns the FastAPI app. Neither imports `dino_teacher`.

```python
# rlvr_curriculum/dino_arena.py
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import nn

from .dino_contract import (FEATURE_DIM, NonFiniteError, Normalization, action_from_index,
                            extract_features)
from .dino_engine import DinoEngine, EngineState
from .dino_train import load_dino_model, predict_actions

@dataclass(slots=True)
class Runtime:
    checkpoint: Path
    device: torch.device
    model: nn.Module
    normalization: Normalization
    feature_version: str
    engine_revision: str
    source_revision: str
    corpus_digest: str
    params: int
    checkpoint_bytes: int
    sha256: str

    def describe(self) -> dict[str, Any]:
        """``{"name": "DinoActionPolicy", "architecture": {"kind": str, "hidden": int,
        "layers": int}, "params": int, "deployed_bytes": int, "checkpoint_sha256": hex,
        "feature_version": str, "engine_revision": str, "source_revision": str,
        "corpus_digest": hex, "device": str}``."""

@dataclass(frozen=True, slots=True)
class Suite:
    name: str
    kind: Literal["random", "min_gap", "max_speed", "bird_heavy", "mixed"]
    seeds: tuple[int, ...]
    frames: int

SUITES: tuple[Suite, ...] = (
    Suite("random-60s", "random", (9001, 9002, 9003), 3600),
    Suite("min-gap-60s", "min_gap", (9101, 9102), 3600),
    Suite("max-speed-60s", "max_speed", (9201, 9202), 3600),
    Suite("bird-heavy-60s", "bird_heavy", (9301, 9302), 3600),
    Suite("mixed-120s", "mixed", (9401, 9402, 9403), 7200),
)

def model_action(runtime: Runtime, state: EngineState) -> tuple[int, tuple[float, float, float], float]:
    """The single inference entry point every live and arena path calls (tests
    monkeypatch this). Returns ``(index, probabilities, latency_ms)``; raise
    ``NonFiniteError`` on a non-finite probability."""

def load_runtime(checkpoint: Path, device: torch.device | None = None) -> Runtime:
    """Load once; refuse on hash/schema mismatch via ``load_dino_model``. Compute the
    SHA-256 of the checkpoint bytes itself and reuse it in ``describe``."""

def run_suite(runtime: Runtime, suite: Suite) -> dict[str, Any]:
    """Run every seed in ``suite``; each run is a fresh ``DinoEngine`` stepped headless.
    Teacher-free: ``unsafe_choices`` counts decision frames on which the engine
    registered a collision (certified unsafe-action verdicts are M6's job)."""

def run_selected_seeds(runtime: Runtime, seeds: Sequence[int], frames: int = 3600) -> dict[str, Any]:
    """Ad-hoc suite over explicit seeds; same report shape as ``run_suite``."""
```

`run_suite` / `run_selected_seeds` report shape:

```python
{
  "suite": str, "kind": str, "seeds": [9001, 9002, 9003],
  "scenarios": int, "frames": int, "obstacles": int, "crashes": int,
  "unsafe_choices": int, "latency_ms": {"p50": float, "p95": float, "p99": float},
  "per_seed": [{"seed": int, "frames": int, "obstacles": int, "crashes": int,
                "unsafe_choices": int, "distance": float}],
}
```

```python
# rlvr_curriculum/dino_web.py
from __future__ import annotations
import json, threading, time, uuid
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .dino_arena import (SUITES, Runtime, load_runtime, model_action, run_selected_seeds,
                         run_suite)
from .dino_contract import InvalidActionError, NonFiniteError, action_from_index, extract_features
from .dino_engine import DinoEngine

def write_violation(output_dir: Path, run_id: str, payload: dict[str, Any]) -> Path:
    """Persist ``violations/{run_id}.json`` with ``{run_id, frame, reason, action,
    probabilities, state}`` via ``atomic_write_json``; return the path."""

class PlayStartRequest(BaseModel):
    model_config = {"extra": "forbid"}
    seed: int = 0
    mode: str = "random"          # "random" | "min_gap" | "max_speed" | "bird_heavy"

class PlayStepRequest(BaseModel):
    model_config = {"extra": "forbid"}
    run_id: str

class ArenaRequest(BaseModel):
    model_config = {"extra": "forbid"}
    suite: str
    checkpoint: str | None = None

class CertifyRequest(BaseModel):
    model_config = {"extra": "forbid"}
    output: str | None = None

def create_app(runtime: Runtime, *, static_dir: Path | None = None,
               snapshot_paths: list[Path] | None = None) -> FastAPI:
    """Reuse the ``jobs_lock`` + worker-thread pattern from ``nim_web.create_app``."""

def serve(checkpoint: Path, host: str = "127.0.0.1", port: int = 8000,
          web_dir: Path | None = None, open_browser: bool = False) -> None:
    """``load_runtime`` then ``uvicorn.run``; when ``open_browser`` open ``/``."""
```

Endpoints: `GET /api/model`; `POST /api/play/start`, `POST /api/play/step`; `GET /api/arena/suites`, `POST /api/arena`, `GET /api/arena/{job_id}`; `GET /api/certification`, `POST /api/certify`; `GET /api/replay/{run_id}`, `GET /api/replay/{run_id}/frame/{index}`.

```python
@app.post("/api/play/step")
def play_step(request: PlayStepRequest) -> dict[str, Any]:
    run = _runs[request.run_id]                        # 404 when unknown
    if run["stopped"]:
        return {"run_id": request.run_id, "stopped": True, "reason": run["reason"]}
    try:
        index, probabilities, latency_ms = model_action(runtime, run["engine"].snapshot())
    except NonFiniteError as exc:
        path = write_violation(runtime.checkpoint.parent, request.run_id,
                               {"reason": "non_finite", "detail": str(exc), "frame": run["engine"].snapshot().frame})
        run.update(stopped=True, reason="non_finite", violation=str(path))
        return {"run_id": request.run_id, "stopped": True, "reason": "non_finite"}
    action = action_from_index(index)
    run["engine"].step(action)
    run["frames"].append({"action": action, "probabilities": list(probabilities),
                          "latency_ms": latency_ms, "state": run["engine"].snapshot().as_dict()})
    if run["engine"].is_terminal():
        run.update(stopped=True, reason="collision")
        write_violation(runtime.checkpoint.parent, request.run_id,
                        {"reason": "collision", "action": action, "frame": run["engine"].snapshot().frame,
                         "probabilities": list(probabilities), "state": run["engine"].snapshot().as_dict()})
    return {"run_id": request.run_id, "stopped": run["stopped"], "reason": run["reason"],
            "action": action, "probabilities": list(probabilities), "latency_ms": latency_ms,
            "state": run["engine"].snapshot().as_dict()}
```

- [ ] RED: write `tests/test_dino_arena.py` asserting suite determinism, counts, and no shared mutable state; write `tests/test_dino_web.py` asserting `/api/play/step` equals the frozen argmax, a **mutation test** forcing an invalid/unsafe action stops the run and writes a violation artifact without invoking the teacher (spy), a bad checkpoint makes `load_runtime` raise, a NaN stops and persists, a job moves queued → done, and interventions report `0`.

```python
# tests/test_dino_web.py
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from rlvr_curriculum.dino_contract import NonFiniteError
from rlvr_curriculum.dino_train import predict_one
from rlvr_curriculum.dino_web import create_app, load_runtime

def test_teacher_is_never_invoked(tmp_path: Path, runtime: object) -> None:
    with patch("rlvr_curriculum.dino_teacher.Teacher.evaluate", side_effect=AssertionError("teacher called")):
        client = TestClient(create_app(runtime, snapshot_paths=[]))
        started = client.post("/api/play/start", json={"seed": 3, "mode": "random"}).json()
        for _ in range(20):
            response = client.post("/api/play/step", json={"run_id": started["run_id"]}).json()
            if response["stopped"]:
                break

def test_nan_stops_and_persists_a_violation(tmp_path: Path, runtime: object) -> None:
    with patch("rlvr_curriculum.dino_web.model_action",
               side_effect=NonFiniteError("NaN probability")):
        client = TestClient(create_app(runtime, snapshot_paths=[]))
        started = client.post("/api/play/start", json={"seed": 3, "mode": "random"}).json()
        response = client.post("/api/play/step", json={"run_id": started["run_id"]}).json()
        assert response["stopped"] is True and response["reason"] == "non_finite"

def test_bad_checkpoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        load_runtime(tmp_path / "missing.pt")
```

Run `uv run pytest tests/test_dino_arena.py tests/test_dino_web.py -v` — expect import failure.

- [ ] GREEN: implement `dino_arena.py` and `dino_web.py`, reusing the `jobs_lock` + worker-thread pattern from `nim_web.create_app`; wire `dino-arena`, `dino-web`, `dino-play` into `cli.py`.

```python
def run_suite(runtime: Runtime, suite: Suite) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    latencies: list[float] = []
    for seed in suite.seeds:
        engine = DinoEngine()
        engine.reset(seed)
        crashes = unsafe = obstacles = 0
        for _ in range(suite.frames):
            before = engine.snapshot()
            obstacles += int(before.obstacle is not None)
            index, _probabilities, latency_ms = model_action(runtime, before)
            latencies.append(latency_ms)
            engine.step(action_from_index(index))
            if engine.is_terminal():
                crashes += 1
                unsafe += 1
                break
        per_seed.append({"seed": seed, "frames": before.frame, "obstacles": obstacles,
                         "crashes": crashes, "unsafe_choices": unsafe,
                         "distance": engine.snapshot().distance})
    return {
        "suite": suite.name, "kind": suite.kind, "seeds": list(suite.seeds),
        "scenarios": len(suite.seeds), "frames": sum(row["frames"] for row in per_seed),
        "obstacles": sum(row["obstacles"] for row in per_seed),
        "crashes": sum(row["crashes"] for row in per_seed),
        "unsafe_choices": sum(row["unsafe_choices"] for row in per_seed),
        "latency_ms": _percentiles(latencies), "per_seed": per_seed,
    }
```

Run the tests again — expect all green.

- [ ] Verify: `uv run tiny-rlvr dino-web --checkpoint <artifact>` and exercise `/api/play/step` and `/api/arena` with the TestClient; confirm refusals on hash/schema mismatch.
- [ ] Commit: `feat: add dino arena and fastapi live/arena/proof/replay api`.

**Gate:** Arena/API tests green; live move equals certified inference; mutation test proves no substitution and no teacher fallback.

---

### M9 — React Play / Arena / Proof / Replay Adaptation

**Files:** Create `web/src/pages/{Play,Arena,Proof,Replay}.tsx`, `web/src/components/{DinoLane,PolicyIndicator}.tsx`, `web/src/lib/dino.ts`, `web/src/test/{play,arena,proof,replay}.test.tsx`. Modify `web/src/app/App.tsx` (brand `Tiny Dino`; add tabs + Replay), `web/src/lib/api.ts` (Dino endpoints; keep visual tokens), `web/src/test/app.test.tsx`. Delete Nim-specific `GameBoard/NimPile/Scoreboard/RulesExplainer.tsx` and the Nim Play assertions (the Nim **Python** product stays intact). Reuse `tokens.css`, `Button/SegmentedControl/Metric/ProofMetric/TurnStatus`, and `lib/{formatting,motion}.ts`.

**Interfaces:** `web/src/lib/dino.ts` defines the wire types and the four fetchers; pages consume them.

```ts
// web/src/lib/dino.ts
export type DinoAction = "run" | "jump" | "duck";

export interface DinoObstacle {
  kind: "small" | "large" | "bird";
  x: number; y: number; width: number; height: number;
}

export interface DinoFrameTelemetry {
  frame: number; distance: number; speed: number; score: number;
  obstacleCount: number; action: DinoAction; probability: number; latencyMs: number;
  acceptable: DinoAction[]; margins: Record<DinoAction, number>;
  dinoY: number; obstacles: DinoObstacle[];
}

export interface PlayStepResponse {
  runId: string; stopped: boolean; reason: string;
  action?: DinoAction; probabilities?: [number, number, number]; latencyMs?: number;
  state?: DinoFrameTelemetry;
}

export interface ArenaSuite { name: string; kind: string; seeds: number[]; frames: number; }

export interface ArenaReport {
  suite: string; scenarios: number; frames: number; obstacles: number;
  crashes: number; unsafeChoices: number; latencyMs: { p50: number; p95: number; p99: number };
}

export interface CertificationSnapshot {
  status: string; domainSize: number; statesChecked: number; crashes: number;
  unsafe: number; runtimeInterventions: number; architecture: { kind: string; hidden: number; layers: number };
  params: number; deployedBytes: number; engineRevision: string; checkpointSha256: string;
  domainManifestSha256: string; corpusDigest: string; device: string; timestamp: string;
  claim: string;
}

export interface ReplayFrame { index: number; action: DinoAction; probabilities: [number, number, number];
  acceptable: DinoAction[]; margins: Record<DinoAction, number>; dinoY: number; obstacles: DinoObstacle[]; }

export interface ReplayRun { runId: string; failureFrame: number | null; frames: ReplayFrame[]; }

export async function startPlay(seed: number, mode: string): Promise<{ runId: string; state: DinoFrameTelemetry }>;
export async function stepPlay(runId: string): Promise<PlayStepResponse>;
export async function fetchArenaSuites(): Promise<ArenaSuite[]>;
export async function runArena(suite: string): Promise<{ status: string; jobId?: string; result?: ArenaReport }>;
export async function fetchCertification(): Promise<CertificationSnapshot>;
export async function fetchReplay(runId: string): Promise<ReplayRun>;
```

- [ ] RED: write the four page tests asserting a Dino lane with a visible `UNASSISTED NEURAL POLICY` indicator; a server-authoritative live game with deterministic seed control; model action, probability, latency, speed, score, and obstacle count; failure stops and exposes an exact replay; reduced motion (inject `matchMedia` as `web/src/test/setup.ts` does); Arena suites reporting scenarios/frames/obstacles/crashes/unsafe choices/latency; Proof stating `Every state in the frozen certification domain checked.` with all provenance fields and a clear separation of frozen snapshot vs live rerun; and Replay reconstructing frames plus probabilities, acceptable set, margins, geometry, and the first failing frame.

```tsx
// web/src/test/play.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import Play from "../pages/Play";
import { startPlay, stepPlay } from "../lib/dino";

vi.mock("../lib/dino");

beforeEach(() => {
  vi.mocked(startPlay).mockResolvedValue({
    runId: "run-1",
    state: { frame: 0, distance: 0, speed: 360, score: 0, obstacleCount: 1,
             action: "run", probability: 0.72, latencyMs: 1.4,
             acceptable: ["jump"], margins: { run: -4, jump: 12, duck: 3 },
             dinoY: 0, obstacles: [{ kind: "small", x: 120, y: 0, width: 17, height: 35 }] },
  });
  vi.mocked(stepPlay).mockResolvedValue({
    runId: "run-1", stopped: false, reason: "",
    action: "jump", probabilities: [0.1, 0.8, 0.1], latencyMs: 1.2, state: undefined,
  });
});

test("shows the unassisted indicator and server telemetry", async () => {
  render(<Play />);
  expect(await screen.findByText("UNASSISTED NEURAL POLICY")).toBeVisible();
  await waitFor(() => expect(screen.getByTestId("dino-lane")).toBeInTheDocument());
  expect(screen.getByText(/obstacles/i)).toHaveTextContent("1");
});
```

Run `cd web; npm run test` — expect failures. Then `cd web; npm run typecheck` — expect type errors for the new modules.

- [ ] GREEN: implement the pages/components/lib and adapt `App.tsx` and `api.ts` to the Dino endpoints.

```tsx
// web/src/components/PolicyIndicator.tsx
export function PolicyIndicator({ action, probability, latencyMs }: {
  action: DinoAction; probability: number; latencyMs: number;
}) {
  return (
    <div role="status" data-testid="policy-indicator">
      <span className="policy-badge">UNASSISTED NEURAL POLICY</span>
      <Metric label="Action" value={action} />
      <Metric label="Probability" value={probability.toFixed(3)} />
      <Metric label="Latency" value={`${latencyMs.toFixed(1)} ms`} />
    </div>
  );
}
```

Run the tests and typecheck again — expect all green.

- [ ] Verify: `cd web; npm run typecheck; npm run lint; npm run test; npm run build` — zero failures.
- [ ] Commit: `feat: adapt web app to dino play, arena, proof, and replay`.

**Gate:** Frontend gates green; visible unassisted indicator; Proof wording and provenance match the spec; reduced-motion path covered.

---

### M10 — Full Verification, Visual QA, Soak, and Evidence

**Files:** Create `tests/test_dino_end_to_end.py`, `reports/dino/evidence.{md,json}`, `web/qa/dino-*.png`. Modify `README.md`.

- [ ] RED: write a CPU end-to-end test (tiny sizes) running corpus → train → calibrate → certify → held-out and asserting the report fields plus `selected hash == certified hash`.

```python
# tests/test_dino_end_to_end.py
from pathlib import Path
from rlvr_curriculum.dino_certify import certify
from rlvr_curriculum.dino_dagger import select_candidate
from rlvr_curriculum.dino_dataset import generate_corpus, load_dataset_spec
from rlvr_curriculum.dino_model import ModelSpec
from rlvr_curriculum.dino_train import TrainingSpec, calibrate, train_candidate, write_calibration

ROOT = Path(__file__).resolve().parents[1]

def test_tiny_pipeline_produces_a_consistent_evidence_chain(tmp_path: Path) -> None:
    corpus = generate_corpus(load_dataset_spec(ROOT / "configs" / "dino-corpus.json"))
    spec = TrainingSpec("dino-train-v1", ModelSpec("mlp", 32, 2), 1, 32, 0.01, 0.0,
                        (11,), "cpu", 600, tmp_path, tmp_path / "calibration.json")
    payload = calibrate(spec, corpus)
    write_calibration(payload, tmp_path)
    report = train_candidate(spec, corpus, tmp_path)
    chosen = select_candidate([{**report, "passed_certification": True,
                                "passed_heldout_soak": True,
                                "deployed_bytes": report["deployed_bytes"],
                                "latency_ms": report["latency_ms"]}])
    certification = certify(tmp_path / report["checkpoint"], ROOT / "configs" / "dino-preregistration.json",
                            tmp_path / "cert")
    assert certification["n_failing"] == 0
    assert chosen["checkpoint"] == report["checkpoint"]
```

Run `uv run pytest tests/test_dino_end_to_end.py -v` — expect import/collection failure.

- [ ] GREEN: implement the end-to-end test and generate the evidence artifacts.

```python
# reports/dino/evidence.json shape (written with atomic_write_json)
{
  "schema": 1, "engine_revision": str, "source_revision": str,
  "domain_manifest_sha256": hex, "corpus_digest": hex, "checkpoint_sha256": hex,
  "params": int, "deployed_bytes": int,
  "latency_ms": {"p50": float, "p95": float, "p99": float},
  "certification": {"domain_size": int, "states_checked": int, "unsafe": int,
                    "crashes": int, "runtime_interventions": 0, "n_failing": int},
  "soak": {"frames": int, "crashes": int},
  "failures": [counterexample], "claim": str,   # one entry per counterexample
}
```

- [ ] VERIFY (full gates): `uv run pytest -q`; `uv run ruff check rlvr_curriculum tests run_experiment.py`; `uv run mypy rlvr_curriculum`; `cd web; npm run typecheck; npm run lint; npm run test; npm run build`; and re-run `uv run pytest tests/test_nim_baseline.py -v` to prove the Nim baseline is intact.
- [ ] VERIFY (soak + timing + visual QA): run multi-hour deterministic headless soaks and real-time browser/API smoke tests; record p50/p95/p99 inference latency, dropped frames, and engine speed; capture visual QA at 1440 and 390 widths, light and dark, with reduced motion.
- [ ] Write `reports/dino/evidence.md` and `evidence.json`: engine revision, scenario and dataset manifests, checkpoint SHA-256, parameter count, artifact size, latency, and every failing counterexample — using **only** the permitted claim from the spec §11. If no candidate passes, report the measured failure and retain the best checkpoint as experimental; never weaken the gate or add a safety shield.
- [ ] Commit: `docs: certify and report the tiny dino policy`.

**Gate:** All repository and frontend gates green; Nim baseline green; constrained claim language; shipped checkpoint hash matches the Proof page.

---

## Verification Summary

| Check | Command |
|---|---|
| Nim baseline intact | `uv run pytest tests/test_nim_baseline.py -v` |
| Engine determinism + fixtures | `uv run pytest tests/test_dino_engine.py -v` |
| Contract / normalization | `uv run pytest tests/test_dino_contract.py -v` |
| Teacher + domain (+ import graph) | `uv run pytest tests/test_dino_teacher.py tests/test_dino_domain.py -v` |
| Dataset determinism + splits | `uv run pytest tests/test_dino_dataset.py -v` |
| Model + loss + calibration | `uv run pytest tests/test_dino_model.py tests/test_dino_train.py -v` |
| Certification / held-out / soak | `uv run pytest tests/test_dino_certify.py -v` |
| DAgger + selection + INT8 | `uv run pytest tests/test_dino_dagger.py tests/test_dino_selection.py -v` |
| Arena + API mutation test | `uv run pytest tests/test_dino_arena.py tests/test_dino_web.py -v` |
| Whole repository | `uv run pytest -q` · `uv run ruff check rlvr_curriculum tests run_experiment.py` · `uv run mypy rlvr_curriculum` |
| Frontend | `cd web; npm run typecheck; npm run lint; npm run test; npm run build` |
| Calibration artifact | inspect `artifacts/dino/calibration/calibration.json` before the long run |

## Risks and Mitigations

- **Reference legality:** if adapting the reference repo is not permitted, differential fixtures fall back to pinned-constant plus invariant tests, recorded in `dino-provenance.json` (`license_verdict: "rejected"`) rather than silently skipped.
- **Teacher semantics:** the teacher holds a candidate control state for the encounter horizon (active obstacle clearance plus `CLEARANCE_TAIL_FRAMES`, capped by `lookahead_frames`); `outside_horizon` flags any case whose next obstacle lies beyond that horizon, and such cases are excluded from the claim until resolved.
- **Frontend vs preservation:** the Nim Python product, artifacts, reports, and hashes stay frozen; only the React app is re-pointed to Dino. Nim Python is never deleted silently.
- **Mypy strict:** the new `dino_*` modules stay fully typed; the untyped-calls override list in `pyproject.toml` is not extended.
- **Certification domain size** drives M6 cost: freeze the domain manifest in M3 and interpolate the expected cost in the M5 calibration forecast.
