"""DAgger collection and evidence-driven candidate selection."""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dino_contract import ACTION_ORDER, Normalization, extract_features
from .dino_dataset import Example
from .dino_domain import DomainSpec, enumerate_domain
from .dino_engine import DinoEngine
from .dino_teacher import Teacher
from .dino_train import load_dino_model, predict_actions


@dataclass(frozen=True, slots=True)
class DaggerConfig:
    seeds: tuple[int, ...]
    max_frames: int = 10_000
    lookahead_frames: int = 90


@dataclass(frozen=True, slots=True)
class ExpertRolloutConfig:
    seeds: tuple[int, ...]
    max_frames: int = 20_000


def collect_teacher_rollouts(config: ExpertRolloutConfig) -> list[Example]:
    """Create deterministic offline expert trajectories for initial distillation."""
    examples: list[Example] = []
    for seed in config.seeds:
        engine = DinoEngine()
        engine.reset(seed)
        for _ in range(config.max_frames):
            state = engine.snapshot()
            preferred = Teacher._continuation(state)
            mask = tuple(action == preferred for action in ACTION_ORDER)
            examples.append(
                Example(
                    extract_features(state),
                    (mask[0], mask[1], mask[2]),
                    seed,
                    "expert-rollout",
                    "train",
                    False,
                )
            )
            engine.step(preferred)
            if engine.is_terminal():
                raise RuntimeError(f"teacher crashed on seed {seed} at frame {engine.snapshot().frame}")
    return examples

def collect_visited_states(checkpoint: Path, config: DaggerConfig, device: torch.device | None = None) -> list[Example]:
    chosen = device or torch.device("cpu")
    model, metadata = load_dino_model(checkpoint, chosen)
    norm = Normalization.from_dict(metadata["normalization"])
    examples: list[Example] = []
    for seed in config.seeds:
        engine = DinoEngine(); engine.reset(seed)
        states = []
        for _ in range(config.max_frames):
            state = engine.snapshot(); states.append(state)
            features = torch.tensor([norm.encode(extract_features(state))], dtype=torch.float32)
            selected, _ = predict_actions(model, features, chosen)
            engine.step(ACTION_ORDER[int(selected[0])])
            if engine.is_terminal(): break
        for state in states:
            # DAgger uses the teacher's deterministic receding-horizon policy as
            # the label. Safety of the resulting frozen candidate is checked
            # independently by certification; no teacher enters the live path.
            preferred = Teacher._continuation(state)
            mask = tuple(action == preferred for action in ACTION_ORDER)
            examples.append(
                Example(
                    extract_features(state),
                    (mask[0], mask[1], mask[2]),
                    seed,
                    "dagger",
                    "dagger",
                    True,
                )
            )
    return examples

def select_candidate(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    passing = [report for report in reports if report.get("status") == "PASS" and report.get("heldout_status", "PASS") == "PASS" and report.get("soak_status", "PASS") == "PASS"]
    if not passing: raise ValueError("no candidate passed every gate")
    return min(passing, key=lambda report: (int(report["checkpoint_bytes"]), float(report["p95_ms"])))


def evaluate_int8(checkpoint: Path, output_dir: Path) -> dict[str, object]:
    """Evaluate dynamic INT8 parity; never promotes an artifact with changed actions."""
    model, metadata = load_dino_model(checkpoint, torch.device("cpu"))
    normalization = Normalization.from_dict(metadata["normalization"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        quantized = torch.ao.quantization.quantize_dynamic(  # type: ignore[no-untyped-call]
            model, {nn.Linear}, dtype=torch.qint8
        )
    cases = enumerate_domain(DomainSpec.small())
    features = torch.tensor(
        [normalization.encode(extract_features(case.state)) for case in cases],
        dtype=torch.float32,
    )
    with torch.inference_mode():
        baseline = model(features).argmax(dim=-1)
        candidate = quantized(features).argmax(dim=-1)
    changed = int((baseline != candidate).sum().item())
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "model-int8-state.pt"
    torch.save(quantized.state_dict(), artifact)
    return {
        "cases": len(cases),
        "changed_actions": changed,
        "parity": changed == 0,
        "bytes": artifact.stat().st_size,
        "deployed": False,
        "reason": "FP32 retained" if changed else "eligible after independent recertification",
    }
