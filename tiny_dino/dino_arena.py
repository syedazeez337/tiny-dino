"""Unassisted trajectory runner for Tiny Dino."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from .dino_contract import (
    ACTION_ORDER,
    FEATURE_VERSION,
    Action,
    Normalization,
    extract_features,
    feature_hash,
)
from .dino_engine import DINO_ENGINE_REVISION, DinoEngine, EngineState
from .dino_model import ModelSpec, build_model


@dataclass(slots=True)
class Runtime:
    checkpoint: Path
    model: nn.Module
    metadata: dict[str, Any]
    normalization: Normalization
    device: torch.device
    sha256: str

def load_runtime(checkpoint: Path, device: torch.device | None = None) -> Runtime:
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    chosen = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = cast(dict[str, Any], torch.load(checkpoint, map_location=chosen, weights_only=True))
    if (
        metadata.get("schema") != 1
        or metadata.get("feature_version") != FEATURE_VERSION
        or metadata.get("feature_hash") != feature_hash()
        or metadata.get("engine_revision") != DINO_ENGINE_REVISION
    ):
        raise ValueError("incompatible Dino checkpoint")
    raw = cast(dict[str, object], metadata["model_spec"])
    spec = ModelSpec(str(raw["kind"]), int(cast(int, raw["hidden"])), int(cast(int, raw["layers"])))
    model = build_model(spec).to(chosen)
    model.load_state_dict(metadata["state_dict"])
    model.eval()
    normalization = Normalization.from_dict(cast(dict[str, list[float]], metadata["normalization"]))
    return Runtime(checkpoint, model, metadata, normalization, chosen, hashlib.sha256(checkpoint.read_bytes()).hexdigest())

def model_action(runtime: Runtime, state: EngineState) -> tuple[Action, list[float], float]:
    features = torch.tensor([runtime.normalization.encode(extract_features(state))], dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError("non-finite policy features")
    started = time.perf_counter()
    with torch.inference_mode():
        logits = runtime.model(features.to(runtime.device))
        probabilities = torch.softmax(logits, dim=-1).cpu()
    if logits.shape != (1, len(ACTION_ORDER)) or not torch.isfinite(probabilities).all():
        raise ValueError("invalid policy output")
    selected = int(probabilities.argmax(dim=-1)[0])
    return ACTION_ORDER[selected], probabilities[0].tolist(), (time.perf_counter() - started) * 1000

def run_seed(runtime: Runtime, seed: int, max_frames: int) -> dict[str, object]:
    engine = DinoEngine(); engine.reset(seed)
    latencies: list[float] = []
    actions = {action: 0 for action in ACTION_ORDER}
    for _ in range(max_frames):
        action, _, latency = model_action(runtime, engine.snapshot())
        latencies.append(latency)
        actions[action] += 1
        engine.step(action)
        if engine.is_terminal(): break
    latencies.sort()
    frames = engine.snapshot().frame
    return {"seed": seed, "frames": frames, "distance": engine.snapshot().distance, "crashed": engine.is_terminal(), "actions": actions, "latency_p50_ms": latencies[len(latencies)//2], "latency_p95_ms": latencies[min(len(latencies)-1,int(len(latencies)*0.95))], "runtime_interventions": 0}

def run_suite(runtime: Runtime, seeds: tuple[int, ...], max_frames: int) -> dict[str, object]:
    runs = [run_seed(runtime, seed, max_frames) for seed in seeds]
    frames = sum(run["frames"] for run in runs if isinstance(run["frames"], int))
    return {"runs": runs, "seeds": len(seeds), "crashes": sum(bool(run["crashed"]) for run in runs), "frames": frames, "runtime_interventions": 0}
