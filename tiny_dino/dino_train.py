"""Training, checkpoint, inference, and calibration utilities for Tiny Dino."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from .dino_contract import FEATURE_DIM, FEATURE_VERSION, Normalization, feature_hash
from .dino_dataset import Corpus, corpus_digest
from .dino_engine import DINO_ENGINE_REVISION
from .dino_model import ModelSpec, build_model, parameter_count, set_valued_loss

CHECKPOINT_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    model: ModelSpec
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 0.003
    seed: int = 0
    device: str = "auto"
    boundary_repeats: int = 50


def load_training_spec(path: Path) -> TrainingSpec:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema") != 1:
        raise ValueError("unsupported Dino training config schema")
    return TrainingSpec(
        ModelSpec(str(payload["kind"]), int(cast(int, payload["hidden"])), int(cast(int, payload["layers"]))),
        int(cast(int, payload.get("epochs", 100))),
        int(cast(int, payload.get("batch_size", 512))),
        float(cast(float, payload.get("learning_rate", 0.003))),
        int(cast(int, payload.get("seed", 0))),
        str(payload.get("device", "auto")),
        int(cast(int, payload.get("boundary_repeats", 50))),
    )


def _normalization(features: torch.Tensor) -> Normalization:
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return Normalization(tuple(mean.tolist()), tuple(scale.tolist()))


def train_candidate(spec: TrainingSpec, corpus: Corpus, output_dir: Path) -> dict[str, object]:
    torch.manual_seed(spec.seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu" if spec.device == "auto" else spec.device
    if spec.device != "auto":
        device_name = spec.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec.seed)
        torch.cuda.reset_peak_memory_stats(device)
    rows = [example for example in corpus.examples if example.split == "train"]
    if not rows:
        raise ValueError("training split is empty")
    raw = torch.tensor([example.features for example in rows], dtype=torch.float32)
    allowed = torch.tensor([example.acceptable_mask for example in rows], dtype=torch.bool)
    normalization = _normalization(raw)
    mean = torch.tensor(normalization.mean); scale = torch.tensor(normalization.scale)
    features = (raw - mean) / scale
    features = features.to(device)
    allowed = allowed.to(device)
    model = build_model(spec.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate)
    with torch.no_grad():
        initial = float(set_valued_loss(model(features), allowed).item())
    started = time.perf_counter()
    for _ in range(spec.epochs):
        order = torch.randperm(len(rows), device=device)
        for start in range(0, len(rows), spec.batch_size):
            indices = order[start : start + spec.batch_size]
            loss = set_valued_loss(model(features[indices]), allowed[indices])
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()
    with torch.no_grad():
        final = float(set_valued_loss(model(features), allowed).item())
    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    save_dino_checkpoint(checkpoint, model, spec.model, normalization, corpus_digest(corpus))
    peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {"initial_loss": initial, "final_loss": final, "checkpoint": str(checkpoint), "params": parameter_count(model), "bytes": checkpoint.stat().st_size, "device": str(device), "training_seconds": elapsed, "peak_vram_bytes": peak_vram}

def save_dino_checkpoint(path: Path, model: nn.Module, spec: ModelSpec, normalization: Normalization, corpus_digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": CHECKPOINT_SCHEMA, "state_dict": model.state_dict(), "model_spec": {"kind": spec.kind, "hidden": spec.hidden, "layers": spec.layers}, "normalization": normalization.to_dict(), "corpus_digest": corpus_digest, "feature_version": FEATURE_VERSION, "feature_hash": feature_hash(), "engine_revision": DINO_ENGINE_REVISION, "params": parameter_count(model)}, path)

def load_dino_model(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = cast(dict[str, Any], torch.load(path, map_location=device, weights_only=True))
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("feature_version") != FEATURE_VERSION or payload.get("feature_hash") != feature_hash() or payload.get("engine_revision") != DINO_ENGINE_REVISION:
        raise ValueError("incompatible Dino checkpoint")
    raw = cast(dict[str, object], payload["model_spec"])
    spec = ModelSpec(str(raw["kind"]), int(cast(int, raw["hidden"])), int(cast(int, raw["layers"])))
    model = build_model(spec).to(device)
    model.load_state_dict(payload["state_dict"]); model.eval()
    return model, payload

def predict_actions(model: nn.Module, features: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        probabilities = torch.softmax(model(features.to(device)), dim=-1)
    return probabilities.argmax(dim=-1).cpu(), probabilities.cpu()

def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def calibrate(model: nn.Module, device: torch.device, iterations: int = 500) -> dict[str, object]:
    sample = torch.zeros(1, FEATURE_DIM, device=device)
    durations: list[float] = []
    model = model.to(device).eval()
    for _ in range(iterations):
        start = time.perf_counter(); model(sample)
        if device.type == "cuda": torch.cuda.synchronize(device)
        durations.append((time.perf_counter() - start) * 1000)
    durations.sort()
    def percentile(q: float) -> float:
        return durations[min(len(durations) - 1, int(q * len(durations)))]

    return {"iterations": iterations, "p50_ms": percentile(0.5), "p95_ms": percentile(0.95), "p99_ms": percentile(0.99), "params": parameter_count(model)}
