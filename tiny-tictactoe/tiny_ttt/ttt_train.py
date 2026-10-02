"""Training, checkpoint, and inference utilities for the tic-tac-toe policy."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from .ttt_contract import (
    FEATURE_DIM,
    FEATURE_VERSION,
    Normalization,
    feature_hash,
)
from .ttt_dataset import Corpus, corpus_digest
from .ttt_model import ModelSpec, build_model, parameter_count, set_valued_loss

CHECKPOINT_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    model: ModelSpec = ModelSpec("mlp", 96, 3)
    epochs: int = 400
    batch_size: int = 4096
    learning_rate: float = 0.002
    seed: int = 67
    device: str = "cpu"


def _normalization(features: torch.Tensor) -> Normalization:
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return Normalization(tuple(mean.tolist()), tuple(scale.tolist()))


def train_candidate(spec: TrainingSpec, corpus: Corpus, output_dir: Path) -> dict[str, object]:
    torch.manual_seed(spec.seed)
    device = torch.device(spec.device)
    rows = list(corpus.examples)
    if not rows:
        raise ValueError("corpus is empty")
    raw = torch.tensor([example.features for example in rows], dtype=torch.float32)
    allowed = torch.tensor([example.acceptable_mask for example in rows], dtype=torch.bool)
    normalization = _normalization(raw)
    mean = torch.tensor(normalization.mean)
    scale = torch.tensor(normalization.scale)
    features = ((raw - mean) / scale).to(device)
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
    save_checkpoint(checkpoint, model, spec.model, normalization, corpus_digest(corpus))
    return {
        "initial_loss": initial,
        "final_loss": final,
        "checkpoint": str(checkpoint),
        "params": parameter_count(model),
        "bytes": checkpoint.stat().st_size,
        "device": str(device),
        "training_seconds": elapsed,
        "examples": len(rows),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    spec: ModelSpec,
    normalization: Normalization,
    corpus_digest_value: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "state_dict": model.state_dict(),
            "model_spec": {"kind": spec.kind, "hidden": spec.hidden, "layers": spec.layers},
            "normalization": normalization.to_dict(),
            "corpus_digest": corpus_digest_value,
            "feature_version": FEATURE_VERSION,
            "feature_hash": feature_hash(),
            "params": parameter_count(model),
        },
        path,
    )


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = cast(dict[str, Any], torch.load(path, map_location=device, weights_only=True))
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("feature_version") != FEATURE_VERSION
        or payload.get("feature_hash") != feature_hash()
    ):
        raise ValueError("incompatible tic-tac-toe checkpoint")
    raw = cast(dict[str, object], payload["model_spec"])
    spec = ModelSpec(str(raw["kind"]), int(cast(int, raw["hidden"])), int(cast(int, raw["layers"])))
    model = build_model(spec).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def predict_actions(
    model: nn.Module,
    features: torch.Tensor,
    legal_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked greedy action selection. Returns (indices, probabilities)."""
    with torch.inference_mode():
        logits = model(features.to(device))
        masked = torch.where(
            legal_mask.to(device), logits, torch.full_like(logits, -torch.inf)
        )
        probabilities = torch.softmax(masked, dim=-1)
    return probabilities.argmax(dim=-1).cpu(), probabilities.cpu()


def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
