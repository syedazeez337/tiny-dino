"""Frozen-domain certification for an unassisted Dino checkpoint."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import torch

from .dino_contract import Normalization, extract_features
from .dino_domain import DomainSpec, domain_manifest, enumerate_domain
from .dino_engine import DinoEngine
from .dino_teacher import Teacher
from .dino_train import checkpoint_sha256, load_dino_model, predict_actions
from .io import atomic_write_json


def certify(checkpoint: Path, output_dir: Path, device: torch.device | None = None) -> dict[str, Any]:
    chosen = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_dino_model(checkpoint, chosen)
    normalization = Normalization.from_dict(cast(dict[str, list[float]], metadata["normalization"]))
    cases = enumerate_domain(DomainSpec.small())
    engine = DinoEngine(); engine.reset(0); teacher = Teacher(engine, 90)
    failures: list[dict[str, object]] = []
    certifiable = []
    safe = 0
    for case in cases:
        verdict = teacher.evaluate(case.state)
        if not verdict.acceptable:
            continue
        certifiable.append(case)
        features = torch.tensor([normalization.encode(extract_features(case.state))], dtype=torch.float32)
        selected, probabilities = predict_actions(model, features, chosen)
        action_index = int(selected[0].item())
        action = ("run", "jump", "duck")[action_index]
        if action in verdict.acceptable:
            safe += 1
        else:
            failures.append({"case_id": case.case_id, "action": action, "acceptable": list(verdict.acceptable), "probabilities": probabilities[0].tolist()})
    report: dict[str, Any] = {"status": "PASS" if not failures else "FAIL", "states_checked": len(certifiable), "generated_cases": len(cases), "outside_teacher_horizon": len(cases) - len(certifiable), "safe_actions": safe, "unsafe_actions": len(failures), "n_failing": len(failures), "runtime_interventions": 0, "domain": domain_manifest(certifiable, DomainSpec.small()), "checkpoint_sha256": checkpoint_sha256(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size, "params": metadata["params"], "engine_revision": metadata["engine_revision"], "feature_version": metadata["feature_version"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "certification.json", report)
    with (output_dir / "failures.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case_id", "action", "acceptable", "probabilities")); writer.writeheader()
        for failure in failures: writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in failure.items()})
    return report
