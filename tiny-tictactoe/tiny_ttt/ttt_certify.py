"""Exhaustive certification of the frozen policy over the full domain."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .ttt_contract import Normalization, extract_features, legal_mask_tuple
from .ttt_domain import domain_manifest, enumerate_domain
from .ttt_teacher import Teacher
from .ttt_train import checkpoint_sha256, load_model, predict_actions

CLAIM = (
    "This frozen policy makes a game-theoretically optimal move in every reachable "
    "tic-tac-toe position, so it never loses with unlimited play."
)


@dataclass(frozen=True, slots=True)
class CertificationReport:
    checkpoint_sha256: str
    domain_sha256: str
    positions: int
    decisions: int
    optimal: int
    non_optimal: int
    illegal: int
    perfect_play: bool
    examples: list[dict[str, object]]
    claim: str
    seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "domain_sha256": self.domain_sha256,
            "positions": self.positions,
            "decisions": self.decisions,
            "optimal": self.optimal,
            "non_optimal": self.non_optimal,
            "illegal": self.illegal,
            "perfect_play": self.perfect_play,
            "examples": self.examples,
            "claim": self.claim,
            "seconds": self.seconds,
        }


def certify(checkpoint: Path, device: torch.device | None = None) -> CertificationReport:
    chosen = device or torch.device("cpu")
    model, metadata = load_model(checkpoint, chosen)
    normalization = Normalization.from_dict(metadata["normalization"])
    teacher = Teacher()
    states = enumerate_domain()
    decisions = [state for state in states if not state.is_terminal]
    optimal = 0
    non_optimal = 0
    illegal = 0
    examples: list[dict[str, object]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for state in decisions:
            features = torch.tensor(
                [normalization.encode(extract_features(state, state.player))],
                dtype=torch.float32,
            )
            legal = torch.tensor([legal_mask_tuple(state)], dtype=torch.bool)
            index, _ = predict_actions(model, features, legal, chosen)
            choice = int(index[0])
            verdict = teacher.evaluate(state)
            if choice not in verdict.outcomes:
                illegal += 1
                reason = "illegal"
            elif choice in verdict.acceptable:
                optimal += 1
                continue
            else:
                non_optimal += 1
                reason = "non_optimal"
            if len(examples) < 25:
                examples.append(
                    {
                        "board": list(state.board),
                        "player": state.player,
                        "choice": choice,
                        "acceptable": list(verdict.acceptable),
                        "reason": reason,
                    }
                )
    seconds = time.perf_counter() - started
    manifest = domain_manifest()
    return CertificationReport(
        checkpoint_sha256=checkpoint_sha256(checkpoint),
        domain_sha256=str(manifest["sha256"]),
        positions=len(states),
        decisions=len(decisions),
        optimal=optimal,
        non_optimal=non_optimal,
        illegal=illegal,
        perfect_play=(non_optimal == 0 and illegal == 0),
        examples=examples,
        claim=CLAIM,
        seconds=seconds,
    )


def write_report(report: CertificationReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "certification.json"
    path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    return path
