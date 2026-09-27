from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tiny_dino.dino_arena import load_runtime

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "dino" / "deploy" / "model.pt"
CERTIFICATION = CHECKPOINT.parent / "certification.json"
EVIDENCE = ROOT / "reports" / "dino" / "evidence.json"
SOAK = CHECKPOINT.parent / "soak.json"


def test_deployed_checkpoint_has_one_unbroken_evidence_chain() -> None:
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    certification = json.loads(CERTIFICATION.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    soak = json.loads(SOAK.read_text(encoding="utf-8"))
    runtime = load_runtime(CHECKPOINT)

    assert digest == runtime.sha256 == certification["checkpoint_sha256"]
    assert digest == evidence["checkpoint_sha256"]
    assert digest == soak["checkpoint_sha256"]
    assert certification["status"] == evidence["status"] == "PASS"
    assert certification["unsafe_actions"] == 0
    assert certification["outside_teacher_horizon"] == 0
    assert evidence["heldout"]["crashes"] == 0
    assert soak["crashes"] == 0 and soak["frames"] == 1_000_000
    assert evidence["certification"]["runtime_interventions"] == 0
