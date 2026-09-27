from pathlib import Path

from tiny_dino.dino_certify import certify
from tiny_dino.dino_contract import FEATURE_DIM, Normalization
from tiny_dino.dino_model import ActionMLP, ModelSpec
from tiny_dino.dino_train import save_dino_checkpoint


def test_certification_checks_every_case_and_records_failures(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_dino_checkpoint(checkpoint, ActionMLP(FEATURE_DIM, 16, 1), ModelSpec("mlp", 16, 1), Normalization((0.0,) * FEATURE_DIM, (1.0,) * FEATURE_DIM), "test")
    report = certify(checkpoint, tmp_path / "report")
    assert report["states_checked"] > 0
    assert report["states_checked"] == report["safe_actions"] + report["unsafe_actions"]
    assert report["runtime_interventions"] == 0
    assert (tmp_path / "report" / "certification.json").exists()
