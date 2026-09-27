from tiny_dino.dino_contract import FEATURE_DIM, Normalization
from tiny_dino.dino_dagger import (
    ExpertRolloutConfig,
    collect_teacher_rollouts,
    evaluate_int8,
    select_candidate,
)
from tiny_dino.dino_model import ActionMLP, ModelSpec
from tiny_dino.dino_train import save_dino_checkpoint


def test_selection_requires_pass_then_minimizes_size_and_latency() -> None:
    reports = [
        {"name": "small-fail", "status": "FAIL", "checkpoint_bytes": 10, "p95_ms": 0.1},
        {"name": "large", "status": "PASS", "checkpoint_bytes": 100, "p95_ms": 0.1},
        {"name": "small", "status": "PASS", "checkpoint_bytes": 50, "p95_ms": 0.2},
    ]
    assert select_candidate(reports)["name"] == "small"


def test_teacher_rollouts_are_deterministic_and_cover_controls() -> None:
    config = ExpertRolloutConfig((301, 302), 2_000)
    first = collect_teacher_rollouts(config)
    assert first == collect_teacher_rollouts(config)
    labels = {index for row in first for index, enabled in enumerate(row.acceptable_mask) if enabled}
    assert labels == {0, 1, 2}


def test_int8_evaluation_never_auto_deploys(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_dino_checkpoint(
        checkpoint,
        ActionMLP(FEATURE_DIM, 16, 1),
        ModelSpec("mlp", 16, 1),
        Normalization((0.0,) * FEATURE_DIM, (1.0,) * FEATURE_DIM),
        "test",
    )
    report = evaluate_int8(checkpoint, tmp_path / "int8")
    assert report["cases"] > 0
    assert report["deployed"] is False
    assert (tmp_path / "int8" / "model-int8-state.pt").exists()
