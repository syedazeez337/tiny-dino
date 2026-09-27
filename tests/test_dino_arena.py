from pathlib import Path

import torch

from tiny_dino.dino_arena import load_runtime, run_seed
from tiny_dino.dino_contract import FEATURE_DIM, Normalization
from tiny_dino.dino_model import ActionMLP, ModelSpec
from tiny_dino.dino_train import save_dino_checkpoint


def test_arena_is_deterministic_and_unassisted(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_dino_checkpoint(checkpoint, ActionMLP(FEATURE_DIM, 16, 1), ModelSpec("mlp", 16, 1), Normalization((0.0,) * FEATURE_DIM, (1.0,) * FEATURE_DIM), "test")
    runtime = load_runtime(checkpoint, torch.device("cpu"))
    first = run_seed(runtime, 7, 300)
    second = run_seed(runtime, 7, 300)
    deterministic = ("seed", "frames", "distance", "crashed", "actions")
    assert {key: first[key] for key in deterministic} == {
        key: second[key] for key in deterministic
    }
    assert first["runtime_interventions"] == 0
