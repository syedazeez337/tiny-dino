from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from tiny_dino.dino_arena import load_runtime, model_action
from tiny_dino.dino_contract import FEATURE_DIM, Normalization
from tiny_dino.dino_engine import DinoEngine
from tiny_dino.dino_model import ActionMLP, ModelSpec
from tiny_dino.dino_train import save_dino_checkpoint
from tiny_dino.dino_web import create_app


def _client(tmp_path: Path) -> TestClient:
    checkpoint = tmp_path / "model.pt"
    save_dino_checkpoint(
        checkpoint,
        ActionMLP(FEATURE_DIM, 16, 1),
        ModelSpec("mlp", 16, 1),
        Normalization((0.0,) * FEATURE_DIM, (1.0,) * FEATURE_DIM),
        "test",
    )
    return TestClient(create_app(load_runtime(checkpoint, torch.device("cpu"))))


def test_live_play_is_model_driven_and_unassisted(tmp_path: Path) -> None:
    client = _client(tmp_path)
    started = client.post("/api/play/start", json={"seed": 7}).json()
    result = client.post(
        "/api/play/step", json={"session_id": started["session_id"], "frames": 3}
    ).json()
    assert result["state"]["frame"] == 3
    assert result["action"] in {"run", "jump", "duck"}
    assert result["runtime_interventions"] == 0
    replay = client.get(f"/api/play/replay/{started['session_id']}").json()
    assert replay["exact"] is True
    assert replay["verified"] is True
    assert len(replay["actions"]) == 3
    assert len(replay["frames"]) == 3
    assert replay["final_state"] == result["state"]


def test_live_import_graph_excludes_teacher() -> None:
    root = Path(__file__).resolve().parents[1] / "tiny_dino"
    for name in ("dino_web.py", "dino_arena.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any("dino_teacher" in module for module in imports)
    result = subprocess.run(
        [sys.executable, "-c", "import sys, tiny_dino.dino_web; print('tiny_dino.dino_teacher' in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_live_runtime_fails_closed_on_non_finite_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_dino_checkpoint(checkpoint, ActionMLP(FEATURE_DIM, 16, 1), ModelSpec("mlp", 16, 1), Normalization((0.0,) * FEATURE_DIM, (1.0,) * FEATURE_DIM), "test")
    runtime = load_runtime(checkpoint, torch.device("cpu"))

    class InvalidModel(nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.full((features.shape[0], 3), float("nan"))

    runtime.model = InvalidModel()
    engine = DinoEngine(); engine.reset(1)
    with pytest.raises(ValueError, match="invalid policy output"):
        model_action(runtime, engine.snapshot())
