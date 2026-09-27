from pathlib import Path

import torch

from tiny_dino.dino_contract import FEATURE_DIM, Normalization
from tiny_dino.dino_dataset import DatasetSpec, generate_corpus
from tiny_dino.dino_model import ActionMLP, ModelSpec, build_model, set_valued_loss
from tiny_dino.dino_train import (
    TrainingSpec,
    load_dino_model,
    save_dino_checkpoint,
    train_candidate,
)


def test_candidates_produce_three_deterministic_logits() -> None:
    model = build_model(ModelSpec("mlp", 32, 2))
    inputs = torch.zeros(4, FEATURE_DIM)
    assert model(inputs).shape == (4, 3)
    assert torch.equal(model(inputs), model(inputs))


def test_set_valued_loss_rewards_mass_on_any_allowed_action() -> None:
    logits = torch.tensor([[12.0, 12.0, -12.0]])
    allowed = torch.tensor([[True, True, False]])
    assert set_valued_loss(logits, allowed).item() < 1e-5


def test_checkpoint_round_trip_preserves_metadata(tmp_path: Path) -> None:
    spec = ModelSpec("mlp", 32, 2)
    model = ActionMLP(FEATURE_DIM, 32, 2)
    norm = Normalization((0.0,) * FEATURE_DIM, (1.0,) * FEATURE_DIM)
    path = tmp_path / "model.pt"
    save_dino_checkpoint(path, model, spec, norm, "corpus-hash")
    loaded, metadata = load_dino_model(path, torch.device("cpu"))
    assert torch.equal(model(torch.zeros(1, FEATURE_DIM)), loaded(torch.zeros(1, FEATURE_DIM)))
    assert metadata["corpus_digest"] == "corpus-hash"


def test_training_writes_checkpoint_and_reduces_loss(tmp_path: Path) -> None:
    corpus = generate_corpus(DatasetSpec.small())
    result = train_candidate(TrainingSpec(ModelSpec("mlp", 32, 2), epochs=8, batch_size=128, learning_rate=0.01, seed=7, device="cpu"), corpus, tmp_path)
    assert result["final_loss"] < result["initial_loss"]
    assert result["device"] == "cpu"
    assert (tmp_path / "model.pt").exists()
