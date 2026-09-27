import pytest

from tiny_dino.dino_contract import (
    FEATURE_DIM,
    FEATURE_LAYOUT,
    Normalization,
    extract_features,
    feature_hash,
)
from tiny_dino.dino_engine import DinoEngine


def test_feature_layout_is_fixed_and_complete() -> None:
    assert FEATURE_DIM == 28
    assert len(FEATURE_LAYOUT) == FEATURE_DIM
    assert len(set(FEATURE_LAYOUT)) == FEATURE_DIM


def test_feature_extraction_is_deterministic_and_nonmutating() -> None:
    engine = DinoEngine()
    engine.reset(17)
    state = engine.snapshot()
    assert extract_features(state) == extract_features(state)
    assert engine.snapshot() == state


def test_normalization_round_trip_and_zero_scale_rejection() -> None:
    norm = Normalization(tuple(range(FEATURE_DIM)), (2.0,) * FEATURE_DIM)
    encoded = norm.encode(tuple(float(i + 2) for i in range(FEATURE_DIM)))
    assert encoded == pytest.approx((1.0,) * FEATURE_DIM)
    assert Normalization.from_dict(norm.to_dict()) == norm
    with pytest.raises(ValueError):
        Normalization((0.0,) * FEATURE_DIM, (0.0,) * FEATURE_DIM)


def test_feature_hash_is_stable() -> None:
    assert feature_hash() == feature_hash()
    assert len(feature_hash()) == 64
