"""Versioned deterministic corpus generation for Tiny Dino."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from .dino_contract import extract_features
from .dino_domain import DomainSpec, enumerate_domain
from .dino_engine import DinoEngine
from .dino_teacher import Teacher
from .io import append_jsonl, atomic_write_json

DATASET_VERSION = "dino-corpus-v1"
Split = Literal["train", "dev", "heldout", "dagger"]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    train_seeds: tuple[int, ...]
    dev_seeds: tuple[int, ...]
    heldout_seeds: tuple[int, ...]
    dagger_seeds: tuple[int, ...]
    max_empty_run_fraction: float = 0.2
    lookahead_frames: int = 90

    @classmethod
    def small(cls) -> DatasetSpec:
        return cls((1, 2), (101,), (201,), (301,))


@dataclass(frozen=True, slots=True)
class Example:
    features: tuple[float, ...]
    acceptable_mask: tuple[bool, bool, bool]
    course_seed: int
    family: str
    split: Split
    boundary: bool

    def as_dict(self) -> dict[str, object]:
        return {"features": list(self.features), "acceptable_mask": list(self.acceptable_mask), "course_seed": self.course_seed, "family": self.family, "split": self.split, "boundary": self.boundary}


@dataclass(frozen=True, slots=True)
class Corpus:
    examples: tuple[Example, ...]
    spec: DatasetSpec


def _seed_splits(spec: DatasetSpec) -> tuple[tuple[Split, tuple[int, ...]], ...]:
    return (("train", spec.train_seeds), ("dev", spec.dev_seeds), ("heldout", spec.heldout_seeds), ("dagger", spec.dagger_seeds))


def generate_corpus(spec: DatasetSpec) -> Corpus:
    groups = [set(seeds) for _, seeds in _seed_splits(spec)]
    if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i + 1, len(groups))):
        raise ValueError("dataset seed splits overlap")
    cases = enumerate_domain(DomainSpec.small())
    engine = DinoEngine(); engine.reset(0)
    teacher = Teacher(engine, spec.lookahead_frames)
    examples: list[Example] = []
    for split, seeds in _seed_splits(spec):
        for seed in seeds:
            for case in cases:
                state = replace(case.state, seed=seed)
                verdict = teacher.evaluate(state)
                if not verdict.acceptable:
                    continue
                preferred = Teacher._continuation(state)
                if preferred not in verdict.acceptable:
                    preferred = verdict.acceptable[0]
                mask = (preferred == "run", preferred == "jump", preferred == "duck")
                examples.append(Example(extract_features(state), mask, seed, case.family, split, True))
    return Corpus(tuple(examples), spec)


def corpus_digest(corpus: Corpus) -> str:
    payload = {"version": DATASET_VERSION, "spec": asdict(corpus.spec), "examples": [item.as_dict() for item in corpus.examples]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_corpus(corpus: Corpus, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "corpus.jsonl"
    data_path.unlink(missing_ok=True)
    for example in corpus.examples:
        append_jsonl(data_path, example.as_dict())
    manifest: dict[str, object] = {"version": DATASET_VERSION, "examples": len(corpus.examples), "sha256": corpus_digest(corpus), "spec": asdict(corpus.spec)}
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def load_dataset_spec(payload: Mapping[str, object]) -> DatasetSpec:
    def seeds(name: str) -> tuple[int, ...]:
        raw = payload[name]
        if not isinstance(raw, list) or not all(isinstance(value, int) for value in raw):
            raise ValueError(f"{name} must be an integer list")
        return tuple(raw)
    fraction = payload.get("max_empty_run_fraction", 0.2)
    lookahead = payload.get("lookahead_frames", 90)
    if not isinstance(fraction, (int, float)) or not isinstance(lookahead, int):
        raise TypeError("invalid corpus numeric setting")
    return DatasetSpec(seeds("train_seeds"), seeds("dev_seeds"), seeds("heldout_seeds"), seeds("dagger_seeds"), float(fraction), lookahead)
