"""Teacher-labelled corpus built from the full reachable-position domain."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .ttt_contract import extract_features
from .ttt_domain import domain_manifest, enumerate_domain
from .ttt_engine import GameState
from .ttt_teacher import Teacher

DATASET_VERSION = "ttt-corpus-v1"


@dataclass(frozen=True, slots=True)
class Example:
    features: tuple[float, ...]
    acceptable_mask: tuple[bool, ...]  # length 9
    board: tuple[int, ...]
    player: int

    def as_dict(self) -> dict[str, object]:
        return {
            "features": list(self.features),
            "acceptable_mask": list(self.acceptable_mask),
            "board": list(self.board),
            "player": self.player,
        }


@dataclass(frozen=True, slots=True)
class Corpus:
    examples: tuple[Example, ...]
    manifest: dict[str, object]


def generate_corpus() -> Corpus:
    teacher = Teacher()
    examples: list[Example] = []
    for state in enumerate_domain():
        if state.is_terminal:
            continue
        verdict = teacher.evaluate(state)
        examples.append(
            Example(
                extract_features(state, state.player),
                verdict.acceptable_mask(),
                state.board,
                state.player,
            )
        )
    corpus = Corpus(tuple(examples), {})
    manifest = {
        "version": DATASET_VERSION,
        "examples": len(corpus.examples),
        "domain": domain_manifest(),
        "sha256": _digest(corpus.examples),
    }
    return Corpus(corpus.examples, manifest)


def _digest(examples: tuple[Example, ...]) -> str:
    payload = json.dumps(
        [example.as_dict() for example in examples],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def corpus_digest(corpus: Corpus) -> str:
    return _digest(corpus.examples)
