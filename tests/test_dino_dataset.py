from pathlib import Path

from tiny_dino.dino_dataset import DatasetSpec, corpus_digest, generate_corpus, write_corpus


def test_corpus_is_deterministic_balanced_and_split_isolated(tmp_path: Path) -> None:
    spec = DatasetSpec.small()
    first = generate_corpus(spec)
    second = generate_corpus(spec)
    assert first == second
    assert corpus_digest(first) == corpus_digest(second)
    assert {example.split for example in first.examples} == {"train", "dev", "heldout", "dagger"}
    memberships: dict[tuple[int, str], set[str]] = {}
    for example in first.examples:
        memberships.setdefault((example.course_seed, example.family), set()).add(example.split)
    assert all(len(splits) == 1 for splits in memberships.values())
    assert sum(example.family == "empty" for example in first.examples) / len(first.examples) <= spec.max_empty_run_fraction
    manifest = write_corpus(first, tmp_path)
    assert (tmp_path / "corpus.jsonl").exists()
    assert manifest["sha256"] == corpus_digest(first)


def test_corpus_contains_each_action_label() -> None:
    corpus = generate_corpus(DatasetSpec.small())
    covered = {index for example in corpus.examples for index, allowed in enumerate(example.acceptable_mask) if allowed}
    assert covered == {0, 1, 2}
