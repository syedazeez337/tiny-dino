from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("dino-corpus", "dino-calibrate", "dino-train", "dino-certify",
                            "dino-arena", "dino-web")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/latest"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--expert-frames", type=int, default=20_000)
    parser.add_argument("--web-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", dest="open_browser", action="store_true")
    args = parser.parse_args()
    if args.command in {"dino-corpus", "dino-calibrate", "dino-train"}:
        if not args.config:
            parser.error(f"{args.command} requires --config")
        import json
        from dataclasses import replace

        import torch

        from .dino_dataset import Corpus, Example, generate_corpus, load_dataset_spec, write_corpus
        corpus_path = args.corpus or args.config
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
        corpus = generate_corpus(load_dataset_spec(payload))
        if args.command == "dino-corpus":
            print(write_corpus(corpus, args.output))
            return
        from .dino_model import build_model
        from .dino_train import calibrate, load_training_spec, train_candidate
        spec = load_training_spec(args.config)
        if args.command == "dino-calibrate":
            from .io import atomic_write_json
            args.output.mkdir(parents=True, exist_ok=True)
            result = calibrate(build_model(spec.model), torch.device("cpu"))
            atomic_write_json(args.output / "calibration.json", result)
            print(result)
            return
        from .dino_dagger import ExpertRolloutConfig, collect_teacher_rollouts
        expert = collect_teacher_rollouts(ExpertRolloutConfig(corpus.spec.dagger_seeds, args.expert_frames))
        unique: dict[tuple[tuple[float, ...], tuple[bool, bool, bool]], Example] = {}
        for row in (*corpus.examples, *expert):
            if row.split == "train":
                unique[(row.features, row.acceptable_mask)] = replace(row, split="train")
        rows = tuple(unique.values())
        boundary = tuple(row for row in rows if row.boundary)
        trained = Corpus(boundary * (spec.boundary_repeats - 1) + rows, corpus.spec)
        print(train_candidate(spec, trained, args.output))
        return
    if args.command == "dino-certify":
        if not args.checkpoint:
            parser.error("dino-certify requires --checkpoint")
        from .dino_certify import certify
        print(certify(args.checkpoint, args.output)["status"])
        return
    if args.command == "dino-arena":
        if not args.checkpoint:
            parser.error("dino-arena requires --checkpoint")
        from .dino_arena import load_runtime, run_suite
        print(run_suite(load_runtime(args.checkpoint), (201, 202, 203, 204), 10_000))
        return
    if not args.checkpoint:
        parser.error("dino-web requires --checkpoint")
    from .dino_web import serve
    serve(args.checkpoint, args.host, args.port, args.web_dir, args.open_browser)
