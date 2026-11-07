"""Command line for Tiny Tic-Tac-Toe: train, certify, and play."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .ttt_contract import Normalization, extract_features, legal_mask_tuple
from .ttt_dataset import generate_corpus
from .ttt_engine import EMPTY, O, X, TicTacToeEngine, other
from .ttt_train import TrainingSpec, load_model, predict_actions

SYMBOLS = {EMPTY: ".", X: "X", O: "O"}
DEFAULT_CHECKPOINT = Path("artifacts/model.pt")


def _name(player: int) -> str:
    return "X" if player == X else "O"


def render(board: tuple[int, ...]) -> str:
    rows: list[str] = []
    for row in range(3):
        cells = []
        for col in range(3):
            index = row * 3 + col
            cell = board[index]
            cells.append(SYMBOLS[cell] if cell != EMPTY else str(index + 1))
        rows.append(" " + " | ".join(cells))
        if row < 2:
            rows.append("---+---+---")
    return "\n".join(rows)


def _parse_moves(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(token) - 1 for token in raw.replace(",", " ").split()]


def _load_policy(checkpoint: Path) -> tuple[torch.nn.Module, Normalization]:
    device = torch.device("cpu")
    model, metadata = load_model(checkpoint, device)
    return model, Normalization.from_dict(metadata["normalization"])


def _model_move(
    engine: TicTacToeEngine,
    model: torch.nn.Module,
    normalization: Normalization,
) -> int:
    state = engine.snapshot()
    features = torch.tensor(
        [normalization.encode(extract_features(state, state.player))], dtype=torch.float32
    )
    legal = torch.tensor([legal_mask_tuple(state)], dtype=torch.bool)
    index, probabilities = predict_actions(model, features, legal, torch.device("cpu"))
    cell = int(index[0])
    return cell


def play(human_raw: str, moves_raw: str, checkpoint: Path) -> int:
    human = X if human_raw.lower().startswith("x") else O
    model_player = other(human)
    model, normalization = _load_policy(checkpoint)
    engine = TicTacToeEngine()
    engine.reset()
    queued = _parse_moves(moves_raw)

    print(f"You are {_name(human)}. The model is {_name(model_player)}.")
    print(render(engine.snapshot().board))
    print()

    while not engine.is_terminal():
        state = engine.snapshot()
        if state.player == model_player:
            cell = _model_move(engine, model, normalization)
            engine.step(cell)
            print(f"Model ({_name(model_player)}) plays cell {cell + 1}.")
        else:
            if not queued:
                if sys.stdin.isatty():
                    try:
                        raw = input(f"Your move ({_name(human)}), cell 1-9: ")
                    except EOFError:
                        print("No move provided; stopping.")
                        return 0
                    queued.append(int(raw) - 1)
                else:
                    print("No more moves supplied. Pass --moves \"1,5,9\" to continue.")
                    return 0
            cell = queued.pop(0)
            if not 0 <= cell < 9 or state.board[cell] != EMPTY:
                print(f"Illegal move: cell {cell + 1} is not playable.")
                return 1
            engine.step(cell)
            print(f"You ({_name(human)}) play cell {cell + 1}.")
        print(render(engine.snapshot().board))
        print()

    final = engine.snapshot()
    if final.winner == 3:
        print("Result: draw.")
    elif final.winner == model_player:
        print(f"Result: model ({_name(model_player)}) wins.")
    elif final.winner == human:
        print(f"Result: you ({_name(human)}) win.")
    print(f"RESULT winner={final.winner} human={human}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tiny-ttt")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="distil the policy from the minimax teacher")
    train.add_argument("--output", type=Path, default=Path("artifacts"))
    train.add_argument("--epochs", type=int, default=400)

    certify = sub.add_parser("certify", help="check the policy on every position")
    certify.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    certify.add_argument("--output", type=Path, default=Path("artifacts"))

    game = sub.add_parser("play", help="play against the policy")
    game.add_argument("--human", default="x", help="x or o")
    game.add_argument("--moves", default="", help='your moves, e.g. "5,1,9"')
    game.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)

    web = sub.add_parser("web", help="serve the browser game")
    web.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--open", dest="open_browser", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "train":
        from .ttt_train import train_candidate

        corpus = generate_corpus()
        print(f"corpus: {corpus.manifest['examples']} positions")
        spec = TrainingSpec(epochs=args.epochs)
        report = train_candidate(spec, corpus, args.output)
        print(json_report(report))
        return 0

    if args.command == "certify":
        from .ttt_certify import certify, write_report

        report = certify(args.checkpoint)
        path = write_report(report, args.output)
        print(json_report(report.as_dict()))
        print(f"wrote {path}")
        return 0

    if args.command == "play":
        if not args.checkpoint.exists():
            parser.error(f"checkpoint not found: {args.checkpoint}; run `train` first")
        return play(args.human, args.moves, args.checkpoint)

    if args.command == "web":
        if not args.checkpoint.exists():
            parser.error(f"checkpoint not found: {args.checkpoint}; run `train` first")
        from .ttt_web import serve

        serve(args.checkpoint, args.host, args.port, open_browser=args.open_browser)
        return 0

    return 1


def json_report(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
