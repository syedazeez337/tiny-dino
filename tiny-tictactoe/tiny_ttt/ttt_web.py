"""FastAPI routes + standalone app for playing the tic-tac-toe policy.

`build_router` is the reusable unit: the unified Tiny Games server mounts it at
`/api/ttt`, and `create_app` mounts it at `/api` for the standalone `tiny-ttt web`.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .ttt_contract import Normalization, extract_features, legal_mask_tuple
from .ttt_engine import DRAW, EMPTY, O, X, GameState, TicTacToeEngine
from .ttt_train import checkpoint_sha256, load_model, predict_actions

DEVICE = torch.device("cpu")


def _name(player: int) -> str:
    return "x" if player == X else "o"


def _as_player(text: str) -> int:
    return X if str(text).lower().startswith("x") else O


class Policy:
    """Loads the frozen policy once and serves greedy masked moves."""

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self.model, metadata = load_model(checkpoint, DEVICE)
        self.normalization = Normalization.from_dict(metadata["normalization"])
        self.architecture = dict(metadata["model_spec"])
        self.params = int(metadata["params"])
        self.bytes = checkpoint.stat().st_size
        self.sha256 = checkpoint_sha256(checkpoint)

    def move(self, state: GameState) -> tuple[int, list[float], float]:
        features = torch.tensor(
            [self.normalization.encode(extract_features(state, state.player))],
            dtype=torch.float32,
        )
        legal = torch.tensor([legal_mask_tuple(state)], dtype=torch.bool)
        started = time.perf_counter()
        index, probabilities = predict_actions(self.model, features, legal, DEVICE)
        latency = (time.perf_counter() - started) * 1000.0
        return int(index[0]), [float(value) for value in probabilities[0]], latency

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": "TinyTicTacToe",
            "architecture": self.architecture,
            "params": self.params,
            "bytes": self.bytes,
            "checkpoint_sha256": self.sha256,
            "feature_version": "ttt-features-v1",
        }
        report = self.checkpoint.parent / "certification.json"
        if report.exists():
            data = json.loads(report.read_text(encoding="utf-8"))
            payload["certification"] = {
                "positions": data.get("positions"),
                "decisions": data.get("decisions"),
                "optimal": data.get("optimal"),
                "non_optimal": data.get("non_optimal"),
                "illegal": data.get("illegal"),
                "perfect_play": data.get("perfect_play"),
                "claim": data.get("claim"),
            }
        return payload


@dataclass
class Run:
    run_id: str
    engine: TicTacToeEngine
    human: int
    model_move: int | None = None
    probabilities: list[float] | None = None
    latency_ms: float | None = None


class StartRequest(BaseModel):
    model_config = {"extra": "forbid"}
    human: str = "x"


class MoveRequest(BaseModel):
    model_config = {"extra": "forbid"}
    run_id: str
    cell: int


def build_router(policy: Policy) -> APIRouter:
    """A self-contained tic-tac-toe API, ready to mount under any prefix."""
    router = APIRouter()
    runs: dict[str, Run] = {}

    def model_turn(run: Run) -> None:
        state = run.engine.snapshot()
        cell, probabilities, latency = policy.move(state)
        run.engine.step(cell)
        run.model_move = cell
        run.probabilities = probabilities
        run.latency_ms = latency

    def status(run: Run) -> str:
        winner = run.engine.snapshot().winner
        if winner == 0:
            return "playing"
        if winner == DRAW:
            return "draw"
        return "human_win" if winner == run.human else "model_win"

    def payload(run: Run) -> dict[str, Any]:
        state = run.engine.snapshot()
        return {
            "run_id": run.run_id,
            "board": list(state.board),
            "turn": None if state.is_terminal else _name(state.player),
            "human": _name(run.human),
            "status": status(run),
            "model_move": run.model_move,
            "probabilities": run.probabilities,
            "latency_ms": run.latency_ms,
        }

    @router.get("/model")
    def model_info() -> dict[str, Any]:
        return policy.describe()

    @router.post("/play/start")
    def start(request: StartRequest) -> dict[str, Any]:
        human = _as_player(request.human)
        engine = TicTacToeEngine()
        engine.reset()
        run = Run(uuid.uuid4().hex[:12], engine, human)
        runs[run.run_id] = run
        if engine.snapshot().player != human:
            model_turn(run)
        return payload(run)

    @router.post("/play/move")
    def move(request: MoveRequest) -> dict[str, Any]:
        run = runs.get(request.run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="unknown run")
        state = run.engine.snapshot()
        if state.is_terminal or state.player != run.human:
            return payload(run)
        if not 0 <= request.cell < 9 or state.board[request.cell] != EMPTY:
            raise HTTPException(status_code=400, detail="illegal move")
        run.engine.step(request.cell)
        if not run.engine.is_terminal():
            model_turn(run)
        return payload(run)

    return router


def create_app(policy: Policy, web_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Tiny Tic-Tac-Toe")
    app.include_router(build_router(policy), prefix="/api")
    if web_dir is not None and web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")
    return app


def build_app(checkpoint: Path, web_dir: Path | None = None) -> FastAPI:
    resolved = web_dir or (Path(__file__).resolve().parent.parent / "web")
    return create_app(Policy(checkpoint), resolved)


def serve(
    checkpoint: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    web_dir: Path | None = None,
    open_browser: bool = False,
) -> None:
    import uvicorn

    app = build_app(checkpoint, web_dir)
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(app, host=host, port=port)
