"""Teacher-free FastAPI runtime for the frozen Tiny Dino policy."""
from __future__ import annotations

import threading
import uuid
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .dino_arena import Runtime, load_runtime, model_action, run_suite
from .dino_contract import Action
from .dino_engine import DinoEngine, EngineState

MAX_REPLAY_FRAMES = 100_000

@dataclass(slots=True)
class Session:
    engine: DinoEngine
    initial_state: EngineState
    actions: list[Action]


class StartRequest(BaseModel):
    seed: int = 0


class StepRequest(BaseModel):
    session_id: str
    frames: int = Field(default=1, ge=1, le=120)


class ArenaRequest(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [201, 202, 203, 204])
    max_frames: int = Field(default=10_000, ge=1, le=1_000_000)


def _frame(state: EngineState) -> dict[str, object]:
    return state.as_dict()


def _load_ttt_policy(ttt_checkpoint: Path | None) -> Any | None:
    if ttt_checkpoint is None or not Path(ttt_checkpoint).exists():
        return None
    from tiny_ttt.ttt_web import Policy as TicTacToePolicy

    return TicTacToePolicy(Path(ttt_checkpoint))


def create_app(
    runtime: Runtime,
    *,
    static_dir: Path | None = None,
    certification: dict[str, Any] | None = None,
    ttt_policy: Any | None = None,
) -> FastAPI:
    if certification is not None and certification.get("checkpoint_sha256") != runtime.sha256:
        raise ValueError("certification snapshot does not match the loaded checkpoint")
    app = FastAPI(title="Tiny Games", docs_url=None, redoc_url=None)
    sessions: dict[str, Session] = {}
    lock = threading.Lock()

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "checkpoint": runtime.checkpoint.name}

    @app.get("/api/model")
    def model_info() -> dict[str, object]:
        spec = runtime.metadata["model_spec"]
        return {
            "name": "Tiny Dino Policy",
            "params": runtime.metadata["params"],
            "checkpoint_bytes": runtime.checkpoint.stat().st_size,
            "checkpoint_sha256": runtime.sha256,
            "device": str(runtime.device),
            "engine_revision": runtime.metadata["engine_revision"],
            "feature_version": runtime.metadata["feature_version"],
            "architecture": spec,
            "runtime_interventions": 0,
        }

    @app.post("/api/play/start")
    def start(request: StartRequest) -> dict[str, object]:
        engine = DinoEngine()
        engine.reset(request.seed)
        session_id = uuid.uuid4().hex[:16]
        with lock:
            if len(sessions) >= 128:
                sessions.pop(next(iter(sessions)))
            sessions[session_id] = Session(engine, engine.snapshot(), [])
        return {"session_id": session_id, "state": _frame(engine.snapshot()), "status": "running"}

    @app.post("/api/play/step")
    def step(request: StepRequest) -> dict[str, object]:
        with lock:
            session = sessions.get(request.session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="unknown play session")
            last_action = "run"
            probabilities = [1.0, 0.0, 0.0]
            latency_ms = 0.0
            frames: list[dict[str, object]] = []
            for _ in range(request.frames):
                if session.engine.is_terminal() or len(session.actions) >= MAX_REPLAY_FRAMES:
                    break
                last_action, probabilities, latency_ms = model_action(
                    runtime, session.engine.snapshot()
                )
                session.engine.step(last_action)
                session.actions.append(last_action)
                frames.append(
                    {
                        "action": last_action,
                        "probabilities": probabilities,
                        "latency_ms": latency_ms,
                        "state": _frame(session.engine.snapshot()),
                    }
                )
            state = session.engine.snapshot()
        return {
            "session_id": request.session_id,
            "state": _frame(state),
            "frames": frames,
            "status": "crashed" if session.engine.is_terminal() else "complete" if len(session.actions) >= MAX_REPLAY_FRAMES else "running",
            "action": last_action,
            "probabilities": probabilities,
            "latency_ms": latency_ms,
            "runtime_interventions": 0,
        }

    @app.get("/api/play/replay/{session_id}")
    def replay(session_id: str, start: int = 0, limit: int = 600) -> dict[str, object]:
        if start < 0 or limit < 1 or limit > 600:
            raise HTTPException(status_code=400, detail="invalid replay window")
        with lock:
            session = sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="unknown play session")
            initial_state = session.initial_state
            actions = list(session.actions)
            expected = session.engine.snapshot()
        replay_engine = DinoEngine()
        replay_engine.restore(initial_state)
        frames: list[dict[str, object]] = []
        for index, action in enumerate(actions):
            replay_engine.step(action)
            if start <= index < start + limit:
                frames.append({"index": index + 1, "action": action, "state": _frame(replay_engine.snapshot())})
        reconstructed = replay_engine.snapshot()
        verified = reconstructed == expected
        if not verified:
            raise HTTPException(status_code=500, detail="stored replay failed deterministic verification")
        return {
            "session_id": session_id,
            "initial_state": _frame(initial_state),
            "actions": actions,
            "frames": frames,
            "window_start": start,
            "window_limit": limit,
            "final_state": _frame(reconstructed),
            "verified": verified,
            "exact": True,
        }

    @app.post("/api/arena")
    def arena(request: ArenaRequest) -> dict[str, object]:
        if not request.seeds or len(request.seeds) > 100:
            raise HTTPException(status_code=400, detail="provide between 1 and 100 seeds")
        if len(request.seeds) * request.max_frames > 1_000_000:
            raise HTTPException(status_code=400, detail="arena request exceeds one million frames")
        return run_suite(runtime, tuple(request.seeds), request.max_frames)

    @app.get("/api/certification")
    def proof() -> dict[str, Any]:
        if certification is None:
            raise HTTPException(status_code=404, detail="no frozen certification snapshot")
        return certification

    if ttt_policy is not None:
        from tiny_ttt.ttt_web import build_router as build_ttt_router

        app.include_router(build_ttt_router(ttt_policy), prefix="/api/ttt")

    if static_dir is not None and static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="web")
    return app


def serve(
    checkpoint: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    web_dir: Path | None = None,
    open_browser: bool = False,
    ttt_checkpoint: Path | None = None,
) -> None:
    import json

    import uvicorn

    runtime = load_runtime(checkpoint)
    warm_engine = DinoEngine()
    warm_engine.reset(0)
    model_action(runtime, warm_engine.snapshot())  # pay torch's first-call setup before serving
    cert_path = checkpoint.parent / "certification.json"
    certification = json.loads(cert_path.read_text(encoding="utf-8")) if cert_path.exists() else None
    ttt_policy = _load_ttt_policy(ttt_checkpoint)
    if ttt_policy is not None:
        games = "Dino + Tic-Tac-Toe"
    else:
        games = "Dino"
    print(f"serving Tiny Games ({games}) at http://{host}:{port}/")
    static = web_dir or Path(__file__).resolve().parents[1] / "web" / "dist"
    if open_browser:
        webbrowser.open(f"http://{host}:{port}/")
    uvicorn.run(
        create_app(
            runtime,
            static_dir=static if static.exists() else None,
            certification=certification,
            ttt_policy=ttt_policy,
        ),
        host=host,
        port=port,
        log_level="info",
    )
