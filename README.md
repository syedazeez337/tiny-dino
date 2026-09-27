# Tiny Dino

A 21,699-parameter MLP that plays a deterministic, Chromium-compatible Dino
runner. At run time it has no planner, solver, or safety shield. Every frame, the
model reads a structured state vector and picks `run`, `jump`, or `duck`. That
action is applied unchanged.

The repository contains the game engine, the teacher used to label training
data, training and certification code, the certified checkpoint, the evidence
behind the claim below, and a web app for watching the policy play.

## Result

From `reports/dino/evidence.json` and `artifacts/dino/deploy/`:

| Gate | Value |
|---|---|
| Status | PASS |
| Model | MLP, 3 layers, hidden width 96, 21,699 parameters |
| Checkpoint | `artifacts/dino/deploy/model.pt`, 90,929 bytes, FP32 |
| Frozen certification domain | 5,400/5,400 safe actions, 0 unsafe, 0 exclusions |
| Held-out and soak suite | 1,000,000 frames over seeds 201 to 220, 0 crashes |
| Runtime interventions | 0 |
| CPU latency | 0.0484 ms p50, 0.0545 ms p95, 0.0709 ms p99 |
| INT8 post-training quantization | Rejected: changed 36 certification actions |
| Checkpoint SHA-256 | `b4cd918bb5fae7d0a826ef23a055cbe05e70505f3487243e43d0919813f11109` |

This is a bounded empirical claim over a finite certification domain and a finite
set of seeds. It does not claim that the endless stochastic game is solved.

## How it works

- **Engine** (`tiny_dino/dino_engine.py`): fixed-step 60 FPS simulation of the
  Chromium offline Dino game. Physics constants, obstacle geometry, and
  collision boxes follow Chromium. See [Provenance](#provenance).
- **Contract** (`dino_contract.py`): the state features, their normalization,
  and the action order shared by training and inference.
- **Teacher** (`dino_teacher.py`): an exhaustive look-ahead that labels which
  actions survive from a given state. It is used only to build training data and
  to certify. The live runtime never imports it, and a test enforces that.
- **Domain** (`dino_domain.py`): the frozen set of 5,400 certification states,
  identified by its SHA-256 manifest.
- **Dataset and DAgger** (`dino_dataset.py`, `dino_dagger.py`): a seeded
  corpus with leak-free splits, plus teacher-labelled rollouts.
- **Model and training** (`dino_model.py`, `dino_train.py`): a small MLP
  trained with a set-valued loss, because more than one action can be safe.
- **Certification and arena** (`dino_certify.py`, `dino_arena.py`): check
  every domain state, then run held-out seeds for up to 50,000 frames each.
- **Web** (`dino_web.py`, `web/`): a FastAPI server and a React client that
  draws the live game, the action probabilities, and arena charts.

The design spec and implementation plan are in `docs/specs/` and `docs/plans/`.
They were written in the original research repository, so their file paths
refer to the old `rlvr_curriculum` package and `docs/superpowers/` layout.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). On Windows and
Linux, torch is installed from the PyTorch CUDA 12.8 index. On macOS it comes
from PyPI.

```bash
uv sync
uv run pytest -q
uv run ruff check tiny_dino tests
uv run mypy tiny_dino
```

## Watch the policy play

```bash
cd web
npm ci
npm run build
cd ..
uv run tiny-dino dino-web --checkpoint artifacts/dino/deploy/model.pt --open
```

The server listens on `http://127.0.0.1:8000/` and serves `web/dist`. For
frontend development, keep the server running and run `npm run dev` in `web/`.
Vite proxies `/api` to port 8000.

## Reproduce the evidence

Re-certify the shipped checkpoint over the frozen domain:

```bash
uv run tiny-dino dino-certify --checkpoint artifacts/dino/deploy/model.pt --output runs/certify
```

Run a short arena on seeds 201 to 204, 10,000 frames each:

```bash
uv run tiny-dino dino-arena --checkpoint artifacts/dino/deploy/model.pt
```

Train a candidate from scratch. Training was run on CUDA; to train on the CPU,
set `"device": "cpu"` in the config.

```bash
uv run tiny-dino dino-train --config configs/dino-train-mlp96.json --corpus configs/dino-corpus.json --output runs/mlp96
uv run tiny-dino dino-certify --checkpoint runs/mlp96/model.pt --output runs/mlp96
```

A candidate is selected only if it passes both certification and the soak
suite. The deployed checkpoint came out of a longer search over widths and DAgger
rounds. A fresh run is not guaranteed to reproduce it byte for byte.
`artifacts/dino/deploy/gpu-training.json` records one retrained CUDA candidate
that passed certification but failed the soak suite, and was rejected.

## Repository layout

```
tiny_dino/            engine, contract, teacher, training, certification, web server
tests/                unit tests and the deployment evidence-chain test
configs/              corpus and training configs
artifacts/dino/deploy certified checkpoint and its certification, soak, and latency records
reports/dino/         evidence summary (Markdown and JSON)
docs/                 design spec, plan, and provenance record
web/                  React client
```

## Provenance

The engine reimplements physics constants and collision semantics from
[laya-vs-jev](https://github.com/virajbhartiya/laya-vs-jev) (Apache-2.0,
commit `2854178`). That code is itself a port of the Chromium offline dinosaur
game (BSD-3-Clause). The assisted gameplay in that repository was not used as
evidence of model skill. The details are in `docs/dino-provenance.md`, and the
licence notices are in `NOTICE`.

This project was extracted from branch `worktree-tiny-dino` (commit `b9d40d6`)
of a larger research repository.

## License

MIT for the code in this repository. See `LICENSE`. Third-party notices are in
`NOTICE`, and `web/public/dino/LICENSE.chromium` covers the sprite sheet.
