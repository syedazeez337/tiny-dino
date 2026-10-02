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

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- Node.js 20.19 or later, only for building the web app
- No GPU needed. Every command runs on the CPU. An NVIDIA GPU only speeds up
  training.

On Windows and Linux, `uv sync` installs the CUDA 12.8 build of PyTorch. That
build also runs on machines without a GPU. On macOS, PyTorch comes from PyPI.

## Quick start

Clone, install, and watch the shipped model play:

```bash
git clone https://github.com/syedazeez337/tiny-dino.git
cd tiny-dino
uv sync

cd web
npm ci
npm run build
cd ..

uv run tiny-dino dino-web --checkpoint artifacts/dino/deploy/model.pt --open
```

`--open` opens `http://127.0.0.1:8000/` in your browser. Press **Start run**.
Change the seed to get a different course, and use **Pause**, **Stop**, and
**Replay** (the last two seconds at half speed). The game only advances while its
tab is visible, the same as Chrome's own Dino game.

Stop the server with `Ctrl+C`. If you skip `npm run build`, the server still
starts but serves only the JSON API under `/api/`.

## Check the claim yourself

Each command below uses the shipped checkpoint. The times were measured on a
laptop and are only a rough guide.

Re-certify the model over all 5,400 frozen states (about 30 seconds). This
writes `runs/certify/certification.json`. It should print `PASS` and match
`artifacts/dino/deploy/certification.json` field for field.

```bash
uv run tiny-dino dino-certify --checkpoint artifacts/dino/deploy/model.pt --output runs/certify
```

Play seeds 201 to 204 for 10,000 frames each without the browser (under a
minute). Expect `'crashes': 0` and `'runtime_interventions': 0`.

```bash
uv run tiny-dino dino-arena --checkpoint artifacts/dino/deploy/model.pt
```

Run the test suite (about 6 minutes). This includes a test that ties the
checkpoint hash to the certification, soak, and evidence files.

```bash
uv run pytest -q
```

## Train your own model

`dino-train` builds the teacher-labelled corpus, adds teacher rollouts on the
DAgger seeds, trains, and writes `model.pt` to `--output`. Certify the result
before you trust it.

```bash
# With an NVIDIA GPU (about 5 minutes, mostly spent building the corpus)
uv run tiny-dino dino-train --config configs/dino-train-mlp96.json --corpus configs/dino-corpus.json --output runs/mlp96

# CPU only (about 20 minutes)
uv run tiny-dino dino-train --config configs/dino-train-mlp96.json --corpus configs/dino-corpus.json --device cpu --output runs/mlp96

uv run tiny-dino dino-certify --checkpoint runs/mlp96/model.pt --output runs/mlp96
uv run tiny-dino dino-arena --checkpoint runs/mlp96/model.pt
uv run tiny-dino dino-web --checkpoint runs/mlp96/model.pt --open
```

A fresh run will not reproduce the shipped checkpoint byte for byte, and
passing certification is not enough on its own. In testing, a retrained 96-wide
model passed certification and then crashed on seed 202 at frame 2,327.
`artifacts/dino/deploy/gpu-training.json` records a similar rejected
candidate. The shipped model was selected because it passed both certification
and the 1,000,000-frame soak suite.

`configs/dino-train-mlp32.json` and `configs/dino-train-mlp64.json` train the
smaller widths from the search.

## Command reference

All commands run as `uv run tiny-dino <command> [options]`.

| Command | What it does | Required options |
|---|---|---|
| `dino-web` | Serve the web app and play API | `--checkpoint` |
| `dino-certify` | Check every state in the frozen domain; write `certification.json` and `failures.csv` to `--output` | `--checkpoint` |
| `dino-arena` | Play seeds 201 to 204 for 10,000 frames each and print the results | `--checkpoint` |
| `dino-train` | Build the corpus, add teacher rollouts, and train a candidate | `--config`, `--corpus` |
| `dino-corpus` | Build the teacher-labelled corpus only (about 160,000 examples, several minutes) | `--config` |
| `dino-calibrate` | Time CPU inference for the architecture in a training config. It also rebuilds the corpus first, so it takes several minutes. | `--config`, `--corpus` |

Options:

| Option | Default | Used by |
|---|---|---|
| `--checkpoint PATH` | none | `dino-web`, `dino-certify`, `dino-arena` |
| `--output DIR` | `runs/latest` | `dino-certify`, `dino-train`, `dino-corpus`, `dino-calibrate` |
| `--config PATH` | none | training commands. Use a `configs/dino-train-*.json` file, or `configs/dino-corpus.json` for `dino-corpus`. |
| `--corpus PATH` | `--config` | `dino-train`, `dino-calibrate`: corpus spec, normally `configs/dino-corpus.json` |
| `--device {auto,cpu,cuda}` | from config | `dino-train`: overrides the config's device |
| `--expert-frames N` | `20000` | `dino-train`: teacher rollout frames per DAgger seed |
| `--host HOST` | `127.0.0.1` | `dino-web` |
| `--port PORT` | `8000` | `dino-web` |
| `--web-dir DIR` | `web/dist` | `dino-web`: serve a different frontend build |
| `--open` | off | `dino-web`: open a browser tab |

## Development

```bash
uv run pytest -q
uv run ruff check tiny_dino tests
uv run mypy tiny_dino

cd web
npm run typecheck
npm run lint
npm run test
npm run build
```

For frontend work, keep `dino-web` running and run `npm run dev` in `web/`.
Vite proxies `/api` to port 8000.

## Troubleshooting

- **`the config asks for CUDA but none is available`**: add `--device cpu` to
  `dino-train`.
- **`dino-web` exits with a bind error**: another process holds port 8000.
  Pass `--port 8001` to `dino-web`.
- **The page loads but the game does not move**: browsers pause animation in
  background tabs. Bring the tab to the front.
- **`incompatible Dino checkpoint`**: the checkpoint was built with a different
  engine or feature version than this code. Train a new one.

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
