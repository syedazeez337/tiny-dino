# Tiny Games

Two small neural policies that play classic games. Each one is distilled from an
exact teacher and certified over a finite domain. At run time there is no
planner, solver, or safety shield — every action is a single forward pass of a
roughly 90 KB model.

| Game | Policy | Checkpoint | Teacher | Certification |
|---|---|---|---|---|
| **Tiny Dino** | MLP, 96 wide, 3 layers, 21,699 params | 90,929 B, FP32 | 90-frame look-ahead | 5,400 frozen states, 0 unsafe; 1,000,000-frame soak, 0 crashes |
| **Tiny Tic-Tac-Toe** | MLP, 96 wide, 3 layers, 22,281 params | 93,233 B, FP32 | exact minimax | every reachable position (4,520 decisions): perfect play |

## The shared recipe

Both games are built the same way:

1. A cheap, deterministic engine.
2. An exact teacher that can be brute-forced — used only to label data and to
   certify, never at run time.
3. A compact, fixed-length state vector (tens of numbers, no pixels).
4. A small MLP trained by imitating the teacher's set of acceptable actions.
5. Exhaustive certification over a finite domain, plus a soak suite.

The intelligence is baked in during training. At run time the policy is just a
mapping from the state vector to an action, computed in well under one frame.

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- Node.js 20.19 or later — for the browser interface

## Play both games

One server hosts both games in a single browser interface (light mode). From the
repository root:

```bash
uv sync

cd web
npm ci
npm run build
cd ..

uv run tiny-dino dino-web --checkpoint artifacts/dino/deploy/model.pt --open
```

`--open` opens `http://127.0.0.1:8000/`. The top bar switches between **Dino**
and **Tic-Tac-Toe**:

- **Dino** — press **Start run**. Change the seed, pause, stop, or replay a crash.
  Full details, including how to retrain and re-certify, are in
  [docs/dino.md](docs/dino.md).
- **Tic-Tac-Toe** — pick X or O and click a cell. The model never loses; your
  best result is a draw. Full details are in
  [tiny-tictactoe/README.md](tiny-tictactoe/README.md).

The tic-tac-toe checkpoint is served from `tiny-tictactoe/artifacts/model.pt` by
default; override it with `--ttt-checkpoint PATH`. Tiny Tic-Tac-Toe can also run
standalone, from `tiny-tictactoe/`:

```bash
uv sync
uv run tiny-ttt web --checkpoint artifacts/model.pt --open
```

## Check the claims yourself

Tiny Dino (from the repository root):

```bash
uv run tiny-dino dino-certify --checkpoint artifacts/dino/deploy/model.pt --output runs/certify
uv run tiny-dino dino-arena --checkpoint artifacts/dino/deploy/model.pt
```

Tiny Tic-Tac-Toe (from `tiny-tictactoe/`):

```bash
uv run tiny-ttt certify --checkpoint artifacts/model.pt --output artifacts
uv run tiny-ttt train --output artifacts          # retrain from scratch, a few seconds
```

## Repository layout

```
tiny_dino/         Tiny Dino policy, engine, teacher, training, and web server
tests/ configs/ artifacts/ reports/ web/   Tiny Dino supporting files
docs/dino.md       Full Tiny Dino documentation
tiny-tictactoe/    Tiny Tic-Tac-Toe, a self-contained subproject (package tiny_ttt)
```

Tiny Dino lives at the repository root; Tiny Tic-Tac-Toe is an independent
subproject in `tiny-tictactoe/` with its own `pyproject.toml`. The browser
interface serves both from one server; the standalone commands above run each
game on its own.

## License

MIT for the code in this repository. See [LICENSE](LICENSE); third-party notices
are in [NOTICE](NOTICE).
