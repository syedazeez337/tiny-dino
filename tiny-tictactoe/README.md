# tiny-tictactoe

A 22,281-parameter MLP that plays perfect tic-tac-toe with no search at run time.
It is the same recipe as `tiny-dino`, applied to a game with an **exact** teacher
and a **complete** certification domain.

## Result

- Model: MLP, 3 hidden layers, width 96, **22,281 parameters** (about 93 KB, FP32).
- Input: 28 features (9 cells x empty/mine/opponent, plus board fullness), encoded
  relative to the player the policy controls, so one net plays both X and O.
- Output: 9 logits, one per cell, masked to legal moves.
- Teacher: exact **minimax** (never used at run time).
- Certification domain: **all 5,478 reachable positions** (4,520 of them decisions).

Certification over the entire game:

```
decisions:     4520
optimal:       4520
non_optimal:   0
illegal:       0
perfect_play:  true
```

Because the domain is complete and the teacher is exact, the claim is stronger
than the Dino one: this policy makes a game-theoretically optimal move in **every
reachable position**, so it never loses with unlimited play.

## How it works

Same pipeline as Tiny Dino:

1. **Engine** (`ttt_engine.py`) — deterministic 3x3 rules; the source of truth.
2. **Contract** (`ttt_contract.py`) — the 28-feature vector and 9-action order.
3. **Teacher** (`ttt_teacher.py`) — exact minimax; labels the set of optimal moves.
4. **Domain** (`ttt_domain.py`) — every reachable position, hashed as a manifest.
5. **Dataset** (`ttt_dataset.py`) — teacher-labelled features for every decision.
6. **Model** (`ttt_model.py`) — a small MLP trained with a **set-valued loss**, so
   any optimal move earns full credit (many positions have several).
7. **Train** (`ttt_train.py`) — a few seconds on CPU.
8. **Certify** (`ttt_certify.py`) — checks the frozen policy on every position.

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). The trained model
ships in `artifacts/`, so you can play immediately.

```bash
cd tiny-tictactoe
uv sync
uv run tiny-ttt web --checkpoint artifacts/model.pt --open
```

`--open` opens `http://127.0.0.1:8000/`. Pick a side and click a cell.

### Train and certify yourself

```bash
uv run tiny-ttt train --output artifacts          # a few seconds on CPU
uv run tiny-ttt certify --checkpoint artifacts/model.pt --output artifacts
```

### Play in the terminal

```bash
uv run tiny-ttt play --human x
```

Cells are numbered 1-9 (top-left to bottom-right). You can pre-supply your moves
with `--moves "5,1,9"`.

## Notes

- The whole study is a port of the Tiny Dino recipe; see the sibling `tiny-dino`
  project for the Dino version and its evidence chain.
- It also appears inside the shared Tiny Games browser interface: from the
  repository root, `uv run tiny-dino dino-web --checkpoint
  artifacts/dino/deploy/model.pt` serves both games on one page.
- Tic-tac-toe is public domain, so there are no licensing or provenance concerns.
