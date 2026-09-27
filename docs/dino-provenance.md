# Tiny Dino Policy: Provenance Record

Recorded on 2026-09-22 for the Tiny Dino policy study. This record is the
machine-readable pin that later milestones read (`docs/dino-provenance.json`)
and the human-readable licensing summary for the reference repository.

## Reference pin

| Field | Value |
|---|---|
| Reference repository | `https://github.com/virajbhartiya/laya-vs-jev.git` |
| Reference commit | `28541781c863fcdf20348e5bb3a2073d672fffe6` (`main`) |
| Pinned source path | `laya_mlx/trex/engine.py` |
| Pinned source SHA-256 | `e73cd1902c9195723367416a6b2eb8e0bacdcaf13b145fc38c6f2876a845357d` |
| License verdict | **adapted** |
| Checkpoint SHA-256 | (empty on a clean checkout) |

The reference repository is research input, not the implementation base. The
Dino engine is written fresh against the pinned physics semantics; no model
weights are vendored and no reference file is copied wholesale.

## Adapted elements

Each element below is drawn from the pinned `laya_mlx/trex/engine.py` and
reimplemented in this project:

- fixed-step 60 FPS execution loop;
- Chromium T-Rex physics constants: speed, acceleration, gravity, jump velocity;
- obstacle geometry, sizes, and bird heights;
- two-stage axis-aligned collision-box semantics.

## Rejected elements

- **assisted gameplay** as evidence of independent model skill. The live policy
  chooses every action itself from structured state features; no planner, solver,
  or heuristic may alter a live action.

## Licensing

- The reference repository is licensed **Apache-2.0**. Adapted repository code
  keeps its original attribution and license.
- The T-Rex demo in that repository is a Python port of the Chromium offline
  dinosaur game; its physics constants descend from Chromium. Chromium code and
  assets are licensed **BSD-3-Clause**. The applicable BSD-3-Clause notice is
  reproduced in the repository-level `NOTICE` file, as required when adapting
  Chromium-derived code or assets.

## Nim preservation baseline

The study was developed inside a larger research repository that also held a
bounded-Nim specialist. There, `docs/superpowers/nim-baseline.sha256` and
`tests/test_nim_baseline.py` pinned the Nim modules and configs as of commit
`087e873` so the Dino work could not change them. The `checkpoint_sha256` field
in `dino-provenance.json` is the digest of that Nim checkpoint, not of the Dino
policy. The Nim code is not part of this repository, so the guard was dropped
when the Dino study was extracted. The deployed Dino checkpoint digest is
recorded in `reports/dino/evidence.json`.
