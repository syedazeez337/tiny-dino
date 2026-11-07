# Tiny Dino Policy Design

## 1. Objective

Build a small, genuinely unassisted neural policy that plays a deterministic,
Chromium-compatible Dino runner. Preserve the existing Tiny Nim implementation
and visual language as a historical baseline. Adapt its Play / Arena / Proof
product structure to Dino without allowing a planner, solver, or heuristic to
alter the model's live actions.

The project must distinguish measured bounded certification from the impossible
claim that a policy has been mathematically proved to survive an endless game
forever.

## 2. Success Criteria

The deliverable is successful when one frozen deployment checkpoint:

1. Chooses every live action itself from the structured game state.
2. Produces zero unsafe actions over the frozen bounded certification domain.
3. Produces zero crashes across the preregistered held-out seed suite and soak
   runs.
4. Uses zero runtime planner interventions or substituted actions.
5. Runs locally at substantially less than one 60 FPS frame of latency on the
   target Windows machine.
6. Remains a small specialist. Candidate selection prioritizes the smallest
   model that passes all gates, not a predetermined byte limit.
7. Exposes reproducible evidence: engine revision, scenario manifest, dataset
   manifest, checkpoint SHA-256, parameter count, artifact size, latency, and
   every failing counterexample.

If no candidate passes, the project reports the measured failure and retains
the best checkpoint as experimental. It must not weaken the certification gate
or silently add a safety shield.

## 3. Preserved Baseline

The existing branch at commit `087e873` contains the bounded-Nim specialist,
FastAPI API, React interface, arena, and exhaustive proof workflow. Dino work
must occur on a new isolated branch/worktree based on that commit. Existing Nim
code, artifacts, reports, and checkpoint hashes remain reproducible.

The reference repository is research input, not the implementation base:

- Repository: `https://github.com/virajbhartiya/laya-vs-jev.git`
- Useful elements: pinned Chromium-compatible physics, fixed-step execution,
  planner concepts, course generation, telemetry, crash replay, and tests.
- Rejected element: assisted gameplay as evidence of independent model skill.
- Any adapted Chromium code or assets retain the applicable BSD-3-Clause
  notice; adapted repository code retains its original attribution and license.

Before implementation, record the exact reference commit used. Do not vendor
model weights or copy code without license and provenance review.

## 4. Non-Goals

- Automating `chrome://dino` as the training or certification environment.
- Claiming infinite survival from a finite run.
- Using Laya, Jev, an LLM, or a cloud API during gameplay.
- Adding authentication, databases, leaderboards, analytics, chat, social
  sharing, or unrelated UI frameworks.
- Hiding failures with retries, action substitution, collision suppression, or
  planner intervention.
- Replacing the existing Nim model or rewriting unrelated research code.

## 5. Architecture

The system has six bounded units:

1. **Engine** — deterministic, fixed-step Dino physics and collision logic.
2. **Teacher** — uses exact forward simulation to return the complete set of
   acceptable actions for a state. It is unavailable to the live policy path.
3. **Dataset generator** — produces broad states, trajectory states, boundary
   perturbations, and counterexamples with manifests and stable hashes.
4. **Policy** — a small neural classifier over structured state features.
5. **Certification and arena** — evaluate the frozen deployment artifact
   without changing its actions.
6. **Web application** — exposes Play, Arena, Proof, and replay through the
   existing FastAPI and React stack.

Core game rules, feature extraction, action semantics, and collision decisions
have one Python source of truth. The browser consumes serialized frames and
telemetry; it does not reimplement authoritative physics.

## 6. Engine and Reproducibility

Implement a local engine pinned to a recorded Chromium source revision. It must
advance with deterministic 60 FPS fixed steps and expose explicit reset, step,
snapshot, restore, and collision interfaces.

The engine state includes at least:

- speed, acceleration state, distance, and frame index;
- Dino vertical position, vertical velocity, grounded/jumping/falling state,
  duck state, and current control state;
- active obstacle type, position, width, height, and collision boxes;
- the next obstacle when present;
- RNG state and course seed.

Differential fixtures compare selected trajectories and collision outcomes with
the pinned reference implementation. The engine must support faster-than-real-
time headless execution and exact replay from a serialized initial state plus
actions.

## 7. State and Action Contract

### State features

The first policy uses a fixed normalized vector rather than pixels. It includes:

- current speed;
- Dino y-position and vertical velocity;
- grounded, rising, falling, ducking, and control-state flags;
- frames since the last control transition;
- nearest two obstacles' relative x, width, height, y-position, and type;
- gap between the nearest two obstacles;
- missing-obstacle masks.

Normalization constants are stored with the checkpoint. Feature extraction is
shared by training, certification, API inference, and live play.

### Actions

The policy emits probabilities for exactly three persistent control states:

- `run`: release jump and duck;
- `jump`: hold jump according to the engine's normal jump semantics;
- `duck`: hold duck and release jump.

The engine applies the selected control state unchanged. Illegal or malformed
outputs stop the run and create a policy-violation artifact.

## 8. Teacher

For each state, the teacher restores the engine snapshot and simulates every
candidate action over a bounded lookahead, including delayed consequences and
the next obstacle when relevant. It returns:

- the full set of acceptable actions;
- per-action survival result;
- minimum collision margin;
- clearance frame;
- reason for rejection;
- whether the state is outside the teacher's proven horizon.

The teacher is deterministic. Tests cover obvious cases, jump/duck boundaries,
multiple acceptable actions, two-obstacle interactions, and snapshot isolation.
Teacher disagreement or an unproven horizon is recorded and excluded from
claims until resolved.

## 9. Model Candidates

Train candidates in increasing capacity:

1. MLP: 2 hidden layers, width 32.
2. MLP: 3 hidden layers, width 64.
3. MLP: 3 hidden layers, width 96.
4. Existing-style tiny transformer as a comparison only if the MLP candidates
   fail or materially underperform.

The head has three logits. Primary training uses set-valued cross-entropy:

`loss(state) = -log(sum(probability[action] for action in acceptable_actions))`

Candidate selection is lexicographic: pass certification, pass held-out/soak
gates, then minimize deployed bytes, then minimize p95 latency. Do not select by
training accuracy alone.

After an FP32 candidate passes functional gates, evaluate INT8 post-training
quantization. If it changes any certification action, use quantization-aware
training and recertify. The shipped artifact is the exact artifact whose hash
appears in the Proof page.

## 10. Data and Training

### Initial corpus

Generate a deterministic, versioned corpus covering:

- every obstacle type and legal size;
- bird heights;
- full legal speed range;
- minimum, typical, and maximum gaps;
- single and paired obstacle sequences;
- grounded, rising, apex, falling, landing, and duck-transition states;
- dense perturbations around action-timing and collision boundaries;
- balanced acceptable-action classes.

Trivial empty-track `run` states are capped so they cannot dominate the loss.
Split by course seed and scenario family, not random rows, to prevent trajectory
leakage.

### Training stages

1. Supervised planner distillation on the initial corpus.
2. DAgger iterations: run the unassisted candidate, label its visited states,
   prioritize crashes, unsafe choices, low-margin states, and states with high
   teacher/model disagreement, then retrain on the aggregated corpus.
3. Optional RL fine-tuning only as a preregistered experiment after the
   imitation baseline is frozen. RL may not replace the supervised baseline or
   weaken its gates.

Every run records config, seeds, code commit, dataset hash, training curves,
candidate checkpoint hash, and hardware/runtime metadata.

## 11. Certification

### Frozen bounded domain

Define the exact certification manifest before selecting the final model. It
enumerates a finite lattice of speed, Dino motion/control state, obstacle type,
size, distance, gap, and relevant two-obstacle combinations. Every enumerated
state must be reachable or explicitly identified as a synthetic boundary probe.

For each state:

1. Extract features through the production path.
2. Run the frozen deployment artifact greedily.
3. Record its action and probabilities.
4. Evaluate that action with the teacher only after selection.
5. Fail on unsafe action, malformed output, nondeterminism, non-finite value, or
   feature mismatch.

### Held-out trajectories

Run a preregistered seed suite disjoint from training and DAgger seeds, including
random, minimum-gap, maximum-speed, bird-heavy, and mixed courses. Record every
frame needed to replay a failure.

### Soak and timing

Run multi-hour deterministic headless soaks and real-time browser/API smoke
tests. Measure p50/p95/p99 inference latency, dropped frames, and engine speed.
Latency is reported separately from policy correctness.

### Claim language

Permitted claim after all gates pass:

> This unassisted checkpoint produced zero unsafe choices over the complete
> frozen certification domain and zero crashes across the preregistered
> held-out and soak suites.

Prohibited claim: the endless stochastic game is mathematically solved or the
model can never crash.

## 12. API and Interface

Retain the existing FastAPI + React + TypeScript stack and visual tokens.

### Play

- Server-authoritative live game and deterministic seed.
- Canvas or SVG presentation of the Dino lane.
- Model action, probability, latency, speed, score, and obstacle count.
- Visible `UNASSISTED NEURAL POLICY` indicator.
- Pause, restart, seed, and reduced-motion-safe presentation.
- Failure stops the run and exposes an exact replay.

### Arena

- Run named frozen suites and selected seeds.
- Report scenarios, frames, obstacles, crashes, unsafe choices, and latency.
- Allow comparison of frozen model artifacts without sharing mutable state.
- Long jobs use the existing queued/polled job pattern.

### Proof

- State `Every state in the frozen certification domain checked.`
- Show domain size, held-out seeds, soak frames, crashes, unsafe actions, and
  runtime interventions (`0` by construction).
- Show architecture, parameters, deployed bytes, engine revision, manifest
  hashes, checkpoint SHA-256, device, and certification timestamp.
- Clearly separate frozen snapshots from live reruns.

### Replay

- Reconstruct frames from recorded engine state and actions.
- Show model probabilities, chosen action, teacher acceptable set, margins,
  obstacle geometry, and first failing frame.
- Replay data is diagnostic and never fed back into a claimed frozen checkpoint.

## 13. Error Handling and Integrity

- Missing or hash-mismatched checkpoint: refuse startup.
- Engine/checkpoint schema mismatch: refuse startup.
- Non-finite feature/logit/probability: stop and persist violation.
- Invalid action: stop; never substitute.
- Certification interruption: checkpoint progress and resume without counting
  incomplete cases as passed.
- Background job failure: expose status and diagnostic message without losing
  prior certified snapshots.
- Training or evaluation exceptions: preserve configs and partial artifacts;
  never overwrite completed runs.

## 14. Testing

Required gates include:

- Unit tests for physics, collision boxes, obstacle legality, action semantics,
  normalization, snapshot/restore, teacher labels, and set-valued loss.
- Property tests for deterministic replay and engine invariants.
- Differential fixtures against the pinned reference engine.
- API tests proving live moves come only from the model output.
- Mutation-style tests that force an unsafe/invalid model choice and verify the
  run stops rather than invoking the teacher.
- Dataset determinism, split isolation, and manifest-hash tests.
- Checkpoint load/export/quantization parity tests.
- React tests for Play, Arena, Proof, failures, and reduced motion.
- Rendered visual QA at desktop and mobile widths.
- Repository gates: pytest, ruff, mypy, frontend typecheck, lint, tests, build.

## 15. Orchestration and Change Control

Implementation is delegated to an autonomous coding agent using
`deepseek/deepseek-v4-flash` in `--yolo` mode inside a dedicated managed
worktree. The primary orchestrator supplies the approved spec and plan, limits
each run to one reviewable milestone, inspects diffs and artifacts, and runs
independent verification before accepting the milestone.

The agent may edit only the isolated Dino worktree. It must not push, publish,
deploy, rewrite history, alter the preserved Nim checkpoint, or claim completion
from its own prose. Implementation commits and generated experimental artifacts
remain separate. No bot or agent attribution is added to commits.

## 16. Delivery Stages

1. Provenance and deterministic engine.
2. Teacher and certification-domain generator.
3. Dataset generation and model training pipeline.
4. DAgger and candidate selection.
5. Quantization and deployment-artifact certification.
6. FastAPI integration.
7. React Play / Arena / Proof / replay adaptation.
8. Full verification, visual QA, soak tests, and evidence report.

Each stage must be independently testable and reviewed before the next stage.
