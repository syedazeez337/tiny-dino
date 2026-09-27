# Tiny Dino evidence

Status: **PASS**.

The selected checkpoint is a 21,699-parameter MLP stored in 90,929 bytes. CPU
inference measured 0.0484 ms p50, 0.0545 ms p95, and 0.0709 ms p99, comfortably
inside one 60 FPS frame. The live runtime applies its greedy `run`, `jump`, or
`duck` output unchanged and records zero runtime interventions.

The production training path uses CUDA with batch size 4,096. A full measured
optimization run took 16.477 seconds and peaked at 784,314,368 bytes of VRAM on
the RTX 3050. Candidate selection still uses certification and soak results,
not training device or training loss.

Measured release gates:

- Frozen actionable domain: 5,400/5,400 safe actions, zero exclusions and zero
  unsafe choices.
- Held-out and soak suite: 1,000,000 frames across seeds 201–220, zero crashes.
- Checkpoint SHA-256: `b4cd918bb5fae7d0a826ef23a055cbe05e70505f3487243e43d0919813f11109`.
- Domain SHA-256: `1c8205c10fa9bbad56ed0998544261d04144ba7040ecafac0e16720830029bb4`.
- Engine revision: `dino-engine-v2`.
- INT8 PTQ was evaluated but changed 36 certification actions, so the 90 KB
  FP32 checkpoint remains the deployment artifact.

This unassisted checkpoint produced zero unsafe choices over the complete frozen
certification domain and zero crashes across the preregistered held-out and soak
suites. This is a bounded empirical claim; it does not claim that an endless
stochastic game is mathematically solved.
