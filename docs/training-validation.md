# First Intelligence — CPU training validation

The bounded experiment used the actual DQN trainer, Classic simulation and Arena
evaluator. Initial and checkpoint-reloaded weights were evaluated greedily over
the **same environment seeds 0 through 19**. No demonstration gameplay or synthetic
learning data was substituted.

| Measurement | Result |
| --- | ---: |
| Device | CPU, one Torch intra-op thread |
| Runtime | Windows x64, Python 3.13.12, Torch 2.15.0+cpu |
| Architecture | 11 → 16 → 8 → 3, ReLU, 355 parameters |
| Initialization/training seed | 7 |
| Training episodes | 300 |
| Environment movement steps | 49,329 |
| Measured training-loop duration | 79.6925 seconds |
| Episodes/second | 3.7645 |
| Environment steps/second | 618.9919 |
| Initial evaluation mean score | 0.0 |
| Reloaded trained evaluation mean score | 3.5 |
| Initial evaluation terminations | 20 stagnation |
| Trained evaluation terminations | 14 stagnation, 6 absolute episode cap |
| Training last-100 mean score | 2.32 |
| Training best score | 11 |
| Evaluation best score | 13 |
| Training apple count | 312 |
| Training collision/stagnation counts | 283 / 17 |
| Final epsilon | 0.05 |
| Final Huber loss | 0.20319636 |
| Saved/reloaded model weights | Exactly equal |

Training timing excludes model initialization, pre/post evaluation and final
checkpoint I/O. It measures this development machine and workload, not a promised
speed on every laptop. The dashboard separately reports actual worker throughput,
including visible playback pacing, and does not label target multipliers as achieved.

## Reproduction

Use `tools/validate_training.py --episodes 300 --seed 7` in a Python environment
with `requirements-ai.txt`. It writes the complete configuration, per-seed
evaluation scores/termination reasons, actual score/loss/epsilon histories,
measurements and initial/final checkpoints under `.artifacts/v14-validation/`.
It uses the same trainer and safe checkpoint library as the application.

The experiment differs from UI defaults in these explicit fields:

```json
{
  "episodes": 300,
  "seed": 7,
  "replay_capacity": 10000,
  "epsilon_decay_steps": 10000,
  "target_sync": 250,
  "episode_cap": 1000,
  "checkpoint_interval": 100
}
```

Other values use `TrainingConfig` defaults: Adam 0.001, gamma 0.99, batch 64,
epsilon 1.0→0.05, gradient-norm clip 10, rewards +10/−10/−0.01/−5 and stagnation
100×current length. The bounded command records initial/final checkpoints; the
Training Center worker additionally performs interval and pending manual saves.

The raw trained evaluation scores, in seed order 0–19, were:

`[1, 5, 3, 1, 1, 13, 2, 0, 4, 0, 7, 1, 8, 1, 6, 2, 4, 1, 9, 1]`.

## Interpretation and limits

Mean score improved across this fixed seed set, and weights survived the
checkpoint round trip exactly. This single short experiment does not establish
robust generalization, convergence or mastery. All 20 evaluation games still
terminated through stagnation or the absolute cap rather than board completion.
The small feedforward model only sees 11 binary features, so it cannot reconstruct
the full body layout or plan long routes. Sparse apple rewards and the short run
also limit coverage. Broader seed sets, independent training seeds and architecture
comparisons belong to the next diagnostics milestone; no additional reward
shaping was introduced to improve these reported numbers.

Unit tests separately compare uninterrupted training with full-state resumption
through replay wraparound, optimization and target synchronization. Bitwise
continuation is intended for the same CPU/runtime, not arbitrary Torch versions.
