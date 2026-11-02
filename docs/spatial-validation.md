# v1.5.2 validation

All runs used this Windows checkout, Python 3.13.12, CPU Torch 2.15.0 and
Pygame 2.6.1. Neural timing used one Torch thread and one environment. Tests and
pilots used isolated profiles under `.artifacts/`; historical models were not
used as training outputs. No Git operations were performed.

## Automated checks

The complete suite passes **400 tests**. It includes all previous regression
coverage plus encoder geometry/normalization, A–D architecture compatibility,
forward/backward gradients, actual parameter counts, shared CNN initialization,
structured replay, target synchronization, exact checkpoint resumption, model-only
reload, frozen evaluation, Analysis/CSV and application UI routing.
The final full run completed in **51.64 seconds**. The bounded application startup
smoke command also exited successfully with dummy video/audio devices.

Additional compatibility checks load a format-1 model with all new registry fields
absent and compare exact Q-values/actions. Invalid IDs, schema versions, parameter
counts and action orders are rejected. Tests forbid registry lookups after
initialization while training and playing CNNs. A forward hook verifies the
actual spatial/heading tensors received by Arena. Profiling tests establish that
policy-forward counts exclude optimization forwards and random epsilon actions,
and preserve trajectories, weights and owned RNG states.

Offscreen dummy-device checks rendered Frontier A/B/C/D configuration, a completed
CNN dashboard, CNN Arena and C/D Analysis. Screenshots were inspected for metadata
alignment and readable selectors. This is not a physical-display or audio test.
No Windows executable was rebuilt in this patch; resource version metadata is
updated to 1.5.2.0.

## Legacy numerical and throughput comparison

`tools/benchmark_legacy_compat.py` compares the preserved legacy DQN/environment
source snapshot with the current A implementation. It uses seed 7, default
learning settings, 64 warmup steps and two alternating-order pairs of 4,096 timed
steps. Transition serialization/digest work is included equally on both sides;
this is a short kernel comparison, not an end-to-end worker benchmark.

| Trial | Historical steps/s | Current A steps/s | Transitions and final online weights |
| --- | ---: | ---: | --- |
| 1 | 759.73 | 757.65 | Identical |
| 2 | 722.22 | 715.79 | Identical |
| Median | **740.98** | **736.72** | Identical in both pairs |

The median difference is about **−0.6%** in this measurement. It gives no evidence
of a substantial A regression, with ordinary short-run timing uncertainty.
Raw results: `.artifacts/v152-legacy-compat.json`.

A read-only SHA-256 audit confirms **100 historical checkpoint files unchanged**.
Original `ai/network.py`, `ai/observations.py`, `ai/training_config.py`, `ai/replay.py`,
`ai/actions.py` and `core/game.py` also match their preserved hashes. The DQN and
episode adapters were extended; their numerical legacy behavior is checked above.

## Short training pilots

`tools/smoke_spatial.py` runs the actual Frontier worker in isolated subprocesses.
Seed 7, default DQN/rewards, batch 64, replay 100,000, headless execution and
profiling were used for B/C/D. The requested budget was approximately 10,000 steps,
with a **120-second training cap per model**. Pause is checked at 32-step intervals;
10,000 rounds down to 9,984. Initialization is excluded from training timing.
Timing includes worker control/presentation publication and profiling overhead.

| Model | Steps at pause | Active pilot seconds | Steps/s | Optimizer updates | Updates/s | Last finite loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B | 9,984 | 14.418 | 692.48 | 9,921 | 688.11 | .005504 |
| C | 6,912 | 120.148 | 57.53 | 6,849 | 57.00 | .112401 |
| D | 6,912 | 120.966 | 57.14 | 6,849 | 56.62 | .162064 |

The CNNs hit the time cap. No full 10,000-episode experiment was run. The worker
then finished the current episode to save a resumable boundary, adding a small
amount of training: B **10,072** total steps / 120 episodes; C **6,915** / 71;
D **6,940** / 68. Checkpoint/history counts reflect those actual boundaries.
Process peak working sets during validation were approximately 317/335/331 MiB
for B/C/D. These are process-wide peaks, not model memory or worst-case replay
capacity estimates.

All pilots collected transitions, sampled replay, updated parameters with finite
loss, saved and resumed a checkpoint, optimized again after resume, saved/reloaded
a portable model, performed two seeded frozen Arena evaluations, loaded actual
history in Analysis and exported CSV. C and D initial-weight hashes matched.
Frozen two-game checks only validate execution; they are not useful quality
estimates. Early training maxima of one apple do not support ranking the models.

Raw results and protected pilot artifacts live under:

```text
.artifacts/v152-cpu-pilots/results.json
.artifacts/v152-cpu-pilots/B/result.json
.artifacts/v152-cpu-pilots/C/result.json
.artifacts/v152-cpu-pilots/D/result.json
.artifacts/v152-cpu-pilots/<B|C|D>/experiments/<id>/checkpoints/<id>/recovery-.../
.artifacts/v152-cpu-pilots/<B|C|D>/portable/model_<B|C|D>/
.artifacts/v152-cpu-pilots/<B|C|D>/profile/evaluations/
.artifacts/v152-cpu-pilots/<B|C|D>/exports/
```

These are validation artifacts. Normal user runs save visibly under
`experiments/frontier/<id>/` and shareable model pairs under its `models/` folder.
Use **OPEN EXPERIMENT FILES** to locate a normal run. The pilot result JSON records
the exact absolute paths for each checkpoint, portable model, evaluation and CSV.

## Inference measurements and profiling scope

A separate 256-call greedy policy microbenchmark, after 32 warmup calls, measured
actual forward time on a fixed encoded initial state. It performs no simulation,
replay or optimizer work and is not an Arena game-throughput benchmark.

| Model | Policy forwards/s, forward timer only | Action selections/s, including conversion |
| --- | ---: | ---: |
| A | 12,901 | 7,901 |
| B | 15,233 | 8,938 |
| C | 3,771 | 2,924 |
| D | 3,804 | 2,951 |

Raw result: `.artifacts/v152-inference.json`. These very short CPU timings carry
measurement noise and do not establish that B inference is intrinsically faster
than A. The pilot result files' inclusive action-selection timer is labeled
`neural_inference`; do not interpret its per-step calls as actual neural forwards.
The shipped profiler reports these separately as `action_selection` and
`neural_inference`, tested against actual calls. Optimization updates/s in the
pilot table use active pilot wall time. Serialization happens after the paused
timing snapshot and is not included in those throughput values.

## Files changed or added

- Registries/encoders/networks: `ai/encoders.py`, `ai/model_registry.py`,
  `ai/registered_networks.py`.
- Runtime and storage: `ai/environment.py`, `ai/dqn.py`, `ai/persistence.py`,
  `ai/checkpoints.py`, `ai/frontier.py`, `ai/frontier_store.py`, `ai/arena.py`,
  `ai/evaluation.py`, `ai/profiling.py`, `ai/experiments.py`.
- Analysis: `ai/analysis_stats.py`, `ai/history.py`, `ai/analysis_repository.py`,
  `ai/analysis_export.py`.
- Presentation: `ui/frontier_view.py`, `ui/arena_view.py`, `ui/analysis_view.py`,
  `ui/neural_view.py`.
- Tools: `tools/export_models.py`, `tools/smoke_spatial.py`,
  `tools/benchmark_legacy_compat.py`.
- Tests: `tests/test_spatial_foundations.py`, `tests/test_spatial_integration.py`,
  `tests/test_frontier_profiling.py`, `tests/test_challenge_rework.py`.
- Release/documentation: `core/config.py`, `packaging/windows_version.txt`,
  `README.md`, `CHANGELOG.md`, `models/README.md`, `docs/spatial-architecture.md`,
  `docs/ai-research.md`, this report and `docs/releases/v1.5.2.md`.

## Limits and next release

CPU, single environment and fixed 25×25 spatial observations are intentional.
Replay materialization and convolutional backward passes cost substantially more
than the original MLP. No policy quality claim is made. Exact resumption is at
episode boundaries in a matching runtime. Old unrecorded metrics stay unavailable.
Legacy neuron inspection and its Analysis Experiment A helper remain restricted
to the 11-input laboratory; Frontier handles registered spatial experiments.

The architecture/encoder separation, batched network inputs, structured replay,
versioned persistence and shared evaluation path provide foundations for v1.5.3.
Parallel execution and its synchronization/scaling work remain to be implemented.
