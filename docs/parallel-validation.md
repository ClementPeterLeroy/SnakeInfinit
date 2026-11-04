# v1.5.3 engineering validation

## Outcome and execution choice

Frontier supports 1/2/4/8 environments with one shared CPU learner, using ordered
vectorized or Windows-spawn collection. Auto resolves to vectorized and the
default count remains one with one optimizer update per transition. Larger counts
are selectable without silently changing the learning ratio. There is no CUDA
dependency, network/encoder change or new DQN objective.

Simulation is very cheap relative to optimization. Spawn IPC generally reduces
simulation throughput and does not consistently improve end-to-end training.
Vectorized batched collection is the practical default. Reference-policy MLP
results vary and often slow down at larger counts; CNN reference throughput stays
approximately 45–52 steps/s in this short matrix. There is no general reference-
policy speedup claim.

The explicit vector-step policy raises steps/s mainly by reducing updates per
transition. At eight vectorized environments, A measured 3,771.8 steps/s and D
381.7 steps/s. Those are different data/update budgets, not demonstrated learning
quality improvements. Equal-update results show optimizer throughput remains the
central bottleneck, especially for CNNs.

## Automated tests and compatibility

The final full suite passed **431 tests in 78.11 seconds**. Existing tests are preserved,
with the application-version assertion updated. Coverage includes all architectures,
pool independence, seeds, batched input/action ordering, independent exploration,
replay/global counters, exact episode targets, both optimizer policies, global
epsilon and target synchronization, pause/save/resume, stop/reconstruction, spawn
startup/cleanup/failure, recovery, Analysis/CSV and frozen evaluation.

Partial-checkpoint tests compare subsequent transitions, completion records,
online parameters, optimizer counters and replay RNG state exactly across A/B/C/D
and both learning ratios. Historical-checkpoint tests remove the newly added
counter fields to verify compatible inference of the old optimizer count and
numerically identical continuation. Arena frozen evaluation repeats explicit seeds
without learning. Spawn handshake tests verify no Torch or Pygame module is loaded
inside environment workers.
The executable entry point is also imported under Windows spawn semantics to
verify it does not load the application, Torch or Pygame in child processes.

Offscreen checks rendered the four-environment configuration, completed dashboard,
Arena and a 100-record Analysis history. Screenshots were inspected for labels and
alignment. A bounded startup smoke test uses dummy devices; these checks do not
establish physical-display quality or audible playback. No Windows executable was
rebuilt; source and Windows version resources identify v1.5.3.

Actual historical **Snake DQN Alpha** and **Snake DQN Frontier Distance**
10,000-episode checkpoints were loaded read-only with their original architectures.
A SHA-256 audit confirms **159 protected model/checkpoint/experiment files remain
unchanged**, along with the original network, observation/encoder, registered
network, training-default and game-engine source files. No Git operation occurred.

## Benchmark method

The matrix used Windows 11 build 26200, Python 3.13.12, CPU Torch 2.15.0, one Torch
thread, seed 7 and the existing default DQN/reward settings. It ran **160 trials**:
A/B/C/D × 1/2/4/8 environments × vectorized/spawn, with five measured categories:

- 4,096 environment-only transitions with observations and random actions, no
  inference or optimizer.
- 512 end-to-end transitions, reference update policy.
- 512 end-to-end transitions, vector-step update policy.
- Exactly 128 optimizer updates, reference policy.
- Exactly 128 optimizer updates, vector-step policy.

The 512-transition reference runs perform **449 updates** after replay warmup.
Vector-step runs perform **449 / 225 / 113 / 57 updates** at 1/2/4/8 environments.
At equal 128-update budgets, reference runs collect **191 transitions**, while
vector-policy runs collect **191 / 318 / 572 / 1,080 transitions** respectively.
Thus both equal-data and equal-update views are available; neither makes unlike
budgets equivalent.

The matrix collected **187,240 transitions** and performed **29,312 optimizer
updates**, with **zero worker failures**. Timed loops totalled **377.38 seconds**;
this excludes startup/shutdown and report-writing overhead. The script records
startup separately; no rendering, charting, checkpoint saves or disk writes occur
inside the timed training loop. The actual Frontier worker and its persistence
are exercised separately by integration tests.

These are single, short trials with no steady-state warmup beyond ordinary replay
fill. Small cases are sensitive to initialization, allocator and CPU-frequency
effects. Original one-environment reference training uses its preserved kernel;
other configurations use the ordered batch collector. Seed derivation differs
between that preserved path and the parallel path, so trajectories are not paired
quality experiments. Repeat larger budgets for robust hardware conclusions.

CPU counters include parent and children; recorded utilization ranged approximately
88–404% (100% means one core). The largest sampled parent working set was about
328 MiB; the largest sum of child process peaks was about 192 MiB. These peaks
describe this short process lifetime and need not coincide. Full 100,000-entry
replay occupancy and long episodes require separate memory measurements.

Full trial metrics (steps, updates, episodes, episode length, startup, duration,
CPU/RAM and failures) are retained in
[the benchmark CSV](benchmarks/v1.5.3-parallel.csv). Local raw JSON is
`.artifacts/v153-benchmark.json`. Reproduce with:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_parallel.py --steps 512 --updates 128 --simulation-steps 4096 --output .artifacts\parallel-benchmark-new.json
```

## Results

Simulation columns use 4,096 transitions. Training steps/s columns use 512
transitions. The final two columns compare exactly 128 optimizer updates and
therefore different numbers of collected transitions for vector policies.

### A — Original MLP

| Environments | Strategy | Simulation steps/s | Reference training steps/s | Vector-policy training steps/s | Reference updates/s (128 updates) | Vector-policy updates/s (128 updates) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | vectorized | 34,766.7 | 689.9 | 593.6 | 588.3 | 565.7 |
| 1 | multiprocessing | 7,658.1 | 543.1 | 489.5 | 478.7 | 393.5 |
| 2 | vectorized | 32,794.7 | 571.5 | 1,079.8 | 531.8 | 469.4 |
| 2 | multiprocessing | 11,219.7 | 549.5 | 1,009.7 | 356.4 | 357.0 |
| 4 | vectorized | 32,681.7 | 599.5 | 2,061.9 | 545.3 | 484.7 |
| 4 | multiprocessing | 15,306.7 | 469.1 | 1,410.0 | 455.3 | 349.2 |
| 8 | vectorized | 36,279.1 | 597.3 | 3,771.8 | 466.8 | 419.3 |
| 8 | multiprocessing | 14,115.6 | 621.7 | 2,665.3 | 437.8 | 355.7 |

### B — Distance MLP

| Environments | Strategy | Simulation steps/s | Reference training steps/s | Vector-policy training steps/s | Reference updates/s (128 updates) | Vector-policy updates/s (128 updates) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | vectorized | 29,587.7 | 525.4 | 567.4 | 441.2 | 515.6 |
| 1 | multiprocessing | 6,876.7 | 468.4 | 446.1 | 404.0 | 361.6 |
| 2 | vectorized | 29,275.8 | 548.8 | 797.6 | 346.3 | 315.2 |
| 2 | multiprocessing | 9,290.4 | 467.8 | 850.8 | 425.3 | 372.9 |
| 4 | vectorized | 32,097.6 | 582.7 | 1,877.7 | 459.4 | 454.3 |
| 4 | multiprocessing | 16,111.6 | 530.9 | 1,570.8 | 362.6 | 255.5 |
| 8 | vectorized | 30,264.7 | 462.9 | 2,880.9 | 378.4 | 304.5 |
| 8 | multiprocessing | 17,324.1 | 463.5 | 2,805.2 | 502.9 | 341.1 |

### C — Binary CNN

| Environments | Strategy | Simulation steps/s | Reference training steps/s | Vector-policy training steps/s | Reference updates/s (128 updates) | Vector-policy updates/s (128 updates) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | vectorized | 57,559.4 | 45.6 | 46.4 | 43.5 | 45.3 |
| 1 | multiprocessing | 6,280.1 | 50.7 | 49.1 | 43.1 | 44.7 |
| 2 | vectorized | 58,981.8 | 52.3 | 102.1 | 43.7 | 45.8 |
| 2 | multiprocessing | 11,656.6 | 49.6 | 100.9 | 43.0 | 43.4 |
| 4 | vectorized | 52,697.7 | 46.7 | 191.4 | 44.4 | 42.5 |
| 4 | multiprocessing | 16,320.0 | 49.2 | 193.3 | 43.2 | 45.0 |
| 8 | vectorized | 62,671.8 | 51.5 | 347.9 | 44.3 | 43.0 |
| 8 | multiprocessing | 17,102.0 | 51.1 | 367.8 | 44.8 | 42.2 |

### D — Ordered CNN

| Environments | Strategy | Simulation steps/s | Reference training steps/s | Vector-policy training steps/s | Reference updates/s (128 updates) | Vector-policy updates/s (128 updates) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | vectorized | 59,776.7 | 49.1 | 51.1 | 44.2 | 45.7 |
| 1 | multiprocessing | 6,837.3 | 50.6 | 51.7 | 44.1 | 44.2 |
| 2 | vectorized | 53,101.4 | 51.0 | 101.1 | 43.7 | 44.5 |
| 2 | multiprocessing | 11,893.8 | 51.7 | 101.4 | 43.1 | 44.0 |
| 4 | vectorized | 57,742.8 | 51.6 | 193.9 | 43.6 | 42.7 |
| 4 | multiprocessing | 16,402.4 | 45.3 | 186.1 | 42.5 | 42.6 |
| 8 | vectorized | 60,310.1 | 50.2 | 381.7 | 42.6 | 44.7 |
| 8 | multiprocessing | 19,938.8 | 49.6 | 376.8 | 44.1 | 43.3 |

## Files changed

- New collection/learner modules: `ai/environment_pool.py`, `ai/parallel.py`.
- Execution and accounting: `ai/frontier_config.py`, `ai/frontier.py`, `ai/dqn.py`,
  `ai/profiling.py`, `ai/worker.py`.
- Checkpoints/models: `ai/checkpoints.py`, `ai/frontier_store.py`.
- Histories/exports: `ai/analysis_stats.py`, `ai/history.py`, `ai/analysis_export.py`.
- Presentation/time: `ui/frontier_view.py`, `services/training_time.py`.
- Startup/release resources: `main.py`, `core/config.py`,
  `packaging/windows_version.txt`.
- Tests/tools: `tests/test_parallel.py`, `tests/test_challenge_rework.py`,
  `tools/benchmark_parallel.py`.
- Documentation: `README.md`, `CHANGELOG.md`, `models/README.md`,
  `docs/ai-research.md`, `docs/spatial-architecture.md`,
  `docs/parallel-training.md`, `docs/parallel-validation.md`,
  `docs/benchmarks/v1.5.3-parallel.csv`, `docs/releases/v1.5.3.md`.

## Persistence, limitations and next priorities

Parallel checkpoints extend network formats 1/2 with a separate versioned collector
state, actual optimizer counter, deterministic seeds/action RNGs and partial
action traces. Reconstructing those traces restores quiescent partial episodes
without running optimizer updates. Model-only snapshots remain frozen inference
artifacts. Existing historical single-environment recovery is preserved; parallel
recovery belongs in Frontier. Metadata visibility is atomic, and failed writes do
not replace a previously published checkpoint pointer.

Scheduling is synchronous and ordered, with no asynchronous collection/learning
overlap. More environments do not guarantee more optimizer updates per second.
Identical seeds do not promise identical training across worker counts or runtime
versions. A failed collection batch may consume exploration draws before failure;
recovery does not claim identity with a failure-free run. Storage or numerical
corruption can prevent recovery and is reported. Long partial traces take longer
to reconstruct. Closed workers retain their last usage sample.

No full training-quality study, full 500-game historical reevaluation, CUDA test,
frozen executable build or long-duration soak test was performed. Historical
quality results are preserved references, not newly reproduced measurements.

v1.5.4 should profile CNN optimization and tensor materialization, measure CPU/GPU
crossover and transfer overhead, evaluate Torch threading, and add device selection
only where justified. Async pipelining or better IPC should be considered only if
profiling shows collection is limiting end-to-end learning. Benchmark Laboratory
and neural research remain later roadmap work.

Proposed manual commit title: **v1.5.3 — Parallel Frontier Training**. No staging,
commit, push, tag or release was performed.
