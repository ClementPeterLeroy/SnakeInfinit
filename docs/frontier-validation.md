# v1.5.1 — Frontier validation and benchmark report

This release adds an execution/presentation pipeline, not a new learner. The original
`ai/dqn.py`, `ai/environment.py`, `ai/observations.py` and default training hyperparameters
remain unchanged. Source models and existing baseline experiments are not migrated.

## Controlled benchmark

The actual Legacy `TrainingWorker` and new `FrontierWorker` ran sequentially in fresh
processes with isolated settings, histories and experiment directories. Both used
CPU with one Torch intra-op thread, one Classic environment, 11→16→8→3 ReLU networks
(355 parameters), seed 7 and identical initial weights. All learning defaults were
the same: learning rate .001, gamma .99, batch 64, replay 100,000, epsilon 1→.05 over
100,000 steps, target sync every 1,000 and the existing reward/termination settings.

The episode ceiling was raised to 1,000,000 solely so the benchmark could stop at
exactly **100,000 environment steps**, including a final partial episode. It did not
train to that episode ceiling. Each run paused at the exact same step budget. Legacy
ran headless; Frontier refreshed every 100 episodes. Recovery used 500 episodes to
match Legacy's configured checkpoint cadence; Frontier also produced its 1,000-episode
lightweight milestone. Normal Frontier defaults remain 10,000 episodes/recovery 1,000.

Runtime: Windows x64, Python 3.13.12, Torch 2.15.0+cpu. No concurrent agent benchmark
or test workload ran during the pair. Other applications/OS activity were not
controlled, so this is a single paired observation, not a confidence interval or
hardware-independent claim. The historical 19,413-second baseline is not used to
calculate the measured difference.

| Measurement | Legacy | Frontier |
| --- | ---: | ---: |
| Actual environment steps | 100,000 | 100,000 |
| Actual Adam updates | 99,937 | 99,937 |
| Execution wall time | 170.162 s | 157.703 s |
| Environment steps/s | 587.676 | 634.105 |
| Optimizer updates/s | 587.306 | 633.706 |
| Total worker wall time | 172.769 s | 160.246 s |
| Peak process RAM (Windows working set) | 396.4 MiB | 395.7 MiB |
| Completed episodes | 1,021 | 1,021 |
| Last-100 mean score | 0.44 | 0.44 |
| Best score | 4 | 4 |
| Apples | 219 | 219 |
| Collisions / stagnations | 1,010 / 11 | 1,010 / 11 |
| Final epsilon | 0.05 | 0.05 |
| Final Huber TD loss | 0.00025167746935 | 0.00025167746935 |

Execution timing starts at the first training step and ends at acknowledged pause,
including worker bookkeeping, journals, automatic saves and pause aggregation.
Total wall time also includes worker initialization. Python/Torch module imports
before worker construction are excluded. RAM includes the process and Torch, not
just model parameters. Throughput is measured from wall time, not viewing multipliers.

Frontier measured **7.9% higher throughput in this pair**. This is a modest measured
difference, not a statistically established speed guarantee. The validation also
checked:

- Identical initial online weights.
- Identical SHA-256 digest of all 100,000 complete transition objects.
- Identical final online and target parameter bytes.
- Identical actual Adam-update count and final learning metrics.

No numerical/behavioral differences were observed in those checks. Timing and RAM
differences do not imply a change in policy quality. Frontier published 12 snapshots
(initial, episode intervals and explicit pause), while recording every completed episode.

## Profiling

A separate opt-in 4,096-step Frontier sample used the same seed/network/configuration.
It is excluded from the unprofiled speed comparison. Execution took 7.016 seconds
(9.692 seconds including worker initialization), with 4,033 optimizer updates.
Exclusive timers avoid counting nested encoding, sampling and optimizer time twice.

| Phase | Exclusive seconds |
| --- | ---: |
| Forward/backward, tensor preparation, loss/gradient checks | 5.021 |
| Optimizer updates | 1.328 |
| State encoding | 0.185 |
| Replay sampling | 0.157 |
| Environment stepping | 0.113 |
| Neural inference/action selection | 0.032 |
| Metrics aggregation | 0.0013 |

Serialization recorded zero calls because this short sample did not reach a save
milestone. Serialization/UI timers are installed and report calls on longer runs;
zero is not an estimate of their cost. The remaining time includes orchestration,
record handling and profiling overhead. Forward/backward and optimizer phases used
about 90% of measured active wall time in this sample. The next optimization should
measure minibatch tensor construction and small CPU update overhead before choosing
storage/batching changes; any numerical effect must be validated. Parallel environments,
batched inference and CUDA are not implemented here.

Reproduce in a separate terminal without changing production profile locations:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_frontier.py --steps 100000
.\.venv\Scripts\python.exe tools\benchmark_frontier.py --mode frontier --steps 4096 --profiling
```

Each invocation uses a new ignored `.artifacts/` output directory. The comparison
report records full configuration, metric values, hashes, actual time, peak RAM and
optimizer-update counts. Benchmark capture hooks are diagnostic-only and identical
in both runs; they are not installed in normal application training.

## Tests and compatibility

The complete suite passed **371 tests in 50.58 seconds**. It preserves the existing
354 cases and adds Frontier execution, UI, profiling and failure-recovery coverage.
The 13 worker/artifact tests also passed a dedicated run (18.68 seconds), including
terminal shutdown status and save-retry behavior.

The final integrity audit matched **all 100 protected baseline checkpoint-file
SHA-256 hashes**. The original DQN, observation encoder, environment/reward rules
and training-default source files also match their pre-development hashes. No
production process was stopped or signaled, and no existing model weights were changed.

Automated coverage includes configuration rejection/defaults, headless execution,
actual seed/reason/optimizer recording, refresh frequency, real milestone 1,000,
nonstandard final budgets, automatic final publication, model inference, pause/resume,
boundary save/stop, numerical continuation, rotating retention, interrupted serialization,
exclusive experiment directories and failed final-save retry.

Actual Pygame tests exercise the Legacy/Frontier menu, advanced configuration,
cached Dashboard, shared Analysis, CSV-compatible records, checkpoint discovery,
frozen evaluation associations, Arena and missing-Torch fallback. Existing v1.4/v1.5
checkpoint tests and all Legacy/gameplay regression cases are retained. Tests use
temporary fixture stores with `LOCALAPPDATA`, `APPDATA` and `XDG_DATA_HOME` redirected
to workspace artifacts; no destructive tests target production models.

Configuration, menu and completed Dashboard renders were generated and visually
inspected with dummy SDL at 1280×720. This validates layout in an offscreen render;
actual desktop interaction, audible playback and subjective laptop smoothness have
not been manually verified. No five-hour training run was used for acceptance.

## Files and storage

New runtime modules are `ai/frontier_config.py`, `ai/frontier.py`,
`ai/frontier_store.py`, `ai/profiling.py`, `services/checkpoint_catalog.py` and
`ui/frontier_view.py`. Integration changes are in `app.py`, `ui/menu.py`,
`ai/checkpoints.py`, `ai/analysis_repository.py`, `tools/export_models.py`, version
metadata, README/changelog and ignore rules. New tests are `test_frontier.py`,
`test_frontier_ui.py` and `test_frontier_profiling.py`. The bounded benchmark tool is
`tools/benchmark_frontier.py`.

Default source artifact location:

```text
<project>/experiments/frontier/<experiment-id>/
```

Lightweight `.pt`/JSON models are under `models/`. Canonical complete checkpoint
folders are under `checkpoints/<experiment-id>/`, and `history/history.sqlite3`
retains all committed episodes and optimizer diagnostics. Experiment manifests
record configuration, architecture, seeds, parent, policy and artifact paths. A
small `frontier_index/` in the existing settings profile makes these new libraries
discoverable to Legacy Analysis/Arena without copying or modifying the baseline.
Generic experiment/evaluation association metadata remains in that profile too.

Use **OPEN EXPERIMENT FILES** or the Configuration output path to locate files.
Only `.pt` plus matching model JSON is needed for a model-only publication. Exact
training resume requires the entire canonical checkpoint folder; the descriptive
`..._checkpoint.pt` is a training-state alias, not a standalone resumable bundle.
See [experiment file guide](../experiments/README.md) for loading and sharing.

## Limits, manual checks and v1.5.2 readiness

- One CPU environment and the existing 11-input DQN only. No CUDA, CNN, vectorized
  environment, new observation experiment or Benchmark Laboratory UI is present.
- Exact resumption is tested at episode boundaries under the same runtime, not in
  the middle of a partially completed episode or across arbitrary Torch versions.
- Up to 24 newly completed episodes can remain buffered until a flush; orderly
  pause/save/refresh/completion/shutdown flush them. Recovery always flushes first.
- A recording failure stops with an error instead of silently dropping research
  data. Save failures retain in-memory state, are visible and can be retried.
- Dashboard charts retain bounded display windows; full raw history remains on disk.
  Process CPU/RAM includes other work in the same app. Unavailable measurements are N/A.
- Writer locks and atomic metadata protect normal interrupted-save cases, not every
  hardware, disk-full, filesystem or power-loss failure. Only owned new recovery
  folders are eligible for rotation.
- The benchmark is one pair. Repeat under controlled hardware/load before making
  broader performance claims or changing the learning update/experience ratio.

Manually open Training Center, verify both workflows, run a small disposable Frontier
budget, inspect all four charts, pause/resume, request a boundary save/stop, open the
visible model directory, load its snapshot in Arena and compare/export its training
and seeded evaluation data in Analysis. Check real display scaling and existing audio
separately. Do not use the completed baseline as a destructive test fixture.

The worker/presentation separation, measured update counts, owned seed records,
artifact isolation, compatible readers and controlled benchmark establish a usable
foundation for v1.5.2. Future parallelization still needs explicit validation of seed
streams, replay ordering and update-to-experience ratios; those features are not claimed.
No staging, commits, pushes, tags, releases or GitHub operations were performed.
