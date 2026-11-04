# Changelog

## v1.5.3b — Recovery Repairs & Model Library

- Refuse unsafe partial-batch recovery checkpoints, validate resumable traces,
  preserve the previous verified recovery and report recovery status truthfully.
- Persist every committed completion exactly once on failure, with idempotent
  history flushing and an Analysis-readable emergency journal for SQLite failures.
- Add the shared Alpha / Distance / CNN / oCNN Model Library with immutable
  generation/checkpoint IDs, independent persistent counters and optional descriptions.
- Index legacy experiments conservatively without altering original checkpoints,
  weights or history; preserve evidenced resume lineage and flag unresolved data.
- Unify Frontier, supported Legacy, Arena, Analysis/evaluation and Alpha Laboratory
  selection, distinguish inference from full continuation and protect loaded models.
- Add previewed reversible checkpoint/generation deletion, shared-resource retention,
  interrupted-operation recovery and restoration without overwrite.
- Cover recovery fault injection, registry concurrency, migration, compatibility,
  workflow integration and deletion safety. Preserve architectures, learning defaults
  and the existing development roadmap.

## v1.5.3 — Parallel Frontier Training

- Add 1/2/4/8 independent environments feeding one shared DQN learner and replay.
- Support ordered vectorized collection and Windows-spawn workers with batched
  MLP/CNN inference, independent exploration and deterministic seed derivation.
- Expose reference per-transition and explicit per-vector-step optimizer policies;
  preserve global transition-based epsilon and target synchronization schedules.
- Add exact episode accounting, environment/local sequence journal and CSV fields,
  partial-episode checkpoint restoration, process cleanup and failure recovery.
- Add environment/strategy/update controls, dynamic dashboard metrics, readable
  durations, smoothed completion-rate ETA and actual optimizer counters.
- Preserve historical checkpoints, architectures, encoders, rewards, Legacy and
  frozen Arena/Analysis behavior; add fixed-step/fixed-update CPU benchmark tools.
- Document measured simulation/training tradeoffs, 160 benchmark trials, validation
  and the v1.5.4 optimization priorities. No mandatory CUDA or learning-objective change.

## v1.5.2 — Spatial Intelligence Foundations

- Add separate architecture and observation-encoder registries with validated, versioned identities.
- Add distance19 MLP, binary CNN and ordered-body CNN configurations alongside the original 355-parameter DQN.
- Support structured spatial/heading replay, batched learning, target synchronization and exact boundary recovery.
- Integrate all four configurations with Frontier selection, portable models, Arena, frozen evaluation, Analysis and CSV metadata.
- Preserve legacy model formats, gameplay, learning equations, rewards and training defaults.
- Document exact layers/encodings, controlled research methodology, measured CPU pilots and the research roadmap.
- Add compatibility, geometry, persistence, optimization, UI and end-to-end regression coverage.

## v1.5.1 — Frontier Training Foundations

- Add Legacy / Frontier Training navigation while preserving the existing Legacy
  Configuration, Dashboard, Analysis, Arena and model/checkpoint workflows.
- Add a separate CPU, single-environment headless worker using the unchanged DQN
  trainer, configurable budgets/seeds, advanced settings and 25/100/500-episode refresh.
- Add cached research dashboards with distribution candles, rolling mean/median,
  loss, measured throughput, ETA, optimizer diagnostics and process usage.
- Record full per-episode seeds, rewards, reasons, progress and optimizer/loss
  diagnostics independently of presentation refresh, in Analysis-compatible storage.
- Add isolated visible experiment directories, lightweight milestone/final models,
  automatic final saves, rotating boundary recovery, reload validation and save errors.
- Preserve existing checkpoint compatibility and numerical boundary resumption;
  integrate Frontier artifacts with Arena, Analysis, frozen evaluation and model exports.
- Add opt-in exclusive profiling, matched 100,000-step worker benchmarks, compatibility
  and failure tests, and measured performance documentation. No CUDA, vectorized
  environments, CNNs or new learning algorithm are introduced.

## v1.5.0 — Analysis & Experiment Framework

- Add a first-class Training Center Analysis tab with Distribution, Mean / Median
  and Raw Scores charts, exact hover, grouping, ranges, zooming and scrolling.
- Provide mandatory Model A and optional Model B, independent selectors of real
  saved checkpoints, and comparisons between checkpoints of the same model.
- Distinguish exploratory training history through each checkpoint from its
  recorded frozen evaluations; hide Model B's checkpoint/data controls when absent.
- Calculate type-7 linear quartiles, true min/max whiskers, population deviation,
  raw rolling statistics and side-by-side summaries without fabricating missing data.
- Add asynchronous shared-seed frozen Arena evaluation and separate result storage.
- Add EXPORT RAW / EXPORT GROUPED with readable Windows-compatible model/checkpoint
  filenames, comparison identifiers, metadata sidecars and exclusive creation.
- Persist new per-episode and optimization journals separately from weights while
  preserving read-only compatibility with v1.4 checkpoints.
- Add versioned experiment metadata for the existing baseline; future input
  encoders remain planned. DQN, rewards, encoder and gameplay rules are unchanged.
- Replace the large Dashboard epsilon chart with a compact indicator and recent
  score summary while retaining the other training charts and recorded epsilon.
- Include visible shareable model snapshots under `models/` with a copy-only export
  command, regression coverage, performance measurements and release documentation.

See [release notes](docs/releases/v1.5.0.md) and
[validation](docs/analysis-validation.md) for behavior, compatibility and testing.
