# Changelog

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
