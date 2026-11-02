# v1.5.0 — Analysis validation report

The implementation is in the normal project workspace. No Git operations were
performed. Existing architecture, gameplay, network execution and training logic
were retained; the additions provide analysis, persistence and experiment metadata.

## Files created and modified

New runtime modules:

- `ai/analysis_stats.py`: raw records, heterogeneous dataset metadata and exact statistics.
- `ai/analysis_model.py`: cached chart derivation, range/alignment and comparison validation.
- `ai/history.py`: separate durable journals and read-only legacy history access.
- `ai/analysis_repository.py`: metadata-first catalog, loading and change signatures.
- `ai/analysis_export.py`: raw/grouped UTF-8 CSV and metadata sidecars.
- `ai/evaluation.py`: cancelable shared-seed frozen Arena batches.
- `ai/experiments.py`: generic versioned experiment descriptors and append-only links.
- `ui/analysis_chart.py`: distributions, rolling curves, raw overlays and exact hover.
- `ui/analysis_view.py`: asynchronous first-class Analysis workspace and controls.

Modified runtime/configuration files: `ai/worker.py` (journal/experiment instrumentation),
`ui/training_view.py` (tab integration, Dashboard cleanup and display-failure boundary),
`core/config.py` and `packaging/windows_version.txt` (1.5.0 identity).

New regression files: `tests/test_analysis_statistics.py`,
`tests/test_analysis_history_export.py`, `tests/test_analysis_evaluation.py`,
`tests/test_analysis_ui.py`, `tests/test_analysis_selection.py` and
`tests/test_model_export.py`. The visible `models/` library and
`tools/export_models.py` provide portable, copy-only model snapshots for sharing.
The existing release-identity assertion in
`tests/test_challenge_rework.py` now expects v1.5.0. Documentation changes are
`README.md`, `CHANGELOG.md`, `models/README.md`, `docs/releases/v1.5.0.md` and this report.
Application and Windows version metadata consistently identify v1.5.0.

## Compatibility and safeguards

Before development, the existing profile and checkpoint layout were inspected
read-only. Checkpoints contain `network.json`, immutable weight files,
`training.json` and a monolithic `training.pt`. Raw scores/lengths/rewards are
bounded to 20,000 episodes; losses/epsilon to 2,000 values. Older per-episode
reasons, step counters, epsilon and times were not persisted.

The final saved checkpoint was read without constructing a neural network:

| Recorded property | Value |
| --- | ---: |
| Available episodes | 1–10,000 |
| Cumulative environment steps | 2,540,054 |
| Mean / median training score | 12.2158 / 11 |
| Q1 / Q3 | 5 / 18 |
| Minimum / maximum | 0 / 51 |
| Best recorded episode | 9,800 |
| Read-only initial load, including Torch import | 3.18 s |

These are exploratory training results, not frozen-policy rankings. Full-history
duration can use the saved final elapsed total; reason percentages and step
alignment remain unavailable because the corresponding episode records are missing.

**All 100 protected checkpoint files matched their pre-development SHA-256 hashes**
both before and after the read-only validation. No checkpoint/history/weight data
was rewritten, renamed, deleted or migrated. Tests used temporary fixture stores
and redirected `LOCALAPPDATA`, `APPDATA` and `XDG_DATA_HOME` to workspace artifacts.

The original DQN, environment, encoder, training configuration, checkpoint format,
Arena, network implementation and `main.py` also match their baseline source hashes.
Only the worker's analysis instrumentation changed; actual learning/reward/termination
code did not. No production process was stopped, restarted or signaled by this work.
The two originally identified process IDs were absent at the final read-only process
check, so their continued execution cannot be confirmed. The completed saved run is intact.

## Automated results

The complete CPU/Torch/Pygame suite passed **354 tests in 37.11 seconds**, preserving
the existing 319 cases and adding 35 analysis/model-sharing cases. Coverage includes:

- Exact mean/median, type-7 quartiles, population deviation, IQR, empty/singleton
  data, partial groups, raw rolling statistics and missing fields.
- Different model architectures/counts, actual step alignment, training/evaluation
  separation, matching evaluation seeds/settings and cache reuse.
- Raw/grouped/range/comparison CSV, Unicode, missing blanks, write failures and
  exclusive creation without overwriting; readable model/checkpoint filenames,
  actual checkpoint identity for evaluations, and repeated-export numeric suffixes.
- Read-only v1.4 histories without model construction, frozen repeatable evaluations,
  cancellation, unchanged checkpoint hashes, durable journals and resume branches.
- Continued training after simulated history-write failure, chart control state,
  zoom/pan/reset, grouping, paired-candle hover, large-history rendering, background
  control synchronization and contained chart display failure.
- Mandatory Model A, selecting/deselecting Model B, independent checkpoint controls,
  comparisons within one model, checkpoint-bounded journal data, missing files and
  absent evaluations. Model changes retain valid chart/window/range context.
- Existing Classic/Arcade/challenges, Lab, Arena, checkpoint resumption, training
  and subprocess startup with Torch completely blocked.

Regression command (use an isolated profile; do not target production model paths):

```powershell
$testProfile = Join-Path (Get-Location) '.artifacts\v15-test-profile'
$env:LOCALAPPDATA = Join-Path $testProfile 'local'
$env:APPDATA = Join-Path $testProfile 'roaming'
$env:XDG_DATA_HOME = Join-Path $testProfile 'xdg'
.\.venv\Scripts\python.exe -m pytest -q --basetemp .artifacts\pytest-v15-release --tb=short
```

Run this in a separate terminal so profile overrides do not affect a subsequent
normal game launch in another terminal.

## Measured performance

Windows x64, Python 3.13.12, Pygame 2.6.1, CPU; 100,000 synthetic episode records,
one model, default 100-episode window, 886×300 offscreen chart. Drawing measurements
average 60 frames with exact hover enabled; lookups average 100 cached calls.

| Mode | Background preparation | Cached lookup | Chart drawing |
| --- | ---: | ---: | ---: |
| Distribution | 0.193 s | 1.42 µs | 1.11 ms/frame |
| Mean / Median | 0.345 s | 1.37 µs | 0.88 ms/frame |
| Raw Scores | 0.157 s | 1.55 µs | 0.97 ms/frame |

These measurements cover chart drawing, not full-window frame rate or all laptops.
All 100,000 records remained available to statistics/hover/export. Tests verify
that repeated rendering does not rerun statistical calculations. Legacy Torch
deserialization and first import are more expensive but execute in the background.

## UI, evaluation, exports and experiments

The Analysis screen offers mandatory Model A and optional Model B with independent
selectors of actual saved checkpoints. Deselecting B hides its checkpoint/data
controls and removes its data immediately. Each DATA control explicitly selects
training history through that checkpoint or its existing frozen evaluation files;
missing data is explained, never fabricated. Both slots may compare checkpoints
of the same model.

It retains three chart modes, episode/step
alignment, remembered ranges, zoom/pan/reset, exact hover, mean/median toggles,
side-by-side raw statistics, asynchronous CSV and frozen evaluations. An offscreen
1280×720 render, including checkpoint/data selectors and export labels, was inspected
for alignment and overlap. Live desktop interaction,
audio and subjective smoothness have not been manually verified.

Calculations use type-7 linear quartiles, true min/max whiskers and population
deviation. Incomplete groups show actual n. Full-resolution data supplies every
summary and export; extrema-preserving samples only affect rendering. Missing
legacy values stay N/A/blank. Step comparisons require actual recorded counters.

Frozen evaluation reuses Arena with private, non-trainable weights, epsilon zero,
shared seed lists and shared explicit reward/termination settings. New JSON results
are separate from exploratory training records. Raw/grouped CSV uses unique
exclusive filenames and metadata sidecars; selected range is the default.
**EXPORT RAW** / **EXPORT GROUPED** filenames identify model names, selected
checkpoint episodes/short identities and export type/grouping. Comparisons include
both identities; Windows-invalid characters are removed and duplicates use numeric
suffixes. Existing CSV columns and statistical conventions are retained.

Experiment descriptions retain encoder, dimension, architecture, configuration,
seeds, declared step budget, runs/checkpoints and evaluation associations. New
baseline runs register automatically. Only the existing 11-input Experiment A is
executable; B/C/D remain documented plans. No new learning technique was added.

## Remaining limits and manual testing

- Legacy history cannot reconstruct fields never saved or episodes outside the
  checkpoint's retained window. No automatic migration is attempted.
- Dense distribution views display selected original candles preserving extrema;
  zoom for individual groups. Exact hover and exports retain every group.
- Range duration currently shows N/A unless the range starts at episode 1 and
  includes a real final elapsed value.
- Optimizer journals may lose the final unflushed partial batch after abrupt
  termination; completed episode transactions are durable.
- Future input dimensions are supported analytically and in metadata, not executable
  observation encoders. The lightweight packaged executable still excludes Torch;
  this update does not rebuild or publish an executable.

Follow the [release manual checklist](releases/v1.5.0.md#responsiveness-and-manual-checks)
for desktop navigation, resizing, real checkpoint selection, chart interactions,
evaluation/cancellation, exports, disposable training, gameplay and audio checks.
No staging, commits, pushes, tags, releases or GitHub modifications were performed.
