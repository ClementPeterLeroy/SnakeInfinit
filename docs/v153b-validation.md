# v1.5.3b validation

Validated on 2026-11-04 using the workspace Python 3.13.12 environment with CPU
PyTorch and Pygame. Public source/version resources identify **v1.5.3b — Recovery
Repairs & Model Library**. The Windows numeric resource is `(1, 5, 3, 1)` because
that resource requires integers; its public version strings are `v1.5.3b`.

## Recovery repairs

AUD-01 is repaired with explicit batch recovery safety and independent active-trace
validation before checkpoint files are written. Partial collection or optimizer
failure cannot replace the previous verified recovery. UI status differentiates
new verified, previous verified and unavailable recovery. No parameter rollback is
claimed after optimizer mutation.

AUD-02 is repaired by draining committed events before failure flushing, preserving
completion indices/environment associations and validating duplicate persistence.
An emergency JSON journal covers persistent SQLite flush failure and is merged
read-only into Analysis history. Buffered history remains in use.

Fault injection covers before collection, partial collection, before optimizer
mutation, after partial mutation, terminal cleanup, history persistence, safe
checkpoint resume and repeated recovery/persistence. Both original strict expected
failures are ordinary passing regression tests, without new skips or xfails.

## Automated and application checks

Full regression command, with LOCALAPPDATA/APPDATA/XDG_DATA_HOME redirected into
`.artifacts/v153b-test-profile`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .artifacts\pytest-v153b-release-all --junitxml .artifacts\v153b-tests.xml --tb=short
```

**499 passed, 0 failures, 0 errors, 0 skipped, in 115.29 seconds.** Local JUnit
evidence is `.artifacts/v153b-tests.xml`, including all 13 library safety cases.
The full suite includes existing gameplay, optional-AI startup, architecture,
learning, parallel collectors, recovery, Analysis/export and UI regression coverage.
Source compilation passed. The bounded `main.py --smoke-test` startup also exited
successfully with isolated settings and SDL dummy devices, displaying v1.5.3b in
the startup log. The music asset loaded through the dummy mixer; audible playback
was not tested.

Acceptance checks use real application routing, tensors and workers in temporary
storage, with SDL dummy video/audio:

| Workflow | Result |
| --- | --- |
| A: existing Alpha checkpoint → shared library → Arena | Passed; correct architecture and canonical identity |
| B: new Distance generation → short training → library | Passed; Created becomes Completed with actual checkpoint |
| C: continue Distance checkpoint | Passed; same generation, isolated continuation session |
| D: Alpha/Distance Analysis and frozen evaluation | Passed; independent exact checkpoint selection and saved evaluation |
| E: checkpoint and generation Trash/Restore | Passed in temporary fixtures; unrelated/shared files retained |
| F: application close/reopen | Passed; IDs and family counters unchanged |
| G: original recovery failures | Passed; unsafe save refused and committed history retained exactly once |

Additional tests cover concurrent allocation, family-independent counters,
description/name separation, metadata-only migration, architecture compatibility,
missing/overwritten/moved/corrupt artifacts, no training-state resume, loaded/active
guards, duplicate export ownership, owned evaluation/export movement, shared
history/comparison retention, stale selectors, interrupted move recovery, rollback,
overwrite refusal and path escape.
Lifecycle regressions also verify that an orphaned paused experiment becomes
Interrupted after restart and a failed Legacy final save remains Failed in the registry.

The actual Alpha model was loaded and the Library, Arena and temporary deletion
preview rendered to local PNGs for inspection. The temporary preview was cancelled.
No real user model was deleted. These are windowless rendering checks, not a claim
of interactive desktop or audible playback validation.

## Actual legacy indexing

Representative temporary migration fixtures passed before indexing user storage.
The metadata registry is:

```text
C:\Users\cleme\AppData\Local\SnakeInfinite\model_library\library.sqlite3
```

| Generation | State | Checkpoint/model artifacts |
| --- | --- | ---: |
| Snake DQN Alpha — Gen 1 | Completed | 25 |
| Snake DQN Alpha — Gen 2 | Interrupted | 4 |
| Snake DQN Distance — Gen 1 | Completed | 12 |
| Snake DQN CNN — Gen 1 | Interrupted | 1 |
| Snake DQN CNN — Gen 2 | Created | 0 |
| Snake DQN CNN — Gen 3 | Interrupted | 1 |
| Snake DQN CNN — Gen 4 | Interrupted | 4 |

Total: **7 actual generations and 47 checkpoint/model artifacts**. No oCNN
experiment was fabricated. Checkpoints of an experiment were grouped by evidence;
training-interface names did not create new families. Alpha and Distance historical
evaluation/history references remain indexed and usable. A Created CNN experiment
with no checkpoint is not presented as a loadable trained model.

Two repeated final scans added **0 generations and 0 checkpoints**, retaining all
IDs/numbers. Two historical CSV metadata files lack resolvable source provenance:
`20261102-130102-92519a94-raw.csv.metadata.json` and
`20261102-130642-633b614c-grouped.csv.metadata.json`. They remain untouched and are
shown in Migration Review. No historical provenance was guessed.

SHA-256 verification of **215 protected original files found zero changes**.
This set includes existing model weights/configuration, checkpoints, experiment
artifacts, historical evaluations and histories. Local evidence is
`.artifacts/v153b-final-verification.json`; only the new library metadata was added
to user-data storage. All deletion tests used temporary fixtures.

## Files changed or added

- Recovery, completion persistence and metadata: `ai/parallel.py`,
  `ai/checkpoints.py`, `ai/frontier.py`, `ai/history.py`, `ai/frontier_store.py`.
- Registry and runtime integration: new `ai/model_library.py`,
  `ai/worker.py`, `ai/analysis_repository.py`, `ai/evaluation.py`,
  `tools/export_models.py`.
- Shared UI/routing: new `ui/model_library_view.py`, `app.py`, `ui/menu.py`,
  `ui/frontier_view.py`, `ui/training_view.py`, `ui/arena_view.py`,
  `ui/analysis_view.py`, `ui/neural_view.py`.
- Tests: new `tests/test_recovery_repairs.py`, `tests/test_model_library.py`,
  `tests/test_library_safety.py`, `tests/test_library_workflows.py`; updated
  `tests/test_parallel_audit.py`, `tests/test_first_intelligence_ui.py`,
  `tests/test_neural_ui.py`, `tests/test_challenge_rework.py`.
- Version/documentation: `core/config.py`, `packaging/windows_version.txt`,
  `README.md`, `CHANGELOG.md`, `docs/parallel-training.md`; new
  `docs/model-library.md`, `docs/releases/v1.5.3b.md`, this report.
- Local verification scripts, JSON, JUnit and UI screenshots under `.artifacts/`.

## Limits

Trash uses same-volume moves and does not offer permanent deletion. Shared legacy
history databases and shared/ambiguous records remain physically intact with
logical tombstones. Path/ownership/active-reference conflicts block deletion;
unresolved operation recovery and stale writer locks require review. Keep the
registry backed up to preserve assigned identities/counters. Full trace recovery
validation can take time for long active episodes. A failure of both SQLite and
the emergency storage destination cannot guarantee durability and is reported.

The existing Alpha-only Laboratory remains unchanged in scope. No CUDA, new
architecture, learning algorithm, reward/hyperparameter change, long training study
or roadmap change is included. A packaged Windows executable was not rebuilt.
Audio playback was not physically tested. **No Git state-changing operation,
staging, commit, push, tag, release or publication was performed.**
