# Model Library — v1.5.3b

The shared library identifies **model family → generation → checkpoint**.
Alpha is the original 11-input MLP; Distance is the 19-input MLP; CNN uses binary
body planes and oCNN uses ordered-body planes. Frontier and Legacy are training
interfaces recorded as metadata. Network architectures and DQN defaults are unchanged.

## Identity and lifecycle

An independent experiment allocates an immutable UUID and the next number in its
family: `Snake DQN {Family} — Gen {N}`. Allocation uses a SQLite immediate
transaction and a unique family/number constraint. Descriptions are separate,
optional text. Deletion never decrements counters. Resume retains the generation;
its physical continuation session has a new run ID so old history is preserved.
Seeding a new experiment from weights allocates another generation and fresh
optimizer state; it is not presented as full continuation.

Lifecycle values include Created, Training, Paused, Completed, Interrupted, Failed,
Stopped, Archived, Review and Deleted. A record without a checkpoint is an experiment,
not an available trained model. A dead training lease becomes Interrupted. Live
process leases protect running/paused learners and loaded selections across views.
Completed workers retain a loaded lease until explicitly unloaded or closed.

## Storage and migration

The registry is `<user-data>/model_library/library.sqlite3`, using SQLite/WAL in
the same style as history storage. On Windows the default is
`%LOCALAPPDATA%\SnakeInfinite\model_library\library.sqlite3`. It is a separate
metadata database: original history databases, models, checkpoints and visible
repository exports are not rewritten. Preserve the registry along with backups
to preserve assigned identities and counters.

Tables hold generations, family sequences, run aliases, checkpoint envelopes,
sessions, external references, duplicate locations, leases, managed roots and
Trash operation journals. Lists parse JSON architecture metadata without loading
PyTorch tensors. Loading checks architecture/encoder identity, current checkpoint
identity and weights SHA-256. No replacement checkpoint is silently selected.

Discovery covers Legacy checkpoints, indexed Frontier experiment roots, profile
models and repository `models/`. Explicit training output roots are registered as
managed storage. Checkpoints group by immutable run identity; evidenced resume
parents join their existing generation. Portable exported copies retain checkpoint
provenance. Missing parents stay separate and are marked Review. Unproven manual
imports retain an explicit unavailable-provenance label. Repeated indexing is
idempotent; it does not fabricate experiments, training configuration or evaluation
results. Missing/corrupt/incompatible artifacts remain visible with unavailable status.

Historical evaluation, experiment, export and shared history references are indexed
where their provenance is available. Unassigned references are retained and listed
for review. Details shows paths rather than inventing missing metrics. Existing
model files remain in their original locations, including the visible repository
`models/` folder; the registry does not replace portable exports.

## Selection workflows

- **Browse:** expand a family, choose its generation, then an exact checkpoint.
  Search includes family, generation and description. Checkpoints and generations
  have paging. DETAILS shows configuration, architecture, IDs and known references.
- **New training:** family → NEW GENERATION → configure → start. A Created record
  can be configured later; the allocation is retained even if the experiment never starts.
- **Continue:** Frontier/Legacy library button → existing generation → resumable
  checkpoint → LOAD SELECTED. Legacy supports compatible single-environment Alpha;
  other families and parallel checkpoints use Frontier.
- **Arena:** choose a checkpoint through the same library. Inference-only models
  work; full optimizer/replay state is not required for viewing or frozen evaluation.
- **Analysis:** independent Model A and optional Model B selections support
  cross-family comparison and two checkpoints of one generation. Training history
  through the selected checkpoint remains distinct from frozen-policy evaluation.
  Frozen evaluation records include stable registry IDs when selected through the library.
- **Laboratory:** the existing inspector accepts Alpha only. Other families are
  rejected explicitly; no Universal Laboratory is introduced.

Inference-only artifacts cannot be presented as resumable. Missing `training.pt`
disables full continuation, even when weights remain usable. Existing compatible
programmatic entry points remain available.

## Deletion and restoration

DELETE CHECKPOINT or DELETE GENERATION opens a preview listing affected files and
shared records retained. CANCEL changes nothing. MOVE TO TRASH rechecks the preview
and active references, then moves owned paths to `model_library/Trash/<operation>/`.
The operation journal is committed before filesystem moves; interrupted operations
roll back on registry startup. RESTORE returns files to their original locations
and keeps the original IDs and generation number. It refuses overwrites.

Checkpoint deletion leaves unrelated checkpoints and history. Generation deletion
includes owned sessions/checkpoints, exported copies and independently owned
evaluation/experiment/export files. Shared SQLite history, cross-generation
comparisons and ambiguous shared records remain physically intact and are hidden
from deleted generation selectors. Active/paused training, loaded checkpoints,
writer locks, unsafe paths, links/junctions and ownership conflicts block deletion.
UNLOAD IDLE VIEWS releases inactive views; running or paused sessions must first
be stopped or closed. Stale selectors cannot load deleted entries.

Trash supports same-volume atomic moves only. Cross-volume deletion is rejected.
No permanent-delete action is provided. Unresolved rollback conflicts require
manual review and keep the affected item unavailable. A stale writer lock also
requires review. Application-managed locks protect supported workflows; unrelated
programs editing files during a move are outside that coordination.

## Recovery correctness

AUD-01: a batch begins with recovery safety false and commits true only after
successful collection, replay/optimizer work and terminal-slot cleanup. Save
validates active traces independently before creating checkpoint files. Failure
after partial mutation retains the previous verified checkpoint and reports its
status; it does not claim rollback of model parameters.

AUD-02: committed completion events drain on both normal and exception paths.
Completion indices deduplicate retries; persistent inserts validate matching
content and preserve environment associations. If SQLite flushing remains unavailable,
an emergency JSON journal retains pending records and diagnostics. History readers
merge that journal without duplicate episodes. If both destinations fail, the
worker reports the storage failure rather than claiming durable history.

## Reusable API

`ai.model_library.ModelLibrary(profile, repository=None)` exposes `create`,
`register_run`, `generations`, `get`, `checkpoints`, `index_checkpoint`, `resolve`,
`load`, `scan`, `references`, `acquire`/`release`, `preview_delete`, `delete` and
`restore`. Cross-feature references use generation/checkpoint UUIDs. Paths remain
necessary for existing portable formats and provenance, not as the public identity.
No network service, CUDA dependency, new architecture or learning objective is added.
