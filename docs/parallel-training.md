# v1.5.3 — Parallel Frontier Training

Frontier can collect experience from **1, 2, 4 or 8 independent Classic Snake
environments**. They share one online network, target network, Adam optimizer and
uniform replay buffer. This parallelizes collection and batches policy inference;
it does not train independent networks or average weights. All four v1.5.2 model
architectures and observation encoders are preserved.

## Starting a four-environment run

1. Open **AI → Training Center → Frontier Training** and keep **SOURCE: new**.
2. Choose configuration A/B/C/D, episode budget and master seed.
3. Click **PARALLEL ENVIRONMENTS** until it shows **4**.
4. Keep **STRATEGY: AUTO** for measured, low-overhead vectorized collection, or
   explicitly select **MULTIPROCESSING** to test Windows-spawn collectors.
5. Select the update policy deliberately, then click **START**.

The default remains **one environment, Auto, one update per transition**. It
preserves the original single-environment numerical path. Auto selects vectorized
collection for the chosen count; it does not silently alter count, learning ratio
or model. Short CPU benchmarks show communication overhead and no consistent
reference-policy advantage from maximizing worker count. Rebenchmark your machine
before changing the defaults for a serious experiment.

## Two supported collectors

**Vectorized** collection keeps independent Python environments in the learner's
process. A stable environment-ID loop advances them in groups. It amortizes
inference and scheduling but does not claim NumPy-vectorized simulation or
simultaneous execution on multiple CPU cores.

**Multiprocessing** uses Windows-compatible `spawn` with one environment per child.
The central learner sends actions and receives transitions through owned pipes.
Workers import neither Torch nor Pygame, have no model copies or displays, and
return responses in deterministic environment-ID order. Pipe failures, child
exceptions, unexpected exits and timeouts become visible session errors. Shutdown
joins children and terminates unresponsive owned children as a bounded fallback.
`main.py` uses the guarded entry point and `freeze_support()`.

This synchronous design prioritizes reproducibility and bounded control points.
It has IPC overhead and no actor/learner overlap. Tiny Snake environments often
cost less than moving their observations between processes. CNN optimization can
dominate both collectors, so more simulation workers may not accelerate learning.

## Batched inference and exploration

Each environment samples its own epsilon decision and random action from an owned
deterministic RNG. Greedy observations form one ordered batch: `[B,11]`, `[B,19]`,
or `{spatial:[B,3,25,25], heading:[B,4]}`. The shared network performs one forward
for that greedy subset; actions map back to the original environment IDs.
Random exploration does not require a neural forward.

All actions in a vector batch use the weights at the start of that batch. The
learner then inserts and processes results in ascending environment-ID order.
Consequently, changing environment count changes the training trajectory even
when the same number of optimizer updates is performed.

## Global schedules and learning ratio

The **global environment-step counter** counts every collected transition once,
summed across environments. Epsilon uses that counter, including each transition's
offset within a batch. It is not an independent per-worker decay. Target sync fires
at the existing configured multiples of global transitions, including thresholds
crossed inside a vector batch. Replay sampling stays uniform with its owned RNG.

| Policy | Optimizer schedule after replay warmup | Meaning |
| --- | --- | --- |
| **1 per transition — reference** | One optimizer update after each inserted transition | Preserves the original update-to-data ratio |
| **1 per vector step — fewer updates** | One optimizer update after the final transition in the vector batch | Approximately 1/N updates per transition for N active environments |

Warmup and a partially filled final batch affect the realized ratio. The dashboard
reports actual updates and updates/s; experiment/checkpoint/model metadata records
the selected policy. A higher steps/s with fewer updates is a different data/update
budget, not proof of a faster or better learned policy. Compare both equal-step
and equal-update budgets. The loss, rewards, Adam, Bellman targets, clipping and
architectures are unchanged.

## Seeds, episodes and histories

Parallel episode seeds are deterministic signed-63-bit values from the first
eight little-endian bytes of SHA-256 over:

```text
snake-frontier-seeds-v1:<master seed>:<environment ID>:<local episode sequence>
```

Local episode sequences start at zero. Per-environment exploration RNG seeds use
the same derivation with sequence 2^32; the supported training episode ceiling is
far below that reserved stream ID. The original one-environment reference path
keeps its historical owned-RNG convention for numerical compatibility.

Completion order defines global episode numbering, with environment ID breaking
ties within a batch. Each journal record retains completion index, environment ID,
local sequence, seed, score, length, reward, reason, global step count, epsilon,
active duration and optimization diagnostics. New raw CSV exports append
`environment_id` and `local_sequence`; historical unknown values remain blank.
The original history table remains readable, with parallel identities stored in
an additional table. Training histories and frozen evaluation remain separate.

The scheduler reserves at most the remaining episode slots, producing exactly the
requested completion count. Partial episodes are never counted. A resumed target
shortened below the number of active environments advances only the necessary
slots and retains surplus partial traces for a later extension. Milestone
collection is constrained to land on the actual milestone completion index.

Fixed seeds reproduce an ordered run under the same worker count, execution
policy, implementation and numerical runtime. Vectorized/spawn collectors use the
same ordered protocol. Bitwise equivalence across counts, numerical libraries,
versions or future asynchronous schedulers is not promised. Frozen greedy Arena
evaluation stays single-policy, explicitly seeded, and independent of training
worker count; epsilon exploration is never enabled by parallel training metadata.

## Pause, stop, checkpoints and recovery

Control commands are handled between vector batches. Pause retains all partial
episodes in memory and stops both collection and optimization. Pause + Save writes
a checkpoint immediately at that quiescent boundary, without advancing episodes.
Stop and Stop + Save release child processes; in-memory Resume rebuilds them and
restores the same partial traces. Completed runs also release collectors.

Full parallel recovery extends the existing checkpoint state with execution
configuration, actual optimizer count, vector count, environment sequences/seeds,
action RNG states and partial action traces. Online/target weights, Adam, replay,
replay RNG, global counters, epsilon schedule and metrics retain their existing
validated storage. Partial environments reconstruct deterministically by replaying
their seed/action trace **without inserting replay or running learning updates**.
This supports exact continuation in the tested matching runtime. Reconstruction
cost grows with partial episode length.

Checkpoint metadata is made visible atomically only after tensors and checksums
exist. Parallel state has its own schema version 1, leaving network format-1/2
compatibility intact. Historical single-environment checkpoints remain loadable
and resumable in their original path. Parallel checkpoints resume in Frontier;
Legacy Training reports a clear routing message. Model-only snapshots still
support frozen inference, but cannot resume training.

On collector failure, the worker flushes completed records and attempts a recovery
checkpoint at the last committed learner state, reports failures and closes owned
children. It does not count responses from an incomplete failed collection batch.
Exploration draws already made for that failed batch remain consumed; recovery
does not claim numerical identity to a run in which no failure occurred.
Numerically corrupted learner state or storage errors can prevent recovery and
are reported explicitly; no successful recovery is assumed without a saved path.

Experiment/checkpoint/model metadata identifies collection settings, actual
optimizer count, training duration and session wall duration. Full session wall
time includes pauses; training elapsed time retains the existing active-time
meaning. Normal output remains visible under `experiments/frontier/<id>/`, with
lightweight shareable `.pt`/`.json` model pairs and separate full recovery/history.
No historical experiment is rewritten or migrated.

## Dashboard, timing and profiling

Headers display actual environment count and resolved strategy. Metrics include
global steps, completed episodes, steps/s, episodes/s, actual updates/s and total
updates, active environments, worker health, epsilon, recent scores, loss and
CPU/RAM. Multiprocessing usage combines parent and child reports; one busy core is
about 100% CPU. Peak memory is the sum of process peaks, an upper bound whose peaks
need not occur simultaneously. Closed collectors retain their last usage sample.

Elapsed duration uses seconds/minutes/hours/days. ETA stays **Calculating…** until
at least five completions contribute, then uses an exponentially smoothed recent
completion rate (25% new sample). It adapts to longer episodes but remains an
estimate. Finished runs show **Completed**. Raw durations remain numeric in saved
metadata. Charts refresh every 25/100/500 completions, with all episode records
collected independently of presentation refresh.

Optional profiling separates policy-forward batch calls, action selection,
simulation, encoding, replay, forward/backward, optimizer, serialization and UI.
With spawn collection, parent simulation time includes IPC and worker encoding;
it is not a separate child CPU profile. A neural forward call can cover several
greedy observations. Disabled profiling attaches no timing wrappers.

## Benchmarking and next work

Run the reproducible matrix on your own hardware:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_parallel.py --steps 2048 --updates 256 --simulation-steps 4096 --output .artifacts\parallel-benchmark.json
```

The tool tests A/B/C/D × 1/2/4/8 × vectorized/spawn, separating environment-only
collection from end-to-end inference/replay/optimization. Training has both
equal-environment-step and equal-optimizer-update categories for both policies.
Startup is recorded separately, and no chart or disk I/O occurs in the timed
loop. Reports include steps, updates, completions, episode lengths, durations,
CPU/RAM and failures. Choose a new filename for each report. Short runs are noisy;
repeat larger budgets before drawing hardware or policy conclusions.

See [measured validation](parallel-validation.md) for the release matrix and tests.
No long learning-quality study or full 500-game reevaluation was performed for this
infrastructure patch. Historical supplied frozen results remain **Alpha 37.614
mean / 38 median** and **Frontier Distance 43.822 mean / 44 median**, both after
10,000 training episodes with 500-game evaluations; they were not reproduced here.

v1.5.4 priorities are profiling CNN optimization, batched tensor materialization,
CPU/GPU crossover measurements, transfer overhead, appropriate Torch threading and
only then justified CUDA/device selection. v1.5.5 Benchmark Laboratory and v1.6.x
neural research remain on the roadmap. No CUDA requirement, new encoder, reward
change, prioritized replay or new DQN algorithm is introduced in this release.
