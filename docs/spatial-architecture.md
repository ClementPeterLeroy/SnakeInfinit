# Spatial Intelligence Foundations — v1.5.2

The Pygame-free game engine remains in `core/`. Observation encoders read immutable
game snapshots; models, codecs and DQN belong to `ai/`. Registry metadata imports
without Torch. Constructors load Torch only when a neural model is requested.
Ordinary arcade play still works without the optional dependency.

## Registries and model lineup

`ai/model_registry.py` registers architectures, compatible encoders, constructors,
input shapes, action ordering, schema versions and parameter counts.
`ai/encoders.py` separately registers observation ordering, shape, dtype,
normalization and compatible architectures. Unknown IDs or incompatible metadata
are rejected. A trainer or Arena resolves its encoder/codec once at initialization;
training steps do not perform registry lookups.

| Model | Architecture ID | Encoder ID | Default parameters |
| --- | --- | --- | ---: |
| A | `mlp_11_16_8_3` | `snake-classic-binary11-v1` | 355 |
| B | `mlp_distance19` | `snake-distance19-v1` | 3,459 |
| C | `cnn_spatial_compact` | `snake-spatial-binary-v1` | 188,419 |
| D | `cnn_spatial_compact` | `snake-spatial-ordered-v1` | 188,419 |

All schema versions are 1. All outputs are three linear Q-values, ordered
**Straight, Turn left, Turn right**. No Softmax, pathfinding or rule overrides are
used. Heading order is **up, down, left, right**, matching the original features.

## Vector observations

A keeps the original binary11 encoder and original `SnakeNetwork`: danger
straight/right/left, heading up/down/left/right, food above/below/left/right.
Its layers are Linear(11,16), ReLU, Linear(16,8), ReLU, Linear(8,3):
192 + 136 + 27 = **355** parameters. The legacy configurable 11-input networks
remain supported with their original activation/layer metadata.

B retains those exact first 11 features, then appends free-cell distances in
**N, NE, E, SE, S, SW, W, NW** order. Each ray counts consecutive free cells before
a wall or any current body cell, including the tail, divided by **24**. Adjacent
obstacles give zero; a clear ray across the board gives one. Food does not block
rays. This is a static occupancy sensor; the original immediate-danger features
retain their own legal tail-vacating semantics. B layers default to
Linear(19,64), ReLU, Linear(64,32), ReLU, Linear(32,3):
1,280 + 2,080 + 99 = **3,459** parameters. Two or three hidden layers of 2–512
neurons are configurable in Advanced DQN; activation is ReLU.

## Spatial observations and shared CNN

Both C and D use float32 **[3,25,25]** spatial tensors, indexing `[channel,y,x]`,
and a separate float32 **[4]** heading vector. Channels are body excluding head,
head, and food. Empty cells are zero; head and food cells are one. Absent food
produces an empty food plane. These encoders require the existing 25×25 Classic
AI board.

C assigns one to every body cell. D assigns **(N − i) / N**, where N is the number
of body segments excluding the head and i runs from zero nearest the head to
N−1 at the tail. Thus the nearest segment is one and the tail is 1/N. N=0 writes
no body values; N=1 writes one. No division occurs for an empty body.

| Operation | Single-sample output | Parameters |
| --- | --- | ---: |
| Spatial input | 3×25×25 | 0 |
| Conv 3→32, kernel 3, padding 1, stride 2; ReLU | 32×13×13 | 896 |
| Conv 32→64, kernel 3, padding 1, stride 2; ReLU | 64×7×7 | 18,496 |
| Conv 64→64, kernel 3, padding 1, stride 2; ReLU | 64×4×4 | 36,928 |
| Flatten | 1,024 | 0 |
| Concatenate heading | 1,028 | 0 |
| Linear 1,028→128; ReLU | 128 | 131,712 |
| Linear 128→3 | 3 | 387 |
| **Total for C or D** | **3 Q-values** | **188,419** |

For batch size B, tensors are `[B,3,25,25]` and `[B,4]`, with `[B,3]` outputs.
The same constructor and initialization seed give C and D identical initial
weights. Only the body encoder differs. CNN widths are fixed in this release.

## Training and replay

The existing epsilon schedule, uniform replay sampling, Bellman targets, terminal
mask, Adam, Huber loss, gradient clipping and target synchronization remain shared
across all configurations. No Double/Dueling DQN, parallel environments or CUDA
are introduced. Reward and training defaults remain unchanged.

The legacy A tensor path is retained. B replay uses float32 19-feature vectors.
C/D replay retains immutable, lossless compact observations: head-first ordered
cell positions as little-endian uint16 bytes, food cell, heading and encoder
identity. Adjacent transitions share cached observations. A sampled batch is
materialized directly into contiguous float32 spatial tensors plus heading, for
both online and target networks. The compact storage avoids retaining two dense
25×25 float grids per transition; materialization still has a CPU cost.

## Artifacts and compatibility

Legacy format-1 metadata and weights continue to load without edits. New B/C/D
artifacts use format 2 with explicit architecture/encoder identities and schema
versions, parameters, input shape, actions and parameter count. Network weights
load with `weights_only=True`, size/checksum/shape/dtype/finite checks. CNN replay
uses primitive bytes/lists and tensors, not arbitrary pickled model objects.
Unknown, contradictory or incompatible identities fail clearly.

Frontier publishes visible, isolated directories:

```text
experiments/frontier/<experiment-id>/
    experiment.json
    models/<prefix>_ep010000_model.pt
    models/<prefix>_ep010000_model.json
    checkpoints/<experiment-id>/<checkpoint>/
        network.json
        weights-<id>.pt
        training.json
        training.pt
    history/history.sqlite3
    session.json
```

Prefixes are `snake_dqn_11`, `snake_dqn_distance19`, `snake_cnn_binary`, and
`snake_cnn_ordered`. Share model `.pt` and its matching `.json` together. Metadata
records experiment identity, completed episode and cumulative environment steps.
Full boundary checkpoints also retain online/target weights, optimizer, replay,
training counters, epsilon schedule and owned RNG streams. Resume semantics stay
at completed-episode boundaries under the same runtime; partial episodes remain
in memory only. There is no migration of historical experiments.

Milestones remain episodes 1,000/3,000/5,000/10,000 plus the actual final target;
coincident final/milestone saves are not duplicated. Rotating recovery checkpoints
retain the configured slots. Existing experiments are never overwritten.

Arena automatically restores the matching encoder and uses frozen greedy
inference. Analysis keeps its three charts, independent checkpoint selectors,
optional comparison model, ranges, zoom/scroll and exports. Training history is
clipped through the selected checkpoint and checked against architecture/encoder
identity. Frozen evaluation is stored separately. CSV content is preserved;
metadata sidecars add the model identity and actual CNN parameter count.

## Profiling

Profiling is opt-in. Exclusive timers cover encoding, environment simulation,
action selection, actual policy-network forward calls, replay sampling,
optimization forward/backward, optimizer update, serialization and presentation.
Optimization forwards are excluded from the policy-inference timer. Epsilon-random
actions do not count as neural inference. `inferences_per_second` means actual
policy forwards divided by measured forward time; it is unavailable with profiling
off or no forward calls. Optimizer updates/s and steps/s use active wall time.
Session files retain these metrics. Disabled profiling attaches no wrappers.
