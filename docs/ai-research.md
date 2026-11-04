# SnakeInfinite AI Research

## Chapter 1 — Baseline DQN

The baseline is the original 11→16→8→3 ReLU MLP with **355 parameters**. Its 11
features describe immediate danger, direction and relative food position. Three
linear outputs choose straight, left or right. Training uses CPU DQN with Adam
and Huber loss; defaults are learning rate .001, gamma .99, replay 100,000,
batch 64, target sync 1,000 steps, gradient clip 10, epsilon 1→.05 over 100,000
steps. Rewards are food +10, collision −10, ordinary step −.01, stagnation −5.
The unchanged finite-episode terminal/truncation convention applies.

The project's historical **10,000-episode** baseline reference, supplied with the
v1.5.2 research brief, reports **2.54 million steps**, **5 h 24 min** training and
best training score **51**. Its **500-game frozen evaluation** reports mean
**37.614**, median **38**, minimum **7**, maximum **77**, Q1 **29**, Q3 **46**.
Standard deviation, evaluation collision/stagnation rates and mean episode length
were not included in that reference and are not reconstructed here. These are
historical results, not a new evaluation performed during this patch.

The original observation can react to immediate danger but does not explicitly
represent distant obstacles, the full board, or body segment order. Deterministic
greedy policies can enter safe-looking loops; bounded stagnation terminates them.
More observation information is a research hypothesis, not a guaranteed cure.

The supplied v1.5.1 matched 1,000-episode project reference reports 94,102 steps,
mean score .513, Legacy 600.34 s and Frontier 261.35 s (about 2.30×). That reference
has a different scope from the separately recorded local 100,000-step benchmark
in [Frontier validation](frontier-validation.md). Neither is silently substituted
for the other. New v1.5.2 compatibility timings are in
[spatial validation](spatial-validation.md).

## Chapter 2 — Observation Architecture

Four selectable Frontier configurations now provide:

- A: original immediate-danger binary11 MLP, unchanged.
- B: binary11 plus normalized eight-compass-direction free-cell distances; a
  default 19→64→32→3 MLP with 3,459 parameters.
- C: full-board binary body/head/food planes and heading; 188,419-parameter CNN.
- D: the same CNN with normalized body order instead of binary occupancy.

The [architecture specification](spatial-architecture.md) defines exact feature
ordering, distance conventions, ordered-body formula and every layer dimension.
Distance rays expose obstacle proximity, but discard much spatial geometry.
Full-board planes preserve geometry; learning useful global behavior still costs
samples and computation. Ordered occupancy may help distinguish cells likely to
vacate soon. It provides no explicit future prediction or reachable-space solver.
CPU CNN optimization is materially slower than the small MLPs in measured pilots.

## Chapter 3 — Experimental Methodology

Training trajectories include epsilon exploration and evolving weights. Frozen
evaluation uses saved weights, greedy actions, no optimizer updates and separate
result files. Keep the selected checkpoint and its training history distinct from
evaluation records. Missing records remain unavailable.

For serious comparisons, record architecture/encoder IDs, parameter counts,
hyperparameters, rewards, rules, training seed, checkpoint, environment-step budget,
runtime/software and evaluation seed list. Use multiple training seeds and the same
fixed evaluation seeds, rules and caps for every checkpoint. Reproduce within the
same runtime; cross-version numerical identity is not promised.

Compare equivalent **training environment steps**, not only episodes, since episode
length differs by policy. Frontier currently specifies an episode target and logs
cumulative steps; use real saved boundaries and disclose budget differences. Exact
step-budget experiment orchestration is not introduced here. Record wall-clock
duration alongside sample efficiency and parameter count.

Report frozen mean/median, min/max, Q1/Q3, population standard deviation, collision
and stagnation rates, and mean episode length. Retain per-seed raw records and
descriptive CSV exports. Report training steps, active duration, policy-inference
throughput, optimizer updates/s and profiling scope; separate paused time and
serialization overhead when appropriate.

C versus D is the cleanest encoding comparison because layers, parameter counts,
initialization mechanism and default learning settings are identical. Comparing B
against a CNN also changes capacity and computational structure, so performance
differences cannot be attributed solely to encoding. Repeated seeds and matched
budgets are necessary before ranking models. The short engineering pilots validate
execution and persistence; they do not establish learned-policy superiority.

## Parallel Frontier infrastructure

v1.5.3 supports ordered vectorized or Windows-spawn collection with 1/2/4/8
environments, one learner, batched inference and explicit update-to-data policies.
See [parallel training](parallel-training.md) and
[measured validation](parallel-validation.md). Frozen historical references now
also include the supplied 10,000-episode Distance model result: mean **43.822**,
median **44** over 500 frozen games. These supplied results are not reevaluations
performed by this infrastructure patch.

## Roadmap

| Release | Research scope |
| --- | --- |
| v1.5.3 — Parallel Frontier Training (implemented) | Multiple simultaneous environments, vectorized execution where appropriate, batched inference, shared replay, CPU scaling benchmarks |
| v1.5.4 — CUDA & Frontier Optimization | GPU acceleration, CPU/GPU benchmarking, justified automatic device selection, profiling and optimization |
| v1.5.5 — Benchmark Laboratory | Standardized fixed seeds, optional epsilon-controlled evaluation, per-seed results, comparisons, paired statistics, selected-seed Arena replays and CSV |
| v1.6.x — Neural Intelligence Research | Further encodings, reachable-space awareness, CNN depth/width, temporal/recurrent networks, survival learning, reward design, Double/Dueling DQN investigations |

Earlier proposed 13/16-feature and alternative 19-feature observations remain
deferred hypotheses; their old experiment labels are not the new Frontier A–D
configuration identities. Classical pathfinding and hybrid AI remain deferred
research ideas outside the current pure-neural scope. No CUDA, parallel training,
recurrent models or algorithm changes are implemented in v1.5.2.
