# First Intelligence / v1.4

`observations`, `actions`, and `config` are pure Python. `runtime` probes optional
Torch availability without importing it. `network`, `laboratory`, and `persistence`
are imported only when the user opens the lab and require `requirements-ai.txt`.

`SnakeNetwork` is a CPU `torch.nn.Module`, with a default 11→16→8→3 architecture
and 355 trainable parameters. Captures and inspections are detached snapshots of
actual Torch calculations; manual edits use `no_grad` and refresh those snapshots.

`Laboratory` owns a separate Classic `Game`, supports manual binary observations,
one movement/action at a time, forward comparison, and separate backward/Adam
steps for a supervised MSE example. `LiveController` in `live.py` schedules continuous
untrained argmax control at 1/3/5/10/20 steps per second without blocking the event
loop. Each inference precedes one movement. Pause, Stop and mode switching preserve
the environment; only an explicit Reset starts a new game. Immutable decision
snapshots keep pre-action values and parameters separate from the post-action board.
Observation extraction, inference, reset and stepping are distinct operations so
future scenarios can provide custom states without changing the controller.

No rendering, Pygame, training loops, or neural
assumptions enter the game engine. Relative output actions are Straight/Left/Right.

`ModelStore` saves bounded JSON metadata plus a weights-only state dictionary in
the writable profile. The UI obtains overwrite/replacement confirmation; loading
checks observation/action compatibility and tensor structure. Optimizer state is
not a persisted training checkpoint. See the main README for all features and UI
instructions.

`training_config.py`, `environment.py` and `replay.py` define bounded Classic
episodes, explicit termination/truncation and independently seeded experience
sampling. `dqn.py` owns online/target networks, epsilon-greedy exploration,
Huber/Adam optimization and measured training metrics. `worker.py` owns all live
training tensors and exports immutable snapshots to the Training Center.

`checkpoints.py` combines the existing safe model serializer with full boundary
resume state, including target, Adam, replay, counters and random streams.
`arena.py` performs frozen greedy evaluation without optimization or exploration.
Episode limits apply only to training/Arena, never normal arcade or lab Live Mode.
Both genuine termination and administrative truncation close the finite-return
objective without bootstrapping. See the README and training validation report.
