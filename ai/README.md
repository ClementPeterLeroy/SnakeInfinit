# Neural Foundations / v1.3

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
instructions. V1.4 First Intelligence will introduce DQN separately.
