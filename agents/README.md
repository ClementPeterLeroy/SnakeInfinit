# Controllers and evaluation

The laboratory retains Step and Live modes. `ai/live.py` supplies user-started
continuous inference without training. `ai/dqn.py` trains the network through
`ai/environment.py`; `ai/arena.py` evaluates frozen checkpoints greedily.
Controllers submit `Direction` values to
`Game.turn()` and advance the simulation with `Game.step()`, or use
`Game.step(action)`. `Game.reset(seed=...)` and `Game.get_state()` support
reproducible episodes and detached observations, including stars, hazards, time,
growth, reward, and terminal status. `step(action, dt=...)` supports injected time.
Keep controllers
independent of the Pygame interface. Training-only episode limits belong to the
wrapper; the original simulation and all human arcade rules remain unchanged.
