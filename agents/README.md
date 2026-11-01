# Future controllers

V1.3 has Step and Live laboratory modes. `ai/live.py` supplies user-started,
untrained continuous inference; it does not train or restart episodes. Future
learning controllers can submit `Direction` values to
`Game.turn()` and advance the simulation with `Game.step()`, or use
`Game.step(action)`. `Game.reset(seed=...)` and `Game.get_state()` support
reproducible episodes and detached observations, including stars, hazards, time,
growth, reward, and terminal status. `step(action, dt=...)` supports injected time.
Keep controllers
independent of the Pygame interface. No trained or reinforcement-learning agents
are implemented here.
