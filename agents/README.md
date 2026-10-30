# Future controllers

V1.2 is human-controlled only. Future controllers can submit `Direction` values to
`Game.turn()` and advance the simulation with `Game.step()`, or use
`Game.step(action)`. `Game.reset(seed=...)` and `Game.get_state()` support
reproducible episodes and detached observations, including stars, hazards, time,
growth, reward, and terminal status. `step(action, dt=...)` supports injected time.
Keep controllers
independent of the Pygame interface. No automated agents are implemented here.
