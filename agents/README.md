# Future controllers

V1.1 is human-controlled only. Future controllers can submit `Direction` values to
`Game.turn()` and advance the simulation with `Game.step()`, or use
`Game.step(action)`. `Game.reset(seed=...)` and `Game.get_state()` support
reproducible episodes and detached observations. Keep controllers
independent of the Pygame interface. No automated agents are implemented here.
