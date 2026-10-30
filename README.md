# SnakeInfinite

**v1.1.0 — Gameplay Polish** · Repository: `ClementPeterLeroy/SnakeInfinit`

A playable, minimalist Snake game and the foundation for a future arcade and AI laboratory.
This release contains no neural networks, training, or automated controllers.

## Run

Requires Python **3.12+**, with a Pygame wheel available for your Python version.
Python 3.13 is used for local verification.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

If your interpreter has a different launcher name, substitute its executable in
the first command. After setup, the last command starts the game.

## Play

- Click **SOLO**. Move using arrows, **WASD**, or **ZQSD**, simultaneously supported.
- Quick corner inputs retain one current turn and one following turn. Only one
  turn is applied per movement tick; reversals and repeated key events are ignored.
- Eat apples for one point and one segment each. Walls and your body end the game.
- Press **ESC** to pause or resume. Losing window focus also pauses gameplay.
- Restart or return to the main menu from the pause and game-over overlays.
- Fill all 625 cells to complete the board.
- Settings offer music, effects, optional button/overlay transitions and apple
  feedback, and fixed speeds:
  **CASUAL** (6), **NORMAL** (9), **FAST** (13 cells/second).

The resizable window opens at 1280 × 720. Menus scale uniformly with letterboxing;
the gameplay board is rendered at the viewport resolution with integer-size,
square cells, centered without stretching. Mouse input follows the same viewport.
AI opens an informational placeholder for a future release.

## Settings and assets

Preferences save automatically outside the source tree:

- Windows: `%APPDATA%/snake-ai-laboratory/settings.json`
- Other platforms: `$XDG_CONFIG_HOME/snake-ai-laboratory/settings.json`, falling
  back to `~/.config/snake-ai-laboratory/settings.json`.

Override with `python main.py --settings path/to/settings.json`. Missing or invalid
preferences recover to defaults; valid individual values are preserved. Storage
failures log a warning and keep the session playable.

Original assets are preserved. The apple image loads from `assets/px/apple.png`,
with a red circle fallback. Sound effects use `assets/snd/eating_apple.wav` and
`assets/snd/game_over.wav`. Background music expects
`assets/snd/synth_wave_background.wav` and loops without restarting on screen
changes. Disabling music pauses it; enabling resumes it.

The supplied WAV soundtrack is now present and used directly. No conversion or
additional audio dependency is required. This release selects WAV for music and
effects; FLAC files are not selected as a fallback. Pygame/SDL handles decoding.
Music is loaded once, loops continuously, and pauses/resumes with the Music
setting. Navigation and restarting a game do not restart the track. Missing
assets, unavailable devices, and playback errors are handled without stopping
gameplay and are reported through logging.

## Deterministic, headless simulation

The engine imports only Python's standard library. It runs without a window,
audio device, Pygame import, or rendering clock:

```python
from core.game import Game
from core.types import Direction

game = Game(seed=12345)
initial = game.reset(seed=12345)
events = game.step(Direction.UP)  # Exactly one movement tick, no real-time wait.
state = game.get_state()          # Immutable, detached GameState snapshot.
print(state.snake, state.apple, state.score, state.terminated)
```

The same seed, rules, and action sequence reproduce initial positions and
subsequent food spawns. Every game owns its RNG; global `random` calls and
externally injected RNGs cannot change it. `reset()` without a seed requests
fresh system entropy. A seeded replay should use the same Python runtime.

Existing calls such as `Game("FAST", size=25, rng=random.Random(1))`, `turn()`,
and `step()` remain supported. `turn()` retains its original single pending-turn
contract; the additional deferred human input lives in the application layer.
`step(action)` optionally applies an action before advancing. Terminal games
return no more events until reset. Reading a snapshot never changes engine state.

```python
from core.config import GameRules

rules = GameRules(board_width=31, board_height=21, initial_speed=9.0)
game = Game(rules=rules, seed=42)
```

`GameRules` is immutable and validates dimensions and speed. Legacy speed/size
arguments and explicit rules are alternative constructors and cannot be mixed.
Classic is the only playable mode; the menu still starts a 25 × 25 game.
Reserved progressive-speed, star, and hazard switches reject enabled values
explicitly. Arcade and Challenges can extend these rules in V1.2 without moving
simulation into the UI. Neural networks and automated agents remain unimplemented.

## Architecture and verification

- `core/`: pure Python rules, board, snake, seeded reset, snapshots, and step events.
- `ui/`: views, shared typography, buttons, and cosmetic feedback timers;
  no collision or scoring rules. Image/font caches are bounded during resizing.
- `services/`: atomic JSON persistence, cached images, centralized audio.
- `app.py`: screen navigation, input, fixed-step timing, event-to-audio routing.
- `agents/` and `ai/`: documented extension points only.

Rendering runs at up to 120 FPS, independently of movement ticks. Pausing clears
accumulated movement time and queued input. Resume and restart also reset the
accumulator; even pause/resume in the same frame excludes elapsed menu time.
Tick-boundary rounding is tolerated consistently across rendering rates.
Very long stalled frames cap catch-up at 250 ms to keep the interface responsive.
Optional visual timers never advance the simulation. Disabling Animations removes
the apple ring, +1 flash, overlay fades, and animated button transitions.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py --smoke-test
```

The original 28 tests are preserved. Additional tests cover seeded episode replay
including subsequent food spawns, distinct seeds, RNG ownership, resets, detached
state, rules validation, rectangular boards, input buffering, keyboard aliases,
frame-rate-independent timing, animation toggles, mocked music lifecycle and
failures, and 10,000 simulation steps in a subprocess that forbids importing
Pygame. Application tests use SDL dummy video/audio drivers; these checks do not
prove audible speaker output. `--smoke-test` renders three frames and exits.
