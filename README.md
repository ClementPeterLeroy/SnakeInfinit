# SNAKE INFINITE

**v1.3.0 — Neural Foundations**

A minimalist Python/Pygame arcade game with Classic, Arcade, five challenges,
official startup branding, and local records. Built on the existing v1.0/v1.1
architecture. The optional CPU PyTorch Neural Laboratory adds configurable networks,
real neuron inspection, manual parameter edits, Step and Live simulation, and a
supervised gradient demo.
Classic, Arcade, and the definitive V1.2 challenges keep their existing rules.
Live play uses the current untrained weights. There is no DQN, replay buffer,
automatic episode restart, or reinforcement-learning training.

## Neural Laboratory

Install the optional dependencies into the **same Python environment that runs
main.py**. Ordinary arcade play needs only `requirements.txt`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt --index-url https://download.pytorch.org/whl/cpu
# For an existing environment without pip:
uv pip install --python .venv/Scripts/python.exe -r requirements-ai.txt --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe main.py
```

Choose **AI → NEURAL LABORATORY**. PyTorch loads on demand in a worker; startup and
the arcade screens never import it. Missing or broken AI dependencies show an
explanation while all ordinary modes remain available. Computation uses CPU
float32 tensors. Local verification used PyTorch 2.15.0+cpu on Python 3.13.

The default fully connected `torch.nn.Module` is **11 → 16 → 8 → 3**, with **355
trainable parameters**, counted from the actual model. Configuration allows 1–5
hidden layers, 2–128 neurons each, an optional initialization seed, and ReLU, Tanh,
Sigmoid, or Leaky ReLU (negative slope 0.01). Hidden layers use that activation;
the final layer is linear. Building asks before replacing current weights and
resets optimizer state. Initialization preserves the caller's Torch RNG stream.

The exact 11 binary inputs, in order, are:

1. Danger straight.
2. Danger right.
3. Danger left.
4. Moving up.
5. Moving down.
6. Moving left.
7. Moving right.
8. Food above.
9. Food below.
10. Food left.
11. Food right.

The pure-Python encoder reads an immutable snapshot, detects Classic walls/body
collisions, and allows entering a vacating tail unless growth prevents it. A
buffered heading is treated as committed without consuming it. Food flags use
the ordinary apple; terminal states remain inspectable and absent food produces
four zeros. This first observation model is **Classic only**: stars and challenge
hazards are deliberately excluded. Multiple food-direction flags may be active.

The **three raw outputs** are **Straight, Turn left, Turn right**, in that order.
They are untrained numbers, not probabilities, learned preferences, or meaningful
Q-values yet. The order of relative actions differs from the danger feature order.

Laboratory controls:

| Control | Effect |
| --- | --- |
| Reset environment | New separate Classic game; seed blank for fresh randomness, integer for repeatability |
| Step | Exactly one straight movement tick |
| Forward pass | Recompute current inputs without moving Snake |
| Relative action selector / Apply action | Select Straight/Left/Right, then advance exactly one tick |
| Network action / UNTRAINED (Step Mode) | One user-requested argmax action from the current inputs |
| Eleven input buttons | Toggle binary features for experiments; values need not describe a physically possible board |
| Restore Snake observation | Replace manual values with the current real game snapshot |
| Step Mode / Live Mode | Switch operating modes without resetting the board; Step is the default on entry |
| Play / Pause (Live Mode) | Continuous argmax control from real Snake observations; pause preserves the state and decision |
| Stop | Return to Step Mode, preserving the current environment |
| Speed | Cycle 1, 3, 5, 10, 20 steps/second; default 5 |
| Reset / Reset & Play Again | Reset with the seed field; after game over, explicitly begin a new live session |
| Inspect last decision | Pause and show the latest pre-action observation, activations, outputs and parameters |
| Current State / Forward | Pause and inspect a fresh post-action observation without movement |

Stepping or resetting also restores the real Snake observation. The lab game
does not affect the player's game, audio events, leaderboards, or statistics.
Outputs display the current value and change from the preceding forward pass.

**Live Mode** runs inside the ordinary application loop. For every movement tick,
it extracts the real Classic observation, makes one CPU forward pass, chooses the
highest raw output (ties select the first), and steps the existing game once.
Manual observations never control live play. The output panel shows all three
values to four decimal places, highlights the argmax, and names the selected action.
An untrained network may circle or die quickly; this is expected.

**Decision inspection:** the graph, inputs, output panel and neuron inspector
share an immutable **LAST DECISION / PRE-ACTION** snapshot. It includes the
observation, hidden/output activations, actual weights/biases, selected action,
and states before/after movement. The adjacent **POST-ACTION BOARD** shows where
the Snake moved. These are explicitly different moments. Pause and Stop preserve
the snapshot; **Current State / Forward** or **Restore Snake Observation** switches
to the current board's inputs. **Inspect Last Decision** restores the retained
decision. Editing inputs/parameters pauses execution and shows a refreshed current
calculation; replacing or updating weights clears the historical decision.

Live timing uses a separate accumulator, independent of rendering FPS. Each frame
performs at most eight movement/inference ticks, retaining ordinary catch-up work.
A stall producing more than five seconds of backlog pauses execution and explicitly
reports the canceled backlog; Play starts a fresh clock. Paused time never accrues.
Mode changes, resets, and speed changes clear fractional timing to prevent bursts;
changing speed with pending whole ticks also pauses and reports cancellation.
Changing tabs, leaving the lab, losing window focus, or beginning a numeric edit
pauses live play. Resizing remains interactive. Game over preserves the final
board, decision, score, step count, and cause of death. There is no automatic restart.

Click graph neurons, or use **Layer / Next** and **Select index** (zero-based),
to inspect every actual neuron. The inspector shows real incoming inputs and
weights, bias, `z = sum(w*x) + b`, activation, and final output. Scroll over the
inspector to reach every incoming connection. Click a weight row or **Bias**, edit
its value, then **Apply edit**. Edits run under `torch.no_grad()`, refresh the
forward capture, reset Adam state, and gain a `*` marker. Parameters must be finite
and within ±1e6; updates that overflow the network are rejected and rolled back.
Large layers show at most 16 explicitly labeled groups and representative links;
the inspector always shows exact individual values.

Activation intensity uses the fixed normalization `tanh(abs(a))`, without rescaling
between frames: zero is dark green, positive values become brighter emerald,
negative values use muted purple. Grouped nodes show the mean activation of their
members. The argmax output has a green ring. Static signed connection artwork is
cached for each displayed capture/geometry; selection and dynamic nodes remain
interactive without rebuilding connections on every rendered frame.

Numeric fields accept typing, Backspace, Enter to finish, Ctrl+A to clear, and
Escape to leave editing. Escape also cancels confirmation dialogs or returns to
the AI menu. The interface scales with the existing application viewport.

**Gradient demo:** supply three target numbers and a learning rate (default
0.001, greater than zero and at most 1), then click **1. Backpropagate**. The model
clears old gradients, calculates mean squared error, and calls `loss.backward()`.
Select any weight or bias to inspect its gradient. **2. Optimizer step** applies
one Adam update, refreshes outputs, and displays MSE before/after. Gradients from
that backward pass remain available as an explicitly labeled snapshot; another
update requires another backward pass. Changing inputs, environment, or parameters
invalidates pending gradients. This supervised example is **not DQN**, and its
targets are educational values, not optimal Snake actions. Adam is not guaranteed
to reduce loss on every user-chosen example or learning rate.

**Models:** enter a local name and save, or select an existing model and load.
Overwrite and load both require confirmation. JSON stores architecture, seed,
feature/action contracts, version, update count, and manual-edit markers. A separate
PyTorch state dictionary stores parameters; a checksum detects accidental changes.
Loading validates bounds, shape, dtype, finite values, and compatibility, using
[`weights_only=True`](https://docs.pytorch.org/docs/stable/notes/serialization.html)
with no unsafe fallback. Load only trusted local models; a checksum does not prove
authenticity. Optimizer state and the live environment are not saved. Files live in
`%LOCALAPPDATA%/SnakeInfinite/models/` (or the existing platform profile); a custom
`--settings` path puts models beside that settings file. Bundled assets stay read-only.

See the [v1.3 release notes](docs/releases/v1.3.0.md) for verification and limitations.

## Windows application

Launch `dist/release/SnakeInfinite/SnakeInfinite.exe`. Keep the **entire
SnakeInfinite folder**, including `_internal`, together when copying it. This is
a standalone Windows x64 onedir build; it does not require a separate Python
installation. Validation on a clean machine without Python is still recommended.

Existing `dist/` executables from v1.2.1 are not updated by source changes. Rebuild
to obtain the new arcade menu/version. The existing spec explicitly excludes
PyTorch and NumPy: the lightweight executable intentionally shows the dependency
explanation for the lab. Installing Torch into a separate Python does not add it
to a frozen executable. Run from source for the functional lab in v1.3. A future
separate AI-enabled Windows onedir build would need its own spec, CPU Torch/native
libraries and licenses, a larger download, and independent clean-machine testing.

The official video plays once before the menu, preserving its aspect ratio.
The supplied file is `assets/px/SnakeInfinite_Startup.mp4`; the resolver also
accepts `SnakeInfinite_Startup_6s.mp4` in supported asset locations. It never
creates a replacement animation. Persistent logs are at
`%LOCALAPPDATA%/SnakeInfinite/logs/SnakeInfinite.log`.

Build on Windows with Python 3.13 x64:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_windows.ps1 -Diagnostic
# Validate the diagnostic version, then build the console-free release:
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_windows.ps1
```

If this existing environment has no pip, use
`uv pip install --python .venv/Scripts/python.exe -r requirements-build.txt`.
`-Python path/to/python.exe` selects another build interpreter. The spec includes
only runtime artwork/video/sounds, application modules, and required Pygame/PyAV
libraries. Tests, caches, unused artwork, and pytest are excluded. PyInstaller's
[onedir spec](https://pyinstaller.org/en/stable/spec-files.html) produces
`dist/diagnostic/` with a console and `dist/release/` without a console.

```powershell
.\.venv\Scripts\python.exe tools/validate_windows_build.py dist/release/SnakeInfinite/SnakeInfinite.exe --output .artifacts/windows-release-validation
```

The validator launches the executable twice from outside the repository, with an
isolated profile and native offscreen Windows window. It checks video, runtime
assets, audio-channel activity, all seven modes, pause/restart/exit, and persisted
data. It does not assess audible speaker output. See the
[Windows checklist](docs/windows-validation.md) and [maintenance notes](docs/releases/v1.2.1.md).

## Run

Requires Python **3.12+**; Python 3.13 is used for local verification.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Substitute your interpreter if its launcher name differs. Dependencies are
Pygame, PyAV for the MP4, and pytest. The core requires only the standard library.

## Play

Select **SOLO**, then **CLASSIC**, **ARCADE**, or **CHALLENGES**. All playable
buttons and restarts enter a synchronized **3–2–1–GO!** countdown. Gameplay starts
at GO; movement, survival, stars, hazards, and movement accumulation stay frozen
before it. Countdown pauses when focus is lost.

- Move with **arrows**, **WASD**, or **ZQSD**, simultaneously supported.
- One turn is committed per tick; one additional valid corner is retained for
  the following tick. Reversals, repeated keys, and excess inputs are ignored.
- **ESC** pauses/resumes; losing focus also pauses. Pause clears movement input.
  Restart uses a fresh countdown with the same selected mode.
- Board edges, your body, active wall obstacles, and active mines are lethal.
  Warning markers are not lethal. Entering a vacating tail is legal unless growing.
- Menus use the mouse. Name entry supports text, Backspace, Enter to save, and
  Escape or Cancel to skip leaderboard entry.

The resizable 1280 × 720 interface preserves aspect ratio with `#121616`
letterboxing. Gameplay uses whole-pixel square cells at viewport resolution.

## Modes

**Classic:** 25 × 25 cells, three starting segments, apples only. Each apple gives
one point and one segment. Constant speed: Casual 6, Normal 9, Fast 13
cells/second, selected in Settings. Filling the board completes the game.

**Arcade:** apples stay available while stars appear every 15–25 active seconds.
A star lasts six seconds, shows an expiry ring and timer, and grants two points
plus two segments through pending growth. Only one star exists at once, avoiding
food, snake, and hazards. Optional stars yield space to apples on nearly full boards.

| Score | Cells/second |
| --- | --- |
| 0–9 | 7 |
| 10–24 | 8 |
| 25–49 | 9 |
| 50–74 | 10 |
| 75–99 | 11 |
| 100–149 | 12 |
| 150+ | 13 |

Speed changes preserve accumulated simulation seconds.

**Challenges:** all five are immediately playable through the shared engine.

| Challenge | Objective and rules |
| --- | --- |
| The Centurion | Reach **70 points**, constant 9 cells/second, apples only. |
| Star Hunter | Collect **10 stars**. Stars spawn every 10–15 seconds, expire after six; apples remain available. Constant 9 cells/second. |
| Speed Demon | **50 SEGMENTS · 20 BLOCKS/SEC · SURVIVE 45 SECONDS**. A legal serpentine start has a clear escape route; maximum difficulty. |
| The Labyrinth | Reach **30 points**. Permanent walls accumulate: first warning after five seconds, then one wall every eight seconds, without a four-wall global cap. |
| Minefield | Reach **25 points**. Mines spawn every eight seconds, warn for one, stay active for five, then detonate across a **3×3** area. At most three mines at once. |

Walls spawn at least five Manhattan cells from the head, avoid occupied/food/
committed-next cells, and preserve an apple route using reachability checks.
Placement is revalidated after the one-second warning. Once active, walls remain
until the game ends: no relocation, expiration, or removal to make room for food.
Unsafe/crowded spawn attempts are skipped. Rules optionally support waves of one
to four cells per cycle, independently of the total accumulated wall count.

Mines avoid occupied and committed-next cells, with an activation safety check.
At detonation, the headless engine checks **every snake segment** against the
mine's cell and its eight neighbors, clipped to the board. A hit records death
cause `mine_explosion`; simultaneous blasts produce one game-over event. The mine
is then removed. The brief explosion display creates no delayed collision zone.
When a movement tick and detonation share a timestamp, the detonation checks the
current body before that tick moves the snake.
The last active second previews the blast radius. Red/orange effects follow
Animations; static detonation outlines and a notice remain with animations off.
Countdown and pause freeze survival, warnings, active mines, and detonation time.

Selection shows difficulty, completion, and best score/survival/completion time.
Victories have a distinct overlay with challenge name and final statistics.

**Coming soon:** Against the Clock, Claustrophobia, No Turning Back, The Gauntlet,
Endurance, and The Hunter. The Hunter has a crimson snake marker and description
only; no hostile snake, AI, pathfinding, or collision mechanics are implemented.

## Audio, branding, and effects

Original supplied assets remain in their existing locations:

- `assets/px/logo.png`: official menu logo, scaled proportionally.
- `assets/px/Icon.png`: official window icon.
- `assets/branding/snake_infinite_icon.ico`: Windows icon derived from that PNG;
  rebuild with `python tools/build_icon.py` without changing the original.
- `assets/px/SnakeInfinite_Startup.mp4`: supplied six-second H.264 startup video,
  played once per launch. This file has **no audio stream**. Menu return and
  restarting games do not replay it. Its actual supplied filename is preserved.
- `assets/px/apple.png`, `star.png`: collectible sprites.
- `assets/snd/synth_wave_background.wav`: continuously looping background music.
- `eating_apple.wav`, `eating_star.wav`, `game_over.wav`: effects in `assets/snd/`.
- `3.wav`, `2.wav`, `1.wav`, `go.wav`: synchronized countdown effects.
- `super_voice.wav`: once per game on crossing 25 points, including 24 → 26.
- `game_over_voice.wav`: deaths below 20. At 20+ use ordinary game-over audio.
  Missing/disabled voices fall back to ordinary effects when enabled.

Music, Sound Effects, Voice Announcements, and Animations have independent
settings. Countdown audio follows Sound Effects. Countdown and voices share a
reserved channel; the previous announcement stops before the next plays. Music
pauses/resumes with its setting, without restarting on navigation or game restart.

Optional effects include green/gold particle bursts, floating +1/+2, star glow,
overlay fades, subtle death shake, and high-score celebration. Disabling Animations
preserves essential countdown text, expiry indicators, and hazard warnings.

Startup uses [PyAV](https://pyav.org/docs/stable/api/container.html) in a worker
thread with a bounded queue. Frames follow video timestamps, fit without cropping,
and release decoder resources afterward while Pygame continues processing events.
A short crossfade reveals the main menu when Animations is enabled.
Missing/unsupported video logs an error and continues to the menu. Embedded video
audio is unsupported by this player; the supplied silent video needs none.

The supplied logo has a baked-in darker background than `#121616`. Its approved
artwork is unchanged; a transparent or matching-background export is needed for
perfect blending. Missing logo/icon assets are logged without substitute artwork.
Missing collectible sprites use simple gameplay markers. Unavailable audio
devices and playback failures never prevent play.

## Local records and settings

User files live outside the source tree and read-only bundle:

- Windows: `%LOCALAPPDATA%/SnakeInfinite/`
- Other platforms: `$XDG_DATA_HOME/SnakeInfinite/`, falling back to
  `~/.local/share/SnakeInfinite/`.

Missing settings/records are copied from the previous
`%APPDATA%/snake-ai-laboratory/` location, the old XDG config location, or a legacy
application directory. Originals are never deleted and existing destination files
are never replaced. Copy failures are logged and the app remains usable.

`settings.json` migrates older preferences with Voice enabled by default.
`records.json` stores leaderboards, statistics, and challenge records. Override
with `python main.py --settings path/to/settings.json`; records are placed beside
that settings file. Writes are atomic where supported.

Separate top tens cover Classic Casual/Normal/Fast and Arcade. Positive scores
qualify when a board has space or beat its tenth entry. Names are limited to 12
characters. Confirmation saves name, score, mode, applicable speed, and UTC
date/time. Cancellation skips score entry. Challenges have separate records.

Statistics track games, apples, stars, total/highest/average score, active playtime,
deaths, challenge attempts, and completions. Each run is recorded once on terminal,
restart, menu exit, or app close. Abandoned active runs count in statistics but
not leaderboard entry; countdowns canceled before GO do not count as attempts.

Challenge records retain attempts, victories, completion status, best score,
fastest completion, longest survival, last completion date, and last death cause.
Existing records are preserved. Corrupt files are
backed up as `records.recovered-<id>.json`; valid entries are salvaged. If backup
fails, the original stays untouched. Storage failures are logged and shown on the
leaderboard. All records are **local only**, without online syncing.

## Architecture and headless engine

```text
main.py                 entry point
app.py                  input, screens, fixed-step timing, session lifecycle
core/
  config.py             immutable rules and Arcade speed thresholds
  challenges.py         five configurations and future challenge names
  game.py               seeded reset, movement, growth, timers, objectives, events
  board.py, snake.py    board geometry and movement
  hazards.py            safe placement, apple reachability, clipped blast geometry
  types.py              immutable observations, hazards, explosions, status and events
services/
  assets.py, audio.py   image caching and centralized audio
  persistence.py       settings validation and atomic writes
  records.py           scores, statistics, challenge records and recovery
  video.py              bounded startup decoder worker
  resources.py          development/frozen read-only resource resolution
  userdata.py           writable profiles and copy-only legacy migration
  diagnostic_logging.py persistent rotating logs
  build_validation.py   opt-in local distribution diagnostics
ui/
  menu.py, catalog.py, leaderboard_view.py, settings_view.py
  game_view.py, pause_view.py, countdown.py, effects.py, components.py
tools/build_icon.py     approved PNG to Windows ICO helper
tools/validate_windows_build.py  external-cwd executable validation
SnakeInfinite.spec      Windows onedir packaging configuration
build_windows.ps1       diagnostic and console-free release builds
tests/                  earlier coverage plus Arcade and integration tests
ai/
  observations.py, actions.py, config.py  pure observation/action/config contracts
  network.py             optional CPU Torch model and real layer captures
  laboratory.py          separate step-based game and staged supervised demo
  live.py                nonblocking live scheduler and immutable decision snapshots
  persistence.py         validated JSON + weights-only model storage
  runtime.py             availability probe without importing Torch
ui/neural_view.py        configuration, inspector, editing, gradients, model library
ui/neural_graph.py       bounded individual/grouped activation rendering
agents/                  future learning controllers; untrained live control is in ai/live.py
assets/                 original artwork, video, sounds, and derived ICO
```

The core imports no Pygame, audio, UI, window state, clocks, or real-time delays.

```python
from core.config import arcade_rules
from core.game import Game
from core.types import Direction

game = Game(rules=arcade_rules(), seed=12345)
initial = game.reset(seed=12345)
events = game.step(Direction.UP)  # One tick; elapsed time defaults to its interval.
state = game.get_state()         # Immutable, detached observation.
print(state.score, state.terminated, state.reward, state.star, state.hazards)
```

`step(action, dt=seconds)` injects time for one movement tick.
`advance_time(seconds)` advances timed mechanics without movement. Do not advance
the same duration through both calls. The application advances time per consumed
movement interval, so timed mechanics have movement-tick precision. Countdown and
pause never call the engine. Rendering is independent, capped at 120 FPS; stalled
game frames limit catch-up to 250 ms.

Observations include score, terminal status, elapsed time, food, hazards, pending
growth, counts, death cause, the latest simultaneous explosion group, and last-step
reward (points gained, or −1 on death). Explosion records describe completed
instantaneous events, not active danger zones. Identical
seeds, rules, actions, and time sequences reproduce episodes on the same Python
runtime. Each game owns its RNG; unseeded resets request fresh entropy. Existing
`Game("FAST", size=25, rng=...)`, `turn()`, and `step()` calls remain supported.
`Application.action("start")` remains an immediate programmatic developer/test
entry; every player-facing start uses the countdown.

## Verification and roadmap

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py --smoke-test
```

Earlier tests remain, with the resized-navigation test updated for mode selection
and countdown, and challenge assertions aligned with the definitive V1.2 rules.
Coverage includes Arcade, stars/growth, all challenge victories, the 50-segment
20-cell/second start, permanent wall accumulation, body-sensitive instantaneous
explosions, reproducibility, countdown freezing, voices, local record recovery,
statistics, UI, and real MP4 decoding. Headless tests forbid importing Pygame.
Maintenance tests add frozen resource lookup, migration, initialization failures,
and irregular frame timestamps.
Dummy-device tests verify rendering/routing without proving audible output.
`--smoke-test` renders three frames and exits with decoder cleanup; normal startup
plays the complete video.

Neural tests cover observations, tail movement, seeded initialization, 355 actual
parameters, every supported activation and layer bound, exact calculations, edits,
gradient reset/update, persistence corruption, and Pygame control flows. Torch
test modules explicitly skip when the optional dependency is absent. Compatibility
checks also run arcade modes with Torch imports forbidden.
Live tests cover inference-before-movement, one forward per step, FPS-independent
speed, pause/resume/stop/reset, snapshot accuracy, bounded catch-up, game-over,
tab/focus transitions, resized controls and quit handling.

Next: **v1.4 — First Intelligence**, introducing Deep Q-Learning. Reinforcement
learning, replay buffers, automated training episodes, trained policies, and spectator
tools remain outside v1.3.
