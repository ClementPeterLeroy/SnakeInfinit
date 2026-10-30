# SNAKE INFINITE

**v1.2.0 — Definitive Arcade Edition**

A minimalist Python/Pygame arcade game with Classic, Arcade, five challenges,
official startup branding, and local records. Built on the existing v1.0/v1.1
architecture. No automated snakes, neural networks, or training dependencies.

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

User files live outside the source tree:

- Windows: `%APPDATA%/snake-ai-laboratory/`
- Other platforms: `$XDG_CONFIG_HOME/snake-ai-laboratory/`, falling back to
  `~/.config/snake-ai-laboratory/`.

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
ui/
  menu.py, catalog.py, leaderboard_view.py, settings_view.py
  game_view.py, pause_view.py, countdown.py, effects.py, components.py
tools/build_icon.py     approved PNG to Windows ICO helper
tests/                  earlier coverage plus Arcade and integration tests
agents/, ai/            documented future extension points
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
Dummy-device tests verify rendering/routing without proving audible output.
`--smoke-test` renders three frames and exits with decoder cleanup; normal startup
plays the complete video.

Next: **v1.3 — Neural Foundations**, introducing configurable PyTorch neural
networks. No PyTorch, DQN, training, or automated agents are included in v1.2.
