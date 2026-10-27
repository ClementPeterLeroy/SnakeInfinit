# SNAKE

**v1.0.0 — Foundation** · Repository: `snake-ai-laboratory`

A playable, minimalist Snake game and the foundation for a future AI laboratory.
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
- Eat apples for one point and one segment each. Walls and your body end the game.
- Press **ESC** to pause or resume. Losing window focus also pauses gameplay.
- Restart or return to the main menu from the pause and game-over overlays.
- Fill all 625 cells to complete the board.
- Settings offer music, effects, cosmetic button transitions, and fixed speeds:
  **CASUAL** (6), **NORMAL** (9), **FAST** (13 cells/second).

The resizable window opens at 1280 × 720. The interface scales uniformly with
letterboxing as needed, preserving square board cells and aligned mouse input.
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

**The supplied assets contain `synth_wave_background.flac`, but no corresponding
WAV.** V1 deliberately uses only the filenames authorized by the brief. To enable
the soundtrack, provide the named WAV in `assets/snd/`. Missing files or audio
devices are tolerated and reported through logging.

## Architecture and verification

- `core/`: pure Python simulation; `Game.turn()` buffers one valid turn and
  `Game.step()` advances a cell and returns events. Terminal states stop updates.
- `ui/`: views, shared typography, and buttons; no collision or scoring rules.
- `services/`: atomic JSON persistence, cached images, centralized audio.
- `app.py`: screen navigation, input, fixed-step timing, event-to-audio routing.
- `agents/` and `ai/`: documented extension points only.

Rendering runs at up to 120 FPS, independently of movement ticks. Pausing clears
accumulated movement time. Very long stalled frames cap catch-up at 250 ms to
keep the interface responsive.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py --smoke-test
```

Tests cover movement buffering, collisions, legal tail entry, food placement,
growth, full-board completion, speed, preference recovery, input, pause timing,
resized rendering, missing assets, and unavailable audio. Application tests use
SDL dummy video/audio drivers. `--smoke-test` renders three frames and exits.
