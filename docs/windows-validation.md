# Windows validation — v1.2.1

Build diagnostic first, then release with `build_windows.ps1`. Copy the complete
output folder: `SnakeInfinite.exe` requires its accompanying `_internal` folder.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools/validate_windows_build.py dist/diagnostic/SnakeInfinite/SnakeInfinite.exe --output .artifacts/windows-diagnostic-validation
.\.venv\Scripts\python.exe tools/validate_windows_build.py dist/release/SnakeInfinite/SnakeInfinite.exe --output .artifacts/windows-release-validation
```

The real executable launches twice from Windows Temp, using an isolated profile.
Reports cover native Windows video, 15 runtime assets, frame decoding/menu
transition, audio-channel activity, icon loading, all seven modes,
pause/resume/restart/exit, and reopened settings/records/statistics. A second
process checks values persisted by the first. Reports/screenshots are saved in
the requested output directory; the isolated profile is retained for inspection.

`--validate-build report.json` is opt-in. Without `--settings` it creates a fresh
validation profile instead of injecting scores into normal player records.
Normal launches never run this diagnostic scenario.

## Executed locally on 2026-11-01

- Windows 11 x64, CPython 3.13.12 build environment, PyInstaller 6.22.3,
  Pygame 2.6.1, PyAV 17.1.0.
- Full automated suite: **171 passed**. V1.2 challenge/gameplay tests remain.
- Diagnostic and console-free release executables both passed external-working-
  directory validation, including all seven modes and persistence across processes.
- Final release validation: intro-to-menu **6.319s / 6.336s**, 144 frames decoded.
  An ordinary no-flag launch also completed the full intro and exited cleanly
  through a native Windows close event in **6.948s**.
- Repaired `.venvsnake` source launch from Windows Temp passed, decoding all
  144 frames and reaching the menu in **6.496s**.
- Packaging audit: **15 runtime assets**, official icon and version 1.2.1.0
  embedded; diagnostic PE subsystem 3, release subsystem 2 (no console).
  No bundled pytest/test modules or unused source artwork. Release folder: **109.8 MiB**.
- Startup and menu screenshots captured from the release were visually inspected.
- Reports: `.artifacts/windows-release-final-validation/`,
  `.artifacts/windows-diagnostic-validation/`,
  `.artifacts/windows-source-validation.json`, and
  `.artifacts/windows-packaging-audit.json`.

These checks used native Windows rendering and mixer activity, with isolated
profiles. They do not constitute a human gameplay/audio assessment or a clean
machine test without Python installed.

## Manual checklist

- [ ] Copy the release folder to another directory or clean Windows x64 machine.
- [ ] Double-click to launch; confirm no console appears.
- [ ] Watch the full intro and resize its window; check aspect ratio and transition.
- [ ] Visually confirm the official window/taskbar icon and menu logo.
- [ ] Listen to music, countdown, effects, and voice announcements.
- [ ] Play Classic, Arcade, and all five challenges with the original controls.
- [ ] Pause, resume, restart, and exit; confirm no intro replay during navigation.
- [ ] Save settings and a leaderboard entry, then close and reopen the application.
- [ ] Check user data/logs under `%LOCALAPPDATA%/SnakeInfinite/`, with no bundle writes.
- [ ] Validate on a clean machine without Python; a local build does not imply this test.

Mixer checks establish playback activity, not audible output. Rendered screenshots
do not substitute for a human playthrough or clean-machine acceptance test.
