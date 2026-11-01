"""Launch the real executable twice from outside the repository, in an isolated profile."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def validate_normal_launch(executable, external, env, profile):
    """Exercise the ordinary entry point and close through the actual window."""
    import ctypes
    from ctypes import wintypes
    log = profile / "logs/SnakeInfinite.log"
    offset = log.stat().st_size if log.exists() else 0
    process = subprocess.Popen([str(executable)], cwd=external, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    started = time.monotonic()
    played = False
    try:
        while process.poll() is None and time.monotonic() - started < 25:
            latest = log.read_bytes()[offset:].decode("utf-8", errors="replace") if log.exists() else ""
            if "Startup finished; outcome=played; transitioning to main menu" in latest:
                played = True
                break
            time.sleep(0.05)
        if not played:
            raise RuntimeError("Normal executable launch did not complete startup playback")
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        closed = []
        @callback_type
        def close_window(hwnd, _):
            owner = wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == process.pid and ctypes.windll.user32.GetWindowTextLengthW(hwnd):
                ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE -> SDL QUIT
                closed.append(True)
            return True
        ctypes.windll.user32.EnumWindows(close_window, 0)
        process.communicate(timeout=15)
        assert closed and process.returncode == 0
        latest = log.read_bytes()[offset:].decode("utf-8", errors="replace")
        assert latest.count("Startup invoked once;") == 1
        assert latest.count("transitioning to main menu") == 1
        return round(time.monotonic() - started, 3)
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Retain these local diagnostic files for troubleshooting; no player data is used.
    external = Path(tempfile.mkdtemp(prefix="SnakeInfinite-Windows-validation-"))
    env = os.environ.copy()
    env["LOCALAPPDATA"] = str(external / "local")
    env["APPDATA"] = str(external / "roaming")
    env["SDL_VIDEO_WINDOW_POS"] = "20000,20000"  # Native window, outside the desktop.
    env.pop("SDL_VIDEODRIVER", None)
    env.pop("SDL_AUDIODRIVER", None)
    settings = external / "local/SnakeInfinite/settings.json"
    results = []
    for index in (1, 2):
        report = output / f"launch-{index}.json"
        command = [str(executable), "--settings", str(settings), "--validate-build", str(report)]
        completed = subprocess.run(command, cwd=external, env=env, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=90,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        (output / f"launch-{index}.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"Executable validation failed ({completed.returncode}); see {output}")
        result = json.loads(report.read_text(encoding="utf-8"))
        if not result["success"] or not result["frozen"] or result["video_driver"] != "windows":
            raise RuntimeError(f"Frozen native-window validation failed: {result}")
        if Path(result["cwd"]) != external:
            raise RuntimeError("Executable did not run from the external working directory")
        results.append(result)
    assert results[1]["initial_speed"] == "FAST"
    assert results[1]["initial_validation_score_present"]
    # Ordinary entry point: no diagnostic/smoke flags and no settings override.
    normal_seconds = validate_normal_launch(executable, external, env, settings.parent)
    assert settings.is_file() and (settings.parent / "records.json").is_file()
    assert (settings.parent / "logs/SnakeInfinite.log").is_file()
    summary = dict(success=True, executable=str(executable), external_cwd=str(external),
                   profile=str(settings.parent), separate_process_persistence=True,
                   default_entry_launch=True, native_windows_video=True,
                   normal_launch_full_intro=True, normal_launch_and_exit_seconds=normal_seconds,
                   startup_seconds=[result["startup_seconds"] for result in results])
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
