"""Opt-in local distribution diagnostics, never used by normal gameplay."""
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
import time
from uuid import uuid4
import pygame
from core.config import VERSION
from core.challenges import CHALLENGES
from services.resources import bundled_assets, resource_root
from services.userdata import user_data_dir


def validate_build(settings_path: Path | None, output: Path):
    from app import Application
    # Diagnostics must not inject test records into a player's normal profile.
    settings_path = settings_path if settings_path is not None else (
        user_data_dir() / "validation" / uuid4().hex / "settings.json")
    report = dict(version=VERSION, frozen=bool(getattr(sys, "frozen", False)),
                  cwd=str(Path.cwd()), resources=str(resource_root()),
                  data=str(settings_path.parent), checks={})
    checks = report["checks"]
    app = None
    try:
        assets = bundled_assets()
        missing = [str(path) for path in assets.values() if not path.is_file()]
        checks["all_15_runtime_assets"] = not missing
        if missing:
            raise RuntimeError("Bundled assets missing: " + ", ".join(missing))
        app = Application(settings_path)
        report["initial_speed"] = app.settings.speed
        report["initial_validation_score_present"] = any(
            row["name"] == "BUILD-CHECK" for row in app.records.data["leaderboards"]["classic_NORMAL"])
        checks["icon_loaded"] = app.assets.originals.get("Icon.png") is not None
        report["video_driver"] = pygame.display.get_driver()
        checks["native_windows_window"] = sys.platform == "win32" and bool(pygame.display.get_wm_info().get("window"))
        app.start_intro()
        video = app.video
        started = time.monotonic()
        clock = pygame.time.Clock()
        while app.screen == "intro" and time.monotonic() - started < 20:
            dt = clock.tick(120) / 1000
            for event in pygame.event.get():
                app.handle_event(event)
            app.update(dt)
            app.draw(dt)
        checks["six_second_video_to_menu"] = (app.screen == "menu" and video.error is None
            and video.displayed_frames > 0 and 5.5 <= time.monotonic() - started < 12)
        report["startup_seconds"] = round(time.monotonic() - started, 3)
        report["decoded_frames"] = video.decoded_frames
        checks["decoder_released"] = app.video is None and not video.thread.is_alive()
        output.parent.mkdir(parents=True, exist_ok=True)
        video.draw(app.window)
        pygame.image.save(app.window, str(output.with_name(output.stem + "-startup.png")))
        app.draw(0.3)
        pygame.image.save(app.window, str(output.with_name(output.stem + "-menu.png")))
        app.start_intro()
        checks["intro_only_once"] = app.screen == "menu"
        # Ensure the current persisted setting cannot hide an available audio device.
        app.settings.music = True
        app.settings.sound_effects = True
        app.settings.voice_announcements = True
        app.audio.apply_settings()
        checks["music_playback_active"] = bool(app.audio.ready and app.audio.music_loaded and pygame.mixer.music.get_busy())
        report["mixer_format"] = pygame.mixer.get_init()
        checks["all_countdown_and_voice_sounds_loaded"] = len(app.audio.counts) == 4 and len(app.audio.voices) == 2
        mode_results = {}
        for mode in ("classic", "arcade", *(entry.id for entry in CHALLENGES)):
            app.action("play:" + mode)
            app.draw(0.01)
            sound = bool(app.audio.ready and app.audio.announcement_channel.get_busy())
            frozen = app.game.get_state()
            for _ in range(3):
                app.update(1)
                app.draw(0.01)
            valid = app.screen == "game" and app.game.get_state() == frozen
            app.update(app.game.interval)
            moved = app.game.snake.body[0] != frozen.snake[0]
            elapsed = app.game.elapsed
            app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            app.update(2)
            paused = app.screen == "pause" and app.game.elapsed == elapsed
            app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            resumed = app.screen == "game"
            rules = app.game.rules
            app.action("play")
            restarted = app.screen == "countdown" and app.game.elapsed == 0 and app.game.rules == rules
            app.action("menu")
            mode_results[mode] = all((valid, moved, paused, resumed, restarted, sound))
        report["mode_checks"] = mode_results
        checks["all_seven_modes_countdown_pause_restart"] = all(mode_results.values())
        app.action("speed:FAST")
        app.records.add_score("classic_NORMAL", "BUILD-CHECK", 17, "classic", "NORMAL")
        app.action("exit")
        checks["exit_action"] = not app.running
        app.close()
        app = Application(settings_path)
        checks["settings_persist_after_reopen"] = app.settings.speed == "FAST"
        checks["leaderboard_persists_after_reopen"] = any(
            row["name"] == "BUILD-CHECK" and row["score"] == 17
            for row in app.records.data["leaderboards"]["classic_NORMAL"])
        checks["statistics_persist_after_reopen"] = app.records.data["statistics"]["games"] >= 7
        checks["user_data_outside_resources"] = not settings_path.resolve().is_relative_to(resource_root().resolve())
        report["success"] = all(checks.values())
        if not report["success"]:
            raise RuntimeError("Distribution checks failed: " + ", ".join(key for key, value in checks.items() if not value))
    except Exception as error:
        report["success"] = False
        report["error"] = str(error)
        logging.exception("Distribution validation failed")
        raise
    finally:
        if app is not None:
            app.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logging.info("Distribution validation report: %s success=%s", output, report.get("success"))
