"""Exercise actual Pygame input, timing, rendering, and audio without a display."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import pytest
from app import Application
from core.types import Direction, GameEvent


@pytest.fixture
def app(tmp_path):
    application = Application(tmp_path / "settings.json")
    yield application
    application.close()


def test_fixed_timing_pause_resume_and_restart(app):
    app.action("start")
    head = app.game.snake.body[0]
    app.update(app.game.interval / 2)
    assert app.game.snake.body[0] == head
    app.update(app.game.interval / 2)
    assert app.game.snake.body[0] == (head[0] + 1, head[1])
    app.action("pause")
    paused_head = app.game.snake.body[0]
    app.update(50)
    app.action("resume")
    app.update(0)
    assert app.game.snake.body[0] == paused_head
    assert app.accumulator == 0
    app.action("start")
    assert app.game.score == 0
    assert app.game.snake.body[0] == head


@pytest.mark.parametrize("key", [pygame.K_w, pygame.K_z, pygame.K_UP])
def test_keyboard_aliases(app, key):
    app.action("start")
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
    assert app.game.snake.pending == Direction.UP


def test_mouse_navigation_and_resized_coordinate_mapping(app):
    app.window = pygame.display.set_mode((900, 900), pygame.RESIZABLE)
    view = app.viewport()
    position = (round(view.x + 640 * view.width / 1280),
                round(view.y + 325 * view.height / 720))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=position))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=position))
    assert app.screen == "modes"
    # SOLO now opens mode selection; verify the same resized mouse mapping there.
    position = (round(view.x + 640 * view.width / 1280),
                round(view.y + 303 * view.height / 720))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=position))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=position))
    assert app.screen == "countdown"
    for _ in range(3):
        app.update(1)
    assert app.screen == "game"
    for screen in ("game", "pause", "over", "menu", "settings", "ai"):
        app.action(screen)
        app.draw(0.016)


def test_terminal_sound_event_forwarded_once(app, monkeypatch):
    events = []
    monkeypatch.setattr(app.audio, "handle", lambda batch: events.extend(batch))
    app.action("start")
    app.game.snake.body = [(24, 0), (23, 0), (22, 0)]
    app.update(1)
    app.update(1)
    assert events == [GameEvent.LOST]
    assert app.screen == "over"


def test_missing_image_has_renderable_fallback(app):
    assert app.assets.image("missing.png", 20) is None
    app.assets.originals["apple.png"] = None
    app.action("start")
    app.draw(0.01)


def test_audio_device_failure_is_nonfatal(app, monkeypatch):
    from services.audio import AudioManager
    pygame.mixer.quit()
    def fail():
        raise pygame.error("No audio device")
    monkeypatch.setattr(pygame.mixer, "init", fail)
    audio = AudioManager(app.settings)
    audio.handle([GameEvent.ATE, GameEvent.LOST])
    audio.apply_settings()
    audio.close()
    assert not audio.ready
