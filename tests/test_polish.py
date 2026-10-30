import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import pytest
from app import Application
from core.board import Board
from core.game import Game
from core.types import Direction
from ui.game_view import board_layout


@pytest.fixture
def application(tmp_path):
    app = Application(tmp_path / "settings.json")
    yield app
    app.close()


def press(app, key, **kwargs):
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, **kwargs))


def test_two_fast_corners_execute_on_separate_ticks(application):
    app = application
    app.action("start")
    press(app, pygame.K_UP)
    press(app, pygame.K_LEFT)
    assert app.game.snake.body[0] == (12, 12)
    app.update(app.game.interval)
    assert app.game.snake.body[0] == (12, 11)
    app.update(app.game.interval)
    assert app.game.snake.body[0] == (11, 11)


def test_reversals_repeats_and_buffer_overflow_are_ignored(application):
    app = application
    app.action("start")
    press(app, pygame.K_LEFT)  # Immediate reversal.
    assert app.game.snake.pending is None
    press(app, pygame.K_UP)
    press(app, pygame.K_DOWN)  # Reverses the buffered turn.
    assert app.queued_turn is None
    press(app, pygame.K_RIGHT, repeat=True)
    assert app.queued_turn is None
    press(app, pygame.K_LEFT)
    press(app, pygame.K_RIGHT)  # Cannot overwrite a valid deferred corner.
    assert app.queued_turn == Direction.LEFT


def test_pause_and_restart_clear_fractional_time_and_buffer(application):
    app = application
    app.action("start")
    app.update(app.game.interval * 0.9)
    press(app, pygame.K_UP)
    press(app, pygame.K_LEFT)
    press(app, pygame.K_ESCAPE)
    app.update(100)
    press(app, pygame.K_ESCAPE)
    assert app.game.snake.pending is None and app.queued_turn is None
    app.update(app.game.interval * 0.2)
    assert app.game.snake.body[0] == (12, 12)
    app.action("start")
    assert app.accumulator == 0 and app.queued_turn is None


@pytest.mark.parametrize("fps", [30, 60, 120, 240])
def test_render_rate_never_accelerates_simulation(application, fps):
    app = application
    app.action("start")
    app.game = Game(seed=7)
    app.game.apple = (0, 0)
    for _ in range(fps):
        app.update(1 / fps)
    assert app.game.snake.body[0] == (21, 12)


@pytest.mark.parametrize("key,direction", [
    (pygame.K_a, Direction.LEFT), (pygame.K_q, Direction.LEFT),
    (pygame.K_LEFT, Direction.LEFT), (pygame.K_s, Direction.DOWN),
    (pygame.K_DOWN, Direction.DOWN), (pygame.K_d, Direction.RIGHT),
    (pygame.K_RIGHT, Direction.RIGHT)])
def test_remaining_keyboard_aliases(application, key, direction):
    app = application
    app.action("start")
    app.game.snake.direction = Direction.UP
    if direction == Direction.DOWN:
        app.game.snake.direction = Direction.RIGHT
    press(app, key)
    assert app.game.snake.pending == direction


def test_feedback_disabled_immediately_without_affecting_rules(application):
    app = application
    app.action("start")
    app.game.apple = (13, 12)
    app.update(app.game.interval)
    assert app.feedback.apple_remaining > 0
    assert app.game.score == 1
    state = app.game.get_state()
    app.settings.animations = False
    app.draw(0.01)
    assert app.feedback.apple_remaining == 0
    assert app.game.get_state() == state
    app.action("pause")
    app.draw(0.01)
    assert app.feedback.overlay_opacity == 1


def test_overlay_fade_and_music_survive_navigation(application, monkeypatch):
    app = application
    calls = []
    monkeypatch.setattr(pygame.mixer.music, "play", lambda *args: calls.append(args))
    app.action("start")
    app.action("pause")
    assert app.feedback.overlay_opacity == 0
    app.draw(0.08)
    assert app.feedback.overlay_opacity == 0.5
    app.draw(0.08)
    assert app.feedback.overlay_opacity == 1
    for screen in ("menu", "settings", "ai", "menu"):
        app.action(screen)
        app.draw(0.01)
    assert calls == []


@pytest.mark.parametrize("size", [(1280, 720), (853, 480), (1920, 1080), (640, 360)])
def test_board_uses_integer_square_cells_and_stays_centered(size):
    board = Board(31, 17)
    rect, cell = board_layout(size, board)
    assert isinstance(cell, int)
    assert rect.width == 31 * cell and rect.height == 17 * cell
    assert abs(rect.centerx - size[0] // 2) <= 1
    assert pygame.Rect((0, 0), size).contains(rect)
