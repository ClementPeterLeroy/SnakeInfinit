import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import time
from unittest.mock import Mock
import pygame
import pytest
from app import Application
from core.challenges import BY_ID
from core.config import ROOT
from core.types import Direction, GameEvent, Hazard
from services.audio import AudioManager
from services.persistence import Settings
from services.video import StartupVideo


@pytest.fixture
def application(tmp_path):
    app = Application(tmp_path / "settings.json")
    yield app
    app.close()


def test_countdown_synchronized_once_and_engine_frozen(application, monkeypatch):
    app = application
    sounds = []
    monkeypatch.setattr(app.audio, "countdown", sounds.append)
    app.action("play:arcade")
    state = app.game.get_state()
    for _ in range(5):
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        app.update(0.5)
        assert app.game.get_state() == state
        assert app.accumulator == 0 and app.screen == "countdown"
    app.update(0.5)
    assert app.screen == "game" and app.game.get_state() == state
    assert sounds == ["3", "2", "1", "GO!"]
    app.update(app.game.interval)
    assert app.game.snake.body[0] == (13, 12)
    assert sounds == ["3", "2", "1", "GO!"]


def test_challenge_countdown_pause_and_restart_keep_timers_frozen(application):
    app = application
    app.action("play:speed_demon")
    app.update(1)
    app.action("pause")
    app.update(100)
    assert app.game.elapsed == 0 and app.countdown.elapsed == 1
    app.action("resume")
    assert app.screen == "countdown"
    app.update(2)
    assert app.screen == "game" and app.game.elapsed == 0
    app.update(1 / 20)
    assert app.game.elapsed == 1 / 20
    app.action("pause")
    app.update(100)
    assert app.game.elapsed == 1 / 20
    app.action("play")
    assert app.screen == "countdown"
    assert app.game.rules == BY_ID["speed_demon"].rules
    assert app.game.elapsed == 0 and len(app.game.snake.body) == 50


def test_session_statistics_record_once_and_countdown_cancel_not_a_game(application):
    app = application
    app.action("play:classic")
    app.action("menu")
    assert app.records.data["statistics"]["games"] == 0
    app.action("play:classic")
    app.update(3)
    app.game.snake.body = [(24, 0), (23, 0), (22, 0)]
    app.update(app.game.interval)
    for _ in range(3):
        app.update(1)
        app.draw(0.1)
        app.finish_session()
    app.action("menu")
    assert app.records.data["statistics"]["games"] == 1
    assert app.records.data["statistics"]["deaths"] == 1


def test_name_entry_limit_confirmation_and_cancellation(application):
    app = application
    app.action("play:classic")
    app.update(3)
    app.game.score = 7
    app.game.snake.body = [(24, 0), (23, 0), (22, 0)]
    app.update(app.game.interval)
    assert app.screen == "name"
    app.handle_event(pygame.event.Event(pygame.TEXTINPUT, text="ABCDEFGHIJKLMNO"))
    assert app.player_name == "ABCDEFGHIJKL"
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
    assert app.screen == "over"
    assert app.records.data["leaderboards"]["classic_NORMAL"][0]["score"] == 7
    app.action("play")
    app.update(3)
    app.game.score = 8
    app.game.snake.body = [(24, 0), (23, 0), (22, 0)]
    app.update(app.game.interval)
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert len(app.records.data["leaderboards"]["classic_NORMAL"]) == 1
    assert app.records.data["statistics"]["games"] == 2


def test_all_new_screens_render_and_warnings_survive_animation_toggle(application):
    app = application
    for screen in ("modes", "challenges", "leaderboard", "settings"):
        app.action(screen)
        app.draw(0.2)
    app.action("challenges")
    app.action("challenge_page")
    app.draw(0.2)
    assert not any(control.action.startswith("play:") for control in app.controls)
    app.action("play:minefield")
    app.update(3)
    app.game.advance_time(8)
    app.settings.animations = False
    app.draw(0.2)
    assert len(app.game.hazards) == 1
    assert not app.game.hazards[0].active(app.game.elapsed)
    app.game.advance_time(1)
    app.draw(0.2)


@pytest.fixture
def mock_mixer(monkeypatch):
    mixer = Mock()
    mixer.get_init.return_value = (44100, -16, 2)
    mixer.Sound.side_effect = lambda path: Mock(name=path.name)
    monkeypatch.setattr(pygame, "mixer", mixer)
    return mixer


@pytest.mark.parametrize("score,voiced", [(0, True), (19, True), (20, False), (26, False)])
def test_game_over_voice_threshold_and_channel_priority(mock_mixer, score, voiced):
    audio = AudioManager(Settings())
    audio.game_score = score
    audio.handle([GameEvent.LOST])
    if voiced:
        audio.announcement_channel.play.assert_called_once_with(audio.voices["lost"])
        audio.sounds[GameEvent.LOST].play.assert_not_called()
    else:
        audio.announcement_channel.play.assert_not_called()
        audio.sounds[GameEvent.LOST].play.assert_called_once()


def test_voices_independent_of_effects_and_share_countdown_channel(mock_mixer):
    settings = Settings(sound_effects=False)
    audio = AudioManager(settings)
    audio.handle([GameEvent.SUPER])
    audio.announcement_channel.play.assert_called_once_with(audio.voices["super"])
    settings.sound_effects = True
    audio.countdown("3")
    audio.countdown("2")
    audio.handle([GameEvent.SUPER])
    assert audio.announcement_channel.play.call_count == 4
    assert audio.announcement_channel.stop.call_count >= 4
    mock_mixer.Channel.assert_called_once_with(0)
    settings.voice_announcements = False
    before = audio.announcement_channel.play.call_count
    audio.handle([GameEvent.SUPER])
    assert audio.announcement_channel.play.call_count == before
    audio.countdown("GO!")
    assert audio.announcement_channel.play.call_count == before + 1


def test_missing_voice_falls_back_to_ordinary_sound(mock_mixer):
    def load(path):
        if path.name.endswith("voice.wav"):
            raise FileNotFoundError("voice missing")
        return Mock()
    mock_mixer.Sound.side_effect = load
    audio = AudioManager(Settings())
    audio.game_score = 2
    audio.handle([GameEvent.LOST, GameEvent.SUPER])
    audio.sounds[GameEvent.LOST].play.assert_called_once()


def test_startup_failure_continues_to_menu_once(application, monkeypatch):
    class BrokenVideo:
        def __init__(self, path):
            self.closed = False
        def update(self, dt):
            return True
        def close(self):
            self.closed = True
    monkeypatch.setattr("app.StartupVideo", BrokenVideo)
    app = application
    app.start_intro()
    assert app.screen == "intro"
    video = app.video
    app.update(0.1)
    assert app.screen == "menu" and app.video is None and video.closed
    app.start_intro()
    assert app.screen == "menu"


def test_real_supplied_video_decodes_and_resources_close():
    video = StartupVideo(ROOT / "assets/px/SnakeInfinite_Startup.mp4")
    started = time.monotonic()
    finished = False
    while not finished and time.monotonic() - started < 15:
        finished = video.update(0.04)
        time.sleep(0.002)  # Let the bounded decoder produce frames.
    assert finished and video.error is None
    assert video.current is not None and video.current.get_size() == (1280, 512)
    assert video.duration == pytest.approx(6)
    video.close()
    assert not video.thread.is_alive()


def test_missing_video_reports_failure_without_crashing(tmp_path):
    video = StartupVideo(tmp_path / "missing.mp4")
    video.thread.join(timeout=5)
    assert video.update(0.01) and video.error is not None
    video.close()


def test_startup_crossfade_releases_frame_and_respects_animations(application):
    app = application
    app.intro_exit_frame = pygame.Surface((1280, 720))
    app.intro_fade_remaining = 0.25
    app.draw(0.1)
    assert app.intro_fade_remaining == pytest.approx(0.15)
    app.draw(0.2)
    assert app.intro_exit_frame is None
    app.intro_exit_frame = pygame.Surface((1280, 720))
    app.settings.animations = False
    app.draw(0.01)
    assert app.intro_exit_frame is None


@pytest.mark.parametrize("fps", [30, 60, 120, 240])
def test_speed_demon_moves_exactly_twenty_cells_per_active_second(application, fps):
    app = application
    app.action("play:speed_demon")
    app.update(3)
    app.game.apple = (0, 0)
    head = app.game.snake.body[0]
    for _ in range(fps):
        app.update(1 / fps)
    assert app.game.snake.body[0] == (head[0], head[1] + 20)
    assert app.game.elapsed == pytest.approx(1)


def test_countdown_and_pause_freeze_mine_detonation(application):
    app = application
    app.action("play:minefield")
    app.game.hazards = [Hazard((18, 18), "mine", 0, 1)]
    app.game.next_hazard_at = None
    app.update(2.9)
    assert app.game.elapsed == 0 and not app.game.explosions
    app.update(0.1)
    app.update(0.5)
    time_before_pause = app.game.elapsed
    app.action("pause")
    app.update(100)
    assert app.game.elapsed == time_before_pause and not app.game.explosions
    app.action("resume")
    app.update(0.6)
    assert app.game.explosions and not app.game.hazards


def test_explosion_feedback_remains_clear_without_animations_and_records_one_death(application):
    app = application
    app.action("play:minefield")
    app.update(3)
    app.game.hazards = [Hazard((9, 12), "mine", 0, app.game.interval)]
    app.game.next_hazard_at = None
    app.settings.animations = False
    app.update(app.game.interval)
    assert app.screen == "over" and app.game.death_cause == "mine_explosion"
    app.draw(0.01)
    assert app.feedback.explosion_remaining == 0
    assert app.feedback.explosion_notice_remaining > 0 and app.feedback.explosions
    app.update(100)
    assert app.records.data["statistics"]["deaths"] == 1
    assert app.records.data["challenges"]["minefield"]["last_death_cause"] == "mine_explosion"
