"""Exercise audio lifecycle through mocks without requiring speakers or files."""
from unittest.mock import Mock
import pygame
import pytest
from core.types import GameEvent
from services.audio import AudioManager
from services.persistence import Settings


@pytest.fixture
def mixer(monkeypatch):
    fake = Mock()
    fake.get_init.return_value = (44100, -16, 2)
    monkeypatch.setattr(pygame, "mixer", fake)
    return fake


def test_wav_loops_once_and_toggles_resume_without_restart(mixer):
    settings = Settings()
    audio = AudioManager(settings)
    assert audio.music_path.name == "synth_wave_background.wav"
    mixer.music.load.assert_called_once_with(audio.music_path)
    mixer.music.play.assert_called_once_with(-1)
    for _ in range(5):
        audio.apply_settings()
    mixer.music.play.assert_called_once_with(-1)
    mixer.music.unpause.assert_not_called()
    settings.music = False
    audio.apply_settings()
    audio.apply_settings()
    mixer.music.pause.assert_called_once()
    settings.music = True
    audio.apply_settings()
    mixer.music.unpause.assert_called_once()
    mixer.music.play.assert_called_once_with(-1)


def test_initially_disabled_music_starts_only_when_enabled(mixer):
    settings = Settings(music=False)
    audio = AudioManager(settings)
    mixer.music.play.assert_not_called()
    settings.music = True
    audio.apply_settings()
    mixer.music.play.assert_called_once_with(-1)


def test_missing_music_preserves_sound_effects(mixer):
    mixer.music.load.side_effect = FileNotFoundError("No soundtrack")
    audio = AudioManager(Settings())
    assert audio.ready and not audio.music_loaded
    audio.apply_settings()
    mixer.music.play.assert_not_called()
    audio.handle([GameEvent.ATE])
    audio.sounds[GameEvent.ATE].play.assert_called_once()


def test_missing_effects_do_not_disable_music(mixer):
    mixer.Sound.side_effect = pygame.error("Missing sound")
    audio = AudioManager(Settings())
    assert audio.sounds == {} and audio.music_started
    audio.handle([GameEvent.ATE, GameEvent.LOST])


def test_no_audio_device_is_safe(mixer):
    mixer.get_init.return_value = None
    mixer.init.side_effect = pygame.error("No device")
    audio = AudioManager(Settings())
    assert not audio.ready
    audio.handle([GameEvent.ATE])
    audio.apply_settings()
    audio.close()
    mixer.music.play.assert_not_called()


def test_playback_failure_is_nonfatal(mixer):
    mixer.music.play.side_effect = pygame.error("Device disconnected")
    audio = AudioManager(Settings())
    assert not audio.ready
    audio.handle([GameEvent.LOST])
    audio.close()
    audio.close()


def test_effect_toggle_stops_active_sounds(mixer):
    settings = Settings()
    audio = AudioManager(settings)
    settings.sound_effects = False
    audio.apply_settings()
    audio.handle([GameEvent.ATE])
    audio.sounds[GameEvent.ATE].stop.assert_called()
    audio.sounds[GameEvent.ATE].play.assert_not_called()
