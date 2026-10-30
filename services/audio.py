"""One audio owner; the application forwards simulation events here."""
import logging
import pygame
from core.config import ROOT
from core.types import GameEvent


class AudioManager:
    def __init__(self, settings):
        self.settings = settings
        self.ready = False
        self.music_loaded = False
        self.music_started = False
        self.music_paused = False
        self.music_path = ROOT / "assets" / "snd" / "synth_wave_background.wav"
        self.sounds = {}
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.ready = True
        except (pygame.error, OSError) as error:
            logging.warning("Audio unavailable: %s", error)
            return
        for event, name in ((GameEvent.ATE, "eating_apple.wav"),
                            (GameEvent.LOST, "game_over.wav")):
            try:
                self.sounds[event] = pygame.mixer.Sound(ROOT / "assets" / "snd" / name)
                self.sounds[event].set_volume(0.65)
            except (pygame.error, OSError) as error:
                logging.warning("Sound unavailable: %s", error)
        try:
            pygame.mixer.music.load(self.music_path)
            pygame.mixer.music.set_volume(0.3)
            self.music_loaded = True
        except (pygame.error, OSError) as error:
            logging.warning("Music unavailable: %s", error)
        self.apply_settings()

    def apply_settings(self):
        if not self.ready or not pygame.mixer.get_init():
            return
        try:
            if not self.settings.sound_effects:
                for sound in self.sounds.values():
                    sound.stop()
            if not self.music_loaded:
                return
            if self.settings.music:
                if not self.music_started:
                    pygame.mixer.music.play(-1)
                    self.music_started = True
                elif self.music_paused:
                    pygame.mixer.music.unpause()
                self.music_paused = False
            elif self.music_started and not self.music_paused:
                pygame.mixer.music.pause()
                self.music_paused = True
        except (pygame.error, OSError) as error:
            self._disable(error)

    def _disable(self, error):
        logging.warning("Audio playback unavailable: %s", error)
        self.ready = False

    def handle(self, events):
        if self.ready and pygame.mixer.get_init() and self.settings.sound_effects:
            try:
                for event in events:
                    if event in self.sounds:
                        self.sounds[event].play()
            except (pygame.error, OSError) as error:
                self._disable(error)

    def close(self):
        if pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
                pygame.mixer.stop()
            except pygame.error as error:
                logging.warning("Audio shutdown unavailable: %s", error)
        self.ready = False
