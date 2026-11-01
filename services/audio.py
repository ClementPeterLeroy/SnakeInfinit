"""One audio owner; the application forwards simulation events here."""
import logging
import pygame
from services.resources import resource_path
from core.types import GameEvent


class AudioManager:
    def __init__(self, settings):
        self.settings = settings
        self.ready = False
        self.music_loaded = False
        self.music_started = False
        self.music_paused = False
        self.music_path = resource_path("assets", "snd", "synth_wave_background.wav")
        self.sounds = {}
        self.voices = {}
        self.counts = {}
        self.game_score = 0
        self.suspended = False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.ready = True
            pygame.mixer.set_reserved(2)
            self.announcement_channel = pygame.mixer.Channel(0)
        except (pygame.error, OSError) as error:
            logging.warning("Audio unavailable: %s", error)
            return
        for event, name in ((GameEvent.ATE, "eating_apple.wav"),
                            (GameEvent.LOST, "game_over.wav"),
                            (GameEvent.STAR, "eating_star.wav")):
            try:
                self.sounds[event] = pygame.mixer.Sound(resource_path("assets", "snd", name))
                self.sounds[event].set_volume(0.65)
            except (pygame.error, OSError) as error:
                logging.warning("Sound unavailable: %s", error)
        for table, entries in ((self.voices, (("super", "super_voice.wav"), ("lost", "game_over_voice.wav"))),
                               (self.counts, tuple((word, word.lower().replace("!", "") + ".wav")
                                                  for word in ("3", "2", "1", "GO!")))):
            for key, name in entries:
                try:
                    table[key] = pygame.mixer.Sound(resource_path("assets", "snd", name))
                except (pygame.error, OSError) as error:
                    logging.warning("Announcement unavailable (%s): %s", name, error)
        try:
            pygame.mixer.music.load(self.music_path)
            pygame.mixer.music.set_volume(0.3)
            self.music_loaded = True
            logging.info("Background music loaded: %s; mixer=%s", self.music_path, pygame.mixer.get_init())
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
            if not self.settings.voice_announcements:
                self.announcement_channel.stop()
            if not self.music_loaded:
                return
            if self.settings.music and not self.suspended:
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
        if self.ready and pygame.mixer.get_init():
            try:
                for event in events:
                    if event == GameEvent.SUPER:
                        self.announce("super")
                    low_death = event == GameEvent.LOST and self.game_score < 20
                    if low_death and self.announce("lost"):
                        continue
                    if event in self.sounds and self.settings.sound_effects:
                        self.sounds[event].play()
            except (pygame.error, OSError) as error:
                self._disable(error)

    def announce(self, name):
        if not self.ready or not self.settings.voice_announcements or name not in self.voices:
            return False
        try:
            self.announcement_channel.stop()
            self.announcement_channel.play(self.voices[name])
            return True
        except pygame.error as error:
            self._disable(error)
            return False

    def countdown(self, word):
        self.stop_announcements()
        if self.ready and self.settings.sound_effects and word in self.counts:
            try:
                self.announcement_channel.play(self.counts[word])
            except pygame.error as error:
                self._disable(error)

    def stop_announcements(self):
        if self.ready and pygame.mixer.get_init():
            try:
                self.announcement_channel.stop()
            except pygame.error as error:
                self._disable(error)

    def suspend_music(self, suspended):
        self.suspended = suspended
        self.apply_settings()

    def close(self):
        if pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
                pygame.mixer.stop()
            except pygame.error as error:
                logging.warning("Audio shutdown unavailable: %s", error)
        self.ready = False
