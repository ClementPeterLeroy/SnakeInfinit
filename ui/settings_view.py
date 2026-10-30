"""Preference controls, with values read directly from the current settings."""
import pygame
from ui.components import MUTED, PANEL, backdrop, button, label
from core.config import SPEEDS


def buttons(settings):
    result = []
    for index, (key, title) in enumerate((("music", "MUSIC"),
                                          ("sound_effects", "SOUND EFFECTS"),
                                          ("animations", "ANIMATIONS"),
                                          ("voice_announcements", "VOICE"))):
        result.append(button("ON" if getattr(settings, key) else "OFF", 824,
                             230 + index * 69, 112, key, getattr(settings, key)))
    for index, speed in enumerate(SPEEDS):
        result.append(button(speed, 615 + index * 110, 528, 100,
                             "speed:" + speed, settings.speed == speed))
    result.append(button("BACK", 500, 632, 280, "menu"))
    return result


def draw(surface):
    backdrop(surface, "S N A K E  /  PREFERENCES", "SETTINGS", "Make yourself at home.")
    for index, (title, hint) in enumerate((("Music", "Background soundtrack"),
                                          ("Sound effects", "Apple and game-over feedback"),
                                          ("Animations", "Particles, transitions and feedback"),
                                          ("Voice announcements", "SUPER and game-over voice"))):
        y = 226 + index * 69
        label(surface, title, (344, y), 23, center=False)
        label(surface, hint, (344, y + 32), 15, MUTED, False)
    pygame.draw.line(surface, PANEL, (344, 510), (936, 510), 2)
    label(surface, "Classic speed", (344, 526), 23, center=False)
    label(surface, "6 / 9 / 13 cells per second", (344, 562), 15, MUTED, False)
    label(surface, "Preferences save automatically. Countdown audio follows Sound Effects.", (640, 603), 15, MUTED)
