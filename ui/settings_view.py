"""Preference controls, with values read directly from the current settings."""
import pygame
from ui.components import MUTED, PANEL, backdrop, button, label
from core.config import SPEEDS


def buttons(settings):
    result = []
    for index, (key, title) in enumerate((("music", "MUSIC"),
                                          ("sound_effects", "SOUND EFFECTS"),
                                          ("animations", "ANIMATIONS"))):
        result.append(button("ON" if getattr(settings, key) else "OFF", 824,
                             263 + index * 77, 112, key, getattr(settings, key)))
    for index, speed in enumerate(SPEEDS):
        result.append(button(speed, 615 + index * 110, 505, 100,
                             "speed:" + speed, settings.speed == speed))
    result.append(button("BACK", 500, 613, 280, "menu"))
    return result


def draw(surface):
    backdrop(surface, "S N A K E  /  PREFERENCES", "SETTINGS", "Make yourself at home.")
    for index, (title, hint) in enumerate((("Music", "Background soundtrack"),
                                          ("Sound effects", "Apple and game-over feedback"),
                                          ("Animations", "Buttons, overlays and apple feedback"))):
        y = 259 + index * 77
        label(surface, title, (344, y), 23, center=False)
        label(surface, hint, (344, y + 32), 15, MUTED, False)
    pygame.draw.line(surface, PANEL, (344, 487), (936, 487), 2)
    label(surface, "Game speed", (344, 503), 23, center=False)
    label(surface, "6 / 9 / 13 cells per second", (344, 539), 15, MUTED, False)
    label(surface, "Preferences save automatically.", (640, 587), 15, MUTED)
