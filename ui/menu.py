"""Main menu and the intentionally informational AI placeholder."""
from core.config import VERSION, SUBTITLE
from ui.components import BG, GREEN, MUTED, backdrop, button, label


def buttons(ai=False):
    if ai:
        return [button("BACK", 500, 420, 280, "menu")]
    return [button("SOLO", 500, 300, 280, "start", True),
            button("AI", 500, 366, 280, "ai"),
            button("SETTINGS", 500, 432, 280, "settings"),
            button("EXIT", 1110, 632, 120, "exit")]


def draw(surface, ai=False):
    if ai:
        backdrop(surface, "S N A K E  /  FUTURE EXPLORATION", "AI LABORATORY")
        label(surface, "Neural network training and AI spectator mode", (640, 300), 24)
        label(surface, "will be available in a future update.", (640, 339), 24, MUTED)
        return
    surface.fill(BG)
    label(surface, "THE CLASSIC. A NEW FOUNDATION.", (640, 172), 15, GREEN)
    label(surface, "S N A K E", (640, 232), 64)
    label(surface, "25 × 25  /  ONE MORE APPLE", (640, 535), 15, MUTED)
    label(surface, f"{VERSION}  /  {SUBTITLE}", (48, 650), 16, MUTED, False)
