"""Main menu and the intentionally informational AI placeholder."""
from core.config import VERSION, SUBTITLE
from ui.components import BG, GREEN, MUTED, backdrop, button, label


def buttons(ai=False):
    if ai:
        return [button("BACK", 500, 420, 280, "menu")]
    return [button("SOLO", 500, 300, 280, "modes", True),
            button("AI", 500, 364, 280, "ai"),
            button("LEADERBOARD", 500, 428, 280, "leaderboard"),
            button("SETTINGS", 500, 492, 280, "settings"),
            button("EXIT", 1110, 632, 120, "exit")]


def draw(surface, ai=False, assets=None):
    if ai:
        backdrop(surface, "S N A K E  /  FUTURE EXPLORATION", "AI LABORATORY")
        label(surface, "Neural network training and AI spectator mode", (640, 300), 24)
        label(surface, "will be available in a future update.", (640, 339), 24, MUTED)
        return
    surface.fill(BG)
    logo = assets.fit("logo.png", (740, 255)) if assets is not None else None
    if logo is not None:
        surface.blit(logo, logo.get_rect(center=(640, 155)))
    else:
        label(surface, "Official logo unavailable", (640, 180), 24, MUTED)
    label(surface, "ONE MORE APPLE. INFINITE POSSIBILITIES.", (640, 584), 14, MUTED)
    label(surface, f"{VERSION}  /  {SUBTITLE}", (48, 650), 16, MUTED, False)
