"""Main menu and optional artificial-intelligence section."""
from core.config import VERSION, SUBTITLE
from ui.components import BG, GREEN, MUTED, backdrop, button, label


def buttons(ai=False):
    if ai:
        return [button("NEURAL LABORATORY", 440, 292, 400, "open_neural", True),
                button("TRAINING (COMING SOON)", 440, 362, 400, "training"),
                button("BACK", 500, 452, 280, "menu")]
    return [button("SOLO", 500, 300, 280, "modes", True),
            button("AI", 500, 364, 280, "ai"),
            button("LEADERBOARD", 500, 428, 280, "leaderboard"),
            button("SETTINGS", 500, 492, 280, "settings"),
            button("EXIT", 1110, 632, 120, "exit")]


def draw(surface, ai=False, assets=None):
    if ai:
        from ai.runtime import torch_available
        backdrop(surface, "SNAKE INFINITE / PHASE II", "ARTIFICIAL INTELLIGENCE")
        message = "Inspect real CPU PyTorch networks. Training arrives in v1.4." if torch_available() else "Optional PyTorch required: install requirements-ai.txt. Arcade remains available."
        label(surface, message, (640, 550), 18, MUTED)
        return
    surface.fill(BG)
    logo = assets.fit("logo.png", (740, 255)) if assets is not None else None
    if logo is not None:
        surface.blit(logo, logo.get_rect(center=(640, 155)))
    else:
        label(surface, "Official logo unavailable", (640, 180), 24, MUTED)
    label(surface, "ONE MORE APPLE. INFINITE POSSIBILITIES.", (640, 584), 14, MUTED)
    label(surface, f"{VERSION}  /  {SUBTITLE}", (48, 650), 16, MUTED, False)
