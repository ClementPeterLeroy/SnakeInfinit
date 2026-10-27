"""Modal pause and game-end overlays."""
import pygame
from ui.components import BORDER, MUTED, PANEL, button, label


def buttons(paused):
    if paused:
        return [button("RESUME", 500, 326, 280, "resume", True),
                button("RESTART", 500, 390, 280, "start"),
                button("MAIN MENU", 500, 454, 280, "menu")]
    return [button("PLAY AGAIN", 500, 358, 280, "start", True),
            button("MAIN MENU", 500, 424, 280, "menu")]


def draw(surface, game, paused):
    shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    shade.fill((8, 12, 10, 200))
    surface.blit(shade, (0, 0))
    rect = pygame.Rect(440, 192, 400, 352)
    pygame.draw.rect(surface, PANEL, rect, border_radius=18)
    pygame.draw.rect(surface, BORDER, rect, 1, border_radius=18)
    title = "PAUSED" if paused else "BOARD COMPLETE" if game.apple is None else "GAME OVER"
    label(surface, title, (640, 255), 33)
    label(surface, "Take a breath." if paused else f"Score: {game.score}", (640, 301), 21, MUTED)
