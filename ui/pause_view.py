"""Modal pause and game-end overlays."""
import pygame
from ui.components import BORDER, MUTED, PANEL, button, label


def buttons(paused):
    if paused:
        return [button("RESUME", 500, 326, 280, "resume", True),
                button("RESTART", 500, 390, 280, "play"),
                button("MAIN MENU", 500, 454, 280, "menu")]
    return [button("PLAY AGAIN", 500, 432, 280, "play", True),
            button("MAIN MENU", 500, 496, 280, "menu")]


def draw(surface, game, paused, high_score=False):
    shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    shade.fill((8, 12, 10, 200))
    surface.blit(shade, (0, 0))
    rect = pygame.Rect(384, 176, 512, 400)
    pygame.draw.rect(surface, PANEL, rect, border_radius=18)
    pygame.draw.rect(surface, BORDER, rect, 1, border_radius=18)
    from core.types import GameStatus
    from core.challenges import BY_ID
    victory = game.status == GameStatus.WON
    title = "PAUSED" if paused else "CHALLENGE COMPLETE!" if victory and game.rules.challenge_id else "BOARD COMPLETE" if victory else "GAME OVER"
    label(surface, title, (640, 255), 33)
    if not paused and game.death_cause == "mine_explosion":
        label(surface, "MINE EXPLOSION", (640, 211), 16, (255, 160, 102))
    label(surface, "Take a breath." if paused else f"Score: {game.score}", (640, 301), 21, MUTED)
    if not paused:
        if game.rules.challenge_id:
            label(surface, BY_ID[game.rules.challenge_id].name, (640, 340), 18)
        label(surface, f"{game.elapsed:.1f}s active / {game.apples_eaten} apples / {game.stars_collected} stars", (640, 372), 17, MUTED)
        if high_score:
            label(surface, "NEW HIGH SCORE!", (640, 404), 18, (245, 211, 97))
