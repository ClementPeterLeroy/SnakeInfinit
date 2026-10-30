"""Render at viewport resolution so cells and grid lines use whole pixels."""
import pygame
import math
from core.challenges import BY_ID
from core.hazards import explosion_cells
from ui.components import BG, BORDER, GREEN, MUTED, PANEL, label


def board_layout(size, board):
    width, height = size
    scale = min(width / 1280, height / 720)
    top, bottom = round(94 * scale), round(60 * scale)
    cell = max(1, int(min((width - 64 * scale) / board.width,
                          (height - top - bottom) / board.height)))
    extent = (cell * board.width, cell * board.height)
    return pygame.Rect((width - extent[0]) // 2,
                       top + (height - top - bottom - extent[1]) // 2, *extent), cell


def draw(surface, game, assets, feedback=None):
    surface.fill(BG)
    width, height = surface.get_size()
    scale = min(width / 1280, height / 720)

    def text(value, x, y, size=20, color=MUTED):
        label(surface, value, (round(x * scale), round(y * scale)),
              max(1, round(size * scale)), color)

    mode = BY_ID[game.rules.challenge_id].name if game.rules.challenge_id else game.rules.mode.upper()
    text(mode, 146, 44, 20, GREEN)
    if game.rules.wall_enabled:
        text(f"WALLS: {sum(h.kind == 'wall' and h.active(game.elapsed) for h in game.hazards)}", 146, 70, 15)
    elif game.rules.mines_enabled:
        text("MINES / 3×3 BLAST", 146, 70, 15, (239, 159, 94))
    text("ESC  /  PAUSE", 1140, 44, 16)
    score_box = pygame.Rect(round(width / 2 - 92 * scale), round(13 * scale),
                            max(1, round(184 * scale)), max(1, round(64 * scale)))
    pygame.draw.rect(surface, PANEL, score_box, border_radius=max(1, round(10 * scale)))
    text("SCORE", 640, 29, 12)
    text(f"{game.score:02d}", 640, 54, 29, GREEN)
    board, cell = board_layout(surface.get_size(), game.board)
    pygame.draw.rect(surface, PANEL, board)
    for index in range(game.board.width + 1):
        x = board.left + index * cell
        pygame.draw.line(surface, (32, 40, 36), (x, board.top), (x, board.bottom))
    for index in range(game.board.height + 1):
        y = board.top + index * cell
        pygame.draw.line(surface, (32, 40, 36), (board.left, y), (board.right, y))
    pygame.draw.rect(surface, BORDER, board.inflate(2, 2), 1)
    for hazard in game.hazards:
        x, y = hazard.cell
        rect = pygame.Rect(board.x + x * cell + 1, board.y + y * cell + 1, max(1, cell - 2), max(1, cell - 2))
        active = hazard.active(game.elapsed)
        color = (102, 121, 116) if hazard.kind == "wall" else (235, 76, 90)
        if not active:
            pygame.draw.rect(surface, (240, 187, 76), rect, max(1, cell // 10), border_radius=2)
            label(surface, "!", rect.center, max(8, round(cell * 0.7)), (255, 214, 108))
        elif hazard.kind == "wall":
            pygame.draw.rect(surface, color, rect, border_radius=2)
        else:
            pygame.draw.circle(surface, color, rect.center, max(1, cell // 3))
            pygame.draw.line(surface, color, rect.topleft, rect.bottomright, max(1, cell // 10))
            pygame.draw.line(surface, color, rect.topright, rect.bottomleft, max(1, cell // 10))
            if hazard.expires_at is not None and hazard.expires_at - game.elapsed <= 1:
                for px, py in explosion_cells(game.board, hazard.cell):
                    warning = pygame.Rect(board.x + px * cell, board.y + py * cell, cell, cell)
                    pygame.draw.rect(surface, (154, 83, 56), warning, 1)
    if feedback and feedback.explosion_notice_remaining > 0:
        for blast in feedback.explosions:
            for x, y in blast.cells:
                rect = pygame.Rect(board.x + x * cell, board.y + y * cell, cell, cell)
                if feedback.explosion_remaining > 0:
                    glow = pygame.Surface((cell, cell), pygame.SRCALPHA)
                    glow.fill((255, 102, 45, round(150 * feedback.explosion_remaining / 0.4)))
                    surface.blit(glow, rect)
                pygame.draw.rect(surface, (242, 130, 70), rect, 1)
        text("MINE DETONATED — INSTANT BLAST", 640, 94, 16, (255, 160, 102))
    if game.star is not None:
        x, y = game.star
        center = (board.x + x * cell + cell // 2, board.y + y * cell + cell // 2)
        sprite = assets.image("star.png", max(1, cell - 2))
        if sprite is not None:
            surface.blit(sprite, sprite.get_rect(center=center))
        else:
            label(surface, "*", center, max(8, cell), (255, 218, 93))
        ring = pygame.Rect(0, 0, cell + 4, cell + 4)
        ring.center = center
        pygame.draw.arc(surface, (255, 218, 93), ring, -math.pi / 2,
                        -math.pi / 2 + math.tau * game.star_remaining / game.rules.star_lifetime, max(1, cell // 12))
        text(f"STAR {game.star_remaining:.1f}s", 1050, 70, 15, (255, 218, 93))
    if game.apple is not None:
        x, y = game.apple
        center = (board.x + x * cell + cell // 2, board.y + y * cell + cell // 2)
        apple = assets.image("apple.png", max(1, cell - max(2, round(cell * 0.15))))
        if apple is not None:
            surface.blit(apple, apple.get_rect(center=center))
        else:
            pygame.draw.circle(surface, (236, 95, 92), center, max(1, cell // 3))
    if feedback and feedback.apple_remaining and feedback.eaten_cell is not None:
        x, y = feedback.eaten_cell
        center = (board.x + x * cell + cell // 2, board.y + y * cell + cell // 2)
        strength = feedback.apple_remaining / 0.28
        tint = (255, 218, 93) if feedback.star_effect else GREEN
        color = tuple(round(PANEL[i] + (tint[i] - PANEL[i]) * strength) for i in range(3))
        pygame.draw.circle(surface, color, center, max(1, round(cell * (1.2 - strength * 0.5))), 1)
        for index in range(10):
            angle = index * math.tau / 10
            radius = cell * (1 - strength) * 1.7
            position = (round(center[0] + math.cos(angle) * radius), round(center[1] + math.sin(angle) * radius))
            pygame.draw.circle(surface, color, position, max(1, round(cell * 0.08 * strength)))
        label(surface, "+2" if feedback.star_effect else "+1",
              (center[0], round(center[1] - cell * (1.2 - strength))), max(8, round(16 * scale)), color)
    for index, (x, y) in enumerate(game.snake.body):
        gap = max(1, round(cell * 0.06)) if cell > 3 else 0
        rect = pygame.Rect(board.x + x * cell + gap, board.y + y * cell + gap,
                           max(1, cell - 2 * gap), max(1, cell - 2 * gap))
        color = (176, 244, 190) if index == 0 else GREEN
        pygame.draw.rect(surface, color, rect, border_radius=max(1, round(cell * 0.24)))
    text("MOVE  Arrow keys / WASD / ZQSD", 640, 691, 16)
    if game.rules.star_target:
        text(f"STARS: {game.stars_collected} / {game.rules.star_target}", 640, 669, 18, (255, 218, 93))
    elif game.rules.survival_target:
        remaining = max(0, math.ceil(game.rules.survival_target - game.elapsed))
        text(f"START: {game.rules.initial_length} SEGMENTS / {1 / game.interval:g} BLOCKS/SEC / SURVIVE: {remaining // 60:02d}:{remaining % 60:02d}", 640, 669, 16, GREEN)
    elif game.rules.score_target:
        text(f"TARGET: {game.score} / {game.rules.score_target}", 640, 669, 18, GREEN)
    elif game.rules.mode == "arcade":
        text(f"{1 / game.interval:g} CELLS / SECOND", 640, 669, 16, MUTED)
    if feedback and feedback.apple_remaining:
        text("+2" if feedback.star_effect else "+1", 756, 52, 19, (255, 218, 93) if feedback.star_effect else GREEN)
    if feedback and feedback.celebration_remaining:
        text("NEW HIGH SCORE!", 640, 94, 20, (255, 218, 93))
        for index in range(12):
            x = round(width * (index + 1) / 13)
            y = round(100 * scale + (0.8 - feedback.celebration_remaining) * 100)
            pygame.draw.circle(surface, (255, 218, 93), (x, y), max(1, round(2 * scale)))
