"""Render at viewport resolution so cells and grid lines use whole pixels."""
import pygame
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

    text("S N A K E", 132, 44, 23, GREEN)
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
        color = tuple(round(PANEL[i] + (GREEN[i] - PANEL[i]) * strength) for i in range(3))
        pygame.draw.circle(surface, color, center, max(1, round(cell * (1.2 - strength * 0.5))), 1)
    for index, (x, y) in enumerate(game.snake.body):
        gap = max(1, round(cell * 0.06)) if cell > 3 else 0
        rect = pygame.Rect(board.x + x * cell + gap, board.y + y * cell + gap,
                           max(1, cell - 2 * gap), max(1, cell - 2 * gap))
        color = (176, 244, 190) if index == 0 else GREEN
        pygame.draw.rect(surface, color, rect, border_radius=max(1, round(cell * 0.24)))
    text("MOVE  Arrow keys / WASD / ZQSD", 640, 691, 16)
    if feedback and feedback.apple_remaining:
        text("+1", 756, 52, 19, GREEN)
