"""Draw the square board and HUD without changing simulation state."""
import pygame
from ui.components import BG, BORDER, GREEN, MUTED, PANEL, label


def draw(surface, game, assets):
    surface.fill(BG)
    label(surface, "S N A K E", (52, 32), 23, GREEN, False)
    label(surface, f"SCORE  {game.score:02d}", (640, 45), 24)
    label(surface, "ESC  /  PAUSE", (1080, 34), 16, MUTED, False)
    cell = 22
    extent = cell * game.board.size
    board = pygame.Rect((1280 - extent) // 2, 100, extent, extent)
    pygame.draw.rect(surface, PANEL, board)
    for index in range(game.board.size + 1):
        x = board.left + index * cell
        y = board.top + index * cell
        pygame.draw.line(surface, (32, 40, 36), (x, board.top), (x, board.bottom))
        pygame.draw.line(surface, (32, 40, 36), (board.left, y), (board.right, y))
    pygame.draw.rect(surface, BORDER, board.inflate(2, 2), 1)
    if game.apple is not None:
        x, y = game.apple
        center = (board.x + x * cell + cell // 2, board.y + y * cell + cell // 2)
        apple = assets.image("apple.png", cell - 4)
        if apple is not None:
            surface.blit(apple, apple.get_rect(center=center))
        else:
            pygame.draw.circle(surface, (236, 95, 92), center, cell // 3)
    for index, (x, y) in enumerate(game.snake.body):
        rect = pygame.Rect(board.x + x * cell + 1, board.y + y * cell + 1, cell - 2, cell - 2)
        pygame.draw.rect(surface, (164, 242, 180) if index == 0 else GREEN, rect, border_radius=5)
    label(surface, "MOVE  Arrow keys / WASD / ZQSD", (640, 688), 16, MUTED)
