"""Deterministic step-based rules. No Pygame, clocks, audio, or UI imports."""
import random
from core.board import Board
from core.config import BOARD_SIZE, SPEEDS
from core.snake import Snake
from core.types import Direction, GameEvent, GameStatus


class Game:
    def __init__(self, speed: str = "NORMAL", *, size: int = BOARD_SIZE,
                 rng: random.Random | None = None):
        if size < 3:
            raise ValueError("Board must fit the initial three-segment snake")
        self.board = Board(size)
        x, y = max(2, size // 2), size // 2
        self.snake = Snake([(x, y), (x - 1, y), (x - 2, y)])
        self.rng = rng if rng is not None else random.Random()
        self.interval = 1 / SPEEDS[speed]
        self.apple = self.board.spawn_apple(self.snake.body, self.rng)
        self.score = 0
        self.status = GameStatus.RUNNING

    def turn(self, direction: Direction) -> bool:
        return self.status == GameStatus.RUNNING and self.snake.turn(direction)

    def step(self) -> list[GameEvent]:
        if self.status != GameStatus.RUNNING:
            return []
        head = self.snake.next_head()
        grow = head == self.apple
        occupied = self.snake.body if grow else self.snake.body[:-1]
        if not self.board.contains(head) or head in occupied:
            self.status = GameStatus.LOST
            return [GameEvent.LOST]
        self.snake.advance(head, grow)
        if not grow:
            return []
        self.score += 1
        self.apple = self.board.spawn_apple(self.snake.body, self.rng)
        if self.apple is None:
            self.status = GameStatus.WON
            return [GameEvent.ATE, GameEvent.WON]
        return [GameEvent.ATE]
