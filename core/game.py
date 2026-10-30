"""Deterministic step-based rules. No Pygame, clocks, audio, or UI imports."""
import random
from core.board import Board
from core.config import BOARD_SIZE, SPEEDS, GameRules
from core.snake import Snake
from core.types import Direction, GameEvent, GameStatus, GameState


class Game:
    def __init__(self, speed: str | None = None, *, size: int | None = None,
                 rng: random.Random | None = None, seed: int | None = None,
                 rules: GameRules | None = None):
        if rules is not None and (speed is not None or size is not None):
            raise ValueError("Supply rules or legacy speed/size arguments, not both")
        if rng is not None and seed is not None:
            raise ValueError("Supply rng or seed, not both")
        self.rules = rules if rules is not None else GameRules(
            board_width=BOARD_SIZE if size is None else size,
            board_height=BOARD_SIZE if size is None else size,
            initial_speed=SPEEDS["NORMAL" if speed is None else speed])
        self.board = Board(self.rules.board_width, self.rules.board_height)
        self.rng = random.Random(seed)
        if rng is not None:
            # Retain RNG injection compatibility without sharing mutable state.
            self.rng.setstate(rng.getstate())
        self._reset_state()

    def _reset_state(self):
        x, y = max(2, self.board.width // 2), self.board.height // 2
        self.snake = Snake([(x, y), (x - 1, y), (x - 2, y)])
        self.interval = 1 / self.rules.initial_speed
        self.apple = self.board.spawn_apple(self.snake.body, self.rng)
        self.score = 0
        self.status = GameStatus.RUNNING

    def reset(self, seed: int | None = None) -> GameState:
        """Start fresh with these rules. None requests fresh system entropy."""
        self.rng.seed(seed)
        self._reset_state()
        return self.get_state()

    @property
    def terminated(self) -> bool:
        return self.status != GameStatus.RUNNING

    def get_state(self) -> GameState:
        return GameState(self.board.width, self.board.height, tuple(self.snake.body),
                         self.snake.direction, self.snake.pending, self.apple,
                         self.score, self.status, self.interval)

    def turn(self, direction: Direction) -> bool:
        return self.status == GameStatus.RUNNING and self.snake.turn(direction)

    def step(self, action: Direction | None = None) -> list[GameEvent]:
        """Advance exactly one cell, optionally submitting a direction first."""
        if self.status != GameStatus.RUNNING:
            return []
        if action is not None:
            self.turn(action)
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
