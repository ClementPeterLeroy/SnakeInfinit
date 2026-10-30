"""Shared simulation value types."""
from enum import Enum
from dataclasses import dataclass

Cell = tuple[int, int]


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class GameStatus(Enum):
    RUNNING = "running"
    LOST = "lost"
    WON = "won"


class GameEvent(Enum):
    ATE = "ate"
    LOST = "lost"
    WON = "won"


@dataclass(frozen=True)
class GameState:
    """Detached observation: callers cannot mutate the engine through a snapshot."""

    board_width: int
    board_height: int
    snake: tuple[Cell, ...]
    direction: Direction
    pending_direction: Direction | None
    apple: Cell | None
    score: int
    status: GameStatus
    interval: float

    @property
    def terminated(self) -> bool:
        return self.status != GameStatus.RUNNING
