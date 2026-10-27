"""Shared simulation value types."""
from enum import Enum

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
