"""Relative actions shared by Step and Live modes; no learning policy."""
from enum import IntEnum
from core.types import Direction
from ai.observations import DIRECTIONS


class RelativeAction(IntEnum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2


ACTION_NAMES = ("Straight", "Turn left", "Turn right")


def relative_direction(direction: Direction, action: RelativeAction | int) -> Direction:
    if isinstance(action, bool):
        raise ValueError("Unknown relative action")
    action = RelativeAction(action)
    offset = {RelativeAction.STRAIGHT: 0, RelativeAction.LEFT: -1, RelativeAction.RIGHT: 1}[action]
    return DIRECTIONS[(DIRECTIONS.index(direction) + offset) % 4]
