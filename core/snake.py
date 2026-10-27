"""Snake position and a single pending turn per movement tick."""
from dataclasses import dataclass, field
from core.types import Cell, Direction


@dataclass
class Snake:
    body: list[Cell]
    direction: Direction = Direction.RIGHT
    pending: Direction | None = field(default=None, init=False)

    def turn(self, direction: Direction) -> bool:
        dx, dy = self.direction.value
        if self.pending is not None or direction == self.direction:
            return False
        if direction.value == (-dx, -dy):
            return False
        self.pending = direction
        return True

    def next_head(self) -> Cell:
        if self.pending is not None:
            self.direction = self.pending
            self.pending = None
        x, y = self.body[0]
        dx, dy = self.direction.value
        return x + dx, y + dy

    def advance(self, head: Cell, grow: bool) -> None:
        self.body.insert(0, head)
        if not grow:
            self.body.pop()
