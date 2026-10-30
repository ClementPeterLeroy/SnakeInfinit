"""Board bounds and uniformly sampled unoccupied food positions."""
import random
from dataclasses import dataclass
from core.types import Cell


@dataclass(frozen=True)
class Board:
    # `size` remains the width for compatibility with the original square board.
    size: int = 25
    height: int | None = None

    def __post_init__(self):
        if self.height is None:
            object.__setattr__(self, "height", self.size)

    @property
    def width(self) -> int:
        return self.size

    def contains(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def spawn_apple(self, occupied: list[Cell], rng: random.Random) -> Cell | None:
        blocked = set(occupied)
        free = [(x, y) for y in range(self.height) for x in range(self.width)
                if (x, y) not in blocked]
        return rng.choice(free) if free else None
