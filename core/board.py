"""Board bounds and uniformly sampled unoccupied food positions."""
import random
from dataclasses import dataclass
from core.types import Cell


@dataclass(frozen=True)
class Board:
    size: int = 25

    def contains(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size

    def spawn_apple(self, occupied: list[Cell], rng: random.Random) -> Cell | None:
        blocked = set(occupied)
        free = [(x, y) for y in range(self.size) for x in range(self.size)
                if (x, y) not in blocked]
        return rng.choice(free) if free else None
