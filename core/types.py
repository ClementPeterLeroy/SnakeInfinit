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
    STAR = "star"
    SUPER = "super"
    STAR_SPAWNED = "star_spawned"
    STAR_EXPIRED = "star_expired"
    HAZARD_WARNING = "hazard_warning"
    HAZARD_ACTIVE = "hazard_active"
    HAZARD_EXPIRED = "hazard_expired"
    MINE_EXPLODED = "mine_exploded"


@dataclass(frozen=True)
class Hazard:
    cell: Cell
    kind: str
    activates_at: float
    expires_at: float | None = None

    def active(self, elapsed: float) -> bool:
        return self.activates_at <= elapsed and (self.expires_at is None or elapsed < self.expires_at)


@dataclass(frozen=True)
class Explosion:
    """An instantaneous detonation, retained for observations and UI feedback."""
    cell: Cell
    cells: tuple[Cell, ...]
    detonated_at: float


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
    elapsed: float = 0.0
    star: Cell | None = None
    star_remaining: float = 0.0
    pending_growth: int = 0
    apples_eaten: int = 0
    stars_collected: int = 0
    hazards: tuple[Hazard, ...] = ()
    next_star_at: float | None = None
    next_hazard_at: float | None = None
    super_announced: bool = False
    reward: float = 0.0
    death_cause: str | None = None
    explosions: tuple[Explosion, ...] = ()

    @property
    def terminated(self) -> bool:
        return self.status != GameStatus.RUNNING
