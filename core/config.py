"""Source constants; user preferences live in a separate JSON file."""
from pathlib import Path
from dataclasses import dataclass
import math

ROOT = Path(__file__).resolve().parent.parent
VERSION = "v1.5.0"
SUBTITLE = "Analysis & Experiment Framework"
BOARD_SIZE = 25
SPEEDS = {"CASUAL": 6, "NORMAL": 9, "FAST": 13}


@dataclass(frozen=True)
class GameRules:
    """One validated ruleset for all modes; classic rejects non-classic switches."""

    board_width: int = BOARD_SIZE
    board_height: int = BOARD_SIZE
    initial_speed: float = 9.0
    progressive_speed: bool = False
    stars_enabled: bool = False
    hazards_enabled: bool = False
    mode: str = "classic"
    challenge_id: str | None = None
    initial_length: int = 3
    score_target: int | None = None
    star_target: int | None = None
    survival_target: float | None = None
    star_interval: tuple[float, float] = (15.0, 25.0)
    star_lifetime: float = 6.0
    wall_enabled: bool = False
    mines_enabled: bool = False
    hazard_interval: float = 8.0
    warning_duration: float = 1.0
    mine_lifetime: float = 5.0
    wall_spawn_count: int = 1

    def __post_init__(self):
        if (type(self.board_width) is not int or type(self.board_height) is not int
                or self.board_width < 3 or self.board_height < 3):
            raise ValueError("Board dimensions must be integers of at least 3 cells")
        if (isinstance(self.initial_speed, bool)
                or not isinstance(self.initial_speed, (int, float))
                or not math.isfinite(self.initial_speed) or self.initial_speed <= 0):
            raise ValueError("Speed must be a finite positive number")
        flags = (self.progressive_speed, self.stars_enabled, self.hazards_enabled,
                 self.wall_enabled, self.mines_enabled)
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("Rule switches must be booleans")
        if self.mode not in ("classic", "arcade", "challenge"):
            raise ValueError("Unknown game mode")
        if self.mode == "classic" and (any(flags) or self.wall_enabled or self.mines_enabled):
            raise NotImplementedError("Classic mode does not enable arcade mechanics")
        if self.wall_enabled and self.mines_enabled:
            raise ValueError("Select one hazard behavior per ruleset")
        if type(self.wall_spawn_count) is not int or not 1 <= self.wall_spawn_count <= 4:
            raise ValueError("Wall waves must contain one to four cells")
        if self.hazards_enabled != (self.wall_enabled or self.mines_enabled):
            raise ValueError("Hazards require an enabled wall or mine behavior")
        if (type(self.initial_length) is not int or self.initial_length < 3
                or self.initial_length >= self.board_width * self.board_height):
            raise ValueError("Initial length must fit the board and leave a free cell")
        for value in (self.score_target, self.star_target):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("Objective targets must be positive integers")
        if (not isinstance(self.star_interval, tuple) or len(self.star_interval) != 2
                or any(type(value) not in (int, float) for value in self.star_interval)):
            raise ValueError("Star interval must be a pair of numbers")
        for value in (*self.star_interval, self.star_lifetime, self.hazard_interval,
                      self.warning_duration, self.mine_lifetime):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Timers must be finite positive numbers")
        if len(self.star_interval) != 2 or self.star_interval[0] > self.star_interval[1]:
            raise ValueError("Invalid star interval")
        if self.survival_target is not None and (
                type(self.survival_target) not in (int, float)
                or not math.isfinite(self.survival_target) or self.survival_target <= 0):
            raise ValueError("Survival target must be positive")


ARCADE_SPEEDS = ((0, 7), (10, 8), (25, 9), (50, 10), (75, 11), (100, 12), (150, 13))


def arcade_rules() -> GameRules:
    return GameRules(mode="arcade", initial_speed=7, progressive_speed=True, stars_enabled=True)
