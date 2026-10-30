"""Source constants; user preferences live in a separate JSON file."""
from pathlib import Path
from dataclasses import dataclass
import math

ROOT = Path(__file__).resolve().parent.parent
VERSION = "v1.1.0"
SUBTITLE = "Gameplay Polish"
BOARD_SIZE = 25
SPEEDS = {"CASUAL": 6, "NORMAL": 9, "FAST": 13}


@dataclass(frozen=True)
class GameRules:
    """Immutable classic rules; reserved switches explicitly reject future modes."""

    board_width: int = BOARD_SIZE
    board_height: int = BOARD_SIZE
    initial_speed: float = 9.0
    progressive_speed: bool = False
    stars_enabled: bool = False
    hazards_enabled: bool = False

    def __post_init__(self):
        if (type(self.board_width) is not int or type(self.board_height) is not int
                or self.board_width < 3 or self.board_height < 3):
            raise ValueError("Board dimensions must be integers of at least 3 cells")
        if (isinstance(self.initial_speed, bool)
                or not isinstance(self.initial_speed, (int, float))
                or not math.isfinite(self.initial_speed) or self.initial_speed <= 0):
            raise ValueError("Speed must be a finite positive number")
        flags = (self.progressive_speed, self.stars_enabled, self.hazards_enabled)
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("Rule switches must be booleans")
        if any(flags):
            raise NotImplementedError("Only classic rules are supported in v1.1")
