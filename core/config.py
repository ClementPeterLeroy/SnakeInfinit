"""Source constants; user preferences live in a separate JSON file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "v1.0.0"
SUBTITLE = "Foundation"
BOARD_SIZE = 25
SPEEDS = {"CASUAL": 6, "NORMAL": 9, "FAST": 13}
