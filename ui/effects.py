"""Short cosmetic timers, deliberately separate from movement timing."""
from dataclasses import dataclass
from core.types import Cell


@dataclass
class Feedback:
    eaten_cell: Cell | None = None
    apple_remaining: float = 0.0
    overlay_elapsed: float = 0.16

    def consume(self, cell: Cell):
        self.eaten_cell = cell
        self.apple_remaining = 0.28

    def enter_overlay(self):
        self.overlay_elapsed = 0.0

    def update(self, dt: float, animations: bool):
        if not animations:
            self.apple_remaining = 0.0
            self.eaten_cell = None
            self.overlay_elapsed = 0.16
            return
        self.apple_remaining = max(0.0, self.apple_remaining - dt)
        self.overlay_elapsed = min(0.16, self.overlay_elapsed + dt)

    @property
    def overlay_opacity(self):
        return self.overlay_elapsed / 0.16
