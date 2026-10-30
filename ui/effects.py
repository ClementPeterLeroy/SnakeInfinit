"""Short cosmetic timers, deliberately separate from movement timing."""
from dataclasses import dataclass
import math
from core.types import Cell, Explosion


@dataclass
class Feedback:
    eaten_cell: Cell | None = None
    apple_remaining: float = 0.0
    overlay_elapsed: float = 0.16
    star_effect: bool = False
    death_remaining: float = 0.0
    celebration_remaining: float = 0.0
    explosions: tuple[Explosion, ...] = ()
    explosion_remaining: float = 0.0
    explosion_notice_remaining: float = 0.0

    def detonate(self, explosions):
        self.explosions = tuple(explosions)
        self.explosion_remaining = 0.4
        self.explosion_notice_remaining = 0.8

    def consume(self, cell: Cell, star=False):
        self.eaten_cell = cell
        self.apple_remaining = 0.28
        self.star_effect = star

    def enter_overlay(self):
        self.overlay_elapsed = 0.0

    def update(self, dt: float, animations: bool):
        # A static detonation notice remains essential feedback with effects off.
        self.explosion_notice_remaining = max(0, self.explosion_notice_remaining - dt)
        if not animations:
            self.apple_remaining = 0.0
            self.eaten_cell = None
            self.overlay_elapsed = 0.16
            self.death_remaining = 0
            self.celebration_remaining = 0
            self.explosion_remaining = 0
            return
        self.apple_remaining = max(0.0, self.apple_remaining - dt)
        self.overlay_elapsed = min(0.16, self.overlay_elapsed + dt)
        self.death_remaining = max(0, self.death_remaining - dt)
        self.celebration_remaining = max(0, self.celebration_remaining - dt)
        self.explosion_remaining = max(0, self.explosion_remaining - dt)

    @property
    def overlay_opacity(self):
        return self.overlay_elapsed / 0.16

    @property
    def shake_offset(self):
        return round(math.sin(self.death_remaining * 150) * 4 * self.death_remaining / 0.2)
