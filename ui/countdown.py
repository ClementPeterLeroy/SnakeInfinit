"""Presentation-only countdown. The engine does not know it exists."""
from dataclasses import dataclass


@dataclass
class Countdown:
    elapsed: float = 0.0
    stage: int = 0

    @property
    def word(self):
        return ("3", "2", "1", "GO!")[self.stage]

    def advance(self, dt):
        self.elapsed += max(0, dt)
        changes = []
        while self.stage < 3 and self.elapsed >= self.stage + 1:
            self.stage += 1
            changes.append(self.word)
        return changes
