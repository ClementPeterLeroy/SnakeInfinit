"""Shared palette, typography, and animated mouse buttons."""
from dataclasses import dataclass
import pygame

BG = (18, 22, 22)
PANEL = (26, 32, 31)
GREEN = (123, 225, 154)
TEXT = (230, 237, 232)
MUTED = (135, 151, 142)
BORDER = (48, 62, 54)
_fonts = {}


def reset_fonts():
    """Discard font handles before restarting or shutting down SDL's font module."""
    _fonts.clear()


def font(size):
    if size not in _fonts:
        if len(_fonts) >= 128:
            _fonts.pop(next(iter(_fonts)))
        _fonts[size] = pygame.font.SysFont("segoeui,dejavusans,arial", size)
    return _fonts[size]


def label(surface, text, position, size=20, color=TEXT, center=True):
    rendered = font(size).render(text, True, color)
    rect = rendered.get_rect(center=position) if center else rendered.get_rect(topleft=position)
    surface.blit(rendered, rect)


@dataclass
class Button:
    text: str
    rect: pygame.Rect
    action: str
    primary: bool = False
    hover: float = 0
    text_size: int = 19

    def draw(self, surface, mouse, dt, animations, pressed=False):
        target = float(self.rect.collidepoint(mouse))
        self.hover = (min(1, self.hover + dt * 9) if target else max(0, self.hover - dt * 9)) if animations else target
        amount = self.hover if animations else target
        base = (97, 198, 129) if self.primary else PANEL
        color = tuple(min(255, int(value + amount * 15)) for value in base)
        if pressed and target:
            color = tuple(max(0, value - 22) for value in color)
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        if not self.primary:
            pygame.draw.rect(surface, GREEN if target else BORDER, self.rect, 1, border_radius=10)
        label(surface, self.text, self.rect.center, self.text_size, BG if self.primary else TEXT)


def button(text, x, y, width, action, primary=False):
    return Button(text, pygame.Rect(x, y, width, 50), action, primary)


def backdrop(surface, eyebrow, title, subtitle=""):
    surface.fill(BG)
    label(surface, eyebrow, (640, 95), 15, GREEN)
    label(surface, title, (640, 152), 44)
    if subtitle:
        label(surface, subtitle, (640, 207), 18, MUTED)
