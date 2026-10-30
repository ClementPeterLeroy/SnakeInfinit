"""Wrap a scaled copy of the approved PNG artwork in a Windows ICO container."""
from io import BytesIO
from pathlib import Path
import struct
import pygame


def build_icon(source: Path, target: Path):
    image = pygame.image.load(str(source))
    rect = image.get_rect().fit(pygame.Rect(0, 0, 256, 256))
    canvas = pygame.Surface((256, 256), pygame.SRCALPHA)
    canvas.blit(pygame.transform.smoothscale(image, rect.size), rect)
    buffer = BytesIO()
    pygame.image.save(canvas, buffer, "PNG")
    payload = buffer.getvalue()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(struct.pack("<HHH", 0, 1, 1)
                       + struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(payload), 22)
                       + payload)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    build_icon(root / "assets/px/Icon.png", root / "assets/branding/snake_infinite_icon.ico")
