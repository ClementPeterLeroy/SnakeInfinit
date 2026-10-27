"""Safe, aspect-preserving image loading with size caching."""
import logging
import pygame
from core.config import ROOT


class AssetManager:
    def __init__(self):
        self.originals = {}
        self.scaled = {}

    def image(self, name: str, size: int) -> pygame.Surface | None:
        key = (name, size)
        if key in self.scaled:
            return self.scaled[key]
        if name not in self.originals:
            try:
                self.originals[name] = pygame.image.load(ROOT / "assets" / "px" / name).convert_alpha()
            except (pygame.error, OSError) as error:
                logging.warning("Image unavailable: %s", error)
                self.originals[name] = None
        source = self.originals[name]
        result = None
        if source is not None:
            ratio = size / max(source.get_size())
            dimensions = tuple(max(1, round(value * ratio)) for value in source.get_size())
            result = pygame.transform.smoothscale(source, dimensions)
        self.scaled[key] = result
        return result
