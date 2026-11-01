"""Safe, aspect-preserving image loading with size caching."""
import logging
from collections import OrderedDict
import pygame
from services.resources import resource_path


class AssetManager:
    def __init__(self):
        self.originals = {}
        self.scaled = OrderedDict()

    def image(self, name: str, size: int) -> pygame.Surface | None:
        key = (name, size)
        if key in self.scaled:
            self.scaled.move_to_end(key)
            return self.scaled[key]
        if name not in self.originals:
            try:
                self.originals[name] = pygame.image.load(resource_path("assets", "px", name)).convert_alpha()
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
        if len(self.scaled) > 64:
            self.scaled.popitem(last=False)
        return result

    def fit(self, name, bounds):
        """Fit supplied artwork inside a rectangle without cropping or distortion."""
        source = self.image(name, max(bounds))
        if source is None:
            return None
        rect = source.get_rect().fit(pygame.Rect((0, 0), bounds))
        return pygame.transform.smoothscale(source, rect.size)
