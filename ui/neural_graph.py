"""Bounded rendering of real activations; grouped nodes are explicitly identified."""
import math
import pygame
from ui.components import GREEN, MUTED, BORDER, TEXT, label


class ConnectionCache:
    """Keep one static connection surface per detached forward capture/geometry."""
    def __init__(self):
        self.source = None
        self.geometry = None
        self.surface = None

    def draw(self, target, rect, groups, lab, snapshot):
        source = snapshot.weights if snapshot else lab.captured
        geometry = (rect.size, tuple(len(nodes) for nodes in groups))
        if self.source is not source or self.geometry != geometry:
            self.source, self.geometry = source, geometry
            self.surface = pygame.Surface(rect.size, pygame.SRCALPHA)
            for column in range(1, len(groups)):
                weights = snapshot.weights[column - 1] if snapshot else lab.network.layers[column - 1].weight.detach().tolist()
                for left in groups[column - 1]:
                    for right in groups[column]:
                        weight = weights[right[2]][left[2]]
                        color = (43, 86, 61) if weight >= 0 else (76, 57, 85)
                        pygame.draw.line(self.surface, color, (left[0] - rect.x, left[1] - rect.y),
                                         (right[0] - rect.x, right[1] - rect.y))
        target.blit(self.surface, rect.topleft)


def activation_color(value):
    """Fixed, signed tanh(|a|) intensity: zero is dark; no per-frame rescaling."""
    intensity = math.tanh(abs(value))
    return ((36, 65 + int(150 * intensity), 60 + int(65 * intensity)) if value >= 0
            else (85 + int(90 * intensity), 68, 117))


def draw_network(surface, rect, lab, selected, snapshot=None, connection_cache=None):
    sizes = snapshot.config.sizes if snapshot else lab.network.config.sizes
    observation = snapshot.observation if snapshot else lab.observation
    captured = snapshot.captured if snapshot else lab.captured
    values = [observation] + [capture.outputs for capture in captured]
    groups = []
    hits = []
    for column, size in enumerate(sizes):
        count = min(16, size)
        nodes = []
        x = rect.left + 20 + column * (rect.width - 40) / (len(sizes) - 1)
        for row in range(count):
            start, end = row * size // count, (row + 1) * size // count
            y = rect.top + 38 + (row + .5) * (rect.height - 64) / count
            value = sum(values[column][start:end]) / (end - start)
            nodes.append((x, y, start, end, value))
        groups.append(nodes)
        name = "Input" if column == 0 else "Output" if column == len(sizes) - 1 else f"H{column}"
        label(surface, f"{name}: {size}", (x, rect.top + 10), 13, TEXT)
    aggregated = any(size > 16 for size in sizes)
    cache = connection_cache if connection_cache is not None else ConnectionCache()
    cache.draw(surface, rect, groups, lab, snapshot)
    for column, nodes in enumerate(groups):
        for x, y, start, end, value in nodes:
            color = activation_color(value)
            radius = min(6, max(3, int((rect.height - 64) / len(nodes) / 2) - 1))
            pygame.draw.circle(surface, color, (round(x), round(y)), radius)
            if column == len(groups) - 1 and start == max(range(3), key=lambda i: values[-1][i]):
                pygame.draw.circle(surface, GREEN, (round(x), round(y)), 8, 1)
            if selected[0] == column - 1 and start <= selected[1] < end:
                pygame.draw.circle(surface, GREEN, (round(x), round(y)), radius + 3, 2)
            hit_height = min(18, (rect.height - 64) / len(nodes))
            hits.append((pygame.Rect(x - 10, y - hit_height / 2, 20, hit_height), column - 1, start, end))
    label(surface, "Groups + representative links; click, then choose exact index" if aggregated else "Click a neuron to inspect its actual values",
          (rect.centerx, rect.bottom + 8), 12, MUTED)
    return hits
