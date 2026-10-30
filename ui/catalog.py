"""Mode and challenge selection, including clearly unavailable future entries."""
import pygame
from core.challenges import CHALLENGES, COMING_SOON
from ui.components import GREEN, MUTED, PANEL, BORDER, backdrop, button, label


def mode_buttons():
    return [button("CLASSIC", 500, 278, 280, "play:classic", True),
            button("ARCADE", 500, 346, 280, "play:arcade"),
            button("CHALLENGES", 500, 414, 280, "challenges"),
            button("BACK", 500, 542, 280, "menu")]


def draw_modes(surface):
    backdrop(surface, "SNAKE INFINITE / SOLO", "SELECT MODE", "A familiar classic. A new kind of challenge.")
    label(surface, "Classic: your pace. Arcade: rising speed and timed stars.", (640, 503), 17, MUTED)


def challenge_buttons(page):
    result = [button("BACK", 96, 638, 160, "modes"),
              button("COMING SOON" if page == 0 else "PLAYABLE", 944, 638, 240, "challenge_page")]
    if page == 0:
        result += [button("PLAY", 1024, 173 + index * 88, 120, "play:" + entry.id, True)
                   for index, entry in enumerate(CHALLENGES)]
    return result


def draw_challenges(surface, records, page):
    surface.fill((18, 22, 22))
    label(surface, "CHALLENGES" if page == 0 else "ON THE HORIZON", (640, 69), 38)
    label(surface, "All five challenges are available now." if page == 0 else "Future challenges — unavailable in v1.2", (640, 117), 17, MUTED)
    if page == 0:
        for index, entry in enumerate(CHALLENGES):
            y = 164 + index * 88
            pygame.draw.rect(surface, PANEL, (96, y, 1088, 78), border_radius=10)
            record = records.get(entry.id, {})
            label(surface, entry.name, (120, y + 7), 21, GREEN, False)
            label(surface, entry.objective, (120, y + 42), 16, MUTED, False)
            label(surface, f"DIFFICULTY {entry.difficulty}/5", (540, y + 15), 14, GREEN, False)
            label(surface, "COMPLETE" if record.get("completed") else "NOT COMPLETED", (710, y + 10), 15, MUTED, False)
            best = record.get("best_time")
            performance = f"Best {best:.1f}s / {record.get('best_score', 0)} pts" if best is not None else f"Best {record.get('best_score', 0)} pts / {record.get('best_survival', 0):.1f}s"
            label(surface, performance, (710, y + 40), 15, MUTED, False)
    else:
        for index, name in enumerate(COMING_SOON):
            x, y = 96 + (index % 2) * 556, 167 + (index // 2) * 144
            pygame.draw.rect(surface, PANEL, (x, y, 532, 126), border_radius=12)
            label(surface, name, (x + 24, y + 17), 23, GREEN, False)
            label(surface, "COMING SOON", (x + 24, y + 55), 14, MUTED, False)
            if name == "THE HUNTER":
                label(surface, "Survive while a hostile snake hunts you.", (x + 24, y + 88), 16, MUTED, False)
                for dx, dy in ((0, 0), (12, 0), (24, 0), (24, -12), (36, -12)):
                    pygame.draw.rect(surface, (190, 44, 72), (x + 440 + dx, y + 42 + dy, 11, 11), border_radius=3)
