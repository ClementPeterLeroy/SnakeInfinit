"""Local leaderboard tabs, lifetime statistics, and name entry."""
import pygame
from ui.components import GREEN, MUTED, PANEL, BORDER, backdrop, button, label

TABS = (("CASUAL", "classic_CASUAL"), ("NORMAL", "classic_NORMAL"),
        ("FAST", "classic_FAST"), ("ARCADE", "arcade"), ("STATISTICS", "statistics"))


def buttons(tab):
    return [button(title, 186 + index * 184, 203, 172, "tab:" + key, tab == key)
            for index, (title, key) in enumerate(TABS)] + [button("BACK", 500, 632, 280, "menu")]


def draw(surface, store, tab):
    backdrop(surface, "SNAKE INFINITE / LOCAL RECORDS", "LEADERBOARD")
    if tab == "statistics":
        stats = store.data["statistics"]
        values = [("GAMES PLAYED", stats["games"]), ("APPLES EATEN", stats["apples"]),
                  ("STARS COLLECTED", stats["stars"]), ("TOTAL SCORE", stats["score"]),
                  ("HIGHEST SCORE", stats["highest_score"]), ("AVERAGE SCORE", f"{store.average_score:.1f}"),
                  ("ACTIVE PLAYTIME", f"{stats['playtime'] / 60:.1f} min"), ("DEATHS", stats["deaths"]),
                  ("CHALLENGE ATTEMPTS", stats["challenge_attempts"]), ("COMPLETIONS", stats["challenge_completions"])]
        for index, (title, value) in enumerate(values):
            x, y = 246 + index % 2 * 450, 285 + index // 2 * 52
            label(surface, title, (x, y), 15, MUTED, False)
            label(surface, str(value), (x + 300, y), 20, GREEN, False)
    else:
        rows = store.data["leaderboards"][tab]
        label(surface, "RANK      PLAYER", (248, 276), 14, MUTED, False)
        label(surface, "SCORE                 DATE (UTC)", (760, 276), 14, MUTED, False)
        if not rows:
            label(surface, "Your next run could be the first record.", (640, 403), 23, MUTED)
        for index, row in enumerate(rows):
            y = 311 + index * 25
            label(surface, f"{index + 1:02d}          {row['name'][:12]}", (248, y), 18, center=False)
            label(surface, str(row["score"]), (780, y), 18, GREEN, False)
            label(surface, row["date"][:10], (920, y), 16, MUTED, False)
    label(surface, "Scores are stored locally on this computer and are not synchronized online.", (640, 586), 15, MUTED)
    if not store.last_save_ok:
        label(surface, "Storage unavailable: this session's changes could not be saved.", (640, 610), 15, (239, 152, 115))


def name_buttons():
    return [button("SAVE", 424, 451, 200, "save_name", True),
            button("CANCEL", 656, 451, 200, "cancel_name")]


def draw_name(surface, name, score):
    shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    shade.fill((8, 12, 10, 220))
    surface.blit(shade, (0, 0))
    pygame.draw.rect(surface, PANEL, (344, 190, 592, 350), border_radius=16)
    label(surface, "TOP 10 SCORE", (640, 244), 34, GREEN)
    label(surface, f"{score} points — enter your name", (640, 297), 21)
    pygame.draw.rect(surface, BORDER, (424, 345, 432, 64), 1, border_radius=8)
    label(surface, name or "PLAYER", (640, 377), 27)
    label(surface, "12 characters / Enter to save / Esc to cancel", (640, 425), 15, MUTED)
