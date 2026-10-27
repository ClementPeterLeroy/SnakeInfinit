"""Application state, input routing, and fixed-timestep scheduling."""
import argparse
from pathlib import Path
import pygame
from core.game import Game
from core.types import Direction, GameStatus
from services.assets import AssetManager
from services.audio import AudioManager
from services.persistence import SettingsStore
from ui import game_view, menu, pause_view, settings_view
from ui.components import BG, reset_fonts

KEYS = {
    pygame.K_UP: Direction.UP, pygame.K_w: Direction.UP, pygame.K_z: Direction.UP,
    pygame.K_DOWN: Direction.DOWN, pygame.K_s: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT, pygame.K_a: Direction.LEFT, pygame.K_q: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT, pygame.K_d: Direction.RIGHT,
}


class Application:
    def __init__(self, settings_path: Path | None = None):
        reset_fonts()
        pygame.display.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        pygame.display.set_caption("SNAKE • v1.0.0 — Foundation")
        self.canvas = pygame.Surface((1280, 720))
        self.clock = pygame.time.Clock()
        self.store = SettingsStore(settings_path)
        self.settings = self.store.load()
        self.audio = AudioManager(self.settings)
        self.assets = AssetManager()
        self.game = None
        self.screen = "menu"
        self.running = True
        self.accumulator = 0.0
        self.pressed = None
        self.controls = []
        self.rebuild_controls()

    def rebuild_controls(self):
        if self.screen in ("menu", "ai"):
            self.controls = menu.buttons(self.screen == "ai")
        elif self.screen == "settings":
            self.controls = settings_view.buttons(self.settings)
        elif self.screen in ("pause", "over"):
            self.controls = pause_view.buttons(self.screen == "pause")
        else:
            self.controls = []
        self.pressed = None

    def action(self, action):
        self.accumulator = 0
        if action == "exit":
            self.running = False
        elif action == "start":
            self.game = Game(self.settings.speed)
            self.screen = "game"
        elif action == "resume":
            self.screen = "game"
        elif action in ("music", "sound_effects", "animations"):
            setattr(self.settings, action, not getattr(self.settings, action))
            self.store.save(self.settings)
            self.audio.apply_settings()
        elif action.startswith("speed:"):
            self.settings.speed = action.split(":")[1]
            self.store.save(self.settings)
        else:
            self.screen = action
        self.rebuild_controls()

    def viewport(self):
        width, height = self.window.get_size()
        scale = min(width / 1280, height / 720)
        size = (max(1, round(1280 * scale)), max(1, round(720 * scale)))
        return pygame.Rect((width - size[0]) // 2, (height - size[1]) // 2, *size)

    def mouse_position(self, position):
        view = self.viewport()
        return ((position[0] - view.x) * 1280 / view.width,
                (position[1] - view.y) * 720 / view.height)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.WINDOWFOCUSLOST and self.screen == "game":
            self.action("pause")
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.screen == "game":
                    self.action("pause")
                elif self.screen == "pause":
                    self.action("resume")
                elif self.screen in ("settings", "ai", "over"):
                    self.action("menu")
            elif self.screen == "game" and event.key in KEYS:
                self.game.turn(KEYS[event.key])
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            position = self.mouse_position(event.pos)
            self.pressed = next((b.action for b in self.controls if b.rect.collidepoint(position)), None)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            position = self.mouse_position(event.pos)
            action = next((b.action for b in self.controls
                           if b.action == self.pressed and b.rect.collidepoint(position)), None)
            self.pressed = None
            if action:
                self.action(action)

    def update(self, dt):
        if self.screen != "game":
            self.accumulator = 0
            return
        self.accumulator += dt
        while self.accumulator >= self.game.interval:
            self.accumulator -= self.game.interval
            self.audio.handle(self.game.step())
            if self.game.status != GameStatus.RUNNING:
                self.action("over")
                break

    def draw(self, dt):
        if self.screen in ("menu", "ai"):
            menu.draw(self.canvas, self.screen == "ai")
        elif self.screen == "settings":
            settings_view.draw(self.canvas)
        else:
            game_view.draw(self.canvas, self.game, self.assets)
            if self.screen in ("pause", "over"):
                pause_view.draw(self.canvas, self.game, self.screen == "pause")
        mouse = self.mouse_position(pygame.mouse.get_pos())
        for control in self.controls:
            control.draw(self.canvas, mouse, dt, self.settings.animations,
                         self.pressed == control.action)
        view = self.viewport()
        self.window.fill(BG)
        self.window.blit(pygame.transform.smoothscale(self.canvas, view.size), view)
        pygame.display.flip()

    def run(self, frame_limit: int | None = None):
        frames = 0
        try:
            while self.running:
                # Limit a stalled frame's catch-up to keep the UI responsive.
                dt = min(self.clock.tick(120) / 1000, 0.25)
                before = self.screen
                for event in pygame.event.get():
                    self.handle_event(event)
                # Time elapsed on a paused/menu screen never enters a new game.
                self.update(0 if before != self.screen else dt)
                self.draw(dt)
                frames += 1
                if frame_limit is not None and frames >= frame_limit:
                    break
        finally:
            self.close()

    def close(self):
        self.audio.close()
        reset_fonts()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="SNAKE v1.0.0 — Foundation")
    parser.add_argument("--settings", type=Path, help="Override the user settings JSON path")
    parser.add_argument("--smoke-test", action="store_true", help="Render three frames and exit")
    args = parser.parse_args()
    Application(args.settings).run(3 if args.smoke_test else None)
