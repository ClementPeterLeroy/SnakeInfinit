"""Application state, input routing, and fixed-timestep scheduling."""
import argparse
from pathlib import Path
import pygame
from core.game import Game
from core.config import VERSION, SUBTITLE, ROOT, SPEEDS, GameRules, arcade_rules
from core.challenges import BY_ID
from core.types import Direction, GameStatus, GameEvent
from services.assets import AssetManager
from services.audio import AudioManager
from services.persistence import SettingsStore
from services.records import RecordsStore
from services.video import StartupVideo
from ui import game_view, menu, pause_view, settings_view, catalog, leaderboard_view
from ui.components import BG, reset_fonts
from ui.effects import Feedback
from ui.countdown import Countdown
from ui.components import label, GREEN, MUTED

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
        pygame.display.set_caption(f"SNAKE INFINITE • {VERSION} — {SUBTITLE}")
        self.canvas = pygame.Surface((1280, 720), pygame.SRCALPHA)
        self.scene = None
        self.clock = pygame.time.Clock()
        self.store = SettingsStore(settings_path)
        self.settings = self.store.load()
        self.audio = AudioManager(self.settings)
        self.assets = AssetManager()
        icon = self.assets.image("Icon.png", 64)
        if icon is not None:
            pygame.display.set_icon(icon)
        self.records = RecordsStore(self.store.path.with_name("records.json"))
        self.selected_rules = GameRules(initial_speed=SPEEDS[self.settings.speed])
        self.selected_speed = self.settings.speed
        self.challenge_page = 0
        self.leaderboard_tab = "classic_NORMAL"
        self.countdown = None
        self.go_remaining = 0.0
        self.resume_screen = "game"
        self.session_started = False
        self.session_saved = False
        self.pending_entry = None
        self.player_name = ""
        self.new_high_score = False
        self.video = None
        self.intro_attempted = False
        self.intro_exit_frame = None
        self.intro_fade_remaining = 0.0
        self.game = None
        self.screen = "menu"
        self.running = True
        self.accumulator = 0.0
        self.queued_turn = None
        self.feedback = Feedback()
        self.timing_epoch = 0
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
        elif self.screen == "modes":
            self.controls = catalog.mode_buttons()
        elif self.screen == "challenges":
            self.controls = catalog.challenge_buttons(self.challenge_page)
        elif self.screen == "leaderboard":
            self.controls = leaderboard_view.buttons(self.leaderboard_tab)
        elif self.screen == "name":
            self.controls = leaderboard_view.name_buttons()
        else:
            self.controls = []
        self.pressed = None

    def action(self, action):
        self.accumulator = 0
        self.timing_epoch += 1
        if action in ("pause", "resume", "start", "menu", "play") or action.startswith("play:"):
            self.queued_turn = None
        if action == "pause" and self.game is not None:
            self.game.snake.pending = None
        if action in ("pause", "over"):
            self.feedback.enter_overlay()
        if action == "exit":
            self.finish_session()
            self.running = False
        elif action == "challenge_page":
            self.challenge_page = 1 - self.challenge_page
        elif action.startswith("tab:"):
            self.leaderboard_tab = action.split(":", 1)[1]
        elif action in ("save_name", "cancel_name"):
            if action == "save_name" and self.pending_entry:
                key, score, mode, speed = self.pending_entry
                self.records.add_score(key, self.player_name, score, mode, speed)
            self.pending_entry = None
            pygame.key.stop_text_input()
            self.screen = "over"
        elif action == "play" or action.startswith("play:"):
            self.finish_session()
            if ":" in action:
                mode = action.split(":", 1)[1]
                if mode == "classic":
                    self.selected_speed = self.settings.speed
                    self.selected_rules = GameRules(initial_speed=SPEEDS[self.selected_speed])
                elif mode == "arcade":
                    self.selected_rules = arcade_rules()
                else:
                    self.selected_rules = BY_ID[mode].rules
            self.begin_game(countdown=True)
        elif action == "start":
            # Retained programmatic immediate-start API. All playable UI buttons
            # use `play`, which always enters the presentation countdown.
            self.finish_session()
            self.selected_speed = self.settings.speed
            self.selected_rules = GameRules(initial_speed=SPEEDS[self.selected_speed])
            self.begin_game(countdown=False)
        elif action == "pause":
            self.resume_screen = self.screen if self.screen in ("game", "countdown") else "game"
            self.screen = "pause"
            self.audio.stop_announcements()
        elif action == "resume":
            self.screen = self.resume_screen
        elif action in ("music", "sound_effects", "animations", "voice_announcements"):
            setattr(self.settings, action, not getattr(self.settings, action))
            self.store.save(self.settings)
            self.audio.apply_settings()
        elif action.startswith("speed:"):
            self.settings.speed = action.split(":")[1]
            self.store.save(self.settings)
        else:
            if action == "menu":
                self.finish_session()
                self.audio.stop_announcements()
                self.pending_entry = None
                pygame.key.stop_text_input()
            self.screen = action
        self.rebuild_controls()

    def begin_game(self, *, countdown):
        self.audio.stop_announcements()
        self.game = Game(rules=self.selected_rules)
        self.feedback = Feedback()
        self.session_started = not countdown
        self.session_saved = False
        self.pending_entry = None
        self.new_high_score = False
        self.go_remaining = 0
        self.countdown = Countdown() if countdown else None
        self.screen = "countdown" if countdown else "game"
        if countdown:
            self.audio.countdown("3")

    def finish_session(self, qualify=False):
        if not self.game or not self.session_started or self.session_saved:
            return
        self.session_saved = True
        if qualify and self.game.rules.mode != "challenge":
            key = "arcade" if self.game.rules.mode == "arcade" else "classic_" + self.selected_speed
            rows = self.records.data["leaderboards"][key]
            self.new_high_score = self.game.score > (rows[0]["score"] if rows else 0)
            if self.records.qualifies(key, self.game.score):
                self.pending_entry = (key, self.game.score, self.game.rules.mode,
                                      self.selected_speed if self.game.rules.mode == "classic" else None)
                self.player_name = ""
        self.records.record_game(self.game)

    def start_intro(self):
        if self.intro_attempted:
            return
        self.intro_attempted = True
        self.video = StartupVideo(ROOT / "assets" / "px" / "SnakeInfinite_Startup.mp4")
        self.audio.suspend_music(True)
        self.screen = "intro"
        self.rebuild_controls()

    def buffer_direction(self, direction):
        """One immediate turn plus one subsequent corner; consume one per tick."""
        if self.game.turn(direction):
            return
        pending = self.game.snake.pending
        if pending is None or self.queued_turn is not None or direction == pending:
            return
        dx, dy = pending.value
        if direction.value != (-dx, -dy):
            self.queued_turn = direction

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
        elif event.type == pygame.WINDOWFOCUSLOST and self.screen in ("game", "countdown"):
            self.action("pause")
        elif event.type == pygame.TEXTINPUT and self.screen == "name":
            self.player_name = (self.player_name + "".join(c for c in event.text if c.isprintable()))[:12]
        elif event.type == pygame.KEYDOWN:
            if self.screen == "name":
                if event.key == pygame.K_RETURN:
                    self.action("save_name")
                elif event.key == pygame.K_ESCAPE:
                    self.action("cancel_name")
                elif event.key == pygame.K_BACKSPACE:
                    self.player_name = self.player_name[:-1]
                return
            if event.key == pygame.K_ESCAPE:
                if self.screen in ("game", "countdown"):
                    self.action("pause")
                elif self.screen == "pause":
                    self.action("resume")
                elif self.screen in ("settings", "ai", "over", "modes", "challenges", "leaderboard"):
                    self.action("menu")
            elif self.screen == "game" and event.key in KEYS:
                # OS key-repeat must not displace a deliberate corner input.
                if not getattr(event, "repeat", False):
                    self.buffer_direction(KEYS[event.key])
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
        if self.screen == "intro":
            if self.video.update(dt):
                if self.settings.animations and getattr(self.video, "current", None) is not None:
                    self.intro_exit_frame = pygame.Surface((1280, 720))
                    self.video.draw(self.intro_exit_frame)
                    self.intro_fade_remaining = 0.25
                self.video.close()
                self.video = None
                self.audio.suspend_music(False)
                self.action("menu")
            return
        if self.screen == "countdown":
            self.accumulator = 0
            for word in self.countdown.advance(dt):
                self.audio.countdown(word)
                if word == "GO!":
                    self.session_started = True
                    self.screen = "game"
                    self.go_remaining = 0.6
                    self.timing_epoch += 1
                    self.rebuild_controls()
            return
        if self.screen != "game":
            self.accumulator = 0
            return
        self.accumulator += dt
        # Tiny rounding differences at exact tick boundaries must not lose a
        # movement step at 60/120/240 FPS. This tolerance is sub-nanosecond.
        while self.accumulator + self.game.interval * 1e-9 >= self.game.interval:
            self.accumulator = max(0.0, self.accumulator - self.game.interval)
            eaten_cell = self.game.apple
            events = self.game.step()
            self.audio.game_score = self.game.score
            self.audio.handle(events)
            if GameEvent.MINE_EXPLODED in events:
                self.feedback.detonate(self.game.explosions)
            if GameEvent.ATE in events and self.settings.animations:
                self.feedback.consume(eaten_cell)
            if GameEvent.STAR in events and self.settings.animations:
                self.feedback.consume(self.game.snake.body[0], star=True)
            if self.game.status != GameStatus.RUNNING:
                self.finish_session(qualify=True)
                self.action("over")
                if self.game.status == GameStatus.LOST and self.settings.animations:
                    self.feedback.death_remaining = 0.2
                if self.new_high_score and self.settings.animations:
                    self.feedback.celebration_remaining = 0.8
                if self.pending_entry:
                    self.screen = "name"
                    pygame.key.start_text_input()
                    self.rebuild_controls()
                break
            if self.queued_turn is not None:
                self.game.turn(self.queued_turn)
                self.queued_turn = None

    def draw(self, dt):
        self.feedback.update(dt, self.settings.animations)
        view = self.viewport()
        if self.scene is None or self.scene.get_size() != view.size:
            self.scene = pygame.Surface(view.size)
        self.scene.fill(BG)
        self.canvas.set_alpha(255)
        if self.screen == "intro":
            self.video.draw(self.scene)
            self.canvas.fill((0, 0, 0, 0))
        elif self.screen in ("menu", "ai"):
            menu.draw(self.canvas, self.screen == "ai", self.assets)
        elif self.screen == "settings":
            settings_view.draw(self.canvas)
        elif self.screen == "modes":
            catalog.draw_modes(self.canvas)
        elif self.screen == "challenges":
            catalog.draw_challenges(self.canvas, self.records.data["challenges"], self.challenge_page)
        elif self.screen == "leaderboard":
            leaderboard_view.draw(self.canvas, self.records, self.leaderboard_tab)
        else:
            game_view.draw(self.scene, self.game, self.assets, self.feedback)
            self.canvas.fill((0, 0, 0, 0))
            if self.screen in ("pause", "over"):
                pause_view.draw(self.canvas, self.game, self.screen == "pause", self.new_high_score)
                self.canvas.set_alpha(round(255 * self.feedback.overlay_opacity))
            elif self.screen == "name":
                leaderboard_view.draw_name(self.canvas, self.player_name, self.game.score)
            elif self.screen == "countdown" or self.go_remaining > 0:
                word = self.countdown.word if self.screen == "countdown" else "GO!"
                if self.screen == "countdown":
                    pygame.draw.rect(self.canvas, (*BG, 238), (548, 272, 184, 180), border_radius=16)
                    label(self.canvas, word, (640, 351), 100, GREEN)
                    label(self.canvas, "GET READY", (640, 427), 18, MUTED)
                else:
                    label(self.canvas, word, (640, 200), 54, GREEN)
            if self.screen == "game":
                self.go_remaining = max(0, self.go_remaining - dt)
        mouse = self.mouse_position(pygame.mouse.get_pos())
        for control in self.controls:
            control.draw(self.canvas, mouse, dt, self.settings.animations,
                         self.pressed == control.action)
        self.scene.blit(pygame.transform.smoothscale(self.canvas, view.size), (0, 0))
        if self.intro_exit_frame is not None:
            if self.screen == "menu" and self.settings.animations:
                self.intro_fade_remaining = max(0, self.intro_fade_remaining - dt)
                frame = pygame.transform.smoothscale(self.intro_exit_frame, view.size)
                frame.set_alpha(round(255 * self.intro_fade_remaining / 0.25))
                self.scene.blit(frame, (0, 0))
            else:
                self.intro_fade_remaining = 0
            if self.intro_fade_remaining == 0:
                self.intro_exit_frame = None
        self.window.fill(BG)
        offset = self.feedback.shake_offset if self.settings.animations else 0
        self.window.blit(self.scene, view.move(offset, 0))
        pygame.display.flip()

    def run(self, frame_limit: int | None = None):
        frames = 0
        try:
            self.start_intro()
            while self.running:
                # Limit a stalled frame's catch-up to keep the UI responsive.
                dt = min(self.clock.tick(120) / 1000, 0.25)
                before = self.timing_epoch
                for event in pygame.event.get():
                    self.handle_event(event)
                # Time elapsed on a paused/menu screen never enters a new game.
                self.update(0 if before != self.timing_epoch else dt)
                self.draw(dt)
                frames += 1
                if frame_limit is not None and frames >= frame_limit:
                    break
        finally:
            self.close()

    def close(self):
        self.finish_session()
        if self.video is not None:
            self.video.close()
            self.video = None
        self.audio.close()
        reset_fonts()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=f"SnakeInfinite {VERSION} — {SUBTITLE}")
    parser.add_argument("--settings", type=Path, help="Override the user settings JSON path")
    parser.add_argument("--smoke-test", action="store_true", help="Render three frames and exit")
    args = parser.parse_args()
    Application(args.settings).run(3 if args.smoke_test else None)
