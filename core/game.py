"""Deterministic step-based rules. No Pygame, clocks, audio, or UI imports."""
import random
import math
from core.board import Board
from core.config import BOARD_SIZE, SPEEDS, ARCADE_SPEEDS, GameRules
from core.snake import Snake
from core.types import Direction, GameEvent, GameStatus, GameState, Hazard, Explosion
from core.hazards import safe_candidate, valid_activation, explosion_cells


class Game:
    def __init__(self, speed: str | None = None, *, size: int | None = None,
                 rng: random.Random | None = None, seed: int | None = None,
                 rules: GameRules | None = None):
        if rules is not None and (speed is not None or size is not None):
            raise ValueError("Supply rules or legacy speed/size arguments, not both")
        if rng is not None and seed is not None:
            raise ValueError("Supply rng or seed, not both")
        self.rules = rules if rules is not None else GameRules(
            board_width=BOARD_SIZE if size is None else size,
            board_height=BOARD_SIZE if size is None else size,
            initial_speed=SPEEDS["NORMAL" if speed is None else speed])
        self.board = Board(self.rules.board_width, self.rules.board_height)
        self.rng = random.Random(seed)
        if rng is not None:
            # Retain RNG injection compatibility without sharing mutable state.
            self.rng.setstate(rng.getstate())
        self._reset_state()

    def _reset_state(self):
        x, y = max(2, self.board.width // 2), self.board.height // 2
        self.snake = Snake([(x, y), (x - 1, y), (x - 2, y)])
        if self.rules.initial_length > 3:
            # A contiguous serpentine body with a free forward escape route.
            path = [(px, py) for py in range(self.board.height)
                    for px in (range(self.board.width) if py % 2 == 0
                               else range(self.board.width - 1, -1, -1))]
            body = list(reversed(path[:self.rules.initial_length]))
            head, neck = body[:2]
            direction = Direction((head[0] - neck[0], head[1] - neck[1]))
            if body[0][0] in (0, self.board.width - 1):
                # Finish long starts one cell before the row's boundary so the
                # initial direction has a free forward cell.
                body = list(reversed(path[1:self.rules.initial_length + 1]))
                head, neck = body[:2]
                direction = Direction((head[0] - neck[0], head[1] - neck[1]))
            self.snake = Snake(body, direction)
        self.interval = 1 / self.rules.initial_speed
        self.hazards = []
        self.star = None
        self.star_expires_at = None
        self.elapsed = 0.0
        self.pending_growth = 0
        self.apples_eaten = 0
        self.stars_collected = 0
        self.super_announced = False
        self.last_reward = 0.0
        self.death_cause = None
        self.explosions = []
        self.next_star_at = self.rng.uniform(*self.rules.star_interval) if self.rules.stars_enabled else None
        self.next_hazard_at = (5.0 if self.rules.wall_enabled else self.rules.hazard_interval) if self.rules.hazards_enabled else None
        self.apple = None
        self.apple = self._spawn_cell()
        self.score = 0
        self.status = GameStatus.RUNNING

    def reset(self, seed: int | None = None) -> GameState:
        """Start fresh with these rules. None requests fresh system entropy."""
        self.rng.seed(seed)
        self._reset_state()
        return self.get_state()

    @property
    def terminated(self) -> bool:
        return self.status != GameStatus.RUNNING

    def get_state(self) -> GameState:
        return GameState(self.board.width, self.board.height, tuple(self.snake.body),
                         self.snake.direction, self.snake.pending, self.apple,
                         self.score, self.status, self.interval, self.elapsed,
                         self.star, self.star_remaining, self.pending_growth,
                         self.apples_eaten, self.stars_collected, tuple(self.hazards),
                         self.next_star_at, self.next_hazard_at, self.super_announced,
                         self.last_reward, self.death_cause, tuple(self.explosions))

    @property
    def star_remaining(self):
        return max(0.0, self.star_expires_at - self.elapsed) if self.star_expires_at is not None else 0.0

    @property
    def active_hazards(self):
        return {hazard.cell for hazard in self.hazards if hazard.active(self.elapsed)}

    def _spawn_cell(self):
        blocked = self.snake.body + [hazard.cell for hazard in self.hazards]
        blocked += [cell for cell in (getattr(self, "apple", None), self.star) if cell is not None]
        return self.board.spawn_apple(blocked, self.rng)

    def _ensure_apple(self, events):
        """Food takes priority over optional objects on a nearly full board."""
        if self.apple is not None:
            return
        self.apple = self._spawn_cell()
        if self.apple is None and self.star is not None:
            self.star = self.star_expires_at = None
            self.next_star_at = self.elapsed + self.rng.uniform(*self.rules.star_interval)
            events.append(GameEvent.STAR_EXPIRED)
            self.apple = self._spawn_cell()
        # Walls are permanent and mines finish their deterministic lifecycle.
        # Neither is silently removed to make room for food.

    def _lose(self, cause, events):
        if not self.terminated:
            self.status = GameStatus.LOST
            self.death_cause = cause
            self.last_reward = -1.0
            events.append(GameEvent.LOST)

    def advance_time(self, dt: float) -> list[GameEvent]:
        """Explicit active simulation time; never reads a wall clock or moves."""
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Simulation time must be finite and non-negative")
        if self.terminated:
            return []
        events = []
        # Process timed transitions in chronological order even for a large dt.
        target = self.elapsed + dt
        while True:
            deadlines = [value for value in (self.next_star_at, self.star_expires_at,
                                             self.next_hazard_at) if value is not None]
            deadlines += [time for hazard in self.hazards
                          for time in (hazard.activates_at, hazard.expires_at)
                          if time is not None and time > self.elapsed + 1e-9]
            if self.rules.survival_target is not None:
                deadlines.append(self.rules.survival_target)
            due = min(deadlines, default=float("inf"))
            if due > target + 1e-9:
                self.elapsed = target
                break
            self.elapsed = max(self.elapsed, due)
            if self.star is not None and self.elapsed + 1e-9 >= self.star_expires_at:
                self.star = self.star_expires_at = None
                self.next_star_at = self.elapsed + self.rng.uniform(*self.rules.star_interval)
                events.append(GameEvent.STAR_EXPIRED)
            if self.next_star_at is not None and self.elapsed + 1e-9 >= self.next_star_at:
                self.star = self._spawn_cell()
                self.next_star_at = None if self.star is not None else self.elapsed + 1
                if self.star is not None:
                    self.star_expires_at = self.elapsed + self.rules.star_lifetime
                    events.append(GameEvent.STAR_SPAWNED)
            kept = []
            explosion_hit = False
            for hazard in self.hazards:
                if hazard.expires_at is not None and self.elapsed + 1e-9 >= hazard.expires_at:
                    if hazard.kind == "mine":
                        blast = Explosion(hazard.cell, explosion_cells(self.board, hazard.cell), self.elapsed)
                        # Keep just the most recent simultaneous group. This is
                        # bounded, deterministic across update chunk sizes, and
                        # does not turn the visual trail into a collision zone.
                        if self.explosions and abs(self.explosions[-1].detonated_at - self.elapsed) > 1e-9:
                            self.explosions.clear()
                        self.explosions.append(blast)
                        explosion_hit |= any(cell in blast.cells for cell in self.snake.body)
                        events.append(GameEvent.MINE_EXPLODED)
                    events.append(GameEvent.HAZARD_EXPIRED)
                elif abs(hazard.activates_at - self.elapsed) < 1e-8:
                    if valid_activation(self, hazard):
                        kept.append(hazard)
                        events.append(GameEvent.HAZARD_ACTIVE)
                    else:
                        events.append(GameEvent.HAZARD_EXPIRED)
                else:
                    kept.append(hazard)
            self.hazards = kept
            if explosion_hit:
                # Resolve the whole simultaneous group before emitting one death.
                self._lose("mine_explosion", events)
                break
            if self.rules.survival_target is not None and self.elapsed + 1e-9 >= self.rules.survival_target:
                self.status = GameStatus.WON
                events.append(GameEvent.WON)
                break
            if self.next_hazard_at is not None and self.elapsed + 1e-9 >= self.next_hazard_at:
                kind = "wall" if self.rules.wall_enabled else "mine"
                count = self.rules.wall_spawn_count if kind == "wall" else 1
                for _ in range(count):
                    if kind == "mine" and len(self.hazards) >= 3:
                        break
                    cell = safe_candidate(self, kind)
                    if cell is None:
                        break  # No safe placement: skip, never replace an old wall.
                    active = self.elapsed + self.rules.warning_duration
                    expiry = active + self.rules.mine_lifetime if kind == "mine" else None
                    self.hazards.append(Hazard(cell, kind, active, expiry))
                    events.append(GameEvent.HAZARD_WARNING)
                self.next_hazard_at = self.elapsed + self.rules.hazard_interval
            if self.apple is None:
                self._ensure_apple(events)
        return events

    def turn(self, direction: Direction) -> bool:
        return self.status == GameStatus.RUNNING and self.snake.turn(direction)

    def step(self, action: Direction | None = None, *, dt: float | None = None) -> list[GameEvent]:
        """Advance exactly one cell, optionally submitting a direction first."""
        if self.status != GameStatus.RUNNING:
            return []
        if action is not None:
            self.turn(action)
        self.last_reward = 0.0
        events = self.advance_time(self.interval if dt is None else dt)
        if self.terminated:
            return events
        head = self.snake.next_head()
        apple_eaten = head == self.apple
        star_eaten = self.star is not None and head == self.star
        grow = apple_eaten or star_eaten or self.pending_growth > 0
        occupied = self.snake.body if grow else self.snake.body[:-1]
        if not self.board.contains(head) or head in occupied or head in self.active_hazards:
            obstacle = next((hazard.kind for hazard in self.hazards
                             if hazard.cell == head and hazard.active(self.elapsed)), None)
            cause = ("board_edge" if not self.board.contains(head) else "self_collision"
                     if head in occupied else "mine_contact" if obstacle == "mine" else "wall_contact")
            self._lose(cause, events)
            return events
        before = self.score
        if apple_eaten:
            self.score += 1
            self.apples_eaten += 1
            self.pending_growth += 1
            self.apple = None
            events.append(GameEvent.ATE)
        if star_eaten:
            self.score += 2
            self.stars_collected += 1
            self.pending_growth += 2
            self.star = self.star_expires_at = None
            self.next_star_at = self.elapsed + self.rng.uniform(*self.rules.star_interval)
            events.append(GameEvent.STAR)
        self.snake.advance(head, grow)
        if self.pending_growth:
            self.pending_growth -= 1
        self.last_reward = float(self.score - before)
        if before < 25 <= self.score and not self.super_announced:
            self.super_announced = True
            events.append(GameEvent.SUPER)
        if self.rules.progressive_speed:
            self.interval = 1 / max(rate for threshold, rate in ARCADE_SPEEDS if self.score >= threshold)
        if self.apple is None:
            self._ensure_apple(events)
        won = (len(self.snake.body) == self.board.width * self.board.height
               or self.rules.score_target is not None and self.score >= self.rules.score_target
               or self.rules.star_target is not None and self.stars_collected >= self.rules.star_target)
        if won:
            self.status = GameStatus.WON
            events.append(GameEvent.WON)
        return events
