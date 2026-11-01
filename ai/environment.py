"""Classic episode wrapper. Administrative limits never change arcade rules."""
from dataclasses import dataclass
from core.game import Game
from core.types import GameStatus
from ai.actions import relative_direction
from ai.observations import encode_observation
from ai.training_config import TrainingConfig


@dataclass(frozen=True)
class Transition:
    observation: tuple
    action: int
    reward: float
    next_observation: tuple
    terminated: bool
    truncated: bool
    reason: str | None

    @property
    def done(self):
        return self.terminated or self.truncated


class Episode:
    def __init__(self, config=TrainingConfig(), seed=0):
        self.config = config
        self.game = Game(seed=seed)
        self.steps = self.without_food = 0
        self.done = False
        self.reason = None
        self.total_reward = 0.

    def reset(self, seed):
        self.game.reset(seed)
        self.steps = self.without_food = 0
        self.done, self.reason, self.total_reward = False, None, 0.
        return self.observation()

    def observation(self):
        return encode_observation(self.game.get_state())

    def step(self, action):
        if self.done:
            raise ValueError("Reset a completed episode before stepping")
        observation, score = self.observation(), self.game.score
        self.game.step(relative_direction(self.game.snake.direction, action))
        self.steps += 1
        ate = self.game.score > score
        self.without_food = 0 if ate else self.without_food + 1
        terminated, truncated, reason = self.game.terminated, False, None
        reward = self.config.food_reward if ate else self.config.step_reward
        if self.game.status == GameStatus.WON:
            reason = 'board_completion'
        elif terminated:
            reason = 'wall_collision' if self.game.death_cause == 'board_edge' else 'self_collision'
            reward = self.config.collision_reward
        elif self.without_food >= self.config.stagnation_factor * len(self.game.snake.body):
            truncated, reason, reward = True, 'stagnation', self.config.stagnation_reward
        elif self.steps >= self.config.episode_cap:
            truncated, reason = True, 'episode_step_limit'
        self.done, self.reason = terminated or truncated, reason
        self.total_reward += reward
        return Transition(observation, int(action), reward, self.observation(), terminated, truncated, reason)
