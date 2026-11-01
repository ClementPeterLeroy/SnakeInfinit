"""Validated, serializable DQN hyperparameters; no tensor or display imports."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 10000
    learning_rate: float = .001
    gamma: float = .99
    replay_capacity: int = 100000
    batch_size: int = 64
    epsilon_start: float = 1.
    epsilon_min: float = .05
    epsilon_decay_steps: int = 100000
    target_sync: int = 1000
    checkpoint_interval: int = 500
    seed: int = 0
    food_reward: float = 10.
    collision_reward: float = -10.
    step_reward: float = -.01
    stagnation_reward: float = -5.
    stagnation_factor: int = 100
    episode_cap: int = 10000
    gradient_clip: float = 10.

    def __post_init__(self):
        limits = dict(episodes=(1,1000000), replay_capacity=(1,1000000), batch_size=(1,4096),
            epsilon_decay_steps=(1,1000000000), target_sync=(1,1000000000),
            checkpoint_interval=(1,1000000), seed=(0,2**63-1), stagnation_factor=(1,100000), episode_cap=(1,1000000))
        for key, (low, high) in limits.items():
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} must be an integer from {low} to {high}")
        for key in ('learning_rate','gamma','epsilon_start','epsilon_min','food_reward',
                    'collision_reward','step_reward','stagnation_reward','gradient_clip'):
            value = getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not 0 < self.learning_rate <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("Learning rate must be in (0,1], gamma in [0,1]")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1")
        if self.batch_size > self.replay_capacity or not 0 < self.gradient_clip <= 10000:
            raise ValueError("Batch must fit replay; gradient clip must be in (0,10000]")
        if any(abs(getattr(self,key))>1e6 for key in ('food_reward','collision_reward','step_reward','stagnation_reward')):
            raise ValueError('Reward magnitudes must be at most 1e6')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw,dict) or set(raw) != set(cls().to_dict()):
            raise ValueError("Invalid training configuration fields")
        return cls(**raw)
