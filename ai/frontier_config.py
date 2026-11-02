"""Execution policy, separate from the unchanged DQN hyperparameters."""
from dataclasses import asdict,dataclass
from ai.training_config import TrainingConfig


@dataclass(frozen=True)
class FrontierConfig:
    refresh_episodes: int = 100
    headless: bool = True
    recovery_interval: int = 1000
    recovery_slots: int = 2
    milestones: tuple[int,...] = (1000,3000,5000,10000)
    profiling: bool = False

    def __post_init__(self):
        if self.refresh_episodes not in (25,100,500):raise ValueError('Refresh must be 25, 100 or 500 episodes')
        if type(self.headless) is not bool or type(self.profiling) is not bool:raise ValueError('Invalid execution flags')
        if type(self.recovery_interval) is not int or not 1<=self.recovery_interval<=1000000:raise ValueError('Recovery interval must be 1..1,000,000 episodes')
        if type(self.recovery_slots) is not int or not 1<=self.recovery_slots<=10:raise ValueError('Recovery slots must be 1..10')
        if not isinstance(self.milestones,tuple) or any(type(n) is not int or n<1 for n in self.milestones):raise ValueError('Invalid milestones')

    def to_dict(self):return asdict(self)
