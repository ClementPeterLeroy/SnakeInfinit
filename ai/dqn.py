"""CPU DQN independent of Pygame. One environment tick per train_step()."""
from collections import deque
import copy
import math
import random
import torch
from ai.network import SnakeNetwork
from ai.config import NetworkConfig
from ai.environment import Episode
from ai.replay import ReplayBuffer
from ai.training_config import TrainingConfig


def bellman_targets(rewards, next_values, done, gamma):
    # Finite episodic return: both true termination and administrative truncation
    # close an episode. Neither bootstraps into the reset environment.
    return rewards + gamma * next_values * (~done).to(rewards.dtype)


class Trainer:
    def __init__(self, config=TrainingConfig(), network=None, architecture=None):
        self.config = config
        if network is not None:self.online=copy.deepcopy(network)
        elif architecture is not None and not isinstance(architecture,NetworkConfig):
            from ai.model_registry import build_model
            self.online=build_model(architecture)
        else:self.online=SnakeNetwork(architecture or NetworkConfig(seed=config.seed))
        self.codec=None
        encoder=None
        if not isinstance(self.online.config,NetworkConfig):
            from ai.encoders import get_encoder
            from ai.registered_networks import ObservationCodec
            encoder=get_encoder(self.online.config.encoder_id)
            self.codec=ObservationCodec(self.online.config)
        if any(p.device.type!='cpu' or not torch.isfinite(p).all() for p in self.online.parameters()):
            raise ValueError('Training requires finite CPU parameters')
        self.online.requires_grad_(True)
        self.target = copy.deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity, config.seed + 1)
        self.action_rng = random.Random(config.seed + 2)
        self.environment_rng = random.Random(config.seed + 3)
        self.torch_rng_state = torch.Generator(device='cpu').manual_seed(config.seed).get_state()
        self.environment = Episode(config,encoder=encoder)
        self.at_boundary = True
        self.completed = self.total_steps = self.apples = self.collisions = self.stagnations = self.best = 0
        self.loss = self.gradient_norm = None
        self.optimizer_updates = 0
        self.elapsed = 0.
        self.cumulative_reward=0.
        self.scores, self.lengths, self.rewards = deque(maxlen=20000), deque(maxlen=20000), deque(maxlen=20000)
        self.losses, self.epsilons = deque(maxlen=2000), deque(maxlen=2000)
        self.last_reason = None

    @property
    def epsilon(self):
        fraction = min(1., self.total_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (self.config.epsilon_min - self.config.epsilon_start)

    def choose_action(self, observation, explore=True):
        if explore and self.action_rng.random() < self.epsilon:
            return self.action_rng.randrange(3)
        with torch.no_grad():
            values = self.online(torch.tensor(observation, dtype=torch.float32) if self.codec is None else self.codec.single(observation))
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite Q-values; training stopped")
        return values.argmax().item()

    def optimize(self):
        if len(self.replay) < self.config.batch_size:
            return None
        batch = self.replay.sample(self.config.batch_size)
        observations = torch.tensor([t.observation for t in batch], dtype=torch.float32) if self.codec is None else self.codec.stack([t.observation for t in batch])
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.tensor([t.next_observation for t in batch], dtype=torch.float32) if self.codec is None else self.codec.stack([t.next_observation for t in batch])
        done = torch.tensor([t.done for t in batch], dtype=torch.bool)
        predicted = self.online(observations).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            expected = bellman_targets(rewards, self.target(next_states).max(1).values, done, self.config.gamma)
        loss = torch.nn.functional.smooth_l1_loss(predicted, expected)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite TD loss; training stopped")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer_updates += 1
        if any(not torch.isfinite(p).all() for p in self.online.parameters()):
            raise ValueError("Non-finite model parameters; training stopped")
        self.loss, self.gradient_norm = loss.item(), norm.item()
        self.losses.append(self.loss)
        return self.loss

    def train_step(self):
        if self.completed >= self.config.episodes:
            return None
        if self.at_boundary:
            self.environment.reset(self.environment_rng.randrange(2**63))
            self.at_boundary = False
        action = self.choose_action(self.environment.observation())
        score_before=self.environment.game.score
        transition = self.environment.step(action)
        self.replay.append(transition)
        self.total_steps += 1
        self.apples += self.environment.game.score-score_before
        self.cumulative_reward+=transition.reward
        self.optimize()
        if self.total_steps % self.config.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.epsilons.append(self.epsilon)
        if transition.done:
            self.completed += 1
            self.at_boundary = True
            score = self.environment.game.score
            self.best = max(self.best, score)
            self.scores.append(score)
            self.lengths.append(self.environment.steps)
            self.rewards.append(self.environment.total_reward)
            self.last_reason = transition.reason
            self.collisions += int(transition.terminated and transition.reason != 'board_completion')
            self.stagnations += int(transition.reason == 'stagnation')
        return transition

    def metrics(self):
        recent = list(self.scores)[-100:]
        return dict(episode=min(self.completed + int(not self.at_boundary), self.config.episodes),
            completed=self.completed, score=self.environment.game.score, best=self.best,
            average=sum(recent)/len(recent) if recent else 0., epsilon=self.epsilon, loss=self.loss,
            steps=self.total_steps, apples=self.apples, collisions=self.collisions, stagnations=self.stagnations,
            elapsed=self.elapsed, gradient_norm=self.gradient_norm, last_reason=self.last_reason,
            average_length=sum(self.lengths)/len(self.lengths) if self.lengths else 0.,
            reward_total=self.cumulative_reward, parameter_count=self.online.parameter_count,
            collision_frequency=self.collisions/max(1,self.completed), stagnation_frequency=self.stagnations/max(1,self.completed),
            apples_per_step=self.apples/max(1,self.total_steps))
