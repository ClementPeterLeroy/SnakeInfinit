"""Frozen greedy evaluation with separate statistics and bounded spectator games."""
import math
import random
import torch
from ai.environment import Episode
from ai.training_config import TrainingConfig
from ai.config import NetworkConfig


class Arena:
    def __init__(self, network, config=None, seed=0, name='Model', episode=0):
        self.network=network.eval()
        self.network.requires_grad_(False)
        self.codec=None
        encoder=None
        if not isinstance(network.config,NetworkConfig):
            from ai.registered_networks import ObservationCodec
            from ai.encoders import get_encoder
            encoder=get_encoder(network.config.encoder_id)
            self.codec=ObservationCodec(network.config)
        self.environment=Episode(config or TrainingConfig(),seed,encoder=encoder)
        self.seed,self.name,self.checkpoint_episode=seed,name,episode
        self.rng=random.Random()
        self.running=False
        self.multiplier=1
        self.accumulator=0.
        self.auto_next=False
        self.auto_delay=1.
        self.delay_remaining=0.
        self.games=self.best=0
        self.total_score=0
        self.last_reason=None
        self.recorded=False

    def pause(self):
        self.running=False
        self.accumulator=0.

    def play(self):
        self.accumulator=0.
        self.running=not self.environment.done or self.auto_next

    def reset(self,seed=None):
        self.pause()
        if seed is not None:
            if type(seed) is not int or not 0<=seed<2**63:
                raise ValueError('Arena seed must be an integer from 0 to 2^63-1')
            self.seed=seed
        self.environment.reset(self.seed)
        self.recorded=False
        self.delay_remaining=0.

    def new_game(self):
        seed=self.rng.randrange(2**63)
        while seed==self.seed:
            seed=self.rng.randrange(2**63)
        self.reset(seed)

    def step(self):
        if self.environment.done:
            return None
        with torch.inference_mode():
            observation=self.environment.observation()
            output=self.network(torch.tensor(observation,dtype=torch.float32) if self.codec is None else self.codec.single(observation))
        if not torch.isfinite(output).all():
            self.pause()
            raise ValueError('Non-finite Arena Q-values')
        transition=self.environment.step(output.argmax().item())
        if transition.done and not self.recorded:
            self.recorded=True
            self.games+=1
            score=self.environment.game.score
            self.total_score+=score
            self.best=max(self.best,score)
            self.last_reason=transition.reason
            self.delay_remaining=self.auto_delay
            self.accumulator=0.
            if not self.auto_next:
                self.pause()
        return transition

    def advance(self,dt):
        if not math.isfinite(dt) or dt<0:
            raise ValueError('Arena elapsed time must be finite and nonnegative')
        if not self.running:
            return 0
        if self.environment.done:
            if self.auto_next:
                self.delay_remaining=max(0.,self.delay_remaining-dt)
                if self.delay_remaining==0:
                    self.new_game()
                    self.play()
            return 0
        self.accumulator+=dt
        if self.accumulator>5:
            self.pause()
            raise ValueError('Arena paused: excessive playback backlog canceled')
        count=0
        interval=1/(9*self.multiplier)
        while self.running and not self.environment.done and self.accumulator+1e-10>=interval and count<16:
            self.accumulator=max(0.,self.accumulator-interval)
            self.step()
            count+=1
        return count
