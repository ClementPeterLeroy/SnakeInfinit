"""Bounded replay with its own reproducible uniform sampling stream."""
import random


class ReplayBuffer:
    def __init__(self, capacity=100000, seed=0):
        self.items = []
        self.capacity, self.position = capacity, 0
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def append(self, transition):
        if len(self.items) < self.capacity:
            self.items.append(transition)
        else:
            self.items[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, size):
        if not 0 < size <= len(self.items):
            raise ValueError("Replay sample must fit available transitions")
        return self.rng.sample(self.items, size)
