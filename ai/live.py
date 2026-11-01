"""Nonblocking, user-started inference scheduler over the existing laboratory game."""
from dataclasses import dataclass
import math
from ai.actions import RelativeAction
from ai.config import NetworkConfig
from ai.observations import encode_observation
from core.types import GameState

SPEEDS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class DecisionSnapshot:
    """Detached pre-action calculation paired explicitly with its resulting board."""
    before: GameState
    after: GameState
    observation: tuple
    captured: tuple
    outputs: tuple
    action: RelativeAction
    config: NetworkConfig
    weights: tuple
    biases: tuple
    step: int

    def inspect(self, layer, neuron):
        if not 0 <= layer < len(self.captured) or not 0 <= neuron < len(self.biases[layer]):
            raise IndexError("Snapshot neuron index out of range")
        data = self.captured[layer]
        return dict(inputs=data.inputs, weights=self.weights[layer][neuron], bias=self.biases[layer][neuron],
                    weighted_sum=data.sums[neuron], output=data.outputs[neuron],
                    activation=self.config.activation if layer < len(self.captured) - 1 else "linear")


class LiveController:
    MAX_STEPS_PER_FRAME = 8
    MAX_BACKLOG_SECONDS = 5.0

    def __init__(self, laboratory):
        self.lab = laboratory
        self.mode = "step"
        self.running = False
        self.speed = 5
        self.accumulator = 0.0
        self.decision = None
        self.previous_decision = None
        self.message = "Step Mode / CPU / raw untrained outputs"
        self.model_label='UNTRAINED'

    def pause(self, message="Live paused / decision preserved"):
        self.running = False
        self.accumulator = 0.0
        self.message = message

    def set_mode(self, mode):
        if mode not in ("step", "live"):
            raise ValueError("Unknown laboratory operating mode")
        self.pause("Step Mode / environment preserved" if mode == "step" else "Live ready / press PLAY")
        self.mode = mode

    def play(self):
        self.mode = "live"
        self.accumulator = 0.0
        self.running = not self.lab.game.terminated
        self.message = f"LIVE / {self.model_label} / argmax control" if self.running else "GAME OVER / reset to play again"

    def stop(self):
        self.set_mode("step")

    def set_speed(self, speed):
        if type(speed) is not int or speed not in SPEEDS:
            raise ValueError("Choose 1, 3, 5, 10 or 20 steps/second")
        if self.accumulator * self.speed >= 1:
            # Explicitly report cancellation of scheduled catch-up work.
            self.pause("Speed changed: pending catch-up canceled; press PLAY")
        self.accumulator = 0.0
        self.speed = speed

    def reset(self, seed=None):
        self.pause("Environment reset / press PLAY")
        state = self.lab.reset(seed)
        self.decision = self.previous_decision = None
        return state

    def infer(self):
        """Read the supplied game's current state; no reset or movement assumptions."""
        before = self.lab.game.get_state()
        self.lab.observation = encode_observation(before)
        self.lab.input_source = "Decision snapshot / PRE-ACTION"
        self.lab.invalidate_gradients()
        self.lab.forward()
        action = RelativeAction(max(range(3), key=lambda i: self.lab.outputs[i]))
        return before, action

    def tick(self):
        if self.lab.game.terminated:
            self.pause("GAME OVER / final board and last decision preserved")
            return False
        before, action = self.infer()
        # Capture parameters as well as activations, so later edits cannot rewrite history.
        weights = tuple(tuple(tuple(row) for row in layer.weight.detach().tolist()) for layer in self.lab.network.layers)
        biases = tuple(tuple(layer.bias.detach().tolist()) for layer in self.lab.network.layers)
        self.lab.step(action, refresh=False)
        self.previous_decision = self.decision
        self.decision = DecisionSnapshot(before, self.lab.game.get_state(), self.lab.observation,
            self.lab.captured, self.lab.outputs, action, self.lab.network.config, weights, biases,
            self.lab.simulation_steps)
        if self.lab.game.terminated:
            self.pause("GAME OVER / final board and last decision preserved")
        return True

    def advance(self, dt):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Elapsed time must be finite and nonnegative")
        if not self.running:
            return 0
        self.accumulator += dt
        if self.accumulator > self.MAX_BACKLOG_SECONDS:
            pending = self.accumulator
            self.pause(f"Live paused: {pending:.2f}s backlog canceled after a long stall. Press PLAY to resume.")
            return 0
        count = 0
        interval = 1 / self.speed
        while self.running and self.accumulator + 1e-10 >= interval and count < self.MAX_STEPS_PER_FRAME:
            self.accumulator = max(0.0, self.accumulator - interval)
            count += int(self.tick())
        # All remaining ordinary catch-up work stays in the accumulator.
        return count
