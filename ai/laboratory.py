"""Classic lab environment, detached forward inspection, and supervised demo steps."""
import copy
import math
import torch
from core.game import Game
from ai.actions import RelativeAction, relative_direction
from ai.config import NetworkConfig
from ai.network import SnakeNetwork
from ai.observations import encode_observation, validate_observation


class Laboratory:
    def __init__(self, config=NetworkConfig(), environment_seed=0):
        self.game = Game(seed=environment_seed)
        self.environment_seed = environment_seed
        self.observation = encode_observation(self.game.get_state())
        self.input_source = "Snake snapshot"
        self.learning_rate = 0.001
        self.target = (0.0, 1.0, 0.0)
        self.previous_outputs = None
        self.outputs = None
        self.demo_steps = 0
        self.simulation_steps = 0
        self.replace_network(SnakeNetwork(config))

    def invalidate_gradients(self):
        self.network.zero_grad(set_to_none=True)
        self.gradients = None
        self.gradient_ready = False
        self.loss_before = self.loss_after = None

    def replace_network(self, network):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        self.previous_outputs = self.outputs = None
        self.demo_steps = 0
        self.invalidate_gradients()
        self.forward()

    def build(self, config):
        self.replace_network(SnakeNetwork(config))

    def forward(self):
        captured = self.network.capture(self.observation)
        self.previous_outputs = self.outputs
        self.captured = captured
        self.outputs = captured[-1].outputs
        return self.outputs

    def snake_inputs(self):
        self.observation = encode_observation(self.game.get_state())
        self.input_source = "Snake snapshot"
        self.invalidate_gradients()
        return self.forward()

    def set_input(self, index, value):
        if type(index) is not int or not 0 <= index < 11:
            raise IndexError("Observation index out of range")
        old = self.observation
        values = list(old)
        values[index] = value
        self.observation = validate_observation(values)
        try:
            self.forward()
        except ValueError:
            self.observation = old
            raise
        self.input_source = "Manual binary inputs"
        self.invalidate_gradients()

    def reset(self, seed=None):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("Environment seed must be blank or an integer from 0 to 2^63−1")
        self.environment_seed = seed
        self.game.reset(seed=seed)
        self.simulation_steps = 0
        self.snake_inputs()
        return self.game.get_state()

    def step(self, action=RelativeAction.STRAIGHT, *, refresh=True):
        if self.game.terminated:
            return []
        direction = self.game.snake.pending or self.game.snake.direction
        if self.game.snake.pending is not None:
            raise ValueError("Commit one lab action at a time; clear external buffered input first")
        events = self.game.step(relative_direction(direction, action))
        self.simulation_steps += 1
        if refresh:
            self.snake_inputs()
        return events

    def network_action(self):
        # This performs exactly one user-requested action, never an episode loop.
        self.forward()
        action = RelativeAction(max(range(3), key=lambda index: self.outputs[index]))
        self.step(action)
        return action

    def edit_parameter(self, layer, neuron, value, input_index=None):
        self.network.inspect(self.captured, layer, neuron)
        target = self.network.layers[layer]
        parameter = target.bias[neuron] if input_index is None else target.weight[neuron, input_index]
        previous = parameter.detach().item()
        edited = self.network.edited_parameters.copy()
        self.network.edit(layer, neuron, value, input_index)
        try:
            self.forward()
        except ValueError:
            self.network.edit(layer, neuron, previous, input_index)
            self.network.edited_parameters = edited
            self.invalidate_gradients()
            raise
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.invalidate_gradients()

    def backward(self, target, learning_rate=0.001):
        try:
            target = tuple(float(value) for value in target)
            learning_rate = float(learning_rate)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Supply three finite targets and a finite learning rate") from error
        if len(target) != 3 or any(not math.isfinite(value) or abs(value) > 1e6 for value in target):
            raise ValueError("Targets must be three finite numbers within ±1e6")
        if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be greater than 0 and at most 1")
        self.invalidate_gradients()
        self.target, self.learning_rate = target, learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        output = self.network(torch.tensor(self.observation, dtype=torch.float32))
        loss = torch.nn.functional.mse_loss(output, torch.tensor(target, dtype=torch.float32))
        if not torch.isfinite(loss):
            raise ValueError("Loss overflowed; reduce parameter magnitudes")
        loss.backward()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
               for parameter in self.network.parameters()):
            self.invalidate_gradients()
            raise ValueError("Gradients overflowed; reduce parameter magnitudes")
        self.loss_before = loss.detach().item()
        self.gradients = tuple(dict(weights=layer.weight.grad.detach().clone(),
                                    biases=layer.bias.grad.detach().clone()) for layer in self.network.layers)
        self.gradient_ready = True
        return self.loss_before

    def gradient(self, layer, neuron, input_index=None):
        if self.gradients is None:
            return None
        gradient = self.gradients[layer]
        return (gradient["biases"][neuron] if input_index is None
                else gradient["weights"][neuron, input_index]).item()

    def optimizer_step(self):
        if not self.gradient_ready:
            raise ValueError("Run BACKPROPAGATE before an optimizer step")
        original = {key: value.clone() for key, value in self.network.state_dict().items()}
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        try:
            self.optimizer.step()
            self.forward()
            self.loss_after = torch.nn.functional.mse_loss(
                torch.tensor(self.outputs), torch.tensor(self.target)).item()
            if not math.isfinite(self.loss_after):
                raise ValueError("Optimizer update overflowed")
        except (ValueError, RuntimeError):
            self.network.load_state_dict(original)
            self.optimizer.load_state_dict(optimizer_state)
            self.forward()
            self.invalidate_gradients()
            raise
        self.gradient_ready = False
        self.optimizer.zero_grad(set_to_none=True)
        self.demo_steps += 1
        return self.loss_after
