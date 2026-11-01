"""Actual CPU PyTorch layers, forward captures, and individual parameter edits."""
from dataclasses import dataclass
import math
import secrets
import torch
from torch import nn
from ai.config import NetworkConfig
from ai.observations import validate_observation

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
                "leaky_relu": nn.LeakyReLU}


@dataclass(frozen=True)
class LayerCapture:
    inputs: tuple[float, ...]
    sums: tuple[float, ...]
    outputs: tuple[float, ...]


class SnakeNetwork(nn.Module):
    def __init__(self, config: NetworkConfig = NetworkConfig()):
        super().__init__()
        self.config = config
        self.edited_parameters = set()
        # Initialization is repeatable without perturbing a caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed if config.seed is not None else secrets.randbits(63))
            self.layers = nn.ModuleList(nn.Linear(left, right, device="cpu", dtype=torch.float32)
                                       for left, right in zip(config.sizes, config.sizes[1:]))
        self.activation = _ACTIVATIONS[config.activation]()

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, inputs):
        if inputs.ndim not in (1, 2) or inputs.shape[-1] != 11:
            raise ValueError("Network expects an 11-feature tensor")
        if inputs.device.type != "cpu" or not torch.isfinite(inputs).all():
            raise ValueError("Inputs must be finite CPU values")
        for index, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if index < len(self.layers) - 1:
                inputs = self.activation(inputs)
        return inputs  # Raw outputs: deliberately no Softmax.

    def capture(self, observation):
        inputs = torch.tensor(validate_observation(observation), dtype=torch.float32)
        captured = []
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                sums = layer(inputs)
                outputs = self.activation(sums) if index < len(self.layers) - 1 else sums
                if not torch.isfinite(sums).all() or not torch.isfinite(outputs).all():
                    raise ValueError("Network values overflowed; reduce parameter magnitudes")
                captured.append(LayerCapture(tuple(inputs.tolist()), tuple(sums.tolist()), tuple(outputs.tolist())))
                inputs = outputs
        return tuple(captured)

    def inspect(self, capture, layer_index, neuron_index):
        if type(layer_index) is not int or not 0 <= layer_index < len(self.layers):
            raise IndexError("Layer index out of range")
        layer = self.layers[layer_index]
        if type(neuron_index) is not int or not 0 <= neuron_index < layer.out_features:
            raise IndexError("Neuron index out of range")
        data = capture[layer_index]
        return dict(layer=layer_index, neuron=neuron_index, inputs=data.inputs,
                    weights=tuple(layer.weight[neuron_index].detach().tolist()),
                    bias=layer.bias[neuron_index].detach().item(), weighted_sum=data.sums[neuron_index],
                    activation=self.config.activation if layer_index < len(self.layers) - 1 else "linear",
                    output=data.outputs[neuron_index])

    def edit(self, layer_index, neuron_index, value, input_index=None):
        if type(layer_index) is not int or not 0 <= layer_index < len(self.layers):
            raise IndexError("Layer index out of range")
        value = float(value)
        if not math.isfinite(value) or abs(value) > 1e6:
            raise ValueError("Parameter must be finite and between −1e6 and 1e6")
        layer = self.layers[layer_index]
        if type(neuron_index) is not int or not 0 <= neuron_index < layer.out_features:
            raise IndexError("Neuron index out of range")
        if input_index is not None and (type(input_index) is not int or not 0 <= input_index < layer.in_features):
            raise IndexError("Input index out of range")
        with torch.no_grad():
            if input_index is None:
                layer.bias[neuron_index] = value
            else:
                layer.weight[neuron_index, input_index] = value
        self.edited_parameters.add((layer_index, neuron_index, input_index))
        self.zero_grad(set_to_none=True)
