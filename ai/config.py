"""Validated architecture metadata. Importing this module does not import PyTorch."""
from dataclasses import asdict, dataclass

ACTIVATIONS = ("relu", "tanh", "sigmoid", "leaky_relu")


@dataclass(frozen=True)
class NetworkConfig:
    hidden_sizes: tuple[int, ...] = (16, 8)
    activation: str = "relu"
    seed: int | None = 0
    input_size: int = 11
    output_size: int = 3

    def __post_init__(self):
        if (not isinstance(self.hidden_sizes, tuple) or not 1 <= len(self.hidden_sizes) <= 5
                or any(type(size) is not int or not 2 <= size <= 128 for size in self.hidden_sizes)):
            raise ValueError("Use 1–5 hidden layers, each containing 2–128 neurons")
        if self.activation not in ACTIVATIONS:
            raise ValueError("Unsupported activation function")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError("Initialization seed must be blank or an integer from 0 to 2^63−1")
        if type(self.input_size) is not int or type(self.output_size) is not int or (self.input_size, self.output_size) != (11, 3):
            raise ValueError("This observation/action model requires 11 inputs and 3 outputs")

    @property
    def sizes(self):
        return (self.input_size, *self.hidden_sizes, self.output_size)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or set(raw) != {"hidden_sizes", "activation", "seed", "input_size", "output_size"}:
            raise ValueError("Invalid network architecture metadata")
        if not isinstance(raw["hidden_sizes"], list):
            raise ValueError("Hidden sizes must be a JSON array")
        return cls(**{**raw, "hidden_sizes": tuple(raw["hidden_sizes"])})
