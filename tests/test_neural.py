"""Real optional CPU Torch tests; explicitly skipped in arcade-only environments."""
import json
import hashlib
import pytest
torch = pytest.importorskip("torch", reason="Optional requirements-ai.txt not installed")
from ai.config import NetworkConfig, ACTIVATIONS
from ai.network import SnakeNetwork
from ai.laboratory import Laboratory
from ai.persistence import ModelStore
from core.types import Direction


def test_default_architecture_parameter_count_and_rng_isolation():
    torch.manual_seed(901)
    state = torch.get_rng_state().clone()
    network = SnakeNetwork()
    assert torch.equal(torch.get_rng_state(), state)
    assert network.config.sizes == (11, 16, 8, 3)
    assert network.parameter_count == 355
    assert network(torch.zeros(11)).shape == (3,)
    assert network(torch.zeros(5, 11)).shape == (5, 3)
    assert all(p.device.type == "cpu" for p in network.parameters())
    other = SnakeNetwork()
    assert all(torch.equal(a, b) for a, b in zip(network.parameters(), other.parameters()))
    different = SnakeNetwork(NetworkConfig(seed=90))
    assert not torch.equal(network.layers[0].weight, different.layers[0].weight)


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("sizes", [(2,), (16, 8), (128,) * 5])
def test_configurable_actual_layers_and_activations(activation, sizes):
    model = SnakeNetwork(NetworkConfig(sizes, activation, 123))
    assert len(model.layers) == len(sizes) + 1
    assert model.parameter_count == sum((a + 1) * b for a, b in zip(model.config.sizes, model.config.sizes[1:]))
    captured = model.capture([1] * 11)
    with torch.no_grad():
        inputs = torch.ones(11)
        for index, layer in enumerate(model.layers):
            sums = layer(inputs)
            inputs = model.activation(sums) if index < len(model.layers) - 1 else sums
            assert captured[index].sums == pytest.approx(sums.tolist())
            assert captured[index].outputs == pytest.approx(inputs.tolist())


def test_inspector_calculation_edit_and_raw_outputs():
    model = SnakeNetwork(NetworkConfig((2,), "tanh"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.edit(0, 0, .5, 0)
    model.edit(0, 0, .2)
    model.edit(1, 0, 2, 0)
    model.edit(1, 1, -3)
    captured = model.capture([1] + [0] * 10)
    data = model.inspect(captured, 0, 0)
    assert data['weights'][0] == .5
    assert data['bias'] == pytest.approx(.2)
    assert data['weighted_sum'] == pytest.approx(.7)
    assert data['output'] == pytest.approx(torch.tanh(torch.tensor(.7)).item())
    assert captured[-1].outputs[1] == -3  # no Softmax
    assert model.layers[0].weight.is_leaf
    assert (0, 0, 0) in model.edited_parameters and (0, 0, None) in model.edited_parameters
    for bad in (float('nan'), float('inf'), 1e7):
        with pytest.raises(ValueError):
            model.edit(0, 0, bad)
    with pytest.raises(IndexError):
        model.edit(-1, 0, 0)


def test_lab_environment_single_actions_reset_and_terminal():
    lab = Laboratory(environment_seed=21)
    first = lab.game.get_state()
    lab.forward()
    assert lab.game.get_state() == first
    lab.step(1)
    assert lab.game.snake.direction == Direction.UP
    assert lab.game.elapsed == pytest.approx(first.interval)
    assert lab.reset(21) == first
    with torch.no_grad():
        for parameter in lab.network.parameters():
            parameter.zero_()
        lab.network.layers[-1].bias.copy_(torch.tensor([0., -1., 5.]))
    assert lab.network_action() == 2
    assert lab.game.snake.direction == Direction.DOWN
    for _ in range(40):
        lab.step()
    state = lab.game.get_state()
    assert lab.game.terminated and lab.step(1) == []
    lab.network_action()
    assert lab.game.get_state() == state
    lab.reset(21)
    assert not lab.game.terminated


def test_manual_inputs_parameters_recalculate_and_restore():
    lab = Laboratory()
    initial = lab.observation
    old = lab.outputs
    lab.set_input(0, 1 - initial[0])
    assert lab.previous_outputs == old
    assert lab.input_source.startswith("Manual")
    lab.edit_parameter(2, 0, 10)
    assert lab.outputs[0] > 9
    assert lab.network.edited_parameters == {(2, 0, None)}
    lab.snake_inputs()
    assert lab.observation == initial and lab.input_source == "Snake snapshot"
    with pytest.raises(ValueError):
        lab.set_input(0, 2)


def test_backward_gradient_reset_inspection_and_optimizer_update():
    lab = Laboratory()
    parameters = [p.detach().clone() for p in lab.network.parameters()]
    outputs = lab.outputs
    expected = sum((x - y) ** 2 for x, y in zip(outputs, (0, 1, 0))) / 3
    assert lab.backward((0, 1, 0)) == pytest.approx(expected)
    gradients = [p.grad.clone() for p in lab.network.parameters()]
    assert lab.gradient(2, 0) == pytest.approx(2 * outputs[0] / 3)
    assert lab.gradient(2, 0, 0) == pytest.approx(lab.network.layers[2].weight.grad[0, 0].item())
    assert all(torch.equal(a, b) for a, b in zip(parameters, lab.network.parameters()))
    lab.backward((0, 1, 0))
    assert all(torch.equal(a, b.grad) for a, b in zip(gradients, lab.network.parameters()))
    after = lab.optimizer_step()
    assert after < expected and lab.demo_steps == 1
    assert lab.previous_outputs == outputs
    assert all(p.grad is None for p in lab.network.parameters())
    assert lab.gradient(2, 0) is not None  # retained explicit snapshot
    with pytest.raises(ValueError):
        lab.optimizer_step()
    lab.backward((0, 1, 0))
    lab.set_input(0, 1)
    assert not lab.gradient_ready and lab.gradients is None
    for target, lr in [((1, 2), .1), ((0, float('nan'), 0), .1), ((0, 1, 0), 0)]:
        with pytest.raises(ValueError):
            lab.backward(target, lr)


def test_model_round_trip_overwrite_and_metadata(tmp_path):
    store = ModelStore(tmp_path / "models")
    model = SnakeNetwork(NetworkConfig((4, 3), "sigmoid", 9))
    model.edit(0, 2, .25, 3)
    folder = store.save("demo", model, demo_steps=4)
    loaded, metadata = store.load("demo")
    assert loaded.config == model.config
    assert loaded.edited_parameters == model.edited_parameters
    assert metadata["supervised_demo_steps"] == 4 and not metadata["reinforcement_learning"]
    assert loaded.capture([0] * 11) == model.capture([0] * 11)
    assert store.names() == ["demo"]
    with pytest.raises(FileExistsError):
        store.save("demo", model)
    store.save("demo", model, overwrite=True)
    assert len(list(folder.glob("weights*.pt"))) == 1
    assert store.load("demo")[0].config == model.config
    with pytest.raises(ValueError):
        store.save("../outside", model)


@pytest.mark.parametrize("corruption", ["json", "config", "checksum", "shape", "nan", "actions", "path", "pickle"])
def test_corrupt_or_incompatible_models_rejected(tmp_path, corruption):
    store = ModelStore(tmp_path)
    model = SnakeNetwork()
    folder = store.save("broken", model)
    path = folder / "network.json"
    metadata = json.loads(path.read_text())
    if corruption == "json":
        path.write_text("{invalid")
    else:
        weights = folder / metadata['weights_file']
        if corruption == "config":
            metadata['architecture']['hidden_sizes'] = [999999]
        elif corruption == "checksum":
            weights.write_bytes(b"corrupt")
        elif corruption in ("shape", "nan", "pickle"):
            state = model.state_dict()
            if corruption == "shape":
                state['layers.0.weight'] = torch.zeros(2, 2)
            elif corruption == "nan":
                state['layers.0.weight'][0, 0] = float('nan')
            else:
                state = {"unsupported": complex(2, 1)}
            torch.save(state, weights)
            metadata['sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()
        elif corruption == "actions":
            metadata['actions'] = list(reversed(metadata['actions']))
        else:
            metadata['weights_file'] = "../../escape.pt"
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        store.load("broken")
