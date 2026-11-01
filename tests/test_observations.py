"""The Classic observation/action contract stays pure Python."""
from dataclasses import replace
import subprocess
import sys
import pytest
from ai.observations import FEATURE_NAMES, encode_observation, validate_observation
from ai.actions import relative_direction
from ai.config import NetworkConfig
from core.game import Game
from core.types import Direction, GameStatus


def test_feature_order_and_food_diagonal_without_mutation():
    game = Game(seed=34)
    game.apple = (0, 0)
    before, rng = game.get_state(), game.rng.getstate()
    assert FEATURE_NAMES == ("Danger straight", "Danger right", "Danger left", "Moving up",
                             "Moving down", "Moving left", "Moving right", "Food above",
                             "Food below", "Food left", "Food right")
    assert encode_observation(before) == (0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0)
    assert game.get_state() == before and game.rng.getstate() == rng


@pytest.mark.parametrize("direction,index", [(Direction.UP, 3), (Direction.DOWN, 4),
                                              (Direction.LEFT, 5), (Direction.RIGHT, 6)])
def test_direction_encoding(direction, index):
    state = replace(Game(seed=0).get_state(), snake=((0, 0),), direction=direction, apple=(4, 5))
    observation = encode_observation(state)
    assert observation[index] == 1 and sum(observation[3:7]) == 1
    assert observation[7:] == (0, 1, 0, 1)


def test_danger_order_walls_body_and_vacating_tail():
    state = replace(Game().get_state(), snake=((0, 0), (0, 1), (1, 1)), direction=Direction.UP, apple=(5, 5))
    assert encode_observation(state)[:3] == (1, 0, 1)
    loop = replace(state, snake=((1, 1), (1, 2), (2, 2), (2, 1)), direction=Direction.RIGHT)
    assert encode_observation(loop)[:3] == (0, 1, 0)  # tail goes away, neck doesn't
    assert encode_observation(replace(loop, pending_growth=1))[0] == 1
    assert encode_observation(replace(loop, apple=(2, 1)))[0] == 1


def test_buffered_heading_terminal_and_seeded_snapshots():
    game = Game(seed=111)
    second = Game(seed=111)
    assert encode_observation(game.get_state()) == encode_observation(second.get_state())
    game.turn(Direction.UP)
    state = game.get_state()
    assert encode_observation(state)[3:7] == (1, 0, 0, 0)
    assert game.get_state() == state
    terminal = replace(state, status=GameStatus.WON, apple=None)
    assert encode_observation(terminal)[7:] == (0, 0, 0, 0)


@pytest.mark.parametrize("values", [[], [0] * 10, [0] * 12, [float('nan')] * 11, [2] * 11, None])
def test_bad_observations(values):
    with pytest.raises(ValueError):
        validate_observation(values)


@pytest.mark.parametrize("direction,left,right", [(Direction.UP, Direction.LEFT, Direction.RIGHT),
    (Direction.DOWN, Direction.RIGHT, Direction.LEFT), (Direction.LEFT, Direction.DOWN, Direction.UP),
    (Direction.RIGHT, Direction.UP, Direction.DOWN)])
def test_relative_actions(direction, left, right):
    assert relative_direction(direction, 0) == direction
    assert relative_direction(direction, 1) == left
    assert relative_direction(direction, 2) == right
    with pytest.raises(ValueError):
        relative_direction(direction, 3)


@pytest.mark.parametrize("kwargs", [{"hidden_sizes": ()}, {"hidden_sizes": (1,)}, {"hidden_sizes": (129,)},
    {"hidden_sizes": (4,) * 6}, {"hidden_sizes": (True,)}, {"activation": "softmax"},
    {"seed": -1}, {"seed": True}, {"input_size": 10}, {"output_size": 4}])
def test_config_rejects_invalid_architecture(kwargs):
    with pytest.raises(ValueError):
        NetworkConfig(**kwargs)


def test_headless_encoder_executes_without_tensor_or_display_imports():
    script = '''
import builtins
old = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in ('pygame', 'torch'):
        raise AssertionError(name)
    return old(name, *args, **kwargs)
builtins.__import__ = guard
from core.game import Game
from ai.observations import encode_observation
from ai.actions import relative_direction
game = Game(seed=0)
for i in range(3000):
    if game.terminated: game.reset(seed=i)
    game.step(relative_direction(game.snake.direction, i % 3))
    assert len(encode_observation(game.get_state())) == 11
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
