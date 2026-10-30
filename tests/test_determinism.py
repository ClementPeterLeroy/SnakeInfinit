"""Reproducible episodes and observations without rendering dependencies."""
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError
import pytest
from core.config import GameRules
from core.game import Game
from core.types import Direction, GameEvent, GameStatus


def test_seed_reproduces_action_sequence_and_food_spawns():
    first, second = Game(size=8, seed=12345), Game(size=8, seed=12345)
    # Fixed circuit: every cell is visited, so the replay includes many meals.
    path = [(x, 0) for x in range(8)]
    for y in range(1, 8):
        path.extend((x, y) for x in (range(7, 0, -1) if y % 2 else range(1, 8)))
    path.extend((0, y) for y in range(7, 0, -1))
    index = path.index(first.snake.body[0])
    meals = 0
    for _ in range(1500):
        previous, target = path[index], path[(index + 1) % len(path)]
        action = Direction((target[0] - previous[0], target[1] - previous[1]))
        events = first.step(action)
        assert events == second.step(action)
        assert first.get_state() == second.get_state()
        meals += GameEvent.ATE in events
        index = (index + 1) % len(path)
        if first.terminated:
            break
    assert meals >= 10
    assert first.status != GameStatus.LOST


def test_different_seeds_produce_different_food_sequences():
    sequences = []
    for seed in (1, 2):
        game = Game(seed=seed)
        sequences.append([game.board.spawn_apple(game.snake.body, game.rng) for _ in range(20)])
    assert sequences[0] != sequences[1]


def test_reset_replays_initial_state_and_clears_episode():
    game = Game(seed=12345)
    initial = game.get_state()
    game.apple = (13, 12)
    game.step()
    game.turn(Direction.UP)
    game.status = GameStatus.LOST
    assert game.reset(seed=12345) == initial
    assert game.score == 0 and not game.terminated
    assert game.snake.pending is None
    assert game.step() == Game(seed=12345).step()


def test_global_random_and_injected_rng_are_not_shared():
    injected = random.Random(123)
    before = injected.getstate()
    first = Game(rng=injected)
    assert injected.getstate() == before
    second = Game(rng=injected)
    random.seed(99)
    for _ in range(10):
        assert first.board.spawn_apple(first.snake.body, first.rng) == second.board.spawn_apple(second.snake.body, second.rng)
        random.random()


def test_unseeded_reset_requests_system_entropy(monkeypatch):
    game = Game(seed=4)
    calls = []
    original = game.rng.seed
    def record(seed):
        calls.append(seed)
        original(seed)
    monkeypatch.setattr(game.rng, "seed", record)
    game.reset()
    assert calls == [None]


def test_snapshot_is_detached_and_immutable():
    game = Game(seed=0)
    state = game.get_state()
    game.step(Direction.UP)
    assert state.snake[0] == (12, 12)
    assert state.direction == Direction.RIGHT
    with pytest.raises(FrozenInstanceError):
        state.score = 999


def test_headless_ten_thousand_steps_without_pygame():
    script = '''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'pygame' or name.startswith('pygame.'):
        raise AssertionError('Engine imported Pygame')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from core.game import Game
from core.types import Direction
game = Game(seed=10)
episodes = 0
for _ in range(10000):
    if game.terminated:
        episodes += 1
        game.reset(seed=episodes)
    game.step(Direction.RIGHT)
assert episodes > 100
assert 'pygame' not in sys.modules
print('10000 headless steps completed')
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "10000 headless steps completed" in result.stdout


def test_classic_rules_defaults_and_legacy_compatibility():
    rules = GameRules()
    assert (rules.board_width, rules.board_height, rules.initial_speed) == (25, 25, 9)
    assert not rules.progressive_speed and not rules.stars_enabled and not rules.hazards_enabled
    assert Game(seed=1).get_state() == Game(rules=rules, seed=1).get_state()
    assert Game("FAST", size=10).interval == 1 / 13


def test_rectangular_rules_survive_reset_and_respect_bounds():
    rules = GameRules(board_width=11, board_height=7, initial_speed=6)
    game = Game(rules=rules, seed=3)
    assert game.board.contains((10, 6))
    assert not game.board.contains((11, 6))
    assert not game.board.contains((10, 7))
    for _ in range(20):
        assert game.board.contains(game.apple)
        assert game.apple not in game.snake.body
        game.step()
        if game.terminated:
            game.reset(seed=4)
    assert game.rules == rules and game.interval == 1 / 6


@pytest.mark.parametrize("kwargs", [dict(board_width=2), dict(board_height=0),
    dict(board_width=4.5), dict(initial_speed=0), dict(initial_speed=float('nan')),
    dict(initial_speed=float('inf')), dict(initial_speed=True), dict(stars_enabled=1)])
def test_invalid_rules_rejected(kwargs):
    with pytest.raises(ValueError):
        GameRules(**kwargs)


@pytest.mark.parametrize("flag", ["progressive_speed", "stars_enabled", "hazards_enabled"])
def test_future_mechanics_explicitly_unavailable(flag):
    with pytest.raises(NotImplementedError):
        GameRules(**{flag: True})


def test_conflicting_configuration_is_rejected():
    with pytest.raises(ValueError):
        Game("FAST", rules=GameRules())
    with pytest.raises(ValueError):
        Game(seed=1, rng=random.Random())
