import random
import pytest
from core.game import Game
from core.types import Direction, GameEvent, GameStatus


def test_initial_state():
    game = Game(rng=random.Random(7))
    assert game.board.size == 25
    assert len(game.snake.body) == 3
    assert game.snake.direction == Direction.RIGHT
    assert game.apple not in game.snake.body
    assert game.score == 0


def test_eating_grows_and_emits_one_event():
    game = Game(rng=random.Random(0))
    game.apple = (13, 12)
    assert game.step() == [GameEvent.ATE]
    assert game.score == 1
    assert len(game.snake.body) == 4
    assert game.apple not in game.snake.body


def test_wall_collision_emits_once_and_freezes():
    game = Game()
    game.snake.body = [(24, 5), (23, 5), (22, 5)]
    assert game.step() == [GameEvent.LOST]
    assert game.status == GameStatus.LOST
    assert game.step() == []
    assert game.snake.body[0] == (24, 5)


def test_body_collision():
    game = Game()
    game.snake.body = [(5, 5), (5, 6), (6, 6), (6, 5), (7, 5)]
    game.apple = (0, 0)
    assert game.step() == [GameEvent.LOST]


def test_tail_cell_is_legal_when_it_vacates():
    game = Game()
    game.snake.body = [(5, 5), (5, 6), (6, 6), (6, 5)]
    game.apple = (0, 0)
    assert game.step() == []
    assert game.snake.body == [(6, 5), (5, 5), (5, 6), (6, 6)]


def test_full_board_completes_without_spawning():
    game = Game(size=3)
    game.snake.body = [(1, 0), (0, 0), (0, 1), (1, 1),
                       (2, 1), (2, 2), (1, 2), (0, 2)]
    game.apple = (2, 0)
    assert game.step() == [GameEvent.ATE, GameEvent.WON]
    assert game.status == GameStatus.WON
    assert game.apple is None
    assert len(set(game.snake.body)) == 9
    assert game.step() == []


@pytest.mark.parametrize("speed,rate", [("CASUAL", 6), ("NORMAL", 9), ("FAST", 13)])
def test_speed_constant_after_eating(speed, rate):
    game = Game(speed)
    game.apple = (13, 12)
    game.step()
    assert game.interval == 1 / rate


def test_spawn_uses_only_free_cell():
    game = Game(size=3)
    occupied = [(x, y) for x in range(3) for y in range(3) if (x, y) != (2, 2)]
    assert game.board.spawn_apple(occupied, random.Random()) == (2, 2)
    assert game.board.spawn_apple(occupied + [(2, 2)], random.Random()) is None
