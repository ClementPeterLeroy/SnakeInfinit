from core.snake import Snake
from core.types import Direction


def test_single_buffer_prevents_reversal_between_ticks():
    snake = Snake([(5, 5), (4, 5), (3, 5)])
    assert not snake.turn(Direction.LEFT)
    assert snake.turn(Direction.UP)
    assert not snake.turn(Direction.LEFT)
    assert snake.next_head() == (5, 4)
    assert snake.turn(Direction.LEFT)
    assert snake.body[0] == (5, 5)  # Input never advances position.


def test_same_direction_does_not_consume_turn_buffer():
    snake = Snake([(5, 5), (4, 5), (3, 5)])
    assert not snake.turn(Direction.RIGHT)
    assert snake.turn(Direction.DOWN)
