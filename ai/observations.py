"""Classic-only observation model. No Pygame or tensor-library dependency."""
import math
from core.types import Direction, GameState

FEATURE_NAMES = ("Danger straight", "Danger right", "Danger left", "Moving up",
                 "Moving down", "Moving left", "Moving right", "Food above",
                 "Food below", "Food left", "Food right")
DIRECTIONS = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)


def validate_observation(values) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Observation must contain 11 binary numbers") from error
    if len(result) != 11 or any(not math.isfinite(value) or value not in (0.0, 1.0) for value in result):
        raise ValueError("Observation must contain 11 binary numbers")
    return result


def encode_observation(state: GameState) -> tuple[float, ...]:
    """Use the committed heading if a turn is pending; never consume that turn.

    Only Classic wall/body rules and the apple are represented. Stars and
    challenge hazards are deliberately outside this first observation model.
    Terminal snapshots remain inspectable; absent food sets all food flags to 0.
    """
    if not state.snake:
        raise ValueError("A Snake observation requires a nonempty body")
    direction = state.pending_direction or state.direction
    index = DIRECTIONS.index(direction)
    x, y = state.snake[0]
    danger = []
    for heading in (direction, DIRECTIONS[(index + 1) % 4], DIRECTIONS[(index - 1) % 4]):
        dx, dy = heading.value
        cell = x + dx, y + dy
        grows = state.pending_growth > 0 or cell == state.apple
        body = state.snake if grows else state.snake[:-1]
        danger.append(float(not (0 <= cell[0] < state.board_width and 0 <= cell[1] < state.board_height)
                            or cell in body))
    moving = [float(direction == heading) for heading in
              (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)]
    food = [0.0] * 4 if state.apple is None else [
        float(state.apple[1] < y), float(state.apple[1] > y),
        float(state.apple[0] < x), float(state.apple[0] > x)]
    return validate_observation((*danger, *moving, *food))
