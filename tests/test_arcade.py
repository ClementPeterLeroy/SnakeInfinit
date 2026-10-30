import random
import pytest
from core.config import GameRules, arcade_rules, ARCADE_SPEEDS
from core.challenges import CHALLENGES, BY_ID
from core.game import Game
from core.hazards import committed_cell, reachable
from core.types import Direction, GameEvent, GameStatus, Hazard


@pytest.mark.parametrize("threshold,speed", ARCADE_SPEEDS)
def test_arcade_speed_thresholds(threshold, speed):
    game = Game(rules=arcade_rules(), seed=7)
    game.score = max(0, threshold - 1)
    if threshold:
        game.apple = (13, 12)
    else:
        game.apple = (0, 0)
    game.step(dt=0)
    assert game.interval == 1 / speed


def test_classic_has_no_timed_arcade_mechanics():
    game = Game(seed=4)
    game.advance_time(1000)
    assert game.star is None and not game.hazards
    assert game.interval == 1 / 9


def test_star_spawn_and_expiry_keep_apple_available():
    game = Game(rules=arcade_rules(), seed=5)
    apple = game.apple
    assert 15 <= game.next_star_at <= 25
    assert GameEvent.STAR_SPAWNED in game.advance_time(game.next_star_at)
    assert game.star not in game.snake.body and game.star != apple
    assert game.star_remaining == 6
    assert GameEvent.STAR_EXPIRED in game.advance_time(6)
    assert game.star is None and game.apple == apple
    assert 15 <= game.next_star_at - game.elapsed <= 25


def place_star(game, cell=(13, 12)):
    game.star = cell
    game.star_expires_at = game.elapsed + 6
    game.next_star_at = None
    game.apple = (0, 0)


def test_star_scoring_pending_growth_and_apples_overlap():
    game = Game(rules=arcade_rules(), seed=1)
    place_star(game)
    assert GameEvent.STAR in game.step(dt=0)
    assert (game.score, len(game.snake.body), game.pending_growth) == (2, 4, 1)
    assert game.last_reward == 2
    game.apple = (14, 12)
    game.step(dt=0)
    assert (game.score, len(game.snake.body), game.pending_growth) == (3, 5, 1)
    game.step(dt=0)
    assert len(game.snake.body) == 6 and game.pending_growth == 0
    assert game.apples_eaten == 1 and game.stars_collected == 1


def test_pending_growth_blocks_tail_entry():
    game = Game()
    game.snake.body = [(5, 5), (5, 6), (6, 6), (6, 5)]
    game.apple = (0, 0)
    game.pending_growth = 1
    assert GameEvent.LOST in game.step(dt=0)


def test_super_threshold_handles_star_jump_once_and_resets():
    game = Game(rules=arcade_rules())
    game.score = 24
    place_star(game)
    assert GameEvent.SUPER in game.step(dt=0)
    assert game.score == 26
    game.apple = (14, 12)
    assert GameEvent.SUPER not in game.step(dt=0)
    game.reset(seed=1)
    assert not game.super_announced and game.pending_growth == 0


@pytest.mark.parametrize("challenge_id,target", [("centurion", 70), ("labyrinth", 30), ("minefield", 25)])
def test_score_challenges_complete_at_target(challenge_id, target):
    game = Game(rules=BY_ID[challenge_id].rules)
    game.score = target - 1
    game.apple = (13, 12)
    assert GameEvent.WON in game.step(dt=0)
    assert game.score == target and game.status == GameStatus.WON


def test_star_hunter_requires_tenth_star_not_score():
    game = Game(rules=BY_ID["star_hunter"].rules, seed=4)
    assert 10 <= game.next_star_at <= 15
    game.score = 100
    game.stars_collected = 9
    place_star(game)
    assert GameEvent.WON in game.step(dt=0)
    assert game.stars_collected == 10


def test_speed_demon_legal_long_start_and_active_survival():
    game = Game(rules=BY_ID["speed_demon"].rules, seed=3)
    assert len(game.snake.body) == len(set(game.snake.body)) == 50
    assert all(game.board.contains(cell) for cell in game.snake.body)
    assert committed_cell(game) not in game.snake.body
    for first, second in zip(game.snake.body, game.snake.body[1:]):
        assert abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
    assert game.interval == 1 / 20
    game.step()
    assert not game.terminated
    game.advance_time(44.9 - game.elapsed)
    assert not game.terminated
    assert game.advance_time(0.1) == [GameEvent.WON]
    assert game.elapsed == pytest.approx(45)


def test_labyrinth_warns_and_accumulates_permanent_walls():
    game = Game(rules=BY_ID["labyrinth"].rules, seed=4)
    game.advance_time(5)
    assert len(game.hazards) == 1 and not game.active_hazards
    head = game.snake.body[0]
    wall = game.hazards[0]
    assert abs(wall.cell[0] - head[0]) + abs(wall.cell[1] - head[1]) >= 5
    assert wall.cell not in game.snake.body and wall.cell not in (game.apple, committed_cell(game))
    assert reachable(game, wall.cell)
    game.advance_time(1)
    assert len(game.active_hazards) == 1
    game.advance_time(24)
    assert len(game.hazards) == len(game.active_hazards) == 4
    before = {hazard.cell for hazard in game.hazards}
    game.advance_time(8)
    after = {hazard.cell for hazard in game.hazards}
    assert len(after) == 5 and before < after
    for _ in range(20):
        before = set(game.active_hazards)
        game.advance_time(8)
        assert before <= game.active_hazards
    assert len(game.active_hazards) > 4


def test_warning_revalidated_before_wall_activation():
    game = Game(rules=BY_ID["labyrinth"].rules, seed=1)
    game.advance_time(5)
    cell = game.hazards[0].cell
    game.snake.body[0] = cell
    game.advance_time(1)
    assert cell not in game.active_hazards


def test_reachability_rejects_apple_isolation():
    game = Game(size=5)
    game.snake.body = [(1, 1), (0, 1), (0, 2)]
    game.apple = (4, 4)
    game.hazards = [Hazard((3, 4), "wall", 0)]
    assert not reachable(game, (4, 3))


def test_mine_warning_activation_expiry_and_capacity():
    game = Game(rules=BY_ID["minefield"].rules, seed=6)
    assert GameEvent.HAZARD_WARNING in game.advance_time(8)
    mine = game.hazards[0]
    assert not mine.active(game.elapsed)
    assert mine.cell not in game.snake.body and mine.cell not in (game.apple, committed_cell(game))
    assert GameEvent.HAZARD_ACTIVE in game.advance_time(1)
    assert mine.cell in game.active_hazards
    events = game.advance_time(5)
    assert GameEvent.HAZARD_EXPIRED in events and GameEvent.MINE_EXPLODED in events
    assert not game.hazards
    for _ in range(50):
        game.advance_time(1)
        assert len(game.active_hazards) <= 3


@pytest.mark.parametrize("kind", ["mine", "wall"])
def test_active_hazard_collision(kind):
    game = Game()
    game.hazards = [Hazard((13, 12), kind, 0, 10 if kind == "mine" else None)]
    assert GameEvent.LOST in game.step(dt=0)
    assert game.last_reward == -1


def test_warning_cell_is_not_lethal():
    game = Game()
    game.hazards = [Hazard((13, 12), "mine", 1, 6)]
    game.apple = (0, 0)
    game.step(dt=0)
    assert not game.terminated


@pytest.mark.parametrize("rules", [arcade_rules(), BY_ID["labyrinth"].rules, BY_ID["minefield"].rules])
def test_timed_modes_reproduce_across_chunking_and_reset(rules):
    first, second = Game(rules=rules, seed=123), Game(rules=rules, seed=123)
    first.advance_time(60)
    for _ in range(60):
        second.advance_time(1)
    assert first.get_state() == second.get_state()
    assert first.reset(seed=123) == Game(rules=rules, seed=123).get_state()


def test_invalid_time_rejected_and_terminal_time_freezes():
    game = Game()
    for dt in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            game.advance_time(dt)
    game.status = GameStatus.LOST
    game.advance_time(100)
    assert game.elapsed == 0


def test_nearly_full_board_keeps_apple_by_releasing_optional_star():
    game = Game(rules=GameRules(mode="arcade", stars_enabled=True, board_width=3, board_height=3))
    game.snake.body = [(1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (1, 2)]
    game.snake.direction = Direction.RIGHT
    game.apple = (2, 0)
    game.star = (0, 2)
    game.star_expires_at = 10
    game.next_star_at = None
    game.step(dt=0)
    assert game.apple == (0, 2) and game.star is None
    assert not game.terminated
