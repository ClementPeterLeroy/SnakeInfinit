"""Definitive v1.2 challenge rules, without rendering-dependent collisions."""
from dataclasses import replace
import subprocess
import sys
import pytest
from core.board import Board
from core.challenges import BY_ID
from core.config import VERSION, SUBTITLE
from core.game import Game
from core.hazards import explosion_cells, committed_cell, reachable
from core.types import Direction, GameEvent, GameStatus, Hazard
from services.records import RecordsStore


def mine_game(cell, *, expires_at=5.0):
    game = Game(rules=BY_ID["minefield"].rules, seed=123)
    game.next_hazard_at = None
    game.hazards = [Hazard(cell, "mine", 0, expires_at)]
    return game


def test_version_and_unchanged_challenge_objectives():
    assert VERSION == "v1.5.0" and SUBTITLE == "Analysis & Experiment Framework"
    assert BY_ID["centurion"].rules.score_target == 70
    hunter = BY_ID["star_hunter"].rules
    assert hunter.star_target == 10 and hunter.star_interval == (10, 15)
    assert hunter.star_lifetime == 6


def test_speed_demon_has_a_full_second_of_safe_initial_forward_movement():
    game = Game(rules=BY_ID["speed_demon"].rules, seed=1)
    assert len(game.snake.body) == len(set(game.snake.body)) == 50
    assert all(game.board.contains(cell) for cell in game.snake.body)
    assert game.board.contains(committed_cell(game))
    assert committed_cell(game) not in game.snake.body
    assert game.interval == 0.05 and game.rules.survival_target == 45
    game.apple = (0, 0)
    for _ in range(20):
        game.step()
    assert game.elapsed == pytest.approx(1)
    assert not game.terminated


def test_permanent_walls_stay_safe_and_accumulate_without_a_global_cap():
    game = Game(rules=BY_ID["labyrinth"].rules, seed=123)
    previous = set()
    for index in range(14):
        game.advance_time(5 if index == 0 else 7)
        warnings = [h for h in game.hazards if not h.active(game.elapsed)]
        assert len(warnings) == 1
        for wall in warnings:
            head = game.snake.body[0]
            assert abs(wall.cell[0] - head[0]) + abs(wall.cell[1] - head[1]) >= 5
            assert wall.cell not in game.snake.body
            assert wall.cell not in (game.apple, game.star, committed_cell(game))
            assert reachable(game, wall.cell)
            assert wall.expires_at is None
        game.advance_time(1)
        assert previous < game.active_hazards
        previous = set(game.active_hazards)
    assert len(previous) == 14


def test_wall_waves_are_per_cycle_and_not_a_total_limit():
    rules = replace(BY_ID["labyrinth"].rules, wall_spawn_count=4)
    game = Game(rules=rules, seed=42)
    game.advance_time(6)
    first = set(game.active_hazards)
    assert len(first) == 4
    game.advance_time(8)
    assert len(game.active_hazards) == 8 and first < game.active_hazards


@pytest.mark.parametrize("count", [0, 5, True, 1.5])
def test_invalid_wall_wave_sizes_are_rejected(count):
    with pytest.raises(ValueError):
        replace(BY_ID["labyrinth"].rules, wall_spawn_count=count)


def test_wall_spawns_skip_crowded_board_instead_of_forcing_placement():
    rules = replace(BY_ID["labyrinth"].rules, board_width=3, board_height=3)
    game = Game(rules=rules, seed=1)
    # No position on this board can be five cells away from the initial head.
    assert game.advance_time(100) == []
    assert not game.hazards and not game.terminated


def test_food_respawn_cannot_remove_permanent_walls():
    rules = replace(BY_ID["labyrinth"].rules, board_width=3, board_height=3)
    game = Game(rules=rules)
    game.snake.body = [(1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (1, 2)]
    game.snake.direction = Direction.RIGHT
    game.apple = (2, 0)
    wall = Hazard((0, 2), "wall", 0)
    game.hazards = [wall]
    events = game.step(dt=0)
    assert game.hazards == [wall] and wall.cell in game.active_hazards
    assert GameEvent.HAZARD_EXPIRED not in events


def test_food_respawn_cannot_defuse_a_mine_before_detonation():
    rules = replace(BY_ID["minefield"].rules, board_width=3, board_height=3)
    game = Game(rules=rules)
    game.next_hazard_at = None
    game.snake.body = [(1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (1, 2)]
    game.snake.direction = Direction.RIGHT
    game.apple = (2, 0)
    mine = Hazard((0, 2), "mine", 0, 5)
    game.hazards = [mine]
    game.step(dt=0)
    assert game.hazards == [mine]
    events = game.advance_time(5)
    assert GameEvent.MINE_EXPLODED in events and game.death_cause == "mine_explosion"


def test_mine_capacity_includes_warnings_and_active_mines():
    rules = replace(BY_ID["minefield"].rules, hazard_interval=0.5)
    game = Game(rules=rules, seed=123)
    detonations = 0
    for _ in range(80):
        detonations += game.advance_time(0.25).count(GameEvent.MINE_EXPLODED)
        assert len(game.hazards) <= 3
    assert detonations >= 3


@pytest.mark.parametrize("center,count", [((12, 12), 9), ((0, 0), 4), ((24, 24), 4),
                                        ((0, 12), 6), ((24, 0), 4), ((12, 24), 6)])
def test_blast_is_exact_three_by_three_neighborhood_clipped_to_board(center, count):
    board = Board(25)
    cells = explosion_cells(board, center)
    expected = {(x, y) for y in range(25) for x in range(25)
                if max(abs(x - center[0]), abs(y - center[1])) <= 1}
    assert set(cells) == expected and len(cells) == count


@pytest.mark.parametrize("center", [(13, 13), (9, 12)])
def test_explosion_kills_head_or_body_at_exact_detonation_time(center):
    game = mine_game(center)
    if center == (9, 12):
        assert game.snake.body[0] not in explosion_cells(game.board, center)
        assert game.snake.body[-1] in explosion_cells(game.board, center)
    game.advance_time(4.99)
    assert not game.terminated
    events = game.advance_time(20)
    assert game.elapsed == 5
    assert events.count(GameEvent.MINE_EXPLODED) == events.count(GameEvent.LOST) == 1
    assert game.status == GameStatus.LOST and game.death_cause == "mine_explosion"
    assert game.last_reward == -1
    assert not game.hazards
    state = game.get_state()
    assert state.death_cause == "mine_explosion" and state.explosions[0].detonated_at == 5
    assert game.advance_time(20) == [] and game.step() == []
    assert game.get_state() == state


def test_surviving_explosion_is_instantaneous_without_a_lingering_collision_zone():
    game = mine_game((15, 12), expires_at=1)
    game.snake.body = [(13, 12), (12, 12), (11, 12)]
    game.apple = (0, 0)
    events = game.step(dt=1)
    assert GameEvent.MINE_EXPLODED in events and GameEvent.LOST not in events
    # Detonation resolved before this movement tick. Its old 3x3 area is safe now.
    assert game.snake.body[0] == (14, 12)
    game.step(dt=0)
    assert game.snake.body[0] == (15, 12) and not game.terminated
    assert GameEvent.MINE_EXPLODED not in game.advance_time(20)
    assert len(game.explosions) == 1 and not game.hazards


def test_mine_lifecycle_warns_activates_then_detonates_after_five_active_seconds():
    game = Game(rules=BY_ID["minefield"].rules, seed=6)
    assert GameEvent.HAZARD_WARNING in game.advance_time(8)
    mine = game.hazards[0]
    assert not mine.active(game.elapsed)
    assert GameEvent.HAZARD_ACTIVE in game.advance_time(1)
    assert mine.active(game.elapsed)
    assert GameEvent.MINE_EXPLODED not in game.advance_time(4.99)
    assert GameEvent.MINE_EXPLODED in game.advance_time(0.01)
    assert game.explosions[0].detonated_at == pytest.approx(14)
    assert not game.hazards


@pytest.mark.parametrize("fatal", [True, False])
def test_simultaneous_explosions_all_resolve_with_at_most_one_death(fatal):
    game = mine_game((13, 13) if fatal else (1, 1), expires_at=1)
    game.hazards.append(Hazard((9, 12) if fatal else (22, 22), "mine", 0, 1))
    events = game.advance_time(1)
    assert events.count(GameEvent.MINE_EXPLODED) == 2
    assert events.count(GameEvent.LOST) == int(fatal)
    assert len(game.explosions) == 2 and not game.hazards
    assert game.advance_time(1) == []


def test_reset_clears_explosion_and_death_state_reproducibly():
    game = mine_game((13, 13))
    game.advance_time(5)
    assert game.death_cause and game.explosions
    state = game.reset(seed=123)
    assert state.death_cause is None and not state.explosions
    assert state == Game(rules=BY_ID["minefield"].rules, seed=123).get_state()


def test_explosion_death_cause_persists_with_challenge_statistics(tmp_path):
    store = RecordsStore(tmp_path / "records.json")
    game = mine_game((9, 12))
    game.advance_time(5)
    store.record_game(game)
    loaded = RecordsStore(store.path)
    assert loaded.data["statistics"]["deaths"] == 1
    assert loaded.data["challenges"]["minefield"]["last_death_cause"] == "mine_explosion"


def test_reworked_challenges_and_explosions_remain_headless_and_seeded():
    script = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'pygame' or name.startswith('pygame.'):
        raise AssertionError('Pygame imported into simulation')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from core.challenges import BY_ID
from core.game import Game
from core.types import Hazard
for challenge in ('speed_demon', 'labyrinth', 'minefield'):
    first = Game(rules=BY_ID[challenge].rules, seed=777)
    second = Game(rules=BY_ID[challenge].rules, seed=777)
    for _ in range(30):
        assert first.advance_time(1) == second.advance_time(1)
        assert first.get_state() == second.get_state()
game = Game(rules=BY_ID['minefield'].rules)
game.hazards = [Hazard((9,12), 'mine', 0, 1)]
game.advance_time(1)
assert game.death_cause == 'mine_explosion'
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
