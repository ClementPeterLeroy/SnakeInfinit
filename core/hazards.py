"""Safe obstacle placement, including conservative apple reachability checks."""
from collections import deque


def explosion_cells(board, center):
    """The exact clipped Chebyshev-radius-one neighborhood, in stable row order."""
    x, y = center
    return tuple((px, py) for py in range(y - 1, y + 2)
                 for px in range(x - 1, x + 2) if board.contains((px, py)))


def committed_cell(game):
    x, y = game.snake.body[0]
    dx, dy = (game.snake.pending or game.snake.direction).value
    return x + dx, y + dy


def reachable(game, candidate):
    start = game.snake.body[0]
    target = game.apple
    if target is None:
        return True
    blocked = set(game.snake.body[1:-1]) | {hazard.cell for hazard in game.hazards} | {candidate}
    pending = deque([start])
    seen = {start}
    while pending:
        x, y = pending.popleft()
        if (x, y) == target:
            return True
        for cell in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if game.board.contains(cell) and cell not in seen and cell not in blocked:
                seen.add(cell)
                pending.append(cell)
    return False


def valid_activation(game, hazard):
    cell = hazard.cell
    head = game.snake.body[0]
    if cell in game.snake.body or cell in (game.apple, game.star, committed_cell(game)):
        return False
    if hazard.kind == "wall":
        return (abs(cell[0] - head[0]) + abs(cell[1] - head[1]) >= 5
                and reachable(game, cell))
    return True


def safe_candidate(game, kind, exclude=()):
    head = game.snake.body[0]
    occupied = set(game.snake.body) | {h.cell for h in game.hazards}
    occupied.update(exclude)
    occupied.update(cell for cell in (game.apple, game.star, committed_cell(game)) if cell is not None)
    if kind == "wall" and game.board.width * game.board.height - len(occupied) <= 1:
        return None  # Leave room for food rather than forcing a crowded spawn.
    cells = [(x, y) for y in range(game.board.height) for x in range(game.board.width)
             if (x, y) not in occupied and abs(x - head[0]) + abs(y - head[1]) >= 5]
    game.rng.shuffle(cells)
    return next((cell for cell in cells if kind != "wall" or reachable(game, cell)), None)
