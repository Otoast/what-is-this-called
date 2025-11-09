"""
Endgame-aware Tron AI (a1k0n-inspired) with boost support.

Key features:
- Supports boost moves (e.g., "LEFT:BOOST", "UP:BOOST") when foo.bar() returns True.
- Uses minimax + alpha-beta pruning.
- Switches to endgame exhaustive mode when players are disconnected.
- Works with 'R', 'B', '.' board representation and dummy_state format.
"""

import math
import copy
from collections import deque

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------

MAX_DEPTH = 6
ENDGAME_COMPONENT_MULTIPLIER = 1000
ENDGAME_EXHAUSTIVE_THRESHOLD = 20  # Switch to full search if few cells left

MOVES = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "UP:BOOST": (0, -2),
    "DOWN:BOOST": (0, 2),
    "LEFT:BOOST": (-2, 0),
    "RIGHT:BOOST": (2, 0),
}

# --------------------------------------------------------------
# GRID UTILITIES
# --------------------------------------------------------------


def print_board(grid):
    for row in grid:
        print(' '.join(row))

def in_bounds(x, y, grid):
    return 0 <= x < len(grid[0]) and 0 <= y < len(grid)

def valid_move(grid, x, y):
    """A valid move is within bounds and not already occupied."""
    return in_bounds(x, y, grid) and grid[y][x] == '.'


def move_position(pos, direction):
    dx, dy = direction
    return pos[0] + dx, pos[1] + dy

def _convert_board(state):
    board = [['.' for _ in range(20)] for _ in range(18)]
    for trail in state["agent1_trail"]:
        board[trail[1]][trail[0]] = 'R'
    for trail in state["agent2_trail"]:
        board[trail[1]][trail[0]] = 'B'
    return board

def convert_trail(trail):
    return [(y - 1, x - 1) for (y, x) in trail]

def copy_grid(grid):
    return [row[:] for row in grid]


def count_empty(grid):
    return sum(row.count('.') for row in grid)

# ---- attempt_move now supports multi-step (boost) moves safely ----
def attempt_move(grid, head, head_char, direction):
    """
    Try to move along `direction` (dx,dy). If the move length > 1 (boost),
    verify each intermediate cell is free and mark the trail for every step.
    Returns (new_grid, new_head) or (grid, None) if invalid.
    """
    dx, dy = direction
    steps = max(abs(dx), abs(dy))
    step_dx = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_dy = 0 if dy == 0 else (1 if dy > 0 else -1)

    # validate entire path
    path = []
    cx, cy = head
    for i in range(1, steps + 1):
        tx, ty = cx + step_dx * i, cy + step_dy * i
        if not in_bounds(tx, ty, grid) or grid[ty][tx] != '.':
            return grid, None
        path.append((tx, ty))

    # apply path (mark trail at each visited cell)
    new_grid = copy_grid(grid)
    for (tx, ty) in path:
        new_grid[ty][tx] = head_char

    new_head = path[-1]
    return new_grid, new_head

# --------------------------------------------------------------
# FLOOD-FILL UTILITIES
# --------------------------------------------------------------

def bfs_distance_map(grid, start):
    """Compute distances from start to every cell using BFS."""
    h, w = len(grid), len(grid[0])
    dist = [[math.inf] * w for _ in range(h)]
    sx, sy = start
    if not in_bounds(sx, sy, grid) or grid[sy][sx] != '.':
        return dist
    dist[sy][sx] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if valid_move(grid, nx, ny) and dist[ny][nx] > dist[y][x] + 1:
                dist[ny][nx] = dist[y][x] + 1
                q.append((nx, ny))
    return dist


def flood_fill(grid, start):
    """Return all reachable cells from start position."""
    visited = set()
    q = deque([start])
    start_handled = False
    while q:
        x, y = q.popleft()
        if (x, y) in visited or not in_bounds(x, y, grid):
            continue
        elif grid[y][x] != '.' and start_handled:
            continue
        visited.add((x, y))
        start_handled = True
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            q.append((x + dx, y + dy))
    return visited

def players_disconnected(grid, my_head, opponent_head):
    """Return True if players have no overlapping reachable space."""
    red_reach = flood_fill(grid, my_head)
    blue_reach = flood_fill(grid, opponent_head)
    return len(red_reach.intersection(blue_reach)) == 0



# --------------------------------------------------------------
# EVALUATION FUNCTIONS
# --------------------------------------------------------------

def open_neighbor_bonus(grid, x, y):
    """Count number of open adjacent cells (used in evaluation)."""
    return sum(1 for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)] if valid_move(grid, x + dx, y + dy))


def evaluate_voronoi(grid, my_head, opponent_head):
    """Voronoi area control heuristic with openness bias."""
    red_dist = bfs_distance_map(grid, my_head)
    blue_dist = bfs_distance_map(grid, opponent_head)
    score = 0
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y][x] != '.':
                continue
            rd, bd = red_dist[y][x], blue_dist[y][x]
            if rd < bd:
                score += 1 + open_neighbor_bonus(grid, x, y)
            elif bd < rd:
                score -= 1 + open_neighbor_bonus(grid, x, y)
    return score


def evaluate(grid, my_head, opponent_head):
    """Combine Voronoi heuristic and endgame flood-fill separation."""
    if players_disconnected(grid, my_head, opponent_head):
        red_area = flood_fill(grid, my_head)
        blue_area = flood_fill(grid, opponent_head)
        return ENDGAME_COMPONENT_MULTIPLIER * (len(red_area) - len(blue_area))
    else:
        return evaluate_voronoi(grid, my_head, opponent_head)


# --------------------------------------------------------------
# MINIMAX WITH BOOST SUPPORT
# --------------------------------------------------------------

# ---- greedy_move: iterate direction tuples; ignore boost moves for greedy heuristic ----
def greedy_move(grid, head):
    """Pick the non-boost move with the most open neighbors (tie-breaker for walls)."""
    best, best_score = None, -math.inf
    for move_name, direction in MOVES.items():
        if ":BOOST" in move_name:
            continue  # skip boost moves in greedy playout heuristic
        nx, ny = move_position(head, direction)
        if not valid_move(grid, nx, ny):
            continue
        sc = open_neighbor_bonus(grid, nx, ny)
        if sc > best_score:
            best_score, best = sc, direction
    return best

# ---- minimax: don't reuse new_grid across moves; handle boosts properly ----

def minimax(grid, my_head, opponent_head, my_color, opponent_color, depth, alpha, beta, maximizing, red_boosts_remaining, blue_boosts_remaining):
    if depth == MAX_DEPTH:
        return evaluate(grid, my_head, opponent_head)
    # Endgame check: disconnected heads
    if players_disconnected(grid, my_head, opponent_head):
        empties = count_empty(grid)
        if empties <= ENDGAME_EXHAUSTIVE_THRESHOLD:
            return exhaustive_endgame(grid, my_head, opponent_head, my_color, opponent_color)
        else:
            return evaluate(grid, my_head, opponent_head)

    if maximizing:
        max_eval = -math.inf
        for move_name, direction in MOVES.items():
            # boost availability check
            if ":BOOST" in move_name and red_boosts_remaining <= 0:
                continue

            # simulate move from the original grid for every branch
            new_grid, new_red = attempt_move(grid, my_head, my_color, direction)
            if new_red is None:
                continue

            next_red_boosts = red_boosts_remaining - 1 if ":BOOST" in move_name else red_boosts_remaining
            val = minimax(new_grid, new_red, opponent_head, my_color, opponent_color, depth + 1, alpha, beta, False, next_red_boosts, blue_boosts_remaining)
            max_eval = max(max_eval, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        # if no legal moves, return a large negative (dead) evaluation
        if max_eval == -math.inf:
            return -ENDGAME_COMPONENT_MULTIPLIER  # trapped player
        return max_eval

    else:
        min_eval = math.inf
        for move_name, direction in MOVES.items():
            if ":BOOST" in move_name and blue_boosts_remaining <= 0:
                continue

            new_grid, new_blue = attempt_move(grid, opponent_head, opponent_color, direction)
            if new_blue is None:
                continue

            next_blue_boosts = blue_boosts_remaining - 1 if ":BOOST" in move_name else blue_boosts_remaining
            val = minimax(new_grid, my_head, new_blue, my_color, opponent_color, depth + 1, alpha, beta, True, red_boosts_remaining, next_blue_boosts)
            min_eval = min(min_eval, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        if min_eval == math.inf:
            return ENDGAME_COMPONENT_MULTIPLIER  # opponent trapped
        return min_eval

# --------------------------------------------------------------
# EXHAUSTIVE ENDGAME SEARCH (a1k0n-style)
# --------------------------------------------------------------

def greedy_move(grid, head):
    """Pick the non-boost move with the most open neighbors (tie-breaker for walls)."""
    best, best_score = None, -math.inf
    for move_name, direction in MOVES.items():
        if ":BOOST" in move_name:
            continue  # skip boost moves in greedy playout heuristic
        nx, ny = move_position(head, direction)
        if not valid_move(grid, nx, ny):
            continue
        sc = open_neighbor_bonus(grid, nx, ny)
        if sc > best_score:
            best_score, best = sc, direction
    return best

def greedy_playout(grid, my_head, opponent_head, my_color, opponent_color):
    g = copy_grid(grid)
    r, b = my_head, opponent_head
    while True:
        mr, mb = greedy_move(g, r), greedy_move(g, b)
        moved = False
        if mr:
            r = move_position(r, mr)
            g[r[1]][r[0]] = my_color
            moved = True
        if mb:
            b = move_position(b, mb)
            g[b[1]][b[0]] = opponent_color
            moved = True
        if not moved:
            break
    ra, ba = flood_fill(g, r), flood_fill(g, b)
    return ENDGAME_COMPONENT_MULTIPLIER * (len(ra) - len(ba))


def exhaustive_endgame(grid, my_head, opponent_head, my_color, opponent_color):
    """Fully explore all remaining moves to find optimal endgame play."""
    empties = count_empty(grid)
    max_depth = empties + 2  # enough depth to play all empty cells
    cache = {}

    def key(g, rh, bh, maximizing):
        return (tuple(tuple(r) for r in g), rh, bh, maximizing)

    def dfs(g, rh, bh, maximizing, depth):
        # If no more depth or both stuck, evaluate final outcome
        r_stuck = all(not valid_move(g, *move_position(rh, dir)) for dir in [(1,0),(-1,0),(0,1),(0,-1)])
        b_stuck = all(not valid_move(g, *move_position(bh, dir)) for dir in [(1,0),(-1,0),(0,1),(0,-1)])
        if depth >= max_depth or (r_stuck and b_stuck):
            return greedy_playout(g, rh, bh, my_color, opponent_color)

        k = key(g, rh, bh, maximizing)
        if k in cache:
            return cache[k]

        if maximizing:
            best = -math.inf
            for dir in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = move_position(rh, dir)
                if not valid_move(g, nx, ny):
                    continue
                new_g = copy_grid(g)
                new_g[ny][nx] = my_color
                best = max(best, dfs(new_g, (nx, ny), bh, False, depth+1))
            cache[k] = best
            return best
        else:
            best = math.inf
            for dir in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = move_position(bh, dir)
                if not valid_move(g, nx, ny):
                    continue
                new_g = copy_grid(g)
                new_g[ny][nx] = opponent_color
                best = min(best, dfs(new_g, rh, (nx, ny), True, depth+1))
            cache[k] = best
            return best

    return dfs(grid, my_head, opponent_head, True, 0)

