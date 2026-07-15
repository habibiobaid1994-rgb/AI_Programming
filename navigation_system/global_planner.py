import heapq
import math
from dataclasses import dataclass

SQRT2 = math.sqrt(2.0)  # diagonal step cost on the grid


@dataclass
class PlanResult:
    path: list
    cost: float
    nodes_expanded: int  # I track this to compare A* against Dijkstra
    success: bool


def _to_cell(point):
    # World coordinates are floats, but the search works on integer grid cells.
    return (int(round(point[0])), int(round(point[1])))


def _octile(a, b):
    # Octile distance: the exact cost of moving on an empty 8-connected grid.
    # It never overestimates, so A* stays optimal (admissible heuristic).
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


def _neighbours(col, row, grid, height, width):
    # 8-connected neighbours. Straight moves cost 1, diagonals cost sqrt(2).
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]
    for dc, dr, cost in steps:
        nc, nr = col + dc, row + dr
        if not (0 <= nc < width and 0 <= nr < height):
            continue
        if grid[nr, nc] == 1:
            continue
        # Don't let the robot cut diagonally through the corner of a wall,
        # otherwise the path would clip obstacles it should go around.
        if dc != 0 and dr != 0:
            if grid[row, nc] == 1 or grid[nr, col] == 1:
                continue
        yield nc, nr, cost


def plan_path(env, start, goal, method="astar"):
    # A* and Dijkstra share the exact same code. The only difference is whether
    # we add the heuristic to the priority, so I implemented them together.
    grid = env.grid
    height, width = grid.shape

    start_cell = _to_cell(start)
    goal_cell = _to_cell(goal)

    # If either end sits on a wall there is nothing to plan.
    if grid[start_cell[1], start_cell[0]] == 1 or grid[goal_cell[1], goal_cell[0]] == 1:
        return PlanResult([], math.inf, 0, False)

    use_heuristic = method == "astar"

    g_score = {start_cell: 0.0}   # best known cost from the start to each cell
    came_from = {}                # to rebuild the path at the end
    counter = 0                   # tie-breaker so the heap never compares cells
    open_heap = [(0.0, 0, start_cell)]
    closed = set()
    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        # A cell can end up in the heap more than once; skip old copies.
        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal_cell:
            return PlanResult(_reconstruct(came_from, current),
                              g_score[current], nodes_expanded, True)

        col, row = current
        for nc, nr, step_cost in _neighbours(col, row, grid, height, width):
            neighbour = (nc, nr)
            if neighbour in closed:
                continue
            tentative = g_score[current] + step_cost
            # Only keep this neighbour if we found a cheaper way to reach it.
            if tentative < g_score.get(neighbour, math.inf):
                g_score[neighbour] = tentative
                came_from[neighbour] = current
                if use_heuristic:
                    # A*: cost so far plus estimated cost to the goal.
                    priority = tentative + _octile(neighbour, goal_cell)
                else:
                    # Dijkstra: just the cost so far, so it spreads evenly.
                    priority = tentative
                counter += 1
                heapq.heappush(open_heap, (priority, counter, neighbour))

    # Frontier ran out without reaching the goal -> no path exists.
    return PlanResult([], math.inf, nodes_expanded, False)


def _reconstruct(came_from, current):
    # Walk the came_from links backwards from the goal to the start.
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return [(float(c), float(r)) for c, r in path]
