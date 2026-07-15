import math
from dataclasses import dataclass

import numpy as np


# A moving obstacle: a circle that drifts with a constant velocity and bounces
# off the arena walls. These are NOT stored in the grid, so the global planner
# cannot see them - only the local planner reacts to them at run time.
@dataclass
class DynamicObstacle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float

    def step(self, dt, width, height):
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Bounce off the left/right and top/bottom borders.
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx
        elif self.x + self.radius > width:
            self.x = width - self.radius
            self.vx = -self.vx

        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy
        elif self.y + self.radius > height:
            self.y = height - self.radius
            self.vy = -self.vy


class Environment:
    # The world is a 2D occupancy grid: grid[row, col] == 1 means "wall".
    # World (x, y) maps to (col, row) so the same map is used by both the
    # discrete global search and the continuous local motion.
    def __init__(self, width=50, height=50):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.dynamic_obstacles = []
        self._distance = None  # cached distance transform, built lazily

    def add_border(self):
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        self._distance = None

    def add_rectangle(self, x0, y0, x1, y1):
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(self.width, int(x1))
        y1 = min(self.height, int(y1))
        self.grid[y0:y1, x0:x1] = 1
        self._distance = None

    def add_dynamic_obstacle(self, x, y, vx, vy, radius):
        self.dynamic_obstacles.append(DynamicObstacle(x, y, vx, vy, radius))

    def in_bounds(self, col, row):
        return 0 <= col < self.width and 0 <= row < self.height

    def is_free(self, col, row):
        return self.in_bounds(col, row) and self.grid[row, col] == 0

    def move_dynamic_obstacles(self, dt):
        for ob in self.dynamic_obstacles:
            ob.step(dt, self.width, self.height)

    @property
    def distance_field(self):
        # Build it once on first use, then reuse the cached array.
        if self._distance is None:
            self._distance = self._compute_distance_transform()
        return self._distance

    def _compute_distance_transform(self):
        # The local planner needs the TRUE (Euclidean) distance to the nearest
        # wall thousands of times, so I precompute it once and cache it. After
        # this, a distance query is just an O(1) array lookup.
        #
        # I first tried a grid (BFS) distance where a diagonal step counts as 1,
        # but that under-measures the real distance (it under-counts diagonal
        # gaps). So I compute the real straight-line distance from every free cell
        # to the closest occupied cell instead, which is what the local planner
        # scores as clearance. The map is small and this runs only once.
        occupied = np.argwhere(self.grid == 1)  # (row, col) of every wall cell

        dist = np.full((self.height, self.width), np.inf)
        if occupied.size == 0:
            return dist

        obst_rows = occupied[:, 0]
        obst_cols = occupied[:, 1]

        # For each free cell, distance to the nearest wall cell centre.
        for row in range(self.height):
            for col in range(self.width):
                if self.grid[row, col] == 1:
                    dist[row, col] = 0.0
                    continue
                dr = obst_rows - row
                dc = obst_cols - col
                dist[row, col] = float(np.sqrt(np.min(dr * dr + dc * dc)))

        return dist

    def nearest_obstacle_distance(self, x, y):
        col = int(round(x))
        row = int(round(y))
        if not self.in_bounds(col, row):
            return 0.0
        return float(self.distance_field[row, col])

    def dynamic_clearance(self, x, y):
        # Distance from a point to the closest moving obstacle (edge, not centre).
        if not self.dynamic_obstacles:
            return math.inf
        best = math.inf
        for ob in self.dynamic_obstacles:
            d = math.hypot(x - ob.x, y - ob.y) - ob.radius
            if d < best:
                best = d
        return best

    def is_collision(self, x, y, robot_radius=0.0):
        # True if the robot would overlap a wall, leave the map, or touch a
        # moving obstacle. Used both while planning and after each real move.
        col = int(round(x))
        row = int(round(y))
        if not self.in_bounds(col, row):
            return True
        if self.grid[row, col] == 1:
            return True
        if self.nearest_obstacle_distance(x, y) < robot_radius:
            return True
        for ob in self.dynamic_obstacles:
            if math.hypot(x - ob.x, y - ob.y) < ob.radius + robot_radius:
                return True
        return False


SCENARIOS = ["simple", "maze", "trap", "dynamic"]


# Each scenario is designed to stress a different part of the system:
#   simple  - baseline that should just work
#   maze    - narrow corridors where planning ahead matters
#   trap    - a U-shaped dead end that breaks local-only planners
#   dynamic - moving obstacles that force real-time reaction
def make_scenario(name, width=50, height=50):
    env = Environment(width, height)
    env.add_border()

    if name == "simple":
        env.add_rectangle(15, 10, 22, 30)
        env.add_rectangle(30, 22, 38, 40)

    elif name == "maze":
        env.add_rectangle(12, 0, 15, 38)
        env.add_rectangle(24, 12, 27, 50)
        env.add_rectangle(36, 0, 39, 38)

    elif name == "trap":
        env.add_rectangle(20, 18, 24, 42)
        env.add_rectangle(36, 18, 40, 42)
        env.add_rectangle(20, 38, 40, 42)

    elif name == "dynamic":
        env.add_rectangle(15, 10, 22, 28)
        env.add_rectangle(30, 24, 38, 40)
        env.add_dynamic_obstacle(26, 16, 0.0, 1.2, 1.5)
        env.add_dynamic_obstacle(28, 33, -1.1, 0.0, 1.5)
        env.add_dynamic_obstacle(24, 42, 1.0, -0.8, 1.2)

    else:
        raise ValueError("unknown scenario: " + str(name))

    return env
