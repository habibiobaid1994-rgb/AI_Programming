import math
from dataclasses import dataclass, field

from .global_planner import plan_path
from .local_planner import DWAPlanner, PRESETS
from .robot import RobotState, RobotConfig, motion


# Everything I want to measure about a single run ends up here. This is what
# the experiments read to build the result tables.
@dataclass
class NavigationLog:
    success: bool = False
    steps: int = 0
    path_length: float = 0.0
    replans: int = 0
    reason: str = ""          # why the run ended (goal reached, blocked, ...)
    nodes_expanded: int = 0
    trajectory: list = field(default_factory=list)
    # Clearance = distance to the nearest obstacle (wall or moving one) recorded
    # at every step. I keep the worst (min) and the average so I can actually
    # measure how safe each preset drives, not just claim it.
    min_clearance: float = math.inf
    clearance_sum: float = 0.0

    @property
    def avg_clearance(self):
        if self.steps == 0:
            return 0.0
        return self.clearance_sum / self.steps


# The Navigator glues the two layers together: the global planner gives a route,
# the local planner drives along it, and we replan globally if we get stuck.
class Navigator:
    def __init__(self, env, start, goal, config=None, weights=None,
                 global_method="astar", carrot_lookahead=4.0,
                 goal_tolerance=1.2, max_replans=40, stall_limit=60):
        self.env = env
        self.start = start
        self.goal = goal
        self.config = config if config is not None else RobotConfig()
        self.weights = weights if weights is not None else PRESETS["balanced"]
        self.global_method = global_method
        self.carrot_lookahead = carrot_lookahead
        self.goal_tolerance = goal_tolerance
        self.max_replans = max_replans
        self.stall_limit = stall_limit

        self.planner = DWAPlanner(self.config, self.weights)
        self.state = None
        self.global_path = []
        self.carrot_index = 0
        self.log = NavigationLog()
        self.last_trajectories = []
        self.chosen_trajectory = []
        self.finished = False

    def reset(self):
        heading = math.atan2(self.goal[1] - self.start[1],
                             self.goal[0] - self.start[0])
        self.state = RobotState(float(self.start[0]), float(self.start[1]), heading)
        result = plan_path(self.env, self.start, self.goal, self.global_method)
        self.global_path = result.path
        self.log = NavigationLog(nodes_expanded=result.nodes_expanded)
        self.log.trajectory.append((self.state.x, self.state.y))
        self.carrot_index = 0
        self.finished = False
        self.best_goal_dist = math.hypot(self.goal[0] - self.state.x,
                                         self.goal[1] - self.state.y)
        self.stall_steps = 0
        if not result.success:
            self.finished = True
            self.log.reason = "no global path"
        return result

    def _carrot(self):
        # "Carrot on a stick": pick a point a little way ahead on the global
        # path and let the local planner chase it. This is how the reactive
        # DWA layer ends up following the long-term global route.
        if not self.global_path:
            return self.goal

        # First find the point on the path we are currently closest to, so we
        # don't aim backwards at bits of the route we already passed.
        px, py = self.state.x, self.state.y
        best_index = self.carrot_index
        best_dist = math.inf
        for i in range(self.carrot_index, len(self.global_path)):
            gx, gy = self.global_path[i]
            d = math.hypot(gx - px, gy - py)
            if d < best_dist:
                best_dist = d
                best_index = i
        self.carrot_index = best_index

        # Then walk forward until we are lookahead cells ahead and aim there.
        target_index = best_index
        for i in range(best_index, len(self.global_path)):
            gx, gy = self.global_path[i]
            if math.hypot(gx - px, gy - py) >= self.carrot_lookahead:
                target_index = i
                break
            target_index = i
        return self.global_path[target_index]

    def _replan(self):
        result = plan_path(self.env, (self.state.x, self.state.y),
                           self.goal, self.global_method)
        self.log.nodes_expanded += result.nodes_expanded
        if result.success:
            self.global_path = result.path
            self.carrot_index = 0
            self.log.replans += 1
            return True
        return False

    def step(self):
        # One control tick. Returns False once the run is over.
        if self.finished:
            return False

        # Close enough to the goal? Then we are done.
        dist_to_goal = math.hypot(self.goal[0] - self.state.x,
                                  self.goal[1] - self.state.y)
        if dist_to_goal <= self.goal_tolerance:
            self.finished = True
            self.log.success = True
            self.log.reason = "goal reached"
            return False

        # Ask the local planner for the best feasible motion toward the carrot.
        carrot = self._carrot()
        outcome = self.planner.plan(self.state, carrot, self.env)

        # No feasible motion means we are boxed in: try a global replan once.
        if outcome is None:
            if self.log.replans >= self.max_replans or not self._replan():
                self.finished = True
                self.log.reason = "blocked"
                return False
            carrot = self._carrot()
            outcome = self.planner.plan(self.state, carrot, self.env)
            if outcome is None:
                self.finished = True
                self.log.reason = "blocked"
                return False

        best, trajectories = outcome
        self.last_trajectories = trajectories      # kept for the animation
        self.chosen_trajectory = best.trajectory

        # Apply the chosen motion and let the world (moving obstacles) advance.
        prev = (self.state.x, self.state.y)
        self.state = motion(self.state, best.v, best.omega, self.config.dt)
        self.env.move_dynamic_obstacles(self.config.dt)

        self.log.steps += 1
        self.log.path_length += math.hypot(self.state.x - prev[0],
                                           self.state.y - prev[1])
        self.log.trajectory.append((self.state.x, self.state.y))

        # Record how much room the robot actually had this step.
        clearance = min(self.env.nearest_obstacle_distance(self.state.x, self.state.y),
                        self.env.dynamic_clearance(self.state.x, self.state.y))
        self.log.min_clearance = min(self.log.min_clearance, clearance)
        self.log.clearance_sum += clearance

        if self.env.is_collision(self.state.x, self.state.y, self.config.radius * 0.5):
            self.finished = True
            self.log.reason = "collision"
            return False

        # Stall detection: if we go a long time without getting meaningfully
        # closer to the goal, we are probably circling or stuck, so replan.
        current_dist = math.hypot(self.goal[0] - self.state.x,
                                  self.goal[1] - self.state.y)
        if current_dist < self.best_goal_dist - 0.5:
            self.best_goal_dist = current_dist
            self.stall_steps = 0
        else:
            self.stall_steps += 1

        if self.stall_steps >= self.stall_limit:
            self.stall_steps = 0
            if self.log.replans >= self.max_replans or not self._replan():
                self.finished = True
                self.log.reason = "stuck"
                return False

        return True

    def run(self, max_steps=2000):
        self.reset()
        while not self.finished and self.log.steps < max_steps:
            self.step()
        if not self.finished and self.log.steps >= max_steps:
            self.log.reason = "timeout"
        return self.log
