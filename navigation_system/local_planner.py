import math
from dataclasses import dataclass

from .robot import motion


# The five weights of my multi-objective score (extension option D). Each one
# says how much a different goal matters when choosing a motion.
@dataclass
class DWAWeights:
    alpha: float = 1.0     # heading towards the goal
    beta: float = 1.0      # distance to the nearest obstacle
    gamma: float = 1.0     # forward speed (progress)
    delta: float = 1.0     # average clearance (safety)
    epsilon: float = 1.0   # smoothness (comfort / energy)


# Four "personalities". Changing the weights changes how the robot drives,
# which is the whole point of the weight study in the report.
PRESETS = {
    "balanced": DWAWeights(alpha=1.0, beta=1.0, gamma=0.7, delta=0.6, epsilon=0.4),
    "aggressive": DWAWeights(alpha=1.4, beta=0.5, gamma=1.4, delta=0.2, epsilon=0.2),
    "cautious": DWAWeights(alpha=0.8, beta=1.6, gamma=0.4, delta=1.5, epsilon=0.6),
    "smooth": DWAWeights(alpha=1.1, beta=0.9, gamma=0.7, delta=0.5, epsilon=0.9),
}


# One (v, omega) motion option together with its rolled-out path and the raw
# value of each objective. The final score gets filled in later by _score.
class Candidate:
    def __init__(self, v, omega, trajectory, heading, obstacle, velocity, clearance, smoothness):
        self.v = v
        self.omega = omega
        self.trajectory = trajectory
        self.heading = heading
        self.obstacle = obstacle
        self.velocity = velocity
        self.clearance = clearance
        self.smoothness = smoothness
        self.score = 0.0


class DWAPlanner:
    def __init__(self, config, weights=None, safe_distance=3.0):
        self.config = config
        self.weights = weights if weights is not None else PRESETS["balanced"]
        self.safe_distance = safe_distance

    def _dynamic_window(self, state):
        # The "dynamic window" is the set of speeds and turn rates the robot can
        # actually reach in the next step, given its current velocity and its
        # acceleration limits. This is what keeps every choice physically feasible.
        c = self.config
        v_min = max(c.min_speed, state.v - c.max_accel * c.dt)
        v_max = min(c.max_speed, state.v + c.max_accel * c.dt)
        w_min = max(-c.max_omega, state.omega - c.max_domega * c.dt)
        w_max = min(c.max_omega, state.omega + c.max_domega * c.dt)
        return v_min, v_max, w_min, w_max

    def _frange(self, start, stop, step):
        # Like range() but for floats, and it always includes the end value so
        # the fastest reachable speed is never accidentally skipped.
        values = []
        n = int((stop - start) / step) + 1
        for i in range(n + 1):
            value = start + i * step
            if value > stop + 1e-9:
                break
            values.append(value)
        if not values or abs(values[-1] - stop) > 1e-9:
            values.append(stop)
        return values

    def _rollout(self, state, v, omega, env):
        # Simulate holding this (v, omega) for a short time and record where the
        # robot would go. If any point on the way hits something, the whole
        # option is thrown away (return None). I also track the closest and the
        # average clearance so the score can reward safe trajectories.
        c = self.config
        steps = max(1, int(c.predict_time / c.dt))
        points = []
        min_clear = math.inf
        clear_sum = 0.0
        current = state.copy()
        for _ in range(steps):
            current = motion(current, v, omega, c.dt)
            static_clear = env.nearest_obstacle_distance(current.x, current.y)
            dyn_clear = env.dynamic_clearance(current.x, current.y)
            # Clip at safe_distance: once we are far enough away, extra distance
            # should not matter, otherwise the robot just parks in open space.
            clear = min(static_clear, dyn_clear, self.safe_distance)
            if env.is_collision(current.x, current.y, c.radius):
                return None, None, None, None
            min_clear = min(min_clear, clear)
            clear_sum += clear
            points.append((current.x, current.y))
        avg_clear = clear_sum / steps
        return points, min_clear, avg_clear, current.theta

    def plan(self, state, goal, env):
        # Try every (v, omega) inside the dynamic window, keep the ones that do
        # not crash, score them, and drive the best one.
        c = self.config
        v_min, v_max, w_min, w_max = self._dynamic_window(state)

        candidates = []
        for v in self._frange(v_min, v_max, c.v_resolution):
            for omega in self._frange(w_min, w_max, c.omega_resolution):
                points, min_clear, avg_clear, final_theta = self._rollout(
                    state, v, omega, env)
                if points is None:
                    continue
                # Heading term: how well the robot ends up pointing at the goal.
                # I wrap the angle error to [-pi, pi] so turning either way is fair.
                end = points[-1]
                bearing = math.atan2(goal[1] - end[1], goal[0] - end[0])
                error = math.atan2(math.sin(bearing - final_theta),
                                   math.cos(bearing - final_theta))
                heading = math.pi - abs(error)
                velocity = v
                # Smoothness: punish sharp turns and sudden changes in turn rate.
                smoothness = -(abs(omega) + abs(omega - state.omega))
                candidates.append(
                    Candidate(v, omega, points, heading, min_clear,
                              velocity, avg_clear, smoothness))

        if not candidates:
            return None  # everything crashes -> caller will try to replan

        self._score(candidates)
        # Prefer options that actually move; only stand still if there is truly
        # nothing else, which stops the robot from freezing to play it safe.
        moving = [cand for cand in candidates if cand.v > 1e-3]
        pool = moving if moving else candidates
        best = max(pool, key=lambda cand: cand.score)
        trajectories = [cand.trajectory for cand in candidates]
        return best, trajectories

    def _score(self, candidates):
        # Weighted sum of the five objectives. Everything is normalised to
        # [0, 1] first (see below) so the weights are directly comparable.
        w = self.weights
        headings = _normalize([cand.heading for cand in candidates])
        obstacles = _normalize([cand.obstacle for cand in candidates])
        velocities = _normalize([cand.velocity for cand in candidates])
        clearances = _normalize([cand.clearance for cand in candidates])
        smoothness = _normalize([cand.smoothness for cand in candidates])

        for i, cand in enumerate(candidates):
            cand.score = (w.alpha * headings[i]
                          + w.beta * obstacles[i]
                          + w.gamma * velocities[i]
                          + w.delta * clearances[i]
                          + w.epsilon * smoothness[i])


def _normalize(values):
    # Rescale a list of raw scores to [0, 1]. Without this, a term with a big
    # numeric range would dominate no matter what its weight was.
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [0.5 for _ in values]
    low = min(finite)
    high = max(finite)
    if high - low < 1e-9:
        return [1.0 for _ in values]
    result = []
    for v in values:
        if not math.isfinite(v):
            v = high
        result.append((v - low) / (high - low))
    return result
