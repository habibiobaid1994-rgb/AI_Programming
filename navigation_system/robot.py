import math
from dataclasses import dataclass


# The robot is modelled as a unicycle: it has a position, a heading, and it can
# only drive forward and turn. This class just holds those numbers together.
@dataclass
class RobotState:
    x: float
    y: float
    theta: float          # heading angle in radians
    v: float = 0.0        # current forward speed
    omega: float = 0.0    # current turn rate

    def copy(self):
        # I copy the state a lot when rolling out trajectories, so having a
        # helper keeps the DWA code readable.
        return RobotState(self.x, self.y, self.theta, self.v, self.omega)


# All the physical limits live here so they are easy to tweak in one place.
# These are what make the problem hard: the robot cannot jump to any speed.
@dataclass
class RobotConfig:
    max_speed: float = 3.0
    min_speed: float = 0.0
    max_omega: float = 2.5          # fastest it can turn
    max_accel: float = 2.0          # how fast speed can change per second
    max_domega: float = 3.5         # how fast the turn rate can change
    radius: float = 0.6             # robot size, used for collision checks
    dt: float = 0.2                 # control time step
    v_resolution: float = 0.1       # how finely we sample speeds in DWA
    omega_resolution: float = 0.15  # how finely we sample turn rates in DWA
    predict_time: float = 1.5       # how far ahead each rollout looks


def motion(state, v, omega, dt):
    # Move the robot one step using the unicycle equations. I update the
    # heading first and then use the new heading for the position update,
    # which behaves a bit better over a discrete step.
    new = state.copy()
    new.theta = state.theta + omega * dt
    new.x = state.x + v * math.cos(new.theta) * dt
    new.y = state.y + v * math.sin(new.theta) * dt
    new.v = v
    new.omega = omega
    return new
