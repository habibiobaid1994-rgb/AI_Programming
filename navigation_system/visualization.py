import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


# Live animation of a run. This is what I show in the demo. It redraws the map,
# the global route, the DWA candidate fan, the chosen move and the robot each frame.
def _draw_static(ax, env, navigator):
    ax.clear()
    ax.imshow(env.grid, cmap="Greys", origin="lower",
              extent=[0, env.width, 0, env.height], vmin=0, vmax=1)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")

    if navigator.global_path:
        xs = [p[0] for p in navigator.global_path]
        ys = [p[1] for p in navigator.global_path]
        ax.plot(xs, ys, "--", color="royalblue", linewidth=1.5, label="global route")

    ax.plot(navigator.start[0], navigator.start[1], "o", color="green", markersize=9)
    ax.plot(navigator.goal[0], navigator.goal[1], "*", color="gold",
            markersize=16, markeredgecolor="black")


def run_visualization(navigator, title="Navigation", max_steps=2000, interval=40, save_path=None):
    if save_path:
        plt.switch_backend("Agg")
    navigator.reset()
    env = navigator.env

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(_frame):
        alive = navigator.step()
        _draw_static(ax, env, navigator)

        for traj in navigator.last_trajectories:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, color="cyan", linewidth=0.4, alpha=0.5)

        if navigator.chosen_trajectory:
            xs = [p[0] for p in navigator.chosen_trajectory]
            ys = [p[1] for p in navigator.chosen_trajectory]
            ax.plot(xs, ys, color="magenta", linewidth=2.0)

        travelled = navigator.log.trajectory
        if len(travelled) > 1:
            xs = [p[0] for p in travelled]
            ys = [p[1] for p in travelled]
            ax.plot(xs, ys, color="orange", linewidth=2.0)

        for ob in env.dynamic_obstacles:
            circle = patches.Circle((ob.x, ob.y), ob.radius, color="red", alpha=0.7)
            ax.add_patch(circle)

        state = navigator.state
        robot = patches.Circle((state.x, state.y), navigator.config.radius,
                               color="blue")
        ax.add_patch(robot)
        import math
        ax.plot([state.x, state.x + math.cos(state.theta) * 1.5],
                [state.y, state.y + math.sin(state.theta) * 1.5],
                color="white", linewidth=1.5)

        status = navigator.log.reason if navigator.finished else "running"
        ax.set_title("%s  |  steps=%d  replans=%d  %s"
                     % (title, navigator.log.steps, navigator.log.replans, status))
        ax.legend(loc="upper left", fontsize=8)

        return []

    def frames():
        count = 0
        while not navigator.finished and count < max_steps:
            yield count
            count += 1

    anim = FuncAnimation(fig, update, frames=frames, interval=interval,
                         blit=False, repeat=False, save_count=max_steps,
                         cache_frame_data=False)

    if save_path:
        anim.save(save_path, writer="pillow", fps=20)
        plt.close(fig)
    else:
        plt.show()

    return navigator.log
