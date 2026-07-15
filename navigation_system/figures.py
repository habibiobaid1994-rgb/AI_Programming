import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .environment import make_scenario, SCENARIOS
from .global_planner import plan_path
from .navigator import Navigator
from .robot import RobotConfig, RobotState
from .local_planner import PRESETS, DWAPlanner
from .experiments import planner_comparison, weight_study, START, GOAL

FIGURE_DIR = "figures"


def _ensure_dir():
    if not os.path.isdir(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)


def draw_scenario(name):
    draw_env = make_scenario(name)
    result = plan_path(draw_env, START, GOAL, "astar")

    run_env = make_scenario(name)
    nav = Navigator(run_env, START, GOAL, config=RobotConfig(),
                    weights=PRESETS["balanced"], global_method="astar")
    log = nav.run()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(draw_env.grid, cmap="Greys", origin="lower",
              extent=[0, draw_env.width, 0, draw_env.height], vmin=0, vmax=1)

    if result.success:
        xs = [p[0] for p in result.path]
        ys = [p[1] for p in result.path]
        ax.plot(xs, ys, "--", color="royalblue", linewidth=2, label="A* route")

    if len(log.trajectory) > 1:
        tx = [p[0] for p in log.trajectory]
        ty = [p[1] for p in log.trajectory]
        ax.plot(tx, ty, "-", color="darkorange", linewidth=2.2,
                label="robot path")

    for ob in draw_env.dynamic_obstacles:
        ax.add_patch(patches.Circle((ob.x, ob.y), ob.radius, color="red",
                                    alpha=0.65, label="obstacle (start)"))
    for ob in run_env.dynamic_obstacles:
        ax.add_patch(patches.Circle((ob.x, ob.y), ob.radius, color="red",
                                    alpha=0.2))

    ax.plot(START[0], START[1], "o", color="green", markersize=10, label="start")
    ax.plot(GOAL[0], GOAL[1], "*", color="gold", markersize=16,
            markeredgecolor="black", label="goal")
    ax.set_xlim(0, draw_env.width)
    ax.set_ylim(0, draw_env.height)
    ax.set_aspect("equal")
    ax.set_title("Scenario: %s  (%d steps, %d replans)"
                 % (name, log.steps, log.replans))
    _dedup_legend(ax)

    path = os.path.join(FIGURE_DIR, "scenario_" + name + ".png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _dedup_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = handle
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)


def draw_planner_comparison():
    rows = planner_comparison()
    names = [r["scenario"] for r in rows]
    dij = [r["dijkstra_nodes"] for r in rows]
    ast = [r["astar_nodes"] for r in rows]

    x = range(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width / 2 for i in x], dij, width, label="Dijkstra", color="#8888cc")
    ax.bar([i + width / 2 for i in x], ast, width, label="A*", color="#cc7755")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Nodes expanded")
    ax.set_title("Global planner efficiency: A* vs Dijkstra")
    ax.legend()

    path = os.path.join(FIGURE_DIR, "planner_comparison.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_weight_study():
    rows = weight_study("dynamic")
    presets = [r["preset"] for r in rows]
    steps = [r["steps"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(presets, steps, color=["#cc7755", "#77aa77", "#7799cc", "#bb77bb"])
    ax.set_ylabel("Steps to goal")
    ax.set_title("Weight presets on the dynamic map")

    path = os.path.join(FIGURE_DIR, "weight_study.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_distance_field():
    # Show the precomputed distance transform as a heatmap. This is the
    # representation the local planner leans on for fast clearance queries.
    env = make_scenario("maze")
    field = env.distance_field.copy()
    field[env.grid == 1] = 0  # draw walls as the darkest colour

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(field, cmap="viridis", origin="lower",
                   extent=[0, env.width, 0, env.height])
    fig.colorbar(im, ax=ax, shrink=0.8, label="cells to nearest wall")
    ax.set_title("Distance transform (maze map)")
    ax.set_aspect("equal")

    path = os.path.join(FIGURE_DIR, "distance_field.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_dwa_candidates():
    # Freeze one moment in time and draw the whole "fan" of candidate motions
    # the DWA evaluates, with the chosen one highlighted. Helps explain how the
    # local planner actually picks a move.
    env = make_scenario("simple")
    # Longer horizon and a bit more sampling just for this snapshot, so the fan
    # of possible trajectories is wide enough to actually see.
    config = RobotConfig(predict_time=3.0, v_resolution=0.15, omega_resolution=0.1)
    planner = DWAPlanner(config, PRESETS["balanced"])

    # Robot already moving so its dynamic window (reachable motions) is wide.
    state = RobotState(10.0, 8.0, 0.9, v=2.5)
    outcome = planner.plan(state, GOAL, env)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(env.grid, cmap="Greys", origin="lower",
              extent=[0, env.width, 0, env.height], vmin=0, vmax=1)

    if outcome is not None:
        best, trajectories = outcome
        for i, traj in enumerate(trajectories):
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            label = "candidate motions" if i == 0 else None
            ax.plot(xs, ys, color="deepskyblue", linewidth=0.8, alpha=0.7, label=label)
        bx = [p[0] for p in best.trajectory]
        by = [p[1] for p in best.trajectory]
        ax.plot(bx, by, color="magenta", linewidth=2.6, label="chosen motion")

    ax.plot(state.x, state.y, "o", color="blue", markersize=9, label="robot")
    ax.set_xlim(4, 26)
    ax.set_ylim(4, 26)
    ax.set_aspect("equal")
    ax.set_title("DWA dynamic window: candidate motions")
    _dedup_legend(ax)

    path = os.path.join(FIGURE_DIR, "dwa_candidates.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_preset_paths():
    # Overlay the paths driven by every preset on the same dynamic map so the
    # different "personalities" can be compared side by side.
    colours = {"aggressive": "#d1495b", "balanced": "#2e8b57",
               "smooth": "#3d6bb3", "cautious": "#9b5fb0"}

    base = make_scenario("dynamic")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(base.grid, cmap="Greys", origin="lower",
              extent=[0, base.width, 0, base.height], vmin=0, vmax=1)

    for preset, colour in colours.items():
        env = make_scenario("dynamic")
        nav = Navigator(env, START, GOAL, config=RobotConfig(),
                        weights=PRESETS[preset], global_method="astar")
        log = nav.run()
        xs = [p[0] for p in log.trajectory]
        ys = [p[1] for p in log.trajectory]
        ax.plot(xs, ys, "-", color=colour, linewidth=1.8,
                label="%s (%d steps)" % (preset, log.steps))

    ax.plot(START[0], START[1], "o", color="green", markersize=10, label="start")
    ax.plot(GOAL[0], GOAL[1], "*", color="gold", markersize=16,
            markeredgecolor="black", label="goal")
    ax.set_xlim(0, base.width)
    ax.set_ylim(0, base.height)
    ax.set_aspect("equal")
    ax.set_title("Same map, four presets: different driving styles")
    ax.legend(loc="lower right", fontsize=7)

    path = os.path.join(FIGURE_DIR, "preset_paths.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_clearance():
    # Bar chart of worst-case (min) and average clearance per preset. This is my
    # measured safety evidence: it shows cautious really does keep the largest
    # worst-case margin around the moving obstacles.
    rows = weight_study("dynamic")
    presets = [r["preset"] for r in rows]
    mins = [r["min_clear"] for r in rows]
    avgs = [r["avg_clear"] for r in rows]

    x = range(len(presets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width / 2 for i in x], mins, width, label="worst-case (min)",
           color="#cc5555")
    ax.bar([i + width / 2 for i in x], avgs, width, label="average",
           color="#5599cc")
    ax.set_xticks(list(x))
    ax.set_xticklabels(presets)
    ax.set_ylabel("Clearance (cells to nearest obstacle)")
    ax.set_title("Measured safety per preset (dynamic map)")
    ax.legend()

    path = os.path.join(FIGURE_DIR, "clearance.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_tradeoff():
    # Scatter of path length against steps for each preset. Makes the
    # multi-objective trade-off obvious: no single preset wins everything.
    rows = weight_study("dynamic")
    fig, ax = plt.subplots(figsize=(6, 5))
    for row in rows:
        ax.scatter(row["steps"], row["path_length"], s=120)
        ax.annotate(row["preset"], (row["steps"], row["path_length"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Steps to goal (time)")
    ax.set_ylabel("Path length (distance)")
    ax.set_title("Speed vs distance trade-off per preset")
    ax.grid(True, linestyle=":", alpha=0.5)

    path = os.path.join(FIGURE_DIR, "tradeoff.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_architecture():
    # Simple block diagram of the two-layer architecture. Drawn by hand with
    # matplotlib boxes so it lives in the repo and needs no extra tools.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis("off")

    def box(x, y, w, h, text, colour):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02",
            facecolor=colour, edgecolor="black", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9, wrap=True)

    box(0.05, 0.80, 0.9, 0.13, "Environment\n(occupancy grid + distance transform + obstacles)", "#dbe9f4")
    box(0.05, 0.52, 0.40, 0.15, "Global planner\nA* / Dijkstra\n(once + on replan)", "#f4e2c8")
    box(0.55, 0.52, 0.40, 0.15, "Local planner (DWA)\nmulti-objective score", "#e6f4d8")
    box(0.28, 0.24, 0.44, 0.15, "Navigator\ncarrot-following + replanning", "#f4d8e4")
    box(0.28, 0.03, 0.44, 0.10, "Robot (unicycle motion model)", "#eeeeee")

    # Arrows drawn between the box edges (not through the text).
    arrow = dict(arrowstyle="->", linewidth=1.4, color="black")
    ax.annotate("", xy=(0.25, 0.67), xytext=(0.25, 0.80), arrowprops=arrow)   # env -> global
    ax.annotate("", xy=(0.75, 0.67), xytext=(0.75, 0.80), arrowprops=arrow)   # env -> local
    ax.annotate("", xy=(0.40, 0.39), xytext=(0.28, 0.52), arrowprops=arrow)   # global -> navigator
    ax.annotate("", xy=(0.60, 0.39), xytext=(0.72, 0.52), arrowprops=arrow)   # local -> navigator
    ax.annotate("", xy=(0.50, 0.13), xytext=(0.50, 0.24), arrowprops=arrow)   # navigator -> robot

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("System architecture")

    path = os.path.join(FIGURE_DIR, "architecture.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all():
    _ensure_dir()
    saved = []
    for name in SCENARIOS:
        saved.append(draw_scenario(name))
    saved.append(draw_planner_comparison())
    saved.append(draw_weight_study())
    saved.append(draw_distance_field())
    saved.append(draw_dwa_candidates())
    saved.append(draw_preset_paths())
    saved.append(draw_tradeoff())
    saved.append(draw_clearance())
    saved.append(draw_architecture())
    for path in saved:
        print("saved " + path)
    return saved


if __name__ == "__main__":
    generate_all()
