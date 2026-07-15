from .environment import make_scenario, SCENARIOS
from .global_planner import plan_path
from .local_planner import PRESETS
from .navigator import Navigator
from .robot import RobotConfig

START = (5, 5)
GOAL = (44, 44)


def planner_comparison():
    # Experiment 1: run both planners on every map and count how many nodes
    # each one expands. They return the same cost, but A* should expand fewer.
    rows = []
    for name in SCENARIOS:
        env = make_scenario(name)
        dij = plan_path(env, START, GOAL, "dijkstra")
        ast = plan_path(env, START, GOAL, "astar")
        rows.append({
            "scenario": name,
            "dijkstra_nodes": dij.nodes_expanded,
            "astar_nodes": ast.nodes_expanded,
            "cost": round(ast.cost, 2),
        })
    return rows


def weight_study(scenario="dynamic", presets=None):
    # Experiment 2: run each weight preset on the same map and compare how they
    # drive (steps, distance, replans). Shows the speed/safety/smoothness trade-off.
    if presets is None:
        presets = ["aggressive", "balanced", "smooth", "cautious"]
    rows = []
    for preset in presets:
        env = make_scenario(scenario)
        nav = Navigator(env, START, GOAL, config=RobotConfig(),
                        weights=PRESETS[preset], global_method="astar")
        log = nav.run()
        rows.append({
            "preset": preset,
            "steps": log.steps,
            "path_length": round(log.path_length, 1),
            "replans": log.replans,
            "min_clear": round(log.min_clearance, 2),
            "avg_clear": round(log.avg_clearance, 2),
            "success": log.success,
        })
    return rows


def _print_table(title, rows, columns):
    print("\n" + title)
    print("-" * len(title))
    header = "  ".join(str(c).ljust(14) for c in columns)
    print(header)
    for row in rows:
        line = "  ".join(str(row[c]).ljust(14) for c in columns)
        print(line)


def run_all():
    comp = planner_comparison()
    _print_table("Experiment 1 - A* vs Dijkstra (nodes expanded)", comp,
                 ["scenario", "dijkstra_nodes", "astar_nodes", "cost"])

    study = weight_study("dynamic")
    _print_table("Experiment 2 - Weight study on the dynamic map", study,
                 ["preset", "steps", "path_length", "replans",
                  "min_clear", "avg_clear", "success"])


if __name__ == "__main__":
    run_all()
