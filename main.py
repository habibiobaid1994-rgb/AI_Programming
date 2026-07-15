import argparse

from navigation_system.environment import make_scenario, SCENARIOS
from navigation_system.local_planner import PRESETS
from navigation_system.navigator import Navigator
from navigation_system.robot import RobotConfig

START = (5, 5)
GOAL = (44, 44)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intelligent Autonomous Navigation System")
    parser.add_argument("--scenario", choices=SCENARIOS, default="dynamic")
    parser.add_argument("--global", dest="global_method",
                        choices=["astar", "dijkstra"], default="astar")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="balanced")
    parser.add_argument("--save", default=None,
                        help="save the animation to this gif path instead of showing it")
    parser.add_argument("--experiments", action="store_true",
                        help="run the experiments and print the result tables")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiments:
        from navigation_system.experiments import run_all
        run_all()
        return

    env = make_scenario(args.scenario)
    navigator = Navigator(env, START, GOAL, config=RobotConfig(),
                          weights=PRESETS[args.preset],
                          global_method=args.global_method)

    from navigation_system.visualization import run_visualization
    title = "%s (%s, %s)" % (args.scenario, args.global_method, args.preset)
    log = run_visualization(navigator, title=title, save_path=args.save)

    print("result: %s" % log.reason)
    print("steps: %d" % log.steps)
    print("path length: %.1f" % log.path_length)
    print("replans: %d" % log.replans)


if __name__ == "__main__":
    main()
