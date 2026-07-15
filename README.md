# Intelligent Autonomous Navigation System

A robot navigation system that plans a global route with **A\*/Dijkstra** and
drives it in real time with a **Dynamic Window Approach (DWA)** local planner,
dodging moving obstacles and replanning when blocked. The DWA score is extended
into a **multi-objective** optimiser (heading, obstacle distance, velocity,
clearance, smoothness) whose weights create different driving personalities.

AI Programming — Summer Semester 2026.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python main.py                                  # live demo, dynamic map
python main.py --scenario trap                  # trap (dead end) demo
python main.py --scenario maze --global dijkstra
python main.py --scenario dynamic --preset cautious
python main.py --experiments                    # print the result tables
python -m navigation_system.figures             # regenerate the figures/
```

### Options

- `--scenario {simple, maze, trap, dynamic}`
- `--global {astar, dijkstra}`
- `--preset {balanced, aggressive, cautious, smooth}`
- `--save path.gif` — save the animation instead of showing it
- `--experiments` — run the experiments and print the tables

On screen: **cyan** = candidate moves, **magenta** = chosen move, **orange** =
path travelled, **blue dashed** = the global route, **red** = moving obstacles.

## Project layout

```
main.py                     launcher / CLI
navigation_system/
    environment.py          occupancy grid, obstacles, distance transform, scenarios
    robot.py                unicycle motion model and limits
    global_planner.py       A* and Dijkstra from scratch
    local_planner.py        Dynamic Window Approach + multi-objective score
    navigator.py            global + local control loop and replanning
    visualization.py        live matplotlib animation
    experiments.py          A* vs Dijkstra and weight-study tables
    figures.py              saves the PNG figures
figures/                    generated plots (scenarios, planner comparison,
                            weight study, distance field, DWA candidate fan,
                            preset paths, trade-off scatter, architecture)

build_docs.py               turns the .md documents into .html
```





This is a planning/optimization project, not a machine-learning one, so there is
no CSV dataset. The input is the environment: a grid map (each cell free or
blocked), a start point, a goal point, and a list of moving obstacles (each with
a position, a velocity, and a radius). The output is a path.
