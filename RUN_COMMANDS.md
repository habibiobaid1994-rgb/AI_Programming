# Run Commands — Every Scenario, Step by Step

This file lists every command I can run for the project, with a short note on
what each one does and what to expect. Copy–paste them one at a time.

---

## 0. One-time setup (do this first, once)

Open a terminal in the project folder (`Final_AI_programming`) and run:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- `python -m venv .venv` creates a private environment for the project.
- `source .venv/bin/activate` turns it on (you'll see `(.venv)` in the prompt).
- `pip install -r requirements.txt` installs numpy, matplotlib, and markdown.

**Every time I open a new terminal, I first run:**

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

---

## 1. The four scenarios (live animation)

Each command opens a window showing the robot driving. Close the window when
done. On screen: **blue dashed** = planned route, **cyan** = candidate moves,
**magenta** = chosen move, **orange** = path travelled, **red** = moving obstacles.

### 1.1 Simple room (basic check)

```bash
python main.py --scenario simple
```

Expected: the robot drives an almost straight route to the goal; planned route
and actual path match closely because there is nothing to dodge.

### 1.2 Maze room (tight corridors)

```bash
python main.py --scenario maze
```

Expected: the robot follows the planned route through narrow corridors. This
shows why planning ahead matters.

### 1.3 Trap room (U-shaped dead end)

```bash
python main.py --scenario trap
```

Expected: the planned route goes **around** the dead end and the robot follows
it. This is the clearest proof that the reactive driver alone would fail (it
would drive into the U and get stuck).

### 1.4 Dynamic room (moving obstacles)

```bash
python main.py --scenario dynamic
```

Expected: red circles move around; the robot swerves and loops to let them pass,
then rejoins its route. This is the real-time dodging in action.

> Tip: running `python main.py` with no options defaults to the **dynamic** room
> with the **balanced** personality and the **A\*** planner.

---

## 2. Switch the global planner (A* vs Dijkstra)

Use `--global` to choose the route-finder. The default is `astar`.

```bash
python main.py --scenario maze --global astar        # A* (default, faster search)
python main.py --scenario maze --global dijkstra     # Dijkstra (the baseline)
```

Both produce the **same route**; the difference (how many squares each examines)
is measured in the experiments (Section 5).

---

## 3. Switch the driving personality (weights)

Use `--preset` to change how the robot drives. The default is `balanced`.

```bash
python main.py --scenario dynamic --preset balanced      # sensible all-rounder
python main.py --scenario dynamic --preset aggressive    # fast & bold
python main.py --scenario dynamic --preset cautious      # careful, big margins
python main.py --scenario dynamic --preset smooth        # gentle, soft turns
```

Watch how the path shape and speed change even though the room is identical.

---

## 4. Mix and match (any room + any planner + any personality)

You can combine all the options. A few useful examples:

```bash
python main.py --scenario trap --global dijkstra --preset cautious
python main.py --scenario dynamic --global astar --preset aggressive
python main.py --scenario maze --global astar --preset smooth
```

---

## 5. Run the experiments (print the result tables)

```bash
python main.py --experiments
```

Expected output: two tables in the terminal —
1. **A\* vs Dijkstra** — squares expanded per room (A* far fewer; same route cost).
2. **Weight study** — steps / distance / re-plans for each personality on the
   dynamic room.

These are the numbers used in my report and study guide.

---

## 6. Save an animation as a GIF (instead of opening a window)

Use `--save` with a filename ending in `.gif`. Useful for putting a clip in the
presentation.

```bash
python main.py --scenario dynamic --save demo_dynamic.gif
python main.py --scenario trap --save demo_trap.gif
```

Expected: no window opens; a `.gif` file is written into the project folder.

---

## 7. Regenerate all the figures (the pictures in figures/)

```bash
python -m navigation_system.figures
```

Expected: it prints `saved figures/...` for each of the 11 pictures (the four
scenarios, the A* vs Dijkstra chart, the weight study, the distance map, the DWA
candidate fan, the preset paths, the trade-off scatter, and the architecture
diagram).

---

## 8. Rebuild the documents as HTML (optional)

Turns the `.md` files (report, slides, study guide, etc.) into printable HTML.

```bash
python build_docs.py
```

Expected: it prints `built REPORT.html`, `built SLIDES.html`, and so on. Open any
`.html` in a browser, then Print → Save as PDF if you need a PDF.

---

## Quick reference — all options for `main.py`

| Option | Values | Meaning |
| --- | --- | --- |
| `--scenario` | `simple`, `maze`, `trap`, `dynamic` | which room |
| `--global` | `astar`, `dijkstra` | which route-finder |
| `--preset` | `balanced`, `aggressive`, `cautious`, `smooth` | driving personality |
| `--save` | e.g. `clip.gif` | save an animation instead of showing it |
| `--experiments` | (flag) | print the result tables and exit |

**The three commands I use most:**

```bash
python main.py --scenario dynamic          # my main live demo
python main.py --scenario trap             # shows why global planning is needed
python main.py --experiments               # the numbers/tables
```
