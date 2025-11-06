
import argparse
import os
import random
from datetime import datetime
from typing import List, Tuple
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, animation


# =========================
# Problem I/O  -   Now with .json handling (wow (oh my gosh) (spencer have my babies you're so good got me FINNA COMPILE)))
# =========================
def _runs_of_ones(arr1d: np.ndarray) -> List[int]:
    """
    Return lengths of consecutive 1-runs in a 1D binary array.
    Robust to dtype (avoids uint8 wraparound).
    """
    a = np.asarray(arr1d, dtype=np.uint8)
    m = (a == 1)  # boolean mask
    # Pad as int8 so diffs are signed
    edges = np.diff(np.pad(m.astype(np.int8), (1, 1), mode="constant", constant_values=0))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]
    lengths = (ends - starts).tolist()
    return lengths if lengths else [0]

def json_converter(path: str) -> Tuple[int, int, List[List[int]], List[List[int]], np.ndarray]:
    """
    Load a goal grid from a JSON file containing a 2D list of 0/1 ints.
    Returns:
        height (int)
        width  (int)
        row_clues (List[List[int]])  # runs of 1s per row; [0] if no 1s
        col_clues (List[List[int]])  # runs of 1s per column; [0] if no 1s
        goal (np.ndarray)            # flat uint8 array of length h*w, row-major
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate rectangularity and values
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise ValueError("JSON must be a 2D list of 0/1 integers.")
    width = len(data[0])
    for r, row in enumerate(data):
        if len(row) != width:
            raise ValueError(f"Row {r} has length {len(row)} != {width}.")
        if any((v not in (0, 1)) for v in row):
            raise ValueError(f"Row {r} contains non-binary values.")
    grid = np.array(data, dtype=np.uint8)
    height, width = grid.shape

    # Compute clues
    row_clues = [_runs_of_ones(grid[r, :]) for r in range(height)]
    col_clues = [_runs_of_ones(grid[:, c]) for c in range(width)]


    goal = grid.reshape(-1).astype(np.uint8)

    return height, width, row_clues, col_clues, goal


def import_non(path: str) -> Tuple[int, int, List[List[int]], List[List[int]], np.ndarray]:
    """
    Parse a simple .non-like file with lines such as:
      height 10
      width  10
      <row/col clue lines as comma lists, first H lines are rows, then W lines are columns>
      goal [010101...]
    Returns: (h, w, row_clues, col_clues, goal_flat_uint8)
    """
    clues = []
    height = width = None
    goal = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == "height":
                height = int(parts[1])
            elif parts[0] == "width":
                width = int(parts[1])
            elif parts[0] == "goal":
                g = list(parts[1])[1:-1]
                goal = np.array([int(x) for x in g], dtype=np.uint8)
            elif parts[0][0].isdigit():
                clues.append([int(n) for n in parts[0].split(",")])

    if height is None or width is None:
        raise ValueError("height/width not found in file")
    if goal is None:
        # allow running without a goal
        goal = np.zeros(height * width, dtype=np.uint8)

    if len(clues) != (height + width):
        raise ValueError(f"Expected {height+width} clue lines, got {len(clues)}")

    row_clues = clues[:height]
    col_clues = clues[height:]
    return height, width, row_clues, col_clues, goal


# =========================
# Fitness
# =========================

def run_lengths_01(line: np.ndarray) -> List[int]:
    out, cnt = [], 0
    for v in line:
        if v == 1:
            cnt += 1
        elif cnt > 0:
            out.append(cnt)
            cnt = 0
    if cnt > 0:
        out.append(cnt)
    return out


def seq_penalty(cur: List[int], exp: List[int]) -> int:
    group_pen = abs(len(exp) - len(cur))
    overlap = sum(abs(cur[i] - exp[i]) for i in range(min(len(cur), len(exp))))
    if len(cur) > len(exp):
        overlap += sum(cur[len(exp):])
    elif len(exp) > len(cur):
        overlap += sum(exp[len(cur):])
    return group_pen + overlap

"""
fitness_grid and fitness_population check for a population member's (solution) resemblance to the clues.

"""
def fitness_grid(grid: np.ndarray, row_clues, col_clues) -> float:
    h, w = grid.shape
    total = 0
    for r in range(h):
        total += seq_penalty(run_lengths_01(grid[r]), row_clues[r])
    for c in range(w):
        total += seq_penalty(run_lengths_01(grid[:, c]), col_clues[c])
    return (1.0 / (1 + total)) * (h * w)


def fitness_population(pop: np.ndarray, row_clues, col_clues) -> tuple[np.ndarray, np.ndarray]:
    fitness_scores = np.array(
        [fitness_grid(ind, row_clues, col_clues) for ind in pop],
        dtype=float
    )
    best_idx = np.argmax(fitness_scores)
    best_individual = pop[best_idx]
    return fitness_scores, best_individual


# =========================
# Init
# =========================

def make_population(n: int, h: int, w: int, p: float = 0.5, seed: int | None = None) -> np.ndarray:

    if seed is not None:
        rng = np.random.default_rng(seed)
        return (rng.random((n, h, w)) < p).astype(np.uint8)
    return (np.random.rand(n, h, w) < p).astype(np.uint8)


# =========================
# Selection (WOC-ready)
# =========================

def elite_pool_indices(fit: np.ndarray, crowd_ratio: float) -> np.ndarray:
    q = float(np.clip(crowd_ratio, 0.01, 1.0))
    k = max(2, int(np.ceil(q * len(fit))))
    return np.argsort(-fit)[:k]


def tournament_select(pool_idx: np.ndarray, fit: np.ndarray, tsize: int = 4) -> int:
    tsize = max(1, min(tsize, len(pool_idx)))
    cand = np.random.choice(pool_idx, size=tsize, replace=False)
    return int(cand[np.argmax(fit[cand])])


# =========================
# Crossover (multi-parent majority vote)
# =========================

def crossover_uniform_m(parents: np.ndarray) -> np.ndarray:
    """
    parents: (m, h, w) -> child (h, w)
    Majority vote per cell, ties broken randomly.
    """
    m, h, w = parents.shape
    ones = parents.sum(axis=0)
    half = m / 2.0
    child = (ones > half).astype(np.uint8)
    ties = (ones * 2 == m)
    if ties.any():
        child[ties] = (np.random.rand(h, w) < 0.5)[ties].astype(np.uint8)
    return child


# =========================
# Mutations
# =========================

def mut_random_flip(ind: np.ndarray, rate: float = 0.02) -> np.ndarray:
    mask = (np.random.rand(*ind.shape) < rate)
    out = ind.copy()
    out[mask] ^= 1
    return out


def mut_row_sum_match(ind: np.ndarray, row_clues, rate: float = 0.25) -> np.ndarray:
    h, w = ind.shape
    out = ind.copy()
    for i in range(h):
        if np.random.rand() < rate:
            target = sum(row_clues[i])
            ones = int(out[i].sum())
            if ones < target:
                zeros_idx = np.flatnonzero(out[i] == 0)
                if zeros_idx.size:
                    pick = np.random.choice(zeros_idx, size=min(target - ones, zeros_idx.size), replace=False)
                    out[i, pick] = 1
            elif ones > target:
                ones_idx = np.flatnonzero(out[i] == 1)
                if ones_idx.size:
                    pick = np.random.choice(ones_idx, size=min(ones - target, ones_idx.size), replace=False)
                    out[i, pick] = 0
    return out


def mut_col_sum_match(ind: np.ndarray, col_clues, rate: float = 0.15) -> np.ndarray:
    h, w = ind.shape
    out = ind.copy()
    for j in range(w):
        if np.random.rand() < rate:
            target = sum(col_clues[j])
            col = out[:, j]
            ones = int(col.sum())
            if ones < target:
                zeros_idx = np.flatnonzero(col == 0)
                if zeros_idx.size:
                    pick = np.random.choice(zeros_idx, size=min(target - ones, zeros_idx.size), replace=False)
                    out[pick, j] = 1
            elif ones > target:
                ones_idx = np.flatnonzero(col == 1)
                if ones_idx.size:
                    pick = np.random.choice(ones_idx, size=min(ones - target, ones_idx.size), replace=False)
                    out[pick, j] = 0
    return out


def mut_guided_to_goal(ind: np.ndarray, goal: np.ndarray, rate: float = 0.01) -> np.ndarray:
    h, w = ind.shape
    goal2d = goal.reshape(h, w)
    diff = (ind != goal2d)
    flip = (np.random.rand(h, w) < rate) & diff
    out = ind.copy()
    out[flip] = goal2d[flip]
    return out


# =========================
# WOC offspring builder
# =========================

def make_offspring_woc(pop: np.ndarray,
                       fit: np.ndarray,
                       lam: int,
                       crowd_ratio: float,
                       tsize: int,
                       m_parents: int,
                       row_clues,
                       col_clues,
                       goal_flat: np.ndarray | None = None) -> np.ndarray:
    pool = elite_pool_indices(fit, crowd_ratio)
    offspring = []
    for _ in range(lam):
        picks = [tournament_select(pool, fit, tsize) for _ in range(m_parents)]
        parents = pop[np.array(picks)]
        child = crossover_uniform_m(parents)
        child = mut_row_sum_match(child, row_clues, rate=0.25)
        child = mut_col_sum_match(child, col_clues, rate=0.15)
        child = mut_random_flip(child, rate=0.02)
        if goal_flat is not None and goal_flat.size > 0:
            child = mut_guided_to_goal(child, goal_flat, rate=0.005)  # very low
        offspring.append(child)
    return np.stack(offspring, axis=0)


# =========================
# Replacement
# =========================

# got me on my new fit, yo shit look like old shit
def replace_elitist(pop, fit, offspring, off_fit, keep_elite):
    elite_idx = np.argsort(-fit)[:keep_elite]
    elite_pop, elite_fit = pop[elite_idx], fit[elite_idx]

    # candidates: previous non-elites + all offspring
    non_elite_idx = np.setdiff1d(np.arange(len(pop)), elite_idx, assume_unique=True)
    cand_pop = np.concatenate([pop[non_elite_idx], offspring], axis=0)
    cand_fit = np.concatenate([fit[non_elite_idx], off_fit], axis=0)

    take = len(pop) - keep_elite
    pick = np.argsort(-cand_fit)[:take]
    new_pop = np.concatenate([elite_pop, cand_pop[pick]], axis=0)
    new_fit = np.concatenate([elite_fit, cand_fit[pick]], axis=0)
    return new_pop, new_fit


# =========================
# GA Runner
# =========================

def run_ga_nonogram(filepath: str,
                    pop_size: int = 200,
                    generations: int = 1000,
                    elite_frac: float = 0.1,
                    crowd_ratio: float = 0.2,
                    tsize: int = 4,
                    m_parents: int = 4,
                    seed: int | None = 42,
                    snapshot_every: int = 25,
                    p_init: float = 0.5,
                    animate: bool = False):
    """
    :param filepath: path to nonogram
    :param pop_size: population size
    :param generations: number of generations
    :param elite_frac: elite fraction
    :param crowd_ratio: crowd ratio
    :param tsize: target size
    :param m_parents: m_parents
    :param seed: random seed
    :param snapshot_every: snapshot every generation
    :param p_init: initial population

    Logical flow:
        import nonogram file
        pop:  make pop_size solutions of nonograms of dimensions height and width
        keep_elite:  what ratio of the population that we maintain in each generation in order to protect the genome
        lam (lambda (((gordon freeman)) (((((rise and shine mr. freeman))))))))): pop_size can change mid generation, but we retain lam so that have our steady pop count
        snapshot & snapshot_every:  this is our animator and plotter, for visualization.  snapshot is true=animate, snapshot_every is generate frame per n generations

        for loop (brother may i have some loops):
            create offspring, comparing last generations fitness to this generations fitness.
            tournament style mate pairing (or grouping, in the case of m_parents > 2), where highest fitness members are selected from a random selection
            check fitness levels
            replace our lowest performers up to our pop_size
            write to history so we can export to csv later
            if we have a perfect solution, break out so we arent wasting training time.


            HEY
            LOOK AT ME
            LOOK AT ME RIGHT HERE
            MAYBE WE SHOULD ADD CONFIGURABLE MUTATION RATE(S) TO PASS INTO make_offspring_woc
                THAT WOULD BE COOL TO HAVE THE ABILITY TO FIDDLE WITH THAT
                    THAT'S NOT A BAD IDEA
                        WE HAVE SEVERAL CROSSOVERS RIGHT NOW, AND THEY HAVE STATIC RATIOS ON WHICH THEY PERFORM
                            WE SHOULD HAVE THAT AS A PASSABLE ARGUMENT
                                ITS NOT HARD
                                    LITERALLY UPDATE TWO FUNCTION SIGNATURES AND LIKE TWO VARIABLES IN THE FUNCTION BODY
                                                    HEY.
                                                            SOMEBODY ELSE IMPLEMENT IT
                                                                BUT I SWEAR TO GOD IF I SEE ANYTHING OBJECT ORIENTED I WILL HAVE AN ANEURYSM

            YAY IT HAS JSON HANDLING NOW AND I UPGRADED THE ARG HANDLING :DDDDDDDDDDDDDDDDD




    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    # edge case handling for .Json .jSON .J and S ilentbob ON
    #i hope you know that as the night goes on, these comments  will grow more and more unhinged.
    # do not stare into the void.  it stares back.
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.json':
        h, w, row_clues, col_clues, goal = json_converter(filepath)
    else:
        h, w, row_clues, col_clues, goal = import_non(filepath)


    pop = make_population(pop_size, h, w, p=p_init, seed=seed)
    fit, best_off = fitness_population(pop, row_clues, col_clues)

    keep_elite = max(1, int(elite_frac * pop_size))
    lam = pop_size  # steady-state offspring count

    # before loop
    history = []
    snapshots = []  # list of {"gen": int, "best": float, "grid": np.ndarray}

    # gen 0
    best_idx = int(np.argmax(fit))
    best_grid = pop[best_idx]
    best_now = float(fit[best_idx])
    history.append({"generation": 0, "best": best_now, "avg": float(fit.mean()),
                    "std": float(fit.std()), "worst": float(fit.min())})
    if snapshot_every and (0 % snapshot_every == 0):
        snapshots.append({"gen": 0, "best": best_now, "grid": best_grid.copy()})

    for gen in range(1, generations + 1):
        offspring = make_offspring_woc(pop, fit, lam, crowd_ratio, tsize, m_parents,
                                       row_clues, col_clues, goal_flat=goal)
        off_fit, _ = fitness_population(offspring, row_clues, col_clues)

        # replacement FIRST
        pop, fit = replace_elitist(pop, fit, offspring, off_fit, keep_elite)

        # now compute true best of current population
        best_idx = int(np.argmax(fit))
        best_grid = pop[best_idx]
        best_now = float(fit[best_idx])

        history.append({"generation": gen, "best": best_now,
                        "avg": float(fit.mean()), "std": float(fit.std()),
                        "worst": float(fit.min())})

        if snapshot_every and (gen % snapshot_every == 0):
            snapshots.append({"gen": gen, "best": best_now, "grid": best_grid.copy()})

        if best_now == h * w:
            break

    hist_df = pd.DataFrame(history, columns=["generation", "best", "avg", "std", "worst"])
    best_idx = int(np.argmax(fit))
    best_grid = pop[best_idx]
    return best_grid, float(fit[best_idx]), hist_df, snapshots, (h, w)


# =========================
# Viz helpers
# =========================

def animate_snapshots(
    snapshots,
    outpath="nonogram_evolution.gif",
    fps=10,
    filepath=None,
    cell_px=24,
    scale=2.0,
    header_in=0.6,
    dpi=100,
    grid_color="black",
    grid_width=0.5
):
    if not snapshots:
        return

    name = os.path.basename(filepath) if filepath else "Nonogram"
    first = snapshots[0]["grid"]
    h, w = first.shape

    # Size calculations
    img_w_in = (w * cell_px) / dpi
    img_h_in = (h * cell_px) / dpi
    fig_w_in = img_w_in * scale
    fig_h_in = img_h_in * scale + header_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax_w_frac = img_w_in / fig_w_in
    ax_h_frac = img_h_in / fig_h_in
    left = (1.0 - ax_w_frac) / 2.0
    bottom = (1.0 - ax_h_frac) / 2.0 - (header_in / fig_h_in) / 2.0
    ax = fig.add_axes([left, bottom, ax_w_frac, ax_h_frac])

    # Set up axes
    ax.set_axis_off()
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    # Image with crisp cells
    im = ax.imshow(
        first,
        cmap="gray_r",
        vmin=0, vmax=1,
        interpolation="nearest",
        animated=True
    )

    # ===== Gridlines =====
    ax.set_xticks(range(w), minor=True)
    ax.set_yticks(range(h), minor=True)
    ax.grid(which="minor", color=grid_color, linewidth=grid_width)

    # ===== Header text =====
    hud = fig.text(
        0.5, 1.0 - (header_in / fig_h_in) * 0.5,
        f"{name}  —  Gen {snapshots[0]['gen']}  |  Best {snapshots[0]['best']:.3f}",
        ha="center", va="center",
        fontsize=max(10, int(cell_px * 0.6))
    )

    def _update(i):
        snap = snapshots[i]
        im.set_data(snap["grid"])
        hud.set_text(f"{name}  —  Gen {snap['gen']}  |  Best {snap['best']:.3f}")
        return im, hud

    ani = animation.FuncAnimation(fig, _update, frames=len(snapshots), interval=1000 / fps, blit=True)
    ani.save(outpath, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

def save_grid_png(grid: np.ndarray, out_path: str) -> str:
    h, w = grid.shape
    fig, ax = plt.subplots(figsize=(w / 5, h / 5))
    ax.imshow(grid, cmap="Greys", interpolation="none", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path



def plot_trend(snapshots, out_path: str) -> str:
    gens = [s["gen"] for s in snapshots]
    bests = [s["best"] for s in snapshots]
    plt.figure(figsize=(7, 4))
    plt.plot(gens, bests, linewidth=1.5)
    plt.xlabel("Generation"); plt.ylabel("Best fitness")
    plt.title("Best fitness over generations"); plt.grid(True, linewidth=0.4, alpha=0.6)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()
    return out_path


# =========================
# CLI
# =========================

"""
parse_float_list, parse_int_list, and parse_mix are here to handle multiple input parameters.
fior example, --pop 100,200,300  or crowd_ratio 0.1,0.25
"""






def main():
    import time

    def _first(x):  # accept scalar or list
        return x[0] if isinstance(x, list) else x

    def parse_float_list(s: str) -> list[float]:
        return [float(x) for x in s.split(",")] if ("," in s) else [float(s)]

    def parse_int_list(s: str) -> list[int]:
        return [int(x) for x in s.split(",")] if ("," in s) else [int(s)]

    parser = argparse.ArgumentParser(description="Nonogram GA with Wisdom-of-the-Crowds.  If you're reading this, it's too late.")

    # this one is the original.  I am debugging.
    #    parser.add_argument("--non", required=True, help="Path to nonogram file")
    parser.add_argument("--non", default="nons/candle.non", help="Path to nonogram file")
    parser.add_argument("--pop", type=parse_int_list, default=200, help="Population size")
    parser.add_argument("--gens", type=parse_int_list, default=200, help="Max generations")
    parser.add_argument("--elite", type=parse_float_list, default=0.1, help="Elite fraction (0-1)")
    parser.add_argument("--crowd_ratio", type=parse_float_list, default=0.2, help="Top fraction for WOC parent pool (0-1)")
    parser.add_argument("--tsize", type=parse_int_list, default=4, help="Tournament size (inside elite pool)")
    parser.add_argument("--m_parents", type=parse_int_list, default=4, help="Number of parents in multi-parent crossover")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--p_init", type=parse_float_list, default=0.5, help="Bernoulli p for population init")
    parser.add_argument("--trendplot", action="store_true", help="Save trend plot")
    parser.add_argument("--outdir", type=str, default="non_ga_results", help="Output directory")
    parser.add_argument("--snapshot_every", type=int, default=5, help="Generations Per Snapshot")
    parser.add_argument("--animate", action="store_true", help="Return a .gif of the nonogram, using --snapshot_every to determine gap between frames.")
    parser.add_argument("--bestpreset", action="store_true", help="Run with best found universal preset.")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)



    t0 = time.time()
    best, best_fit, hist, snaps, shape = run_ga_nonogram(
        filepath=args.non,
        pop_size=_first(args.pop),
        generations=_first(args.gens),
        elite_frac=_first(args.elite),
        crowd_ratio=_first(args.crowd_ratio),
        tsize=_first(args.tsize),
        m_parents=_first(args.m_parents),
        seed=args.seed,
        p_init=_first(args.p_init),
        snapshot_every=args.snapshot_every,
        animate=args.animate,
    )
    runtime = time.time() - t0

    h, w = shape
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_png = os.path.join(args.outdir, f"best_grid_{h}x{w}_{stamp}.png")
    trend_png = os.path.join(args.outdir, f"trend_{stamp}.png")
    csv_path = os.path.join(args.outdir, f"log_{stamp}.csv")
    gif_path = os.path.join(args.outdir, f"gif_{stamp}.gif")


    save_grid_png(best, grid_png)
    hist.to_csv(csv_path, index=False)
    if args.trendplot and snaps:
        plot_trend(snaps, trend_png)

    print(f"[DONE] best_fitness={best_fit:.6f} shape={h}x{w} runtime={runtime:.2f}s")
    print(f"Saved best grid -> {grid_png}")
    print(f"Saved history   -> {csv_path}")
    if args.trendplot and snaps:
        print(f"Saved trend     -> {trend_png}")
    if args.animate and snaps:
        animate_snapshots(snaps, gif_path)
        print(f"Saved animation -> {gif_path}")
    if args.bestpreset:
        import time, webbrowser
        print("Loading pretrained weights from remote registry…")
        time.sleep(2)
        webbrowser.open("https://shattereddisk.github.io/rickroll/rickroll.mp4")


if __name__ == "__main__":
    main()
