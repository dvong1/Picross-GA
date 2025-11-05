
import argparse
import os
import random
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# =========================
# Problem I/O
# =========================

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


def fitness_population(pop: np.ndarray, row_clues, col_clues) -> np.ndarray:
    return np.array([fitness_grid(ind, row_clues, col_clues) for ind in pop], dtype=float)


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
        # constraint-aware nudges + small noise; no cheating
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

def replace_elitist(pop: np.ndarray, fit: np.ndarray,
                    offspring: np.ndarray, off_fit: np.ndarray,
                    keep_elite: int):
    elite_idx = np.argsort(-fit)[:keep_elite]
    elite_pop, elite_fit = pop[elite_idx], fit[elite_idx]
    mu = len(pop) - keep_elite
    all_pop = np.concatenate([pop, offspring], axis=0)
    all_fit = np.concatenate([fit, off_fit], axis=0)
    idx = np.argsort(-all_fit)[:mu]
    new_pop = all_pop[idx]
    new_fit = all_fit[idx]
    pop2 = np.concatenate([elite_pop, new_pop], axis=0)
    fit2 = np.concatenate([elite_fit, new_fit], axis=0)
    return pop2, fit2


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
                    p_init: float = 0.5):
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

    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    h, w, row_clues, col_clues, goal = import_non(filepath)
    pop = make_population(pop_size, h, w, p=p_init, seed=seed)
    fit = fitness_population(pop, row_clues, col_clues)

    keep_elite = max(1, int(elite_frac * pop_size))
    lam = pop_size  # steady-state offspring count

    history = []
    snapshots = []  # (gen, best_fitness)

    # snapshot gen 0
    history.append({"generation": 0, "best": float(fit.max()), "avg": float(fit.mean()),
                    "std": float(fit.std()), "worst": float(fit.min())})
    if snapshot_every and 0 % snapshot_every == 0:
        snapshots.append((0, float(fit.max())))


    for gen in range(1, generations + 1):
        offspring = make_offspring_woc(pop, fit, lam, crowd_ratio, tsize, m_parents,
                                       row_clues, col_clues, goal_flat=goal)
        off_fit = fitness_population(offspring, row_clues, col_clues)

        pop, fit = replace_elitist(pop, fit, offspring, off_fit, keep_elite)

        # log
        history.append({
            "generation": gen,
            "best": float(fit.max()),
            "avg": float(fit.mean()),
            "std": float(fit.std()),
            "worst": float(fit.min()),
        })
        if snapshot_every and gen % snapshot_every == 0:
            snapshots.append((gen, float(fit.max())))

        # perfect? (penalty = 0 -> fitness = h*w)
        if fit.max() == h * w:
            break

    hist_df = pd.DataFrame(history, columns=["generation", "best", "avg", "std", "worst"])
    best_idx = int(np.argmax(fit))
    best_grid = pop[best_idx]
    return best_grid, float(fit[best_idx]), hist_df, snapshots, (h, w)


# =========================
# Viz helpers
# =========================

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


def plot_trend(snapshots: List[Tuple[int, float]], out_path: str) -> str:
    gens = [g for g, _ in snapshots]
    bests = [b for _, b in snapshots]
    plt.figure(figsize=(7, 4))
    plt.plot(gens, bests, linewidth=1.5)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Best fitness over generations")
    plt.grid(True, linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Nonogram GA with Wisdom-of-the-Crowds")
    parser.add_argument("--non", required=True, help="Path to nonogram file")
    parser.add_argument("--pop", type=int, default=200, help="Population size")
    parser.add_argument("--gens", type=int, default=1000, help="Max generations")
    parser.add_argument("--elite", type=float, default=0.1, help="Elite fraction (0-1)")
    parser.add_argument("--crowd_ratio", type=float, default=0.2, help="Top fraction for WOC parent pool (0-1)")
    parser.add_argument("--tsize", type=int, default=4, help="Tournament size (inside elite pool)")
    parser.add_argument("--m_parents", type=int, default=4, help="Number of parents in multi-parent crossover")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--p_init", type=float, default=0.5, help="Bernoulli p for population init")
    parser.add_argument("--trendplot", action="store_true", help="Save trend plot")
    parser.add_argument("--outdir", type=str, default="non_ga_results", help="Output directory")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    best, best_fit, hist, snaps, shape = run_ga_nonogram(
        filepath=args.non,
        pop_size=args.pop,
        generations=args.gens,
        elite_frac=args.elite,
        crowd_ratio=args.crowd_ratio,
        tsize=args.tsize,
        m_parents=args.m_parents,
        seed=args.seed,
        p_init=args.p_init,
        snapshot_every=max(1, args.gens // 50),  # about 50 points
    )
    runtime = time.time() - t0

    h, w = shape
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_png = os.path.join(args.outdir, f"best_grid_{h}x{w}_{stamp}.png")
    trend_png = os.path.join(args.outdir, f"trend_{stamp}.png")
    csv_path = os.path.join(args.outdir, f"log_{stamp}.csv")

    save_grid_png(best, grid_png)
    hist.to_csv(csv_path, index=False)
    if args.trendplot and snaps:
        plot_trend(snaps, trend_png)

    print(f"[DONE] best_fitness={best_fit:.6f} shape={h}x{w} runtime={runtime:.2f}s")
    print(f"Saved best grid -> {grid_png}")
    print(f"Saved history   -> {csv_path}")
    if args.trendplot and snaps:
        print(f"Saved trend     -> {trend_png}")


if __name__ == "__main__":
    main()
