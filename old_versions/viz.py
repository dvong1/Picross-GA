import numpy as np
import matplotlib.pyplot as plt
from data import importnon, filepath

# Display solution / goal state. You need to first extract the data from the .non file for all viz functions
def displayGoalImage(goal, height, width):
    grid = np.array(goal).reshape((height, width))

    fig, ax = plt.subplots(figsize=(width / 5, height / 5))

    ax.imshow(grid, cmap='Greys', interpolation='none', vmin=0, vmax=1)

    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

# Display static image for any chromosome sample from population
def displayChromosome(chromosome, height, width, fitness=None, generation=None):
    grid = np.array(chromosome).reshape((height, width))

    fig, ax = plt.subplots(figsize=(width / 5, height / 5))

    ax.imshow(grid, cmap='Greys', interpolation='none', vmin=0, vmax=1)

    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])

    title = ""
    if generation is not None:
        title += f"Generation: {generation}  "
    if fitness is not None:
        title += f"Fitness: {fitness:.4f}"
    plt.title(title, fontsize=10)

    plt.show()
    
# Display the nonogram Puzzle initial state. All empty with clues visible.
def displayNonogramPuzzle(height, width, clues):
    # Split the clues
    row_clues = clues[:height]
    col_clues = clues[height:]

    # Create an empty grid
    grid = np.zeros((height, width))

    fig, ax = plt.subplots(figsize=(width / 2, height / 2))

    # Show the grid
    ax.imshow(grid, cmap='Greys', vmin=0, vmax=1)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    # Hide major ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Display row clues on the left
    max_row_clues = max(len(c) for c in row_clues)
    for i, clue in enumerate(row_clues):
        text = " ".join(map(str, clue))
        ax.text(-max_row_clues, i, text, ha='right', va='center', fontsize=10)

    # Display column clues on top
    max_col_clues = max(len(c) for c in col_clues)
    for j, clue in enumerate(col_clues):
        for k, n in enumerate(reversed(clue)):
            ax.text(j, - (max_col_clues - k), str(n), ha='center', va='center', fontsize=10)

    # Adjust margins to make space for top and left clues
    plt.subplots_adjust(left=0.2, top=1 - max_col_clues * 0.05, bottom=0.05)

    plt.show()


# Dynamic visualization for chromosomes evolving through GA. Use Ctrl+C in terminal to quit early
def displayChromosomeLive(chromosome, height, width, generation=None, fitness=None):
    plt.clf()  
    grid = np.array(chromosome).reshape((height, width))
    plt.imshow(grid, cmap='Greys', interpolation='none', vmin=0, vmax=1)
    
    # Hide ticks
    plt.xticks([])
    plt.yticks([])
    
    # Build title
    title = ""
    if generation is not None:
        title += f"Generation: {generation}  "
    if fitness is not None:
        title += f"Fitness: {fitness:.4f}"
    plt.title(title, fontsize=10)
    
    plt.pause(0.05)  # Pause to allow the figure to update

if __name__ == '__main__':
    h, w, c, g = importnon(filepath)
    displayGoalImage(goal=g, height=h, width=w)
    displayNonogramPuzzle(height=h, width=w, clues=c)