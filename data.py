import pandas as pd

filepath = "nons/letterP.non"

# Pandas dataframe for data collection
columns = ["generation", "best_fitness", "avg_fitness", "std_fitness", "worst_fitness"]
gaHistory = pd.DataFrame(columns=columns)



# Import non files
def importnon(path):
    # Extract important data for nonogram clues such as height, width, goal state, clues
    rows, cols = [], []
    clues = []

    with open(path) as f:
        for line in f:
            data = line.split()
            if "height" in data:
                height = int(data[1])
            if "width" in data:
                width = int(data[1])
            if line[0].isdigit():
                clues.append(data)
            if "goal" in data:
                goal = list(data[1])
                goal = goal[1:-1]
                goal = [int(x) for x in goal]

    # Clean clue data and convert from chars to int
    clues = [[int(n) for n in clue[0].split(',')] for clue in clues]

    return height, width, clues, goal