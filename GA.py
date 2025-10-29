import random
from data import importnon, filepath

# Genetic Algorithm {
#   initialize population;
#   evaluate population;
#   while TerminationCriteriaNotSatisfied{
#       Select parents for reproduction;
#       Perform recombination and mutation;
#       Evaluate population;
#   }
# } 

populationSize = 10
maxGenerations = 5000 # Termination Criteria


##### INITIALIZATION #############

# Initialize fitness dict to keep track of fitness of each population sample
def setFitnessDict(fitnessDict, population, height, width, clues):
    for i, chromosome in enumerate(population):
        fitnessDict[i] = evaluate(chromosome=chromosome, height=height, width=width, clues=clues)

    return fitnessDict

# Update fitness dict when genetic operator occurs
def updateFitnessDict(fitnessDict, population, replacement, height, width, clues):
    fitnessDict[replacement] = evaluate(chromosome=population[replacement], height=height, width=width, clues=clues)

    return fitnessDict

# Create a randomized population of size N
def generatePopulation(height, width):
    population = []

    while(len(population)) < populationSize:
        chromosome = [] 

        for row in range(0, height):
            randomRow = random.choices(range(0, 2), k=width)
            chromosome.append(randomRow)

        population.append(chromosome)

    return population
    
###### EVALUATION #######
def evaluate(chromosome, height, width, clues):
    # Use constraint satisfaction as evaluation function
    # To do this, we have must make 2 comparisons, does the row's sequence satisfy the constraints from the clues?
    # Vice verse for the columns
    # Then how can we compare how similar a chromosome's sequences are to the nonogram's constraints?

    row_clues = clues[:height]   # Splice clues into rows and columns
    col_clues = clues[height:] 

    rowConstraints = []         # Save sample's sequences from each row and each column
    colConstraints = []

    total_penalty = 0           # Penalty to define fitness. How far off is rowConstraints from row_clues?

    # Generate current sequence on all rows
    for row in chromosome:
        count = 0
        currentRowConstraint = []

        for pixel in row:
            if pixel == 1:
                count += 1
            elif pixel == 0 and count > 0:
                currentRowConstraint.append(count)
                count = 0

        if count > 0:
            currentRowConstraint.append(count)

        rowConstraints.append(currentRowConstraint)

    # Generate current sequence for all columns
    for col_index in range(width):
        column = []
        for row_index in range(height):
            # Build column
            column.append(chromosome[row_index][col_index])
            count = 0
            currentColConstraint = []
        for pixel in column:
            if pixel == 1:
                count += 1
            elif pixel == 0 and count > 0:
                currentColConstraint.append(count)
                count = 0
        if count > 0:
            currentColConstraint.append(count)

        colConstraints.append(currentColConstraint)

    # print(f"Chromosome's Rows sequences: {rowConstraints}")
    # print(f"Chromosome's Column sequences: {colConstraints}")
    # print(f"Puzzle's Rows Constraints: {row_clues}")
    # print(f"Puzzle'Column Constraints: {col_clues}")

    # Compare the sequences of each row to the puzzle's row clues. Calculate penalty
    for comparison in zip(rowConstraints, row_clues):
        currentRow = comparison[0]
        expectedRow = comparison[1]

        # Penalty for difference in number of contiguous blocks
        group_penalty = abs(len(expectedRow) - len(currentRow))

        # Penalty for difference in block sizes for overlapping blocks
        block_penalty = sum(abs(currentRow[i] - expectedRow[i]) for i in range(min(len(currentRow), len(expectedRow))))

        # Penalty for extra blocks in currentRow or missing blocks in expectedRow
        if len(currentRow) > len(expectedRow):
            block_penalty += sum(currentRow[len(expectedRow):])
        elif len(expectedRow) > len(currentRow):
            block_penalty += sum(expectedRow[len(currentRow):])

        row_penalty = group_penalty + block_penalty
        total_penalty += row_penalty

        # print(f"Current row: {currentRow}")
        # print(f"Expected row: {expectedRow}")
        # print(f"Group penalty: {group_penalty}, Block penalty: {block_penalty}, Row penalty: {row_penalty}")

    # Calculate penalty from comparing column clues to column sequences
    for comparison in zip(colConstraints, col_clues):
        currentCol = comparison[0]
        expectedCol = comparison[1]

        group_penalty = abs(len(expectedCol) - len(currentCol))

        block_penalty = sum(abs(currentCol[i] - expectedCol[i]) for i in range(min(len(currentCol), len(expectedCol))))

        if len(currentCol) > len(expectedCol):
            block_penalty += sum(currentCol[len(expectedCol):])
        elif len(expectedCol) > len(currentCol):
            block_penalty += sum(expectedCol[len(currentCol):])

        col_penalty = group_penalty + block_penalty
        total_penalty += col_penalty

    # Define fitness on a scale from 0-1
    fitness = (1 / (1 + total_penalty)) * width * height

    return fitness

#### GENETIC CROSSOVERS #####

# Swap rows? - mutation
# Take a few rows from parent1 and a few rows from paretn2 - crossover
# Are there constraint aware operators?


def crossover1(chromosome):
    return 0



#### GENETIC RECOMBINATION ####

def constraint_aware_mutation(chromosome, clues, height, width, mutation_rate=0.2):
    """
    chromosome: 2D list (height x width)
    clues: list of row or column clues
    mutation_rate: probability to mutate a row/column
    """
    new_chromosome = [row[:] for row in chromosome]  # deep copy

    # Mutate rows
    for i in range(height):
        if random.random() < mutation_rate:
            clue = clues[i]
            row = new_chromosome[i]

            # Total number of 1s we should have in this row
            target_ones = sum(clue)

            # Current number of 1s
            current_ones = sum(row)

            # Add or remove 1s to match clue sum
            if current_ones < target_ones:
                # Add 1s randomly in 0 positions
                zeros = [idx for idx, val in enumerate(row) if val == 0]
                for _ in range(target_ones - current_ones):
                    if zeros:
                        idx = random.choice(zeros)
                        row[idx] = 1
                        zeros.remove(idx)
            elif current_ones > target_ones:
                # Remove 1s randomly
                ones = [idx for idx, val in enumerate(row) if val == 1]
                for _ in range(current_ones - target_ones):
                    if ones:
                        idx = random.choice(ones)
                        row[idx] = 0
                        ones.remove(idx)

            # Optional: randomly shift blocks to better match contiguous groups
            # (more advanced, can be implemented later)

    return new_chromosome

def column_constraint_aware_mutation(chromosome, clues, height, width, mutation_rate=0.5):
    """
    Mutates a chromosome with respect to column constraints.
    
    chromosome: 2D list (height x width)
    clues: full list of row + column clues
    mutation_rate: probability to mutate each column
    """
    import random

    # Extract column clues
    col_clues = clues[height:]  # columns are after the row clues

    # Deep copy of chromosome
    new_chromosome = [row[:] for row in chromosome]

    # Iterate over each column
    for j in range(width):
        if random.random() < mutation_rate:
            clue = col_clues[j]
            # Extract current column
            col = [new_chromosome[i][j] for i in range(height)]

            target_ones = sum(clue)
            current_ones = sum(col)

            if current_ones < target_ones:
                zeros = [i for i, val in enumerate(col) if val == 0]
                for _ in range(target_ones - current_ones):
                    if zeros:
                        idx = random.choice(zeros)
                        col[idx] = 1
                        zeros.remove(idx)
            elif current_ones > target_ones:
                ones = [i for i, val in enumerate(col) if val == 1]
                for _ in range(current_ones - target_ones):
                    if ones:
                        idx = random.choice(ones)
                        col[idx] = 0
                        ones.remove(idx)

            # Write back the mutated column
            for i in range(height):
                new_chromosome[i][j] = col[i]

    return new_chromosome

if __name__ == '__main__':
    height, width, clues, goal = importnon(filepath) # Retrieve info from .non file
    
    population = generatePopulation(width=width, height=height)

