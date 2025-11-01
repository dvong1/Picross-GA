import GA
import viz
import data
import matplotlib.pyplot as plt
from data import gaHistory
import random


filepath = "nons/ubuntu.non"

populationSize = 50
generation = 0
maxGenerations = 1000


if __name__ == '__main__':
    h, w, c, goal = data.importnon(filepath)

    # Establish empty frame for animation
    # plt.ion()  
    # plt.figure(figsize=(w/5, h/5))
    
    # Initialize GA population. Then evaluate all samples.
    p = GA.generatePopulation(height=h, width=w, clues=c)
    gaHistory = data.updateHistory(df=gaHistory, generation=generation, population=p)

    # GA Loop + animation for sample 1 of population.
    while generation < maxGenerations:
        # Crossover 1
        p1, p2 = GA.selectParents(p)
        child = GA.crossover1(parent1=p1, parent2=p2)
        worstSample = min(p, key=lambda p:p.fitness)
        worstSample = child

        # Crossover 2
        p1, p2 = GA.selectParents(p)
        child = GA.row_based_crossover(parent1=p1, parent2=p2)
        worstSample = min(p, key=lambda p:p.fitness)
        worstSample = child

        # Mutation 1
        mutateIdx = random.randrange(len(p))
        possibleMutation = p[mutateIdx]
        p[mutateIdx] = GA.constraint_aware_mutation(chromosome=possibleMutation, clues=c, height=h, width=w)

        # Mutation 2
        mutateIdx = random.randrange(len(p))
        possibleMutation = p[mutateIdx]
        p[mutateIdx] = GA.column_constraint_aware_mutation(chromosome=possibleMutation, clues=c, height=h, width=w)

        # Mutation 3
        mutateIdx = random.randrange(len(p))
        possibleMutation = p[mutateIdx]
        p[mutateIdx] = GA.guided_mutation(chromosome=possibleMutation, goal_state=goal, clues=c)


        gaHistory = data.updateHistory(df=gaHistory, generation=generation, population=p)

        best = max(p, key=lambda c: c.fitness)
        # print(best.fitness)

        viz.displayChromosomeLive(best.chromosome, h, w, generation=generation, fitness=best.fitness)
        generation += 1

    # plt.ioff()
    # plt.show()

    # viz.displayChromosome(best.chromosome, h, w, fitness=best.fitness, generation=generation)

    print(gaHistory.head(2))
    print(gaHistory.tail(2))
    