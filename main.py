import GA
import viz
import data
import matplotlib.pyplot as plt

filepath = "nons/letterP.non"

populationSize = 10
generation = 0
maxGenerations = 5000
fitnessDict = {}


if __name__ == '__main__':
    h, w, c, goal = data.importnon(filepath)

    # Establish empty frame for animation
    plt.ion()  
    plt.figure(figsize=(w/5, h/5))
    
    # Initialize GA population. Then evaluate all samples.
    p = GA.generatePopulation(height=h, width=w)
    fitnessDict = GA.setFitnessDict(fitnessDict, population=p, height=h, width=w, clues=c)

    # GA Loop + animation for sample 1 of population.
    while generation < maxGenerations:
        p[0] = GA.constraint_aware_mutation(p[0], clues=c, height=h, width=w, mutation_rate=0.5)
        p[0] = GA.column_constraint_aware_mutation(p[0], clues=c, height=h, width=w)
        fitnessDict = GA.updateFitnessDict(fitnessDict=fitnessDict, population=p, replacement=0, height=h, width=w, clues=c)
        viz.displayChromosomeLive(p[0], h, w, generation=generation, fitness=fitnessDict[0])
        generation += 1

    plt.ioff()
    plt.show()
    