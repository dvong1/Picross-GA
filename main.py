import GA
import viz
import data
import matplotlib.pyplot as plt


filepath = "nons/letterP.non"

populationSize = 10
generation = 0
maxGenerations = 5000


if __name__ == '__main__':
    h, w, c, goal = data.importnon(filepath)

    # Establish empty frame for animation
    plt.ion()  
    plt.figure(figsize=(w/5, h/5))
    
    # Initialize GA population. Then evaluate all samples.
    p = GA.generatePopulation(height=h, width=w, clues=c)

    # GA Loop + animation for sample 1 of population.
    while generation < maxGenerations:
        p[0] = GA.constraint_aware_mutation(p[0], clues=c, height=h, width=w, mutation_rate=0.9)
        print(p[0].fitness)
        p[0] = GA.column_constraint_aware_mutation(p[0], clues=c, height=h, width=w, mutation_rate=0.9)
        viz.displayChromosomeLive(p[0].chromosome, h, w, generation=generation, fitness=p[0].fitness)
        generation += 1

    plt.ioff()
    plt.show()
    