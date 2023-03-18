# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 09:19:19 2023

@author: danio
"""

import pygad
import numpy as np
import math

def endurance(x, y, z, u, v, w):
    return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w)

gene_space = {'low':0,'high':1}


# funkcja fitness
def fitness_func(solution, solution_idx):
    if np.any(solution <0):
        return 0
    elif np.any(solution>=1):
        return 0
    else:
        x,y,z,u,v,w = solution
        return endurance(x,y,z,u,v,w)

fitness_function = fitness_func

# ile chromosomów w populacji
# ile genów ma chromosom
sol_per_pop = 30
num_genes = 6

# ilu rodziców do rozmnażania
# ile pokolen
# ilu rodziców zachować
num_parents_mating = 15
num_generations = 30
keep_parents = 2


#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 17


#inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,)

#uruchomienie algorytmu
ga_instance.run()

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()


#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
