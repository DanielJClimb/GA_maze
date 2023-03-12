# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 10:20:36 2023

@author: danio
"""
import pygad

gene_space=[0, 1, 2, 3]

walls = [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0],[10,0],[11,0],
         [0,1],                  [4,1],                  [8,1],              [11,1],
         [0,2],[1,2],[2,2],                  [6,2],      [8,2],[9,2],       [11,2],
         [0,3],                  [4,3],      [6,3],                         [11,3],
         [0,4],      [2,4],      [4,4],[5,4],            [8,4],[9,4],       [11,4],
         [0,5],            [3,5],[4,5],                  [8,5],             [11,5],
         [0,6],                              [6,6],                  [10,6],[11,6],
         [0,7],      [2,7],            [5,7],[6,7],      [8,7],             [11,7],
         [0,8],      [2,8],[3,8],[4,8],                  [8,8],[9,8],       [11,8],
         [0,9],      [2,9],      [4,9],[5,9],       [7,9],     [9,9],       [11,9],
        [0,10],      [2,10],                                                [11,10],
       [0,11],[1,11],[2,11],[3,11],[4,11],[5,11],[6,11],[7,11],[8,11],[9,11],[10,11],[11,11]]

start = [1,1]
exi = [10,10]


# 0 - ruch w góre
# 1 - ruch w prawo
# 2 - ruch w dół
# 3 ruch w lewo
def movement(x, l):
    if x == 0:
        l[1] -= 1
        if l in walls:
            l[1] +=1
    elif x == 1:
        l[0] +=1
        if l in walls:
            l[0] -=1
    elif x == 2:
        l[1] +=1
        if l in walls:
            l[1] -=1
    elif x ==3:
        l[0] -= 1
        if l in walls:
            l[0] +=1
    return l

def penalty(s):
    k = 0
    for i in range(1, len(s)):
        if abs(s[i] - s[i-1]) == 2:
            k += 5
    return k
    
    
def fitness(solution, solution_idx):
    location = [1,1]
    for i in solution:
        if penalty(solution) > 0:
            kara = penalty(solution)
            fitnes = abs(location[1] - exi[1]) + abs(location[0] - exi[0]) + kara
            break
        else:
            movement(i, location)
            fitnes = abs(location[1] - exi[1]) + abs(location[0] - exi[0])
    return -fitnes
 
fitness_function = fitness

#ile chromsomów w populacji
#ile genow ma chromosom          
sol_per_pop = 150
num_genes = 30

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 75
num_generations = 300
keep_parents = 3

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"
        

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 5

stop_criteria = 'reach_0'


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
                       mutation_percent_genes=mutation_percent_genes,
                       stop_criteria = stop_criteria)

#uruchomienie algorytmu
ga_instance.run()


# funkcja correction usuwa z naszego rozwiązania ruchy, w których
# pozostalismy w miejscu(brak ruchu spowodowany jest chęcia wejscia w scianę).
# Zostało to zrobione za pomocą funkcji, która od nowa przechodzi
# sciezke, ponieważ funkcja ruchu jest tak zdefiniowana, ze uzycie jej
# w petli po kazdym ruchu rozwiazania sprawialo, ze tablica wypelniala się
# w calosci lokalizacja ostatniego ruchu
def correction(s):
    l = [1,1]
    p = []
    for x in s:
        if x == 0:
            l[1] -= 1
            if l in walls:
                l[1] +=1
                p.append(False)
            else:
                p.append(True)
        elif x == 1:
            l[0] +=1
            if l in walls:
                l[0] -=1
                p.append(False)
            else:
                p.append(True)
        elif x == 2:
            l[1] +=1
            if l in walls:
                l[1] -=1
                p.append(False)
            else:
                p.append(True)
        elif x ==3:
            l[0] -= 1
            if l in walls:
                l[0] +=1
                p.append(False)
            else:
                p.append(True)
    return s[p]


#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
solution = correction(solution)
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Exit found in {steps} steps".format(steps=len(solution)))
#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()

location = [1,1]
print('Lokalizacje po ruchach')
for i in solution:
    movement(i, location)
    print(location)
