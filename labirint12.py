# -*- coding: utf-8 -*-
"""

@author: danio
"""
import pygad
import numpy as np
import time
import matplotlib.pyplot as plt

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


# 0 - up
# 1 - right
# 2 - down
# 3 -left

def movement(x, l):
    '''
    Function update location, based on number indicating next move.
    Then, if new location is in walls, function chnages location back to
    previous one, in which we were before hit.
    
    
    Parameters
    ----------
    x : int
        Number from set [0,1,2,3] indicating direction of next move
    l : list
        List that contains current location

    Returns
    -------
    l : list
        Function returns list with location after move
    '''
    t = l
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
    return t

def penalty(s):
    '''
    Function measures the penalty for wasting moves byb heading back to the 
    location, from which we have just move.

    Parameters
    ----------
    s : numpy.array
        Array, which contains directions of every move

    Returns
    -------
    k : int
        Penalty for wasting moves.

    '''
    k = 0
    for i in range(1, len(s)):
        if abs(s[i] - s[i-1]) == 2:
            k += 5
    return k
    
    
def fitness(solution, solution_idx):
    '''
    Function measures fintess of our solution, by calculating how far
    are we from exit in current location.
    
    Parameters
    ----------
    solution : numpy.array
        Array with directions of each move
    solution_idx : index
        Index of numpy.array

    Returns
    -------
    int
        Fitness of our solution

    '''
    location = [1,1]
    for i in solution:
        if penalty(solution) > 0:
            kara = penalty(solution)
            fitnes = abs(location[1] - exi[1]) + abs(location[0] - exi[0]) + kara
            break
        else:
            location = movement(i, location)
            fitnes = abs(location[1] - exi[1]) + abs(location[0] - exi[0])
    return -fitnes
 
fitness_function = fitness

#chromosoms in population
#number of genes in each chromosom        
sol_per_pop = 150
num_genes = 30

#how many parents we choose for mating
#how many generations 
#percentage of parents to keep
num_parents_mating = 75
num_generations = 300
keep_parents = 3

#type of parents selection
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"
        

#type of crossover
crossover_type = "single_point"

#type of mutation
#chance of mutation(%)
mutation_type = "random"
mutation_percent_genes = 5

#we want ot stop, when we reach exit =  fitness equals 0
stop_criteria = 'reach_0'


#icreating our algorithm
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

#runnig algorithm
ga_instance.run()


# kolor scian niebieski
# kolor pol czerwony
# zielona linia sciezka


def correction(s):
    '''
    Function take our solution and clean it from movements in which we 
    stayed in place(our next movement would be hitting the wall). 
    Every move is checked. If it would be hitting the wall, p list
    is appended with False, else it is appended with True. At the end p is 
    used to obtain clean solution.

    Parameters
    ----------
    s : numpy.array
        array, which includes set of moves

    Returns
    -------
    numpy.array
        Array containg cleaned path

    '''
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


#Summary: best solution(chromosom + fitness)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
solution = correction(solution)
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Exit found in {steps} steps".format(steps=len(solution)))
#plot, which show, how our fitness was changing
ga_instance.plot_fitness()

location = [1,1]
moves = [[1,1]]
print('Location after each move')
for i in solution:
    location = movement(i, location)
    t = location.copy()
    moves.append(t)
    print(location)


place_x =[]
for i in range(12):
    place_x.extend(list(np.arange(0,12)))
place_y =[]
for i in range(12):
    for j in range(12):
        place_y.append(i)
x = [x[0] for x in walls]
y = [y[1] for y in walls]
path_x =[x[0] for x in moves]
path_y = [y[1] for y in moves]

fig, ax = plt.subplots()
ax.scatter(place_x, place_y, color='red')
ax.scatter(x,y, color='blue')
ax.scatter(1,1, color = 'black')
ax.scatter(10,10, color='orange')
plt.plot(path_x, path_y, color='green')
plt.show()
# kolor niebieski sciany
# kolor czerwony pola
# kolor zielony droga
# kolor czarny start
# kolor pomarańczowy koniec


mean_time = []
for i in range(10):
    start = time.time()
    ga_instance.run()
    end = time.time()
    mean_time.append(end -start)

mean_time = np.array(mean_time)
mean = np.mean(mean_time)
print('Mean time: {mean}'.format(mean = mean))