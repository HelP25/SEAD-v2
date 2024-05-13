import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from SEAD_v2.Optimise.Safe_corridor import *
from SEAD_v2.Assets import *

def results_analysis(ga, first_front, length, first_time, first_front_history, striker):
    print(f"First time that we have a good solution: {first_time}")
    print(f'There are {len(first_front)} solutions')

    # Removing all the solutions where a jammer is detected
    new_first_front = []
    for ind in first_front:
        # Updating the battle space according to the individual tested
        for i, jammer in enumerate(Jammer.list):
            jammer.update(ind[i][0], ind[i][1])
            jammer.targets(ind[i][2])
        if any_detection(5) == 1:   # Checking if a jammer is detected
            new_first_front.append(ind)
        # Resetting the battle context
        for radar in sensor_iads.list:
            radar.jammers_targeting = []
    first_front = new_first_front.copy()

    plt.close("all")

    # Saving the solutions before removing their useless individuals
    old_fitness = {}
    old_cost_operation = {}
    old_first_front = first_front.copy()
    for k in range(len(old_first_front)):
        cost = 0
        old_fitness[k + 1] = ga.fitness(first_front[k])
        for jammer in Jammer.list:
            jammer.update(first_front[k][i][0], first_front[k][i][1])
            cost += jammer.use_cost + abs(
                old_fitness[k + 1][2]) * jammer.fuel_consumption  # Calculating the cost of the jammer
        old_cost_operation[k + 1] = cost  # Saving the cost of the solution


    # Removing all the jammers that aren't useful in the remaining solutions
    fitness = {}    # Saving the fitness of the solution
    cost_operation = {} # Saving the operation cost of the solution
    nb_jammers = {} # Saving the number of jammers of the solution
    for k in range(len(first_front)):   # Checking all the solutions
        cost = 0
        jammers_removed = []    # List of the jammers removed
        plt.subplot(len(first_front) // 2 + 1, 2, k + 1)
        fitness[k + 1] = ga.fitness(first_front[k]) # Fitness of the solution without removing jammers

        # Testing to remove one by one the jammers
        for i, jammer in enumerate(Jammer.list):
            jammer_at_stake = first_front[k][i] # Saving the jammer's genome to be removed
            first_front[k].pop(i)   # Removing the jammer in the genome of the individual
            Jammer.list.remove(jammer)  # Removing the object from the list of objects
            jammers_removed.append(jammer)  # Saving the object removed
            corridor_test = ga.fitness(first_front[k], simplification=True) # Calculating the new fitness
            if not (fitness[k + 1][0] - corridor_test[0] <= 50 and corridor_test[0] > 0 and corridor_test[1] == 1): # If the jammer can't be removed
                first_front[k].append(jammer_at_stake)  # Adding back the genome of the removed jammer
                Jammer.list.append(jammer)  # Adding back the removed object
                jammers_removed.remove(jammer)
            else:
                fitness[k + 1] = list(fitness[k + 1])
                fitness[k + 1][0] = corridor_test[0]    # Updating the fitness now that a jammer has been removed successfully
                fitness[k + 1] = tuple(fitness[k + 1])
        nb_jammers[k + 1] = len(first_front[k]) # Updating the number of jammers in the solution

        # Updating the graph with less jammers
        for i, jammer in enumerate(Jammer.list):
            jammer.update(first_front[k][i][0], first_front[k][i][1])
            plt.plot(jammer.X, jammer.Y, 'r4', markersize=10)
            plt.text(jammer.X, jammer.Y, jammer.name)
            jammer.targets(first_front[k][i][2])
            cost += jammer.use_cost + abs(fitness[k + 1][2]) * jammer.fuel_consumption  # Calculating the cost of the jammer
        cost_operation[k + 1] = cost    # Saving the cost of the solution

        # Plotting the radars
        for radar in sensor_iads.list:
            plt.plot(radar.X, radar.Y, 'bs', markersize=10)
            plt.text(radar.X, radar.Y, radar.name)
            radar.outline_detection(striker)

        Jammer.list.extend(jammers_removed) # To go back to the original list of objects for the next solution to study

    # Printing the tables
    old_table = [[f'Solution number {k + 1}', old_fitness[k + 1][0], old_fitness[k + 1][1], old_fitness[k + 1][2], ga.nb_jammers,
              old_cost_operation[k + 1]] for k in range(len(old_first_front))]
    table = [[f'Solution number {k + 1}', fitness[k + 1][0], fitness[k + 1][1], fitness[k + 1][2], nb_jammers[k + 1],
              cost_operation[k + 1]] for k in range(len(first_front))]
    headers = ['Solutions', 'Corridor width (km)', '2nd Objective function', '3rd objective function', 'Number of jammer used',
               'Cost of the operation (US$)']
    print('Results before removing the useless jammers: ')
    print(tabulate(old_table, headers, tablefmt='fancy_grid', floatfmt='.2f', numalign='right'))
    print('Results now that the useless jammer have been removed:')
    print(tabulate(table, headers, tablefmt='fancy_grid', floatfmt='.2f', numalign='right'))

    plt.show()

    print(f'length of the first front: {length}')
    plt.show()


    # Displaying the animation of the evolution of the first front depending on the iterations
    def animate(optimizer):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')

        def update(i):
            x = [ind[0] for ind in optimizer.first_front_history[i]]
            y = [ind[1] for ind in optimizer.first_front_history[i]]
            z = [ind[2] for ind in optimizer.first_front_history[i]]
            ax.clear()
            ax.scatter(x, y, z)
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_zlabel('Objective 3')
            ax.set_title(f'Iteration {i}')
            ax.set_xlim(-150, 100)
            ax.set_ylim(-1, 3)
            ax.set_zlim(-950, -650)
            return ax

        ani = animation.FuncAnimation(fig, update, frames=len(optimizer.first_front_history), interval=500)
        plt.show()

    animate(ga)

    # Displaying a graph of the position of all the jammers of all the individuals of the final population
    pop = ga.population

    xs = np.array([[j[0] for j in idx] for idx in pop]).flatten()
    ys = np.array([[j[1] for j in idx] for idx in pop]).flatten()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(xs, ys)
    plt.show()

