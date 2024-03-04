import numpy as np

from SEAD_v2.Assets import *
from SEAD_v2.Optimise.Safe_corridor import *
import random
from scipy.stats.qmc import LatinHypercube


class MultiObjGeneticAlgorithm:
    best_individual = None
    def __init__(self, X0, Y0, nb_jammers, aircraft_secured, security_width, population_size, chance_to_mutate, chance_to_crossover):
        self.population_size = population_size
        self.chance_to_mutate = chance_to_mutate
        self.chance_to_crossover = chance_to_crossover
        self.nb_jammers = nb_jammers
        self.aircraft_secured = aircraft_secured
        self.security_width = security_width
        self.X0 = X0
        self.Y0 = Y0
        self.population = [self.create_random_genome() for _ in range(population_size)]
        for _ in range(self.nb_jammers): # creation of our jammers
            jammer = Jammer(self.X0,self.Y0)


    def create_random_genome(self):

        # Creation of the borders within the individuals are generated
        maxX = max([radar.X for radar in sensor_iads.list])
        maxY = max([radar.Y for radar in sensor_iads.list])
        minY = min([radar.Y for radar in sensor_iads.list])

        # Generation of an individual using the LHS
        sample = LatinHypercube(3)
        genome = sample.random(self.nb_jammers).tolist()
        for i in range(self.nb_jammers):
            genome[i][0] = int(genome[i][0] * maxX)
            genome[i][1] = int(minY + genome[i][1] * (maxY - minY))
            genome[i][2] = random.choice(sensor_iads.list)
        return genome

    def select_individuals(self):
        # Perform non-dominated sorting
        fronts, definition_pop = self.fast_non_dominated_sorting([[individual, self.fitness(individual)] for individual in self.population])

        # Calculates the crowding distances
        crowding_distances = self.crowding_distance(fronts)

        # Selection
        selected_individuals = []
        i = 0
        ind = False
        while i < len(fronts) and ind == False:
            # Add all the individuals from the front i in the selected population
            selected_individuals.extend([definition_pop[individual] for individual in fronts[i]])

            # If the length of the selected population reaches the length of the population, then the selection must stop
            if len(selected_individuals) >= self.population_size:
                ind = True
            else:
                front = fronts[i]
                front = sorted(front, key=lambda x: crowding_distances[x], reverse=True)
                selected_individuals.extend(definition_pop[individual] for individual in front[:self.population_size-len(selected_individuals)])
            i+=1

        return selected_individuals


    def crowding_distance(self, fronts):
        """
        Calculates the crowding distance for all the Pareto fronts.
        :param fronts: A 3D list
        :return: A dictionary with the individuals defined by their fitness and the crowding distance related
        """

        # Initialize crowding distances
        distances ={}

        for front in fronts:
            front_size = len(front)

            if front_size <= 2:
                # If there are less than 2 individuals in the front, then the distances must be infinite
                distances.update({individual: np.inf for individual in front})

            else:
                # Sort the front by each objective
                sorted_front = [sorted(front, key=lambda x: x[i]) for i in range(3)]

                # Calculate the crowding distance for every individual of the front
                for i in range(3):
                    distances_i = [0.0] * front_size
                    distances_i[0] = np.inf
                    distances_i[-1] = np.inf
                    for j in range(1, front_size - 1):
                        distances_i[j] = (sorted_front[i][j + 1][i] - sorted_front[i][j - 1][i]) / (sorted_front[i][-1][i] - sorted_front[i][0][i])
                    # Add the crowding distances to every individual
                    for j in range(front_size):
                        if front[j] in distances:
                            distances[front[j]] += distances_i[j]
                        else:
                            distances[front[j]] = distances_i[j]
        return distances

    def fast_non_dominated_sorting(self, population):
        fronts=[[]]
        S = [[] for _ in range(self.population_size)]
        domination_count = [0 for _ in range(self.population_size)]
        definition_pop = {individual[1]: individual[0] for individual in population}
        pop_fitness = definition_pop.keys()

        for i, p in enumerate(pop_fitness):
            for j, q in enumerate(pop_fitness):
                if (all(q[k] <= p[k] for k in range(3)) and any(q[k] < p[k] for k in range(3))):
                    S[i].append((j, q))
                elif (all(p[k] <= q[k] for k in range(3)) and any(p[k] < q[k] for k in range(3))):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append((i, p))

        k = 0
        while fronts[k]:
            Q = []
            for l, (i, p) in enumerate(fronts[k]):
                for j, q in S[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        Q.append((j, q))
                fronts[k][l] = p
            k += 1
            fronts.append(Q)
        return fronts, definition_pop


    def crossover(self, P1, P2):

        # The crossover is only be done with a certain probability: chance_to_crossover
        if random.random() < self.chance_to_crossover:
            C1 = [[] for i in range(self.nb_jammers)]
            C2 = [[] for i in range(self.nb_jammers)]
            def distance(i, radar, Cx):
                '''
                Provides the distance between a radar and a jammer from an individual Cx
                Parameters
                ----------
                i : index of the jammer
                radar : object of the class sensor_iads
                Cx : individual of the population
                Returns : float
                -------
                '''
                return np.linalg.norm(np.array((radar.X, radar.Y)) -
                                  np.array((Cx[i][0], Cx[i][1])))

            # For each jammer of the individual, its coordinates and radar targeted are modified according to the same method
            for i in range(self.nb_jammers):
                radars = sensor_iads.list.copy()
                C1[i].append(P1[i][0])
                C1[i].append(P2[i][1])
                closest_radar = min(radars, key=lambda radar: distance(i, radar, C1))
                C1[i].append(closest_radar)
                radars.remove(closest_radar)
                C2[i].append(P2[i][0])
                C2[i].append(P1[i][1])
                closest_radar = min(radars, key=lambda radar: distance(i, radar, C2))
                C2[i].append(closest_radar)
                radars.remove(closest_radar)
        else:
            C1 = P1.copy()
            C2 = P2.copy()
        return C1,C2

    def mutation(self, individual):
        new_vector = [[] for i in range(self.nb_jammers)]
        maxX = max([radar.X for radar in sensor_iads.list])# Boundaries are set because it is the only method
                                                           # that can give solution outside the workspace
        if random.random() < self.chance_to_mutate:
            for i in range(self.nb_jammers):
                for j in range(len(individual[i]) - 1):
                    strnb = str(individual[i][j])
                    strnb = strnb[::-1]# The method switches the first and the last digits of the coordinates of the jammers
                    if j == 0 and int(strnb) > maxX:# to not go outside the boundaries
                        new_vector[i].append(2*maxX - int(strnb))
                    else:
                        new_vector[i].append(int(strnb))
                new_vector[i].append(individual[i][-1])
        else:
            new_vector = individual.copy()
        return new_vector

    def fitness(self, genome):

        # Updating the battle space context with the DNA of an individual by overwriting the characteristics of the objects Jammer
        for i, jammer in enumerate(Jammer.list):
            jammer.update(genome[i][0], genome[i][1])
            jammer.targets(genome[i][2])

        # Calculation of the different objective function
        objective_function_1_value = corridor_width(self.aircraft_secured, self.security_width) - any_detection(40)
        objective_function_2_value = safe_distance()
        objective_function_3_value = time_constraint(self.X0, self.Y0)

        # Resetting the allocations after the calculation of the fitness
        for radar in sensor_iads.list:
            radar.jammers_targeting = []
        return (objective_function_1_value, objective_function_2_value, objective_function_3_value)


    def next_generation(self):
        graded_individuals = self.select_individuals()
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(graded_individuals, 2)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            new_population.append(child1)
            new_population.append(child2)
        self.population = new_population

    def run(self, generations_count):
        i = 0
        j = 0
        while i < generations_count and j < 7:
            i += 1
            self.next_generation()
        #     best_individual_fitness = self.fitness(self.best_individual)
        #     if best_individual_fitness[0] <= 0 and i >= 10:# In the case the initial population wasn't good enough and it doesn't converge
        #         self.population = [self.create_random_genome() for _ in range(self.population_size)]
        #         i = 0
        #         j += 1
        # if j == 7:
        #     print("The algorithm has been ended prematurely because it wasn't able to find a corridor")
        fronts, def_pop = self.fast_non_dominated_sorting([[individual, self.fitness(individual)] for individual in self.population])
        first_front = [def_pop[individual] for individual in fronts[0]]
        #Find the best individual in the final population
        return first_front
