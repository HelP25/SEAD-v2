import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from SEAD_v2.Assets import *
from SEAD_v2.Optimise.Safe_corridor import *
import random
from scipy.stats.qmc import LatinHypercube


class MultiObjGeneticAlgorithm:
    def __init__(self, X0, Y0, nb_jammers, aircraft_secured, security_width, population_size, chance_to_mutate,
                 chance_to_crossover):
        self.population_size = population_size
        self.chance_to_mutate = chance_to_mutate
        self.chance_to_crossover = chance_to_crossover
        self.nb_jammers = nb_jammers
        self.aircraft_secured = aircraft_secured
        self.security_width = security_width
        self.best_individuals = None
        self.first_front = None
        self.first_front_history = []
        self.X0 = X0
        self.Y0 = Y0
        self.opt_line = min([radar.X for radar in sensor_iads.list]) - sensor_iads.list[0].get_detection_range(
            aircraft_secured)
        self.population = [self.create_random_genome() for _ in
                           range(population_size)]  # Creation of the initial population
        # self.population.append([[569, 631, next(radar for radar in sensor_iads.list if radar.name == 'radar4')],
        #                         [609, 587, next(radar for radar in sensor_iads.list if radar.name == 'radar3')],
        #                         [530, 505, next(radar for radar in sensor_iads.list if radar.name == 'radar4')],
        #                         [310, 480, next(radar for radar in sensor_iads.list if radar.name == 'radar2')],
        #                         [370, 450, next(radar for radar in sensor_iads.list if radar.name == 'radar2')]])
        for _ in range(self.nb_jammers):  # creation of our jammers
            jammer = Jammer(self.X0, self.Y0)

    def create_random_genome(self):
        '''
        Function that aims at creating a random individual for the creation of a population
        Returns: an individual also defined by its genome
        -------

        '''

        # Creation of the borders within the individuals are generated
        maxX = max([radar.X for radar in sensor_iads.list])
        maxY = max([radar.Y for radar in sensor_iads.list])
        minY = min([radar.Y for radar in sensor_iads.list])

        # Generation of an individual using the LHS
        sample = LatinHypercube(3)
        genome = sample.random(self.nb_jammers).tolist()  # List of tuples of three random numbers between 0 and 1
        for i in range(self.nb_jammers):
            genome[i][0] = int(genome[i][0] * maxX)  # To have our random numbers between the boundaries
            genome[i][1] = int(minY + genome[i][1] * (maxY - minY))
            genome[i][2] = random.choice(sensor_iads.list)
        return genome

    def hierarchy(self, fronts, front_level, power):
        """
        This function aims at adding a hierarchy to the objective functions by boosting the individuals with a good
        first objective function
        Parameters
        ----------
        fronts: list of the individuals sorted by Pareto fronts
        power: the number of fronts the individual must be moved forward

        Returns: the new sorted individuals list
        -------

        """
        # All the interesting individuals of the power-first fronts must be moved to the first front
        for i in range(1, min([power, len(fronts)])):
            for ind in fronts[i]:  # Checking all the individuals of the front
                if ind[0] > 0:  # If its first objective function has an interesting value
                    fronts[0].append(ind)  # Adding the individual to the first front
                    fronts[i].remove(ind)  # Removing it from its current front
                    front_level[ind] = 1
        # The others must be moved power fronts above
        for i in range(power, len(fronts)):
            for ind in fronts[i]:
                if ind[0] > 0:
                    fronts[i - power].append(ind)
                    fronts[i].remove(ind)
                    front_level[ind] = front_level[ind] - power
        return fronts, front_level

    def select_individuals(self, population):
        """
        Function that selects the individuals from a list to create a population using elitism
        Parameters
        ----------
        population: list of individuals with a greater size than the one of the population we want to create

        Returns: list of the most interesting individuals from the list in parameter of the function
        -------

        """
        # Perform non-dominated sorting
        fronts, definition_pop, front_level = self.fast_non_dominated_sorting(
            [[individual, self.fitness(individual)] for individual in population])

        fronts, front_level = self.hierarchy(fronts, front_level, 2)  # Applying the hierarchy aspect

        self.best_individuals = [definition_pop[ind] for ind in fronts[0] if ind[0] > 0]

        # Calculates the crowding distances
        crowding_distances = self.crowding_distance(fronts)

        # Selection
        selected_individuals = []
        i = 0
        ind = False
        # The selected individuals is first filled up with as many whole fronts as possible
        while i < len(fronts) and ind == False:
            # If, by adding the next front, the size of the selected individuals get bigger than the size of the
            # population, the process must be stopped
            if len(selected_individuals) + len(fronts[i]) > self.population_size:
                ind = True
            else:
                # Add all the individuals from the front i in the selected population
                selected_individuals.extend([definition_pop[individual] for individual in fronts[i]])
            i += 1
        # If the size of the selected population does not reach the size of the previous population, the individuals of
        # the next front must be selected according to their crowding distance
        if len(selected_individuals) < self.population_size:
            front = fronts[i - 1]  # Because at the end of the while loop, i is increased
            front = sorted(front, key=lambda x: crowding_distances[x], reverse=True)  # Sorting the individuals by
            # their crowding distance
            selected_individuals.extend(definition_pop[individual]
                                        for individual in front[:self.population_size - len(selected_individuals)])
        return selected_individuals

    def crowding_distance(self, fronts):
        """
        Calculates the crowding distance for all the Pareto fronts.
        :param fronts: A 3D list
        :return: A dictionary with the individuals defined by their fitness and the crowding distance related
        """

        # Initialize crowding distances
        distances = {}

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
                    for j in range(1, front_size - 1):  # Distance related to the neighbours normalised
                        distances_i[j] = (sorted_front[i][j + 1][i] - sorted_front[i][j - 1][i]) / (
                                    sorted_front[i][-1][i] - sorted_front[i][0][i])
                    # Add the crowding distances to every individual
                    for j in range(front_size):
                        if front[j] in distances:
                            distances[front[j]] += distances_i[j]  # Distances can be added as they are normalised
                        else:
                            distances[front[j]] = distances_i[j]  # Creating a new definition in the dictionary
        return distances

    def fast_non_dominated_sorting(self, population):
        """
        Function that performs a fast non dominated sorting on a list of individuals regarding their fitness
        Parameters
        ----------
        population: list of individuals defined by a list containing their genome and fitness

        Returns: a list of the individuals sorted out in the different fronts
                 the dictionary that relates the fitness of an individual to its genome
        -------

        """
        # Initialisation
        fronts = [[]]  # Creation of the fronts list
        S = [[] for _ in range(len(population))]  # Creation of the list of set of solutions dominated by another one
        domination_count = [0 for _ in range(len(population))]
        definition_pop = {individual[1]: individual[0] for individual in population}
        pop_fitness = [individual[1] for individual in population]
        front_level = {}

        # Determination of dominance relations
        for p in range(len(pop_fitness)):
            for q in range(len(pop_fitness)):
                if all(pop_fitness[q][k] <= pop_fitness[p][k] for k in range(3)) and any(pop_fitness[q][k] < pop_fitness[p][k] for k in range(3)):
                    if q not in S[p]:
                        S[p].append(q)  # If p dominates q, add q to the set of solutions dominated by p
                elif all(pop_fitness[p][k] <= pop_fitness[q][k] for k in range(3)) and any(pop_fitness[p][k] < pop_fitness[q][k] for k in range(3)):
                    domination_count[p] += 1  # Increment the domination counter of p
            if domination_count[p] == 0:
                front_level[pop_fitness[p]] = 1
                if p not in fronts[0]:
                    fronts[0].append(p)  # Then p belongs to the first front

        # Creation of the fronts
        k = 0  # Initialize the front counter
        while fronts[k]:
            Q = []  # Used to store the members of the next front
            for p in fronts[k]:
                for q in S[p]:
                    domination_count[q] -= 1  # If the relation of dominance from the individuals of the previous
                    # front is withdrawn
                    if domination_count[q] == 0:  # If q does not have any dominating individual anymore
                        front_level[pop_fitness[q]] = k + 2
                        if q not in Q:
                            Q.append(q)  # Then q belongs to the next front
            k += 1
            fronts.append(Q)  # Creation of the next front
        for front in fronts:
            for i, ind in enumerate(front):
                front[i] = pop_fitness[ind]
        self.first_front = fronts[0]
        return fronts[:-1], definition_pop, front_level

    def distance(self, radar, jammer):
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
                              np.array((jammer[0], jammer[1])))

    def crossover(self, P1, P2):
        """
        Function that performs the crossover method
        Parameters
        ----------
        P1: an individual defined by its genome and seen as the first parent
        P2: an individual defined by its genome and seen as the second parent

        Returns: two individuals defined by their genome and seen as the children
        -------

        """

        # The crossover is only be done with a certain probability: chance_to_crossover
        if random.random() < self.chance_to_crossover:
            # Initialisation
            C1 = [[] for i in range(self.nb_jammers)]
            C2 = [[] for i in range(self.nb_jammers)]

            # For each jammer of the individual, its coordinates and radar targeted are modified according to the same method
            for i in range(self.nb_jammers):
                radars = sensor_iads.list.copy()
                C1[i].append(P1[i][0])
                C1[i].append(P2[i][1])  # The ordinates of the jammer i from the two parents are being exchanged
                closest_radar = min(radars, key=lambda radar: self.distance(radar, C1[i]))  # And the new jammer created
                C1[i].append(closest_radar)  # now targets the closest radar
                radars.remove(closest_radar)  # To avoid focusing only one radar
                C2[i].append(P2[i][0])
                C2[i].append(P1[i][1])
                closest_radar = min(radars, key=lambda radar: self.distance(radar, C2[i]))
                C2[i].append(closest_radar)
                radars.remove(closest_radar)
            random.shuffle(C1)  # The jammers are now shuffled into the genome to avoid having a useless crossover
            random.shuffle(C2)  # by doing it twice on the same two individuals
        else:
            C1 = P1.copy()
            C2 = P2.copy()
        return C1, C2

    def mutation(self, individual):
        """
        Function that performs the mutation method
        Parameters
        ----------
        individual: an individual defined by his genome

        Returns: an individual mutated
        -------

        """
        # Initialisation
        new_vector = [[] for i in range(self.nb_jammers)]
        maxX = max([radar.X for radar in sensor_iads.list])  # Boundaries are set because it is the only method
        # that can give solution outside the workspace
        if random.random() < self.chance_to_mutate:
            # Updating the battle space context with the DNA of an individual by overwriting the characteristics of the objects Jammer
            for i, jammer in enumerate(Jammer.list):
                jammer.update(individual[i][0], individual[i][1])
                jammer.targets(individual[i][2])

            # Calculation of the center of the corridor if there is one or at least where there is a sweet spot
            center_corridor = find_corridor(self.aircraft_secured, self.security_width)[1]

            # Calculation if the new position of the jammers
            for i in range(self.nb_jammers):
                # If the jammers are on the left side of the optimal line
                if individual[i][0] < self.opt_line:
                    # The jammers are moved toward the point center_corridor of a random distance according to
                    # an affine function
                    heading = (center_corridor[1] - individual[i][1]) / (center_corridor[0] - individual[i][0])
                    new_abscissa = int(
                        individual[i][0] + (center_corridor[0] - individual[i][0]) * random.gauss(0.5, 0.17))
                    new_ordinate = int(heading * (new_abscissa - center_corridor[0]) + center_corridor[1])
                    individual[i][0] = new_abscissa
                    individual[i][1] = new_ordinate
                    # To keep a bit of diversity the target is chosen as the closest one since the move is random
                    closest_radar = max(sensor_iads.list, key=lambda radar: self.distance(radar, individual[i]))
                    individual[i][2] = closest_radar

            # Resetting the allocations after the calculation of the fitness
            for radar in sensor_iads.list:
                radar.jammers_targeting = []

        return individual

    def old_mutation(self, individual):
        new_vector = [[] for i in range(self.nb_jammers)]
        maxX = max([radar.X for radar in sensor_iads.list])  # Boundaries are set because it is the only method
        # that can give solution outside the workspace
        if random.random() < self.chance_to_mutate:
            for i in range(self.nb_jammers):
                for j in range(len(individual[i]) - 1):
                    strnb = str(individual[i][j])
                    strnb = strnb[
                            ::-1]  # The method switches the first and the last digits of the coordinates of the jammers
                    if j == 0 and int(strnb) > maxX:  # to not go outside the boundaries
                        new_vector[i].append(2 * maxX - int(strnb))
                    else:
                        new_vector[i].append(int(strnb))
                new_vector[i].append(individual[i][-1])
        else:
            new_vector = individual.copy()
        return new_vector

    def fitness(self, genome):
        """
        Calculates the value of the objective functions for an individual
        Parameters
        ----------
        genome: genome of an individual

        Returns: a tuple of values of the objective functions
        -------

        """

        # Updating the battle space context with the DNA of an individual by overwriting the characteristics of the objects Jammer
        for i, jammer in enumerate(Jammer.list):
            jammer.update(genome[i][0], genome[i][1])
            jammer.targets(genome[i][2])

        # Calculation of the different objective function
        width = find_corridor(self.aircraft_secured, self.security_width)[0]
        objective_function_1_value = width - any_detection(40)
        objective_function_2_value = safe_distance(self.opt_line) * objective_function_1_value
        objective_function_3_value = time_constraint(self.X0, self.Y0)

        # Resetting the allocations after the calculation of the fitness
        for radar in sensor_iads.list:
            radar.jammers_targeting = []
        return (objective_function_1_value, objective_function_2_value, objective_function_3_value)

    def next_generation(self):
        """
        Calculates the next generation of the population
        -------

        """
        fronts, def_pop, front_level = self.fast_non_dominated_sorting(
            [[individual, self.fitness(individual)] for individual in self.population])
        fronts, front_level = self.hierarchy(fronts, front_level, 2)
        distances = self.crowding_distance(fronts)
        mating_pool = []
        for _ in range(self.population_size // 2):
            mating_pool.append(self.binary_tournament_selection(front_level, distances, def_pop))
        new_population = self.population.copy()
        while len(new_population) < self.population_size * 2:
            parent1, parent2 = mating_pool[random.randint(0, len(mating_pool) - 1)]
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.old_mutation(child1)
            child2 = self.old_mutation(child2)
            if child1 == None and child2 == None:
                child1 = self.old_mutation(parent1)
                child2 = self.old_mutation(parent2)
            new_population.append(child1)
            new_population.append(child2)
        self.population = self.select_individuals(new_population)
        # print(len(self.population))

    def binary_tournament_selection(self, front_level, distances, def_pop):
        ind_in_tournament = random.choices(list(front_level.keys()), k=4)
        max_level = max(list(front_level.values()))
        tournament = {}
        for ind in ind_in_tournament:
            if distances[ind] == np.inf:
                distances[ind] = 3
            tournament[ind] = max_level - front_level[ind] + distances[ind] / 3.01
        selection = sorted(ind_in_tournament, key=lambda ind: tournament[ind], reverse=True)
        return [def_pop[ind] for ind in selection[:2]]

    def next_generation_old(self):
        """
        Calculates the next generation of the population
        -------

        """
        fronts, def_pop, front_level = self.fast_non_dominated_sorting(
            [[individual, self.fitness(individual)] for individual in self.population])
        fronts, front_level = self.hierarchy(fronts, front_level, 2)
        distances = self.crowding_distance(fronts)
        mating_pool = [self.binary_tournament_selection_old(front_level, distances, def_pop) for _ in
                       range(self.population_size)]
        new_population = self.population.copy()
        while len(new_population) < self.population_size * 2:
            parent1, parent2 = random.sample(mating_pool, 2)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.old_mutation(child1)
            child2 = self.old_mutation(child2)
            if child1 == None and child2 == None:
                child1 = self.old_mutation(parent1)
                child2 = self.old_mutation(parent2)
            new_population.append(child1)
            new_population.append(child2)
        self.population = self.select_individuals(new_population)
        print(len(self.population))

    def binary_tournament_selection_old(self, front_level, distances, def_pop):
        individual1, individual2 = random.choices(list(front_level.keys()), k=2)

        if front_level[individual1] > front_level[individual2]:
            return def_pop[individual1]
        elif front_level[individual1] == front_level[individual2]:
            if distances[individual1] >= distances[individual2]:
                return def_pop[individual1]
            else:
                return def_pop[individual2]
        else:
            return def_pop[individual2]


    def run(self, generations_count):
        i = 0
        j = 0
        first_time = np.inf
        first_time_good = None

        while i < generations_count and j < 7:
            i += 1
            print(f'{i}/{generations_count}')
            self.next_generation()
            #self.next_generation_old()
            fronts, def_pop, front_level = self.fast_non_dominated_sorting(
                [[individual, self.fitness(individual)] for individual in self.population])
            print(len(list(def_pop.keys())))
            self.first_front_history.append(self.first_front)
            for ind in self.first_front:
                if ind[0] > 0 and first_time == np.inf:
                    first_time = i
                    first_time_good = def_pop[ind]
        #     best_individual_fitness = self.fitness(self.best_individual)
        #     if best_individual_fitness[0] <= 0 and i >= 10:# In the case the initial population wasn't good enough and it doesn't converge
        #         self.population = [self.create_random_genome() for _ in range(self.population_size)]
        #         i = 0
        #         j += 1
        # if j == 7:
        #     print("The algorithm has been ended prematurely because it wasn't able to find a corridor")
        fronts, def_pop, front_level = self.fast_non_dominated_sorting(
            [[individual, self.fitness(individual)] for individual in self.population])
        length = len(self.first_front)
        first_front = [def_pop[individual] for individual in self.first_front if individual[0] > 0]
        # Find the best individual in the final population
        return first_front, length, first_time, self.first_front_history
