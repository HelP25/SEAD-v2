import numpy as np

from Assets import *
from Optimise.Safe_corridor.Corridor_width_v2 import *
from Optimise.Safe_corridor import *
import random
from scipy.stats.qmc import LatinHypercube


class SingleObjGeneticAlgorithm:
    best_individual = None
    def __init__(self, nb_jammers, aircraft_secured, security_width, population_size, chance_to_mutate, chance_to_crossover):
        self.population_size = population_size
        self.chance_to_mutate = chance_to_mutate
        self.chance_to_crossover = chance_to_crossover
        self.nb_jammers = nb_jammers
        self.aircraft_secured = aircraft_secured
        self.security_width = security_width
        self.population = [self.create_random_genome() for _ in range(population_size)]
        for _ in range(self.nb_jammers):    # creation of our jammers
            jammer = Jammer(0,0)


    def create_random_genome(self):
        '''
        Function that aims at creating a random individual for the creation of a population
        Returns: an individual also defined by its genome
        -------

        '''

        # Creation of the borders within the individuals are generated
        maxX = max([radar.X for radar in sensor_iads.list]) + sensor_iads.list[0].get_detection_range(self.aircraft_secured)
        maxY = max([radar.Y for radar in sensor_iads.list])
        minY = min([radar.Y for radar in sensor_iads.list])

        # Generation of an individual using the LHS
        sample = LatinHypercube(3)
        genome = sample.random(self.nb_jammers).tolist()    # List of tuples of three random numbers between 0 and 1
        for i in range(self.nb_jammers):
            genome[i][0] = int(genome[i][0] * maxX) # Abscissa
            genome[i][1] = int(minY + genome[i][1] * (maxY - minY)) # Ordinate
            genome[i][2] = random.choice(sensor_iads.list)  # The target

        # Generation of an individual simply using random
        # genome = []
        # for _ in range(self.nb_jammers):
        #     maxX = max([radar.X for radar in sensor_iads.list])
        #     maxY = max([radar.Y for radar in sensor_iads.list])
        #     minY = min([radar.Y for radar in sensor_iads.list])
        #     X = random.randrange(0, maxX)
        #     Y = random.randrange(minY, maxY)
        #     genome.append([X, Y, random.choice(sensor_iads.list)])

        return genome

    def select_individuals(self):
        """
         Function that selects the individuals from a list to create a population using elitism
         Parameters

         Returns: list of the most interesting individuals from the list in parameter of the function
         -------

         """
        # Calculation of the sum of all the fitness to be able to calculate a probability for every individual to be selected
        total_fitness = sum(self.fitness(individual) for individual in self.population)
        min_fitness = min(self.fitness(individual) for individual in self.population)

        # Selection of the individuals according to their probability, the roulette selection.
        graded_individuals = random.choices(self.population,
                                            weights=[(self.fitness(individual) - min_fitness + 1)/
                                                     (total_fitness - (min_fitness - 1)*len(self.population))
                                                     for individual in self.population],
                                            k=self.population_size) # one individual with a high probability can be selected several times

        return graded_individuals

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
        # The crossover is only done with a certain probability: chance_to_crossover
        if random.random() < self.chance_to_crossover:

            # Initialisation of the children
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
                C1[i].append(P2[i][1])  # The ordinates of the jammer i from the two parents are being exchanged
                closest_radar = min(radars, key=lambda radar: distance(i, radar, C1))  # And the new jammer created
                C1[i].append(closest_radar)                                            # now targets the closest radar
                radars.remove(closest_radar)    # To avoid focusing only one radar
                C2[i].append(P2[i][0])
                C2[i].append(P1[i][1])
                closest_radar = min(radars, key=lambda radar: distance(i, radar, C2))
                C2[i].append(closest_radar)
                radars.remove(closest_radar)
            random.shuffle(C1)  # The jammers are now shuffled into the genome to avoid having a useless crossover
            random.shuffle(C2)  # by doing it twice on the same two individuals
        else:   # When the probability is not verified
            C1 = P1.copy()
            C2 = P2.copy()
        return C1,C2

    def mutation(self, individual):
        """
        Function that performs the mutation method
        Parameters
        ----------
        individual: an individual defined by his genome

        Returns: an individual mutated
        -------

        """
        new_vector = [[] for i in range(self.nb_jammers)]
        maxX = max([radar.X for radar in sensor_iads.list]) # Boundaries are set because it is the only method
                                                            # that can give solution outside the workspace
        if random.random() < self.chance_to_mutate:
            for i in range(self.nb_jammers):    # Each jammer's coordinates are modified in the individual
                for j in range(len(individual[i]) - 1):
                    strnb = str(individual[i][j])
                    strnb = strnb[::-1] # The method switches the first and the last digits of the coordinates of the jammers
                    if j == 0 and int(strnb) > maxX:    # to not go outside the boundaries
                        new_vector[i].append(2*maxX - int(strnb))
                    else:
                        new_vector[i].append(int(strnb))
                new_vector[i].append(individual[i][-1])
        else:
            new_vector = individual.copy()
        return new_vector

    def fitness(self, genome):
        """
        Calculates the value of the objective function for an individual
        Parameters
        ----------
        genome: genome of an individual

        Returns: the value of the objective function
        -------

        """
        # Updating the battle space context with the DNA of an individual by overwriting the characteristics of the objects Jammer
        for i, jammer in enumerate(Jammer.list):
            jammer.update(genome[i][0], genome[i][1])
            jammer.targets(genome[i][2])

        fitness = corridor_width(self.aircraft_secured, self.security_width) - any_detection(40)

        # Resetting the allocations after the calculation of the fitness
        for radar in sensor_iads.list:
            radar.jammers_targeting = []
        return fitness


    def next_generation(self):
        """
        Calculates the next generation of the population
        -------

        """
        graded_individuals = self.select_individuals()
        self.best_individual = max(graded_individuals, key=lambda individual: self.fitness(individual))
        graded_individuals.remove(self.best_individual)# Keeping the best individual aside to avoid loosing it
                                                       # with a crossover or a mutation
        new_population = []
        # Creation of the next population
        while len(new_population) < self.population_size - 1:
            parent1, parent2 = random.sample(graded_individuals, 2) # Randomly choose 2 parents within the list of the
                                                                       # selected individuals
            child1, child2 = self.crossover(parent1, parent2)   # The children are bred from the parents
            child1 = self.mutation(child1)  # They can mutate
            child2 = self.mutation(child2)
            new_population.append(child1)   # They are added to the new population
            new_population.append(child2)
        new_population.append(self.best_individual)# Adding back the best individual
        self.population = new_population

    def run(self, generations_count):
        i = 0
        j = 0   # Used to avoid non convergence
        while i < generations_count and j < 7:
            i += 1
            self.next_generation()
            best_individual_fitness = self.fitness(self.best_individual)
            if best_individual_fitness <= 0 and i >= 10:# In the case the initial population wasn't good enough and it doesn't converge
                self.population = [self.create_random_genome() for _ in range(self.population_size)]
                i = 0
                j += 1
        if j == 7:
            print("The algorithm has been ended prematurely because it wasn't able to find a corridor")
        # Find the best individual in the final population
        return self.best_individual, best_individual_fitness



