from SEAD_v2.Assets import *
from SEAD_v2.Optimise.Safe_corridor import *
import random

class GeneticAlgorithm:
    def __init__(self, nb_jammers, aircraft_secured, security_width, population_size, chance_to_mutate):
        self.population_size = population_size
        self.chance_to_mutate = chance_to_mutate
        self.nb_jammers = nb_jammers
        self.aircraft_secured = aircraft_secured
        self.security_width = security_width
        self.population = [self.create_random_genome() for _ in range(population_size)]
        for i in range(self.nb_jammers): # creation of our jammers
            jammer = Jammer(0,0)


    def create_random_genome(self):
        genome = []
        for _ in range(self.nb_jammers):
            maxX = max([radar.X for radar in sensor_iads.list])
            maxY = max([radar.Y for radar in sensor_iads.list])
            minY = min([radar.Y for radar in sensor_iads.list])
            X = random.randrange(0, maxX)
            Y = random.randrange(minY, maxY)
            genome.append([X, Y, random.choice(sensor_iads.list)])
        return genome

    def select_individuals(self):
        total_fitness = sum(self.fitness(individual) for individual in self.population)
        min_fitness = min(self.fitness(individual) for individual in self.population)
        graded_individuals = random.choices(self.population,
                                            weights=[abs((self.fitness(individual) - min_fitness)/ total_fitness) for individual in self.population],
                                            k=self.population_size) # can draw many times the same individual

        return graded_individuals

    def crossover(self, P1, P2):
        C1 = [[] for i in range(self.nb_jammers)]
        C2 = [[] for i in range(self.nb_jammers)]
        for i in range(self.nb_jammers):
            C1[i].append(P1[i][0])
            C1[i].append(P2[i][1])
            radars = sensor_iads.list.copy()
            radars.remove(P1[i][-1])
            C1[i].append(random.choice(radars))
            C2[i].append(P2[i][0])
            C2[i].append(P1[i][1])
            radars = sensor_iads.list.copy()
            radars.remove(P2[i][-1])
            C2[i].append(random.choice(radars))
        return C1,C2

    def mutation(self, individual):
        new_vector = [[] for i in range(self.nb_jammers)]
        if random.random() < self.chance_to_mutate:
            for i in range(self.nb_jammers):
                for j in range(len(individual[i]) - 1):
                    strnb = str(individual[i][j])
                    strnb = strnb[::-1]
                    new_vector[i].append(int(strnb))
                new_vector[i].append(individual[i][-1])
        else:
            new_vector = individual.copy()
        return new_vector

    def fitness(self, genome):
        for i, jammer in enumerate(Jammer.list):
            jammer.update(genome[i][0], genome[i][1])
            jammer.targets(genome[i][2])
        fitness = corridor_width(self.aircraft_secured, self.security_width) - any_detection(30)
        for radar in sensor_iads.list:
            radar.jammers_targeting = []
        return fitness



    def next_generation(self):
        graded_individuals = self.select_individuals()
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(graded_individuals, 2)# Once the individual is picked, it is removed from the
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            new_population.append(child1)
            new_population.append(child2)
        self.population = new_population

    def run(self, generations_count):
        for _ in range(generations_count):
            self.next_generation()
        # Find the best individual in the final population
        best_individual = max(self.population, key=lambda individual: self.fitness(individual))
        return best_individual, self.fitness(best_individual)



