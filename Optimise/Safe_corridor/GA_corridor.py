from SEAD_v2.Assets import *
from SEAD_v2.Optimise.Safe_corridor import *
import random

class GeneticAlgorithm:
    def __init__(self, nb_jammers, aircraft_secured, security_width, population_size, chance_to_mutate, graded_retain_percent, chance_retain_non_graded):
        self.population_size = population_size
        self.chance_to_mutate = chance_to_mutate
        self.graded_retain_percent = graded_retain_percent
        self.chance_retain_non_graded = chance_retain_non_graded
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
        tournament_size = 2
        tournament_individuals = random.sample(self.population, tournament_size)
        graded_individuals = sorted(tournament_individuals, key=lambda ind: ind.fitness, reverse=True)

        # Retain a percentage of the best individuals
        retain_count = int(self.graded_retain_percent * self.population_size / 100)
        graded_individuals = graded_individuals[:retain_count]

        # Retain a percentage of the non-graded individuals
        non_graded_count = int(self.chance_retain_non_graded * self.population_size / 100)
        non_graded_individuals = [ind for ind in self.population if ind not in graded_individuals]
        non_graded_individuals = sorted(non_graded_individuals, key=lambda ind: ind.fitness, reverse=True)
        graded_individuals.extend(non_graded_individuals[:non_graded_count])
        return graded_individuals

    def crossover(self, P1, P2):
        C1 = []
        C1.append(P1[0])
        C1.append(P2[1])
        radars = sensor_iads.list.copy()
        radars.remove(P1[-1])
        C1.append(random.choice(radars))
        C2 = []
        C2.append(P2[0])
        C2.append(P1[1])
        radars = sensor_iads.list.copy()
        radars.remove(P2[-1])
        C2.append(random.choice(radars))
        return C1,C2

    def mutation(self, jammer_vector):
        new_vector = []
        if random.random() < self.chance_to_mutate:
            for i in range(len(jammer_vector) -1):
                strnb = str(jammer_vector[i])
                strnb = strnb[::-1]
                new_vector.append(int(strnb))
            new_vector.append(jammer_vector[-1])
        else:
            new_vector = jammer_vector.copy()
        return new_vector

    def fitness(self, genome):
        for i in range(len(Jammer.list)):
            Jammer.list[i].update(genome[i][0], genome[i][0])
            Jammer.list[i].targets(genome[i][2])
        fitness = corridor_width(self.aircraft_secured, self.security_width) * any_detection(1)



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
        for _ in range(generations_count):
            self.next_generation()
        # Find the best individual in the final population
        best_individual = max(self.population, key=lambda ind: ind.fitness)
        print(f"Best individual: {best_individual.genome}, Fitness: {best_individual.fitness}")

#
# #Test
# striker = aircraft(573,702)
# radar1 = sensor_iads(600, 300)
# radar2 = sensor_iads(700, 400)
# radar3 = sensor_iads(650, 550)
# radar4 = sensor_iads(750, 550)
# radar5 = sensor_iads(550, 650)
#
# ga = GeneticAlgorithm(4, striker, 2, 100, 0.1, 65, 20)
# genome = ga.create_random_genome()
# ga.fitness(genome)
# print(len(Jammer.list))


