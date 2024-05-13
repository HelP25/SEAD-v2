import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from SEAD_v2.Optimise import *
from SEAD_v2.Assets import *

plt.close('all')

# Test

"""
radar1 = sensor_iads(400, 400)
radar2 = sensor_iads(400, 600)
radar3 = sensor_iads(400, 700)
jammer1 = Jammer(330, 510)
jammer2 = Jammer(295, 530)
jammer3 = Jammer(150,330)
radar1.targeted_by([])
radar2.targeted_by([jammer2])
radar3.targeted_by([jammer3])
for radar in sensor_iads.list:
    radar.outline_detection(jammer2)
print(f"The width is equal to: {find_corridor(jammer1, 2)}")

plt.show()
"""
#  Scénario
"""
radar1 = sensor_iads(600, 300)
radar2 = sensor_iads(700, 400)
radar3 = sensor_iads(650, 550)
radar9 = sensor_iads(750, 550)
radar4 = sensor_iads(550, 650)
radar5 = sensor_iads(600, 750)
radar6 = sensor_iads(600, 850)
radar7 = sensor_iads(600, 950)
radar8 = sensor_iads(600, 1050)


striker = aircraft(573,702)
jammer1 = Jammer(300, 500)
jammer2 = Jammer(436, 650)
jammer3 = Jammer(490, 725)
jammer4 = Jammer(450, 750)
jammer5 = Jammer(490, 500)

# Test
#radar3.targeted_by([jammer1])
#radar4.targeted_by([jammer2])
#radar5.targeted_by([jammer3])
#radar6.targeted_by([jammer4])
#radar9.targeted_by([jammer5])



print(corridor_width(striker, 2))
#print(any_detection(20))
#print(means_cost())
#print(safe_distance())
#print(time_constraint(100, 400))
#plt.legend()

"""

#  Test Single objective
"""
striker = aircraft(200, 302)
radar1 = sensor_iads(600, 300)
radar2 = sensor_iads(700, 400)
radar3 = sensor_iads(650, 550)
radar4 = sensor_iads(550, 650)

ga = SingleObjGeneticAlgorithm(2, striker, 2, 100, 0.1, 0.2)

solution, fitness = ga.run(50)
plt.close("all")

for i, jammer in enumerate(Jammer.list):
    jammer.update(solution[i][0], solution[i][1])
    jammer.targets(solution[i][2])
    plt.plot(jammer.X, jammer.Y, 'r4', markersize=10)
    plt.text(jammer.X, jammer.Y, jammer.name)
    print(f'{jammer.name} targets {jammer.target.name}')
for radar in sensor_iads.list:
    plt.plot(radar.X, radar.Y, 'bs', markersize=10)
    plt.text(radar.X, radar.Y, radar.name)
    radar.outline_detection(striker)
    print(f'{radar.name} is targeted by {[jammer.name for jammer in radar.jammers_targeting]}')
print(f"The fitness is equal to : {fitness}")

plt.legend()
plt.show()

"""
#  Test Multi objective

striker = aircraft(200, 302)
radar1 = sensor_iads(600, 300)
radar2 = sensor_iads(700, 400)
radar3 = sensor_iads(650, 550)
radar4 = sensor_iads(550, 650)
radar5 = sensor_iads(600, 750)
radar6 = sensor_iads(700, 850)

ga = MultiObjGeneticAlgorithm(0, 300, 8, striker, 2, 75, 0.2, 0.9)

first_front, length, first_time, first_front_history = ga.run(50)

results_analysis(ga, first_front, length, first_time, first_front_history, striker)


