import matplotlib.pyplot as plt
import numpy as np

from SEAD_v2.Optimise import *
from SEAD_v2.Assets import *

plt.close('all')

'''
#Test
radar1 = sensor_iads(400, 300)
EA18G = Jammer(210, 300)
print("EA18G is detected before jamming: ")
print(radar1.detection(EA18G))
plt.pause(2)
F16 = aircraft(362,342)
print("EA18G is detected while jamming: ")
print(radar1.detection(EA18G, [EA18G]))
print("F16 is detected while EA18G is jamming: ")
print(radar1.detection(F16,[EA18G]))
plt.pause(4)
EA18G.update(310)
print('F16 is detected while EA18G is jamming: ')
print(radar1.detection(F16,[EA18G]))
plt.pause(3)
EA6B = Jammer(330, 280)
print('F16 is detected while EA18G is jamming: ')
print(radar1.detection(F16,[EA18G, EA6B]))
'''

#Scénario
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

#Test

striker = aircraft(200,302)
radar1 = sensor_iads(600, 300)
radar2 = sensor_iads(700, 400)
radar3 = sensor_iads(650, 550)
radar4 = sensor_iads(550, 650)

ga = GeneticAlgorithm(4, striker, 2, 20, 0.2)
solution, fitness = ga.run(30)
plt.close("all")


for i, jammer in enumerate(Jammer.list):
    jammer.update(solution[i][0], solution[i][1])
    jammer.targets(solution[i][2])
    plt.plot(jammer.X, jammer.Y, 'r4', markersize=10, label=jammer.name)
    print(f'{jammer.name} targets {jammer.target.name}')
for radar in sensor_iads.list:
    plt.plot(radar.X, radar.Y, 'bs', markersize=10, label=radar.name)
    radar.get_detection_range(striker, radar.jammers_targeting)
    print(f'{radar.name} is targeted by {[jammer.name for jammer in radar.jammers_targeting]}')
print(f"La fitness est de: {fitness}")

plt.legend()

plt.show()



