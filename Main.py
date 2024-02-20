import matplotlib.pyplot as plt
import numpy as np
from Assets import *
from Optimise import *

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

#Test
radar3.targeted_by([jammer1])
radar4.targeted_by([jammer2])
radar5.targeted_by([jammer3])
radar6.targeted_by([jammer4])
radar9.targeted_by([jammer5])



print(corridor_width(striker, 2))
print(any_detection(20))
print(means_cost())
print(safe_distance())
print(time_constraint(100, 400))
#plt.legend()
plt.show()
"""


