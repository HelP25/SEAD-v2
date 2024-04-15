import matplotlib.pyplot as plt
import numpy as np
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
plt.close("all")
ga = MultiObjGeneticAlgorithm(0, 300, 5, striker, 2, 50, 0.1, 1)

first_front, length, first_time, first_front_history = ga.run(50)
plt.close("all")
print(f"First time that we have a good solution: {first_time}")
print(f'There are {len(first_front)} solutions')

new_first_front = []
for ind in first_front:
    for i, jammer in enumerate(Jammer.list):
        jammer.update(ind[i][0], ind[i][1])
        jammer.targets(ind[i][2])
    if any_detection(5) == 1:
        new_first_front.append(ind)
    for radar in sensor_iads.list:
        radar.jammers_targeting = []

first_front = new_first_front.copy()



for k in range(len(first_front)):
    plt.subplot(len(first_front)//2 +1, 2, k+1 )
    for i, jammer in enumerate(Jammer.list):
        jammer.update(first_front[k][i][0], first_front[k][i][1])
        jammer.targets(first_front[k][i][2])
        plt.plot(jammer.X, jammer.Y, 'r4', markersize=10)
        plt.text(jammer.X, jammer.Y, jammer.name)
    for radar in sensor_iads.list:
        plt.plot(radar.X, radar.Y, 'bs', markersize=10)
        plt.text(radar.X, radar.Y, radar.name)
        radar.outline_detection(striker)
    fitness = ga.fitness(first_front[k])
    print(f"The fitness of the {k+1}th solution is : {fitness}")
print(f'length of the first front: {length}')
plt.show()

pop = ga.population
xs = np.array([[j[0] for j in idx] for idx in pop]).flatten()
ys = np.array([[j[1] for j in idx] for idx in pop]).flatten()

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(xs, ys)
plt.show()

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
        ax.set_xlim(-300, 100)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-750, -500)
        return ax

    ani = animation.FuncAnimation(fig, update, frames=len(optimizer.first_front_history), interval=500)
    plt.show()

animate(ga)
"""
striker = aircraft(200, 302)
radar1 = sensor_iads(600, 300)
radar2 = sensor_iads(700, 400)
radar3 = sensor_iads(650, 550)
radar4 = sensor_iads(550, 650)

jammer1 = Jammer(569, 631)
jammer2 = Jammer(609, 587)
jammer3 = Jammer(505, 530)
jammer4 = Jammer(310, 480)
jammer5 = Jammer(370, 450)
jammer1.targets(radar4)
jammer2.targets(radar3)
jammer3.targets(radar4)
jammer4.targets(radar2)
jammer5.targets(radar2)

for jammer in Jammer.list:
    plt.plot(jammer.X, jammer.Y, 'r4', markersize=10)
    plt.text(jammer.X, jammer.Y, jammer.name)
for radar in sensor_iads.list:
    plt.plot(radar.X, radar.Y, 'bs', markersize=10)
    plt.text(radar.X, radar.Y, radar.name)
    radar.outline_detection(striker)
print(find_corridor(striker, 2))
plt.show()
"""