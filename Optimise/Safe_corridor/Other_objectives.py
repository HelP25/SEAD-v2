import matplotlib.pyplot as plt
import numpy as np
from SEAD_v2.Assets import *

def any_detection(weight):
    """
    :param allocation_matrix: the matrix of all the assets and their allocation
    :param weight: how much the value of the function is increased when the assets or not detected
    :return: the value of the objective function
    """
    total = 0
    for radar in sensor_iads.list:
        for aerial_vehical in aircraft.list:
            if radar.detection(aerial_vehical):
                total += weight
    return total

def means_cost():
    return len(sensor_iads.list) - len(Jammer.list)

def safe_distance(x):
    # sum = 0
    # for jammer in Jammer.list:
    #     sum += (jammer.X - x)**2 / (jammer.target.jamming_power_1jammer(jammer, jammer) *1e10)
    # return -1 * sum
    ranges = []
    for radar in sensor_iads.list:
        for jammer in Jammer.list:
            ranges += [radar.range(jammer)]
    mean = np.mean(ranges)
    s = np.std(ranges, ddof=1)
    return mean/s

def time_constraint(X0, Y0):
    distance = []
    for aerial_vehicule in aircraft.list:
        distance += [np.linalg.norm(np.array([X0, Y0]) - np.array(aerial_vehicule.coordinates))]
    time = - max(distance)
    return time