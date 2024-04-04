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
            if radar.detection(aerial_vehical): # Check for any aircraft to be detected by any radar
                total += weight
    return total


def safe_distance(x):
    """
    Objective function that calculates the inverse of the variation coefficient
    Parameters

    Returns: the inverse of the variation coefficient
    -------

    """
    # sum = 0
    # for jammer in Jammer.list:
    #     sum += (jammer.X - x)**2 / (jammer.target.jamming_power_1jammer(jammer, jammer) *1e10)
    # return -1 * sum
    ranges = []
    # Creation of a list with all the ranges between the jammers and the radars
    for radar in sensor_iads.list:
        for jammer in Jammer.list:
            ranges += [radar.range(jammer)]
    mean = np.mean(ranges)  # Calculation of the mean
    s = np.std(ranges, ddof=1)  # Calculation of the standard deviation
    return mean/s

def time_constraint(X0, Y0):
    """
    Calculates the time constraint as a distance
    Parameters
    ----------
    X0: initial abscissa
    Y0: initial ordinate

    Returns
    -------

    """
    distance = []
    # Creation of a list with all the distances between the initial position and the final one of every jammer
    for aerial_vehicule in aircraft.list:
        distance += [np.linalg.norm(np.array([X0, Y0]) - np.array(aerial_vehicule.coordinates))]
    time = - max(distance)
    return time