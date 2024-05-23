import matplotlib.pyplot as plt
import numpy as np
from Assets import *


# noinspection PyUnreachableCode
def corridor_width(aircraft_secured, security_width):
    '''
    Be careful, radars must be in reading order on the map, so that two radars created in succession are two neighboring
    radars on the map.
    :param aircraft_secured: the aircraft which has to go through the enemy defences
    :param security_width: take into account the minimum width the safe corridor must have
    :return: the width of the safe corridor
    '''
    # Generate a dictionary of all radar ranges with their indices
    ranges_with_indices = {(i+1, j+1): sensor_iads.list[i].range(sensor_iads.list[j])
                            for i in range(len(sensor_iads.list))
                            for j in range(i + 1, len(sensor_iads.list))
                           }

    # Sort the ranges in ascending order
    sorted_ranges = sorted(ranges_with_indices.items(), key=lambda x: x[1])

    # Calculate widths between two circles of detection range
    widths = {key: (value - sensor_iads.list[key[0]-1].get_detection_range(aircraft_secured)
                    -sensor_iads.list[key[1]-1].get_detection_range(aircraft_secured))
              for key, value in sorted_ranges}

    # Create the basic widths list
    basic_widths = []
    for i in range(1, len(sensor_iads.list)):
        basic_widths += [[(i, i+1), widths.get((i, i+1))]]

    # Sort the basic widths in ascending order
    basic_widths = sorted(basic_widths, key=lambda x: x[1])

    cpt = 0
    Test = False
    while cpt < len(basic_widths) and Test is False:# The highest basic width is not always the good one, so we have to check all the positives basic widths
        Test = True
        # Determine if there is a potential safe corridor
        if basic_widths[-cpt-1][1] - security_width <= 0:# The highest one is where the corridor will be in first,
            # if there is one
            return basic_widths[-cpt-1][1] - security_width # If the highest basic width is not positive, then there is no corridor
            # so the fitness must be penalised by having a negative value

        # Check if other widths obstruct the potential corridor
        problematic_widths = []
        for key, value in widths.items():
            if key[0] <= basic_widths[-cpt-1][0][0] and key[1] >= basic_widths[-cpt-1][0][1]:# If the basic width is positive,
                # then the other widths which could obstruct the probable safe corridor must be positive too
                problematic_widths += [value] # If they are positive, then they must be taken into account to determine the width of the safe corridor
                if value - security_width <= 0:
                    Test = False # If there are some other detection range that overlap around the basic width, then there
                    # is no longer corridor possible for this basic width and the next highest one needs to be checked
        cpt += 1
    if Test is False:
        problematic_widths.remove(basic_widths[-cpt-2][1])
        return min(problematic_widths)# returns the negative width which is the more problematic to penalise the fitness
    # Determine the final width of the safe corridor
    final_width = min(problematic_widths)# The width is equal to the smallest interesting widths
    return final_width
