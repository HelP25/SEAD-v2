import numpy as np

from Assets import *
from Optimise.Safe_corridor import *

def assessment(algo, solution):
    """
    Assesses the solutions given, find out if they can be improved by withdrawing some useless assets and returns some marks
    characterising how relevant are the tactics compared to one another (are the jammers highly put at risk,
    Parameters
    ----------
    algo: the genetic algorithm used to find the solutions
    solutions: list of solution returned by the ga

    Returns
    -------
    """
