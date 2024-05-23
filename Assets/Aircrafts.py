import matplotlib.pyplot as plt
import numpy as np


class aircraft:
    rcs = 2  # Radar cross section
    list = []  # List of all the aircrafts created
    def __init__(self, X, Y, name = 'aircraft'):
        self.X = X
        self.Y = Y
        self.coordinates = [self.X, self.Y]  # Coordinates of the asset
        self.target = None  # Used for the allocation, every aircraft can get a target allocated
        aircraft.list += [self]  # List of all the aircrafts
        self.name = name  # Name specific to every asset
        self.fuel_consumption = 10 # dollars/km
        if self.name == 'aircraft':  # To specify the type of aircraft it is to differentiate them on the graph
            point = 'r>'
            self.name += f'{len(aircraft.list)}'
        elif self.name[:3] == 'jam':
            point = 'r4'
        else:
            point = 'b4'
        self.point, = plt.plot(self.X, self.Y, point, markersize=10)  # Plotting the aircraft
        legend = plt.text(self.X, self.Y, self.name)  # Adding a legend

    # Updates the position of the aircraft on the map if it is moved
    def update(self,x = None,y = None):
        if x is None:   # we can modify only the abscissa
            x = self.X
        if y is None:   # or the ordinate
            y = self.Y
        self.X = x  # modification of the coordinates
        self.Y = y
        self.coordinates = [self.X, self.Y]
        self.point.set_data([x],[y])    # update on the graph


    # Allocates a radar to the aircraft as its target
    def targets(self, radarX):
        self.target = radarX
        if self not in radarX.jammers_targeting:    # If the jammer hasn't been allocated before to a radar
            radarX.jammers_targeting.append(self)   # Then it has to be done


class Jammer(aircraft):
    #Physical characteristics
    Gj = 3  # Gain of the antenna
    Lj = 3  # Loss of the antenna
    Pj = 1  # Power of the jamming antenna
    Bj = 30e6   # Bandwidth of the jamming effect
    list = []   # List
    def __init__(self,X,Y):
        Jammer.list += [self]   # Adding the jammer to the jammer list
        self.name = f'jammer{len(Jammer.list)}' # Defining the name of the jammer
        self.use_cost = 12000 # Cost of the use of the jammer
        super().__init__(X, Y, self.name)



class Weasel(aircraft):
    #Physical characteristics
    nb_missiles = 4
    power_missiles = 2
    list = []
    def __init__(self, X, Y):
        Weasel.list += [self]
        self.name = f'weasel{len(Weasel.list)}'
        super().__init__(X,Y,self.name)



