import matplotlib.pyplot as plt
import numpy as np


class aircraft:
    rcs = 2
    list = []
    def __init__(self, X, Y, color = None):
        if color is None:
            self.color = 'Aircraft'
            point = 'r>'
        elif color == 'Jammer':
            self.color = color
            point = 'r4'
        else:
            self.color = color
            point = 'b4'
        self.X = X
        self.Y = Y
        self.coordinates = [self.X, self.Y]
        self.target = None
        aircraft.list += [self]
        self.name = f'aircraft{len(aircraft.list)}'
        self.point, = plt.plot(self.X, self.Y, point, markersize=10, label=self.name)


    def update(self,x = None,y = None):#to update the position of the aircraft on the map if it moved
        if x is None:
            x = self.X
        if y is None:
            y = self.Y
        self.X = x
        self.Y = y
        self.coordinates = [self.X, self.Y]
        self.point.set_data([x],[y])

    def targets(self, radarX):
        self.target = radarX
        if self not in radarX.jammers_targeting:
            radarX.jammers_targeting.append(self)


class Jammer(aircraft):
    #Physical caracteristics
    Gj = 3
    Lj = 3
    Pj = 1
    Bj = 30e6
    list = []
    def __init__(self,X,Y):
        self.color = 'Jammer'
        super().__init__(X, Y, self.color)
        Jammer.list += [self]
        self.name = f'jammer{len(Jammer.list)}'
        plt.text(self.X, self.Y, self.name)



class Weasel(aircraft):
    #Physical caracteristics
    nb_missiles = 4
    power_missiles = 2
    list = []
    def __init__(self, X, Y):
        self.name = 'Weasel'
        super().__init__(X,Y,self.name)
        Weasel.list += [self]
        self.name = f'weasel{len(Weasel.list)}'

pass
