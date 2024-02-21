import matplotlib.pyplot as plt
import numpy as np


def from_dB(G):  # Definition of a function to convert from dB to whatever unit
    return 10 ** (G / 10)

# Creation of the different assets in the battlespace
class sensor_iads:
    #Physical caracteristics
    G = 33.6 #gain
    F = 7 #noise
    L = 8 #loss
    Pt = 10 * np.log10(7.2e3) #emitted power
    Rmin = 9 #minimum signal-to-noise ration
    l = -8 #wavelength
    Br = 1.37e6 #bandwidth
    list = []
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.jammers_targeting = []
        self.coordinates = [X,Y]
        self.point, = plt.plot(self.X,self.Y,'bs', markersize=10, label='Radar')
        sensor_iads.list += [self]
        self.name = f'radar{len(sensor_iads.list)}'

    def update(self,x = None,y = None):#allows to update the position of the asset when its coordinates are modified
        if x is None:#we can modify only the absissa
            x = self.X
        if y is None:#or the ordinate
            y = self.Y
        self.X = x#modification of the coordinates
        self.Y = y
        self.point.set_data([x],[y])#update on the graph

    def targeted_by(self, jammers):
        self.jammers_targeting = jammers.copy()
        for jammer in jammers:
            jammer.targets(self)

    def range(self, asset):#provides the distance between an asset and the radar
        return np.linalg.norm(np.array((self.X, self.Y)) -
                                  np.array((asset.X, asset.Y)))

    def detection(self, aircraft,
                  jammers = None):#tells if the asset is within the detection range of a radar
        if self.range(aircraft) <= self.get_detection_range(aircraft, jammers):
            return True
        return False

    #Function that calculates the range of the sensor, either or not it has been jammed
    def get_detection_range(self, aircraft, jammers = None):
        print(self)
        print(jammers)
        #Calculation of the range
        # Range when the jammer is the aircraft to keep out of the detection range of the radar
        # detection_range = ((from_dB(self.Pt) * from_dB(self.G)**2 * aircraft.rcs * from_dB(self.l)**2)
        #                  / ((4 * np.pi)**3 * from_dB(self.L) * from_dB(self.F) * 1.38e-23 * self.Br
        #                     + (4 * np.pi * from_dB(self.L) * jammer.Pj * from_dB(Gj) * self.Br)
        #                     / (from_dB(Lj) * Bj)))**0.5 / 1000
        # Range when the jammer reduces the detection range to protect a friendly asset from being detected (depends
        # on the distance between the jammer and the radar targeted)
        detection_range = ((from_dB(self.Pt) * from_dB(self.G) ** 2 * aircraft.rcs * from_dB(self.l) ** 2)
                     / ((4 * np.pi) ** 3 * from_dB(self.L) * from_dB(self.F) * 1.38e-23 * self.Br
                        + self.jamming_power(jammers))) ** 0.25 / 1000
        # Outline of the range
        print(detection_range)
        theta = np.linspace(0, 2 * np.pi, 100)
        x = self.X + detection_range * np.cos(theta)
        y = self.Y + detection_range * np.sin(theta)
        plt.plot(x, y, label='Range')
        return detection_range

    def jamming_power(self, jammers):#Calculate the jamming power needed to determine the detection range
        if jammers is None:#if there isn't any jammer that is jamming
            return 0
        power = 0
        for jammer in jammers:
            power += ((4 * np.pi * from_dB(self.L) * jammer.Pj * from_dB(jammer.Gj) * self.Br)
                      /((self.range(jammer) * 1000) ** 2 * from_dB(jammer.Lj) * jammer.Bj))
        return power