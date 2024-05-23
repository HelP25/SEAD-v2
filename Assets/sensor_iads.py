import matplotlib.pyplot as plt
import numpy as np
from Assets import *


def from_dB(G):  # Definition of a function to convert from dB to whatever unit
    return 10 ** (G / 10)


# Creation of the different assets in the battle space
class sensor_iads:
    # Physical characteristics
    G = 33.6  # gain
    F = 7  # noise
    L = 8  # loss
    Pt = 10 * np.log10(7.2e3)  # emitted power
    Rmin = 9  # minimum signal-to-noise ration
    l = -8  # wavelength
    Br = 1.37e6  # bandwidth
    list = []
    k = 0.08  # Coefficient to approximate the lobes of the radar
    beam_width = np.deg2rad(30)

    def __init__(self, X, Y):
        self.X = X  # Coordinates of the radar
        self.Y = Y
        self.jammers_targeting = [] # List of the jammers that are targeting the radar
        self.coordinates = [X, Y]
        sensor_iads.list += [self]  # List of all the radars
        self.name = f'radar{len(sensor_iads.list)}' # Name of the radar to be written on the graph
        self.point, = plt.plot(self.X, self.Y, 'bs', markersize=10, label=self.name)

    # Updates the position of the radar on the graph if its coordinates are modified
    def update(self, x=None, y=None):
        if x is None:   # we can modify only the abscissa
            x = self.X
        if y is None:   # or the ordinate
            y = self.Y
        self.X = x  # modification of the coordinates
        self.Y = y
        self.point.set_data([x], [y])   # update on the graph

    # Allocates a list of jammers to the radar that they target
    def targeted_by(self, jammers):
        self.jammers_targeting = jammers.copy() # The jammers targeting the radar are put in the jammers_targeting list
        for jammer in jammers:
            jammer.targets(self)    # Also, to every jammer is allocated the radar that it targets


    # Calculates the distance between an asset and the radar
    def range(self, asset):
        return np.linalg.norm(np.array((self.X, self.Y)) -
                              np.array((asset.X, asset.Y)))


    # Tells if the asset is within the detection range of a radar
    def detection(self, aircraft):
        if self.range(aircraft) <= self.get_detection_range(aircraft):
            return True
        return False


    # Outline the detection range of the radar
    def outline_detection(self, aircraft):
        x = []
        y = []
        alpha = np.linspace(0, 2 * np.pi, 100)
        for a in alpha:
            x.append(self.X + self.get_detection_range(aircraft, a) * np.cos(a))
            y.append(self.Y + self.get_detection_range(aircraft, a) * np.sin(a))
        plt.plot(x, y, label='Range')


    # Function that calculates the detection range of the sensor, either or not it has been jammed
    def get_detection_range(self, aircraft, alpha=None):
        # Calculation of the range
        # Range when the jammer is the aircraft to keep out of the detection range of the radar
        # detection_range = ((from_dB(self.Pt) * from_dB(self.G)**2 * aircraft.rcs * from_dB(self.l)**2)
        #                  / ((4 * np.pi)**3 * from_dB(self.L) * from_dB(self.F) * 1.38e-23 * self.Br
        #                     + (4 * np.pi * from_dB(self.L) * jammer.Pj * from_dB(Gj) * self.Br)
        #                     / (from_dB(Lj) * Bj)))**0.5 / 1000
        # Range when the jammer reduces the detection range to protect a friendly asset from being detected (depends
        # on the distance between the jammer and the radar targeted)
        detection_range = ((from_dB(self.Pt) * from_dB(self.G) ** 2 * aircraft.rcs * from_dB(self.l) ** 2)
                           / ((4 * np.pi) ** 3 * from_dB(self.L) * from_dB(self.F) * 1.38e-23 * self.Br
                              + self.jamming_power(aircraft, alpha))) ** 0.25 / 1000
        # The equation comes from: Wang, Q. and Yao, D. (2017). Research on Electronic Jamming Airspace Planning. 2017 International Conference on Computer Technology, Electronics and Communication (ICCTEC). doi:https://doi.org/10.1109/icctec.2017.00180.
        return detection_range

    # Calculates the jamming power needed to determine the detection range
    def jamming_power(self, aircraft, alpha):
        jammers = self.jammers_targeting.copy()
        if jammers == []:   # if there isn't any jammer that is jamming
            return 0
        power = 0
        for jammer in jammers:
            # Basic model
            # power += ((4 * np.pi * from_dB(self.L) * jammer.Pj * from_dB(jammer.Gj) * self.Br)
            #           / ((self.range(jammer) * 1000) ** 2 * from_dB(jammer.Lj) * jammer.Bj))
            # More realistic model
            power += self.jamming_power_1jammer(aircraft, jammer, alpha)
        return power

    # Calculates the jamming power of only one jammer received by the radar
    def jamming_power_1jammer(self, aircraft, jammer, alpha=None):
        return ((4 * np.pi * from_dB(self.L) * jammer.Pj * from_dB(jammer.Gj) * self.Br)
                / ((self.range(jammer) * 1000) ** 2 * from_dB(jammer.Lj) * jammer.Bj / self.G_theta(aircraft, jammer,
                                                                                                    alpha)))


    # Calculates the gain that describes the effect of the angle of attack of the jamming compared to where is heading
    # the main beam
    def G_theta(self, aircraft, jammer, alpha):
        if alpha is None:   # alpha is used to simulate a different heading to the aircraft to draw the detection range
            angle_to_aircraft = np.arctan2(aircraft.Y - self.Y, aircraft.X - self.X)
        else:
            angle_to_aircraft = alpha

        angle_to_jammer = np.arctan2(self.Y - jammer.Y, self.X - jammer.X)
        angle_difference = np.abs(np.mod(angle_to_aircraft - angle_to_jammer, 2 * np.pi) - np.pi)   # Angle between the
        # jammer and the aircraft the radar is trying to detect
        # Then, the theoretic equation used to calculate the gain
        if abs(angle_difference) <= self.beam_width / 2:    # It is where we get back to the basic model
            return 1
        elif abs(angle_difference) > self.beam_width / 2 and abs(angle_to_jammer) <= np.pi / 2:
            return self.k * (self.beam_width / angle_difference) ** 2   # Where it is smoothed to have a continuous function
        else:   # Where the detection range is almost not modified
            return self.k * (2 * self.beam_width / np.pi) ** 2
