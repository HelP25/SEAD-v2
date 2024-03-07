from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union

from SEAD_v2.Assets import *


def find_corridor(aircraft_secured, security_width):
    radars = sensor_iads.list.copy()

    # Create a list of Polygons representing the detection zones of each radar
    detection_zones = []
    for radar in radars:
        points = []
        for a in np.linspace(0, 2 * np.pi, 100):
            x = radar.X + radar.get_detection_range(aircraft_secured, a) * np.cos(a)
            y = radar.Y + radar.get_detection_range(aircraft_secured, a) * np.sin(a)
            points.append((x, y))
        detection_zones.append(Polygon(points))

    # Create a MultiPolygon representing the union of all the detection zones
    combined_zones = list(unary_union(detection_zones).geoms)


    # If the combined_zones is in several parts, then it means that there is at least one corridor
    if len(combined_zones) > 1:
        width = []  # widths list of all the corridor found
        for i, zone1 in enumerate(combined_zones):
            for zone2 in combined_zones[i+1:]:
                diff = zone1.symmetric_difference(zone2)
                # If among all the other zones, there is not anyone in between zone and combined_zones.geoms[j],
                # then it means that there is a corridor between those two zones
                if any(diff.intersects(zone) for zone in detection_zones if not zone.intersects(zone1) and not zone.intersects(zone2)):
                    # The width of the corridor must be added to the list of the widths of all the corridors found
                    width.append(zone1.distance(zone2))
        max_width = max(width)  # The best corridor is the one with the widest width

        #   The best width is returned if it is superior to the security width
        if max_width - security_width >= 0:
            return max_width
        else:
            return max_width - security_width

    # If there is no corridor, we return the width of the overlap with the minimum width
    else:
        width = []  # widths list of all the overlaps
        for i, zone in enumerate(detection_zones):
            for j in range(i + 1, len(detection_zones)):
                overlap = zone.intersection(detection_zones[j])
                if not overlap.is_empty:
                    width.append(np.sqrt(overlap.area)) # the width is recovered by doing a dimensional operation which
                    # is true within one factor but the same is used for every calculation
        min_width = min(width) - security_width
        return -min_width
