from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union

from SEAD_v2.Assets import *


def find_corridor(aircraft_secured, security_width):
    radars = sensor_iads.list.copy()    # Collect all the radars within a list

    # Create a list of Polygons representing the detection zones of each radar
    detection_zones = []
    for radar in radars:
        points = []
        for a in np.linspace(0, 2 * np.pi, 100):    # Outlines the detection range from every direction
            x = radar.X + radar.get_detection_range(aircraft_secured, a) * np.cos(a)
            y = radar.Y + radar.get_detection_range(aircraft_secured, a) * np.sin(a)
            points.append((x, y))
        detection_zones.append(Polygon(points))

    # Create a MultiPolygon representing the union of all the detection zones
    combined_zones = unary_union(detection_zones)


    # If the combined_zones is in several parts, then it means that there is at least one corridor
    if combined_zones.geom_type == 'MultiPolygon':
        combined_zones = list(combined_zones.geoms)
        width = []  # widths list of all the corridor found
        for i, zone1 in enumerate(combined_zones):
            for zone2 in combined_zones[i+1:]:
                convex_hull = zone1.union(zone2).convex_hull
                # If among all the other zones, there is not anyone in between zone1 and zone2,
                # then it means that there is a corridor between those two zones
                if not any(convex_hull.intersects(zone) for zone in detection_zones if not zone.intersects(zone1) and
                                                                                       not zone.intersects(zone2)):
                    corridor = convex_hull.difference(zone1).difference(zone2)  # Calculation of the shape of the corridor
                    corridor_center = corridor.centroid.coords[0]   # Calculation of the center of the corridor
                    # The width of the corridor must be added to the list of the widths of all the corridors found
                    width.append([zone1.distance(zone2), corridor_center])
        max_width = max(width, key=lambda x: x[0])  # The best corridor is the one with the widest width

        #   The best width is returned if it is superior to the security width
        if max_width[0] - security_width >= 0:
            return max_width
        else:   # Otherwise, this maximum width minus the security width is returned to penalise it a little bit
            max_width[0] -= security_width
            return max_width

    # If there is no corridor, we return the width of the overlap with the minimum width
    else:
        width = []  # widths list of all the overlaps
        for i, zone1 in enumerate(detection_zones):
            for zone2 in detection_zones[i+1:]:
                overlap = zone1.intersection(zone2)
                # The overlap must not be empty and if it is contained into another detection zone, then it means that
                # it is not the overlap we are looking to reduce by jamming; because even if it becomes positive,
                # it will not create a safe corridor: there is another detection zone overlapping in between
                if not overlap.is_empty and not any(zone.contains(overlap) for zone in detection_zones
                                                    if not zone1.contains(zone) and not zone2.contains(zone)):
                    center_overlap = overlap.centroid.coords[0]
                    width.append([np.sqrt(overlap.area), center_overlap]) # the width is recovered by doing a
                    # dimensional operation which is true within one factor but the same is used for every calculation
        min_width = min(width, key=lambda x: x[0])  # The most plausible future corridor is where there is the minimum width
        min_width[0] = -min_width[0]    # Set as negative to fit the maximisation goal
        return min_width
