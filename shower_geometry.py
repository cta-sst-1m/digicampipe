import numpy as np


def line_point_distance(p1, c, p):

    # Distance between a straight line and a point in space.
    # p1:   numpy 1d array, reference point on the line
    # c:    numpy 1d array, direction cosines of the line
    # p:    numpy 1d array, point in space
    # From Konrad's hessio rec_tools.h, line_point_distance function

    a = np.cross((p1 - p), c)

    return np.sqrt(np.sum(a**2)/np.sum(c**2))


def impact_parameter(x_core, y_core, telpos, theta, phi):

    cx = np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi))  # direction cosines
    cy = np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi))  #
    cz = np.cos(np.deg2rad(theta))                          #

    direction = np.array([cx, cy, cz]).T
    impact_point = np.array([x_core, y_core, np.zeros(x_core.shape[0])]).T

    impact = []
    for i in range(direction.shape[0]):

        impact.append(line_point_distance(impact_point[i],
                                          direction[i],
                                          telpos))

    return impact
