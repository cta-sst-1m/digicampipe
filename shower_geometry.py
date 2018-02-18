
import numpy as np

def line_point_distance(xp1, yp1, zp1, cx, cy, cz, x, y, z):

    #  Distance between a straight line and a point in space.

    # xp1, yp1, zp1:  reference point on the line
    # cx, cy, cz:  direction cosines of the line
    # x, y, z:	point in space

    # From Konrad's hessio rec_tools.h, line_point_distance function

    a1 = (y-yp1)*cz - (z-zp1)*cy
    a2 = (z-zp1)*cx - (x-xp1)*cz
    a3 = (x-xp1)*cy - (y-yp1)*cx
    a  = a1*a1 + a2*a2 + a3*a3
    b = cx*cx + cy*cy + cz*cz

    return np.sqrt(a/b)


def impact_parameter(x_core, y_core, telpos_x, telpos_y, telpos_z, theta, phi):

    cx = np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi))  # direction cosines
    cy = np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi))  #
    cz = np.cos(np.deg2rad(theta))                          #

    impact = line_point_distance(x_core, y_core, 0., cx, cy, cz, telpos_x, telpos_y, telpos_z)

    return impact
