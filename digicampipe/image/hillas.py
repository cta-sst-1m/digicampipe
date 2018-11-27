"""
For further reference on Hillas parameters refer to :
http://adsabs.harvard.edu/abs/1993ApJ...404..206R
and
https://github.com/cta-observatory/ctapipe
"""

import numpy as np


def correct_hillas(hillas, source_x=0, source_y=0):

    source_x = np.atleast_1d(source_x)
    source_y = np.atleast_1d(source_y)

    hillas['x'] = hillas['x'] - source_x[:, None]
    hillas['y'] = hillas['y'] - source_y[:, None]
    hillas['r'] = np.sqrt(hillas['x'] ** 2.0 + hillas['y'] ** 2.0)
    hillas['phi'] = np.arctan2(hillas['y'], hillas['x'])
    hillas['psi'] = hillas['psi'] - np.zeros(source_y.shape)[:, None]

    return hillas


def compute_alpha(phi, psi):
    """
    :param phi: Polar angle of shower centroid
    :param psi: Orientation of shower major axis
    :return:
    """

    # phi and psi range [-np.pi, +np.pi]
    alpha = np.abs(phi - psi)
    alpha = np.minimum(np.abs(np.pi - alpha), alpha)

    return alpha


def compute_miss(r, alpha):
    """
    :param r: Shower centroid distance to center of coordinates
    :param alpha: Shower orientation to center of coordinates
    :return:
    """
    miss = r * np.sin(alpha)

    return miss

