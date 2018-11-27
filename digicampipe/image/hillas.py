import numpy as np


def correct_hillas(hillas, source_x=0, source_y=0):  # cyril

    hillas['x'] = hillas['x'] - source_x
    hillas['y'] = hillas['y'] - source_y
    hillas['r'] = np.sqrt(hillas['x'] ** 2.0 + hillas['y'] ** 2.0)
    hillas['phi'] = np.arctan2(hillas['y'], hillas['x'])

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


# etienne from scan_crab_cluster.c
def correct_alpha_3(data, source_x=0, source_y=0):
    d_x = np.cos(data['psi'])
    d_y = np.sin(data['psi'])
    to_c_x = source_x - data['x']
    to_c_y = source_y - data['y']
    to_c_norm = np.sqrt(to_c_x ** 2.0 + to_c_y ** 2.0)
    to_c_x = to_c_x / to_c_norm
    to_c_y = to_c_y / to_c_norm
    p_scal_1 = d_x * to_c_x + d_y * to_c_y
    p_scal_2 = -d_x * to_c_x + -d_y * to_c_y
    alpha_c_1 = abs(np.arccos(p_scal_1))
    alpha_c_2 = abs(np.arccos(p_scal_2))
    alpha_cetienne = alpha_c_1
    alpha_cetienne[alpha_c_2 < alpha_c_1] = alpha_c_2[alpha_c_2 < alpha_c_1]
    data.loc[:, 'alpha'] = 180.0 / np.pi * alpha_cetienne
    data.loc[:, 'r'] = to_c_norm
    data.loc[:, 'miss'] = data['r'] * np.sin(data['alpha'])
    return data


def correct_alpha_4(data, sources_x, sources_y):
    sources_x = np.array(sources_x)
    sources_y = np.array(sources_y)
    d_x = np.cos(data['psi'])
    d_y = np.sin(data['psi'])
    to_c_x = sources_x[None, :] - data['x'][:, None]
    to_c_y = sources_y[None, :] - data['y'][:, None]
    to_c_norm = np.sqrt(to_c_x ** 2.0 + to_c_y ** 2.0)
    to_c_x = to_c_x / to_c_norm
    to_c_y = to_c_y / to_c_norm
    p_scal_1 = d_x[:, None] * to_c_x + d_y[:, None] * to_c_y
    p_scal_2 = -d_x[:, None] * to_c_x - d_y[:, None] * to_c_y
    alpha_c_1 = abs(np.arccos(p_scal_1))
    alpha_c_2 = abs(np.arccos(p_scal_2))
    mask = (alpha_c_2 < alpha_c_1)
    alpha_cetienne = alpha_c_1
    alpha_cetienne[mask] = alpha_c_2[mask]
    alphas = 180.0 / np.pi * alpha_cetienne
    return alphas
