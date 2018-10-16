import astropy.units as u
import numpy as np


def correct_hillas(data, source_x=0, source_y=0):  # cyril

    data['x'] = data['x'] - source_x
    data['y'] = data['y'] - source_y
    data['r'] = np.sqrt(data['x'] ** 2.0 + data['y'] ** 2.0)
    data['phi'] = np.arctan2(data['y'], data['x'])

    # data = compute_alpha_and_miss(data)

    return data


def compute_alpha(hillas_parameters):
    data = hillas_parameters

    alpha = np.cos(data['phi'] - data['psi'])
    alpha = np.arccos(alpha)
    alpha = np.remainder(alpha, np.pi / 2 * u.rad)
    alpha = alpha.to(u.deg)

    return alpha


def compute_miss(hillas_parameters, alpha):
    miss = hillas_parameters['r'] * np.sin(alpha)

    return miss


def correct_alpha_2(data, source_x=0,
                    source_y=0):  # cyril from prod_alpha_plot.c

    x = data['cen_x'] - source_x
    y = data['cen_y'] - source_y
    data['r'] = np.sqrt(x ** 2.0 + y ** 2.0)
    phi = np.arctan(y / x)
    calpha = np.sin(phi) * np.sin(data['psi']) + np.cos(phi) * np.cos(
        data['psi'])
    alpha = np.arccos(calpha)
    for i in range(len(alpha)):
        if alpha[i] > np.pi / 2.0:
            alpha[i] = np.pi - alpha[i]
    # delta_alpha = np.arctan2(source_y,source_x)
    # delta_alpha -= np.arctan2(-1.0*data['cen_y'],-1.0*data['cen_x'])
    # alpha_r = alpha + delta_alpha;
    # miss_c = r*TMath::Sin(alpha_c);
    # miss_r = r*TMath::Sin(alpha_r);
    data['alpha'] = alpha
    data['alpha'] = np.rad2deg(data['alpha'])  # conversion to degrees
    data['miss'] = data['r'] * np.sin(data['alpha'])
    return data


"""
def alpha_roland(datas, source_x=0, source_y=0): #roland from prod_alpha_plot.c

    alpha = np.sin(datas['miss']/datas['r'])
    alpha2 = np.arctan2(-datas['cen_y'], -datas['cen_x'])
    alpha2 = alpha2 - datas['psi'] + np.pi
    for i in range(len(alpha2)):
        if (alpha2[i] > np.pi):
            alpha2[i] = alpha2[i] - 2*np.pi
        elif (alpha2[i] < -np.pi):
            alpha2[i] = alpha2[i] + 2*np.pi

    delta_alpha2 = np.arctan2(source_y-datas['cen_y'],source_x-datas['cen_x'])
    delta_alpha2 -= np.arctan2(-datas['cen_y'], -datas['cen_x'])
    alpha2_crab = alpha2 + delta_alpha2

    for i in range(len(alpha2_crab)):
        if (alpha2_crab[i] > 2*np.pi):
            alpha2_crab[i] = alpha2_crab[i] - 2*np.pi
        elif (alpha2_crab[i] < -2*np.pi):
            alpha2_crab[i] = alpha2_crab[i] + 2*np.pi

    for i in range(len(alpha2_crab)):
        if (alpha2_crab[i] > np.pi):
            alpha2_crab[i] = 2*np.pi - alpha2_crab[i]
        elif (alpha2_crab[i] < -np.pi):
            alpha2_crab[i] = -2*np.pi - alpha2_crab[i]

    alpha2_crab = abs(alpha2_crab)

    for i in range(len(alpha2_crab)):
        if (alpha2_crab[i] > 0.5*np.pi):
            alpha2_crab[i] = np.pi - alpha2_crab[i]

    datas['alpha'] = alpha2_crab
    return datas
"""


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
    for i in range(len(alpha_cetienne)):
        if (alpha_c_2[i] < alpha_c_1[i]):
            alpha_cetienne[i] = alpha_c_2[i]
    data['alpha'] = 180.0 / np.pi * alpha_cetienne
    data['r'] = to_c_norm
    data['miss'] = data['r'] * np.sin(data['alpha'])
    return data
