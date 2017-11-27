from datetime import datetime
import pytz
import sys
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import math

def plot_alpha(datas, **kwargs):

    mask = ~np.isnan(datas['alpha']) * ~np.isinf(datas['alpha'])

    fig = plt.figure(figsize=(14, 12))
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('alpha [rad]')
    ax1.hist(datas['alpha'][mask], bins='auto', **kwargs)

def unit_vector(vector):

    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def correct_alpha(data, source_x, source_y):

    data['cen_x'] = data['cen_x'] - source_x
    data['cen_y'] = data['cen_y'] - source_y
    data['r'] = np.sqrt((data['cen_x'])**2 + (data['cen_y'])**2)
    data['phi'] = np.arctan2(data['cen_y'], data['cen_x'])

    data['alpha'] = np.sin(data['phi']) * np.sin(data['psi']) + np.cos(data['phi']) * np.cos(data['psi'])
    data['alpha'] = np.arccos(data['alpha'])
    # data['alpha'] = np.abs(data['phi'] - data['psi'])
    data['alpha'] = np.remainder(data['alpha'], np.pi/2)
    data['miss'] = data['r'] * np.sin(data['alpha'])

    return data


def main():
    new_cen_cam_x = 0 #61.2
    new_cen_cam_y = 0 #7.5

    data = np.load(sys.argv[1])

    mask = np.ones(data['size'].shape[0], dtype=bool)
    mask = data['size'] > 0

    data_cor = dict()

    for key, val in data.items():

        data_cor[key] = val[mask]

    data_cor = correct_alpha(data_cor, source_x=new_cen_cam_x, source_y=new_cen_cam_y)
    plot_alpha(data_cor)
    plt.show()

if __name__ == '__main__':
    main()
