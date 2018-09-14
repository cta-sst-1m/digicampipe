import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_alpha2d(data):  # plot original data

    N = data['N'].reshape(
        (len(np.unique(data['y'])), len(np.unique(data['x']))))
    x, y = np.meshgrid(np.unique(data['x']), np.unique(data['y']))

    fig = plt.figure(figsize=(9, 8))
    ax1 = fig.add_subplot(111)
    pcm = ax1.pcolormesh(x, y, N, rasterized=True, cmap='nipy_spectral')
    ax1.set_ylabel('FOV Y [mm]')
    ax1.set_xlabel('FOV X [mm]')
    cbar = fig.colorbar(pcm)
    cbar.set_label('N of events')


def plot_alpha2d_mod(data, floor_value,
                     r_max):  # plot based on Etienne's histo_crab.py code

    N = data['N'].reshape(
        (len(np.unique(data['y'])), len(np.unique(data['x']))))
    x, y = np.meshgrid(np.unique(data['x']), np.unique(data['y']))

    # mean threshold
    Nm = np.mean(data['N'])
    N = N - Nm
    mask = N < 0
    N[mask] = floor_value

    # circular fov
    for i in range(len(x)):
        for j in range(len(y)):
            if x[0, i] ** 2.0 + y[j, 0] ** 2.0 > r_max ** 2.0:
                N[i, j] = floor_value

    fig = plt.figure(figsize=(9, 8))
    ax1 = fig.add_subplot(111)
    pcm = ax1.pcolormesh(x, y, N, rasterized=True, cmap='nipy_spectral')
    ax1.set_ylabel('FOV Y [mm]')
    ax1.set_xlabel('FOV X [mm]')
    cbar = fig.colorbar(pcm)
    cbar.set_label('N of events (mean substracted)')


if __name__ == '__main__':
    data = np.load(sys.argv[1])

    plot_alpha2d(data)
    plot_alpha2d_mod(data, floor_value=0, r_max=444.0)
    plt.show()
