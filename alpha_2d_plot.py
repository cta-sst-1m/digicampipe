
import numpy as np
import sys
import matplotlib.pyplot as plt
from optparse import OptionParser


def plot_alpha2d(data):  # plot original data

    N = data['N'].reshape((len(np.unique(data['y'])), len(np.unique(data['x']))))
    x, y = np.meshgrid(np.unique(data['x']), np.unique(data['y']))
    fig = plt.figure(figsize=(9, 8))
    ax1 = fig.add_subplot(111)
    pcm = ax1.pcolormesh(x, y, N, rasterized=True, cmap='nipy_spectral')
    ax1.set_ylabel('FOV Y [mm]')
    ax1.set_xlabel('FOV X [mm]')
    cbar = fig.colorbar(pcm)
    cbar.set_label('N of events')


def plot_alpha2d_mod(data, floor_value, r_max, save_file):  # plot based on Etienne's histo_crab.py code

    N = data['N'].reshape((len(np.unique(data['y'])), len(np.unique(data['x']))))
    x, y = np.meshgrid(np.unique(data['x']), np.unique(data['y']))

    # mean threshold
    Nm = np.mean(data['N'])
    N = N - Nm
    mask = N < 0
    N[mask] = floor_value

    # circular fov
    for i in range(len(x)):
        for j in range(len(y)):
            if x[0, i]**2.0 + y[j, 0]**2.0 > r_max**2.0:
                N[i, j] = floor_value

    fig = plt.figure(figsize=(9, 8))
    ax1 = fig.add_subplot(111)
    pcm = ax1.pcolormesh(x, y, N, rasterized=True, cmap='nipy_spectral')
    ax1.set_ylabel('FOV Y [mm]')
    ax1.set_xlabel('FOV X [mm]')
    cbar = fig.colorbar(pcm)
    cbar.set_label('N of events (mean substracted)')
    #ax1.xaxis.set_ticklabels([])
    #ax1.yaxis.set_ticklabels([])

    if save_file is not None:
        plt.savefig(save_file)
        print('Plot saved as ', save_file)

    '''
    # Cropped image
    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(111)
    pcm = ax1.pcolormesh(x, y, N, rasterized=True, cmap='nipy_spectral')
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.set_xlim([-150, 150])
    ax1.set_ylim([-150, 150])
    
    if save_file is not None:
        plt.savefig(save_file[:-4] + '-cropp.png')
    '''


if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="path", help="path to data files", default='../alpha2d_parametermap_noaddrow/alpha_2d_parametermap_noaddrow_1015.npz')
    parser.add_option("-o", "--output", dest="output", help="output filename with saved image (mean substracted)")
    (options, args) = parser.parse_args()

    data = np.load(options.path)

    plot_alpha2d(data)
    plot_alpha2d_mod(data, floor_value=0, r_max=444.0, save_file=options.output)
    if options.output is None:
        plt.show()
