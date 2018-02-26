import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_lookup2d(data):  

    data_width = data[:, [0, 1, 2]]
    data_length = data[:, [0, 1, 3]]
    data_sigmaw = data[:, [0, 1, 4]]
    data_sigmal = data[:, [0, 1, 5]]

    width = data_width[:, 2].reshape(
                                     (len(np.unique(data_width[:, 0])),
                                      len(np.unique(data_width[:, 1]))
                                      ))

    xw, yw = np.meshgrid(
                         np.unique(data_width[:, 1]),
                         np.unique(data_width[:, 0]))

    length = data_length[:, 2].reshape(
                                       (len(np.unique(data_length[:, 0])),
                                        len(np.unique(data_length[:, 1]))
                                        ))

    xl, yl = np.meshgrid(
                         np.unique(data_length[:, 1]),
                         np.unique(data_length[:, 0]))

    sigmaw = data_sigmaw[:, 2].reshape(
                                       (len(np.unique(data_sigmaw[:, 0])),
                                        len(np.unique(data_sigmaw[:, 1]))
                                        ))

    xsw, ysw = np.meshgrid(
                           np.unique(data_sigmaw[:, 1]),
                           np.unique(data_sigmaw[:, 0]))

    sigmal = data_sigmal[:, 2].reshape(
                                       (len(np.unique(data_sigmal[:, 0])),
                                        len(np.unique(data_sigmal[:, 1]))
                                        ))

    xsl, ysl = np.meshgrid(
                           np.unique(data_sigmal[:, 1]),
                           np.unique(data_sigmal[:, 0]))

    fig, ax = plt.subplots(2, 2, figsize = (14, 12))

    pcm = ax[0, 0].pcolormesh(xl, yl, length, rasterized=True)
    ax[0, 0].set_ylabel('Impact parameter [m]')
    ax[0, 0].set_xlabel('log10 size')
    cbar = fig.colorbar(pcm, ax = ax[0, 0])
    cbar.set_label('mean length')

    pcm = ax[0, 1].pcolormesh(xsl, ysl, sigmal, rasterized=True)
    ax[0, 1].set_ylabel('Impact parameter [m]')
    ax[0, 1].set_xlabel('log10 size')
    cbar = fig.colorbar(pcm, ax = ax[0, 1])
    cbar.set_label('sigma length')

    pcm = ax[1, 0].pcolormesh(xw, yw, width, rasterized=True)
    ax[1, 0].set_ylabel('Impact parameter [m]')
    ax[1, 0].set_xlabel('log10 size')
    cbar = fig.colorbar(pcm, ax = ax[1, 0])
    cbar.set_label('mean width')

    pcm = ax[1, 1].pcolormesh(xsw, ysw, sigmaw, rasterized=True)
    ax[1, 1].set_ylabel('Impact parameter [m]')
    ax[1, 1].set_xlabel('log10  size')
    cbar = fig.colorbar(pcm, ax = ax[1, 1])
    cbar.set_label('sigma width')
