import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def lookup2d(data):

    x, y = np.meshgrid(np.unique(data['size']), np.unique(data['impact']))

    width = data['mean_width'].reshape(
                                     (len(np.unique(data['impact'])),
                                      len(np.unique(data['size']))
                                      ))

    length =  data['mean_length'].reshape(
                                       (len(np.unique(data['impact'])),
                                        len(np.unique(data['size']))
                                        ))

    sigmaw = data['sigma_width'].reshape(
                                       (len(np.unique(data['impact'])),
                                        len(np.unique(data['size']))
                                        ))

    sigmal = data['sigma_length'].reshape(
                                       (len(np.unique(data['impact'])),
                                        len(np.unique(data['size']))
                                        ))

    fig, ax = plt.subplots(2, 2, figsize=(14, 12))

    pcm = ax[0, 0].pcolormesh(x, y, length, rasterized=True)
    ax[0, 0].set_ylabel('Impact parameter [m]')
    ax[0, 0].set_xlabel('log10 size')
    cbar = fig.colorbar(pcm, ax=ax[0, 0])
    cbar.set_label('mean length')

    pcm = ax[0, 1].pcolormesh(x, y, sigmal, rasterized=True)
    ax[0, 1].set_ylabel('Impact parameter [m]')
    ax[0, 1].set_xlabel('log10 size')
    cbar = fig.colorbar(pcm, ax=ax[0, 1])
    cbar.set_label('sigma length')

    pcm = ax[1, 0].pcolormesh(x, y, width, rasterized=True)
    ax[1, 0].set_ylabel('Impact parameter [m]')
    ax[1, 0].set_xlabel('log10 size')
    cbar = fig.colorbar(pcm, ax=ax[1, 0])
    cbar.set_label('mean width')

    pcm = ax[1, 1].pcolormesh(x, y, sigmaw, rasterized=True)
    ax[1, 1].set_ylabel('Impact parameter [m]')
    ax[1, 1].set_xlabel('log10  size')
    cbar = fig.colorbar(pcm, ax=ax[1, 1])
    cbar.set_label('sigma width')


def rswl_norm(rswg, rslg, rswp, rslp):

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    weights_g = np.ones_like(rswg)/float(len(rswg))
    weights_p = np.ones_like(rswp)/float(len(rswp))
    ax[0].hist(rswg, bins=100, weights=weights_g, label='gamma',
               histtype='step', stacked=True, fill=False, linewidth=4,
               color='black', range=[-10, 20])
    ax[0].hist(rswp, bins=100, weights=weights_p, label='proton',
               histtype='step', stacked=True, fill=False, linewidth=4,
               color='red', range=[-10, 20])
    ax[0].set_xlabel('RSW [sigma]')
    ax[0].set_ylabel('Normalised')
    ax[0].legend()

    weights_g = np.ones_like(rslg)/float(len(rslg))
    weights_p = np.ones_like(rslp)/float(len(rslp))
    ax[1].hist(rslg, bins=100, weights=weights_g, label='gamma',
               histtype='step', stacked=True, fill=False, linewidth=4,
               color='black', range=[-10, 20])
    ax[1].hist(rslp, bins=100, weights=weights_p, label='proton',
               histtype='step', stacked=True, fill=False, linewidth=4,
               color='red', range=[-10, 20])
    ax[1].set_xlabel('RSL [sigma]')
    ax[1].set_ylabel('Normalised')
    ax[1].legend()


def efficiency(gamma_cut, efficiency_gammaw, efficiency_gammal,
               efficiency_protonw, efficiency_protonl):

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(gamma_cut, efficiency_gammaw, 'k.', label='gamma')
    ax[0].plot(gamma_cut, efficiency_protonw, 'r.', label='proton')
    ax[0].set_xlabel('RSW gamma cut [sigma]')
    ax[0].set_ylabel('efficiency')
    ax[0].legend()

    ax[1].plot(gamma_cut, efficiency_gammal, 'k.', label='gamma')
    ax[1].plot(gamma_cut, efficiency_protonl, 'r.', label='proton')
    ax[1].set_xlabel('RSL gamma cut [sigma]')
    ax[1].set_ylabel('efficiency')
    ax[1].legend()


def quality(gamma_cut, qualityw, qualityl):

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(gamma_cut, qualityw, 'k-')
    ax[0].set_xlabel('RSW gamma cut [sigma]')
    ax[0].set_ylabel('Quality factor')

    ax[1].plot(gamma_cut, qualityl, 'k-')
    ax[1].set_xlabel('RSL gamma cut [sigma]')
    ax[1].set_ylabel('Quality factor')
