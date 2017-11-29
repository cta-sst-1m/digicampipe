import numpy as np
from digicampipe.visualization import plot
import plot_alpha_corrected
import matplotlib.pyplot as plt


def correct_hillas(data, source_x, source_y):

    data['cen_x'] = data['cen_x'] - source_x
    data['cen_y'] = data['cen_y'] - source_y
    data['r'] = np.sqrt((data['cen_x'])**2 + (data['cen_y'])**2)
    data['phi'] = np.arctan2(data['cen_y'], data['cen_x'])

    data['alpha'] = np.cos(data['phi']-data['psi'])
    data['alpha'] = np.arccos(data['alpha'])

    mask = data['alpha'] > (np.pi / 2)
    data['alpha'][mask] = np.pi - data['alpha'][mask]
    data['alpha'] = np.rad2deg(data['alpha'])
    data['phi'] = np.rad2deg(data['phi'])
    data['psi'] = np.rad2deg(data['psi'])
    # data['alpha'] = np.abs(data['phi'] - data['psi'])
    # data['alpha'] = np.remainder(data['alpha'], np.pi/2)
    data['miss'] = data['r'] * np.sin(data['alpha'])

    return data


def plot_hillas_parameter(parameter, name, units='', axis=None, bins='auto', log_x=False, log_y=False, legend=True, errorbar=False, **kwargs):

    if axis is None:

        fig = plt.figure()
        axis = fig.add_subplot(111)

    hist, bins = np.histogram(parameter, bins=bins)
    axis.step(bins[:-1], hist, where='mid', label=name, **kwargs)
    if errorbar:
        axis.errorbar(bins[:-1], hist, yerr=np.sqrt(hist), fmt='k', linestyle='None', **kwargs)
    # axis.set_xlabel(name + units)
    if log_x:
        axis.set_xscale('log')
    if log_y:
        axis.set_yscale('log')
    # axis.set_ylabel('count')
    if legend:
        axis.legend(loc='best')

    return axis


if __name__ == '__main__':


    """
    directory = '/home/alispach/data/CRAB_01/'
    hillas_filename = directory + 'hillas_crab_all_cut_100pe.npz'
    hillas = dict(np.load(hillas_filename))
    """

    directory = '/home/alispach/data/CRAB_02/'
    hillas_filename = directory + 'crab_2nd.txt'
    hillas = np.loadtxt(hillas_filename)
    hillas = hillas.T
    names = ["size", "cen_x", "cen_y", "length", "width", "r", "phi", "psi", "miss", "skewness", "kurtosis", "event_number", "time_stamp", 'border']
    hillas = dict(zip(names, hillas))

    x, y = 40, 13.5
    hillas = correct_hillas(hillas, source_x=x, source_y=y)

    mask = (np.ones(hillas['size'].shape[0]) > 0)
    print(np.sum(mask))
    mask *= hillas['width']/hillas['length'] < 2/3
    # mask *= hillas['alpha'] < 4
    try:
        mask *= hillas['border'] == 0
    except:
        print('no border parameter')

    for key, val in hillas.items():

        mask *= np.isfinite(val)

    for key, val in hillas.items():
        hillas[key] = val[mask]

    n_bins = 40

    title = 'Hillas parameters'
    figure, axis_array = plt.subplots(3, 4)
    figure.subplots_adjust(top=0.95)

    plot_hillas_parameter(hillas['alpha'], r'$\alpha$ [deg]', axis=axis_array[2, 2], bins=np.arange(0, 90 + 4, 4), errorbar=True)
    plot_hillas_parameter(hillas['cen_x'], '$x$ [mm]', axis=axis_array[0, 0], bins=n_bins)
    plot_hillas_parameter(hillas['cen_y'], '$y$ [mm]', axis=axis_array[0, 1], bins=n_bins)
    plot_hillas_parameter(hillas['length'], 'length [mm]', axis=axis_array[1, 0], bins=n_bins)
    plot_hillas_parameter(hillas['width'], 'width [mm]', axis=axis_array[1, 1], bins=n_bins)
    plot_hillas_parameter(hillas['phi'], '$\phi$ [deg]', axis=axis_array[2, 0], bins=n_bins//2)
    plot_hillas_parameter(hillas['psi'], '$\psi$ [deg]', axis=axis_array[2, 1], bins=n_bins//2)
    plot_hillas_parameter(hillas['miss'], 'miss [mm]', axis=axis_array[0, 3], bins=n_bins)
    plot_hillas_parameter(hillas['skewness'], 'skewness', axis=axis_array[2, 3], legend=True, bins=n_bins)
    plot_hillas_parameter(hillas['kurtosis'], 'kurtosis', axis=axis_array[2, 3], legend=True, bins=n_bins)
    plot_hillas_parameter(hillas['r'], 'r [mm]', axis=axis_array[0, 2], bins=n_bins)
    plot_hillas_parameter(hillas['size'], 'size [a.u.]', axis=axis_array[1, 3], bins=n_bins, log_y=True, log_x=True)
    plot_hillas_parameter(hillas['width']/hillas['length'], 'width/length []', axis=axis_array[1, 2], bins=n_bins)

    plt.show()