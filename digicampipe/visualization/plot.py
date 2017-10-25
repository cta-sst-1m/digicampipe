import matplotlib.pyplot as plt
import numpy as np


def plot_hillas(hillas_dict, title='', **kwargs):

    figure, axis_array = plt.subplots(3, 4)
    figure.suptitle(title)
    figure.subplots_adjust(top=0.95)
    plot_parameter(hillas_dict['cen_x'], name='$x$', units=' [mm]', axis=axis_array[0, 0], **kwargs)
    plot_parameter(hillas_dict['cen_y'], name='$y$', units=' [mm]', axis=axis_array[0, 1], **kwargs)
    plot_parameter(hillas_dict['length'], name='$l$', units=' [mm]', axis=axis_array[0, 2], **kwargs)
    plot_parameter(hillas_dict['width'], name='$w$', units=' [mm]', axis=axis_array[0, 3], **kwargs)
    plot_parameter(hillas_dict['phi'], name='$\phi$', units=' [rad]', axis=axis_array[1, 0], **kwargs)
    plot_parameter(hillas_dict['psi'], name='$\psi$', units=' [rad]', axis=axis_array[1, 1], **kwargs)
    plot_parameter(hillas_dict['miss'], name='miss', units=' [mm]', axis=axis_array[1, 2], **kwargs)
    plot_parameter(hillas_dict['skewness'], name='skewness', units=' []', axis=axis_array[1, 3], **kwargs, alpha=0.5)
    plot_parameter(hillas_dict['kurtosis'], name='kurtosis', units=' []', axis=axis_array[1, 3], **kwargs, alpha=0.5)
    axis_array[1,3].set_yscale('log')
    plot_parameter(hillas_dict['r'], name='r', units=' [mm]', axis=axis_array[2, 0], **kwargs)
    plot_parameter(hillas_dict['alpha'], name=r'$\alpha$', units=' [rad]', axis=axis_array[2, 1], **kwargs)
    plot_parameter(hillas_dict['width'] / hillas_dict['length'], name=r'$\frac{w}{l}$', units=' [rad]', axis=axis_array[2, 2], **kwargs)
    plot_parameter(hillas_dict['size'], name='size', units=' [p.e.]', axis=axis_array[2, 3], log=True, **kwargs)

    return figure


def plot_parameter(parameter, name, units, axis=None, **kwargs):

    parameter = parameter[~(np.isnan(parameter)) * ~np.isinf(parameter)]

    if axis is None:

        plt.figure()
        plt.hist(parameter, label=name, **kwargs)
        plt.xlabel(name + units)
        plt.ylabel('count')
        plt.legend(loc='best')

    else:

        #axis.hist(parameter, label=name + units + '\n mean : %0.1f $\pm$ %0.2f' % (np.mean(parameter), np.std(parameter)/ np.sqrt(len(parameter))), **kwargs)
        axis.hist(parameter, label=name + units , **kwargs)
        #axis.set_xlabel(name + units)
        #axis.set_ylabel('count')
        axis.legend(loc='best')
