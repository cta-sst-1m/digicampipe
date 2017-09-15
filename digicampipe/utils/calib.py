import numpy as np
from scipy.interpolate import splrep, splev


def compute_gain_drop(pedestal, type='std'):

    data = np.load('true_mean_std_nsb.npz')
    nsb_rate = data['nsb_rate']
    std = data['std']
    baseline_shift = data['mean'] - data['mean'][0]

    if type == 'std':

        spline = splrep(std, gain_drop(nsb_rate))

    elif type == 'mean':

        spline = splrep(baseline_shift, gain_drop(nsb_rate))

    else:
        raise('Unknown type %s' % type)

    return splev(pedestal, spline)


def compute_nsb_rate(pedestal, type='std'):

    data = np.load('true_mean_std_nsb.npz')
    nsb_rate = data['nsb_rate']
    std = data['std']
    baseline_shift = data['mean'] - data['mean'][0]

    if type == 'mean':

        spline = splrep(baseline_shift, nsb_rate)

    elif type == 'std':

        spline = splrep(std, nsb_rate)

    else:

        raise('Unknown type %s' % type)

    return splev(pedestal, spline)


def get_gains():

    return np.ones(1296) * 23. # TODO, replace gain of 23 by calib array of gain


def gain_drop(nsb_rate, cell_capacitance=85. * 1E-15, bias_resistance=10. * 1E3):

    return 1. / (1. + nsb_rate * cell_capacitance * bias_resistance * 1E9)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    data = np.load('true_mean_std_nsb.npz')

    nsb_rate = data['nsb_rate']
    nsb = data['nsb_rate']
    std = data['std']
    baseline_shift = data['mean'] - data['mean'][0]

    std_1 = np.linspace(0, 10, num=30)
    baseline_shift_1 = np.linspace(0, 50, num=30)

    plt.figure()
    plt.plot(std_1, compute_gain_drop(std_1, type='std'))
    plt.plot(std, gain_drop(nsb_rate), linestyle='None', marker='o')
    plt.legend()

    plt.figure()
    plt.plot(baseline_shift_1, compute_gain_drop(baseline_shift_1, type='mean'))
    plt.plot(baseline_shift, gain_drop(nsb_rate), linestyle='None', marker='o')
    plt.legend()

    plt.figure()
    plt.plot(std_1, compute_nsb_rate(std_1, type='std'))
    plt.plot(std, nsb_rate, linestyle='None', marker='o')
    plt.legend()

    plt.figure()
    plt.plot(baseline_shift_1, compute_nsb_rate(baseline_shift_1, type='mean'))
    plt.plot(baseline_shift, nsb_rate, linestyle='None', marker='o')
    plt.legend()

    plt.show()

print(data.__dict__)